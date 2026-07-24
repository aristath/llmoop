use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::stream_plan::{TensorIndex, TensorMetadata};
use crate::tensor_storage::{TensorStorage, TensorStorageRange};
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanResidentBuffer, VulkanResidentKernelBufferAccess,
    VulkanResidentKernelBufferBinding, VulkanResidentKernelDispatch, VulkanResidentKernelSequence,
    VulkanResidentKernelSequenceStep, VulkanResidentQueueSubmissionBatch,
    VulkanSharedHostAllocation, VulkanTimelineSemaphore, VulkanTimelineSemaphorePoint,
};
use crate::vulkan_stream_circuit::{
    VulkanActivationSlotBufferOverride, VulkanDescriptorResourceAddress,
    VulkanLoadedReusableKernelArtifact, VulkanLoadedReusableKernelArtifactManifest,
    VulkanKernelScalarBinding, VulkanKernelScalarSource, VulkanPreparedDispatch,
    VulkanPreparedDispatchPlan, VulkanResidentFeedbackControlPlane,
    VulkanReusableKernelArtifactManifest,
};

const DISTRIBUTABLE_PARALLEL_PROJECTION_OP: &str = "parallel_linear_silu_multiply";
const DISTRIBUTABLE_SPARSE_EXPERT_OPS: [&str; 2] = ["sparse_moe_gate_up", "sparse_moe_down"];
const BF16_BYTE_COUNT: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedExecutionPlan {
    pub device_ids: Vec<String>,
    pub storage_buffer_offset_alignment: usize,
    pub dispatches: Vec<VulkanDistributedDispatchPlan>,
    pub dispatch_groups: Vec<VulkanDistributedDispatchGroup>,
    pub shared_input_byte_capacity: usize,
    pub shared_output_byte_capacity: usize,
    pub distributed_parameter_byte_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchSubmission {
    pub dependency_value: u64,
    pub consume_owner_ready_signal: bool,
    pub prepare_owner_continuation: bool,
    pub signal_completion: bool,
    pub use_feedback_indirect: bool,
}

impl VulkanDistributedExecutionPlan {
    pub fn for_placed_components(
        device_ids: &[String],
        storage_buffer_offset_alignment: usize,
    ) -> Result<Self, VulkanDistributedPlanError> {
        validate_device_pool(device_ids)?;
        if storage_buffer_offset_alignment == 0
            || !storage_buffer_offset_alignment.is_power_of_two()
            || !storage_buffer_offset_alignment.is_multiple_of(BF16_BYTE_COUNT)
        {
            return Err(VulkanDistributedPlanError(format!(
                "distributed storage-buffer offset alignment {storage_buffer_offset_alignment} is invalid"
            )));
        }
        Ok(Self {
            device_ids: device_ids.to_vec(),
            storage_buffer_offset_alignment,
            dispatches: Vec::new(),
            dispatch_groups: Vec::new(),
            shared_input_byte_capacity: 0,
            shared_output_byte_capacity: 0,
            distributed_parameter_byte_count: 0,
        })
    }

    pub fn from_prepared_plans(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanReusableKernelArtifactManifest,
        device_ids: &[String],
        storage_buffer_offset_alignment: usize,
    ) -> Result<Self, VulkanDistributedPlanError> {
        validate_device_pool(device_ids)?;
        if storage_buffer_offset_alignment == 0
            || !storage_buffer_offset_alignment.is_power_of_two()
            || !storage_buffer_offset_alignment.is_multiple_of(BF16_BYTE_COUNT)
        {
            return Err(VulkanDistributedPlanError(format!(
                "distributed storage-buffer offset alignment {storage_buffer_offset_alignment} is invalid"
            )));
        }
        let mut dispatches = Vec::new();
        let mut shared_input_byte_capacity = 0usize;
        let mut shared_output_byte_capacity = 0usize;
        let mut distributed_parameter_byte_count = 0usize;

        for (owner_device_id, prepared_plan) in prepared_plans {
            if device_ids.len() < 2 {
                continue;
            }
            if !device_ids
                .iter()
                .any(|device_id| device_id == owner_device_id)
            {
                return Err(VulkanDistributedPlanError(format!(
                    "dispatch owner {:?} is absent from the distributed execution device pool",
                    owner_device_id
                )));
            }
            for dispatch in &prepared_plan.dispatches {
                if dispatch.op != DISTRIBUTABLE_PARALLEL_PROJECTION_OP
                    && !DISTRIBUTABLE_SPARSE_EXPERT_OPS.contains(&dispatch.op.as_str())
                {
                    continue;
                }
                let artifact = artifact_manifest
                    .artifacts
                    .iter()
                    .find(|artifact| artifact.family_id == dispatch.reusable_family_id)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed dispatch {}.{} has no artifact for family {:?}",
                            dispatch.component_id, dispatch.node_id, dispatch.reusable_family_id
                        ))
                    })?;
                let Some(planned) = plan_dispatch(
                    owner_device_id,
                    dispatch,
                    tensor_index,
                    device_ids,
                    artifact.workgroup_count_x,
                    storage_buffer_offset_alignment,
                )?
                else {
                    continue;
                };
                shared_input_byte_capacity =
                    shared_input_byte_capacity.max(planned.input_byte_capacity);
                shared_output_byte_capacity =
                    shared_output_byte_capacity.max(planned.output_byte_capacity);
                distributed_parameter_byte_count = distributed_parameter_byte_count
                    .checked_add(planned.distributed_parameter_byte_count)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed parameter byte count overflowed".to_string(),
                        )
                    })?;
                dispatches.push(planned);
            }
        }

        let dispatch_groups = distributed_dispatch_groups(&dispatches);
        Ok(Self {
            device_ids: device_ids.to_vec(),
            storage_buffer_offset_alignment,
            dispatches,
            dispatch_groups,
            shared_input_byte_capacity,
            shared_output_byte_capacity,
            distributed_parameter_byte_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchGroup {
    pub owner_device_id: String,
    pub dispatches: Vec<VulkanDistributedDispatchPlan>,
}

impl VulkanDistributedDispatchGroup {
    pub fn leader(&self) -> &VulkanDistributedDispatchPlan {
        self.dispatches
            .first()
            .expect("distributed dispatch groups are never empty")
    }

    pub fn tail(&self) -> &VulkanDistributedDispatchPlan {
        self.dispatches
            .last()
            .expect("distributed dispatch groups are never empty")
    }

    pub fn contains_dispatch(&self, dispatch_index: usize) -> bool {
        self.dispatches
            .iter()
            .any(|dispatch| dispatch.dispatch_index == dispatch_index)
    }

    pub fn dispatch_indices(&self) -> Vec<usize> {
        self.dispatches
            .iter()
            .map(|dispatch| dispatch.dispatch_index)
            .collect()
    }
}

fn distributed_dispatch_groups(
    dispatches: &[VulkanDistributedDispatchPlan],
) -> Vec<VulkanDistributedDispatchGroup> {
    let mut groups = Vec::<VulkanDistributedDispatchGroup>::new();
    for dispatch in dispatches {
        if let Some(group) = groups.last_mut()
            && distributed_dispatches_can_share_sequence(group.tail(), dispatch)
        {
            group.dispatches.push(dispatch.clone());
        } else {
            groups.push(VulkanDistributedDispatchGroup {
                owner_device_id: dispatch.owner_device_id.clone(),
                dispatches: vec![dispatch.clone()],
            });
        }
    }
    groups
}

fn distributed_dispatches_can_share_sequence(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    producer.owner_device_id == consumer.owner_device_id
        && producer.component_id == consumer.component_id
        && producer.dispatch_index.checked_add(1) == Some(consumer.dispatch_index)
        && producer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && consumer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && same_distributed_activation(&producer.output_activation, &consumer.input_activation)
        && producer.shards.len() == consumer.shards.len()
        && producer
            .shards
            .iter()
            .zip(&consumer.shards)
            .all(|(producer, consumer)| {
                producer.device_id == consumer.device_id
                    && producer.row_start == consumer.row_start
                    && producer.row_count == consumer.row_count
                    && producer.base_workgroup_z == consumer.base_workgroup_z
            })
}

fn same_distributed_activation(
    left: &VulkanDistributedActivationSlot,
    right: &VulkanDistributedActivationSlot,
) -> bool {
    left.component_id == right.component_id
        && left.signal_id == right.signal_id
        && left.slot == right.slot
        && left.byte_capacity == right.byte_capacity
        && left.signal_byte_capacity == right.signal_byte_capacity
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchPlan {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub reusable_family_id: String,
    pub input_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub output_rows: usize,
    pub input_width: usize,
    pub row_alignment: usize,
    pub input_activation: VulkanDistributedActivationSlot,
    pub auxiliary_input_activations: Vec<VulkanDistributedActivationSlot>,
    pub output_activation: VulkanDistributedActivationSlot,
    pub distribution: VulkanDistributedDispatchDistribution,
    pub distributed_parameter_byte_count: usize,
    pub shards: Vec<VulkanDistributedDispatchShard>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanDistributedDispatchDistribution {
    OutputRows,
    ExpertRange,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationSlot {
    pub binding: usize,
    pub component_id: String,
    pub signal_id: String,
    pub slot: usize,
    pub byte_capacity: usize,
    pub signal_byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferPlan {
    pub allocations: Vec<VulkanDistributedActivationBufferAllocation>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub reference_count: usize,
    pub total_shared_byte_capacity: usize,
}

impl VulkanDistributedActivationBufferPlan {
    pub fn from_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let device_ids = execution_plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut allocations = BTreeMap::<
            VulkanDistributedActivationBufferAllocationKey,
            VulkanDistributedActivationBufferAllocation,
        >::new();

        for dispatch in &execution_plan.dispatches {
            let participant_device_ids = dispatch
                .shards
                .iter()
                .map(|shard| shard.device_id.as_str())
                .collect::<BTreeSet<_>>();
            if participant_device_ids.is_empty() {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has no device shards",
                    dispatch.component_id, dispatch.node_id
                )));
            }
            if !participant_device_ids.contains(dispatch.owner_device_id.as_str()) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} does not include its owner {:?}",
                    dispatch.component_id, dispatch.node_id, dispatch.owner_device_id
                )));
            }
            if let Some(device_id) = participant_device_ids
                .iter()
                .find(|device_id| !device_ids.contains(**device_id))
            {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} uses device {device_id:?} outside the execution pool",
                    dispatch.component_id, dispatch.node_id
                )));
            }

            accumulate_activation_allocation(
                &mut allocations,
                &dispatch.owner_device_id,
                &dispatch.input_activation,
                &participant_device_ids,
                VulkanDistributedActivationAccess::Input,
            )?;
            for activation in &dispatch.auxiliary_input_activations {
                accumulate_activation_allocation(
                    &mut allocations,
                    &dispatch.owner_device_id,
                    activation,
                    &participant_device_ids,
                    VulkanDistributedActivationAccess::Input,
                )?;
            }
            accumulate_activation_allocation(
                &mut allocations,
                &dispatch.owner_device_id,
                &dispatch.output_activation,
                &participant_device_ids,
                VulkanDistributedActivationAccess::Output,
            )?;
        }

        let import_count = allocations.values().try_fold(0usize, |total, allocation| {
            total
                .checked_add(allocation.device_ids.len())
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation import count overflowed".to_string(),
                    )
                })
        })?;
        let reference_count = allocations.values().try_fold(0usize, |total, allocation| {
            total
                .checked_add(allocation.input_use_count)
                .and_then(|count| count.checked_add(allocation.output_use_count))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation reference count overflowed".to_string(),
                    )
                })
        })?;
        let total_shared_byte_capacity =
            allocations.values().try_fold(0usize, |total, allocation| {
                total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation byte capacity overflowed".to_string(),
                    )
                })
            })?;
        let allocations = allocations.into_values().collect::<Vec<_>>();

        Ok(Self {
            allocation_count: allocations.len(),
            allocations,
            import_count,
            reference_count,
            total_shared_byte_capacity,
        })
    }

    pub fn allocation(
        &self,
        owner_device_id: &str,
        component_id: &str,
        slot: usize,
    ) -> Option<&VulkanDistributedActivationBufferAllocation> {
        self.allocations.iter().find(|allocation| {
            allocation.owner_device_id == owner_device_id
                && allocation.component_id == component_id
                && allocation.slot == slot
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferAllocation {
    pub owner_device_id: String,
    pub component_id: String,
    pub slot: usize,
    pub byte_capacity: usize,
    pub signal_ids: Vec<String>,
    pub device_ids: Vec<String>,
    pub input_use_count: usize,
    pub output_use_count: usize,
}

pub struct VulkanDistributedActivationBuffers {
    pub plan: VulkanDistributedActivationBufferPlan,
    pub lane_capacity: usize,
    pub allocations: Vec<VulkanDistributedActivationBuffer>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub total_shared_byte_capacity: usize,
}

impl VulkanDistributedActivationBuffers {
    pub fn allocate<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        Self::allocate_for_lanes(plan, 1, device_for)
    }

    pub fn allocate_for_lanes<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        lane_capacity: usize,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        if lane_capacity == 0 {
            return Err(VulkanDistributedActivationBufferError(
                "distributed activation lane capacity must not be zero".to_string(),
            ));
        }
        let mut allocations = Vec::with_capacity(plan.allocations.len());
        let mut import_count = 0usize;
        let mut total_shared_byte_capacity = 0usize;
        for planned in &plan.allocations {
            let byte_capacity = planned
                .byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(format!(
                        "distributed activation {}.slot_{} lane capacity overflowed",
                        planned.component_id, planned.slot
                    ))
                })?;
            let owner = device_for(&planned.owner_device_id).map_err(|error| {
                VulkanDistributedActivationBufferError(format!(
                    "failed to resolve distributed activation owner {:?}: {error}",
                    planned.owner_device_id
                ))
            })?;
            let peers = planned
                .device_ids
                .iter()
                .filter(|device_id| *device_id != &planned.owner_device_id)
                .map(|device_id| {
                    device_for(device_id).map_err(|error| {
                        VulkanDistributedActivationBufferError(format!(
                            "failed to resolve distributed activation participant {device_id:?}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let shared_allocation = owner
                .create_shared_host_allocation(&peers, byte_capacity)
                .map_err(|error| {
                    VulkanDistributedActivationBufferError(format!(
                        "failed to allocate {} shared activation bytes for {}.slot_{}: {error}",
                        byte_capacity, planned.component_id, planned.slot
                    ))
                })?;
            let mut device_buffers = BTreeMap::new();
            for device_id in &planned.device_ids {
                let device = device_for(device_id).map_err(|error| {
                    VulkanDistributedActivationBufferError(format!(
                        "failed to resolve distributed activation participant {device_id:?}: {error}"
                    ))
                })?;
                let buffer = Arc::new(
                    device
                        .import_shared_host_buffer(Arc::clone(&shared_allocation))
                        .map_err(|error| {
                            VulkanDistributedActivationBufferError(format!(
                                "failed to import {}.slot_{} on {device_id:?}: {error}",
                                planned.component_id, planned.slot
                            ))
                        })?,
                );
                if device_buffers.insert(device_id.clone(), buffer).is_some() {
                    return Err(VulkanDistributedActivationBufferError(format!(
                        "distributed activation {}.slot_{} repeats device {device_id:?}",
                        planned.component_id, planned.slot
                    )));
                }
            }
            import_count = import_count
                .checked_add(device_buffers.len())
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed activation import count overflowed".to_string(),
                    )
                })?;
            total_shared_byte_capacity = total_shared_byte_capacity
                .checked_add(byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed activation byte capacity overflowed".to_string(),
                    )
                })?;
            allocations.push(VulkanDistributedActivationBuffer {
                planned: planned.clone(),
                shared_allocation,
                device_buffers,
            });
        }

        Ok(Self {
            plan: plan.clone(),
            lane_capacity,
            allocation_count: allocations.len(),
            allocations,
            import_count,
            total_shared_byte_capacity,
        })
    }

    pub fn activation_buffer(
        &self,
        owner_device_id: &str,
        component_id: &str,
        slot: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.allocations
            .iter()
            .find(|allocation| {
                allocation.planned.owner_device_id == owner_device_id
                    && allocation.planned.component_id == component_id
                    && allocation.planned.slot == slot
            })
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub fn activation_overrides_for_owner_device(
        &self,
        owner_device_id: &str,
    ) -> Vec<VulkanActivationSlotBufferOverride> {
        self.allocations
            .iter()
            .filter(|allocation| allocation.planned.owner_device_id == owner_device_id)
            .filter_map(|allocation| {
                allocation
                    .device_buffers
                    .get(owner_device_id)
                    .map(|buffer| VulkanActivationSlotBufferOverride {
                        component_id: allocation.planned.component_id.clone(),
                        slot: allocation.planned.slot,
                        buffer: Arc::clone(buffer),
                    })
            })
            .collect()
    }
}

pub struct VulkanDistributedActivationBuffer {
    pub planned: VulkanDistributedActivationBufferAllocation,
    pub shared_allocation: Arc<VulkanSharedHostAllocation>,
    pub device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferError(pub String);

impl Display for VulkanDistributedActivationBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedActivationBufferError {}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedActivationBufferAllocationKey {
    owner_device_id: String,
    component_id: String,
    slot: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDistributedActivationAccess {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchShard {
    pub device_id: String,
    pub row_start: usize,
    pub row_count: usize,
    pub workgroup_count_x: u32,
    pub base_workgroup_z: u32,
    pub output_byte_offset: usize,
    pub output_byte_count: usize,
    pub parameters: Vec<VulkanDistributedParameterFragment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterFragment {
    pub binding: usize,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocationPlan {
    pub allocations: Vec<VulkanDistributedParameterAllocation>,
    pub allocation_count: usize,
    pub tensor_count: usize,
    pub total_byte_capacity: usize,
}

impl VulkanDistributedParameterAllocationPlan {
    pub fn from_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let device_ids = execution_plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut allocations = BTreeMap::<
            VulkanDistributedParameterAllocationKey,
            VulkanDistributedParameterAllocation,
        >::new();

        for dispatch in &execution_plan.dispatches {
            for shard in &dispatch.shards {
                if !device_ids.contains(shard.device_id.as_str()) {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed parameter shard for {}.{} uses device {:?} outside the execution pool",
                        dispatch.component_id, dispatch.node_id, shard.device_id
                    )));
                }
                for fragment in &shard.parameters {
                    let metadata = tensor_index.tensors.get(&fragment.tensor).ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed parameter fragment has no tensor metadata for {:?}",
                            fragment.tensor
                        ))
                    })?;
                    let tensor_byte_count = metadata.byte_count.ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has no byte count",
                            fragment.tensor
                        ))
                    })?;
                    if fragment.byte_count == 0 {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has an empty fragment",
                            fragment.tensor
                        )));
                    }
                    let byte_end = fragment
                        .byte_offset
                        .checked_add(fragment.byte_count)
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "distributed parameter tensor {:?} fragment range overflowed",
                                fragment.tensor
                            ))
                        })?;
                    if byte_end > tensor_byte_count {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has {tensor_byte_count} bytes but a fragment ends at {byte_end}",
                            fragment.tensor
                        )));
                    }
                    let key = VulkanDistributedParameterAllocationKey {
                        device_id: shard.device_id.clone(),
                        tensor: fragment.tensor.clone(),
                        byte_offset: fragment.byte_offset,
                        byte_count: fragment.byte_count,
                    };
                    if let Some(allocation) = allocations.get_mut(&key) {
                        allocation.use_count =
                            allocation.use_count.checked_add(1).ok_or_else(|| {
                                VulkanDistributedPlanError(format!(
                                    "distributed parameter tensor {:?} use count overflowed",
                                    fragment.tensor
                                ))
                            })?;
                    } else {
                        allocations.insert(
                            key,
                            VulkanDistributedParameterAllocation {
                                device_id: shard.device_id.clone(),
                                tensor: fragment.tensor.clone(),
                                byte_offset: fragment.byte_offset,
                                byte_count: fragment.byte_count,
                                use_count: 1,
                            },
                        );
                    }
                }
            }
        }

        validate_tensor_partition_coverage(allocations.values(), tensor_index)?;
        let total_byte_capacity = allocations.values().try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed parameter allocation byte count overflowed".to_string(),
                )
            })
        })?;
        let tensor_count = allocations
            .values()
            .map(|allocation| allocation.tensor.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let allocations = allocations.into_values().collect::<Vec<_>>();

        Ok(Self {
            allocation_count: allocations.len(),
            allocations,
            tensor_count,
            total_byte_capacity,
        })
    }

    pub fn load_from_tensor_index<F>(
        &self,
        tensor_index: &TensorIndex,
        mut write: F,
    ) -> Result<VulkanDistributedParameterLoadReport, VulkanDistributedParameterLoadError>
    where
        F: FnMut(
            &VulkanDistributedParameterAllocation,
            &[u8],
        ) -> Result<(), VulkanDistributedParameterLoadError>,
    {
        let mut allocations_by_tensor = BTreeMap::<
            &str,
            BTreeMap<(usize, usize), Vec<&VulkanDistributedParameterAllocation>>,
        >::new();
        for allocation in &self.allocations {
            allocations_by_tensor
                .entry(&allocation.tensor)
                .or_default()
                .entry((allocation.byte_offset, allocation.byte_count))
                .or_default()
                .push(allocation);
        }

        let mut total_bytes_read = 0usize;
        let mut total_bytes_written = 0usize;
        let mut write_count = 0usize;
        let mut source_files = BTreeSet::new();
        for (tensor, ranges) in allocations_by_tensor {
            let storage = TensorStorage::from_index(tensor_index, tensor)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))?;
            let storage_ranges = ranges
                .keys()
                .map(|(byte_offset, byte_count)| TensorStorageRange {
                    byte_offset: *byte_offset,
                    byte_count: *byte_count,
                })
                .collect::<Vec<_>>();
            let payloads = storage
                .read_partitions(&storage_ranges)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))?;
            total_bytes_read = total_bytes_read
                .checked_add(storage.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterLoadError(
                        "distributed parameter read byte count overflowed".to_string(),
                    )
                })?;
            source_files.insert(storage.source_file);

            for (((_, _), allocations), payload) in ranges.into_iter().zip(payloads) {
                for allocation in allocations {
                    write(allocation, &payload)?;
                    total_bytes_written = total_bytes_written
                        .checked_add(payload.len())
                        .ok_or_else(|| {
                            VulkanDistributedParameterLoadError(
                                "distributed parameter written byte count overflowed".to_string(),
                            )
                        })?;
                    write_count = write_count.checked_add(1).ok_or_else(|| {
                        VulkanDistributedParameterLoadError(
                            "distributed parameter write count overflowed".to_string(),
                        )
                    })?;
                }
            }
        }

        Ok(VulkanDistributedParameterLoadReport {
            tensor_count: self.tensor_count,
            source_file_count: source_files.len(),
            allocation_count: self.allocation_count,
            write_count,
            total_bytes_read,
            total_bytes_written,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocation {
    pub device_id: String,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
    pub use_count: usize,
}

pub struct VulkanDistributedParameterBuffers {
    pub plan: VulkanDistributedParameterAllocationPlan,
    pub buffers: Vec<VulkanDistributedParameterBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanDistributedParameterBuffers {
    pub fn allocate_and_load<'a, F, E>(
        plan: &VulkanDistributedParameterAllocationPlan,
        tensor_index: &TensorIndex,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedParameterBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut buffers = Vec::with_capacity(plan.allocations.len());
        let mut buffer_index = BTreeMap::new();
        let mut total_byte_capacity = 0usize;
        for allocation in &plan.allocations {
            let device = device_for(&allocation.device_id).map_err(|error| {
                VulkanDistributedParameterBufferError(format!(
                    "failed to resolve distributed parameter device {:?}: {error}",
                    allocation.device_id
                ))
            })?;
            let buffer = device
                .create_resident_buffer(allocation.byte_count)
                .map_err(|error| {
                    VulkanDistributedParameterBufferError(format!(
                        "failed to allocate {} distributed bytes for tensor {:?} on {:?}: {error}",
                        allocation.byte_count, allocation.tensor, allocation.device_id
                    ))
                })?;
            total_byte_capacity = total_byte_capacity
                .checked_add(allocation.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(
                        "distributed parameter buffer byte capacity overflowed".to_string(),
                    )
                })?;
            let key = VulkanDistributedParameterAllocationKey::from(allocation);
            if buffer_index.insert(key, buffers.len()).is_some() {
                return Err(VulkanDistributedParameterBufferError(format!(
                    "distributed parameter buffer repeats tensor {:?} range {}..{} on {:?}",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                )));
            }
            buffers.push(VulkanDistributedParameterBufferAllocation {
                allocation: allocation.clone(),
                buffer,
            });
        }
        plan.load_from_tensor_index(tensor_index, |allocation, bytes| {
            let key = VulkanDistributedParameterAllocationKey::from(allocation);
            let index = *buffer_index.get(&key).ok_or_else(|| {
                VulkanDistributedParameterLoadError(format!(
                    "distributed parameter buffer for tensor {:?} range {}..{} on {:?} is missing",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                ))
            })?;
            buffers[index]
                .buffer
                .write_bytes(bytes)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))
        })
        .map_err(|error| VulkanDistributedParameterBufferError(error.to_string()))?;

        Ok(Self {
            plan: plan.clone(),
            buffers,
            total_byte_capacity,
        })
    }

    pub fn parameter_buffer(
        &self,
        device_id: &str,
        tensor: &str,
        byte_offset: usize,
        byte_count: usize,
    ) -> Option<&VulkanDistributedParameterBufferAllocation> {
        self.buffers.iter().find(|buffer| {
            buffer.allocation.device_id == device_id
                && buffer.allocation.tensor == tensor
                && buffer.allocation.byte_offset == byte_offset
                && buffer.allocation.byte_count == byte_count
        })
    }
}

pub struct VulkanDistributedParameterBufferAllocation {
    pub allocation: VulkanDistributedParameterAllocation,
    pub buffer: VulkanResidentBuffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterBufferError(pub String);

impl Display for VulkanDistributedParameterBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedParameterBufferError {}

impl From<VulkanError> for VulkanDistributedParameterBufferError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}
