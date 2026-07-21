use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::stream_plan::{TensorIndex, TensorMetadata};
use crate::tensor_storage::{TensorStorage, TensorStorageRange};
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanResidentBuffer, VulkanResidentKernelBufferAccess,
    VulkanResidentKernelBufferBinding, VulkanResidentKernelDispatch, VulkanResidentKernelSequence,
    VulkanResidentKernelSequenceStep, VulkanSharedHostAllocation,
};
use crate::vulkan_stream_circuit::{
    VulkanActivationSlotBufferOverride, VulkanDescriptorResourceAddress,
    VulkanLoadedReusableKernelArtifactManifest, VulkanPreparedDispatch, VulkanPreparedDispatchPlan,
    VulkanReusableKernelArtifactManifest,
};

const DISTRIBUTABLE_PARALLEL_PROJECTION_OP: &str = "parallel_linear_silu_multiply";
const BF16_BYTE_COUNT: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedExecutionPlan {
    pub device_ids: Vec<String>,
    pub storage_buffer_offset_alignment: usize,
    pub dispatches: Vec<VulkanDistributedDispatchPlan>,
    pub shared_input_byte_capacity: usize,
    pub shared_output_byte_capacity: usize,
    pub distributed_parameter_byte_count: usize,
}

impl VulkanDistributedExecutionPlan {
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
                if dispatch.op != DISTRIBUTABLE_PARALLEL_PROJECTION_OP {
                    continue;
                }
                let artifact = artifact_manifest
                    .artifacts
                    .iter()
                    .find(|artifact| artifact.family_id == dispatch.reusable_family_id)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed dispatch {}.{} has no artifact for family {:?}",
                            dispatch.pedal_id, dispatch.node_id, dispatch.reusable_family_id
                        ))
                    })?;
                let planned = plan_dispatch(
                    owner_device_id,
                    dispatch,
                    tensor_index,
                    device_ids,
                    artifact.workgroup_count_x,
                    storage_buffer_offset_alignment,
                )?;
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

        Ok(Self {
            device_ids: device_ids.to_vec(),
            storage_buffer_offset_alignment,
            dispatches,
            shared_input_byte_capacity,
            shared_output_byte_capacity,
            distributed_parameter_byte_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchPlan {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub pedal_id: String,
    pub node_id: String,
    pub reusable_family_id: String,
    pub input_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub output_rows: usize,
    pub input_width: usize,
    pub row_alignment: usize,
    pub input_activation: VulkanDistributedActivationSlot,
    pub output_activation: VulkanDistributedActivationSlot,
    pub distributed_parameter_byte_count: usize,
    pub shards: Vec<VulkanDistributedDispatchShard>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationSlot {
    pub pedal_id: String,
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
                    dispatch.pedal_id, dispatch.node_id
                )));
            }
            if !participant_device_ids.contains(dispatch.owner_device_id.as_str()) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} does not include its owner {:?}",
                    dispatch.pedal_id, dispatch.node_id, dispatch.owner_device_id
                )));
            }
            if let Some(device_id) = participant_device_ids
                .iter()
                .find(|device_id| !device_ids.contains(**device_id))
            {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} uses device {device_id:?} outside the execution pool",
                    dispatch.pedal_id, dispatch.node_id
                )));
            }

            accumulate_activation_allocation(
                &mut allocations,
                &dispatch.owner_device_id,
                &dispatch.input_activation,
                &participant_device_ids,
                VulkanDistributedActivationAccess::Input,
            )?;
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
        pedal_id: &str,
        slot: usize,
    ) -> Option<&VulkanDistributedActivationBufferAllocation> {
        self.allocations.iter().find(|allocation| {
            allocation.owner_device_id == owner_device_id
                && allocation.pedal_id == pedal_id
                && allocation.slot == slot
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferAllocation {
    pub owner_device_id: String,
    pub pedal_id: String,
    pub slot: usize,
    pub byte_capacity: usize,
    pub signal_ids: Vec<String>,
    pub device_ids: Vec<String>,
    pub input_use_count: usize,
    pub output_use_count: usize,
}

pub struct VulkanDistributedActivationBuffers {
    pub plan: VulkanDistributedActivationBufferPlan,
    pub allocations: Vec<VulkanDistributedActivationBuffer>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub total_shared_byte_capacity: usize,
}

impl VulkanDistributedActivationBuffers {
    pub fn allocate<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut allocations = Vec::with_capacity(plan.allocations.len());
        let mut import_count = 0usize;
        let mut total_shared_byte_capacity = 0usize;
        for planned in &plan.allocations {
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
                .create_shared_host_allocation(&peers, planned.byte_capacity)
                .map_err(|error| {
                    VulkanDistributedActivationBufferError(format!(
                        "failed to allocate {} shared activation bytes for {}.slot_{}: {error}",
                        planned.byte_capacity, planned.pedal_id, planned.slot
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
                                planned.pedal_id, planned.slot
                            ))
                        })?,
                );
                if device_buffers.insert(device_id.clone(), buffer).is_some() {
                    return Err(VulkanDistributedActivationBufferError(format!(
                        "distributed activation {}.slot_{} repeats device {device_id:?}",
                        planned.pedal_id, planned.slot
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
                .checked_add(planned.byte_capacity)
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
            allocation_count: allocations.len(),
            allocations,
            import_count,
            total_shared_byte_capacity,
        })
    }

    pub fn activation_buffer(
        &self,
        owner_device_id: &str,
        pedal_id: &str,
        slot: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.allocations
            .iter()
            .find(|allocation| {
                allocation.planned.owner_device_id == owner_device_id
                    && allocation.planned.pedal_id == pedal_id
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
                        pedal_id: allocation.planned.pedal_id.clone(),
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
    pedal_id: String,
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
                        dispatch.pedal_id, dispatch.node_id, shard.device_id
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

pub struct VulkanDistributedDispatchRunners {
    pub dispatches: Vec<VulkanDistributedDispatchRunner>,
    pub dispatch_count: usize,
    pub shard_count: usize,
}

impl VulkanDistributedDispatchRunners {
    pub fn create<'a, F, E>(
        execution_plan: &VulkanDistributedExecutionPlan,
        parameter_buffers: &VulkanDistributedParameterBuffers,
        activation_buffers: &VulkanDistributedActivationBuffers,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut dispatches = Vec::with_capacity(execution_plan.dispatches.len());
        let mut shard_count = 0usize;
        for planned_dispatch in &execution_plan.dispatches {
            let artifact = loaded_manifest
                .artifact(&planned_dispatch.reusable_family_id)
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "distributed dispatch {}.{} is missing loaded family {:?}",
                        planned_dispatch.pedal_id,
                        planned_dispatch.node_id,
                        planned_dispatch.reusable_family_id
                    ))
                })?;
            let mut shards = Vec::with_capacity(planned_dispatch.shards.len());
            for planned_shard in &planned_dispatch.shards {
                let device = device_for(&planned_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed shard device {:?}: {error}",
                        planned_shard.device_id
                    ))
                })?;
                let input = activation_buffers
                    .activation_buffer(
                        &planned_dispatch.owner_device_id,
                        &planned_dispatch.input_activation.pedal_id,
                        planned_dispatch.input_activation.slot,
                        &planned_shard.device_id,
                    )
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "distributed dispatch {}.{} has no input activation on {:?}",
                            planned_dispatch.pedal_id,
                            planned_dispatch.node_id,
                            planned_shard.device_id
                        ))
                    })?;
                let output = activation_buffers
                    .activation_buffer(
                        &planned_dispatch.owner_device_id,
                        &planned_dispatch.output_activation.pedal_id,
                        planned_dispatch.output_activation.slot,
                        &planned_shard.device_id,
                    )
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "distributed dispatch {}.{} has no output activation on {:?}",
                            planned_dispatch.pedal_id,
                            planned_dispatch.node_id,
                            planned_shard.device_id
                        ))
                    })?;
                let mut bindings = Vec::with_capacity(2 + planned_shard.parameters.len());
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        0,
                        input,
                        planned_dispatch.input_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                );
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        output,
                        planned_shard.output_byte_count,
                    )
                    .with_byte_offset(planned_shard.output_byte_offset)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                );
                for fragment in &planned_shard.parameters {
                    let allocation = parameter_buffers
                        .parameter_buffer(
                            &planned_shard.device_id,
                            &fragment.tensor,
                            fragment.byte_offset,
                            fragment.byte_count,
                        )
                        .ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "distributed dispatch {}.{} has no tensor {:?} range at byte {} with length {} on {:?}",
                                planned_dispatch.pedal_id,
                                planned_dispatch.node_id,
                                fragment.tensor,
                                fragment.byte_offset,
                                fragment.byte_count,
                                planned_shard.device_id
                            ))
                        })?;
                    let binding = u32::try_from(fragment.binding).map_err(|_| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "distributed descriptor binding {} exceeds u32",
                            fragment.binding
                        ))
                    })?;
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            binding,
                            &allocation.buffer,
                            fragment.byte_count,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
                let resident_dispatch = device
                    .create_resident_kernel_dispatch(
                        &artifact.words,
                        &bindings,
                        planned_shard.workgroup_count_x,
                        artifact.artifact.local_size_x,
                        0,
                    )
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to create distributed dispatch {}.{} shard on {:?}: {error}",
                            planned_dispatch.pedal_id,
                            planned_dispatch.node_id,
                            planned_shard.device_id
                        ))
                    })?;
                let sequence = device.create_resident_kernel_sequence().map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to create distributed sequence {}.{} shard on {:?}: {error}",
                        planned_dispatch.pedal_id,
                        planned_dispatch.node_id,
                        planned_shard.device_id
                    ))
                })?;
                device
                    .record_resident_kernel_sequence(
                        &sequence,
                        &[VulkanResidentKernelSequenceStep::new(
                            &resident_dispatch,
                            &[],
                        )],
                    )
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to record distributed dispatch {}.{} shard on {:?}: {error}",
                            planned_dispatch.pedal_id,
                            planned_dispatch.node_id,
                            planned_shard.device_id
                        ))
                    })?;
                shards.push(VulkanDistributedDispatchShardRunner {
                    planned: planned_shard.clone(),
                    resident_dispatch,
                    sequence,
                });
                shard_count = shard_count.checked_add(1).ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed dispatch shard count overflowed".to_string(),
                    )
                })?;
            }
            dispatches.push(VulkanDistributedDispatchRunner {
                planned: planned_dispatch.clone(),
                shards,
            });
        }

        Ok(Self {
            dispatch_count: dispatches.len(),
            dispatches,
            shard_count,
        })
    }

    pub fn dispatch(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedDispatchRunner> {
        self.dispatches.iter().find(|dispatch| {
            dispatch.planned.owner_device_id == owner_device_id
                && dispatch.planned.dispatch_index == dispatch_index
        })
    }

    pub fn run_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dispatch = self
            .dispatch(owner_device_id, dispatch_index)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
                ))
            })?;
        let mut submitted: Vec<(&VulkanComputeDevice, &VulkanDistributedDispatchShardRunner)> =
            Vec::with_capacity(dispatch.shards.len());
        for shard in &dispatch.shards {
            let device = device_for(&shard.planned.device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed shard device {:?}: {error}",
                    shard.planned.device_id
                ))
            })?;
            if let Err(error) = device.submit_recorded_resident_kernel_sequence(&shard.sequence) {
                for (submitted_device, submitted_shard) in &submitted {
                    let _ =
                        submitted_device.wait_resident_kernel_sequence(&submitted_shard.sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.pedal_id, dispatch.planned.node_id, shard.planned.device_id
                )));
            }
            submitted.push((device, shard));
        }
        let mut first_wait_error = None;
        for (device, shard) in &submitted {
            if let Err(error) = device.wait_resident_kernel_sequence(&shard.sequence)
                && first_wait_error.is_none()
            {
                first_wait_error = Some(format!(
                    "failed waiting for distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.pedal_id, dispatch.planned.node_id, shard.planned.device_id
                ));
            }
        }
        if let Some(error) = first_wait_error {
            return Err(VulkanDistributedDispatchRunnerError(error));
        }

        Ok(VulkanDistributedDispatchRun {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index,
            pedal_id: dispatch.planned.pedal_id.clone(),
            node_id: dispatch.planned.node_id.clone(),
            shard_count: dispatch.shards.len(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRun {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub pedal_id: String,
    pub node_id: String,
    pub shard_count: usize,
}

pub struct VulkanDistributedDispatchRunner {
    pub planned: VulkanDistributedDispatchPlan,
    pub shards: Vec<VulkanDistributedDispatchShardRunner>,
}

pub struct VulkanDistributedDispatchShardRunner {
    pub planned: VulkanDistributedDispatchShard,
    pub resident_dispatch: VulkanResidentKernelDispatch,
    pub sequence: VulkanResidentKernelSequence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRunnerError(pub String);

impl Display for VulkanDistributedDispatchRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedDispatchRunnerError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterExclusionPlan {
    pub devices: Vec<VulkanDistributedDeviceParameterExclusions>,
    pub device_count: usize,
    pub unique_tensor_count: usize,
    pub excluded_full_allocation_count: usize,
    pub excluded_full_byte_capacity: usize,
}

impl VulkanDistributedParameterExclusionPlan {
    pub fn from_execution_and_prepared_plans(
        execution_plan: &VulkanDistributedExecutionPlan,
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let mut distributed_dispatch_tensors =
            BTreeMap::<VulkanDistributedDispatchKey, BTreeSet<String>>::new();
        let mut target_tensors = BTreeSet::<(String, String)>::new();
        for dispatch in &execution_plan.dispatches {
            let key = VulkanDistributedDispatchKey::from_distributed(dispatch);
            let tensors = dispatch
                .shards
                .iter()
                .flat_map(|shard| shard.parameters.iter())
                .map(|fragment| fragment.tensor.clone())
                .collect::<BTreeSet<_>>();
            if tensors.is_empty() {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has no parameter tensors",
                    dispatch.pedal_id, dispatch.node_id
                )));
            }
            if distributed_dispatch_tensors
                .insert(key, tensors.clone())
                .is_some()
            {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed execution plan repeats dispatch {}.{} at index {} on {:?}",
                    dispatch.pedal_id,
                    dispatch.node_id,
                    dispatch.dispatch_index,
                    dispatch.owner_device_id
                )));
            }
            target_tensors.extend(
                tensors
                    .into_iter()
                    .map(|tensor| (dispatch.owner_device_id.clone(), tensor)),
            );
        }

        let mut prepared_device_ids = BTreeSet::new();
        let mut matched_dispatches = BTreeSet::new();
        for (device_id, prepared_plan) in prepared_plans {
            if !prepared_device_ids.insert(*device_id) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed parameter exclusion repeats prepared plan for device {device_id:?}"
                )));
            }
            for dispatch in &prepared_plan.dispatches {
                let key = VulkanDistributedDispatchKey::from_prepared(device_id, dispatch);
                let parameter_tensors = dispatch
                    .descriptors
                    .iter()
                    .filter_map(|descriptor| match &descriptor.resource {
                        VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                            Some(tensor.clone())
                        }
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>();
                if let Some(distributed_tensors) = distributed_dispatch_tensors.get(&key) {
                    if &parameter_tensors != distributed_tensors {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed dispatch {}.{} parameter tensors changed between preparation and physical lowering",
                            dispatch.pedal_id, dispatch.node_id
                        )));
                    }
                    matched_dispatches.insert(key);
                } else if let Some(tensor) = parameter_tensors.iter().find(|tensor| {
                    target_tensors.contains(&((*device_id).to_string(), (*tensor).clone()))
                }) {
                    return Err(VulkanDistributedPlanError(format!(
                        "cannot exclude distributed tensor {tensor:?} on {device_id:?}; canonical dispatch {}.{} still uses it",
                        dispatch.pedal_id, dispatch.node_id
                    )));
                }
            }
        }
        if let Some(missing) = distributed_dispatch_tensors
            .keys()
            .find(|key| !matched_dispatches.contains(*key))
        {
            return Err(VulkanDistributedPlanError(format!(
                "distributed dispatch {}.{} at index {} on {:?} is absent from prepared plans",
                missing.pedal_id, missing.node_id, missing.dispatch_index, missing.owner_device_id
            )));
        }

        let mut tensors_by_device = BTreeMap::<String, Vec<String>>::new();
        let mut excluded_full_byte_capacity = 0usize;
        for (device_id, tensor) in &target_tensors {
            let byte_count = tensor_index
                .tensors
                .get(tensor)
                .and_then(|metadata| metadata.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "distributed exclusion tensor {tensor:?} has no byte count"
                    ))
                })?;
            excluded_full_byte_capacity = excluded_full_byte_capacity
                .checked_add(byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed exclusion byte capacity overflowed".to_string(),
                    )
                })?;
            tensors_by_device
                .entry(device_id.clone())
                .or_default()
                .push(tensor.clone());
        }
        let devices = tensors_by_device
            .into_iter()
            .map(|(device_id, tensors)| {
                let total_byte_capacity = tensors.iter().try_fold(0usize, |total, tensor| {
                    total
                        .checked_add(
                            tensor_index.tensors[tensor]
                                .byte_count
                                .expect("validated distributed exclusion byte count"),
                        )
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "distributed device exclusion byte capacity overflowed".to_string(),
                            )
                        })
                })?;
                Ok(VulkanDistributedDeviceParameterExclusions {
                    device_id,
                    tensors,
                    total_byte_capacity,
                })
            })
            .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
        let unique_tensor_count = target_tensors
            .iter()
            .map(|(_, tensor)| tensor.as_str())
            .collect::<BTreeSet<_>>()
            .len();

        Ok(Self {
            device_count: devices.len(),
            devices,
            unique_tensor_count,
            excluded_full_allocation_count: target_tensors.len(),
            excluded_full_byte_capacity,
        })
    }

    pub fn tensors_for_device(&self, device_id: &str) -> BTreeSet<String> {
        self.devices
            .iter()
            .find(|device| device.device_id == device_id)
            .map(|device| device.tensors.iter().cloned().collect())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDeviceParameterExclusions {
    pub device_id: String,
    pub tensors: Vec<String>,
    pub total_byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedDispatchKey {
    owner_device_id: String,
    dispatch_index: usize,
    pedal_id: String,
    node_id: String,
}

impl VulkanDistributedDispatchKey {
    fn from_distributed(dispatch: &VulkanDistributedDispatchPlan) -> Self {
        Self {
            owner_device_id: dispatch.owner_device_id.clone(),
            dispatch_index: dispatch.dispatch_index,
            pedal_id: dispatch.pedal_id.clone(),
            node_id: dispatch.node_id.clone(),
        }
    }

    fn from_prepared(owner_device_id: &str, dispatch: &VulkanPreparedDispatch) -> Self {
        Self {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index: dispatch.dispatch_index,
            pedal_id: dispatch.pedal_id.clone(),
            node_id: dispatch.node_id.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterLoadReport {
    pub tensor_count: usize,
    pub source_file_count: usize,
    pub allocation_count: usize,
    pub write_count: usize,
    pub total_bytes_read: usize,
    pub total_bytes_written: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterLoadError(pub String);

impl Display for VulkanDistributedParameterLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedParameterLoadError {}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedParameterAllocationKey {
    device_id: String,
    tensor: String,
    byte_offset: usize,
    byte_count: usize,
}

impl From<&VulkanDistributedParameterAllocation> for VulkanDistributedParameterAllocationKey {
    fn from(allocation: &VulkanDistributedParameterAllocation) -> Self {
        Self {
            device_id: allocation.device_id.clone(),
            tensor: allocation.tensor.clone(),
            byte_offset: allocation.byte_offset,
            byte_count: allocation.byte_count,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedPlanError(pub String);

impl Display for VulkanDistributedPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedPlanError {}

fn validate_device_pool(device_ids: &[String]) -> Result<(), VulkanDistributedPlanError> {
    if device_ids.is_empty() {
        return Err(VulkanDistributedPlanError(
            "distributed execution device pool must not be empty".to_string(),
        ));
    }
    let mut unique = BTreeSet::new();
    if let Some(device_id) = device_ids
        .iter()
        .find(|device_id| !unique.insert(device_id.as_str()))
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed execution device pool repeats {device_id:?}"
        )));
    }
    Ok(())
}

fn accumulate_activation_allocation(
    allocations: &mut BTreeMap<
        VulkanDistributedActivationBufferAllocationKey,
        VulkanDistributedActivationBufferAllocation,
    >,
    owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
    participant_device_ids: &BTreeSet<&str>,
    access: VulkanDistributedActivationAccess,
) -> Result<(), VulkanDistributedPlanError> {
    if activation.byte_capacity == 0 {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} has zero capacity",
            activation.pedal_id, activation.slot
        )));
    }
    if activation.signal_byte_capacity == 0
        || activation.signal_byte_capacity > activation.byte_capacity
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} has signal {:?} capacity {} outside its {}-byte slot",
            activation.pedal_id,
            activation.slot,
            activation.signal_id,
            activation.signal_byte_capacity,
            activation.byte_capacity
        )));
    }
    let key = VulkanDistributedActivationBufferAllocationKey {
        owner_device_id: owner_device_id.to_string(),
        pedal_id: activation.pedal_id.clone(),
        slot: activation.slot,
    };
    let allocation =
        allocations
            .entry(key)
            .or_insert_with(|| VulkanDistributedActivationBufferAllocation {
                owner_device_id: owner_device_id.to_string(),
                pedal_id: activation.pedal_id.clone(),
                slot: activation.slot,
                byte_capacity: activation.byte_capacity,
                signal_ids: Vec::new(),
                device_ids: Vec::new(),
                input_use_count: 0,
                output_use_count: 0,
            });
    if allocation.byte_capacity != activation.byte_capacity {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} has conflicting capacities {} and {}",
            activation.pedal_id,
            activation.slot,
            allocation.byte_capacity,
            activation.byte_capacity
        )));
    }
    if !allocation.signal_ids.contains(&activation.signal_id) {
        allocation.signal_ids.push(activation.signal_id.clone());
        allocation.signal_ids.sort();
    }
    for device_id in participant_device_ids {
        if !allocation
            .device_ids
            .iter()
            .any(|existing| existing == device_id)
        {
            allocation.device_ids.push((*device_id).to_string());
        }
    }
    allocation.device_ids.sort();
    match access {
        VulkanDistributedActivationAccess::Input => {
            allocation.input_use_count =
                allocation.input_use_count.checked_add(1).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation input use count overflowed".to_string(),
                    )
                })?;
        }
        VulkanDistributedActivationAccess::Output => {
            allocation.output_use_count =
                allocation.output_use_count.checked_add(1).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation output use count overflowed".to_string(),
                    )
                })?;
        }
    }
    Ok(())
}

fn validate_tensor_partition_coverage<'a>(
    allocations: impl Iterator<Item = &'a VulkanDistributedParameterAllocation>,
    tensor_index: &TensorIndex,
) -> Result<(), VulkanDistributedPlanError> {
    let mut ranges_by_tensor = BTreeMap::<&str, BTreeSet<(usize, usize)>>::new();
    for allocation in allocations {
        ranges_by_tensor
            .entry(&allocation.tensor)
            .or_default()
            .insert((allocation.byte_offset, allocation.byte_count));
    }
    for (tensor, ranges) in ranges_by_tensor {
        let tensor_byte_count = tensor_index
            .tensors
            .get(tensor)
            .and_then(|metadata| metadata.byte_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "distributed parameter tensor {tensor:?} has no byte count"
                ))
            })?;
        let mut next_offset = 0usize;
        for (byte_offset, byte_count) in ranges {
            if byte_offset != next_offset {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed parameter tensor {tensor:?} has a gap or overlap at byte {next_offset}"
                )));
            }
            next_offset = next_offset.checked_add(byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "distributed parameter tensor {tensor:?} partition overflowed"
                ))
            })?;
        }
        if next_offset != tensor_byte_count {
            return Err(VulkanDistributedPlanError(format!(
                "distributed parameter tensor {tensor:?} partition covers {next_offset} of {tensor_byte_count} bytes"
            )));
        }
    }
    Ok(())
}

fn plan_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
) -> Result<VulkanDistributedDispatchPlan, VulkanDistributedPlanError> {
    if !dispatch.push_constants.is_empty() {
        return Err(dispatch_error(
            dispatch,
            "cannot yet preserve push constants across physical shards".to_string(),
        ));
    }
    let parameter_descriptors = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                Some((descriptor.binding, tensor.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [
        (first_binding, first_tensor),
        (second_binding, second_tensor),
    ] = parameter_descriptors.as_slice()
    else {
        return Err(dispatch_error(
            dispatch,
            format!(
                "requires exactly two projection matrices, found {} parameters",
                parameter_descriptors.len()
            ),
        ));
    };
    let first = projection_metadata(tensor_index, dispatch, first_tensor)?;
    let second = projection_metadata(tensor_index, dispatch, second_tensor)?;
    if first.shape != second.shape {
        return Err(dispatch_error(
            dispatch,
            format!(
                "projection shapes {:?} and {:?} do not match",
                first.shape, second.shape
            ),
        ));
    }
    let output_rows = first.shape[0];
    let input_width = first.shape[1];
    let artifact_workgroup_count = usize::try_from(artifact_workgroup_count_x).map_err(|_| {
        dispatch_error(
            dispatch,
            "artifact workgroup count exceeds usize".to_string(),
        )
    })?;
    if artifact_workgroup_count == 0 || output_rows % artifact_workgroup_count != 0 {
        return Err(dispatch_error(
            dispatch,
            format!(
                "output row count {output_rows} is incompatible with artifact workgroup count {artifact_workgroup_count}"
            ),
        ));
    }
    let workgroup_row_count = output_rows / artifact_workgroup_count;
    let mut row_alignment = least_common_multiple(
        workgroup_row_count,
        storage_buffer_offset_alignment / BF16_BYTE_COUNT,
    )
    .ok_or_else(|| dispatch_error(dispatch, "row alignment overflowed".to_string()))?;
    if [first.layout.as_deref(), second.layout.as_deref()]
        .contains(&Some("vulkan_bf16_row_pair_u32"))
    {
        row_alignment = least_common_multiple(row_alignment, 2)
            .ok_or_else(|| dispatch_error(dispatch, "row alignment overflowed".to_string()))?;
    }
    let input_byte_capacity = input_width
        .checked_mul(BF16_BYTE_COUNT)
        .ok_or_else(|| dispatch_error(dispatch, "input byte capacity overflowed".to_string()))?;
    let output_byte_capacity = output_rows
        .checked_mul(BF16_BYTE_COUNT)
        .ok_or_else(|| dispatch_error(dispatch, "output byte capacity overflowed".to_string()))?;
    let input_activation = activation_slot(dispatch, 0, input_byte_capacity, "input")?;
    let output_activation = activation_slot(dispatch, 1, output_byte_capacity, "output")?;

    let raw_shards = distribute_rows(
        output_rows,
        device_ids.len(),
        workgroup_row_count,
        row_alignment,
    )
    .map_err(|error| dispatch_error(dispatch, error))?;
    let first_row_bytes = tensor_row_bytes(dispatch, first_tensor, first, output_rows)?;
    let second_row_bytes = tensor_row_bytes(dispatch, second_tensor, second, output_rows)?;
    let mut distributed_parameter_byte_count = 0usize;
    let shards = device_ids
        .iter()
        .zip(raw_shards)
        .map(|(device_id, (row_start, row_count))| {
            let workgroup_count_x =
                u32::try_from(row_count / workgroup_row_count).map_err(|_| {
                    dispatch_error(dispatch, "shard workgroup count exceeds u32".to_string())
                })?;
            let first_fragment = parameter_fragment(
                *first_binding,
                first_tensor,
                first_row_bytes,
                row_start,
                row_count,
                dispatch,
            )?;
            let second_fragment = parameter_fragment(
                *second_binding,
                second_tensor,
                second_row_bytes,
                row_start,
                row_count,
                dispatch,
            )?;
            distributed_parameter_byte_count = distributed_parameter_byte_count
                .checked_add(first_fragment.byte_count)
                .and_then(|total| total.checked_add(second_fragment.byte_count))
                .ok_or_else(|| {
                    dispatch_error(
                        dispatch,
                        "shard parameter byte count overflowed".to_string(),
                    )
                })?;
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.clone(),
                row_start,
                row_count,
                workgroup_count_x,
                output_byte_offset: row_start.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                    dispatch_error(dispatch, "shard output offset overflowed".to_string())
                })?,
                output_byte_count: row_count.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                    dispatch_error(dispatch, "shard output size overflowed".to_string())
                })?,
                parameters: vec![first_fragment, second_fragment],
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;

    Ok(VulkanDistributedDispatchPlan {
        owner_device_id: owner_device_id.to_string(),
        dispatch_index: dispatch.dispatch_index,
        pedal_id: dispatch.pedal_id.clone(),
        node_id: dispatch.node_id.clone(),
        reusable_family_id: dispatch.reusable_family_id.clone(),
        input_byte_capacity,
        output_byte_capacity,
        output_rows,
        input_width,
        row_alignment,
        input_activation,
        output_activation,
        distributed_parameter_byte_count,
        shards,
    })
}

fn projection_metadata<'a>(
    tensor_index: &'a TensorIndex,
    dispatch: &VulkanPreparedDispatch,
    tensor: &str,
) -> Result<&'a TensorMetadata, VulkanDistributedPlanError> {
    let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
        dispatch_error(dispatch, format!("has no tensor metadata for {tensor:?}"))
    })?;
    if metadata.dtype != "BF16" || metadata.shape.len() != 2 {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} must be a rank-2 BF16 matrix, found {} {:?}",
                metadata.dtype, metadata.shape
            ),
        ));
    }
    if !matches!(
        metadata.layout.as_deref(),
        Some("row_major" | "vulkan_bf16_row_pair_u32")
    ) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} has non-shardable layout {:?}",
                metadata.layout
            ),
        ));
    }
    Ok(metadata)
}

fn tensor_row_bytes(
    dispatch: &VulkanPreparedDispatch,
    tensor: &str,
    metadata: &TensorMetadata,
    output_rows: usize,
) -> Result<usize, VulkanDistributedPlanError> {
    let expected = metadata
        .shape
        .iter()
        .try_fold(BF16_BYTE_COUNT, |bytes, dimension| {
            bytes.checked_mul(*dimension)
        });
    let expected = expected.ok_or_else(|| {
        dispatch_error(dispatch, format!("tensor {tensor:?} byte count overflowed"))
    })?;
    let byte_count = metadata.byte_count.unwrap_or(expected);
    if byte_count != expected || !byte_count.is_multiple_of(output_rows) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} byte count {byte_count} does not match BF16 shape {:?}",
                metadata.shape
            ),
        ));
    }
    Ok(byte_count / output_rows)
}

fn activation_slot(
    dispatch: &VulkanPreparedDispatch,
    binding: usize,
    required: usize,
    role: &str,
) -> Result<VulkanDistributedActivationSlot, VulkanDistributedPlanError> {
    let activation = dispatch
        .descriptors
        .iter()
        .find(|descriptor| descriptor.binding == binding)
        .and_then(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::ActivationSlot {
                pedal_id,
                signal_id,
                slot,
                byte_capacity,
                signal_byte_capacity,
            } => Some(VulkanDistributedActivationSlot {
                pedal_id: pedal_id.clone(),
                signal_id: signal_id.clone(),
                slot: *slot,
                byte_capacity: *byte_capacity,
                signal_byte_capacity: *signal_byte_capacity,
            }),
            _ => None,
        })
        .ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("has no resident {role} activation at binding {binding}"),
            )
        })?;
    if activation.signal_byte_capacity < required {
        return Err(dispatch_error(
            dispatch,
            format!(
                "{role} signal has {} bytes but requires {required}",
                activation.signal_byte_capacity
            ),
        ));
    }
    Ok(activation)
}

fn distribute_rows(
    row_count: usize,
    requested_shards: usize,
    workgroup_row_count: usize,
    shard_boundary_row_alignment: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if row_count == 0
        || requested_shards == 0
        || workgroup_row_count == 0
        || shard_boundary_row_alignment == 0
    {
        return Err("row distribution dimensions must not be zero".to_string());
    }
    if !row_count.is_multiple_of(workgroup_row_count)
        || !shard_boundary_row_alignment.is_multiple_of(workgroup_row_count)
    {
        return Err(format!(
            "row count {row_count} and shard boundary {shard_boundary_row_alignment} are incompatible with workgroup width {workgroup_row_count}"
        ));
    }
    let aligned_groups = row_count / shard_boundary_row_alignment;
    let tail_rows = row_count % shard_boundary_row_alignment;
    let shard_count = requested_shards.min(aligned_groups + usize::from(tail_rows != 0));
    let groups_per_shard = aligned_groups / shard_count;
    let remainder = aligned_groups % shard_count;
    let mut row_start = 0usize;
    let mut shards = Vec::with_capacity(shard_count);
    for shard_index in 0..shard_count {
        let group_count = groups_per_shard + usize::from(shard_index < remainder);
        let shard_rows = group_count
            .checked_mul(shard_boundary_row_alignment)
            .and_then(|rows| {
                if shard_index + 1 == shard_count {
                    rows.checked_add(tail_rows)
                } else {
                    Some(rows)
                }
            })
            .ok_or_else(|| "row shard size overflowed".to_string())?;
        if shard_rows == 0 {
            return Err("row distribution produced an empty shard".to_string());
        }
        shards.push((row_start, shard_rows));
        row_start = row_start
            .checked_add(shard_rows)
            .ok_or_else(|| "row shard offset overflowed".to_string())?;
    }
    Ok(shards)
}

fn parameter_fragment(
    binding: usize,
    tensor: &str,
    row_bytes: usize,
    row_start: usize,
    row_count: usize,
    dispatch: &VulkanPreparedDispatch,
) -> Result<VulkanDistributedParameterFragment, VulkanDistributedPlanError> {
    Ok(VulkanDistributedParameterFragment {
        binding,
        tensor: tensor.to_string(),
        byte_offset: row_start.checked_mul(row_bytes).ok_or_else(|| {
            dispatch_error(dispatch, "parameter shard offset overflowed".to_string())
        })?,
        byte_count: row_count.checked_mul(row_bytes).ok_or_else(|| {
            dispatch_error(
                dispatch,
                "parameter shard byte count overflowed".to_string(),
            )
        })?,
    })
}

fn least_common_multiple(left: usize, right: usize) -> Option<usize> {
    left.checked_mul(right / greatest_common_divisor(left, right))
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn dispatch_error(
    dispatch: &VulkanPreparedDispatch,
    message: String,
) -> VulkanDistributedPlanError {
    VulkanDistributedPlanError(format!(
        "distributed dispatch {}.{} {message}",
        dispatch.pedal_id, dispatch.node_id
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::stream_plan::TensorMetadata;
    use crate::vulkan_stream_circuit::{
        VulkanKernelDescriptorUsage, VulkanKernelScalarBinding, VulkanKernelScalarSource,
        VulkanResolvedDescriptorBinding, VulkanReusableKernelArtifact,
    };

    #[test]
    fn plans_balanced_parameter_and_output_shards_from_compiled_contracts() {
        let plan = fixture_plan("row_major");

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.shared_input_byte_capacity, 8);
        assert_eq!(plan.shared_output_byte_capacity, 24);
        assert_eq!(plan.storage_buffer_offset_alignment, 4);
        assert_eq!(plan.distributed_parameter_byte_count, 192);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.owner_device_id, "owner");
        assert_eq!(dispatch.row_alignment, 2);
        assert_eq!(dispatch.input_activation.pedal_id, "pedal");
        assert_eq!(dispatch.input_activation.signal_id, "normalized");
        assert_eq!(dispatch.input_activation.slot, 0);
        assert_eq!(dispatch.output_activation.pedal_id, "pedal");
        assert_eq!(dispatch.output_activation.signal_id, "hidden");
        assert_eq!(dispatch.output_activation.slot, 1);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.output_byte_offset,
                    shard.output_byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                ("owner", 0, 4, 2, 0, 8),
                ("helper-a", 4, 4, 2, 8, 8),
                ("helper-b", 8, 2, 1, 16, 4),
                ("helper-c", 10, 2, 1, 20, 4),
            ]
        );
        assert_eq!(
            dispatch.shards[1]
                .parameters
                .iter()
                .map(|fragment| (
                    fragment.binding,
                    fragment.tensor.as_str(),
                    fragment.byte_offset,
                    fragment.byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![(2, "gate", 32, 32), (3, "up", 32, 32)]
        );
    }

    #[test]
    fn preserves_packed_row_pairs_at_shard_boundaries() {
        let plan = fixture_plan("vulkan_bf16_row_pair_u32");

        assert_eq!(plan.dispatches[0].row_alignment, 2);
        assert!(
            plan.dispatches[0]
                .shards
                .iter()
                .all(|shard| shard.row_start % 2 == 0 && shard.row_count % 2 == 0)
        );
    }

    #[test]
    fn aligns_shared_output_offsets_and_keeps_a_workgroup_aligned_tail() {
        let plan = fixture_plan_result_with_alignment("row_major", 16).unwrap();
        let dispatch = &plan.dispatches[0];

        assert_eq!(dispatch.row_alignment, 8);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.output_byte_offset,
                ))
                .collect::<Vec<_>>(),
            vec![("owner", 0, 8, 4, 0), ("helper-a", 8, 4, 2, 16)]
        );
        assert!(dispatch.shards.iter().all(|shard| {
            shard
                .output_byte_offset
                .is_multiple_of(plan.storage_buffer_offset_alignment)
        }));
    }

    #[test]
    fn plans_one_shared_allocation_per_owner_activation_slot() {
        let execution_plan = fixture_plan("row_major");

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 2);
        assert_eq!(activation_plan.import_count, 8);
        assert_eq!(activation_plan.reference_count, 2);
        assert_eq!(activation_plan.total_shared_byte_capacity, 32);
        assert_eq!(
            activation_plan.allocation("owner", "pedal", 0).unwrap(),
            &VulkanDistributedActivationBufferAllocation {
                owner_device_id: "owner".to_string(),
                pedal_id: "pedal".to_string(),
                slot: 0,
                byte_capacity: 8,
                signal_ids: vec!["normalized".to_string()],
                device_ids: vec![
                    "helper-a".to_string(),
                    "helper-b".to_string(),
                    "helper-c".to_string(),
                    "owner".to_string(),
                ],
                input_use_count: 1,
                output_use_count: 0,
            }
        );
        assert_eq!(
            activation_plan
                .allocation("owner", "pedal", 1)
                .unwrap()
                .output_use_count,
            1
        );
    }

    #[test]
    fn reuses_shared_activation_allocations_across_repeated_dispatches() {
        let mut execution_plan = fixture_plan("row_major");
        let mut repeated = execution_plan.dispatches[0].clone();
        repeated.dispatch_index = 8;
        repeated.input_activation.signal_id = "normalized-again".to_string();
        execution_plan.dispatches.push(repeated);

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 2);
        assert_eq!(activation_plan.import_count, 8);
        assert_eq!(activation_plan.reference_count, 4);
        assert_eq!(activation_plan.total_shared_byte_capacity, 32);
        let input = activation_plan.allocation("owner", "pedal", 0).unwrap();
        assert_eq!(input.input_use_count, 2);
        assert_eq!(
            input.signal_ids,
            vec!["normalized".to_string(), "normalized-again".to_string()]
        );
    }

    #[test]
    fn rejects_conflicting_capacities_for_the_same_activation_slot() {
        let mut execution_plan = fixture_plan("row_major");
        let mut repeated = execution_plan.dispatches[0].clone();
        repeated.dispatch_index = 8;
        repeated.input_activation.byte_capacity = 16;
        execution_plan.dispatches.push(repeated);

        let error = VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("activation pedal.slot_0 has conflicting capacities 8 and 16")
        );
    }

    #[test]
    fn rejects_non_contiguous_projection_layouts() {
        let error = fixture_plan_result("column_major").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("tensor \"gate\" has non-shardable layout Some(\"column_major\")")
        );
    }

    #[test]
    fn rejects_distributed_dispatches_that_cannot_preserve_push_constants() {
        let mut prepared_plan = fixture_prepared_plan();
        prepared_plan.dispatches[0].push_constants = vec![VulkanKernelScalarBinding {
            name: "stream_tick".to_string(),
            scalar_type: "u64".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }];

        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &fixture_artifact_manifest(),
            &["owner".to_string(), "helper".to_string()],
            4,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cannot yet preserve push constants across physical shards")
        );
    }

    #[test]
    fn immutable_parameter_shards_are_reused_by_duplicated_pedals() {
        let mut execution_plan = fixture_plan("row_major");
        let mut duplicate = execution_plan.dispatches[0].clone();
        duplicate.dispatch_index = 8;
        duplicate.pedal_id = "duplicated-pedal".to_string();
        duplicate.node_id = "duplicated-ffn".to_string();
        execution_plan.dispatches.push(duplicate);

        let allocation_plan = VulkanDistributedParameterAllocationPlan::from_execution_plan(
            &execution_plan,
            &fixture_tensor_index("row_major"),
        )
        .unwrap();

        assert_eq!(allocation_plan.allocation_count, 8);
        assert_eq!(allocation_plan.tensor_count, 2);
        assert_eq!(allocation_plan.total_byte_capacity, 192);
        assert!(
            allocation_plan
                .allocations
                .iter()
                .all(|allocation| allocation.use_count == 2)
        );
    }

    #[test]
    fn loads_each_tensor_once_and_streams_verified_shards_to_devices() {
        let execution_plan = fixture_plan("row_major");
        let fixture = DistributedStorageFixture::new();
        let allocation_plan = VulkanDistributedParameterAllocationPlan::from_execution_plan(
            &execution_plan,
            &fixture.tensor_index,
        )
        .unwrap();
        let mut writes = Vec::new();

        let report = allocation_plan
            .load_from_tensor_index(&fixture.tensor_index, |allocation, bytes| {
                writes.push((allocation.clone(), bytes.to_vec()));
                Ok(())
            })
            .unwrap();

        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.source_file_count, 1);
        assert_eq!(report.allocation_count, 8);
        assert_eq!(report.write_count, 8);
        assert_eq!(report.total_bytes_read, 192);
        assert_eq!(report.total_bytes_written, 192);
        let (allocation, bytes) = writes
            .iter()
            .find(|(allocation, _)| {
                allocation.device_id == "helper-a" && allocation.tensor == "gate"
            })
            .unwrap();
        assert_eq!(allocation.byte_offset, 32);
        assert_eq!(allocation.byte_count, 32);
        assert_eq!(bytes, &fixture.gate_bytes[32..64]);
    }

    #[test]
    fn excludes_full_parameters_only_when_all_prepared_uses_are_distributed() {
        let execution_plan = fixture_plan("row_major");
        let prepared_plan = fixture_prepared_plan();

        let exclusions =
            VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
                &execution_plan,
                &[("owner", &prepared_plan)],
                &fixture_tensor_index("row_major"),
            )
            .unwrap();

        assert_eq!(exclusions.device_count, 1);
        assert_eq!(exclusions.unique_tensor_count, 2);
        assert_eq!(exclusions.excluded_full_allocation_count, 2);
        assert_eq!(exclusions.excluded_full_byte_capacity, 192);
        assert_eq!(
            exclusions.tensors_for_device("owner"),
            BTreeSet::from(["gate".to_string(), "up".to_string()])
        );
        assert!(exclusions.tensors_for_device("helper-a").is_empty());
    }

    #[test]
    fn refuses_to_exclude_a_tensor_still_used_by_a_canonical_dispatch() {
        let execution_plan = fixture_plan("row_major");
        let mut prepared_plan = fixture_prepared_plan();
        let mut canonical = prepared_plan.dispatches[0].clone();
        canonical.dispatch_index = 8;
        canonical.node_index = 4;
        canonical.node_id = "canonical-use".to_string();
        canonical.op = "linear".to_string();
        canonical.descriptors.retain(|descriptor| {
            matches!(
                descriptor.resource,
                VulkanDescriptorResourceAddress::PermanentParameter { .. }
            )
        });
        prepared_plan.dispatches.push(canonical);

        let error = VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
            &execution_plan,
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("canonical dispatch pedal.canonical-use still uses it")
        );
    }

    fn fixture_plan(layout: &str) -> VulkanDistributedExecutionPlan {
        fixture_plan_result(layout).unwrap()
    }

    fn fixture_plan_result(
        layout: &str,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        fixture_plan_result_with_alignment(layout, 4)
    }

    fn fixture_plan_result_with_alignment(
        layout: &str,
        storage_buffer_offset_alignment: usize,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        let tensor_index = fixture_tensor_index(layout);
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &[
                "owner".to_string(),
                "helper-a".to_string(),
                "helper-b".to_string(),
                "helper-c".to_string(),
            ],
            storage_buffer_offset_alignment,
        )
    }

    fn fixture_prepared_plan() -> VulkanPreparedDispatchPlan {
        let activation =
            |binding, name: &str, signal: &str, bytes| VulkanResolvedDescriptorBinding {
                binding,
                usage: if binding == 0 {
                    VulkanKernelDescriptorUsage::InputSignal
                } else {
                    VulkanKernelDescriptorUsage::OutputSignal
                },
                name: name.to_string(),
                resource: VulkanDescriptorResourceAddress::ActivationSlot {
                    pedal_id: "pedal".to_string(),
                    signal_id: signal.to_string(),
                    slot: binding,
                    byte_capacity: bytes,
                    signal_byte_capacity: bytes,
                },
            };
        let parameter = |binding, tensor: &str| VulkanResolvedDescriptorBinding {
            binding,
            usage: VulkanKernelDescriptorUsage::Parameter,
            name: tensor.to_string(),
            resource: VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: tensor.to_string(),
                tensor: tensor.to_string(),
                byte_count: Some(96),
            },
        };
        VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![VulkanPreparedDispatch {
                dispatch_index: 7,
                kernel_id: "pedal.ffn".to_string(),
                pedal_id: "pedal".to_string(),
                circuit_id: "circuit".to_string(),
                node_index: 3,
                node_id: "ffn".to_string(),
                op: DISTRIBUTABLE_PARALLEL_PROJECTION_OP.to_string(),
                reusable_family_id: "family".to_string(),
                artifact_path: "ffn.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                descriptors: vec![
                    activation(0, "input", "normalized", 8),
                    activation(1, "output", "hidden", 24),
                    parameter(2, "gate"),
                    parameter(3, "up"),
                ],
                push_constants: Vec::new(),
                uses_stream_tick: false,
            }],
            total_descriptor_count: 4,
        }
    }

    fn fixture_artifact_manifest() -> VulkanReusableKernelArtifactManifest {
        VulkanReusableKernelArtifactManifest::new(vec![VulkanReusableKernelArtifact {
            family_id: "family".to_string(),
            op: DISTRIBUTABLE_PARALLEL_PROJECTION_OP.to_string(),
            path: "ffn.spv".to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            workgroup_count_x: 6,
            descriptor_signature: Vec::new(),
            push_constants: Vec::new(),
            uses_stream_tick: false,
        }])
    }

    fn fixture_tensor_index(layout: &str) -> TensorIndex {
        let metadata = |layout: &str| TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![12, 4],
            logical_shape: None,
            parameter_count: Some(48),
            byte_count: Some(96),
            data_offsets: Some(vec![0, 96]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some(layout.to_string()),
        };
        TensorIndex {
            schema: "llmoop.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([
                ("gate".to_string(), metadata(layout)),
                ("up".to_string(), metadata(layout)),
            ]),
        }
    }

    struct DistributedStorageFixture {
        root: PathBuf,
        tensor_index: TensorIndex,
        gate_bytes: Vec<u8>,
    }

    impl DistributedStorageFixture {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "llmoop-distributed-storage-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            let source = root.join("weights.safetensors");
            let gate_bytes = (0..96).map(|value| value as u8).collect::<Vec<_>>();
            let up_bytes = (0..96)
                .map(|value| 255u8.wrapping_sub(value as u8))
                .collect::<Vec<_>>();
            let header = b"{}";
            let mut file_bytes = Vec::new();
            file_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            file_bytes.extend_from_slice(header);
            file_bytes.extend_from_slice(&gate_bytes);
            file_bytes.extend_from_slice(&up_bytes);
            fs::write(&source, file_bytes).unwrap();
            let metadata = |data_offsets: Vec<usize>, bytes: &[u8]| TensorMetadata {
                dtype: "BF16".to_string(),
                shape: vec![12, 4],
                logical_shape: None,
                parameter_count: Some(48),
                byte_count: Some(96),
                data_offsets: Some(data_offsets),
                source_file: Some(source.to_string_lossy().into_owned()),
                data_sha256: Some(
                    Sha256::digest(bytes)
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                ),
                layout: Some("row_major".to_string()),
            };
            let tensor_index = TensorIndex {
                schema: "llmoop.tensor_index.v1".to_string(),
                tensors: BTreeMap::from([
                    ("gate".to_string(), metadata(vec![0, 96], &gate_bytes)),
                    ("up".to_string(), metadata(vec![96, 192], &up_bytes)),
                ]),
            };
            Self {
                root,
                tensor_index,
                gate_bytes,
            }
        }
    }

    impl Drop for DistributedStorageFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}
