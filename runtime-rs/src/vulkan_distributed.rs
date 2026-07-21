use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::stream_plan::{TensorIndex, TensorMetadata};
use crate::vulkan_stream_circuit::{
    VulkanBoundDescriptorTarget, VulkanMountedPlacedBoundDescriptorTarget,
    VulkanMountedPlacedBoundDispatch, VulkanMountedPlacedBoundDispatchPlan,
    VulkanReusableKernelArtifactManifest,
};

const DISTRIBUTABLE_PARALLEL_PROJECTION_OP: &str = "parallel_linear_silu_multiply";
const BF16_BYTE_COUNT: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedExecutionPlan {
    pub device_ids: Vec<String>,
    pub dispatches: Vec<VulkanDistributedDispatchPlan>,
    pub shared_input_byte_capacity: usize,
    pub shared_output_byte_capacity: usize,
    pub distributed_parameter_byte_count: usize,
}

impl VulkanDistributedExecutionPlan {
    pub fn from_bound_plans(
        bound_plans: &[&VulkanMountedPlacedBoundDispatchPlan],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanReusableKernelArtifactManifest,
        device_ids: &[String],
    ) -> Result<Self, VulkanDistributedPlanError> {
        validate_device_pool(device_ids)?;
        let mut dispatches = Vec::new();
        let mut shared_input_byte_capacity = 0usize;
        let mut shared_output_byte_capacity = 0usize;
        let mut distributed_parameter_byte_count = 0usize;

        for bound_plan in bound_plans {
            if device_ids.len() < 2 {
                continue;
            }
            if !device_ids.contains(&bound_plan.device_id) {
                return Err(VulkanDistributedPlanError(format!(
                    "dispatch owner {:?} is absent from the distributed execution device pool",
                    bound_plan.device_id
                )));
            }
            for dispatch in &bound_plan.dispatches {
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
                    &bound_plan.device_id,
                    dispatch,
                    tensor_index,
                    device_ids,
                    artifact.workgroup_count_x,
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
    pub distributed_parameter_byte_count: usize,
    pub shards: Vec<VulkanDistributedDispatchShard>,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocation {
    pub device_id: String,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
    pub use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedParameterAllocationKey {
    device_id: String,
    tensor: String,
    byte_offset: usize,
    byte_count: usize,
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
    dispatch: &VulkanMountedPlacedBoundDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    artifact_workgroup_count_x: u32,
) -> Result<VulkanDistributedDispatchPlan, VulkanDistributedPlanError> {
    let parameter_descriptors = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.target {
            VulkanMountedPlacedBoundDescriptorTarget::Resident {
                target: VulkanBoundDescriptorTarget::PermanentParameter { tensor, .. },
            } => Some((descriptor.binding, tensor.as_str())),
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
    let mut row_alignment = output_rows / artifact_workgroup_count;
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
    validate_signal_capacity(dispatch, 0, input_byte_capacity, "input")?;
    validate_signal_capacity(dispatch, 1, output_byte_capacity, "output")?;

    let raw_shards = distribute_rows(output_rows, device_ids.len(), row_alignment)
        .map_err(|error| dispatch_error(dispatch, error))?;
    let first_row_bytes = tensor_row_bytes(dispatch, first_tensor, first, output_rows)?;
    let second_row_bytes = tensor_row_bytes(dispatch, second_tensor, second, output_rows)?;
    let mut distributed_parameter_byte_count = 0usize;
    let shards = device_ids
        .iter()
        .zip(raw_shards)
        .map(|(device_id, (row_start, row_count))| {
            let workgroup_count_x = u32::try_from(row_count / row_alignment).map_err(|_| {
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
        distributed_parameter_byte_count,
        shards,
    })
}

fn projection_metadata<'a>(
    tensor_index: &'a TensorIndex,
    dispatch: &VulkanMountedPlacedBoundDispatch,
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
    dispatch: &VulkanMountedPlacedBoundDispatch,
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

fn validate_signal_capacity(
    dispatch: &VulkanMountedPlacedBoundDispatch,
    binding: usize,
    required: usize,
    role: &str,
) -> Result<(), VulkanDistributedPlanError> {
    let capacity = dispatch
        .descriptors
        .iter()
        .find(|descriptor| descriptor.binding == binding)
        .and_then(|descriptor| match &descriptor.target {
            VulkanMountedPlacedBoundDescriptorTarget::Resident {
                target:
                    VulkanBoundDescriptorTarget::ActivationSlot {
                        signal_byte_capacity,
                        ..
                    },
            } => Some(*signal_byte_capacity),
            _ => None,
        })
        .ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("has no resident {role} activation at binding {binding}"),
            )
        })?;
    if capacity < required {
        return Err(dispatch_error(
            dispatch,
            format!("{role} signal has {capacity} bytes but requires {required}"),
        ));
    }
    Ok(())
}

fn distribute_rows(
    row_count: usize,
    requested_shards: usize,
    row_alignment: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if row_count == 0 || requested_shards == 0 || row_alignment == 0 {
        return Err("row distribution dimensions must not be zero".to_string());
    }
    if !row_count.is_multiple_of(row_alignment) {
        return Err(format!(
            "row count {row_count} is not aligned to {row_alignment}"
        ));
    }
    let row_groups = row_count / row_alignment;
    let shard_count = requested_shards.min(row_groups);
    let groups_per_shard = row_groups / shard_count;
    let remainder = row_groups % shard_count;
    let mut row_start = 0usize;
    let mut shards = Vec::with_capacity(shard_count);
    for shard_index in 0..shard_count {
        let group_count = groups_per_shard + usize::from(shard_index < remainder);
        let shard_rows = group_count
            .checked_mul(row_alignment)
            .ok_or_else(|| "row shard size overflowed".to_string())?;
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
    dispatch: &VulkanMountedPlacedBoundDispatch,
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
    dispatch: &VulkanMountedPlacedBoundDispatch,
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

    use super::*;
    use crate::stream_plan::TensorMetadata;
    use crate::vulkan_stream_circuit::{
        VulkanBoundDescriptorTarget, VulkanKernelDescriptorUsage,
        VulkanMountedPlacedBoundDescriptor, VulkanMountedPlacedBoundDescriptorTarget,
        VulkanReusableKernelArtifact,
    };

    #[test]
    fn plans_balanced_parameter_and_output_shards_from_compiled_contracts() {
        let plan = fixture_plan("row_major");

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.shared_input_byte_capacity, 8);
        assert_eq!(plan.shared_output_byte_capacity, 24);
        assert_eq!(plan.distributed_parameter_byte_count, 192);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.owner_device_id, "owner");
        assert_eq!(dispatch.row_alignment, 2);
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
    fn rejects_non_contiguous_projection_layouts() {
        let error = fixture_plan_result("column_major").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("tensor \"gate\" has non-shardable layout Some(\"column_major\")")
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

    fn fixture_plan(layout: &str) -> VulkanDistributedExecutionPlan {
        fixture_plan_result(layout).unwrap()
    }

    fn fixture_plan_result(
        layout: &str,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        let tensor_index = fixture_tensor_index(layout);
        let activation =
            |binding, name: &str, signal: &str, bytes| VulkanMountedPlacedBoundDescriptor {
                binding,
                usage: if binding == 0 {
                    VulkanKernelDescriptorUsage::InputSignal
                } else {
                    VulkanKernelDescriptorUsage::OutputSignal
                },
                name: name.to_string(),
                target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
                    target: VulkanBoundDescriptorTarget::ActivationSlot {
                        buffer_index: binding,
                        pedal_id: "pedal".to_string(),
                        signal_id: signal.to_string(),
                        circuit_id: "circuit".to_string(),
                        slot: binding,
                        byte_capacity: bytes,
                        signal_byte_capacity: bytes,
                    },
                },
            };
        let parameter = |binding, tensor: &str| VulkanMountedPlacedBoundDescriptor {
            binding,
            usage: VulkanKernelDescriptorUsage::Parameter,
            name: tensor.to_string(),
            target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
                target: VulkanBoundDescriptorTarget::PermanentParameter {
                    param_id: tensor.to_string(),
                    tensor: tensor.to_string(),
                    byte_count: Some(96),
                },
            },
        };
        let bound_plan = VulkanMountedPlacedBoundDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            device_id: "owner".to_string(),
            dispatches: vec![VulkanMountedPlacedBoundDispatch {
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
            resident_descriptor_count: 4,
            model_boundary_descriptor_count: 0,
            local_cable_descriptor_count: 0,
            cable_endpoint_descriptor_count: 0,
            incoming_cable_descriptor_count: 0,
            outgoing_cable_descriptor_count: 0,
        };
        let artifact_manifest =
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
            }]);
        VulkanDistributedExecutionPlan::from_bound_plans(
            &[&bound_plan],
            &tensor_index,
            &artifact_manifest,
            &[
                "owner".to_string(),
                "helper-a".to_string(),
                "helper-b".to_string(),
                "helper-c".to_string(),
            ],
        )
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
}
