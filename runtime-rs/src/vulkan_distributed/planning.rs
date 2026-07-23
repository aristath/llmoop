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
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    if DISTRIBUTABLE_SPARSE_EXPERT_OPS.contains(&dispatch.op.as_str()) {
        return plan_sparse_expert_dispatch(
            owner_device_id,
            dispatch,
            tensor_index,
            device_ids,
            artifact_workgroup_count_x,
            storage_buffer_offset_alignment,
        );
    }
    plan_parallel_projection_dispatch(
        owner_device_id,
        dispatch,
        tensor_index,
        device_ids,
        artifact_workgroup_count_x,
        storage_buffer_offset_alignment,
    )
}

fn plan_parallel_projection_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    if !dispatch.push_constants.is_empty() {
        return Ok(None);
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
        // Physical sharding is an optional optimization. Quantized projection
        // families carry auxiliary scale/zero-point tensors in addition to
        // their matrices and need a family-specific sharding contract. Keep
        // those dispatches on their pedal's owner device until such a contract
        // is available; layer placement still distributes the model itself.
        return Ok(None);
    };
    if ![first_tensor, second_tensor].iter().all(|tensor| {
        tensor_index.tensors.get(**tensor).is_some_and(|metadata| {
            metadata.dtype == "BF16"
                && metadata.shape.len() == 2
                && matches!(
                    metadata.layout.as_deref(),
                    Some("row_major" | "vulkan_bf16_row_pair_u32")
                )
        })
    }) {
        return Ok(None);
    }
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
    if raw_shards.len() < 2 {
        return Ok(None);
    }
    let shard_device_ids = std::iter::once(owner_device_id)
        .chain(
            device_ids
                .iter()
                .map(String::as_str)
                .filter(|device_id| *device_id != owner_device_id),
        )
        .take(raw_shards.len())
        .collect::<Vec<_>>();
    let first_row_bytes = tensor_row_bytes(dispatch, first_tensor, first, output_rows)?;
    let second_row_bytes = tensor_row_bytes(dispatch, second_tensor, second, output_rows)?;
    let mut distributed_parameter_byte_count = 0usize;
    let shards = shard_device_ids
        .into_iter()
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
                device_id: device_id.to_string(),
                row_start,
                row_count,
                workgroup_count_x,
                base_workgroup_z: 0,
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

    Ok(Some(VulkanDistributedDispatchPlan {
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
        auxiliary_input_activations: Vec::new(),
        output_activation,
        distribution: VulkanDistributedDispatchDistribution::OutputRows,
        distributed_parameter_byte_count,
        shards,
    }))
}

fn plan_sparse_expert_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    if !dispatch.push_constants.is_empty() || artifact_workgroup_count_x == 0 {
        return Ok(None);
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
    if parameter_descriptors.is_empty() {
        return Ok(None);
    }

    let mut expert_count = None;
    let mut expert_alignment = 1usize;
    let mut parameters = Vec::with_capacity(parameter_descriptors.len());
    for (binding, tensor) in parameter_descriptors {
        let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
            dispatch_error(dispatch, format!("has no tensor metadata for {tensor:?}"))
        })?;
        if metadata.shape.len() < 2
            || !matches!(
                metadata.layout.as_deref(),
                Some("row_major" | "vulkan_bf16_row_pair_u32")
            )
        {
            return Ok(None);
        }
        let tensor_expert_count = metadata.shape[0];
        if tensor_expert_count == 0
            || expert_count.is_some_and(|expected| expected != tensor_expert_count)
        {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "expert tensor {tensor:?} has incompatible leading dimension {}",
                    tensor_expert_count
                ),
            ));
        }
        expert_count = Some(tensor_expert_count);
        let tensor_byte_count = metadata.byte_count.ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("expert tensor {tensor:?} has no byte count"),
            )
        })?;
        if tensor_byte_count == 0 || !tensor_byte_count.is_multiple_of(tensor_expert_count) {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "expert tensor {tensor:?} byte count {tensor_byte_count} is not divisible by {tensor_expert_count} experts"
                ),
            ));
        }
        let bytes_per_expert = tensor_byte_count / tensor_expert_count;
        let tensor_expert_alignment = storage_buffer_offset_alignment
            / greatest_common_divisor(storage_buffer_offset_alignment, bytes_per_expert);
        expert_alignment = least_common_multiple(expert_alignment, tensor_expert_alignment)
            .ok_or_else(|| dispatch_error(dispatch, "expert alignment overflowed".to_string()))?;
        parameters.push((binding, tensor, bytes_per_expert));
    }
    let expert_count = expert_count.expect("non-empty expert parameter set has a leading size");
    let raw_shards = distribute_rows(expert_count, device_ids.len(), 1, expert_alignment)
        .map_err(|error| dispatch_error(dispatch, error))?;
    if raw_shards.len() < 2 {
        return Ok(None);
    }

    let input_activation = activation_slot(dispatch, 0, 1, "primary input")?;
    let routes_activation = activation_slot(dispatch, 1, 1, "expert routes")?;
    let output_activation = activation_slot(dispatch, 2, 1, "output")?;
    let input_byte_capacity = input_activation.signal_byte_capacity;
    let output_byte_capacity = output_activation.signal_byte_capacity;
    let shard_device_ids = std::iter::once(owner_device_id)
        .chain(
            device_ids
                .iter()
                .map(String::as_str)
                .filter(|device_id| *device_id != owner_device_id),
        )
        .take(raw_shards.len())
        .collect::<Vec<_>>();
    let mut distributed_parameter_byte_count = 0usize;
    let shards = shard_device_ids
        .into_iter()
        .zip(raw_shards)
        .map(|(device_id, (expert_start, shard_expert_count))| {
            let parameters = parameters
                .iter()
                .map(|(binding, tensor, bytes_per_expert)| {
                    parameter_fragment(
                        *binding,
                        tensor,
                        *bytes_per_expert,
                        expert_start,
                        shard_expert_count,
                        dispatch,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            distributed_parameter_byte_count = parameters.iter().try_fold(
                distributed_parameter_byte_count,
                |total, fragment| {
                    total.checked_add(fragment.byte_count).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            "expert shard parameter byte count overflowed".to_string(),
                        )
                    })
                },
            )?;
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.to_string(),
                row_start: expert_start,
                row_count: shard_expert_count,
                workgroup_count_x: artifact_workgroup_count_x,
                base_workgroup_z: u32::try_from(expert_start).map_err(|_| {
                    dispatch_error(dispatch, "expert start exceeds u32".to_string())
                })?,
                output_byte_offset: 0,
                output_byte_count: output_byte_capacity,
                parameters,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;

    Ok(Some(VulkanDistributedDispatchPlan {
        owner_device_id: owner_device_id.to_string(),
        dispatch_index: dispatch.dispatch_index,
        pedal_id: dispatch.pedal_id.clone(),
        node_id: dispatch.node_id.clone(),
        reusable_family_id: dispatch.reusable_family_id.clone(),
        input_byte_capacity,
        output_byte_capacity,
        output_rows: expert_count,
        input_width: input_byte_capacity / BF16_BYTE_COUNT,
        row_alignment: expert_alignment,
        input_activation,
        auxiliary_input_activations: vec![routes_activation],
        output_activation,
        distribution: VulkanDistributedDispatchDistribution::ExpertRange,
        distributed_parameter_byte_count,
        shards,
    }))
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
                binding,
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

