struct VulkanResidentPlacedPedalBatchRunner {
    distributed_dispatches: VulkanDistributedPedalBatchRunners,
    lane_capacity: usize,
    slices: Vec<VulkanResidentPedalBatchSliceRunner>,
    cable_transfers: Vec<VulkanPedalBatchCableTransfer>,
}

struct VulkanDistributedPedalBatchRunners {
    dispatches: Vec<VulkanDistributedPedalBatchDispatchRunner>,
}

struct VulkanDistributedPedalBatchDispatchRunner {
    planned: VulkanDistributedDispatchGroup,
    shards: Vec<VulkanDistributedPedalBatchShardRunner>,
}

struct VulkanDistributedPedalBatchShardRunner {
    device_id: String,
    dispatches: Vec<VulkanResidentKernelDispatch>,
    sequence: VulkanResidentKernelSequence,
}

impl VulkanDistributedPedalBatchRunners {
    #[allow(clippy::too_many_arguments)]
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        batch_slices: &[VulkanResidentPedalBatchSliceRunner],
        execution_plan: &VulkanDistributedExecutionPlan,
        parameter_buffers: &VulkanDistributedParameterBuffers,
        lane_capacity: usize,
        execution_mode: VulkanPedalBatchExecutionMode,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let mut dispatches = Vec::with_capacity(execution_plan.dispatches.len());
        for planned in &execution_plan.dispatches {
            for shard in &planned.shards {
                if !devices.contains_key(&shard.device_id) {
                    return Err(
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: shard.device_id.clone(),
                        },
                    );
                }
            }
            let owner_index = placed_slices
                .iter()
                .position(|slice| slice.device_id == planned.owner_device_id)
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: planned.owner_device_id.clone(),
                    },
                )?;
            let package_slice = &placed_slices[owner_index].package_slice;
            let batch_slice = &batch_slices[owner_index];
            let artifact = select_pedal_batch_kernel_artifact_where(
                &package_slice.batch_kernels,
                &planned.pedal_id,
                &planned.node_id,
                execution_mode,
                lane_capacity,
                |artifact| {
                    planned.shards.iter().all(|shard| {
                        devices.get(&shard.device_id).is_some_and(|device| {
                            batch_kernel_artifact_is_supported(device, artifact)
                        })
                    })
                },
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed pedal batch {}.{} has no compatible batch artifact",
                    planned.pedal_id, planned.node_id
                )))
            })?;
            if artifact.batch_mode != VulkanResidentPedalKernelBatchMode::WeightShared {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "distributed pedal batch {}.{} requires a weight-shared artifact",
                        planned.pedal_id, planned.node_id
                    )),
                ));
            }
            let input_key = VulkanPedalBatchSignalKey::Activation {
                pedal_id: planned.input_activation.pedal_id.clone(),
                signal_id: planned.input_activation.signal_id.clone(),
            };
            let auxiliary_input_keys = planned
                .auxiliary_input_activations
                .iter()
                .map(|activation| VulkanPedalBatchSignalKey::Activation {
                    pedal_id: activation.pedal_id.clone(),
                    signal_id: activation.signal_id.clone(),
                })
                .collect::<Vec<_>>();
            let output_key = VulkanPedalBatchSignalKey::Activation {
                pedal_id: planned.output_activation.pedal_id.clone(),
                signal_id: planned.output_activation.signal_id.clone(),
            };
            let input_frame_capacity = batch_slice.signal_buffer(&input_key)?.frame_byte_capacity;
            let output_frame_capacity = batch_slice.signal_buffer(&output_key)?.frame_byte_capacity;
            if input_frame_capacity != planned.input_byte_capacity
                || output_frame_capacity != planned.output_byte_capacity
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "distributed pedal batch {}.{} signal capacities differ from its physical plan",
                        planned.pedal_id, planned.node_id
                    )),
                ));
            }
            for (activation, key) in planned
                .auxiliary_input_activations
                .iter()
                .zip(&auxiliary_input_keys)
            {
                if batch_slice.signal_buffer(key)?.frame_byte_capacity
                    != activation.signal_byte_capacity
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "distributed pedal batch {}.{} auxiliary signal {} differs from its physical plan",
                            planned.pedal_id, planned.node_id, activation.signal_id
                        )),
                    ));
                }
            }
            let input_byte_capacity = planned
                .input_byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "distributed pedal batch input capacity overflowed".to_string(),
                    ))
                })?;
            let workgroup_count_y = u32::try_from(
                lane_capacity
                    .checked_add(artifact.lane_tile_width - 1)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed pedal batch lane count overflowed".to_string(),
                        ))
                    })?
                    / artifact.lane_tile_width,
            )
            .map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed pedal batch workgroup count exceeds u32".to_string(),
                ))
            })?;
            let mut shards = Vec::with_capacity(planned.shards.len());
            for shard in &planned.shards {
                let device = devices.get(&shard.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: shard.device_id.clone(),
                    }
                })?;
                let input = batch_slice.distributed_signal_buffer(&input_key, &shard.device_id)?;
                let output =
                    batch_slice.distributed_signal_buffer(&output_key, &shard.device_id)?;
                let (output_byte_offset, output_byte_capacity) = match planned.distribution {
                    VulkanDistributedDispatchDistribution::OutputRows => {
                        distributed_batch_shard_output_binding_range(
                            planned.output_byte_capacity,
                            lane_capacity,
                            shard.output_byte_offset,
                            shard.output_byte_count,
                        )?
                    }
                    VulkanDistributedDispatchDistribution::ExpertRange => (
                        0,
                        planned
                            .output_byte_capacity
                            .checked_mul(lane_capacity)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed expert output capacity overflowed".to_string(),
                                ))
                            })?,
                    ),
                };
                let mut bindings = Vec::with_capacity(
                    2 + planned.auxiliary_input_activations.len() + shard.parameters.len(),
                );
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        u32::try_from(planned.input_activation.binding).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed pedal batch primary input binding exceeds u32"
                                    .to_string(),
                            ))
                        })?,
                        input,
                        input_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                );
                for (activation, key) in planned
                    .auxiliary_input_activations
                    .iter()
                    .zip(&auxiliary_input_keys)
                {
                    let buffer = batch_slice.distributed_signal_buffer(key, &shard.device_id)?;
                    let byte_capacity = activation
                        .signal_byte_capacity
                        .checked_mul(lane_capacity)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed pedal batch auxiliary input capacity overflowed"
                                    .to_string(),
                            ))
                        })?;
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            u32::try_from(activation.binding).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed pedal batch auxiliary binding exceeds u32"
                                        .to_string(),
                                ))
                            })?,
                            buffer,
                            byte_capacity,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        u32::try_from(planned.output_activation.binding).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed pedal batch output binding exceeds u32".to_string(),
                            ))
                        })?,
                        output,
                        output_byte_capacity,
                    )
                    .with_byte_offset(output_byte_offset)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                );
                for fragment in &shard.parameters {
                    let allocation = parameter_buffers
                        .parameter_buffer(
                            &shard.device_id,
                            &fragment.tensor,
                            fragment.byte_offset,
                            fragment.byte_count,
                        )
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!(
                                    "distributed pedal batch {}.{} is missing tensor {:?} range {}..{} on {:?}",
                                    planned.pedal_id,
                                    planned.node_id,
                                    fragment.tensor,
                                    fragment.byte_offset,
                                    fragment.byte_offset + fragment.byte_count,
                                    shard.device_id
                                ),
                            ))
                        })?;
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            u32::try_from(fragment.binding).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed pedal batch binding exceeds u32".to_string(),
                                ))
                            })?,
                            &allocation.buffer,
                            fragment.byte_count,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
                let mut resident_dispatches = Vec::with_capacity(artifact.stages.len());
                for stage in &artifact.stages {
                    let workgroup_count_x = match planned.distribution {
                        VulkanDistributedDispatchDistribution::ExpertRange => {
                            stage.workgroup_count_x
                        }
                        VulkanDistributedDispatchDistribution::OutputRows => {
                            let rows_per_workgroup = distributed_batch_rows_per_workgroup(
                                planned.output_rows,
                                stage.workgroup_count_x,
                                &planned.pedal_id,
                                &planned.node_id,
                            )?;
                            if !shard.row_start.is_multiple_of(rows_per_workgroup)
                                || !shard.row_count.is_multiple_of(rows_per_workgroup)
                            {
                                return Err(
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                        VulkanError(format!(
                                            "distributed pedal batch {}.{} shard rows {}..{} do not align to {rows_per_workgroup} rows per workgroup",
                                            planned.pedal_id,
                                            planned.node_id,
                                            shard.row_start,
                                            shard.row_start + shard.row_count
                                        )),
                                    ),
                                );
                            }
                            u32::try_from(shard.row_count / rows_per_workgroup).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed pedal batch shard workgroup count exceeds u32"
                                        .to_string(),
                                ))
                            })?
                        }
                    };
                    resident_dispatches.push(
                        device
                            .create_resident_kernel_dispatch_2d_with_base_z(
                                &stage.spirv_words,
                                &bindings,
                                workgroup_count_x,
                                workgroup_count_y,
                                shard.base_workgroup_z,
                                stage.local_size_x,
                                VULKAN_PEDAL_BATCH_CONTROL_BYTE_CAPACITY,
                                Some(format!(
                                    "pedal={} node={} distributed_batch=device:{} rows={}..{} base_z={} distribution={:?}",
                                    planned.pedal_id,
                                    planned.node_id,
                                    shard.device_id,
                                    shard.row_start,
                                    shard.row_start + shard.row_count,
                                    shard.base_workgroup_z,
                                    planned.distribution,
                                )),
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                }
                let sequence = device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                shards.push(VulkanDistributedPedalBatchShardRunner {
                    device_id: shard.device_id.clone(),
                    dispatches: resident_dispatches,
                    sequence,
                });
            }
            dispatches.push(VulkanDistributedPedalBatchDispatchRunner {
                planned: VulkanDistributedDispatchGroup {
                    owner_device_id: planned.owner_device_id.clone(),
                    dispatches: vec![planned.clone()],
                },
                shards,
            });
        }
        let mut dispatches_by_key = dispatches
            .into_iter()
            .map(|runner| {
                let leader = runner.planned.leader();
                (
                    (leader.owner_device_id.clone(), leader.dispatch_index),
                    runner,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut grouped_dispatches = Vec::with_capacity(execution_plan.dispatch_groups.len());
        for planned_group in &execution_plan.dispatch_groups {
            let mut members = planned_group
                .dispatches
                .iter()
                .map(|planned| {
                    dispatches_by_key
                        .remove(&(planned.owner_device_id.clone(), planned.dispatch_index))
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!(
                                    "distributed pedal batch has no physical dispatch {}.{}",
                                    planned.pedal_id, planned.node_id
                                ),
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut leader_runner = members.remove(0);
            for member in members {
                if member.shards.len() != leader_runner.shards.len() {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "distributed pedal batch group {}..{} changes shard count",
                            planned_group.leader().dispatch_index,
                            planned_group.tail().dispatch_index
                        )),
                    ));
                }
                for (leader_shard, member_shard) in
                    leader_runner.shards.iter_mut().zip(member.shards)
                {
                    if leader_shard.device_id != member_shard.device_id {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "distributed pedal batch group {}..{} changes shard device from {:?} to {:?}",
                                planned_group.leader().dispatch_index,
                                planned_group.tail().dispatch_index,
                                leader_shard.device_id,
                                member_shard.device_id
                            )),
                        ));
                    }
                    leader_shard.dispatches.extend(member_shard.dispatches);
                }
            }
            leader_runner.planned = planned_group.clone();
            grouped_dispatches.push(leader_runner);
        }
        if !dispatches_by_key.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "distributed pedal batch left ungrouped physical dispatches".to_string(),
                ),
            ));
        }
        Ok(Self {
            dispatches: grouped_dispatches,
        })
    }

    fn run_dispatch(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        owner_device_id: &str,
        dispatch_index: usize,
        batch_control: &[u8],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let dispatch = self
            .dispatches
            .iter()
            .find(|dispatch| {
                dispatch.planned.owner_device_id == owner_device_id
                    && dispatch.planned.leader().dispatch_index == dispatch_index
            })
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed pedal batch has no dispatch {dispatch_index} owned by {owner_device_id:?}"
                )))
            })?;
        for shard in &dispatch.shards {
            let device = devices.get(&shard.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: shard.device_id.clone(),
                }
            })?;
            let steps = shard
                .dispatches
                .iter()
                .map(|resident| VulkanResidentKernelSequenceStep::new(resident, batch_control))
                .collect::<Vec<_>>();
            device
                .record_resident_kernel_sequence(&shard.sequence, &steps)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        let mut submitted = Vec::<(
            &VulkanComputeDevice,
            &VulkanDistributedPedalBatchShardRunner,
        )>::with_capacity(dispatch.shards.len());
        for shard in &dispatch.shards {
            let device = devices.get(&shard.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: shard.device_id.clone(),
                }
            })?;
            if let Err(error) = device.submit_recorded_resident_kernel_sequence(&shard.sequence) {
                for (submitted_device, submitted_shard) in &submitted {
                    let _ =
                        submitted_device.wait_resident_kernel_sequence(&submitted_shard.sequence);
                }
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    error,
                ));
            }
            submitted.push((device.as_ref(), shard));
        }
        let mut first_error = None;
        for (device, shard) in submitted {
            if let Err(error) = device.wait_resident_kernel_sequence(&shard.sequence)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        if let Some(error) = first_error {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                error,
            ));
        }
        Ok(())
    }
}

fn distributed_batch_shard_output_binding_range(
    frame_byte_capacity: usize,
    lane_capacity: usize,
    shard_byte_offset: usize,
    shard_byte_count: usize,
) -> Result<(usize, usize), VulkanResidentInProcessPlacedRuntimeError> {
    if lane_capacity == 0 || shard_byte_count == 0 {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError("distributed pedal batch output range is empty".to_string()),
        ));
    }
    let shard_end = shard_byte_offset
        .checked_add(shard_byte_count)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed pedal batch shard output end overflowed".to_string(),
            ))
        })?;
    if shard_end > frame_byte_capacity {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "distributed pedal batch shard output range {shard_byte_offset}..{shard_end} exceeds frame capacity {frame_byte_capacity}"
            )),
        ));
    }
    let preceding_lanes = frame_byte_capacity
        .checked_mul(lane_capacity - 1)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed pedal batch output lane span overflowed".to_string(),
            ))
        })?;
    let binding_byte_capacity = preceding_lanes
        .checked_add(shard_byte_count)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed pedal batch output binding span overflowed".to_string(),
            ))
        })?;
    Ok((shard_byte_offset, binding_byte_capacity))
}

fn distributed_batch_rows_per_workgroup(
    output_rows: usize,
    full_workgroup_count_x: u32,
    pedal_id: &str,
    node_id: &str,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    let full_workgroup_count_x = usize::try_from(full_workgroup_count_x).map_err(|_| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
            "distributed pedal batch workgroup count exceeds usize".to_string(),
        ))
    })?;
    if full_workgroup_count_x == 0 || !output_rows.is_multiple_of(full_workgroup_count_x) {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "distributed pedal batch {pedal_id}.{node_id} cannot partition {output_rows} rows across {full_workgroup_count_x} workgroups"
            )),
        ));
    }
    Ok(output_rows / full_workgroup_count_x)
}

