impl VulkanResidentPlacedPedalBatchRunner {
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        lane_capacity: usize,
        execution_mode: VulkanPedalBatchExecutionMode,
        distributed_execution_plan: &VulkanDistributedExecutionPlan,
        distributed_parameter_buffers: &VulkanDistributedParameterBuffers,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let slices = placed_slices
            .iter()
            .map(|slice| {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                VulkanResidentPedalBatchSliceRunner::new(
                    devices,
                    device,
                    slice,
                    lane_capacity,
                    execution_mode,
                    distributed_execution_plan,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let distributed_dispatches = VulkanDistributedPedalBatchRunners::new(
            devices,
            placed_slices,
            &slices,
            distributed_execution_plan,
            distributed_parameter_buffers,
            lane_capacity,
            execution_mode,
        )?;
        let mut cable_transfers = Vec::new();
        for (source_device_index, placed_slice) in placed_slices.iter().enumerate() {
            for outgoing in &placed_slice.mounted.cable_io.outgoing_buffers {
                let destination_device_index = placed_slices
                    .iter()
                    .position(|candidate| {
                        candidate.device_id == outgoing.endpoint.remote_device_id
                            && candidate
                                .mounted
                                .cable_io
                                .incoming_buffer(outgoing.endpoint.cable_index)
                                .is_some()
                    })
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "pedal batch cable {} has no destination device {:?}",
                                outgoing.endpoint.cable_index, outgoing.endpoint.remote_device_id
                            ),
                        ))
                    })?;
                let source = slices[source_device_index].signal_buffer(
                    &VulkanPedalBatchSignalKey::OutgoingCable(outgoing.endpoint.cable_index),
                )?;
                let destination = slices[destination_device_index].signal_buffer(
                    &VulkanPedalBatchSignalKey::IncomingCable(outgoing.endpoint.cable_index),
                )?;
                if source.frame_byte_capacity != destination.frame_byte_capacity
                    || source.buffer.byte_capacity() != destination.buffer.byte_capacity()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "pedal batch cable {} source and destination capacities differ",
                            outgoing.endpoint.cable_index
                        )),
                    ));
                }
                let source_device = devices.get(&placed_slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: placed_slice.device_id.clone(),
                    }
                })?;
                let destination_device = devices
                    .get(&placed_slices[destination_device_index].device_id)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: placed_slices[destination_device_index].device_id.clone(),
                        }
                    })?;
                let byte_len = source.buffer.byte_capacity();
                let binding = if Rc::ptr_eq(source_device, destination_device) {
                    VulkanPedalBatchCableTransferBinding::Resident(Box::new(
                        source_device
                            .create_resident_buffer_copy(
                                &source.buffer,
                                &destination.buffer,
                                byte_len,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    ))
                } else {
                    VulkanPedalBatchCableTransferBinding::Mapped(
                        source
                            .buffer
                            .create_persistently_mapped_copy_to(&destination.buffer, byte_len)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    )
                };
                cable_transfers.push(VulkanPedalBatchCableTransfer {
                    source_device_index,
                    destination_device_index,
                    cable_index: outgoing.endpoint.cable_index,
                    binding,
                });
            }
        }
        Ok(Self {
            distributed_dispatches,
            lane_capacity,
            slices,
            cable_transfers,
        })
    }

    fn slice(
        &self,
        index: usize,
    ) -> Result<&VulkanResidentPedalBatchSliceRunner, VulkanResidentInProcessPlacedRuntimeError>
    {
        self.slices.get(index).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "placed pedal batch has no device slice {index}"
            )))
        })
    }

    fn transfer_cable(
        &self,
        source_device_index: usize,
        destination_device_index: usize,
        cable_index: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.cable_transfers
            .iter()
            .find(|transfer| {
                transfer.source_device_index == source_device_index
                    && transfer.destination_device_index == destination_device_index
                    && transfer.cable_index == cable_index
            })
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "pedal batch has no cable transfer {source_device_index}->{destination_device_index}:{cable_index}"
                )))
            })?
            .run()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_causal_sequence(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device_index: usize,
        owner_device_id: &str,
        mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let device = devices.get(owner_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: owner_device_id.to_string(),
            }
        })?;
        self.slice(device_index)?.run_causal_sequence(
            device,
            mounted,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            |dispatch_index, batch_control| {
                self.distributed_dispatches.run_dispatch(
                    devices,
                    owner_device_id,
                    dispatch_index,
                    batch_control,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_independent_candidates(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device_index: usize,
        owner_device_id: &str,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transaction: &VulkanResidentStateTransactionBank,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let device = devices.get(owner_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: owner_device_id.to_string(),
            }
        })?;
        self.slice(device_index)?.run_independent_candidates(
            device,
            mounted,
            transaction,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            |dispatch_index, batch_control| {
                self.distributed_dispatches.run_dispatch(
                    devices,
                    owner_device_id,
                    dispatch_index,
                    batch_control,
                )
            },
        )
    }
}

fn pedal_batch_signal_target(
    descriptor: &VulkanMountedPlacedBoundDescriptor,
) -> Result<Option<(VulkanPedalBatchSignalKey, usize)>, VulkanResidentInProcessPlacedRuntimeError> {
    let target = match &descriptor.target {
        VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target:
                VulkanBoundDescriptorTarget::ActivationSlot {
                    pedal_id,
                    signal_id,
                    signal_byte_capacity,
                    ..
                },
        } => Some((
            VulkanPedalBatchSignalKey::Activation {
                pedal_id: pedal_id.clone(),
                signal_id: signal_id.clone(),
            },
            *signal_byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::Resident { .. } => None,
        VulkanMountedPlacedBoundDescriptorTarget::ModelInput { .. }
        | VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { .. } => None,
        VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer { cable }
        | VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer { cable } => Some((
            VulkanPedalBatchSignalKey::LocalCable(cable.cable.cable_index),
            cable.byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer { endpoint } => Some((
            VulkanPedalBatchSignalKey::IncomingCable(endpoint.endpoint.cable_index),
            endpoint.byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer { endpoint } => Some((
            VulkanPedalBatchSignalKey::OutgoingCable(endpoint.endpoint.cable_index),
            endpoint.byte_capacity,
        )),
    };
    Ok(target)
}

fn pedal_batch_bindings<'a>(
    mounted: &'a VulkanMountedPlacedStreamCircuit,
    dispatch: &VulkanMountedPlacedBoundDispatch,
    signal_buffers: &'a [VulkanPedalBatchSignalBuffer],
    signal_buffer_indices: &BTreeMap<VulkanPedalBatchSignalKey, usize>,
    lane_index: Option<usize>,
    stream_control_buffer: Option<&'a VulkanResidentBuffer>,
) -> Result<Vec<VulkanResidentKernelBufferBinding<'a>>, VulkanResidentInProcessPlacedRuntimeError> {
    let mut bindings = Vec::with_capacity(
        dispatch.descriptors.len() + usize::from(stream_control_buffer.is_some()),
    );
    for descriptor in &dispatch.descriptors {
        let binding = u32::try_from(descriptor.binding).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "pedal batch descriptor binding exceeds u32".to_string(),
            ))
        })?;
        let access = match descriptor.usage {
            VulkanKernelDescriptorUsage::InputSignal
            | VulkanKernelDescriptorUsage::Parameter
            | VulkanKernelDescriptorUsage::StateRead => VulkanResidentKernelBufferAccess::Read,
            VulkanKernelDescriptorUsage::OutputSignal | VulkanKernelDescriptorUsage::StateWrite => {
                VulkanResidentKernelBufferAccess::Write
            }
            VulkanKernelDescriptorUsage::StateView => VulkanResidentKernelBufferAccess::ReadWrite,
        };
        if let Some((key, frame_byte_capacity)) =
            pedal_batch_signal_target_with_mounted(mounted, descriptor)?
        {
            let index = signal_buffer_indices.get(&key).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "pedal batch descriptor {}.{} has no signal buffer {key:?}",
                    dispatch.pedal_id, dispatch.node_id
                )))
            })?;
            let allocation = &signal_buffers[*index];
            let (byte_offset, byte_len) = if let Some(lane_index) = lane_index {
                (
                    lane_index.checked_mul(frame_byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "pedal batch lane offset overflowed".to_string(),
                        ))
                    })?,
                    frame_byte_capacity,
                )
            } else {
                (0, allocation.buffer.byte_capacity())
            };
            bindings.push(
                VulkanResidentKernelBufferBinding::new(binding, &allocation.buffer, byte_len)
                    .with_byte_offset(byte_offset)
                    .with_access(access),
            );
            continue;
        }
        let (buffer, byte_len) = match &descriptor.target {
            VulkanMountedPlacedBoundDescriptorTarget::Resident { target } => match target {
                VulkanBoundDescriptorTarget::PermanentParameter { tensor, .. } => {
                    let parameter = mounted
                        .parameter_buffers
                        .parameter_buffer(tensor)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!("pedal batch is missing parameter {tensor:?}"),
                            ))
                        })?;
                    (&parameter.buffer, parameter.byte_capacity)
                }
                VulkanBoundDescriptorTarget::StreamStateBuffer {
                    buffer_index,
                    byte_capacity,
                    ..
                }
                | VulkanBoundDescriptorTarget::StreamStateView {
                    buffer_index,
                    byte_capacity,
                    ..
                } => {
                    let state = mounted
                        .buffers
                        .state_buffers
                        .get(*buffer_index)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!("pedal batch has no state buffer {buffer_index}"),
                            ))
                        })?;
                    (&state.buffer, *byte_capacity)
                }
                VulkanBoundDescriptorTarget::BoundaryInput { .. }
                | VulkanBoundDescriptorTarget::BoundaryOutput { .. }
                | VulkanBoundDescriptorTarget::ActivationSlot { .. } => {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "pedal batch descriptor {}.{} has an unbound resident signal target",
                            dispatch.pedal_id, dispatch.node_id
                        )),
                    ));
                }
            },
            _ => unreachable!("signal targets were handled above"),
        };
        bindings.push(
            VulkanResidentKernelBufferBinding::new(binding, buffer, byte_len).with_access(access),
        );
    }
    if let Some(stream_control_buffer) = stream_control_buffer {
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(dispatch.descriptors.len()).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "pedal batch stream-control binding exceeds u32".to_string(),
                    ))
                })?,
                stream_control_buffer,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    Ok(bindings)
}

fn pedal_batch_signal_target_with_mounted(
    mounted: &VulkanMountedPlacedStreamCircuit,
    descriptor: &VulkanMountedPlacedBoundDescriptor,
) -> Result<Option<(VulkanPedalBatchSignalKey, usize)>, VulkanResidentInProcessPlacedRuntimeError> {
    match &descriptor.target {
        VulkanMountedPlacedBoundDescriptorTarget::ModelInput { signal_id } => {
            let allocation = mounted.boundary_io.input_buffer(signal_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "pedal batch has no model input {signal_id:?}"
                )))
            })?;
            Ok(Some((
                VulkanPedalBatchSignalKey::ModelInput(signal_id.clone()),
                allocation.byte_capacity,
            )))
        }
        VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
            let allocation = mounted
                .boundary_io
                .output_buffer(signal_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "pedal batch has no model output {signal_id:?}"
                    )))
                })?;
            Ok(Some((
                VulkanPedalBatchSignalKey::ModelOutput(signal_id.clone()),
                allocation.byte_capacity,
            )))
        }
        _ => pedal_batch_signal_target(descriptor),
    }
}

