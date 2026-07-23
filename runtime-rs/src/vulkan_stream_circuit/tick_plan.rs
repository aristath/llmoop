#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickPlan {
    pub backend_id: String,
    pub device_id: String,
    pub stages: Vec<VulkanMountedPlacedStreamTickStage>,
    pub stage_count: usize,
    pub receive_stage_count: usize,
    pub dispatch_stage_count: usize,
    pub publish_stage_count: usize,
    pub local_cable_read_count: usize,
    pub local_cable_write_count: usize,
    pub incoming_cable_read_count: usize,
    pub outgoing_cable_write_count: usize,
    pub model_input_read_count: usize,
    pub model_output_write_count: usize,
    pub can_execute: bool,
}

impl VulkanMountedPlacedStreamTickPlan {
    pub fn from_mounted_bound_plan(
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
    ) -> Self {
        let mut stages = Vec::new();
        let mut received_endpoints = BTreeSet::<(usize, usize)>::new();
        let mut published_endpoints = BTreeSet::<(usize, usize)>::new();

        let mut local_cable_read_count = 0usize;
        let mut local_cable_write_count = 0usize;
        let mut incoming_cable_read_count = 0usize;
        let mut outgoing_cable_write_count = 0usize;
        let mut model_input_read_count = 0usize;
        let mut model_output_write_count = 0usize;

        for dispatch in &mounted_bound_plan.dispatches {
            let dispatch_stage =
                VulkanMountedPlacedStreamTickDispatch::from_bound_dispatch(dispatch);
            local_cable_read_count += dispatch_stage
                .reads
                .iter()
                .filter(|io| matches!(io, VulkanMountedPlacedStreamTickIo::LocalCableBuffer { .. }))
                .count();
            local_cable_write_count += dispatch_stage
                .writes
                .iter()
                .filter(|io| matches!(io, VulkanMountedPlacedStreamTickIo::LocalCableBuffer { .. }))
                .count();
            incoming_cable_read_count += dispatch_stage
                .reads
                .iter()
                .filter(|io| {
                    matches!(
                        io,
                        VulkanMountedPlacedStreamTickIo::IncomingCableBuffer { .. }
                    )
                })
                .count();
            outgoing_cable_write_count += dispatch_stage
                .writes
                .iter()
                .filter(|io| {
                    matches!(
                        io,
                        VulkanMountedPlacedStreamTickIo::OutgoingCableBuffer { .. }
                    )
                })
                .count();
            model_input_read_count += dispatch_stage
                .reads
                .iter()
                .filter(|io| matches!(io, VulkanMountedPlacedStreamTickIo::ModelSignal { .. }))
                .count();
            model_output_write_count += dispatch_stage
                .writes
                .iter()
                .filter(|io| matches!(io, VulkanMountedPlacedStreamTickIo::ModelSignal { .. }))
                .count();

            for descriptor in &dispatch.descriptors {
                if let VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer { endpoint } =
                    &descriptor.target
                    && received_endpoints
                        .insert((endpoint.endpoint.cable_index, endpoint.buffer_index))
                {
                    stages.push(VulkanMountedPlacedStreamTickStage::ReceiveCable {
                        stage_index: stages.len(),
                        cable_index: endpoint.endpoint.cable_index,
                        endpoint_id: endpoint.endpoint.endpoint_id.clone(),
                        buffer_index: endpoint.buffer_index,
                        byte_capacity: endpoint.byte_capacity,
                        remote_device_id: endpoint.endpoint.remote_device_id.clone(),
                        remote_pedal_id: endpoint.endpoint.remote_pedal_id.clone(),
                    });
                }
            }

            stages.push(VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index: stages.len(),
                dispatch: dispatch_stage,
            });

            for descriptor in &dispatch.descriptors {
                if let VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer { endpoint } =
                    &descriptor.target
                    && published_endpoints
                        .insert((endpoint.endpoint.cable_index, endpoint.buffer_index))
                {
                    stages.push(VulkanMountedPlacedStreamTickStage::PublishCable {
                        stage_index: stages.len(),
                        cable_index: endpoint.endpoint.cable_index,
                        endpoint_id: endpoint.endpoint.endpoint_id.clone(),
                        buffer_index: endpoint.buffer_index,
                        byte_capacity: endpoint.byte_capacity,
                        remote_device_id: endpoint.endpoint.remote_device_id.clone(),
                        remote_pedal_id: endpoint.endpoint.remote_pedal_id.clone(),
                    });
                }
            }
        }

        let receive_stage_count = received_endpoints.len();
        let publish_stage_count = published_endpoints.len();
        let dispatch_stage_count = mounted_bound_plan.dispatches.len();
        let stage_count = stages.len();

        Self {
            backend_id: mounted_bound_plan.backend_id.clone(),
            device_id: mounted_bound_plan.device_id.clone(),
            stages,
            stage_count,
            receive_stage_count,
            dispatch_stage_count,
            publish_stage_count,
            local_cable_read_count,
            local_cable_write_count,
            incoming_cable_read_count,
            outgoing_cable_write_count,
            model_input_read_count,
            model_output_write_count,
            can_execute: false,
        }
    }

    pub fn advance(&self, stream_tick: u64) -> VulkanMountedPlacedStreamTickRun {
        let mut stages = Vec::with_capacity(self.stages.len());
        let mut blocked = None;
        let mut attempted_stage_count = 0usize;
        let mut completed_stage_count = 0usize;

        for stage in &self.stages {
            let status = if blocked.is_some() {
                VulkanMountedPlacedStreamTickStageStatus::Pending
            } else {
                attempted_stage_count += 1;
                let reason = match stage {
                    VulkanMountedPlacedStreamTickStage::ReceiveCable { .. } => {
                        VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable
                    }
                    VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {
                        VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable
                    }
                    VulkanMountedPlacedStreamTickStage::PublishCable { .. } => {
                        VulkanMountedPlacedStreamTickBlockReason::CablePublishTransportUnavailable
                    }
                };
                blocked = Some((stage.stage_index(), reason.clone()));
                VulkanMountedPlacedStreamTickStageStatus::Blocked { reason }
            };
            if matches!(status, VulkanMountedPlacedStreamTickStageStatus::Completed) {
                completed_stage_count += 1;
            }
            stages.push(VulkanMountedPlacedStreamTickStageRun {
                stage_index: stage.stage_index(),
                stage: stage.clone(),
                status,
            });
        }

        let pending_stage_count = stages
            .iter()
            .filter(|stage| {
                matches!(
                    stage.status,
                    VulkanMountedPlacedStreamTickStageStatus::Pending
                )
            })
            .count();
        let status = blocked
            .map(
                |(stage_index, reason)| VulkanMountedPlacedStreamTickRunStatus::Blocked {
                    stage_index,
                    reason,
                },
            )
            .unwrap_or(VulkanMountedPlacedStreamTickRunStatus::Completed);

        VulkanMountedPlacedStreamTickRun {
            backend_id: self.backend_id.clone(),
            device_id: self.device_id.clone(),
            stream_tick,
            stages,
            planned_stage_count: self.stage_count,
            attempted_stage_count,
            completed_stage_count,
            pending_stage_count,
            status,
            can_execute: self.can_execute,
        }
    }

    pub fn advance_with_in_process_transport(
        &self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transport: &mut VulkanInProcessPlacedCableTransport,
        stream_tick: u64,
    ) -> Result<VulkanMountedPlacedStreamTickRun, VulkanMountedPlacedStreamTickTransportError> {
        if self.device_id != mounted.device_id() {
            return Err(
                VulkanMountedPlacedStreamTickTransportError::DeviceMismatch {
                    plan_device_id: self.device_id.clone(),
                    mounted_device_id: mounted.device_id().to_string(),
                },
            );
        }

        let mut stages = Vec::with_capacity(self.stages.len());
        let mut blocked = None;
        let mut attempted_stage_count = 0usize;
        let mut completed_stage_count = 0usize;

        for stage in &self.stages {
            let status = if blocked.is_some() {
                VulkanMountedPlacedStreamTickStageStatus::Pending
            } else {
                attempted_stage_count += 1;
                match stage {
                    VulkanMountedPlacedStreamTickStage::ReceiveCable { cable_index, .. } => {
                        match transport.receive_incoming_cable(mounted, *cable_index) {
                            Ok(_) => {
                                completed_stage_count += 1;
                                VulkanMountedPlacedStreamTickStageStatus::Completed
                            }
                            Err(VulkanPlacedCableTransportError::MissingPacket { .. }) => {
                                let reason =
                                    VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable;
                                blocked = Some((stage.stage_index(), reason.clone()));
                                VulkanMountedPlacedStreamTickStageStatus::Blocked { reason }
                            }
                            Err(error) => {
                                return Err(
                                    VulkanMountedPlacedStreamTickTransportError::Transport(error),
                                );
                            }
                        }
                    }
                    VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {
                        let reason =
                            VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable;
                        blocked = Some((stage.stage_index(), reason.clone()));
                        VulkanMountedPlacedStreamTickStageStatus::Blocked { reason }
                    }
                    VulkanMountedPlacedStreamTickStage::PublishCable { cable_index, .. } => {
                        transport
                            .publish_outgoing_cable(mounted, *cable_index)
                            .map_err(VulkanMountedPlacedStreamTickTransportError::Transport)?;
                        completed_stage_count += 1;
                        VulkanMountedPlacedStreamTickStageStatus::Completed
                    }
                }
            };
            stages.push(VulkanMountedPlacedStreamTickStageRun {
                stage_index: stage.stage_index(),
                stage: stage.clone(),
                status,
            });
        }

        let pending_stage_count = stages
            .iter()
            .filter(|stage| {
                matches!(
                    stage.status,
                    VulkanMountedPlacedStreamTickStageStatus::Pending
                )
            })
            .count();
        let status = blocked
            .map(
                |(stage_index, reason)| VulkanMountedPlacedStreamTickRunStatus::Blocked {
                    stage_index,
                    reason,
                },
            )
            .unwrap_or(VulkanMountedPlacedStreamTickRunStatus::Completed);

        Ok(VulkanMountedPlacedStreamTickRun {
            backend_id: self.backend_id.clone(),
            device_id: self.device_id.clone(),
            stream_tick,
            stages,
            planned_stage_count: self.stage_count,
            attempted_stage_count,
            completed_stage_count,
            pending_stage_count,
            status,
            can_execute: self.can_execute,
        })
    }

    pub fn advance_with_resident_pedalboard_and_in_process_transport(
        &self,
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        transport: &mut VulkanInProcessPlacedCableTransport,
        stream_tick: u64,
    ) -> Result<VulkanMountedPlacedResidentStreamTickRun, VulkanMountedPlacedResidentStreamTickError>
    {
        let mut cursor = self.resident_stream_tick_cursor(stream_tick);
        cursor.advance_with_resident_pedals_and_in_process_transport(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            transport,
        )?;
        Ok(cursor.snapshot())
    }

    pub fn resident_stream_tick_cursor(
        &self,
        stream_tick: u64,
    ) -> VulkanMountedPlacedResidentStreamTickCursor {
        VulkanMountedPlacedResidentStreamTickCursor::new(self.clone(), stream_tick)
    }
}

