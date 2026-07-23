#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentStreamTickRun {
    pub tick_run: VulkanMountedPlacedStreamTickRun,
    pub pedalboard_run: Option<VulkanMountedPlacedResidentPedalboardRun>,
}

impl VulkanMountedPlacedResidentStreamTickRun {
    pub fn pedalboard_dispatch_count(&self) -> usize {
        self.pedalboard_run
            .as_ref()
            .map(VulkanMountedPlacedResidentPedalboardRun::dispatch_count)
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickStage {
    ReceiveCable {
        stage_index: usize,
        cable_index: usize,
        endpoint_id: String,
        buffer_index: usize,
        byte_capacity: usize,
        remote_device_id: String,
        remote_pedal_id: String,
    },
    Dispatch {
        stage_index: usize,
        dispatch: VulkanMountedPlacedStreamTickDispatch,
    },
    PublishCable {
        stage_index: usize,
        cable_index: usize,
        endpoint_id: String,
        buffer_index: usize,
        byte_capacity: usize,
        remote_device_id: String,
        remote_pedal_id: String,
    },
}

impl VulkanMountedPlacedStreamTickStage {
    pub fn stage_index(&self) -> usize {
        match self {
            Self::ReceiveCable { stage_index, .. }
            | Self::Dispatch { stage_index, .. }
            | Self::PublishCable { stage_index, .. } => *stage_index,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub node_id: String,
    pub op: String,
    pub descriptor_count: usize,
    pub resident_descriptor_count: usize,
    pub reads: Vec<VulkanMountedPlacedStreamTickIo>,
    pub writes: Vec<VulkanMountedPlacedStreamTickIo>,
}

impl VulkanMountedPlacedStreamTickDispatch {
    fn from_bound_dispatch(dispatch: &VulkanMountedPlacedBoundDispatch) -> Self {
        let mut resident_descriptor_count = 0usize;
        let mut reads = Vec::new();
        let mut writes = Vec::new();

        for descriptor in &dispatch.descriptors {
            match &descriptor.target {
                VulkanMountedPlacedBoundDescriptorTarget::Resident { .. } => {
                    resident_descriptor_count += 1;
                }
                VulkanMountedPlacedBoundDescriptorTarget::ModelInput { signal_id } => {
                    reads.push(VulkanMountedPlacedStreamTickIo::ModelSignal {
                        signal_id: signal_id.clone(),
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
                    writes.push(VulkanMountedPlacedStreamTickIo::ModelSignal {
                        signal_id: signal_id.clone(),
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer { cable } => {
                    reads.push(VulkanMountedPlacedStreamTickIo::LocalCableBuffer {
                        cable_index: cable.cable.cable_index,
                        buffer_index: cable.buffer_index,
                        byte_capacity: cable.byte_capacity,
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer { cable } => {
                    writes.push(VulkanMountedPlacedStreamTickIo::LocalCableBuffer {
                        cable_index: cable.cable.cable_index,
                        buffer_index: cable.buffer_index,
                        byte_capacity: cable.byte_capacity,
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer { endpoint } => {
                    reads.push(VulkanMountedPlacedStreamTickIo::IncomingCableBuffer {
                        cable_index: endpoint.endpoint.cable_index,
                        buffer_index: endpoint.buffer_index,
                        byte_capacity: endpoint.byte_capacity,
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer { endpoint } => {
                    writes.push(VulkanMountedPlacedStreamTickIo::OutgoingCableBuffer {
                        cable_index: endpoint.endpoint.cable_index,
                        buffer_index: endpoint.buffer_index,
                        byte_capacity: endpoint.byte_capacity,
                    });
                }
            }
        }

        Self {
            dispatch_index: dispatch.dispatch_index,
            kernel_id: dispatch.kernel_id.clone(),
            pedal_id: dispatch.pedal_id.clone(),
            node_id: dispatch.node_id.clone(),
            op: dispatch.op.clone(),
            descriptor_count: dispatch.descriptors.len(),
            resident_descriptor_count,
            reads,
            writes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickIo {
    ModelSignal {
        signal_id: String,
    },
    LocalCableBuffer {
        cable_index: usize,
        buffer_index: usize,
        byte_capacity: usize,
    },
    IncomingCableBuffer {
        cable_index: usize,
        buffer_index: usize,
        byte_capacity: usize,
    },
    OutgoingCableBuffer {
        cable_index: usize,
        buffer_index: usize,
        byte_capacity: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickRun {
    pub backend_id: String,
    pub device_id: String,
    pub stream_tick: u64,
    pub stages: Vec<VulkanMountedPlacedStreamTickStageRun>,
    pub planned_stage_count: usize,
    pub attempted_stage_count: usize,
    pub completed_stage_count: usize,
    pub pending_stage_count: usize,
    pub status: VulkanMountedPlacedStreamTickRunStatus,
    pub can_execute: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickStageRun {
    pub stage_index: usize,
    pub stage: VulkanMountedPlacedStreamTickStage,
    pub status: VulkanMountedPlacedStreamTickStageStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickRunStatus {
    Completed,
    Blocked {
        stage_index: usize,
        reason: VulkanMountedPlacedStreamTickBlockReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickStageStatus {
    Pending,
    Completed,
    Blocked {
        reason: VulkanMountedPlacedStreamTickBlockReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickBlockReason {
    CableReceiveTransportUnavailable,
    KernelDispatchUnavailable,
    CablePublishTransportUnavailable,
    DistributedDispatchPending { dispatch_index: usize },
}
