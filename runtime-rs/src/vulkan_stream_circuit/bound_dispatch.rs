#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedBoundDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub component_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub artifact_path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptors: Vec<VulkanPlacedBoundDescriptor>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedBoundDescriptor {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub name: String,
    pub target: VulkanPlacedBoundDescriptorTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanPlacedBoundDescriptorTarget {
    Resident { target: VulkanBoundDescriptorTarget },
    ModelInput { signal_id: String },
    ModelOutput { signal_id: String },
    LocalEdgeInput { edge: ComponentEdgePlacement },
    LocalEdgeOutput { edge: ComponentEdgePlacement },
    IncomingEdge { edge: ComponentEdgePlacement },
    OutgoingEdge { edge: ComponentEdgePlacement },
}

impl VulkanPlacedBoundDescriptorTarget {
    fn from_bound_target(
        component_id: &str,
        target: &VulkanBoundDescriptorTarget,
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Self {
        match target {
            VulkanBoundDescriptorTarget::BoundaryInput { signal_id } => {
                classify_boundary_input(component_id, signal_id, placed_resident_plan)
            }
            VulkanBoundDescriptorTarget::BoundaryOutput { signal_id } => {
                classify_boundary_output(component_id, signal_id, placed_resident_plan)
            }
            _ => Self::Resident {
                target: target.clone(),
            },
        }
    }
}

fn classify_boundary_input(
    component_id: &str,
    signal_id: &str,
    placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
) -> VulkanPlacedBoundDescriptorTarget {
    if let Some(edge) = placed_resident_plan
        .local_edges
        .iter()
        .find(|edge| {
            edge.destination_component_id == component_id && edge.destination_port_id == signal_id
        })
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::LocalEdgeInput { edge };
    }
    if let Some(edge) = placed_resident_plan
        .incoming_edges
        .iter()
        .find(|edge| {
            edge.destination_component_id == component_id && edge.destination_port_id == signal_id
        })
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::IncomingEdge { edge };
    }
    VulkanPlacedBoundDescriptorTarget::ModelInput {
        signal_id: signal_id.to_string(),
    }
}

fn classify_boundary_output(
    component_id: &str,
    signal_id: &str,
    placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
) -> VulkanPlacedBoundDescriptorTarget {
    if let Some(edge) = placed_resident_plan
        .local_edges
        .iter()
        .find(|edge| edge.source_component_id == component_id && edge.source_port_id == signal_id)
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::LocalEdgeOutput { edge };
    }
    if let Some(edge) = placed_resident_plan
        .outgoing_edges
        .iter()
        .find(|edge| edge.source_component_id == component_id && edge.source_port_id == signal_id)
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::OutgoingEdge { edge };
    }
    VulkanPlacedBoundDescriptorTarget::ModelOutput {
        signal_id: signal_id.to_string(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBoundDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub component_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub artifact_path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptors: Vec<VulkanBoundDescriptor>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBoundDescriptor {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub name: String,
    pub target: VulkanBoundDescriptorTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanBoundDescriptorTarget {
    BoundaryInput {
        signal_id: String,
    },
    BoundaryOutput {
        signal_id: String,
    },
    PermanentParameter {
        param_id: String,
        tensor: String,
        byte_count: Option<usize>,
    },
    ActivationSlot {
        buffer_index: usize,
        component_id: String,
        signal_id: String,
        circuit_id: String,
        slot: usize,
        byte_capacity: usize,
        signal_byte_capacity: usize,
    },
    StreamStateBuffer {
        buffer_index: usize,
        component_id: String,
        state_id: String,
        state_type: String,
        byte_capacity: usize,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    StreamStateView {
        buffer_index: usize,
        component_id: String,
        state_id: String,
        state_type: String,
        byte_capacity: usize,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
}

impl VulkanBoundDescriptorTarget {
    fn from_resource(
        dispatch: &VulkanPreparedDispatch,
        descriptor: &VulkanResolvedDescriptorBinding,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        match &descriptor.resource {
            VulkanDescriptorResourceAddress::BoundaryInput { signal_id } => {
                Ok(Self::BoundaryInput {
                    signal_id: signal_id.clone(),
                })
            }
            VulkanDescriptorResourceAddress::BoundaryOutput { signal_id } => {
                Ok(Self::BoundaryOutput {
                    signal_id: signal_id.clone(),
                })
            }
            VulkanDescriptorResourceAddress::PermanentParameter {
                param_id,
                tensor,
                byte_count,
            } => Ok(Self::PermanentParameter {
                param_id: param_id.clone(),
                tensor: tensor.clone(),
                byte_count: *byte_count,
            }),
            VulkanDescriptorResourceAddress::ActivationSlot {
                component_id,
                signal_id,
                slot,
                byte_capacity,
                signal_byte_capacity,
            } => {
                let buffer_index = buffers
                    .activation_slot_buffer_index(component_id, *slot)
                    .ok_or_else(
                        || VulkanBoundDispatchPlanError::MissingActivationSlotBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            component_id: component_id.clone(),
                            slot: *slot,
                        },
                    )?;
                let buffer = &buffers.activation_slot_buffers[buffer_index];
                validate_bound_byte_capacity(
                    dispatch,
                    descriptor,
                    *byte_capacity,
                    buffer.byte_capacity,
                )?;
                Ok(Self::ActivationSlot {
                    buffer_index,
                    component_id: component_id.clone(),
                    signal_id: signal_id.clone(),
                    circuit_id: buffer.circuit_id.clone(),
                    slot: *slot,
                    byte_capacity: *byte_capacity,
                    signal_byte_capacity: *signal_byte_capacity,
                })
            }
            VulkanDescriptorResourceAddress::StateBuffer {
                component_id,
                state_id,
                state_type,
                byte_capacity,
                static_bytes,
                bytes_per_activation,
            } => {
                let buffer_index =
                    buffers
                        .state_buffer_index(component_id, state_id)
                        .ok_or_else(|| VulkanBoundDispatchPlanError::MissingStateBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            component_id: component_id.clone(),
                            state_id: state_id.clone(),
                        })?;
                let buffer = &buffers.state_buffers[buffer_index];
                validate_bound_byte_capacity(
                    dispatch,
                    descriptor,
                    *byte_capacity,
                    buffer.byte_capacity,
                )?;
                Ok(Self::StreamStateBuffer {
                    buffer_index,
                    component_id: component_id.clone(),
                    state_id: state_id.clone(),
                    state_type: state_type.clone(),
                    byte_capacity: *byte_capacity,
                    static_bytes: *static_bytes,
                    bytes_per_activation: *bytes_per_activation,
                })
            }
            VulkanDescriptorResourceAddress::StateView {
                component_id,
                state_id,
                state_type,
                byte_capacity,
                static_bytes,
                bytes_per_activation,
            } => {
                let buffer_index =
                    buffers
                        .state_buffer_index(component_id, state_id)
                        .ok_or_else(|| VulkanBoundDispatchPlanError::MissingStateBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            component_id: component_id.clone(),
                            state_id: state_id.clone(),
                        })?;
                let buffer = &buffers.state_buffers[buffer_index];
                validate_bound_byte_capacity(
                    dispatch,
                    descriptor,
                    *byte_capacity,
                    buffer.byte_capacity,
                )?;
                Ok(Self::StreamStateView {
                    buffer_index,
                    component_id: component_id.clone(),
                    state_id: state_id.clone(),
                    state_type: state_type.clone(),
                    byte_capacity: *byte_capacity,
                    static_bytes: *static_bytes,
                    bytes_per_activation: *bytes_per_activation,
                })
            }
        }
    }
}

fn validate_bound_byte_capacity(
    dispatch: &VulkanPreparedDispatch,
    descriptor: &VulkanResolvedDescriptorBinding,
    expected_byte_capacity: usize,
    mounted_byte_capacity: usize,
) -> Result<(), VulkanBoundDispatchPlanError> {
    if expected_byte_capacity != mounted_byte_capacity {
        return Err(VulkanBoundDispatchPlanError::ByteCapacityMismatch {
            dispatch_index: dispatch.dispatch_index,
            binding: descriptor.binding,
            expected_byte_capacity,
            mounted_byte_capacity,
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanBoundDispatchPlanError {
    PreparedDispatch(VulkanPreparedDispatchPlanError),
    EdgeIoDeviceMismatch {
        plan_device_id: String,
        edge_io_device_id: String,
    },
    MissingStateBuffer {
        dispatch_index: usize,
        binding: usize,
        component_id: String,
        state_id: String,
    },
    MissingActivationSlotBuffer {
        dispatch_index: usize,
        binding: usize,
        component_id: String,
        slot: usize,
    },
    MissingEdgeEndpointBuffer {
        dispatch_index: usize,
        binding: usize,
        direction: VulkanPlacedEdgeDirection,
        edge_index: usize,
    },
    MissingLocalEdgeBuffer {
        dispatch_index: usize,
        binding: usize,
        edge_index: usize,
    },
    ByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        expected_byte_capacity: usize,
        mounted_byte_capacity: usize,
    },
    LocalEdgeByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        edge_index: usize,
        edge_byte_capacity: Option<usize>,
        mounted_byte_capacity: usize,
    },
    EdgeEndpointByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        edge_index: usize,
        endpoint_byte_capacity: Option<usize>,
        mounted_byte_capacity: usize,
    },
}

impl Display for VulkanBoundDispatchPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreparedDispatch(error) => Display::fmt(error, f),
            Self::EdgeIoDeviceMismatch {
                plan_device_id,
                edge_io_device_id,
            } => write!(
                f,
                "placed bound plan for device {plan_device_id:?} cannot bind edge I/O for device {edge_io_device_id:?}"
            ),
            Self::MissingStateBuffer {
                dispatch_index,
                binding,
                component_id,
                state_id,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing stream state buffer {component_id}.{state_id}"
            ),
            Self::MissingActivationSlotBuffer {
                dispatch_index,
                binding,
                component_id,
                slot,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing activation slot buffer {component_id}.slot_{slot}"
            ),
            Self::MissingEdgeEndpointBuffer {
                dispatch_index,
                binding,
                direction,
                edge_index,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing {direction:?} edge endpoint buffer for edge {edge_index}"
            ),
            Self::MissingLocalEdgeBuffer {
                dispatch_index,
                binding,
                edge_index,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing local edge buffer for edge {edge_index}"
            ),
            Self::ByteCapacityMismatch {
                dispatch_index,
                binding,
                expected_byte_capacity,
                mounted_byte_capacity,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} expects {expected_byte_capacity} bytes but mounted buffer has {mounted_byte_capacity} bytes"
            ),
            Self::LocalEdgeByteCapacityMismatch {
                dispatch_index,
                binding,
                edge_index,
                edge_byte_capacity,
                mounted_byte_capacity,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} local edge {edge_index} expects {edge_byte_capacity:?} bytes but mounted buffer has {mounted_byte_capacity} bytes"
            ),
            Self::EdgeEndpointByteCapacityMismatch {
                dispatch_index,
                binding,
                edge_index,
                endpoint_byte_capacity,
                mounted_byte_capacity,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} edge {edge_index} endpoint expects {endpoint_byte_capacity:?} bytes but mounted buffer has {mounted_byte_capacity} bytes"
            ),
        }
    }
}

impl Error for VulkanBoundDispatchPlanError {}
