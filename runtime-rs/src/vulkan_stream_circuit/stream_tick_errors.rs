#[derive(Debug)]
pub enum VulkanMountedPlacedResidentStreamTickError {
    DeviceMismatch {
        plan_device_id: String,
        mounted_device_id: String,
    },
    BoundPlanDeviceMismatch {
        plan_device_id: String,
        bound_plan_device_id: String,
    },
    DynamicStateCapacityOverflow {
        capacity: usize,
    },
    Transport(VulkanPlacedEdgeTransportError),
    Dispatch(VulkanMountedPlacedResidentKernelDispatchError),
}

impl Display for VulkanMountedPlacedResidentStreamTickError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceMismatch {
                plan_device_id,
                mounted_device_id,
            } => write!(
                f,
                "resident stream tick plan for device {plan_device_id:?} cannot advance mounted device {mounted_device_id:?}"
            ),
            Self::BoundPlanDeviceMismatch {
                plan_device_id,
                bound_plan_device_id,
            } => write!(
                f,
                "resident stream tick plan for device {plan_device_id:?} cannot use mounted bound plan for device {bound_plan_device_id:?}"
            ),
            Self::DynamicStateCapacityOverflow { capacity } => write!(
                f,
                "resident stream tick dynamic-state capacity {capacity} cannot fit in u32 push constants"
            ),
            Self::Transport(error) => Display::fmt(error, f),
            Self::Dispatch(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanMountedPlacedResidentStreamTickError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedResidentKernelDispatchError {
    ExecutionPlanDeviceMismatch {
        plan_device_id: String,
        mounted_device_id: String,
    },
    ExecutionBoundPlanDeviceMismatch {
        plan_device_id: String,
        bound_plan_device_id: String,
    },
    EmptyDispatchSegment {
        device_id: String,
    },
    DispatchSegmentStageOverflow {
        device_id: String,
    },
    NonDispatchStageInSegment {
        device_id: String,
        stage_index: usize,
    },
    MissingSegmentDispatch {
        device_id: String,
        stage_index: usize,
        dispatch_index: usize,
    },
    MissingDispatchSegment {
        device_id: String,
        stage_index: usize,
    },
    MissingDistributedDispatchStage {
        device_id: String,
        dispatch_index: usize,
    },
    DistributedDispatchMismatch {
        device_id: String,
        stage_index: usize,
        expected_dispatch_index: usize,
        completed_dispatch_index: usize,
    },
    MissingExecutionDispatchSegments {
        device_id: String,
    },
    MissingExecutionGraphComponents {
        device_id: String,
    },
    MissingComponentDispatches {
        component_id: String,
    },
    MissingLoadedArtifact {
        dispatch_index: usize,
        family_id: String,
    },
    DescriptorBindingOverflow {
        dispatch_index: usize,
        binding: usize,
    },
    MissingMountedBuffer {
        dispatch_index: usize,
        binding: usize,
        buffer_kind: String,
        buffer_index: usize,
    },
    MissingModelBoundaryBuffer {
        dispatch_index: usize,
        binding: usize,
        direction: VulkanModelBoundaryDirection,
        signal_id: String,
    },
    MissingPermanentParameterBuffer {
        dispatch_index: usize,
        binding: usize,
        param_id: String,
        tensor: String,
        byte_count: Option<usize>,
    },
    PermanentParameterBufferUnavailable {
        dispatch_index: usize,
        binding: usize,
        param_id: String,
        tensor: String,
        byte_count: Option<usize>,
    },
    ModelBoundaryBufferUnavailable {
        dispatch_index: usize,
        binding: usize,
        signal_id: String,
    },
    UnsupportedPushConstantScalar {
        scalar_type: String,
    },
    UnsupportedPushConstantBinding {
        name: String,
        scalar_type: String,
    },
    PushConstantByteCountOverflow,
    ComponentRunnerDescriptorCountOverflow {
        component_id: String,
    },
    ComponentRunnerPushConstantByteCountOverflow {
        component_id: String,
    },
    ExecutionGraphRunnerDescriptorCountOverflow {
        device_id: String,
    },
    ExecutionGraphRunnerPushConstantByteCountOverflow {
        device_id: String,
    },
    Vulkan(VulkanError),
}

impl Display for VulkanMountedPlacedResidentKernelDispatchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExecutionPlanDeviceMismatch {
                plan_device_id,
                mounted_device_id,
            } => write!(
                f,
                "resident execution plan for device {plan_device_id:?} cannot execute mounted device {mounted_device_id:?}"
            ),
            Self::ExecutionBoundPlanDeviceMismatch {
                plan_device_id,
                bound_plan_device_id,
            } => write!(
                f,
                "resident execution plan for device {plan_device_id:?} cannot use mounted bound plan for device {bound_plan_device_id:?}"
            ),
            Self::EmptyDispatchSegment { device_id } => write!(
                f,
                "resident execution plan for device {device_id:?} contains an empty dispatch segment"
            ),
            Self::DispatchSegmentStageOverflow { device_id } => write!(
                f,
                "resident execution plan dispatch segment stage index overflowed for device {device_id:?}"
            ),
            Self::NonDispatchStageInSegment {
                device_id,
                stage_index,
            } => write!(
                f,
                "resident execution plan for device {device_id:?} contains non-dispatch stage {stage_index} inside a dispatch segment"
            ),
            Self::MissingSegmentDispatch {
                device_id,
                stage_index,
                dispatch_index,
            } => write!(
                f,
                "resident execution plan for device {device_id:?} stage {stage_index} references missing bound dispatch {dispatch_index}"
            ),
            Self::MissingDispatchSegment {
                device_id,
                stage_index,
            } => write!(
                f,
                "resident execution plan for device {device_id:?} has no dispatch segment beginning at stage {stage_index}"
            ),
            Self::MissingDistributedDispatchStage {
                device_id,
                dispatch_index,
            } => write!(
                f,
                "resident execution plan for device {device_id:?} cannot replace missing dispatch {dispatch_index}"
            ),
            Self::DistributedDispatchMismatch {
                device_id,
                stage_index,
                expected_dispatch_index,
                completed_dispatch_index,
            } => write!(
                f,
                "resident execution plan for device {device_id:?} stage {stage_index} expected distributed dispatch {expected_dispatch_index}, completed {completed_dispatch_index}"
            ),
            Self::MissingExecutionDispatchSegments { device_id } => write!(
                f,
                "resident execution plan for device {device_id:?} has no dispatch segments"
            ),
            Self::MissingExecutionGraphComponents { device_id } => {
                write!(
                    f,
                    "resident execution_graph runner for device {device_id:?} has no components"
                )
            }
            Self::MissingComponentDispatches { component_id } => {
                write!(f, "component {component_id:?} has no mounted dispatches")
            }
            Self::MissingLoadedArtifact {
                dispatch_index,
                family_id,
            } => write!(
                f,
                "dispatch {dispatch_index} cannot create a resident kernel dispatch because loaded artifact {family_id:?} is missing"
            ),
            Self::DescriptorBindingOverflow {
                dispatch_index,
                binding,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor binding {binding} cannot fit in u32"
            ),
            Self::MissingMountedBuffer {
                dispatch_index,
                binding,
                buffer_kind,
                buffer_index,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing mounted {buffer_kind} buffer {buffer_index}"
            ),
            Self::MissingModelBoundaryBuffer {
                dispatch_index,
                binding,
                direction,
                signal_id,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing mounted model {direction:?} boundary buffer {signal_id:?}"
            ),
            Self::MissingPermanentParameterBuffer {
                dispatch_index,
                binding,
                param_id,
                tensor,
                byte_count,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing mounted permanent parameter {param_id:?} tensor {tensor:?} ({byte_count:?} bytes)"
            ),
            Self::PermanentParameterBufferUnavailable {
                dispatch_index,
                binding,
                param_id,
                tensor,
                byte_count,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references permanent parameter {param_id:?} tensor {tensor:?} ({byte_count:?} bytes), but permanent parameter buffers are not mounted yet"
            ),
            Self::ModelBoundaryBufferUnavailable {
                dispatch_index,
                binding,
                signal_id,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references model boundary signal {signal_id:?}, but model boundary buffers are not mounted yet"
            ),
            Self::UnsupportedPushConstantScalar { scalar_type } => {
                write!(f, "unsupported push-constant scalar type {scalar_type:?}")
            }
            Self::UnsupportedPushConstantBinding { name, scalar_type } => write!(
                f,
                "unsupported push-constant binding {name:?} with scalar type {scalar_type:?}"
            ),
            Self::PushConstantByteCountOverflow => {
                f.write_str("push-constant byte count overflowed")
            }
            Self::ComponentRunnerDescriptorCountOverflow { component_id } => write!(
                f,
                "resident component runner {component_id:?} descriptor count overflowed"
            ),
            Self::ComponentRunnerPushConstantByteCountOverflow { component_id } => write!(
                f,
                "resident component runner {component_id:?} push-constant byte count overflowed"
            ),
            Self::ExecutionGraphRunnerDescriptorCountOverflow { device_id } => write!(
                f,
                "resident execution_graph runner for device {device_id:?} descriptor count overflowed"
            ),
            Self::ExecutionGraphRunnerPushConstantByteCountOverflow { device_id } => write!(
                f,
                "resident execution_graph runner for device {device_id:?} push-constant byte count overflowed"
            ),
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanMountedPlacedResidentKernelDispatchError {}

fn push_constant_byte_count(
    push_constants: &[VulkanKernelScalarBinding],
) -> Result<u32, VulkanMountedPlacedResidentKernelDispatchError> {
    push_constants.iter().try_fold(0u32, |total, binding| {
        let bytes = push_constant_scalar_byte_count(&binding.scalar_type)?;
        total
            .checked_add(bytes)
            .ok_or(VulkanMountedPlacedResidentKernelDispatchError::PushConstantByteCountOverflow)
    })
}

fn push_constant_scalar_byte_count(
    scalar_type: &str,
) -> Result<u32, VulkanMountedPlacedResidentKernelDispatchError> {
    match scalar_type {
        "u32" | "i32" | "f32" => Ok(4),
        "u64" | "i64" | "f64" => Ok(8),
        _ => Err(
            VulkanMountedPlacedResidentKernelDispatchError::UnsupportedPushConstantScalar {
                scalar_type: scalar_type.to_string(),
            },
        ),
    }
}

fn stream_control_push_constant_bytes(
    push_constants: &[VulkanKernelScalarBinding],
    control: VulkanMountedPlacedStreamControl,
) -> Result<Vec<u8>, VulkanMountedPlacedResidentKernelDispatchError> {
    let byte_count = push_constant_byte_count(push_constants)?;
    let mut bytes = Vec::with_capacity(byte_count as usize);

    for binding in push_constants {
        match (binding.name.as_str(), binding.scalar_type.as_str()) {
            ("stream_tick", "u64") => {
                bytes.extend_from_slice(&control.stream_tick.to_le_bytes());
            }
            ("control_flags", "u32") => {
                bytes.extend_from_slice(&control.control_flags.to_le_bytes());
            }
            ("dynamic_state_capacity_activations", "u32") => {
                bytes.extend_from_slice(&control.dynamic_state_capacity_activations.to_le_bytes());
            }
            ("expert_start", "u32") => {
                bytes.extend_from_slice(&0u32.to_le_bytes());
            }
            _ => {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::UnsupportedPushConstantBinding {
                        name: binding.name.clone(),
                        scalar_type: binding.scalar_type.clone(),
                    },
                );
            }
        }
    }

    Ok(bytes)
}
