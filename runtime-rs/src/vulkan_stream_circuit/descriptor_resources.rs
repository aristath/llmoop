#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanKernelDescriptorResource {
    Signal(VulkanSignalBinding),
    Parameter(VulkanParameterBinding),
    State {
        component_id: String,
        binding: VulkanStateBinding,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDescriptorResourcePlan {
    pub backend_id: String,
    pub dynamic_state_capacity_activations: usize,
    pub dispatches: Vec<VulkanDispatchDescriptorResourcePlan>,
    pub total_descriptor_count: usize,
}

impl VulkanDescriptorResourcePlan {
    pub fn from_plans(
        dispatch_plan: &VulkanKernelDispatchPlan,
        resident_plan: &VulkanStreamCircuitResidentPlan,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanDescriptorResourcePlanError> {
        let parameter_index: BTreeMap<_, _> = resident_plan
            .permanent_parameters
            .iter()
            .map(|parameter| (parameter.tensor.as_str(), parameter))
            .collect();
        let state_index: BTreeMap<_, _> = resident_plan
            .stream_state_buffers
            .iter()
            .map(|state| ((state.component_id.as_str(), state.state_id.as_str()), state))
            .collect();
        let activation_index: BTreeMap<_, _> = resident_plan
            .activation_banks
            .iter()
            .flat_map(|bank| {
                bank.slots
                    .iter()
                    .map(move |slot| ((bank.component_id.as_str(), slot.slot), slot))
            })
            .collect();

        let dispatches = dispatch_plan
            .commands
            .iter()
            .map(|command| {
                VulkanDispatchDescriptorResourcePlan::from_command(
                    command,
                    &parameter_index,
                    &state_index,
                    &activation_index,
                    dynamic_state_capacity_activations,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total_descriptor_count = dispatches
            .iter()
            .map(|dispatch| dispatch.descriptors.len())
            .sum();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            dynamic_state_capacity_activations,
            dispatches,
            total_descriptor_count,
        })
    }

    pub fn dispatch(
        &self,
        component_id: &str,
        node_id: &str,
    ) -> Option<&VulkanDispatchDescriptorResourcePlan> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.component_id == component_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDispatchDescriptorResourcePlan {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub component_id: String,
    pub node_id: String,
    pub op: String,
    pub descriptors: Vec<VulkanResolvedDescriptorBinding>,
}

impl VulkanDispatchDescriptorResourcePlan {
    fn from_command(
        command: &VulkanKernelDispatchCommand,
        parameter_index: &BTreeMap<&str, &VulkanResidentParameter>,
        state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
        activation_index: &BTreeMap<(&str, usize), &VulkanResidentActivationSlot>,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanDescriptorResourcePlanError> {
        let descriptors = command
            .descriptor_bindings
            .iter()
            .map(|descriptor| {
                VulkanResolvedDescriptorBinding::from_binding(
                    command,
                    descriptor,
                    parameter_index,
                    state_index,
                    activation_index,
                    dynamic_state_capacity_activations,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            dispatch_index: command.dispatch_index,
            kernel_id: command.kernel_id.clone(),
            component_id: command.component_id.clone(),
            node_id: command.node_id.clone(),
            op: command.op.clone(),
            descriptors,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResolvedDescriptorBinding {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub name: String,
    pub resource: VulkanDescriptorResourceAddress,
}

impl VulkanResolvedDescriptorBinding {
    fn from_binding(
        command: &VulkanKernelDispatchCommand,
        descriptor: &VulkanKernelDescriptorBinding,
        parameter_index: &BTreeMap<&str, &VulkanResidentParameter>,
        state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
        activation_index: &BTreeMap<(&str, usize), &VulkanResidentActivationSlot>,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanDescriptorResourcePlanError> {
        Ok(Self {
            binding: descriptor.binding,
            usage: descriptor.usage.clone(),
            name: descriptor.name.clone(),
            resource: resolve_descriptor_resource(
                command,
                descriptor,
                parameter_index,
                state_index,
                activation_index,
                dynamic_state_capacity_activations,
            )?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanDescriptorResourceAddress {
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
        component_id: String,
        signal_id: String,
        slot: usize,
        byte_capacity: usize,
        signal_byte_capacity: usize,
    },
    StateBuffer {
        component_id: String,
        state_id: String,
        state_type: String,
        byte_capacity: usize,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    StateView {
        component_id: String,
        state_id: String,
        state_type: String,
        byte_capacity: usize,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDescriptorResourcePlanError(pub String);

impl Display for VulkanDescriptorResourcePlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDescriptorResourcePlanError {}

fn resolve_descriptor_resource(
    command: &VulkanKernelDispatchCommand,
    descriptor: &VulkanKernelDescriptorBinding,
    parameter_index: &BTreeMap<&str, &VulkanResidentParameter>,
    state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
    activation_index: &BTreeMap<(&str, usize), &VulkanResidentActivationSlot>,
    dynamic_state_capacity_activations: usize,
) -> Result<VulkanDescriptorResourceAddress, VulkanDescriptorResourcePlanError> {
    match &descriptor.resource {
        VulkanKernelDescriptorResource::Signal(signal) => resolve_signal_descriptor_resource(
            command,
            descriptor,
            signal,
            state_index,
            activation_index,
            dynamic_state_capacity_activations,
        ),
        VulkanKernelDescriptorResource::Parameter(parameter) => {
            let resident = parameter_index
                .get(parameter.tensor.as_str())
                .ok_or_else(|| {
                    VulkanDescriptorResourcePlanError(format!(
                        "{} descriptor {} parameter tensor {:?} is not resident",
                        command.kernel_id, descriptor.binding, parameter.tensor
                    ))
                })?;
            if parameter.byte_count != resident.byte_count {
                return Err(VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} parameter {:?} byte count {:?} does not match resident {:?}",
                    command.kernel_id,
                    descriptor.binding,
                    parameter.tensor,
                    parameter.byte_count,
                    resident.byte_count
                )));
            }
            Ok(VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: parameter.param_id.clone(),
                tensor: parameter.tensor.clone(),
                byte_count: resident.byte_count,
            })
        }
        VulkanKernelDescriptorResource::State { component_id, binding } => {
            resolve_state_descriptor_resource(
                command,
                descriptor,
                component_id,
                binding,
                state_index,
                dynamic_state_capacity_activations,
                false,
            )
        }
    }
}

fn resolve_signal_descriptor_resource(
    command: &VulkanKernelDispatchCommand,
    descriptor: &VulkanKernelDescriptorBinding,
    signal: &VulkanSignalBinding,
    state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
    activation_index: &BTreeMap<(&str, usize), &VulkanResidentActivationSlot>,
    dynamic_state_capacity_activations: usize,
) -> Result<VulkanDescriptorResourceAddress, VulkanDescriptorResourcePlanError> {
    match &signal.resource {
        VulkanSignalResource::BoundaryInput => Ok(VulkanDescriptorResourceAddress::BoundaryInput {
            signal_id: signal.signal_id.clone(),
        }),
        VulkanSignalResource::BoundaryOutput => {
            Ok(VulkanDescriptorResourceAddress::BoundaryOutput {
                signal_id: signal.signal_id.clone(),
            })
        }
        VulkanSignalResource::ActivationSlot {
            component_id,
            slot,
            bytes,
            signal_bytes,
        } => {
            let resident = activation_index
                .get(&(component_id.as_str(), *slot))
                .ok_or_else(|| {
                    VulkanDescriptorResourcePlanError(format!(
                        "{} descriptor {} activation slot {}.{} is not resident",
                        command.kernel_id, descriptor.binding, component_id, slot
                    ))
                })?;
            let byte_capacity = resident.bytes.ok_or_else(|| {
                VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} activation slot {}.{} has unknown byte capacity",
                    command.kernel_id, descriptor.binding, component_id, slot
                ))
            })?;
            if *bytes != Some(byte_capacity) {
                return Err(VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} activation slot {}.{} byte count {:?} does not match resident {}",
                    command.kernel_id, descriptor.binding, component_id, slot, bytes, byte_capacity
                )));
            }
            let signal_byte_capacity = signal_bytes.ok_or_else(|| {
                VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} activation signal {:?} has unknown byte capacity",
                    command.kernel_id, descriptor.binding, signal.signal_id
                ))
            })?;
            if signal_byte_capacity > byte_capacity {
                return Err(VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} activation signal {:?} byte count {} exceeds resident slot capacity {}",
                    command.kernel_id,
                    descriptor.binding,
                    signal.signal_id,
                    signal_byte_capacity,
                    byte_capacity
                )));
            }
            Ok(VulkanDescriptorResourceAddress::ActivationSlot {
                component_id: component_id.clone(),
                signal_id: signal.signal_id.clone(),
                slot: *slot,
                byte_capacity,
                signal_byte_capacity,
            })
        }
        VulkanSignalResource::StateBuffer { .. } | VulkanSignalResource::StateView { .. } => {
            resolve_signal_state_descriptor_resource(
                command,
                descriptor,
                &signal.resource,
                state_index,
                dynamic_state_capacity_activations,
            )
        }
    }
}

fn resolve_signal_state_descriptor_resource(
    command: &VulkanKernelDispatchCommand,
    descriptor: &VulkanKernelDescriptorBinding,
    resource: &VulkanSignalResource,
    state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
    dynamic_state_capacity_activations: usize,
) -> Result<VulkanDescriptorResourceAddress, VulkanDescriptorResourcePlanError> {
    let (component_id, state_id, static_bytes, bytes_per_activation, state_view) = match resource {
        VulkanSignalResource::StateBuffer {
            component_id,
            state_id,
            static_bytes,
            bytes_per_activation,
        } => (
            component_id,
            state_id,
            static_bytes,
            bytes_per_activation,
            false,
        ),
        VulkanSignalResource::StateView {
            component_id,
            state_id,
            static_bytes,
            bytes_per_activation,
        } => (component_id, state_id, static_bytes, bytes_per_activation, true),
        _ => unreachable!("caller only routes state resources"),
    };
    let resident = state_index
        .get(&(component_id.as_str(), state_id.as_str()))
        .ok_or_else(|| {
            VulkanDescriptorResourcePlanError(format!(
                "{} descriptor {} state {}.{} is not resident",
                command.kernel_id, descriptor.binding, component_id, state_id
            ))
        })?;
    if *static_bytes != resident.static_bytes
        || *bytes_per_activation != resident.bytes_per_activation
    {
        return Err(VulkanDescriptorResourcePlanError(format!(
            "{} descriptor {} state {}.{} byte shape {:?}/{:?} does not match resident {:?}/{:?}",
            command.kernel_id,
            descriptor.binding,
            component_id,
            state_id,
            static_bytes,
            bytes_per_activation,
            resident.static_bytes,
            resident.bytes_per_activation
        )));
    }
    let byte_capacity =
        descriptor_state_byte_capacity(resident, dynamic_state_capacity_activations)?;
    if state_view {
        Ok(VulkanDescriptorResourceAddress::StateView {
            component_id: component_id.to_string(),
            state_id: state_id.to_string(),
            state_type: resident.state_type.clone(),
            byte_capacity,
            static_bytes: resident.static_bytes,
            bytes_per_activation: resident.bytes_per_activation,
        })
    } else {
        Ok(VulkanDescriptorResourceAddress::StateBuffer {
            component_id: component_id.to_string(),
            state_id: state_id.to_string(),
            state_type: resident.state_type.clone(),
            byte_capacity,
            static_bytes: resident.static_bytes,
            bytes_per_activation: resident.bytes_per_activation,
        })
    }
}

fn resolve_state_descriptor_resource(
    command: &VulkanKernelDispatchCommand,
    descriptor: &VulkanKernelDescriptorBinding,
    component_id: &str,
    binding: &VulkanStateBinding,
    state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
    dynamic_state_capacity_activations: usize,
    state_view: bool,
) -> Result<VulkanDescriptorResourceAddress, VulkanDescriptorResourcePlanError> {
    let resident = state_index
        .get(&(component_id, binding.state_id.as_str()))
        .ok_or_else(|| {
            VulkanDescriptorResourcePlanError(format!(
                "{} descriptor {} state {}.{} is not resident",
                command.kernel_id, descriptor.binding, component_id, binding.state_id
            ))
        })?;
    if binding.state_type != resident.state_type
        || binding.static_bytes != resident.static_bytes
        || binding.bytes_per_activation != resident.bytes_per_activation
    {
        return Err(VulkanDescriptorResourcePlanError(format!(
            "{} descriptor {} state {}.{} binding does not match resident allocation",
            command.kernel_id, descriptor.binding, component_id, binding.state_id
        )));
    }
    let byte_capacity =
        descriptor_state_byte_capacity(resident, dynamic_state_capacity_activations)?;
    if state_view {
        Ok(VulkanDescriptorResourceAddress::StateView {
            component_id: component_id.to_string(),
            state_id: binding.state_id.clone(),
            state_type: binding.state_type.clone(),
            byte_capacity,
            static_bytes: binding.static_bytes,
            bytes_per_activation: binding.bytes_per_activation,
        })
    } else {
        Ok(VulkanDescriptorResourceAddress::StateBuffer {
            component_id: component_id.to_string(),
            state_id: binding.state_id.clone(),
            state_type: binding.state_type.clone(),
            byte_capacity,
            static_bytes: binding.static_bytes,
            bytes_per_activation: binding.bytes_per_activation,
        })
    }
}

fn descriptor_state_byte_capacity(
    state: &VulkanResidentStateBuffer,
    dynamic_state_capacity_activations: usize,
) -> Result<usize, VulkanDescriptorResourcePlanError> {
    VulkanTransientStateBufferLayout::for_state(state, dynamic_state_capacity_activations)
        .map(|layout| layout.byte_capacity)
        .map_err(|error| VulkanDescriptorResourcePlanError(error.to_string()))
}
