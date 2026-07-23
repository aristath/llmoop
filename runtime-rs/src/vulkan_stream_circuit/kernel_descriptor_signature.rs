#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanKernelDescriptorSlotSignature {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub resource_class: VulkanKernelDescriptorResourceClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_capacity: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanKernelDescriptorResourceClass {
    SignalBuffer,
    ParameterBuffer,
    StateBuffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelDispatchRef {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub circuit_index: usize,
    pub node_index: usize,
    pub node_id: String,
}

impl VulkanKernelDispatchRef {
    fn from_command(command: &VulkanKernelDispatchCommand) -> Self {
        Self {
            dispatch_index: command.dispatch_index,
            kernel_id: command.kernel_id.clone(),
            pedal_id: command.pedal_id.clone(),
            circuit_index: command.circuit_index,
            node_index: command.node_index,
            node_id: command.node_id.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct VulkanReusableKernelKey {
    op: String,
    specialization: String,
    descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    push_constants: Vec<VulkanKernelScalarBinding>,
    uses_stream_tick: bool,
}

impl VulkanReusableKernelKey {
    fn from_command(command: &VulkanKernelDispatchCommand) -> Self {
        Self {
            op: command.op.clone(),
            specialization: command.specialization.clone(),
            descriptor_signature: command
                .descriptor_bindings
                .iter()
                .map(VulkanKernelDescriptorSlotSignature::from_binding)
                .collect(),
            push_constants: command.push_constants.clone(),
            uses_stream_tick: command.uses_stream_tick,
        }
    }

    fn family_id(&self) -> String {
        let contract = serde_json::to_vec(self)
            .expect("reusable Vulkan kernel contract contains only serializable values");
        let digest = Sha256::digest(contract);
        let contract_id = digest
            .iter()
            .take(12)
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!("{}.{}", self.op, contract_id)
    }
}

impl VulkanKernelDescriptorSlotSignature {
    fn from_binding(binding: &VulkanKernelDescriptorBinding) -> Self {
        Self {
            binding: binding.binding,
            usage: binding.usage.clone(),
            resource_class: VulkanKernelDescriptorResourceClass::from_resource(&binding.resource),
            byte_capacity: descriptor_resource_byte_capacity(&binding.resource),
            shape: descriptor_resource_shape(&binding.resource),
        }
    }
}

impl VulkanKernelDescriptorResourceClass {
    fn from_resource(resource: &VulkanKernelDescriptorResource) -> Self {
        match resource {
            VulkanKernelDescriptorResource::Signal(_) => Self::SignalBuffer,
            VulkanKernelDescriptorResource::Parameter(_) => Self::ParameterBuffer,
            VulkanKernelDescriptorResource::State { .. } => Self::StateBuffer,
        }
    }
}

fn descriptor_resource_byte_capacity(resource: &VulkanKernelDescriptorResource) -> Option<usize> {
    match resource {
        VulkanKernelDescriptorResource::Signal(signal) => match &signal.resource {
            VulkanSignalResource::BoundaryInput | VulkanSignalResource::BoundaryOutput => None,
            VulkanSignalResource::StateBuffer {
                static_bytes,
                bytes_per_activation,
                ..
            }
            | VulkanSignalResource::StateView {
                static_bytes,
                bytes_per_activation,
                ..
            } => match (static_bytes, bytes_per_activation) {
                (Some(static_bytes), Some(bytes_per_activation)) => {
                    static_bytes.checked_add(*bytes_per_activation)
                }
                (Some(static_bytes), None) => Some(*static_bytes),
                (None, Some(bytes_per_activation)) => Some(*bytes_per_activation),
                (None, None) => None,
            },
            VulkanSignalResource::ActivationSlot { bytes, .. } => *bytes,
        },
        VulkanKernelDescriptorResource::Parameter(parameter) => parameter.byte_count,
        VulkanKernelDescriptorResource::State { binding, .. } => {
            match (binding.static_bytes, binding.bytes_per_activation) {
                (Some(static_bytes), Some(bytes_per_activation)) => {
                    static_bytes.checked_add(bytes_per_activation)
                }
                (Some(static_bytes), None) => Some(static_bytes),
                (None, Some(bytes_per_activation)) => Some(bytes_per_activation),
                (None, None) => None,
            }
        }
    }
}

fn descriptor_resource_shape(resource: &VulkanKernelDescriptorResource) -> Option<Vec<usize>> {
    match resource {
        VulkanKernelDescriptorResource::Parameter(parameter) => parameter.shape.clone(),
        VulkanKernelDescriptorResource::Signal(_)
        | VulkanKernelDescriptorResource::State { .. } => None,
    }
}

