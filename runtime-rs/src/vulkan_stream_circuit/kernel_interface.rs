#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelInterfacePlan {
    pub backend_id: String,
    pub circuits: Vec<VulkanCircuitKernelInterface>,
}

impl VulkanKernelInterfacePlan {
    pub fn from_binding_plan(binding_plan: &VulkanStreamCircuitBindingPlan) -> Self {
        Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            circuits: binding_plan
                .circuits
                .iter()
                .map(VulkanCircuitKernelInterface::from_binding_plan)
                .collect(),
        }
    }

    pub fn total_kernel_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.kernels.len())
            .sum()
    }

    pub fn kernel(&self, component_id: &str, node_id: &str) -> Option<&VulkanKernelInterface> {
        self.circuits
            .iter()
            .find(|circuit| circuit.component_id == component_id)
            .and_then(|circuit| circuit.kernel(node_id))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCircuitKernelInterface {
    pub component_id: String,
    pub circuit_id: String,
    pub kernels: Vec<VulkanKernelInterface>,
}

impl VulkanCircuitKernelInterface {
    fn from_binding_plan(circuit: &VulkanCircuitBindingPlan) -> Self {
        Self {
            component_id: circuit.component_id.clone(),
            circuit_id: circuit.circuit_id.clone(),
            kernels: circuit
                .nodes
                .iter()
                .map(|node| VulkanKernelInterface::from_node_binding(&circuit.component_id, node))
                .collect(),
        }
    }

    pub fn kernel(&self, node_id: &str) -> Option<&VulkanKernelInterface> {
        self.kernels.iter().find(|kernel| kernel.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelInterface {
    pub kernel_id: String,
    pub component_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub specialization: String,
    pub inputs: Vec<VulkanSignalBinding>,
    pub outputs: Vec<VulkanSignalBinding>,
    pub parameters: Vec<VulkanParameterBinding>,
    pub state_reads: Vec<VulkanStateBinding>,
    pub state_writes: Vec<VulkanStateBinding>,
    pub state_views: Vec<VulkanSignalBinding>,
    pub stream_metadata: VulkanKernelStreamMetadata,
}

impl VulkanKernelInterface {
    fn from_node_binding(component_id: &str, node: &VulkanNodeBinding) -> Self {
        let state_views = node
            .inputs
            .iter()
            .chain(&node.outputs)
            .filter(|binding| matches!(binding.resource, VulkanSignalResource::StateView { .. }))
            .cloned()
            .collect();

        Self {
            kernel_id: format!("{}.{}", component_id, node.node_id),
            component_id: component_id.to_string(),
            node_index: node.node_index,
            node_id: node.node_id.clone(),
            op: node.op.clone(),
            specialization: node.specialization.clone(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            parameters: node.parameters.clone(),
            state_reads: node.state_reads.clone(),
            state_writes: node.state_writes.clone(),
            state_views,
            stream_metadata: VulkanKernelStreamMetadata::for_op(&node.op),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelStreamMetadata {
    pub stream_tick: VulkanKernelScalarBinding,
    pub control_flags: VulkanKernelScalarBinding,
    pub dynamic_state_capacity_activations: VulkanKernelScalarBinding,
    pub uses_stream_tick: bool,
    push_constants: Vec<VulkanKernelScalarBinding>,
}

impl VulkanKernelStreamMetadata {
    fn for_op(op: &str) -> Self {
        Self {
            stream_tick: VulkanKernelScalarBinding::push_constant("stream_tick", "u64"),
            control_flags: VulkanKernelScalarBinding::push_constant("control_flags", "u32"),
            dynamic_state_capacity_activations: VulkanKernelScalarBinding::push_constant(
                "dynamic_state_capacity_activations",
                "u32",
            ),
            uses_stream_tick: matches!(
                op,
                "rotary_position_embedding"
                    | "parallel_head_norm_rope_2way"
                    | "append_state_update"
                    | "scaled_dot_product_attention"
                    | "append_scaled_dot_product_attention"
                    | "per_layer_embedding"
                    | "rg_lru_step"
            ),
            push_constants: if matches!(op, "sparse_moe_gate_up" | "sparse_moe_down") {
                vec![VulkanKernelScalarBinding::push_constant(
                    "expert_start",
                    "u32",
                )]
            } else {
                Vec::new()
            },
        }
    }

    pub fn push_constants(&self) -> Vec<VulkanKernelScalarBinding> {
        self.push_constants.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanKernelScalarBinding {
    pub name: String,
    pub scalar_type: String,
    pub source: VulkanKernelScalarSource,
}

impl VulkanKernelScalarBinding {
    fn push_constant(name: impl Into<String>, scalar_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            scalar_type: scalar_type.into(),
            source: VulkanKernelScalarSource::PushConstant,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanKernelScalarSource {
    PushConstant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelDispatchPlan {
    pub backend_id: String,
    pub commands: Vec<VulkanKernelDispatchCommand>,
}

impl VulkanKernelDispatchPlan {
    pub fn from_binding_plan(binding_plan: &VulkanStreamCircuitBindingPlan) -> Self {
        Self::from_kernel_interfaces(&VulkanKernelInterfacePlan::from_binding_plan(binding_plan))
    }

    pub fn from_kernel_interfaces(interface_plan: &VulkanKernelInterfacePlan) -> Self {
        let mut commands = Vec::with_capacity(interface_plan.total_kernel_count());
        for (circuit_index, circuit) in interface_plan.circuits.iter().enumerate() {
            for kernel in &circuit.kernels {
                commands.push(VulkanKernelDispatchCommand::from_kernel(
                    commands.len(),
                    circuit_index,
                    &circuit.circuit_id,
                    kernel,
                ));
            }
        }

        Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            commands,
        }
    }

    pub fn total_dispatch_count(&self) -> usize {
        self.commands.len()
    }

    pub fn command(&self, component_id: &str, node_id: &str) -> Option<&VulkanKernelDispatchCommand> {
        self.commands
            .iter()
            .find(|command| command.component_id == component_id && command.node_id == node_id)
    }

    pub fn op_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for command in &self.commands {
            *counts.entry(command.op.clone()).or_insert(0) += 1;
        }
        counts
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelDispatchCommand {
    pub dispatch_index: usize,
    pub circuit_index: usize,
    pub kernel_id: String,
    pub component_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub specialization: String,
    pub descriptor_bindings: Vec<VulkanKernelDescriptorBinding>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

impl VulkanKernelDispatchCommand {
    fn from_kernel(
        dispatch_index: usize,
        circuit_index: usize,
        circuit_id: &str,
        kernel: &VulkanKernelInterface,
    ) -> Self {
        Self {
            dispatch_index,
            circuit_index,
            kernel_id: kernel.kernel_id.clone(),
            component_id: kernel.component_id.clone(),
            circuit_id: circuit_id.to_string(),
            node_index: kernel.node_index,
            node_id: kernel.node_id.clone(),
            op: kernel.op.clone(),
            specialization: kernel.specialization.clone(),
            descriptor_bindings: descriptor_bindings_for_kernel(kernel),
            push_constants: kernel.stream_metadata.push_constants(),
            uses_stream_tick: kernel.stream_metadata.uses_stream_tick,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanKernelDescriptorBinding {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub name: String,
    pub resource: VulkanKernelDescriptorResource,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanKernelDescriptorUsage {
    InputSignal,
    OutputSignal,
    Parameter,
    StateRead,
    StateWrite,
    StateView,
}
