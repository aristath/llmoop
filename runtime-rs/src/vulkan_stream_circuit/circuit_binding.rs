#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanStreamCircuitBindingPlan {
    pub backend_id: String,
    pub circuits: Vec<VulkanCircuitBindingPlan>,
}

impl VulkanStreamCircuitBindingPlan {
    pub fn from_plans(
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        resident_plan: &VulkanStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanBindingPlanError> {
        Self::from_plans_with_hosted_components(execution_plan, resource_plan, resident_plan, None)
    }

    pub fn from_placed_resident_plan(
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanBindingPlanError> {
        let hosted_components = placed_resident_plan
            .hosted_component_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        Self::from_plans_with_hosted_components(
            execution_plan,
            resource_plan,
            &placed_resident_plan.resident_plan,
            Some(&hosted_components),
        )
    }

    fn from_plans_with_hosted_components(
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        resident_plan: &VulkanStreamCircuitResidentPlan,
        hosted_components: Option<&BTreeSet<String>>,
    ) -> Result<Self, VulkanBindingPlanError> {
        let hosts_component = |component_id: &str| {
            hosted_components
                .map(|components| components.contains(component_id))
                .unwrap_or(true)
        };
        let hosted_circuit_count = execution_plan
            .circuits
            .iter()
            .filter(|circuit| hosts_component(&circuit.component_id))
            .count();

        if hosted_components.is_none()
            && (execution_plan.circuits.len() != resident_plan.circuit_count
                || resource_plan.circuit_count != resident_plan.circuit_count)
        {
            return Err(VulkanBindingPlanError(format!(
                "execution/resource/resident circuit counts do not match: {}/{}/{}",
                execution_plan.circuits.len(),
                resource_plan.circuit_count,
                resident_plan.circuit_count
            )));
        }
        if hosted_circuit_count != resident_plan.circuit_count {
            return Err(VulkanBindingPlanError(format!(
                "hosted execution/resident circuit counts do not match: {}/{}",
                hosted_circuit_count, resident_plan.circuit_count
            )));
        }

        let parameter_bindings =
            parameter_binding_index(resource_plan, resident_plan, hosted_components)?;
        let state_bindings = state_binding_index(resource_plan, resident_plan)?;
        let activation_bindings = activation_binding_index(resident_plan)?;

        let circuits = execution_plan
            .circuits
            .iter()
            .filter(|circuit| hosts_component(&circuit.component_id))
            .map(|circuit| {
                bind_circuit(
                    circuit,
                    &parameter_bindings,
                    &state_bindings,
                    &activation_bindings,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            circuits,
        })
    }

    pub fn total_node_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.nodes.len())
            .sum()
    }

    pub fn circuit(&self, component_id: &str) -> Option<&VulkanCircuitBindingPlan> {
        self.circuits
            .iter()
            .find(|circuit| circuit.component_id == component_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedStreamCircuitPlan {
    pub backend_id: String,
    pub device_id: String,
    pub placed_resident_plan: VulkanPlacedStreamCircuitResidentPlan,
    pub binding_plan: VulkanStreamCircuitBindingPlan,
    pub kernel_interface_plan: VulkanKernelInterfacePlan,
    pub dispatch_plan: VulkanKernelDispatchPlan,
    pub reusable_kernel_plan: VulkanReusableKernelPlan,
}

impl VulkanPlacedStreamCircuitPlan {
    pub fn from_plans(
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        placed_resident_plan: VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanBindingPlanError> {
        let binding_plan = VulkanStreamCircuitBindingPlan::from_placed_resident_plan(
            execution_plan,
            resource_plan,
            &placed_resident_plan,
        )?;
        let kernel_interface_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);
        let dispatch_plan =
            VulkanKernelDispatchPlan::from_kernel_interfaces(&kernel_interface_plan);
        let reusable_kernel_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_resident_plan.device_id.clone(),
            placed_resident_plan,
            binding_plan,
            kernel_interface_plan,
            dispatch_plan,
            reusable_kernel_plan,
        })
    }

    pub fn prepared_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
        dynamic_state_capacity_activations: usize,
    ) -> Result<VulkanPreparedDispatchPlan, VulkanPreparedDispatchPlanError> {
        let descriptor_plan = VulkanDescriptorResourcePlan::from_plans(
            &self.dispatch_plan,
            &self.placed_resident_plan.resident_plan,
            dynamic_state_capacity_activations,
        )
        .map_err(VulkanPreparedDispatchPlanError::DescriptorResource)?;
        VulkanPreparedDispatchPlan::from_plans(
            &self.dispatch_plan,
            &self.reusable_kernel_plan,
            &descriptor_plan,
            manifest,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCircuitBindingPlan {
    pub component_id: String,
    pub circuit_id: String,
    pub input_ports: Vec<PlannedPort>,
    pub output_ports: Vec<PlannedPort>,
    pub nodes: Vec<VulkanNodeBinding>,
}

impl VulkanCircuitBindingPlan {
    pub fn node(&self, node_id: &str) -> Option<&VulkanNodeBinding> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanNodeBinding {
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub specialization: String,
    pub inputs: Vec<VulkanSignalBinding>,
    pub outputs: Vec<VulkanSignalBinding>,
    pub parameters: Vec<VulkanParameterBinding>,
    pub state_reads: Vec<VulkanStateBinding>,
    pub state_writes: Vec<VulkanStateBinding>,
}

impl VulkanNodeBinding {
    pub fn input(&self, signal_id: &str) -> Option<&VulkanSignalBinding> {
        self.inputs
            .iter()
            .find(|binding| binding.signal_id == signal_id)
    }

    pub fn output(&self, signal_id: &str) -> Option<&VulkanSignalBinding> {
        self.outputs
            .iter()
            .find(|binding| binding.signal_id == signal_id)
    }

    pub fn parameter(&self, param_id: &str) -> Option<&VulkanParameterBinding> {
        self.parameters
            .iter()
            .find(|binding| binding.param_id == param_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSignalBinding {
    pub signal_id: String,
    pub resource: VulkanSignalResource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanSignalResource {
    BoundaryInput,
    BoundaryOutput,
    StateBuffer {
        component_id: String,
        state_id: String,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    StateView {
        component_id: String,
        state_id: String,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    ActivationSlot {
        component_id: String,
        slot: usize,
        bytes: Option<usize>,
        signal_bytes: Option<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanParameterBinding {
    pub param_id: String,
    pub tensor: String,
    pub byte_count: Option<usize>,
    pub shape: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanStateBinding {
    pub component_id: String,
    pub state_id: String,
    pub state_type: String,
    pub static_bytes: Option<usize>,
    pub bytes_per_activation: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBindingPlanError(pub String);

impl Display for VulkanBindingPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanBindingPlanError {}

