use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::stream_plan::{
    CircuitActivationPlan, PlannedNode, SignalProducer, SignalStorage, StreamCircuitExecutionPlan,
    StreamCircuitResourcePlan, TensorIndex,
};
use crate::vulkan_compute::{VulkanComputeDevice, VulkanError, VulkanResidentBuffer};

pub const VULKAN_STREAM_CIRCUIT_BACKEND_ID: &str = "vulkan_stream_circuit_ir";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanStreamCircuitResidentPlan {
    pub backend_id: String,
    pub circuit_count: usize,
    pub permanent_parameters: Vec<VulkanResidentParameter>,
    pub permanent_parameter_bytes: Option<usize>,
    pub stream_state_buffers: Vec<VulkanResidentStateBuffer>,
    pub state_view_signal_count: usize,
    pub activation_banks: Vec<VulkanResidentActivationBank>,
    pub per_stream_static_state_elements: usize,
    pub per_stream_dynamic_state_elements_per_activation: usize,
    pub per_stream_activation_slot_elements: Option<usize>,
    pub per_stream_static_state_bytes: Option<usize>,
    pub per_stream_dynamic_state_bytes_per_activation: Option<usize>,
    pub per_stream_activation_slot_bytes: Option<usize>,
    pub unresolved_parameter_tensors: Vec<String>,
    pub unresolved_activation_slots: Vec<String>,
}

impl VulkanStreamCircuitResidentPlan {
    pub fn from_resource_plan(
        resource_plan: &StreamCircuitResourcePlan,
        tensor_index: Option<&TensorIndex>,
        activation_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanResidentPlanError> {
        let mut permanent_parameters = Vec::with_capacity(resource_plan.parameters.len());
        let mut permanent_parameter_bytes = Some(0usize);
        let mut unresolved_parameter_tensors = Vec::new();

        for parameter in &resource_plan.parameters {
            let metadata = tensor_index.and_then(|index| index.tensors.get(&parameter.tensor));
            let byte_count = metadata.and_then(|metadata| metadata.byte_count);
            match (permanent_parameter_bytes, byte_count) {
                (Some(total), Some(bytes)) => {
                    permanent_parameter_bytes =
                        Some(checked_add(total, bytes, "permanent parameter bytes")?);
                }
                _ => {
                    permanent_parameter_bytes = None;
                    unresolved_parameter_tensors.push(parameter.tensor.clone());
                }
            }

            permanent_parameters.push(VulkanResidentParameter {
                tensor: parameter.tensor.clone(),
                dtype: metadata.map(|metadata| metadata.dtype.clone()),
                shape: metadata.map(|metadata| metadata.shape.clone()),
                byte_count,
                use_count: parameter.uses.len(),
            });
        }

        let mut stream_state_buffers = Vec::with_capacity(resource_plan.state_allocations.len());
        let mut per_stream_static_state_elements = 0usize;
        let mut per_stream_dynamic_state_elements_per_activation = 0usize;

        for state in &resource_plan.state_allocations {
            let static_elements = state.shape.as_ref().and_then(|shape| product(shape));
            if let Some(elements) = static_elements {
                per_stream_static_state_elements = checked_add(
                    per_stream_static_state_elements,
                    elements,
                    "per-stream static state elements",
                )?;
            }
            if let Some(elements) = state.elements_per_activation {
                per_stream_dynamic_state_elements_per_activation = checked_add(
                    per_stream_dynamic_state_elements_per_activation,
                    elements,
                    "per-stream dynamic state elements per activation",
                )?;
            }

            stream_state_buffers.push(VulkanResidentStateBuffer {
                pedal_id: state.pedal_id.clone(),
                state_id: state.state_id.clone(),
                state_type: state.state_type.clone(),
                layout: state.layout.clone(),
                static_elements,
                elements_per_activation: state.elements_per_activation,
                static_bytes: optional_mul(static_elements, activation_element_bytes)?,
                bytes_per_activation: optional_mul(
                    state.elements_per_activation,
                    activation_element_bytes,
                )?,
            });
        }

        let mut activation_banks = Vec::with_capacity(resource_plan.activation_banks.len());
        let mut per_stream_activation_slot_elements = Some(0usize);
        let mut unresolved_activation_slots = Vec::new();

        for bank in &resource_plan.activation_banks {
            let mut slots = Vec::with_capacity(bank.slots.len());
            for slot in &bank.slots {
                match (per_stream_activation_slot_elements, slot.max_elements) {
                    (Some(total), Some(elements)) => {
                        per_stream_activation_slot_elements = Some(checked_add(
                            total,
                            elements,
                            "per-stream activation slot elements",
                        )?);
                    }
                    _ => {
                        per_stream_activation_slot_elements = None;
                        unresolved_activation_slots
                            .push(format!("{}.slot_{}", bank.pedal_id, slot.slot));
                    }
                }

                slots.push(VulkanResidentActivationSlot {
                    slot: slot.slot,
                    signal_ids: slot.signal_ids.clone(),
                    max_elements: slot.max_elements,
                    bytes: optional_mul(slot.max_elements, activation_element_bytes)?,
                });
            }

            activation_banks.push(VulkanResidentActivationBank {
                pedal_id: bank.pedal_id.clone(),
                circuit_id: bank.circuit_id.clone(),
                slot_count: bank.slot_count,
                slots,
            });
        }

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            circuit_count: resource_plan.circuit_count,
            permanent_parameters,
            permanent_parameter_bytes,
            stream_state_buffers,
            state_view_signal_count: resource_plan.state_view_signal_count,
            activation_banks,
            per_stream_static_state_elements,
            per_stream_dynamic_state_elements_per_activation,
            per_stream_activation_slot_elements,
            per_stream_static_state_bytes: optional_mul(
                Some(per_stream_static_state_elements),
                activation_element_bytes,
            )?,
            per_stream_dynamic_state_bytes_per_activation: optional_mul(
                Some(per_stream_dynamic_state_elements_per_activation),
                activation_element_bytes,
            )?,
            per_stream_activation_slot_bytes: optional_mul(
                per_stream_activation_slot_elements,
                activation_element_bytes,
            )?,
            unresolved_parameter_tensors,
            unresolved_activation_slots,
        })
    }

    pub fn activation_bank(&self, pedal_id: &str) -> Option<&VulkanResidentActivationBank> {
        self.activation_banks
            .iter()
            .find(|bank| bank.pedal_id == pedal_id)
    }

    pub fn allocate_stream_buffers(
        &self,
        device: &VulkanComputeDevice,
        dynamic_state_capacity_activations: usize,
    ) -> Result<VulkanStreamCircuitStreamBuffers, VulkanError> {
        let mut state_buffers = Vec::with_capacity(self.stream_state_buffers.len());
        let mut activation_slot_buffers = Vec::new();
        let mut total_byte_capacity = 0usize;

        for state in &self.stream_state_buffers {
            let byte_capacity =
                stream_state_byte_capacity(state, dynamic_state_capacity_activations)?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "stream state buffer allocation",
            )?;
            state_buffers.push(VulkanStreamStateBufferAllocation {
                pedal_id: state.pedal_id.clone(),
                state_id: state.state_id.clone(),
                state_type: state.state_type.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        for bank in &self.activation_banks {
            for slot in &bank.slots {
                let byte_capacity = slot.bytes.ok_or_else(|| {
                    VulkanError(format!(
                        "{} activation slot {} has unknown byte size",
                        bank.pedal_id, slot.slot
                    ))
                })?;
                total_byte_capacity = checked_add_bytes(
                    total_byte_capacity,
                    byte_capacity,
                    "activation slot buffer allocation",
                )?;
                activation_slot_buffers.push(VulkanActivationSlotBufferAllocation {
                    pedal_id: bank.pedal_id.clone(),
                    circuit_id: bank.circuit_id.clone(),
                    slot: slot.slot,
                    signal_ids: slot.signal_ids.clone(),
                    byte_capacity,
                    buffer: device.create_resident_buffer(byte_capacity)?,
                });
            }
        }

        Ok(VulkanStreamCircuitStreamBuffers {
            dynamic_state_capacity_activations,
            state_buffers,
            activation_slot_buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentParameter {
    pub tensor: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub byte_count: Option<usize>,
    pub use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentStateBuffer {
    pub pedal_id: String,
    pub state_id: String,
    pub state_type: String,
    pub layout: Option<String>,
    pub static_elements: Option<usize>,
    pub elements_per_activation: Option<usize>,
    pub static_bytes: Option<usize>,
    pub bytes_per_activation: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentActivationBank {
    pub pedal_id: String,
    pub circuit_id: String,
    pub slot_count: usize,
    pub slots: Vec<VulkanResidentActivationSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentActivationSlot {
    pub slot: usize,
    pub signal_ids: Vec<String>,
    pub max_elements: Option<usize>,
    pub bytes: Option<usize>,
}

pub struct VulkanStreamCircuitStreamBuffers {
    pub dynamic_state_capacity_activations: usize,
    pub state_buffers: Vec<VulkanStreamStateBufferAllocation>,
    pub activation_slot_buffers: Vec<VulkanActivationSlotBufferAllocation>,
    pub total_byte_capacity: usize,
}

pub struct VulkanStreamStateBufferAllocation {
    pub pedal_id: String,
    pub state_id: String,
    pub state_type: String,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

pub struct VulkanActivationSlotBufferAllocation {
    pub pedal_id: String,
    pub circuit_id: String,
    pub slot: usize,
    pub signal_ids: Vec<String>,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

impl VulkanStreamCircuitStreamBuffers {
    pub fn state_buffer(
        &self,
        pedal_id: &str,
        state_id: &str,
    ) -> Option<&VulkanStreamStateBufferAllocation> {
        self.state_buffers
            .iter()
            .find(|buffer| buffer.pedal_id == pedal_id && buffer.state_id == state_id)
    }

    pub fn activation_slot_buffer(
        &self,
        pedal_id: &str,
        slot: usize,
    ) -> Option<&VulkanActivationSlotBufferAllocation> {
        self.activation_slot_buffers
            .iter()
            .find(|buffer| buffer.pedal_id == pedal_id && buffer.slot == slot)
    }
}

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
        if execution_plan.circuits.len() != resident_plan.circuit_count
            || resource_plan.circuit_count != resident_plan.circuit_count
        {
            return Err(VulkanBindingPlanError(format!(
                "execution/resource/resident circuit counts do not match: {}/{}/{}",
                execution_plan.circuits.len(),
                resource_plan.circuit_count,
                resident_plan.circuit_count
            )));
        }

        let parameter_bindings = parameter_binding_index(resource_plan, resident_plan)?;
        let state_bindings = state_binding_index(resident_plan)?;
        let activation_bindings = activation_binding_index(resident_plan)?;

        let circuits = execution_plan
            .circuits
            .iter()
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

    pub fn circuit(&self, pedal_id: &str) -> Option<&VulkanCircuitBindingPlan> {
        self.circuits
            .iter()
            .find(|circuit| circuit.pedal_id == pedal_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCircuitBindingPlan {
    pub pedal_id: String,
    pub circuit_id: String,
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
        pedal_id: String,
        state_id: String,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    StateView {
        pedal_id: String,
        state_id: String,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    ActivationSlot {
        pedal_id: String,
        slot: usize,
        bytes: Option<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanParameterBinding {
    pub param_id: String,
    pub tensor: String,
    pub byte_count: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanStateBinding {
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

pub struct VulkanMountedStreamCircuit {
    pub resident_plan: VulkanStreamCircuitResidentPlan,
    pub binding_plan: VulkanStreamCircuitBindingPlan,
    pub kernel_interface_plan: VulkanKernelInterfacePlan,
    pub dispatch_plan: VulkanKernelDispatchPlan,
    pub reusable_kernel_plan: VulkanReusableKernelPlan,
    pub buffers: VulkanStreamCircuitStreamBuffers,
}

impl VulkanMountedStreamCircuit {
    pub fn from_plans(
        device: &VulkanComputeDevice,
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        resident_plan: VulkanStreamCircuitResidentPlan,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            execution_plan,
            resource_plan,
            &resident_plan,
        )?;
        let kernel_interface_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);
        let dispatch_plan =
            VulkanKernelDispatchPlan::from_kernel_interfaces(&kernel_interface_plan);
        let reusable_kernel_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
        let buffers =
            resident_plan.allocate_stream_buffers(device, dynamic_state_capacity_activations)?;
        Ok(Self {
            resident_plan,
            binding_plan,
            kernel_interface_plan,
            dispatch_plan,
            reusable_kernel_plan,
            buffers,
        })
    }

    pub fn can_execute(&self) -> bool {
        false
    }

    pub fn reusable_kernel_coverage_report<I, S>(
        &self,
        available_family_ids: I,
    ) -> VulkanReusableKernelCoverageReport
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.reusable_kernel_plan
            .coverage_report(available_family_ids)
    }
}

#[derive(Debug)]
pub enum VulkanStreamCircuitMountError {
    Binding(VulkanBindingPlanError),
    Vulkan(VulkanError),
}

impl Display for VulkanStreamCircuitMountError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(error) => Display::fmt(error, f),
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanStreamCircuitMountError {}

impl From<VulkanBindingPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanBindingPlanError) -> Self {
        Self::Binding(error)
    }
}

impl From<VulkanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

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

    pub fn kernel(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanKernelInterface> {
        self.circuits
            .iter()
            .find(|circuit| circuit.pedal_id == pedal_id)
            .and_then(|circuit| circuit.kernel(node_id))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCircuitKernelInterface {
    pub pedal_id: String,
    pub circuit_id: String,
    pub kernels: Vec<VulkanKernelInterface>,
}

impl VulkanCircuitKernelInterface {
    fn from_binding_plan(circuit: &VulkanCircuitBindingPlan) -> Self {
        Self {
            pedal_id: circuit.pedal_id.clone(),
            circuit_id: circuit.circuit_id.clone(),
            kernels: circuit
                .nodes
                .iter()
                .map(|node| VulkanKernelInterface::from_node_binding(&circuit.pedal_id, node))
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
    pub pedal_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub inputs: Vec<VulkanSignalBinding>,
    pub outputs: Vec<VulkanSignalBinding>,
    pub parameters: Vec<VulkanParameterBinding>,
    pub state_reads: Vec<VulkanStateBinding>,
    pub state_writes: Vec<VulkanStateBinding>,
    pub state_views: Vec<VulkanSignalBinding>,
    pub stream_metadata: VulkanKernelStreamMetadata,
}

impl VulkanKernelInterface {
    fn from_node_binding(pedal_id: &str, node: &VulkanNodeBinding) -> Self {
        let state_views = node
            .inputs
            .iter()
            .chain(&node.outputs)
            .filter(|binding| matches!(binding.resource, VulkanSignalResource::StateView { .. }))
            .cloned()
            .collect();

        Self {
            kernel_id: format!("{}.{}", pedal_id, node.node_id),
            pedal_id: pedal_id.to_string(),
            node_index: node.node_index,
            node_id: node.node_id.clone(),
            op: node.op.clone(),
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
            uses_stream_tick: matches!(op, "rotary_position_embedding" | "append_state_update"),
        }
    }

    pub fn push_constants(&self) -> Vec<VulkanKernelScalarBinding> {
        vec![
            self.stream_tick.clone(),
            self.control_flags.clone(),
            self.dynamic_state_capacity_activations.clone(),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

    pub fn command(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanKernelDispatchCommand> {
        self.commands
            .iter()
            .find(|command| command.pedal_id == pedal_id && command.node_id == node_id)
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
    pub pedal_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
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
            pedal_id: kernel.pedal_id.clone(),
            circuit_id: circuit_id.to_string(),
            node_index: kernel.node_index,
            node_id: kernel.node_id.clone(),
            op: kernel.op.clone(),
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanKernelDescriptorUsage {
    InputSignal,
    OutputSignal,
    Parameter,
    StateRead,
    StateWrite,
    StateView,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanKernelDescriptorResource {
    Signal(VulkanSignalBinding),
    Parameter(VulkanParameterBinding),
    State {
        pedal_id: String,
        binding: VulkanStateBinding,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelPlan {
    pub backend_id: String,
    pub total_command_count: usize,
    pub families: Vec<VulkanReusableKernelFamily>,
}

impl VulkanReusableKernelPlan {
    pub fn from_dispatch_plan(dispatch_plan: &VulkanKernelDispatchPlan) -> Self {
        let mut grouped: BTreeMap<VulkanReusableKernelKey, Vec<VulkanKernelDispatchRef>> =
            BTreeMap::new();

        for command in &dispatch_plan.commands {
            grouped
                .entry(VulkanReusableKernelKey::from_command(command))
                .or_default()
                .push(VulkanKernelDispatchRef::from_command(command));
        }

        let mut op_family_indices = BTreeMap::new();
        let families = grouped
            .into_iter()
            .map(|(key, command_refs)| {
                let op_family_index = op_family_indices.entry(key.op.clone()).or_insert(0usize);
                let family_id = if *op_family_index == 0 {
                    key.op.clone()
                } else {
                    format!("{}.signature_{}", key.op, op_family_index)
                };
                *op_family_index += 1;

                VulkanReusableKernelFamily {
                    family_id,
                    op: key.op,
                    descriptor_signature: key.descriptor_signature,
                    push_constants: key.push_constants,
                    uses_stream_tick: key.uses_stream_tick,
                    command_refs,
                }
            })
            .collect();

        Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            total_command_count: dispatch_plan.total_dispatch_count(),
            families,
        }
    }

    pub fn total_family_count(&self) -> usize {
        self.families.len()
    }

    pub fn reusable_family_count(&self) -> usize {
        self.families
            .iter()
            .filter(|family| family.command_refs.len() > 1)
            .count()
    }

    pub fn family(&self, family_id: &str) -> Option<&VulkanReusableKernelFamily> {
        self.families
            .iter()
            .find(|family| family.family_id == family_id)
    }

    pub fn families_for_op(&self, op: &str) -> Vec<&VulkanReusableKernelFamily> {
        self.families
            .iter()
            .filter(|family| family.op == op)
            .collect()
    }

    pub fn coverage_report<I, S>(
        &self,
        available_family_ids: I,
    ) -> VulkanReusableKernelCoverageReport
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let available_family_ids: BTreeSet<String> = available_family_ids
            .into_iter()
            .map(|id| id.as_ref().to_string())
            .collect();
        let mut families = Vec::with_capacity(self.families.len());
        let mut available_family_count = 0usize;
        let mut covered_command_count = 0usize;

        for family in &self.families {
            let available = available_family_ids.contains(&family.family_id);
            if available {
                available_family_count += 1;
                covered_command_count += family.command_refs.len();
            }
            families.push(VulkanReusableKernelFamilyCoverage {
                family_id: family.family_id.clone(),
                op: family.op.clone(),
                command_count: family.command_refs.len(),
                available,
            });
        }

        VulkanReusableKernelCoverageReport {
            backend_id: self.backend_id.clone(),
            required_family_count: self.families.len(),
            available_family_count,
            missing_family_count: self.families.len() - available_family_count,
            required_command_count: self.total_command_count,
            covered_command_count,
            missing_command_count: self.total_command_count - covered_command_count,
            families,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelFamily {
    pub family_id: String,
    pub op: String,
    pub descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
    pub command_refs: Vec<VulkanKernelDispatchRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelCoverageReport {
    pub backend_id: String,
    pub required_family_count: usize,
    pub available_family_count: usize,
    pub missing_family_count: usize,
    pub required_command_count: usize,
    pub covered_command_count: usize,
    pub missing_command_count: usize,
    pub families: Vec<VulkanReusableKernelFamilyCoverage>,
}

impl VulkanReusableKernelCoverageReport {
    pub fn all_available(&self) -> bool {
        self.missing_family_count == 0
    }

    pub fn missing_families(&self) -> Vec<&VulkanReusableKernelFamilyCoverage> {
        self.families
            .iter()
            .filter(|family| !family.available)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelFamilyCoverage {
    pub family_id: String,
    pub op: String,
    pub command_count: usize,
    pub available: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanKernelDescriptorSlotSignature {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub resource_class: VulkanKernelDescriptorResourceClass,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanReusableKernelKey {
    op: String,
    descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    push_constants: Vec<VulkanKernelScalarBinding>,
    uses_stream_tick: bool,
}

impl VulkanReusableKernelKey {
    fn from_command(command: &VulkanKernelDispatchCommand) -> Self {
        Self {
            op: command.op.clone(),
            descriptor_signature: command
                .descriptor_bindings
                .iter()
                .map(VulkanKernelDescriptorSlotSignature::from_binding)
                .collect(),
            push_constants: command.push_constants.clone(),
            uses_stream_tick: command.uses_stream_tick,
        }
    }
}

impl VulkanKernelDescriptorSlotSignature {
    fn from_binding(binding: &VulkanKernelDescriptorBinding) -> Self {
        Self {
            binding: binding.binding,
            usage: binding.usage.clone(),
            resource_class: VulkanKernelDescriptorResourceClass::from_resource(&binding.resource),
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

fn descriptor_bindings_for_kernel(
    kernel: &VulkanKernelInterface,
) -> Vec<VulkanKernelDescriptorBinding> {
    let mut bindings = Vec::new();

    for input in &kernel.inputs {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::InputSignal,
            input.signal_id.clone(),
            VulkanKernelDescriptorResource::Signal(input.clone()),
        );
    }
    for output in &kernel.outputs {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::OutputSignal,
            output.signal_id.clone(),
            VulkanKernelDescriptorResource::Signal(output.clone()),
        );
    }
    for parameter in &kernel.parameters {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::Parameter,
            parameter.param_id.clone(),
            VulkanKernelDescriptorResource::Parameter(parameter.clone()),
        );
    }
    for state in &kernel.state_reads {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::StateRead,
            state.state_id.clone(),
            VulkanKernelDescriptorResource::State {
                pedal_id: kernel.pedal_id.clone(),
                binding: state.clone(),
            },
        );
    }
    for state in &kernel.state_writes {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::StateWrite,
            state.state_id.clone(),
            VulkanKernelDescriptorResource::State {
                pedal_id: kernel.pedal_id.clone(),
                binding: state.clone(),
            },
        );
    }
    for state_view in &kernel.state_views {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::StateView,
            state_view.signal_id.clone(),
            VulkanKernelDescriptorResource::Signal(state_view.clone()),
        );
    }

    bindings
}

fn push_descriptor_binding(
    bindings: &mut Vec<VulkanKernelDescriptorBinding>,
    usage: VulkanKernelDescriptorUsage,
    name: String,
    resource: VulkanKernelDescriptorResource,
) {
    bindings.push(VulkanKernelDescriptorBinding {
        binding: bindings.len(),
        usage,
        name,
        resource,
    });
}

fn parameter_binding_index(
    resource_plan: &StreamCircuitResourcePlan,
    resident_plan: &VulkanStreamCircuitResidentPlan,
) -> Result<BTreeMap<(String, String), VulkanParameterBinding>, VulkanBindingPlanError> {
    let resident_by_tensor: BTreeMap<_, _> = resident_plan
        .permanent_parameters
        .iter()
        .map(|parameter| (parameter.tensor.as_str(), parameter))
        .collect();
    let mut bindings = BTreeMap::new();

    for parameter in &resource_plan.parameters {
        let resident = resident_by_tensor
            .get(parameter.tensor.as_str())
            .ok_or_else(|| {
                VulkanBindingPlanError(format!(
                    "resident plan has no permanent parameter for tensor {:?}",
                    parameter.tensor
                ))
            })?;
        for use_ref in &parameter.uses {
            let key = (use_ref.pedal_id.clone(), use_ref.param_id.clone());
            let previous = bindings.insert(
                key.clone(),
                VulkanParameterBinding {
                    param_id: use_ref.param_id.clone(),
                    tensor: parameter.tensor.clone(),
                    byte_count: resident.byte_count,
                },
            );
            if previous.is_some() {
                return Err(VulkanBindingPlanError(format!(
                    "duplicate parameter binding for {}.{}",
                    key.0, key.1
                )));
            }
        }
    }

    Ok(bindings)
}

fn state_binding_index(
    resident_plan: &VulkanStreamCircuitResidentPlan,
) -> Result<BTreeMap<(String, String), VulkanStateBinding>, VulkanBindingPlanError> {
    let mut bindings = BTreeMap::new();
    for state in &resident_plan.stream_state_buffers {
        let key = (state.pedal_id.clone(), state.state_id.clone());
        let previous = bindings.insert(
            key.clone(),
            VulkanStateBinding {
                state_id: state.state_id.clone(),
                state_type: state.state_type.clone(),
                static_bytes: state.static_bytes,
                bytes_per_activation: state.bytes_per_activation,
            },
        );
        if previous.is_some() {
            return Err(VulkanBindingPlanError(format!(
                "duplicate state binding for {}.{}",
                key.0, key.1
            )));
        }
    }
    Ok(bindings)
}

fn activation_binding_index(
    resident_plan: &VulkanStreamCircuitResidentPlan,
) -> Result<BTreeMap<(String, String), (usize, Option<usize>)>, VulkanBindingPlanError> {
    let mut bindings = BTreeMap::new();
    for bank in &resident_plan.activation_banks {
        for slot in &bank.slots {
            for signal_id in &slot.signal_ids {
                let key = (bank.pedal_id.clone(), signal_id.clone());
                let previous = bindings.insert(key.clone(), (slot.slot, slot.bytes));
                if previous.is_some() {
                    return Err(VulkanBindingPlanError(format!(
                        "duplicate activation binding for {}.{}",
                        key.0, key.1
                    )));
                }
            }
        }
    }
    Ok(bindings)
}

fn bind_circuit(
    circuit: &CircuitActivationPlan,
    parameter_bindings: &BTreeMap<(String, String), VulkanParameterBinding>,
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
    activation_bindings: &BTreeMap<(String, String), (usize, Option<usize>)>,
) -> Result<VulkanCircuitBindingPlan, VulkanBindingPlanError> {
    let nodes = circuit
        .nodes
        .iter()
        .map(|node| {
            bind_node(
                circuit,
                node,
                parameter_bindings,
                state_bindings,
                activation_bindings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(VulkanCircuitBindingPlan {
        pedal_id: circuit.pedal_id.clone(),
        circuit_id: circuit.circuit_id.clone(),
        nodes,
    })
}

fn bind_node(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    parameter_bindings: &BTreeMap<(String, String), VulkanParameterBinding>,
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
    activation_bindings: &BTreeMap<(String, String), (usize, Option<usize>)>,
) -> Result<VulkanNodeBinding, VulkanBindingPlanError> {
    let inputs = node
        .inputs
        .iter()
        .map(|signal_id| {
            bind_signal(
                circuit,
                node,
                signal_id,
                state_bindings,
                activation_bindings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = node
        .outputs
        .iter()
        .map(|signal_id| {
            bind_signal(
                circuit,
                node,
                signal_id,
                state_bindings,
                activation_bindings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parameters = node
        .params
        .iter()
        .map(|param_id| {
            parameter_bindings
                .get(&(circuit.pedal_id.clone(), param_id.clone()))
                .cloned()
                .ok_or_else(|| {
                    VulkanBindingPlanError(format!(
                        "{} node {} parameter {:?} is not bound",
                        circuit.pedal_id, node.id, param_id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state_reads = bind_state_refs(circuit, node, &node.state_reads, state_bindings)?;
    let state_writes = bind_state_refs(circuit, node, &node.state_writes, state_bindings)?;

    Ok(VulkanNodeBinding {
        node_index: node.index,
        node_id: node.id.clone(),
        op: node.op.clone(),
        inputs,
        outputs,
        parameters,
        state_reads,
        state_writes,
    })
}

fn bind_state_refs(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    state_ids: &[String],
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
) -> Result<Vec<VulkanStateBinding>, VulkanBindingPlanError> {
    state_ids
        .iter()
        .map(|state_id| {
            state_bindings
                .get(&(circuit.pedal_id.clone(), state_id.clone()))
                .cloned()
                .ok_or_else(|| {
                    VulkanBindingPlanError(format!(
                        "{} node {} state {:?} is not bound",
                        circuit.pedal_id, node.id, state_id
                    ))
                })
        })
        .collect()
}

fn bind_signal(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    signal_id: &str,
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
    activation_bindings: &BTreeMap<(String, String), (usize, Option<usize>)>,
) -> Result<VulkanSignalBinding, VulkanBindingPlanError> {
    let signal = circuit.signal(signal_id).ok_or_else(|| {
        VulkanBindingPlanError(format!(
            "{} node {} signal {:?} is not planned",
            circuit.pedal_id, node.id, signal_id
        ))
    })?;

    let resource = if signal.is_boundary_output {
        VulkanSignalResource::BoundaryOutput
    } else {
        match signal.storage {
            SignalStorage::Boundary => VulkanSignalResource::BoundaryInput,
            SignalStorage::State => {
                let state = state_bindings
                    .get(&(circuit.pedal_id.clone(), signal_id.to_string()))
                    .ok_or_else(|| {
                        VulkanBindingPlanError(format!(
                            "{} signal {:?} has no state buffer binding",
                            circuit.pedal_id, signal_id
                        ))
                    })?;
                VulkanSignalResource::StateBuffer {
                    pedal_id: circuit.pedal_id.clone(),
                    state_id: state.state_id.clone(),
                    static_bytes: state.static_bytes,
                    bytes_per_activation: state.bytes_per_activation,
                }
            }
            SignalStorage::Activation => {
                let (slot, bytes) = activation_bindings
                    .get(&(circuit.pedal_id.clone(), signal_id.to_string()))
                    .ok_or_else(|| {
                        VulkanBindingPlanError(format!(
                            "{} signal {:?} has no activation slot binding",
                            circuit.pedal_id, signal_id
                        ))
                    })?;
                VulkanSignalResource::ActivationSlot {
                    pedal_id: circuit.pedal_id.clone(),
                    slot: *slot,
                    bytes: *bytes,
                }
            }
            SignalStorage::StateView => {
                let state_id = state_view_state_id(circuit, signal_id)?;
                let state = state_bindings
                    .get(&(circuit.pedal_id.clone(), state_id.clone()))
                    .ok_or_else(|| {
                        VulkanBindingPlanError(format!(
                            "{} state-view signal {:?} has no state buffer binding for {:?}",
                            circuit.pedal_id, signal_id, state_id
                        ))
                    })?;
                VulkanSignalResource::StateView {
                    pedal_id: circuit.pedal_id.clone(),
                    state_id,
                    static_bytes: state.static_bytes,
                    bytes_per_activation: state.bytes_per_activation,
                }
            }
        }
    };

    Ok(VulkanSignalBinding {
        signal_id: signal_id.to_string(),
        resource,
    })
}

fn state_view_state_id(
    circuit: &CircuitActivationPlan,
    signal_id: &str,
) -> Result<String, VulkanBindingPlanError> {
    let signal = circuit.signal(signal_id).ok_or_else(|| {
        VulkanBindingPlanError(format!(
            "{} signal {:?} is not planned",
            circuit.pedal_id, signal_id
        ))
    })?;
    let SignalProducer::Node { node_id } = &signal.producer else {
        return Err(VulkanBindingPlanError(format!(
            "{} state-view signal {:?} is not produced by a node",
            circuit.pedal_id, signal_id
        )));
    };
    let producer = circuit
        .nodes
        .iter()
        .find(|node| &node.id == node_id)
        .ok_or_else(|| {
            VulkanBindingPlanError(format!(
                "{} state-view signal {:?} producer {:?} is not planned",
                circuit.pedal_id, signal_id, node_id
            ))
        })?;
    producer
        .state_writes
        .first()
        .or_else(|| producer.state_reads.first())
        .cloned()
        .ok_or_else(|| {
            VulkanBindingPlanError(format!(
                "{} state-view signal {:?} producer {:?} does not reference state",
                circuit.pedal_id, signal_id, node_id
            ))
        })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentPlanError(pub String);

impl Display for VulkanResidentPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanResidentPlanError {}

fn optional_mul(
    elements: Option<usize>,
    bytes_per_element: Option<usize>,
) -> Result<Option<usize>, VulkanResidentPlanError> {
    match (elements, bytes_per_element) {
        (Some(elements), Some(bytes_per_element)) => Ok(Some(checked_mul(
            elements,
            bytes_per_element,
            "resident byte count",
        )?)),
        _ => Ok(None),
    }
}

fn stream_state_byte_capacity(
    state: &VulkanResidentStateBuffer,
    dynamic_state_capacity_activations: usize,
) -> Result<usize, VulkanError> {
    let static_bytes = state.static_bytes.unwrap_or(0);
    let dynamic_bytes = match state.bytes_per_activation {
        Some(bytes_per_activation) => {
            if dynamic_state_capacity_activations == 0 {
                return Err(VulkanError(format!(
                    "{}.{} requires non-zero dynamic state capacity",
                    state.pedal_id, state.state_id
                )));
            }
            bytes_per_activation
                .checked_mul(dynamic_state_capacity_activations)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{}.{} dynamic state byte capacity overflowed",
                        state.pedal_id, state.state_id
                    ))
                })?
        }
        None => 0,
    };
    let total = static_bytes.checked_add(dynamic_bytes).ok_or_else(|| {
        VulkanError(format!(
            "{}.{} state byte capacity overflowed",
            state.pedal_id, state.state_id
        ))
    })?;
    if total == 0 {
        return Err(VulkanError(format!(
            "{}.{} has unknown or zero byte capacity",
            state.pedal_id, state.state_id
        )));
    }
    Ok(total)
}

fn checked_add_bytes(left: usize, right: usize, label: &str) -> Result<usize, VulkanError> {
    left.checked_add(right)
        .ok_or_else(|| VulkanError(format!("{label} overflowed")))
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, VulkanResidentPlanError> {
    left.checked_add(right)
        .ok_or_else(|| VulkanResidentPlanError(format!("{label} overflowed")))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, VulkanResidentPlanError> {
    left.checked_mul(right)
        .ok_or_else(|| VulkanResidentPlanError(format!("{label} overflowed")))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::stream_circuit::ResolvedLoweredPedalboard;
    use crate::stream_plan::{StreamCircuitExecutionPlan, StreamCircuitResourcePlan};

    fn lfm2_index_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("lowered")
            .join("lfm2_5_230m")
            .join("pedalboard.circuits.json")
    }

    fn lfm2_tensor_index_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("transpiled")
            .join("lfm2_5_230m")
            .join("tensors.json")
    }

    #[test]
    fn plans_lfm2_vulkan_resident_allocations_from_stream_circuit_resources() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();

        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        assert_eq!(resident_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(resident_plan.circuit_count, 14);
        assert_eq!(resident_plan.permanent_parameters.len(), 130);
        assert_eq!(resident_plan.permanent_parameter_bytes, Some(325_166_592));
        assert!(resident_plan.unresolved_parameter_tensors.is_empty());
        assert_eq!(resident_plan.stream_state_buffers.len(), 14);
        assert_eq!(resident_plan.state_view_signal_count, 20);
        assert_eq!(resident_plan.activation_banks.len(), 14);
        assert_eq!(resident_plan.per_stream_static_state_elements, 8 * 3 * 1024);
        assert_eq!(
            resident_plan.per_stream_dynamic_state_elements_per_activation,
            6 * 1024
        );
        assert_eq!(
            resident_plan.per_stream_activation_slot_elements,
            Some(138_240)
        );
        assert_eq!(resident_plan.per_stream_static_state_bytes, Some(49_152));
        assert_eq!(
            resident_plan.per_stream_dynamic_state_bytes_per_activation,
            Some(12_288)
        );
        assert_eq!(
            resident_plan.per_stream_activation_slot_bytes,
            Some(276_480)
        );
        assert!(resident_plan.unresolved_activation_slots.is_empty());

        let conv_in = resident_plan
            .permanent_parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.layers.0.conv.in_proj.weight")
            .unwrap();
        assert_eq!(conv_in.dtype.as_deref(), Some("BF16"));
        assert_eq!(conv_in.shape, Some(vec![3072, 1024]));
        assert_eq!(conv_in.byte_count, Some(6_291_456));
        assert_eq!(conv_in.use_count, 1);

        let layer_00_bank = resident_plan.activation_bank("layer_00").unwrap();
        assert_eq!(
            layer_00_bank
                .slots
                .iter()
                .map(|slot| slot.bytes)
                .collect::<Vec<_>>(),
            vec![Some(5120), Some(6144), Some(5120), Some(5120)]
        );

        let layer_02_bank = resident_plan.activation_bank("layer_02").unwrap();
        assert_eq!(
            layer_02_bank
                .slots
                .iter()
                .map(|slot| slot.bytes)
                .collect::<Vec<_>>(),
            vec![Some(2048), Some(5120), Some(5120), Some(5120)]
        );
    }

    #[test]
    fn resident_plan_keeps_sizes_unknown_without_tensor_and_element_metadata() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let resource_plan = StreamCircuitResourcePlan::from_graph(&graph).unwrap();

        let resident_plan =
            VulkanStreamCircuitResidentPlan::from_resource_plan(&resource_plan, None, None)
                .unwrap();

        assert_eq!(resident_plan.permanent_parameters.len(), 130);
        assert_eq!(resident_plan.permanent_parameter_bytes, None);
        assert_eq!(resident_plan.unresolved_parameter_tensors.len(), 130);
        assert_eq!(resident_plan.per_stream_static_state_elements, 8 * 3 * 1024);
        assert_eq!(
            resident_plan.per_stream_dynamic_state_elements_per_activation,
            6 * 1024
        );
        assert_eq!(resident_plan.per_stream_static_state_bytes, None);
        assert_eq!(resident_plan.per_stream_activation_slot_elements, None);
        assert_eq!(resident_plan.per_stream_activation_slot_bytes, None);
        assert!(!resident_plan.unresolved_activation_slots.is_empty());
    }

    #[test]
    fn allocates_lfm2_per_stream_vulkan_buffers_from_resident_plan() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan stream-circuit allocation: {error}");
                return;
            }
        };
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        let buffers = resident_plan.allocate_stream_buffers(&device, 4).unwrap();

        assert_eq!(buffers.dynamic_state_capacity_activations, 4);
        assert_eq!(buffers.state_buffers.len(), 14);
        assert_eq!(buffers.activation_slot_buffers.len(), 56);
        assert_eq!(buffers.total_byte_capacity, 49_152 + 12_288 * 4 + 276_480);

        let layer_00_state = buffers
            .state_buffers
            .iter()
            .find(|buffer| buffer.pedal_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_state.state_id, "temporal_memory");
        assert_eq!(layer_00_state.byte_capacity, 6_144);
        assert_eq!(layer_00_state.buffer.byte_capacity(), 6_144);

        let layer_02_state = buffers
            .state_buffers
            .iter()
            .find(|buffer| buffer.pedal_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_state.state_id, "kv_memory");
        assert_eq!(layer_02_state.byte_capacity, 8_192);
        assert_eq!(layer_02_state.buffer.byte_capacity(), 8_192);

        let layer_00_slot_1 = buffers
            .activation_slot_buffers
            .iter()
            .find(|buffer| buffer.pedal_id == "layer_00" && buffer.slot == 1)
            .unwrap();
        assert_eq!(layer_00_slot_1.byte_capacity, 6_144);
        assert!(
            layer_00_slot_1
                .signal_ids
                .contains(&"conv_projected".to_string())
        );
        assert_eq!(
            buffers
                .state_buffer("layer_02", "kv_memory")
                .map(|buffer| buffer.byte_capacity),
            Some(8_192)
        );
        assert_eq!(
            buffers
                .activation_slot_buffer("layer_02", 0)
                .map(|buffer| buffer.byte_capacity),
            Some(2_048)
        );
    }

    #[test]
    fn binds_lfm2_nodes_to_vulkan_resident_resources() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            &execution_plan,
            &resource_plan,
            &resident_plan,
        )
        .unwrap();

        assert_eq!(binding_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(binding_plan.circuits.len(), 14);
        assert_eq!(binding_plan.total_node_count(), 242);

        let layer_00 = binding_plan.circuit("layer_00").unwrap();
        let operator_norm = layer_00.node("operator_norm").unwrap();
        assert_eq!(
            operator_norm.input("input_frame").unwrap().resource,
            VulkanSignalResource::BoundaryInput
        );
        assert_eq!(
            operator_norm.parameter("operator_norm").unwrap().tensor,
            "model.layers.0.operator_norm.weight"
        );

        let conv_in = layer_00.node("conv_in_projection").unwrap();
        assert_eq!(
            conv_in.parameter("conv_in_projection").unwrap().tensor,
            "model.layers.0.conv.in_proj.weight"
        );
        assert_eq!(
            conv_in.output("conv_projected").unwrap().resource,
            VulkanSignalResource::ActivationSlot {
                pedal_id: "layer_00".to_string(),
                slot: 1,
                bytes: Some(6144)
            }
        );

        let temporal_update = layer_00.node("temporal_memory_update").unwrap();
        assert_eq!(
            temporal_update.input("temporal_memory").unwrap().resource,
            VulkanSignalResource::StateBuffer {
                pedal_id: "layer_00".to_string(),
                state_id: "temporal_memory".to_string(),
                static_bytes: Some(6144),
                bytes_per_activation: None,
            }
        );
        assert_eq!(
            temporal_update.output("temporal_window").unwrap().resource,
            VulkanSignalResource::StateView {
                pedal_id: "layer_00".to_string(),
                state_id: "temporal_memory".to_string(),
                static_bytes: Some(6144),
                bytes_per_activation: None,
            }
        );

        let layer_02 = binding_plan.circuit("layer_02").unwrap();
        let kv_append = layer_02.node("kv_memory_append").unwrap();
        assert_eq!(
            kv_append.input("kv_memory").unwrap().resource,
            VulkanSignalResource::StateBuffer {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );
        assert_eq!(
            kv_append.output("k_memory").unwrap().resource,
            VulkanSignalResource::StateView {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );
        assert_eq!(
            kv_append.output("v_memory").unwrap().resource,
            VulkanSignalResource::StateView {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );

        let attention = layer_02.node("attention_read").unwrap();
        assert_eq!(
            attention.input("q_positioned").unwrap().resource,
            VulkanSignalResource::ActivationSlot {
                pedal_id: "layer_02".to_string(),
                slot: 2,
                bytes: Some(5120),
            }
        );
        assert!(matches!(
            attention.input("k_memory").unwrap().resource,
            VulkanSignalResource::StateView { .. }
        ));
        assert_eq!(
            attention.output("attention_out").unwrap().resource,
            VulkanSignalResource::ActivationSlot {
                pedal_id: "layer_02".to_string(),
                slot: 0,
                bytes: Some(2048),
            }
        );
    }

    #[test]
    fn kernel_interfaces_describe_lfm2_compiled_pedal_abi() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            &execution_plan,
            &resource_plan,
            &resident_plan,
        )
        .unwrap();

        let kernel_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);

        assert_eq!(kernel_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(kernel_plan.circuits.len(), 14);
        assert_eq!(kernel_plan.total_kernel_count(), 242);

        let conv_in = kernel_plan
            .kernel("layer_00", "conv_in_projection")
            .unwrap();
        assert_eq!(conv_in.kernel_id, "layer_00.conv_in_projection");
        assert_eq!(conv_in.op, "linear");
        assert_eq!(conv_in.inputs.len(), 1);
        assert_eq!(conv_in.outputs.len(), 1);
        assert_eq!(conv_in.parameters.len(), 1);
        assert!(conv_in.state_reads.is_empty());
        assert!(conv_in.state_writes.is_empty());
        assert!(conv_in.state_views.is_empty());
        assert!(!conv_in.stream_metadata.uses_stream_tick);
        assert_eq!(
            conv_in.parameters[0],
            VulkanParameterBinding {
                param_id: "conv_in_projection".to_string(),
                tensor: "model.layers.0.conv.in_proj.weight".to_string(),
                byte_count: Some(6_291_456),
            }
        );
        assert_eq!(
            conv_in.outputs[0].resource,
            VulkanSignalResource::ActivationSlot {
                pedal_id: "layer_00".to_string(),
                slot: 1,
                bytes: Some(6144),
            }
        );

        let q_rope = kernel_plan.kernel("layer_02", "q_rope").unwrap();
        assert_eq!(q_rope.op, "rotary_position_embedding");
        assert!(q_rope.stream_metadata.uses_stream_tick);
        assert_eq!(
            q_rope.stream_metadata.stream_tick,
            VulkanKernelScalarBinding {
                name: "stream_tick".to_string(),
                scalar_type: "u64".to_string(),
                source: VulkanKernelScalarSource::PushConstant,
            }
        );
        assert_eq!(q_rope.stream_metadata.control_flags.name, "control_flags");
        assert_eq!(
            q_rope.outputs[0].resource,
            VulkanSignalResource::ActivationSlot {
                pedal_id: "layer_02".to_string(),
                slot: 2,
                bytes: Some(5120),
            }
        );

        let kv_append = kernel_plan.kernel("layer_02", "kv_memory_append").unwrap();
        assert_eq!(kv_append.op, "append_state_update");
        assert!(kv_append.stream_metadata.uses_stream_tick);
        assert_eq!(kv_append.inputs.len(), 3);
        assert_eq!(kv_append.outputs.len(), 2);
        assert_eq!(kv_append.state_reads.len(), 1);
        assert_eq!(kv_append.state_writes.len(), 1);
        assert_eq!(kv_append.state_views.len(), 2);
        assert_eq!(
            kv_append.inputs[2].resource,
            VulkanSignalResource::StateBuffer {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );
        assert!(
            kv_append
                .state_views
                .iter()
                .all(|view| matches!(view.resource, VulkanSignalResource::StateView { .. }))
        );
        assert_eq!(
            kv_append
                .stream_metadata
                .dynamic_state_capacity_activations
                .name,
            "dynamic_state_capacity_activations"
        );
    }

    #[test]
    fn dispatch_plan_orders_lfm2_kernel_commands_for_stream_ticks() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            &execution_plan,
            &resource_plan,
            &resident_plan,
        )
        .unwrap();

        let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

        assert_eq!(dispatch_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(dispatch_plan.total_dispatch_count(), 242);
        assert_eq!(dispatch_plan.op_counts().get("linear"), Some(&82));

        let first = &dispatch_plan.commands[0];
        assert_eq!(first.dispatch_index, 0);
        assert_eq!(first.circuit_index, 0);
        assert_eq!(first.kernel_id, "layer_00.operator_norm");
        assert_eq!(first.pedal_id, "layer_00");
        assert_eq!(first.node_index, 0);
        assert_eq!(first.op, "rms_norm");
        assert_eq!(first.descriptor_bindings.len(), 3);
        assert_eq!(
            first
                .descriptor_bindings
                .iter()
                .map(|binding| binding.usage.clone())
                .collect::<Vec<_>>(),
            vec![
                VulkanKernelDescriptorUsage::InputSignal,
                VulkanKernelDescriptorUsage::OutputSignal,
                VulkanKernelDescriptorUsage::Parameter,
            ]
        );
        assert_eq!(
            first
                .push_constants
                .iter()
                .map(|binding| binding.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "stream_tick",
                "control_flags",
                "dynamic_state_capacity_activations"
            ]
        );
        assert!(!first.uses_stream_tick);

        let kv_append = dispatch_plan
            .command("layer_02", "kv_memory_append")
            .unwrap();
        assert_eq!(kv_append.dispatch_index, 40);
        assert_eq!(kv_append.circuit_index, 2);
        assert_eq!(kv_append.node_index, 8);
        assert_eq!(kv_append.op, "append_state_update");
        assert!(kv_append.uses_stream_tick);
        assert_eq!(
            kv_append
                .descriptor_bindings
                .iter()
                .map(|binding| (
                    binding.binding,
                    binding.usage.clone(),
                    binding.name.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, VulkanKernelDescriptorUsage::InputSignal, "k_positioned"),
                (1, VulkanKernelDescriptorUsage::InputSignal, "v_projected"),
                (2, VulkanKernelDescriptorUsage::InputSignal, "kv_memory"),
                (3, VulkanKernelDescriptorUsage::OutputSignal, "k_memory"),
                (4, VulkanKernelDescriptorUsage::OutputSignal, "v_memory"),
                (5, VulkanKernelDescriptorUsage::StateRead, "kv_memory"),
                (6, VulkanKernelDescriptorUsage::StateWrite, "kv_memory"),
                (7, VulkanKernelDescriptorUsage::StateView, "k_memory"),
                (8, VulkanKernelDescriptorUsage::StateView, "v_memory"),
            ]
        );
        assert_eq!(
            kv_append.descriptor_bindings[2].resource,
            VulkanKernelDescriptorResource::Signal(VulkanSignalBinding {
                signal_id: "kv_memory".to_string(),
                resource: VulkanSignalResource::StateBuffer {
                    pedal_id: "layer_02".to_string(),
                    state_id: "kv_memory".to_string(),
                    static_bytes: None,
                    bytes_per_activation: Some(2048),
                },
            })
        );
        assert_eq!(
            kv_append.descriptor_bindings[6].resource,
            VulkanKernelDescriptorResource::State {
                pedal_id: "layer_02".to_string(),
                binding: VulkanStateBinding {
                    state_id: "kv_memory".to_string(),
                    state_type: "append_only_attention_memory".to_string(),
                    static_bytes: None,
                    bytes_per_activation: Some(2048),
                },
            }
        );

        let last = dispatch_plan.commands.last().unwrap();
        assert_eq!(last.dispatch_index, 241);
        assert_eq!(last.circuit_index, 13);
        assert_eq!(last.kernel_id, "layer_13.ffn_residual");
        assert_eq!(last.node_index, 15);
        assert_eq!(
            last.descriptor_bindings.last().unwrap().resource,
            VulkanKernelDescriptorResource::Signal(VulkanSignalBinding {
                signal_id: "output_frame".to_string(),
                resource: VulkanSignalResource::BoundaryOutput,
            })
        );
    }

    #[test]
    fn reusable_kernel_plan_collapses_lfm2_dispatches_into_op_families() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            &execution_plan,
            &resource_plan,
            &resident_plan,
        )
        .unwrap();
        let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

        let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);

        assert_eq!(reusable_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(reusable_plan.total_command_count, 242);
        assert_eq!(reusable_plan.total_family_count(), 12);
        assert_eq!(reusable_plan.reusable_family_count(), 12);
        assert_eq!(reusable_plan.families_for_op("rms_norm").len(), 1);

        let linear = reusable_plan.family("linear").unwrap();
        assert_eq!(linear.op, "linear");
        assert_eq!(linear.command_refs.len(), 82);
        assert!(!linear.uses_stream_tick);
        assert_eq!(
            linear.descriptor_signature,
            vec![
                VulkanKernelDescriptorSlotSignature {
                    binding: 0,
                    usage: VulkanKernelDescriptorUsage::InputSignal,
                    resource_class: VulkanKernelDescriptorResourceClass::SignalBuffer,
                },
                VulkanKernelDescriptorSlotSignature {
                    binding: 1,
                    usage: VulkanKernelDescriptorUsage::OutputSignal,
                    resource_class: VulkanKernelDescriptorResourceClass::SignalBuffer,
                },
                VulkanKernelDescriptorSlotSignature {
                    binding: 2,
                    usage: VulkanKernelDescriptorUsage::Parameter,
                    resource_class: VulkanKernelDescriptorResourceClass::ParameterBuffer,
                },
            ]
        );
        assert_eq!(linear.command_refs[0].dispatch_index, 1);
        assert_eq!(
            linear.command_refs[0].kernel_id,
            "layer_00.conv_in_projection"
        );
        assert_eq!(
            linear.command_refs.last().unwrap().kernel_id,
            "layer_13.ffn_down_projection"
        );

        let rope = reusable_plan.family("rotary_position_embedding").unwrap();
        assert_eq!(rope.command_refs.len(), 12);
        assert!(rope.uses_stream_tick);
        assert_eq!(
            rope.push_constants
                .iter()
                .map(|binding| binding.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "stream_tick",
                "control_flags",
                "dynamic_state_capacity_activations"
            ]
        );

        let append = reusable_plan.family("append_state_update").unwrap();
        assert_eq!(append.command_refs.len(), 6);
        assert!(append.uses_stream_tick);
        assert_eq!(
            append
                .descriptor_signature
                .iter()
                .map(|slot| (
                    slot.binding,
                    slot.usage.clone(),
                    slot.resource_class.clone()
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    0,
                    VulkanKernelDescriptorUsage::InputSignal,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
                (
                    1,
                    VulkanKernelDescriptorUsage::InputSignal,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
                (
                    2,
                    VulkanKernelDescriptorUsage::InputSignal,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
                (
                    3,
                    VulkanKernelDescriptorUsage::OutputSignal,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
                (
                    4,
                    VulkanKernelDescriptorUsage::OutputSignal,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
                (
                    5,
                    VulkanKernelDescriptorUsage::StateRead,
                    VulkanKernelDescriptorResourceClass::StateBuffer,
                ),
                (
                    6,
                    VulkanKernelDescriptorUsage::StateWrite,
                    VulkanKernelDescriptorResourceClass::StateBuffer,
                ),
                (
                    7,
                    VulkanKernelDescriptorUsage::StateView,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
                (
                    8,
                    VulkanKernelDescriptorUsage::StateView,
                    VulkanKernelDescriptorResourceClass::SignalBuffer,
                ),
            ]
        );

        let split = reusable_plan.family("split").unwrap();
        assert_eq!(split.command_refs.len(), 8);
        assert_eq!(split.descriptor_signature.len(), 4);
    }

    #[test]
    fn reusable_kernel_coverage_reports_missing_gpu_pedal_circuits() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            &execution_plan,
            &resource_plan,
            &resident_plan,
        )
        .unwrap();
        let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
        let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);

        let empty = reusable_plan.coverage_report(std::iter::empty::<&str>());
        assert!(!empty.all_available());
        assert_eq!(empty.required_family_count, 12);
        assert_eq!(empty.available_family_count, 0);
        assert_eq!(empty.missing_family_count, 12);
        assert_eq!(empty.required_command_count, 242);
        assert_eq!(empty.covered_command_count, 0);
        assert_eq!(empty.missing_command_count, 242);
        assert!(
            empty
                .missing_families()
                .iter()
                .any(|family| family.family_id == "linear" && family.command_count == 82)
        );

        let partial = reusable_plan.coverage_report(["linear", "rms_norm"]);
        assert!(!partial.all_available());
        assert_eq!(partial.available_family_count, 2);
        assert_eq!(partial.missing_family_count, 10);
        assert_eq!(partial.covered_command_count, 82 + 28);
        assert_eq!(partial.missing_command_count, 242 - 82 - 28);
        assert_eq!(partial.missing_families().len(), 10);

        let full = reusable_plan.coverage_report(
            reusable_plan
                .families
                .iter()
                .map(|family| family.family_id.as_str()),
        );
        assert!(full.all_available());
        assert_eq!(full.available_family_count, 12);
        assert_eq!(full.missing_family_count, 0);
        assert_eq!(full.covered_command_count, 242);
        assert_eq!(full.missing_command_count, 0);
    }

    #[test]
    fn mounts_lfm2_stream_circuit_resources_without_claiming_execution() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan stream-circuit mount: {error}");
                return;
            }
        };
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
            &resource_plan,
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        let mounted = VulkanMountedStreamCircuit::from_plans(
            &device,
            &execution_plan,
            &resource_plan,
            resident_plan,
            4,
        )
        .unwrap();

        assert!(!mounted.can_execute());
        assert_eq!(mounted.resident_plan.permanent_parameters.len(), 130);
        assert_eq!(mounted.binding_plan.total_node_count(), 242);
        assert_eq!(mounted.kernel_interface_plan.total_kernel_count(), 242);
        assert_eq!(mounted.dispatch_plan.total_dispatch_count(), 242);
        assert_eq!(mounted.reusable_kernel_plan.total_family_count(), 12);
        assert_eq!(mounted.reusable_kernel_plan.total_command_count, 242);
        let empty_coverage = mounted.reusable_kernel_coverage_report(std::iter::empty::<&str>());
        assert!(!empty_coverage.all_available());
        assert_eq!(empty_coverage.missing_family_count, 12);
        assert_eq!(empty_coverage.missing_command_count, 242);
        assert_eq!(mounted.buffers.state_buffers.len(), 14);
        assert_eq!(mounted.buffers.activation_slot_buffers.len(), 56);
        assert_eq!(mounted.buffers.total_byte_capacity, 374_784);

        let attention = mounted
            .binding_plan
            .circuit("layer_02")
            .unwrap()
            .node("attention_read")
            .unwrap();
        assert!(matches!(
            attention.input("k_memory").unwrap().resource,
            VulkanSignalResource::StateView { .. }
        ));
        assert_eq!(
            mounted
                .buffers
                .activation_slot_buffer("layer_02", 0)
                .map(|buffer| buffer.byte_capacity),
            Some(2_048)
        );
        assert_eq!(
            mounted
                .dispatch_plan
                .command("layer_02", "kv_memory_append")
                .map(|command| command.dispatch_index),
            Some(40)
        );
    }
}
