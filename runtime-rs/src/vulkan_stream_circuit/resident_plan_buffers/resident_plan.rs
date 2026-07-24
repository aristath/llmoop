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
        Self::from_resource_plan_with_hosted_components(
            resource_plan,
            None,
            tensor_index,
            activation_element_bytes,
        )
    }

    fn from_resource_plan_with_hosted_components(
        resource_plan: &StreamCircuitResourcePlan,
        hosted_components: Option<&BTreeSet<String>>,
        tensor_index: Option<&TensorIndex>,
        activation_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanResidentPlanError> {
        let hosts_component = |component_id: &str| {
            hosted_components
                .map(|components| components.contains(component_id))
                .unwrap_or(true)
        };
        let mut permanent_parameters = Vec::with_capacity(resource_plan.parameters.len());
        let mut permanent_parameter_bytes = Some(0usize);
        let mut unresolved_parameter_tensors = Vec::new();

        for parameter in &resource_plan.parameters {
            let hosted_use_count = parameter
                .uses
                .iter()
                .filter(|use_ref| hosts_component(&use_ref.component_id))
                .count();
            if hosted_use_count == 0 {
                continue;
            }
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
                use_count: hosted_use_count,
            });
        }

        let mut stream_state_buffers = Vec::with_capacity(resource_plan.state_allocations.len());
        let mut per_stream_static_state_elements = 0usize;
        let mut per_stream_dynamic_state_elements_per_activation = 0usize;
        let mut per_stream_static_state_bytes = Some(0usize);
        let mut per_stream_dynamic_state_bytes_per_activation = Some(0usize);

        for state in &resource_plan.state_allocations {
            if !hosts_component(&state.component_id) {
                continue;
            }
            if state
                .sharing
                .as_deref()
                .is_some_and(|sharing| sharing.starts_with("shared_from:"))
            {
                continue;
            }
            let clone_from = state
                .sharing
                .as_deref()
                .and_then(|sharing| sharing.strip_prefix("clone_from:"))
                .map(parse_state_source)
                .transpose()
                .map_err(|error| VulkanResidentPlanError(error.to_string()))?;
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

            let state_element_bytes = state.element_bytes.or(activation_element_bytes);
            per_stream_static_state_bytes = optional_add(
                per_stream_static_state_bytes,
                optional_state_contribution_bytes(static_elements, state_element_bytes)?,
                "per-stream static state bytes",
            )?;
            per_stream_dynamic_state_bytes_per_activation = optional_add(
                per_stream_dynamic_state_bytes_per_activation,
                optional_state_contribution_bytes(
                    state.elements_per_activation,
                    state_element_bytes,
                )?,
                "per-stream dynamic state bytes per activation",
            )?;

            stream_state_buffers.push(VulkanResidentStateBuffer {
                component_id: state.component_id.clone(),
                state_id: state.state_id.clone(),
                state_type: state.state_type.clone(),
                layout: state.layout.clone(),
                static_elements,
                elements_per_activation: state.elements_per_activation,
                max_dynamic_activations: state.max_dynamic_activations,
                static_bytes: optional_mul(static_elements, state_element_bytes)?,
                bytes_per_activation: optional_mul(
                    state.elements_per_activation,
                    state_element_bytes,
                )?,
                clone_from,
            });
        }

        let mut activation_banks = Vec::with_capacity(resource_plan.activation_banks.len());
        let mut per_stream_activation_slot_elements = Some(0usize);
        let mut per_stream_activation_slot_bytes = Some(0usize);
        let mut unresolved_activation_slots = Vec::new();

        for bank in &resource_plan.activation_banks {
            if !hosts_component(&bank.component_id) {
                continue;
            }
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
                    }
                }
                let bytes = match slot.max_bytes {
                    Some(bytes) => Some(bytes),
                    None => optional_mul(slot.max_elements, activation_element_bytes)?,
                };
                match (per_stream_activation_slot_bytes, bytes) {
                    (Some(total), Some(bytes)) => {
                        per_stream_activation_slot_bytes = Some(checked_add(
                            total,
                            bytes,
                            "per-stream activation slot bytes",
                        )?);
                    }
                    _ => {
                        per_stream_activation_slot_bytes = None;
                        unresolved_activation_slots
                            .push(format!("{}.slot_{}", bank.component_id, slot.slot));
                    }
                }

                slots.push(VulkanResidentActivationSlot {
                    slot: slot.slot,
                    signal_ids: slot.signal_ids.clone(),
                    max_elements: slot.max_elements,
                    bytes,
                });
            }

            activation_banks.push(VulkanResidentActivationBank {
                component_id: bank.component_id.clone(),
                circuit_id: bank.circuit_id.clone(),
                slot_count: bank.slot_count,
                slots,
            });
        }
        let circuit_count = resource_plan
            .activation_banks
            .iter()
            .filter(|bank| hosts_component(&bank.component_id))
            .count();
        let state_view_signal_count = resource_plan
            .activation_banks
            .iter()
            .filter(|bank| hosts_component(&bank.component_id))
            .map(|bank| bank.state_view_signal_count)
            .sum();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            circuit_count,
            permanent_parameters,
            permanent_parameter_bytes,
            stream_state_buffers,
            state_view_signal_count,
            activation_banks,
            per_stream_static_state_elements,
            per_stream_dynamic_state_elements_per_activation,
            per_stream_activation_slot_elements,
            per_stream_static_state_bytes,
            per_stream_dynamic_state_bytes_per_activation,
            per_stream_activation_slot_bytes,
            unresolved_parameter_tensors,
            unresolved_activation_slots,
        })
    }

    pub fn activation_bank(&self, component_id: &str) -> Option<&VulkanResidentActivationBank> {
        self.activation_banks
            .iter()
            .find(|bank| bank.component_id == component_id)
    }

    pub fn allocate_stream_buffers(
        &self,
        device: &VulkanComputeDevice,
        dynamic_state_capacity_activations: usize,
    ) -> Result<VulkanStreamCircuitStreamBuffers, VulkanError> {
        self.allocate_stream_buffers_with_activation_overrides(
            device,
            dynamic_state_capacity_activations,
            &[],
        )
    }

    pub fn allocate_stream_buffers_with_activation_overrides(
        &self,
        device: &VulkanComputeDevice,
        dynamic_state_capacity_activations: usize,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
    ) -> Result<VulkanStreamCircuitStreamBuffers, VulkanError> {
        let mut state_buffers = Vec::with_capacity(self.stream_state_buffers.len());
        let mut activation_slot_buffers = Vec::new();
        let mut total_byte_capacity = 0usize;
        let mut override_index = BTreeMap::new();

        for activation_override in activation_overrides {
            let key = (
                activation_override.component_id.clone(),
                activation_override.slot,
            );
            if override_index.insert(key, activation_override).is_some() {
                return Err(VulkanError(format!(
                    "activation buffer override repeats {}.slot_{}",
                    activation_override.component_id, activation_override.slot
                )));
            }
            if !device.owns_resident_buffer(&activation_override.buffer) {
                return Err(VulkanError(format!(
                    "activation buffer override {}.slot_{} belongs to a different Vulkan logical device",
                    activation_override.component_id, activation_override.slot
                )));
            }
            if !activation_override.buffer.is_shared_host_backed() {
                return Err(VulkanError(format!(
                    "activation buffer override {}.slot_{} is not backed by shared host memory",
                    activation_override.component_id, activation_override.slot
                )));
            }
            let slot = self
                .activation_bank(&activation_override.component_id)
                .and_then(|bank| {
                    bank.slots
                        .iter()
                        .find(|slot| slot.slot == activation_override.slot)
                })
                .ok_or_else(|| {
                    VulkanError(format!(
                        "activation buffer override {}.slot_{} does not address a resident activation slot",
                        activation_override.component_id, activation_override.slot
                    ))
                })?;
            let required_byte_capacity = slot.bytes.ok_or_else(|| {
                VulkanError(format!(
                    "{} activation slot {} has unknown byte size",
                    activation_override.component_id, activation_override.slot
                ))
            })?;
            if activation_override.buffer.byte_capacity() < required_byte_capacity {
                return Err(VulkanError(format!(
                    "activation buffer override {}.slot_{} has {} bytes but requires {required_byte_capacity}",
                    activation_override.component_id,
                    activation_override.slot,
                    activation_override.buffer.byte_capacity()
                )));
            }
        }

        for state in &self.stream_state_buffers {
            let byte_capacity =
                stream_state_byte_capacity(state, dynamic_state_capacity_activations)?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "stream state buffer allocation",
            )?;
            state_buffers.push(VulkanStreamStateBufferAllocation {
                component_id: state.component_id.clone(),
                state_id: state.state_id.clone(),
                state_type: state.state_type.clone(),
                byte_capacity,
                static_byte_capacity: state.static_bytes,
                bytes_per_activation: state.bytes_per_activation,
                clone_from: state.clone_from.clone(),
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        for bank in &self.activation_banks {
            for slot in &bank.slots {
                let byte_capacity = slot.bytes.ok_or_else(|| {
                    VulkanError(format!(
                        "{} activation slot {} has unknown byte size",
                        bank.component_id, slot.slot
                    ))
                })?;
                total_byte_capacity = checked_add_bytes(
                    total_byte_capacity,
                    byte_capacity,
                    "activation slot buffer allocation",
                )?;
                let activation_override =
                    override_index.remove(&(bank.component_id.clone(), slot.slot));
                let (buffer, shared_across_devices) = match activation_override {
                    Some(activation_override) => (Arc::clone(&activation_override.buffer), true),
                    None => (
                        Arc::new(device.create_resident_buffer(byte_capacity)?),
                        false,
                    ),
                };
                activation_slot_buffers.push(VulkanActivationSlotBufferAllocation {
                    component_id: bank.component_id.clone(),
                    circuit_id: bank.circuit_id.clone(),
                    slot: slot.slot,
                    signal_ids: slot.signal_ids.clone(),
                    byte_capacity,
                    shared_across_devices,
                    buffer,
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
