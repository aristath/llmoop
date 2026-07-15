use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::stream_plan::{StreamCircuitResourcePlan, TensorIndex};
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
    use crate::stream_plan::StreamCircuitResourcePlan;

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
    }
}
