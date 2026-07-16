use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::stream_circuit::{
    CableTransport, PedalCablePlacement, ResolvedLoweredPedalboard, StreamCircuitPlacementPlan,
    StreamCircuitPlacementSpec,
};
use crate::stream_plan::{
    CircuitActivationPlan, PlannedNode, PlannedPort, SignalProducer, SignalStorage,
    StreamCircuitExecutionPlan, StreamCircuitResourcePlan, TensorIndex,
};
use crate::vulkan::{DEFAULT_COMPUTE_LOCAL_SIZE_X, DEFAULT_SPIRV_ENTRY_POINT, read_spirv_words};
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanResidentBuffer, VulkanResidentKernelBufferBinding,
    VulkanResidentKernelDispatch,
};

pub const VULKAN_STREAM_CIRCUIT_BACKEND_ID: &str = "vulkan_stream_circuit_ir";
pub const VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA: &str =
    "llmoop.vulkan_reusable_kernel_artifacts.v1";
const LFM2_TOKEN_EMBEDDING_TRANSDUCER_ID: &str = "input_transducer.token_embedding";
const LFM2_OUTPUT_EMBEDDING_NORM_TRANSDUCER_ID: &str = "output_transducer.embedding_norm";
const LFM2_TIED_OUTPUT_PROJECTION_TRANSDUCER_ID: &str = "output_transducer.tied_output_projection";
const LFM2_GREEDY_SAMPLER_PEDAL_ID: &str = "greedy_sampler";
const LFM2_EMBED_TOKENS_TENSOR: &str = "model.embed_tokens.weight";
const LFM2_EMBEDDING_NORM_TENSOR: &str = "model.embedding_norm.weight";
const LFM2_INPUT_FRAME_SIGNAL: &str = "input_frame";
const LFM2_OUTPUT_FRAME_SIGNAL: &str = "output_frame";
const LFM2_VOCAB_SIZE: usize = 65_536;
const LFM2_HIDDEN_SIZE: usize = 1_024;
const LFM2_FRAME_BYTES: usize = LFM2_HIDDEN_SIZE * 2;
const LFM2_FRAME_WORDS: usize = LFM2_FRAME_BYTES / 4;
const LFM2_LOGITS_BYTES: usize = LFM2_VOCAB_SIZE * 4;
const LFM2_SAMPLER_OUTPUT_BYTES: usize = 16;
const LFM2_EMBED_TOKENS_BYTES: usize = LFM2_VOCAB_SIZE * LFM2_FRAME_BYTES;
const VULKAN_INPUT_EMBEDDING_LOOKUP_LOCAL_SIZE_X: u32 = 256;
const VULKAN_OUTPUT_PROJECTION_LOCAL_SIZE_X: u32 = 64;
const VULKAN_GREEDY_SAMPLER_LOCAL_SIZE_X: u32 = 64;

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
        Self::from_resource_plan_with_hosted_pedals(
            resource_plan,
            None,
            tensor_index,
            activation_element_bytes,
        )
    }

    fn from_resource_plan_with_hosted_pedals(
        resource_plan: &StreamCircuitResourcePlan,
        hosted_pedals: Option<&BTreeSet<String>>,
        tensor_index: Option<&TensorIndex>,
        activation_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanResidentPlanError> {
        let hosts_pedal = |pedal_id: &str| {
            hosted_pedals
                .map(|pedals| pedals.contains(pedal_id))
                .unwrap_or(true)
        };
        let mut permanent_parameters = Vec::with_capacity(resource_plan.parameters.len());
        let mut permanent_parameter_bytes = Some(0usize);
        let mut unresolved_parameter_tensors = Vec::new();

        for parameter in &resource_plan.parameters {
            let hosted_use_count = parameter
                .uses
                .iter()
                .filter(|use_ref| hosts_pedal(&use_ref.pedal_id))
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

        for state in &resource_plan.state_allocations {
            if !hosts_pedal(&state.pedal_id) {
                continue;
            }
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
            if !hosts_pedal(&bank.pedal_id) {
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
        let circuit_count = resource_plan
            .activation_banks
            .iter()
            .filter(|bank| hosts_pedal(&bank.pedal_id))
            .count();
        let state_view_signal_count = resource_plan
            .activation_banks
            .iter()
            .filter(|bank| hosts_pedal(&bank.pedal_id))
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
pub struct VulkanPermanentParameterBufferPlan {
    pub backend_id: String,
    pub device_id: String,
    pub parameters: Vec<VulkanPermanentParameterBuffer>,
    pub parameter_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_tensors: Vec<String>,
}

impl VulkanPermanentParameterBufferPlan {
    pub fn from_placed_resident_plan(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError> {
        let mut parameters = Vec::with_capacity(
            placed_resident_plan
                .resident_plan
                .permanent_parameters
                .len(),
        );
        let mut tensor_ids = BTreeSet::new();
        let mut total_byte_capacity = Some(0usize);
        let mut unresolved_tensors = Vec::new();

        for parameter in &placed_resident_plan.resident_plan.permanent_parameters {
            if !tensor_ids.insert(parameter.tensor.clone()) {
                return Err(VulkanPermanentParameterBufferPlanError(format!(
                    "{} permanent parameter tensor {:?} appears more than once",
                    placed_resident_plan.device_id, parameter.tensor
                )));
            }

            match (total_byte_capacity, parameter.byte_count) {
                (Some(total), Some(bytes)) => {
                    total_byte_capacity = Some(add_parameter_bytes(
                        total,
                        bytes,
                        "permanent parameter buffer plan",
                    )?);
                }
                _ => {
                    total_byte_capacity = None;
                    unresolved_tensors.push(parameter.tensor.clone());
                }
            }

            parameters.push(VulkanPermanentParameterBuffer {
                buffer_index: parameters.len(),
                tensor: parameter.tensor.clone(),
                dtype: parameter.dtype.clone(),
                shape: parameter.shape.clone(),
                byte_capacity: parameter.byte_count,
                use_count: parameter.use_count,
            });
        }

        let parameter_count = parameters.len();
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_resident_plan.device_id.clone(),
            parameters,
            parameter_count,
            total_byte_capacity,
            unresolved_tensors,
        })
    }

    pub fn from_transducer_parameters(
        device_id: impl Into<String>,
        resource_plan: &StreamCircuitResourcePlan,
        tensor_index: Option<&TensorIndex>,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError> {
        let device_id = device_id.into();
        let mut parameters = Vec::with_capacity(resource_plan.transducer_parameters.len());
        let mut tensor_ids = BTreeSet::new();
        let mut total_byte_capacity = Some(0usize);
        let mut unresolved_tensors = Vec::new();

        for parameter in &resource_plan.transducer_parameters {
            if !tensor_ids.insert(parameter.tensor.clone()) {
                return Err(VulkanPermanentParameterBufferPlanError(format!(
                    "{device_id} transducer parameter tensor {:?} appears more than once",
                    parameter.tensor
                )));
            }

            let metadata = tensor_index.and_then(|index| index.tensors.get(&parameter.tensor));
            let byte_count = metadata.and_then(|metadata| metadata.byte_count);
            match (total_byte_capacity, byte_count) {
                (Some(total), Some(bytes)) => {
                    total_byte_capacity = Some(add_parameter_bytes(
                        total,
                        bytes,
                        "transducer parameter buffer plan",
                    )?);
                }
                _ => {
                    total_byte_capacity = None;
                    unresolved_tensors.push(parameter.tensor.clone());
                }
            }

            parameters.push(VulkanPermanentParameterBuffer {
                buffer_index: parameters.len(),
                tensor: parameter.tensor.clone(),
                dtype: metadata.map(|metadata| metadata.dtype.clone()),
                shape: metadata.map(|metadata| metadata.shape.clone()),
                byte_capacity: byte_count,
                use_count: parameter.uses.len(),
            });
        }

        let parameter_count = parameters.len();
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id,
            parameters,
            parameter_count,
            total_byte_capacity,
            unresolved_tensors,
        })
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanPermanentParameterBuffers, VulkanError> {
        let mut buffers = Vec::with_capacity(self.parameters.len());
        let mut total_byte_capacity = 0usize;

        for parameter in &self.parameters {
            let byte_capacity = parameter.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} permanent parameter {:?} has unknown byte capacity",
                    self.device_id, parameter.tensor
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "permanent parameter buffer allocation",
            )?;
            buffers.push(VulkanPermanentParameterBufferAllocation {
                parameter: parameter.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        Ok(VulkanPermanentParameterBuffers {
            plan: self.clone(),
            buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterBuffer {
    pub buffer_index: usize,
    pub tensor: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub byte_capacity: Option<usize>,
    pub use_count: usize,
}

pub struct VulkanPermanentParameterBuffers {
    pub plan: VulkanPermanentParameterBufferPlan,
    pub buffers: Vec<VulkanPermanentParameterBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPermanentParameterBuffers {
    pub fn parameter_buffer(
        &self,
        tensor: &str,
    ) -> Option<&VulkanPermanentParameterBufferAllocation> {
        self.buffers
            .iter()
            .find(|buffer| buffer.parameter.tensor == tensor)
    }

    pub fn load_parameter_from_tensor_index(
        &self,
        tensor_index: &TensorIndex,
        tensor: &str,
    ) -> Result<VulkanPermanentParameterLoadRecord, VulkanPermanentParameterLoadError> {
        let allocation = self.parameter_buffer(tensor).ok_or_else(|| {
            VulkanPermanentParameterLoadError(format!(
                "mounted parameter buffer for tensor {tensor:?} is missing"
            ))
        })?;
        load_parameter_allocation_from_tensor_index(allocation, tensor_index)
    }

    pub fn load_from_tensor_index(
        &self,
        tensor_index: &TensorIndex,
    ) -> Result<VulkanPermanentParameterLoadReport, VulkanPermanentParameterLoadError> {
        let mut records = Vec::with_capacity(self.buffers.len());
        let mut total_bytes_loaded = 0usize;
        let mut source_files = BTreeSet::new();

        for allocation in &self.buffers {
            let record = load_parameter_allocation_from_tensor_index(allocation, tensor_index)?;
            total_bytes_loaded = total_bytes_loaded
                .checked_add(record.byte_count)
                .ok_or_else(|| {
                    VulkanPermanentParameterLoadError(
                        "permanent parameter loaded byte count overflowed".to_string(),
                    )
                })?;
            source_files.insert(record.source_file.clone());
            records.push(record);
        }

        Ok(VulkanPermanentParameterLoadReport {
            parameter_count: self.buffers.len(),
            loaded_count: records.len(),
            total_bytes_loaded,
            source_file_count: source_files.len(),
            records,
        })
    }
}

pub struct VulkanPermanentParameterBufferAllocation {
    pub parameter: VulkanPermanentParameterBuffer,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterLoadReport {
    pub parameter_count: usize,
    pub loaded_count: usize,
    pub total_bytes_loaded: usize,
    pub source_file_count: usize,
    pub records: Vec<VulkanPermanentParameterLoadRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterLoadRecord {
    pub tensor: String,
    pub buffer_index: usize,
    pub source_file: String,
    pub data_start: usize,
    pub data_end: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterLoadError(pub String);

impl Display for VulkanPermanentParameterLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPermanentParameterLoadError {}

fn load_parameter_allocation_from_tensor_index(
    allocation: &VulkanPermanentParameterBufferAllocation,
    tensor_index: &TensorIndex,
) -> Result<VulkanPermanentParameterLoadRecord, VulkanPermanentParameterLoadError> {
    let tensor = &allocation.parameter.tensor;
    let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
        VulkanPermanentParameterLoadError(format!(
            "tensor index has no metadata for mounted parameter tensor {tensor:?}"
        ))
    })?;
    let source_file = metadata.source_file.as_ref().ok_or_else(|| {
        VulkanPermanentParameterLoadError(format!(
            "tensor metadata for {tensor:?} has no source_file"
        ))
    })?;
    let offsets = metadata.data_offsets.as_ref().ok_or_else(|| {
        VulkanPermanentParameterLoadError(format!(
            "tensor metadata for {tensor:?} has no data_offsets"
        ))
    })?;
    if offsets.len() != 2 {
        return Err(VulkanPermanentParameterLoadError(format!(
            "tensor metadata for {tensor:?} has invalid data_offsets {:?}",
            offsets
        )));
    }
    let data_start = offsets[0];
    let data_end = offsets[1];
    if data_end < data_start {
        return Err(VulkanPermanentParameterLoadError(format!(
            "tensor metadata for {tensor:?} has reversed data_offsets {:?}",
            offsets
        )));
    }
    let byte_count = data_end - data_start;
    if byte_count != allocation.byte_capacity {
        return Err(VulkanPermanentParameterLoadError(format!(
            "tensor {tensor:?} byte count {byte_count} does not match mounted buffer capacity {}",
            allocation.byte_capacity
        )));
    }
    if metadata.byte_count != Some(byte_count) {
        return Err(VulkanPermanentParameterLoadError(format!(
            "tensor {tensor:?} metadata byte_count {:?} does not match data_offsets byte count {byte_count}",
            metadata.byte_count
        )));
    }

    let source_path = Path::new(source_file);
    let data_base = safetensors_data_start(source_path)?;
    let absolute_start = data_base
        .checked_add(u64::try_from(data_start).map_err(|_| {
            VulkanPermanentParameterLoadError(format!(
                "tensor {tensor:?} data_start {data_start} cannot fit in u64"
            ))
        })?)
        .ok_or_else(|| {
            VulkanPermanentParameterLoadError(format!(
                "tensor {tensor:?} absolute data offset overflowed"
            ))
        })?;
    let mut file = fs::File::open(source_path).map_err(|error| {
        VulkanPermanentParameterLoadError(format!(
            "failed to open safetensors source {source_file:?}: {error}"
        ))
    })?;
    file.seek(SeekFrom::Start(absolute_start))
        .map_err(|error| {
            VulkanPermanentParameterLoadError(format!(
                "failed to seek safetensors source {source_file:?} to tensor {tensor:?}: {error}"
            ))
        })?;
    let mut bytes = vec![0u8; byte_count];
    file.read_exact(&mut bytes).map_err(|error| {
        VulkanPermanentParameterLoadError(format!(
            "failed to read tensor {tensor:?} from safetensors source {source_file:?}: {error}"
        ))
    })?;
    allocation.buffer.write_bytes(&bytes)?;

    Ok(VulkanPermanentParameterLoadRecord {
        tensor: tensor.clone(),
        buffer_index: allocation.parameter.buffer_index,
        source_file: source_file.clone(),
        data_start,
        data_end,
        byte_count,
    })
}

fn safetensors_data_start(path: &Path) -> Result<u64, VulkanPermanentParameterLoadError> {
    let mut file = fs::File::open(path).map_err(|error| {
        VulkanPermanentParameterLoadError(format!(
            "failed to open safetensors file {:?}: {error}",
            path
        ))
    })?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|error| {
        VulkanPermanentParameterLoadError(format!(
            "failed to read safetensors header length from {:?}: {error}",
            path
        ))
    })?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    8u64.checked_add(header_len).ok_or_else(|| {
        VulkanPermanentParameterLoadError(format!(
            "safetensors data start overflowed for {:?}",
            path
        ))
    })
}

impl From<VulkanError> for VulkanPermanentParameterLoadError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterBufferPlanError(pub String);

impl Display for VulkanPermanentParameterBufferPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPermanentParameterBufferPlanError {}

fn add_parameter_bytes(
    total: usize,
    bytes: usize,
    label: &str,
) -> Result<usize, VulkanPermanentParameterBufferPlanError> {
    total
        .checked_add(bytes)
        .ok_or_else(|| VulkanPermanentParameterBufferPlanError(format!("{label} overflowed")))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanModelBoundaryBufferPlan {
    pub backend_id: String,
    pub device_id: String,
    pub signal_element_bytes: Option<usize>,
    pub inputs: Vec<VulkanModelBoundaryBuffer>,
    pub outputs: Vec<VulkanModelBoundaryBuffer>,
    pub input_count: usize,
    pub output_count: usize,
    pub total_buffer_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_byte_signals: Vec<String>,
}

impl VulkanModelBoundaryBufferPlan {
    pub fn from_placed_plan(
        placed_plan: &VulkanPlacedStreamCircuitPlan,
    ) -> Result<Self, VulkanModelBoundaryBufferPlanError> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut total_byte_capacity = Some(0usize);
        let mut unresolved_byte_signals = Vec::new();
        let signal_element_bytes = placed_plan.placed_resident_plan.signal_element_bytes;

        for circuit in &placed_plan.binding_plan.circuits {
            for input in &circuit.input_ports {
                if placed_plan
                    .placed_resident_plan
                    .local_cables
                    .iter()
                    .chain(&placed_plan.placed_resident_plan.incoming_cables)
                    .any(|cable| {
                        cable.destination_pedal_id == circuit.pedal_id
                            && cable.destination_port_id == input.id
                    })
                {
                    continue;
                }
                let boundary = VulkanModelBoundaryBuffer::from_port(
                    inputs.len(),
                    &circuit.pedal_id,
                    input,
                    signal_element_bytes,
                )?;
                total_byte_capacity =
                    add_optional_boundary_bytes(total_byte_capacity, boundary.byte_capacity)?;
                if boundary.byte_capacity.is_none() {
                    unresolved_byte_signals.push(boundary.signal_id.clone());
                }
                inputs.push(boundary);
            }

            for output in &circuit.output_ports {
                if placed_plan
                    .placed_resident_plan
                    .local_cables
                    .iter()
                    .chain(&placed_plan.placed_resident_plan.outgoing_cables)
                    .any(|cable| {
                        cable.source_pedal_id == circuit.pedal_id
                            && cable.source_port_id == output.id
                    })
                {
                    continue;
                }
                let boundary = VulkanModelBoundaryBuffer::from_port(
                    outputs.len(),
                    &circuit.pedal_id,
                    output,
                    signal_element_bytes,
                )?;
                total_byte_capacity =
                    add_optional_boundary_bytes(total_byte_capacity, boundary.byte_capacity)?;
                if boundary.byte_capacity.is_none() {
                    unresolved_byte_signals.push(boundary.signal_id.clone());
                }
                outputs.push(boundary);
            }
        }

        let input_count = inputs.len();
        let output_count = outputs.len();
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_plan.device_id.clone(),
            signal_element_bytes,
            total_buffer_count: input_count + output_count,
            input_count,
            output_count,
            inputs,
            outputs,
            total_byte_capacity,
            unresolved_byte_signals,
        })
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanModelBoundaryBuffers, VulkanError> {
        let mut input_buffers = Vec::with_capacity(self.inputs.len());
        let mut output_buffers = Vec::with_capacity(self.outputs.len());
        let mut total_byte_capacity = 0usize;

        for boundary in &self.inputs {
            let byte_capacity = boundary.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} model input boundary {:?} has unknown byte capacity",
                    self.device_id, boundary.signal_id
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "model input boundary buffer allocation",
            )?;
            input_buffers.push(VulkanModelBoundaryBufferAllocation {
                boundary: boundary.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        for boundary in &self.outputs {
            let byte_capacity = boundary.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} model output boundary {:?} has unknown byte capacity",
                    self.device_id, boundary.signal_id
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "model output boundary buffer allocation",
            )?;
            output_buffers.push(VulkanModelBoundaryBufferAllocation {
                boundary: boundary.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        Ok(VulkanModelBoundaryBuffers {
            plan: self.clone(),
            input_buffers,
            output_buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanModelBoundaryBuffer {
    pub buffer_index: usize,
    pub signal_id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub pedal_id: String,
    pub port_id: String,
    pub source_signal_id: Option<String>,
}

impl VulkanModelBoundaryBuffer {
    fn from_port(
        buffer_index: usize,
        pedal_id: &str,
        port: &PlannedPort,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanModelBoundaryBufferPlanError> {
        let element_count = product(&port.shape).ok_or_else(|| {
            VulkanModelBoundaryBufferPlanError(format!(
                "{} model boundary port {:?} shape {:?} overflows",
                pedal_id, port.id, port.shape
            ))
        })?;
        if element_count == 0 {
            return Err(VulkanModelBoundaryBufferPlanError(format!(
                "{} model boundary port {:?} shape {:?} has zero elements",
                pedal_id, port.id, port.shape
            )));
        }
        let byte_capacity = signal_element_bytes
            .map(|bytes| {
                element_count.checked_mul(bytes).ok_or_else(|| {
                    VulkanModelBoundaryBufferPlanError(format!(
                        "{} model boundary port {:?} byte capacity overflowed",
                        pedal_id, port.id
                    ))
                })
            })
            .transpose()?;

        Ok(Self {
            buffer_index,
            signal_id: port.source.clone().unwrap_or_else(|| port.id.clone()),
            signal: port.signal.clone(),
            shape: port.shape.clone(),
            element_count,
            byte_capacity,
            pedal_id: pedal_id.to_string(),
            port_id: port.id.clone(),
            source_signal_id: port.source.clone(),
        })
    }
}

fn add_optional_boundary_bytes(
    total: Option<usize>,
    byte_capacity: Option<usize>,
) -> Result<Option<usize>, VulkanModelBoundaryBufferPlanError> {
    match (total, byte_capacity) {
        (Some(total), Some(bytes)) => total.checked_add(bytes).map(Some).ok_or_else(|| {
            VulkanModelBoundaryBufferPlanError(
                "model boundary total byte capacity overflowed".to_string(),
            )
        }),
        _ => Ok(None),
    }
}

pub struct VulkanModelBoundaryBuffers {
    pub plan: VulkanModelBoundaryBufferPlan,
    pub input_buffers: Vec<VulkanModelBoundaryBufferAllocation>,
    pub output_buffers: Vec<VulkanModelBoundaryBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanModelBoundaryBuffers {
    pub fn input_buffer(&self, signal_id: &str) -> Option<&VulkanModelBoundaryBufferAllocation> {
        self.input_buffers
            .iter()
            .find(|buffer| buffer.boundary.signal_id == signal_id)
    }

    pub fn output_buffer(&self, signal_id: &str) -> Option<&VulkanModelBoundaryBufferAllocation> {
        self.output_buffers
            .iter()
            .find(|buffer| buffer.boundary.signal_id == signal_id)
    }
}

pub struct VulkanModelBoundaryBufferAllocation {
    pub boundary: VulkanModelBoundaryBuffer,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanModelBoundaryDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanModelBoundaryBufferPlanError(pub String);

impl Display for VulkanModelBoundaryBufferPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanModelBoundaryBufferPlanError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedStreamCircuitResidentPlan {
    pub backend_id: String,
    pub device_id: String,
    pub hosted_pedal_ids: Vec<String>,
    pub signal_element_bytes: Option<usize>,
    pub local_cables: Vec<PedalCablePlacement>,
    pub incoming_cables: Vec<PedalCablePlacement>,
    pub outgoing_cables: Vec<PedalCablePlacement>,
    pub resident_plan: VulkanStreamCircuitResidentPlan,
}

impl VulkanPlacedStreamCircuitResidentPlan {
    pub fn from_resource_plan_for_device(
        resource_plan: &StreamCircuitResourcePlan,
        placement_plan: &StreamCircuitPlacementPlan,
        device_id: impl Into<String>,
        tensor_index: Option<&TensorIndex>,
        activation_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanResidentPlanError> {
        let device_id = device_id.into();
        if device_id.is_empty() {
            return Err(VulkanResidentPlanError(
                "Vulkan placed resident plan device_id must not be empty".to_string(),
            ));
        }
        let hosted_pedal_ids = placement_plan
            .pedals
            .iter()
            .filter(|pedal| pedal.device_id == device_id)
            .map(|pedal| pedal.pedal_id.clone())
            .collect::<Vec<_>>();
        let hosted_pedal_set = hosted_pedal_ids.iter().cloned().collect::<BTreeSet<_>>();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan_with_hosted_pedals(
            resource_plan,
            Some(&hosted_pedal_set),
            tensor_index,
            activation_element_bytes,
        )?;
        let local_cables = placement_plan
            .cables
            .iter()
            .filter(|cable| {
                cable.source_device_id == device_id && cable.destination_device_id == device_id
            })
            .cloned()
            .collect();
        let incoming_cables = placement_plan
            .cables
            .iter()
            .filter(|cable| {
                cable.source_device_id != device_id && cable.destination_device_id == device_id
            })
            .cloned()
            .collect();
        let outgoing_cables = placement_plan
            .cables
            .iter()
            .filter(|cable| {
                cable.source_device_id == device_id && cable.destination_device_id != device_id
            })
            .cloned()
            .collect();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id,
            hosted_pedal_ids,
            signal_element_bytes: activation_element_bytes,
            local_cables,
            incoming_cables,
            outgoing_cables,
            resident_plan,
        })
    }

    pub fn hosts_pedal(&self, pedal_id: &str) -> bool {
        self.hosted_pedal_ids
            .iter()
            .any(|hosted| hosted == pedal_id)
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

    pub fn state_buffer_index(&self, pedal_id: &str, state_id: &str) -> Option<usize> {
        self.state_buffers
            .iter()
            .position(|buffer| buffer.pedal_id == pedal_id && buffer.state_id == state_id)
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

    pub fn activation_slot_buffer_index(&self, pedal_id: &str, slot: usize) -> Option<usize> {
        self.activation_slot_buffers
            .iter()
            .position(|buffer| buffer.pedal_id == pedal_id && buffer.slot == slot)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableIoPlan {
    pub backend_id: String,
    pub device_id: String,
    pub signal_element_bytes: Option<usize>,
    pub local_cables: Vec<VulkanPlacedLocalCable>,
    pub endpoints: Vec<VulkanPlacedCableEndpoint>,
    pub local_cable_count: usize,
    pub incoming_endpoint_count: usize,
    pub outgoing_endpoint_count: usize,
    pub total_buffer_count: usize,
    pub total_endpoint_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_byte_cables: Vec<usize>,
}

impl VulkanPlacedCableIoPlan {
    pub fn from_placed_resident_plan(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
        let mut local_cables = Vec::with_capacity(placed_resident_plan.local_cables.len());
        for cable in &placed_resident_plan.local_cables {
            local_cables.push(VulkanPlacedLocalCable::from_cable(
                local_cables.len(),
                &placed_resident_plan.device_id,
                cable,
                placed_resident_plan.signal_element_bytes,
            )?);
        }

        let mut endpoints = Vec::with_capacity(
            placed_resident_plan.incoming_cables.len() + placed_resident_plan.outgoing_cables.len(),
        );

        for cable in &placed_resident_plan.incoming_cables {
            endpoints.push(VulkanPlacedCableEndpoint::from_cable(
                endpoints.len(),
                VulkanPlacedCableDirection::Incoming,
                &placed_resident_plan.device_id,
                cable,
                placed_resident_plan.signal_element_bytes,
            )?);
        }
        for cable in &placed_resident_plan.outgoing_cables {
            endpoints.push(VulkanPlacedCableEndpoint::from_cable(
                endpoints.len(),
                VulkanPlacedCableDirection::Outgoing,
                &placed_resident_plan.device_id,
                cable,
                placed_resident_plan.signal_element_bytes,
            )?);
        }

        let local_cable_count = local_cables.len();
        let incoming_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedCableDirection::Incoming)
            .count();
        let outgoing_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedCableDirection::Outgoing)
            .count();
        let unresolved_byte_cables = local_cables
            .iter()
            .filter(|cable| cable.byte_capacity.is_none())
            .map(|cable| cable.cable_index)
            .chain(
                endpoints
                    .iter()
                    .filter(|endpoint| endpoint.byte_capacity.is_none())
                    .map(|endpoint| endpoint.cable_index),
            )
            .collect::<Vec<_>>();
        let total_byte_capacity = local_cables
            .iter()
            .map(|cable| cable.byte_capacity)
            .chain(endpoints.iter().map(|endpoint| endpoint.byte_capacity))
            .try_fold(Some(0usize), |total, byte_capacity| {
                match (total, byte_capacity) {
                    (Some(total), Some(bytes)) => Some(total.checked_add(bytes).ok_or_else(|| {
                        VulkanPlacedCableIoPlanError(
                            "placed cable buffer byte capacity overflowed".to_string(),
                        )
                    }))
                    .transpose(),
                    _ => Ok(None),
                }
            })?;

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_resident_plan.device_id.clone(),
            signal_element_bytes: placed_resident_plan.signal_element_bytes,
            local_cables,
            local_cable_count,
            total_buffer_count: local_cable_count + endpoints.len(),
            total_endpoint_count: endpoints.len(),
            endpoints,
            incoming_endpoint_count,
            outgoing_endpoint_count,
            total_byte_capacity,
            unresolved_byte_cables,
        })
    }

    pub fn endpoint(
        &self,
        direction: VulkanPlacedCableDirection,
        cable_index: usize,
    ) -> Option<&VulkanPlacedCableEndpoint> {
        self.endpoints
            .iter()
            .find(|endpoint| endpoint.direction == direction && endpoint.cable_index == cable_index)
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanPlacedCableIoBuffers, VulkanError> {
        let mut local_buffers = Vec::with_capacity(self.local_cable_count);
        let mut incoming_buffers = Vec::with_capacity(self.incoming_endpoint_count);
        let mut outgoing_buffers = Vec::with_capacity(self.outgoing_endpoint_count);
        let mut total_byte_capacity = 0usize;

        for cable in &self.local_cables {
            let byte_capacity = cable.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} local cable {} has unknown byte capacity",
                    self.device_id, cable.cable_index
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "placed local cable buffer allocation",
            )?;
            local_buffers.push(VulkanPlacedLocalCableBufferAllocation {
                cable: cable.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        for endpoint in &self.endpoints {
            let byte_capacity = endpoint.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} endpoint {} for cable {} has unknown byte capacity",
                    self.device_id, endpoint.endpoint_id, endpoint.cable_index
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "placed cable endpoint buffer allocation",
            )?;
            let allocation = VulkanPlacedCableBufferAllocation {
                endpoint: endpoint.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            };
            match endpoint.direction {
                VulkanPlacedCableDirection::Incoming => incoming_buffers.push(allocation),
                VulkanPlacedCableDirection::Outgoing => outgoing_buffers.push(allocation),
            }
        }

        Ok(VulkanPlacedCableIoBuffers {
            plan: self.clone(),
            local_buffers,
            incoming_buffers,
            outgoing_buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedLocalCable {
    pub buffer_index: usize,
    pub cable_id: String,
    pub cable_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub device_id: String,
    pub source_pedal_id: String,
    pub source_port_id: String,
    pub source_pedal_port: Option<String>,
    pub destination_pedal_id: String,
    pub destination_port_id: String,
    pub destination_pedal_port: Option<String>,
    pub transport: CableTransport,
}

impl VulkanPlacedLocalCable {
    fn from_cable(
        buffer_index: usize,
        device_id: &str,
        cable: &PedalCablePlacement,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
        let CableTransport::LocalBuffer {
            device_id: transport_device_id,
        } = &cable.transport
        else {
            return Err(VulkanPlacedCableIoPlanError(format!(
                "cable {} is not a local cable",
                cable.cable_index
            )));
        };

        if transport_device_id != device_id
            || cable.source_device_id != device_id
            || cable.destination_device_id != device_id
        {
            return Err(VulkanPlacedCableIoPlanError(format!(
                "local cable {} is not fully resident on device {:?}",
                cable.cable_index, device_id
            )));
        }

        let element_count = cable_element_count(cable)?;
        let byte_capacity = cable_byte_capacity(cable, element_count, signal_element_bytes)?;

        Ok(Self {
            buffer_index,
            cable_id: format!("cable_{}_local", cable.cable_index),
            cable_index: cable.cable_index,
            signal: cable.signal.clone(),
            shape: cable.shape.clone(),
            element_count,
            byte_capacity,
            device_id: device_id.to_string(),
            source_pedal_id: cable.source_pedal_id.clone(),
            source_port_id: cable.source_port_id.clone(),
            source_pedal_port: cable.source_pedal_port.clone(),
            destination_pedal_id: cable.destination_pedal_id.clone(),
            destination_port_id: cable.destination_port_id.clone(),
            destination_pedal_port: cable.destination_pedal_port.clone(),
            transport: cable.transport.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableEndpoint {
    pub endpoint_index: usize,
    pub endpoint_id: String,
    pub direction: VulkanPlacedCableDirection,
    pub cable_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub local_device_id: String,
    pub remote_device_id: String,
    pub local_pedal_id: String,
    pub remote_pedal_id: String,
    pub local_port_id: String,
    pub remote_port_id: String,
    pub local_pedal_port: Option<String>,
    pub remote_pedal_port: Option<String>,
    pub transport: CableTransport,
}

impl VulkanPlacedCableEndpoint {
    fn from_cable(
        endpoint_index: usize,
        direction: VulkanPlacedCableDirection,
        device_id: &str,
        cable: &PedalCablePlacement,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
        let CableTransport::CrossDevice {
            from_device_id,
            to_device_id,
        } = &cable.transport
        else {
            return Err(VulkanPlacedCableIoPlanError(format!(
                "cable {} is not a cross-device cable",
                cable.cable_index
            )));
        };

        match direction {
            VulkanPlacedCableDirection::Incoming => {
                if to_device_id != device_id || cable.destination_device_id != device_id {
                    return Err(VulkanPlacedCableIoPlanError(format!(
                        "incoming cable {} does not terminate on device {:?}",
                        cable.cable_index, device_id
                    )));
                }
            }
            VulkanPlacedCableDirection::Outgoing => {
                if from_device_id != device_id || cable.source_device_id != device_id {
                    return Err(VulkanPlacedCableIoPlanError(format!(
                        "outgoing cable {} does not originate on device {:?}",
                        cable.cable_index, device_id
                    )));
                }
            }
        }

        let element_count = cable_element_count(cable)?;
        let byte_capacity = cable_byte_capacity(cable, element_count, signal_element_bytes)?;

        let (
            local_device_id,
            remote_device_id,
            local_pedal_id,
            remote_pedal_id,
            local_port_id,
            remote_port_id,
            local_pedal_port,
            remote_pedal_port,
        ) = match direction {
            VulkanPlacedCableDirection::Incoming => (
                cable.destination_device_id.clone(),
                cable.source_device_id.clone(),
                cable.destination_pedal_id.clone(),
                cable.source_pedal_id.clone(),
                cable.destination_port_id.clone(),
                cable.source_port_id.clone(),
                cable.destination_pedal_port.clone(),
                cable.source_pedal_port.clone(),
            ),
            VulkanPlacedCableDirection::Outgoing => (
                cable.source_device_id.clone(),
                cable.destination_device_id.clone(),
                cable.source_pedal_id.clone(),
                cable.destination_pedal_id.clone(),
                cable.source_port_id.clone(),
                cable.destination_port_id.clone(),
                cable.source_pedal_port.clone(),
                cable.destination_pedal_port.clone(),
            ),
        };
        let direction_suffix = match direction {
            VulkanPlacedCableDirection::Incoming => "in",
            VulkanPlacedCableDirection::Outgoing => "out",
        };

        Ok(Self {
            endpoint_index,
            endpoint_id: format!("cable_{}_{}", cable.cable_index, direction_suffix),
            direction,
            cable_index: cable.cable_index,
            signal: cable.signal.clone(),
            shape: cable.shape.clone(),
            element_count,
            byte_capacity,
            local_device_id,
            remote_device_id,
            local_pedal_id,
            remote_pedal_id,
            local_port_id,
            remote_port_id,
            local_pedal_port,
            remote_pedal_port,
            transport: cable.transport.clone(),
        })
    }
}

fn cable_element_count(cable: &PedalCablePlacement) -> Result<usize, VulkanPlacedCableIoPlanError> {
    let element_count = product(&cable.shape).ok_or_else(|| {
        VulkanPlacedCableIoPlanError(format!(
            "cable {} signal shape {:?} overflows",
            cable.cable_index, cable.shape
        ))
    })?;
    if element_count == 0 {
        return Err(VulkanPlacedCableIoPlanError(format!(
            "cable {} signal shape {:?} has zero elements",
            cable.cable_index, cable.shape
        )));
    }
    Ok(element_count)
}

fn cable_byte_capacity(
    cable: &PedalCablePlacement,
    element_count: usize,
    signal_element_bytes: Option<usize>,
) -> Result<Option<usize>, VulkanPlacedCableIoPlanError> {
    match signal_element_bytes {
        Some(bytes_per_element) => element_count
            .checked_mul(bytes_per_element)
            .map(Some)
            .ok_or_else(|| {
                VulkanPlacedCableIoPlanError(format!(
                    "cable {} byte capacity overflowed",
                    cable.cable_index
                ))
            }),
        None => Ok(None),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanPlacedCableDirection {
    Incoming,
    Outgoing,
}

pub struct VulkanPlacedCableIoBuffers {
    pub plan: VulkanPlacedCableIoPlan,
    pub local_buffers: Vec<VulkanPlacedLocalCableBufferAllocation>,
    pub incoming_buffers: Vec<VulkanPlacedCableBufferAllocation>,
    pub outgoing_buffers: Vec<VulkanPlacedCableBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPlacedCableIoBuffers {
    pub fn local_buffer(
        &self,
        cable_index: usize,
    ) -> Option<(usize, &VulkanPlacedLocalCableBufferAllocation)> {
        self.local_buffers
            .iter()
            .enumerate()
            .find(|(_, buffer)| buffer.cable.cable_index == cable_index)
    }

    pub fn local_cable_buffer(
        &self,
        cable_index: usize,
    ) -> Option<&VulkanPlacedLocalCableBufferAllocation> {
        self.local_buffers
            .iter()
            .find(|buffer| buffer.cable.cable_index == cable_index)
    }

    pub fn buffer(
        &self,
        direction: VulkanPlacedCableDirection,
        cable_index: usize,
    ) -> Option<(usize, &VulkanPlacedCableBufferAllocation)> {
        match direction {
            VulkanPlacedCableDirection::Incoming => self
                .incoming_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.cable_index == cable_index),
            VulkanPlacedCableDirection::Outgoing => self
                .outgoing_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.cable_index == cable_index),
        }
    }

    pub fn incoming_buffer(
        &self,
        cable_index: usize,
    ) -> Option<&VulkanPlacedCableBufferAllocation> {
        self.incoming_buffers
            .iter()
            .find(|buffer| buffer.endpoint.cable_index == cable_index)
    }

    pub fn outgoing_buffer(
        &self,
        cable_index: usize,
    ) -> Option<&VulkanPlacedCableBufferAllocation> {
        self.outgoing_buffers
            .iter()
            .find(|buffer| buffer.endpoint.cable_index == cable_index)
    }
}

pub struct VulkanPlacedLocalCableBufferAllocation {
    pub cable: VulkanPlacedLocalCable,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

pub struct VulkanPlacedCableBufferAllocation {
    pub endpoint: VulkanPlacedCableEndpoint,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableIoPlanError(pub String);

impl Display for VulkanPlacedCableIoPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPlacedCableIoPlanError {}

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
        Self::from_plans_with_hosted_pedals(execution_plan, resource_plan, resident_plan, None)
    }

    pub fn from_placed_resident_plan(
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanBindingPlanError> {
        let hosted_pedals = placed_resident_plan
            .hosted_pedal_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        Self::from_plans_with_hosted_pedals(
            execution_plan,
            resource_plan,
            &placed_resident_plan.resident_plan,
            Some(&hosted_pedals),
        )
    }

    fn from_plans_with_hosted_pedals(
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        resident_plan: &VulkanStreamCircuitResidentPlan,
        hosted_pedals: Option<&BTreeSet<String>>,
    ) -> Result<Self, VulkanBindingPlanError> {
        let hosts_pedal = |pedal_id: &str| {
            hosted_pedals
                .map(|pedals| pedals.contains(pedal_id))
                .unwrap_or(true)
        };
        let hosted_circuit_count = execution_plan
            .circuits
            .iter()
            .filter(|circuit| hosts_pedal(&circuit.pedal_id))
            .count();

        if hosted_pedals.is_none()
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
            parameter_binding_index(resource_plan, resident_plan, hosted_pedals)?;
        let state_bindings = state_binding_index(resident_plan)?;
        let activation_bindings = activation_binding_index(resident_plan)?;

        let circuits = execution_plan
            .circuits
            .iter()
            .filter(|circuit| hosts_pedal(&circuit.pedal_id))
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCircuitBindingPlan {
    pub pedal_id: String,
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
    pub shape: Option<Vec<usize>>,
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

    pub fn link_reusable_kernels(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> VulkanLinkedReusableKernelPlan {
        self.reusable_kernel_plan.link_artifacts(manifest)
    }

    pub fn descriptor_resource_plan(
        &self,
    ) -> Result<VulkanDescriptorResourcePlan, VulkanDescriptorResourcePlanError> {
        VulkanDescriptorResourcePlan::from_plans(
            &self.dispatch_plan,
            &self.resident_plan,
            self.buffers.dynamic_state_capacity_activations,
        )
    }

    pub fn prepared_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanPreparedDispatchPlan, VulkanPreparedDispatchPlanError> {
        let descriptor_plan = self
            .descriptor_resource_plan()
            .map_err(VulkanPreparedDispatchPlanError::DescriptorResource)?;
        VulkanPreparedDispatchPlan::from_plans(
            &self.dispatch_plan,
            &self.reusable_kernel_plan,
            &descriptor_plan,
            manifest,
        )
    }

    pub fn bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let prepared_plan = self
            .prepared_dispatch_plan(manifest)
            .map_err(VulkanBoundDispatchPlanError::PreparedDispatch)?;
        VulkanBoundDispatchPlan::from_prepared_plan(&prepared_plan, &self.buffers)
    }
}

#[derive(Debug)]
pub enum VulkanStreamCircuitMountError {
    Binding(VulkanBindingPlanError),
    BoundaryIo(VulkanModelBoundaryBufferPlanError),
    CableIo(VulkanPlacedCableIoPlanError),
    PermanentParameters(VulkanPermanentParameterBufferPlanError),
    Vulkan(VulkanError),
}

impl Display for VulkanStreamCircuitMountError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(error) => Display::fmt(error, f),
            Self::BoundaryIo(error) => Display::fmt(error, f),
            Self::CableIo(error) => Display::fmt(error, f),
            Self::PermanentParameters(error) => Display::fmt(error, f),
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

impl From<VulkanModelBoundaryBufferPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanModelBoundaryBufferPlanError) -> Self {
        Self::BoundaryIo(error)
    }
}

impl From<VulkanPlacedCableIoPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanPlacedCableIoPlanError) -> Self {
        Self::CableIo(error)
    }
}

impl From<VulkanPermanentParameterBufferPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanPermanentParameterBufferPlanError) -> Self {
        Self::PermanentParameters(error)
    }
}

impl From<VulkanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

pub struct VulkanMountedPlacedStreamCircuit {
    pub placed_plan: VulkanPlacedStreamCircuitPlan,
    pub parameter_buffers: VulkanPermanentParameterBuffers,
    pub buffers: VulkanStreamCircuitStreamBuffers,
    pub boundary_io: VulkanModelBoundaryBuffers,
    pub cable_io: VulkanPlacedCableIoBuffers,
}

impl VulkanMountedPlacedStreamCircuit {
    pub fn from_placed_plan(
        device: &VulkanComputeDevice,
        placed_plan: VulkanPlacedStreamCircuitPlan,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        let buffers = placed_plan
            .placed_resident_plan
            .resident_plan
            .allocate_stream_buffers(device, dynamic_state_capacity_activations)?;
        let parameter_buffer_plan = VulkanPermanentParameterBufferPlan::from_placed_resident_plan(
            &placed_plan.placed_resident_plan,
        )?;
        let parameter_buffers = parameter_buffer_plan.allocate_buffers(device)?;
        let boundary_io_plan = VulkanModelBoundaryBufferPlan::from_placed_plan(&placed_plan)?;
        let boundary_io = boundary_io_plan.allocate_buffers(device)?;
        let cable_io_plan =
            VulkanPlacedCableIoPlan::from_placed_resident_plan(&placed_plan.placed_resident_plan)?;
        let cable_io = cable_io_plan.allocate_buffers(device)?;
        Ok(Self {
            placed_plan,
            parameter_buffers,
            buffers,
            boundary_io,
            cable_io,
        })
    }

    pub fn can_execute(&self) -> bool {
        false
    }

    pub fn device_id(&self) -> &str {
        &self.placed_plan.device_id
    }

    pub fn descriptor_resource_plan(
        &self,
    ) -> Result<VulkanDescriptorResourcePlan, VulkanDescriptorResourcePlanError> {
        VulkanDescriptorResourcePlan::from_plans(
            &self.placed_plan.dispatch_plan,
            &self.placed_plan.placed_resident_plan.resident_plan,
            self.buffers.dynamic_state_capacity_activations,
        )
    }

    pub fn prepared_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanPreparedDispatchPlan, VulkanPreparedDispatchPlanError> {
        let descriptor_plan = self
            .descriptor_resource_plan()
            .map_err(VulkanPreparedDispatchPlanError::DescriptorResource)?;
        VulkanPreparedDispatchPlan::from_plans(
            &self.placed_plan.dispatch_plan,
            &self.placed_plan.reusable_kernel_plan,
            &descriptor_plan,
            manifest,
        )
    }

    pub fn bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let prepared_plan = self
            .prepared_dispatch_plan(manifest)
            .map_err(VulkanBoundDispatchPlanError::PreparedDispatch)?;
        VulkanBoundDispatchPlan::from_prepared_plan(&prepared_plan, &self.buffers)
    }

    pub fn placed_bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanPlacedBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let bound_plan = self.bound_dispatch_plan(manifest)?;
        Ok(VulkanPlacedBoundDispatchPlan::from_bound_plan(
            &bound_plan,
            &self.placed_plan.placed_resident_plan,
        ))
    }

    pub fn mounted_placed_bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanMountedPlacedBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let placed_bound_plan = self.placed_bound_dispatch_plan(manifest)?;
        VulkanMountedPlacedBoundDispatchPlan::from_placed_bound_plan(
            &placed_bound_plan,
            &self.cable_io,
        )
    }

    pub fn stream_tick_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanMountedPlacedStreamTickPlan, VulkanBoundDispatchPlanError> {
        let mounted_bound_plan = self.mounted_placed_bound_dispatch_plan(manifest)?;
        Ok(VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(
            &mounted_bound_plan,
        ))
    }

    pub fn advance_stream_tick(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
        stream_tick: u64,
    ) -> Result<VulkanMountedPlacedStreamTickRun, VulkanMountedPlacedStreamTickError> {
        let tick_plan = self.stream_tick_plan(manifest)?;
        Ok(tick_plan.advance(stream_tick))
    }

    pub fn resident_kernel_dispatch_readiness_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<VulkanMountedPlacedResidentKernelDispatchReadinessPlan, VulkanBoundDispatchPlanError>
    {
        let mounted_bound_plan = self.mounted_placed_bound_dispatch_plan(manifest)?;
        Ok(
            VulkanMountedPlacedResidentKernelDispatchReadinessPlan::from_mounted_bound_plan(
                self,
                &mounted_bound_plan,
                loaded_manifest,
            ),
        )
    }

    pub fn resident_kernel_dispatch_readiness_for_bound_dispatch(
        &self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> VulkanMountedPlacedResidentKernelDispatchStatus {
        if loaded_manifest
            .artifact(&dispatch.reusable_family_id)
            .is_none()
        {
            return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked {
                error: VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                    dispatch_index: dispatch.dispatch_index,
                    family_id: dispatch.reusable_family_id.clone(),
                },
            };
        }

        let (descriptor_count, workgroup_count_x) =
            match self.resident_kernel_buffer_bindings_for_bound_dispatch(dispatch) {
                Ok(bindings) => {
                    let workgroup_count_x =
                        match resident_kernel_dispatch_workgroup_count_x(dispatch, &bindings) {
                            Ok(workgroup_count_x) => workgroup_count_x,
                            Err(error) => {
                                return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked {
                                    error,
                                };
                            }
                        };
                    (bindings.len(), workgroup_count_x)
                }
                Err(error) => {
                    return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked { error };
                }
            };
        let push_constant_byte_count = match push_constant_byte_count(&dispatch.push_constants) {
            Ok(bytes) => bytes,
            Err(error) => {
                return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked { error };
            }
        };

        VulkanMountedPlacedResidentKernelDispatchStatus::Instantiable {
            descriptor_count,
            workgroup_count_x,
            local_size_x: dispatch.local_size_x,
            push_constant_byte_count,
        }
    }

    pub fn resident_kernel_buffer_bindings_for_bound_dispatch<'a>(
        &'a self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
    ) -> Result<
        Vec<VulkanResidentKernelBufferBinding<'a>>,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        dispatch
            .descriptors
            .iter()
            .map(|descriptor| self.resident_kernel_buffer_binding(dispatch, descriptor))
            .collect()
    }

    pub fn create_resident_kernel_dispatch_for_bound_dispatch(
        &self,
        device: &VulkanComputeDevice,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<VulkanResidentKernelDispatch, VulkanMountedPlacedResidentKernelDispatchError> {
        let artifact = loaded_manifest
            .artifact(&dispatch.reusable_family_id)
            .ok_or_else(|| {
                VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                    dispatch_index: dispatch.dispatch_index,
                    family_id: dispatch.reusable_family_id.clone(),
                }
            })?;
        let buffer_bindings = self.resident_kernel_buffer_bindings_for_bound_dispatch(dispatch)?;
        let workgroup_count_x =
            resident_kernel_dispatch_workgroup_count_x(dispatch, &buffer_bindings)?;
        device
            .create_resident_kernel_dispatch(
                &artifact.words,
                &buffer_bindings,
                workgroup_count_x,
                dispatch.local_size_x,
                push_constant_byte_count(&dispatch.push_constants)?,
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }

    pub fn create_resident_pedal_runner(
        &self,
        device: &VulkanComputeDevice,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        pedal_id: &str,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<
        VulkanMountedPlacedResidentPedalRunner,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        VulkanMountedPlacedResidentPedalRunner::from_mounted_bound_plan(
            device,
            self,
            mounted_bound_plan,
            pedal_id,
            loaded_manifest,
        )
    }

    pub fn create_resident_pedalboard_runner<I, S>(
        &self,
        device: &VulkanComputeDevice,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        pedal_ids: I,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<
        VulkanMountedPlacedResidentPedalboardRunner,
        VulkanMountedPlacedResidentKernelDispatchError,
    >
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        VulkanMountedPlacedResidentPedalboardRunner::from_mounted_bound_plan(
            device,
            self,
            mounted_bound_plan,
            pedal_ids,
            loaded_manifest,
        )
    }

    fn resident_kernel_buffer_binding<'a>(
        &'a self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        descriptor: &VulkanMountedPlacedBoundDescriptor,
    ) -> Result<VulkanResidentKernelBufferBinding<'a>, VulkanMountedPlacedResidentKernelDispatchError>
    {
        let binding = u32::try_from(descriptor.binding).map_err(|_| {
            VulkanMountedPlacedResidentKernelDispatchError::DescriptorBindingOverflow {
                dispatch_index: dispatch.dispatch_index,
                binding: descriptor.binding,
            }
        })?;
        let (buffer, byte_len) = match &descriptor.target {
            VulkanMountedPlacedBoundDescriptorTarget::Resident { target } => {
                self.resident_kernel_buffer_for_resident_target(dispatch, descriptor, target)?
            }
            VulkanMountedPlacedBoundDescriptorTarget::ModelInput { signal_id } => {
                let allocation = self.boundary_io.input_buffer(signal_id).ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingModelBoundaryBuffer {
                        dispatch_index: dispatch.dispatch_index,
                        binding: descriptor.binding,
                        direction: VulkanModelBoundaryDirection::Input,
                        signal_id: signal_id.clone(),
                    }
                })?;
                (&allocation.buffer, allocation.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
                let allocation = self.boundary_io.output_buffer(signal_id).ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingModelBoundaryBuffer {
                        dispatch_index: dispatch.dispatch_index,
                        binding: descriptor.binding,
                        direction: VulkanModelBoundaryDirection::Output,
                        signal_id: signal_id.clone(),
                    }
                })?;
                (&allocation.buffer, allocation.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer { cable }
            | VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer { cable } => {
                let allocation = self
                    .cable_io
                    .local_buffers
                    .get(cable.buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "local_cable".to_string(),
                            buffer_index: cable.buffer_index,
                        }
                    })?;
                (&allocation.buffer, cable.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer { endpoint } => {
                let allocation = self
                    .cable_io
                    .incoming_buffers
                    .get(endpoint.buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "incoming_cable".to_string(),
                            buffer_index: endpoint.buffer_index,
                        }
                    })?;
                (&allocation.buffer, endpoint.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer { endpoint } => {
                let allocation = self
                    .cable_io
                    .outgoing_buffers
                    .get(endpoint.buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "outgoing_cable".to_string(),
                            buffer_index: endpoint.buffer_index,
                        }
                    })?;
                (&allocation.buffer, endpoint.byte_capacity)
            }
        };

        Ok(VulkanResidentKernelBufferBinding::new(
            binding, buffer, byte_len,
        ))
    }

    fn resident_kernel_buffer_for_resident_target<'a>(
        &'a self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        descriptor: &VulkanMountedPlacedBoundDescriptor,
        target: &VulkanBoundDescriptorTarget,
    ) -> Result<(&'a VulkanResidentBuffer, usize), VulkanMountedPlacedResidentKernelDispatchError>
    {
        match target {
            VulkanBoundDescriptorTarget::PermanentParameter {
                param_id,
                tensor,
                byte_count,
            } => {
                let allocation = self.parameter_buffers.parameter_buffer(tensor).ok_or_else(
                    || {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingPermanentParameterBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            param_id: param_id.clone(),
                            tensor: tensor.clone(),
                            byte_count: *byte_count,
                        }
                    },
                )?;
                Ok((&allocation.buffer, allocation.byte_capacity))
            }
            VulkanBoundDescriptorTarget::BoundaryInput { signal_id }
            | VulkanBoundDescriptorTarget::BoundaryOutput { signal_id } => Err(
                VulkanMountedPlacedResidentKernelDispatchError::ModelBoundaryBufferUnavailable {
                    dispatch_index: dispatch.dispatch_index,
                    binding: descriptor.binding,
                    signal_id: signal_id.clone(),
                },
            ),
            VulkanBoundDescriptorTarget::ActivationSlot {
                buffer_index,
                byte_capacity,
                ..
            } => {
                let allocation = self
                    .buffers
                    .activation_slot_buffers
                    .get(*buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "activation_slot".to_string(),
                            buffer_index: *buffer_index,
                        }
                    })?;
                Ok((&allocation.buffer, *byte_capacity))
            }
            VulkanBoundDescriptorTarget::StreamStateBuffer {
                buffer_index,
                byte_capacity,
                ..
            }
            | VulkanBoundDescriptorTarget::StreamStateView {
                buffer_index,
                byte_capacity,
                ..
            } => {
                let allocation =
                    self.buffers
                        .state_buffers
                        .get(*buffer_index)
                        .ok_or_else(|| {
                            VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                                dispatch_index: dispatch.dispatch_index,
                                binding: descriptor.binding,
                                buffer_kind: "stream_state".to_string(),
                                buffer_index: *buffer_index,
                            }
                        })?;
                Ok((&allocation.buffer, *byte_capacity))
            }
        }
    }
}

pub struct VulkanResidentInputEmbeddingTransducerRunner {
    pub transducer_id: String,
    pub parameter_tensor: String,
    pub output_signal_id: String,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
    resident_dispatch: VulkanResidentKernelDispatch,
}

impl VulkanResidentInputEmbeddingTransducerRunner {
    pub fn from_mounted_lfm2_token_embedding(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transducer_parameter_buffers: &VulkanPermanentParameterBuffers,
        spirv_words: &[u32],
    ) -> Result<Self, VulkanResidentInputEmbeddingTransducerRunnerError> {
        let embedding_weight = transducer_parameter_buffers
            .parameter_buffer(LFM2_EMBED_TOKENS_TENSOR)
            .ok_or_else(|| {
                VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                    tensor: LFM2_EMBED_TOKENS_TENSOR.to_string(),
                }
            })?;
        validate_lfm2_embedding_weight(embedding_weight)?;

        let output_frame = mounted
            .boundary_io
            .input_buffer(LFM2_INPUT_FRAME_SIGNAL)
            .ok_or_else(|| {
                VulkanResidentInputEmbeddingTransducerRunnerError::MissingModelInputBuffer {
                    signal_id: LFM2_INPUT_FRAME_SIGNAL.to_string(),
                }
            })?;
        if output_frame.byte_capacity != LFM2_FRAME_BYTES {
            return Err(
                VulkanResidentInputEmbeddingTransducerRunnerError::InvalidOutputFrameByteCapacity {
                    signal_id: LFM2_INPUT_FRAME_SIGNAL.to_string(),
                    byte_capacity: output_frame.byte_capacity,
                    expected_byte_capacity: LFM2_FRAME_BYTES,
                },
            );
        }

        let bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                &embedding_weight.buffer,
                embedding_weight.byte_capacity,
            ),
            VulkanResidentKernelBufferBinding::new(
                1,
                &output_frame.buffer,
                output_frame.byte_capacity,
            ),
        ];
        let workgroup_count_x = u32::try_from(
            LFM2_FRAME_WORDS.div_ceil(VULKAN_INPUT_EMBEDDING_LOOKUP_LOCAL_SIZE_X as usize),
        )
        .map_err(|_| VulkanResidentInputEmbeddingTransducerRunnerError::WorkgroupCountOverflow)?;
        let resident_dispatch = device.create_resident_kernel_dispatch(
            spirv_words,
            &bindings,
            workgroup_count_x,
            VULKAN_INPUT_EMBEDDING_LOOKUP_LOCAL_SIZE_X,
            std::mem::size_of::<u32>() as u32,
        )?;

        Ok(Self {
            transducer_id: LFM2_TOKEN_EMBEDDING_TRANSDUCER_ID.to_string(),
            parameter_tensor: LFM2_EMBED_TOKENS_TENSOR.to_string(),
            output_signal_id: LFM2_INPUT_FRAME_SIGNAL.to_string(),
            descriptor_count: resident_dispatch.descriptor_count(),
            workgroup_count_x: resident_dispatch.workgroup_count_x(),
            push_constant_byte_count: resident_dispatch.push_constant_byte_count(),
            resident_dispatch,
        })
    }

    pub fn run_token_id(
        &self,
        device: &VulkanComputeDevice,
        token_id: u32,
    ) -> Result<
        VulkanResidentInputEmbeddingTransducerRun,
        VulkanResidentInputEmbeddingTransducerRunnerError,
    > {
        device.run_resident_kernel_dispatch(&self.resident_dispatch, &token_id.to_le_bytes())?;
        Ok(VulkanResidentInputEmbeddingTransducerRun {
            transducer_id: self.transducer_id.clone(),
            token_id,
            output_signal_id: self.output_signal_id.clone(),
            dispatch_count: 1,
            descriptor_count: self.descriptor_count,
            workgroup_count_x: self.workgroup_count_x,
            push_constant_byte_count: self.push_constant_byte_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInputEmbeddingTransducerRun {
    pub transducer_id: String,
    pub token_id: u32,
    pub output_signal_id: String,
    pub dispatch_count: usize,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
}

#[derive(Debug)]
pub enum VulkanResidentInputEmbeddingTransducerRunnerError {
    MissingTransducerParameterBuffer {
        tensor: String,
    },
    InvalidEmbeddingWeight {
        tensor: String,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        byte_capacity: usize,
    },
    MissingModelInputBuffer {
        signal_id: String,
    },
    InvalidOutputFrameByteCapacity {
        signal_id: String,
        byte_capacity: usize,
        expected_byte_capacity: usize,
    },
    WorkgroupCountOverflow,
    Vulkan(VulkanError),
}

impl Display for VulkanResidentInputEmbeddingTransducerRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTransducerParameterBuffer { tensor } => {
                write!(
                    f,
                    "missing transducer parameter buffer for tensor {tensor:?}"
                )
            }
            Self::InvalidEmbeddingWeight {
                tensor,
                dtype,
                shape,
                byte_capacity,
            } => write!(
                f,
                "transducer embedding tensor {tensor:?} has dtype {dtype:?}, shape {shape:?}, and {byte_capacity} bytes"
            ),
            Self::MissingModelInputBuffer { signal_id } => {
                write!(f, "missing model input boundary buffer {signal_id:?}")
            }
            Self::InvalidOutputFrameByteCapacity {
                signal_id,
                byte_capacity,
                expected_byte_capacity,
            } => write!(
                f,
                "model input boundary buffer {signal_id:?} has {byte_capacity} bytes, expected {expected_byte_capacity}"
            ),
            Self::WorkgroupCountOverflow => {
                f.write_str("input embedding transducer workgroup count overflowed")
            }
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentInputEmbeddingTransducerRunnerError {}

impl From<VulkanError> for VulkanResidentInputEmbeddingTransducerRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

fn validate_lfm2_embedding_weight(
    allocation: &VulkanPermanentParameterBufferAllocation,
) -> Result<(), VulkanResidentInputEmbeddingTransducerRunnerError> {
    if allocation.parameter.dtype.as_deref() != Some("BF16")
        || allocation.parameter.shape.as_deref() != Some(&[LFM2_VOCAB_SIZE, LFM2_HIDDEN_SIZE])
        || allocation.byte_capacity != LFM2_EMBED_TOKENS_BYTES
    {
        return Err(
            VulkanResidentInputEmbeddingTransducerRunnerError::InvalidEmbeddingWeight {
                tensor: allocation.parameter.tensor.clone(),
                dtype: allocation.parameter.dtype.clone(),
                shape: allocation.parameter.shape.clone(),
                byte_capacity: allocation.byte_capacity,
            },
        );
    }
    Ok(())
}

pub struct VulkanResidentOutputTransducerRunner {
    pub transducer_id: String,
    pub input_signal_id: String,
    pub logits_byte_capacity: usize,
    pub dispatch_count: usize,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    normalized_frame_buffer: VulkanResidentBuffer,
    logits_buffer: VulkanResidentBuffer,
    embedding_norm_dispatch: VulkanResidentKernelDispatch,
    tied_projection_dispatch: VulkanResidentKernelDispatch,
}

impl VulkanResidentOutputTransducerRunner {
    pub fn from_mounted_lfm2_output_transducer(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transducer_parameter_buffers: &VulkanPermanentParameterBuffers,
        embedding_norm_spirv_words: &[u32],
        tied_projection_spirv_words: &[u32],
    ) -> Result<Self, VulkanResidentOutputTransducerRunnerError> {
        let output_frame = mounted
            .boundary_io
            .output_buffer(LFM2_OUTPUT_FRAME_SIGNAL)
            .ok_or_else(
                || VulkanResidentOutputTransducerRunnerError::MissingModelOutputBuffer {
                    signal_id: LFM2_OUTPUT_FRAME_SIGNAL.to_string(),
                },
            )?;
        if output_frame.byte_capacity != LFM2_FRAME_BYTES {
            return Err(
                VulkanResidentOutputTransducerRunnerError::InvalidInputFrameByteCapacity {
                    signal_id: LFM2_OUTPUT_FRAME_SIGNAL.to_string(),
                    byte_capacity: output_frame.byte_capacity,
                    expected_byte_capacity: LFM2_FRAME_BYTES,
                },
            );
        }

        let embedding_norm_weight = transducer_parameter_buffers
            .parameter_buffer(LFM2_EMBEDDING_NORM_TENSOR)
            .ok_or_else(|| {
                VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                    tensor: LFM2_EMBEDDING_NORM_TENSOR.to_string(),
                }
            })?;
        validate_lfm2_embedding_norm_weight(embedding_norm_weight)?;
        let embedding_weight = transducer_parameter_buffers
            .parameter_buffer(LFM2_EMBED_TOKENS_TENSOR)
            .ok_or_else(|| {
                VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                    tensor: LFM2_EMBED_TOKENS_TENSOR.to_string(),
                }
            })?;
        validate_lfm2_output_embedding_weight(embedding_weight)?;

        let normalized_frame_buffer = device.create_resident_buffer(LFM2_FRAME_BYTES)?;
        let logits_buffer = device.create_resident_buffer(LFM2_LOGITS_BYTES)?;

        let embedding_norm_bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                &output_frame.buffer,
                output_frame.byte_capacity,
            ),
            VulkanResidentKernelBufferBinding::new(1, &normalized_frame_buffer, LFM2_FRAME_BYTES),
            VulkanResidentKernelBufferBinding::new(
                2,
                &embedding_norm_weight.buffer,
                embedding_norm_weight.byte_capacity,
            ),
        ];
        let embedding_norm_dispatch = device.create_resident_kernel_dispatch(
            embedding_norm_spirv_words,
            &embedding_norm_bindings,
            1,
            DEFAULT_COMPUTE_LOCAL_SIZE_X,
            0,
        )?;

        let projection_workgroup_count_x =
            u32::try_from(LFM2_VOCAB_SIZE.div_ceil(VULKAN_OUTPUT_PROJECTION_LOCAL_SIZE_X as usize))
                .map_err(|_| VulkanResidentOutputTransducerRunnerError::WorkgroupCountOverflow)?;
        let tied_projection_bindings = [
            VulkanResidentKernelBufferBinding::new(0, &normalized_frame_buffer, LFM2_FRAME_BYTES),
            VulkanResidentKernelBufferBinding::new(
                1,
                &embedding_weight.buffer,
                embedding_weight.byte_capacity,
            ),
            VulkanResidentKernelBufferBinding::new(2, &logits_buffer, LFM2_LOGITS_BYTES),
        ];
        let tied_projection_dispatch = device.create_resident_kernel_dispatch(
            tied_projection_spirv_words,
            &tied_projection_bindings,
            projection_workgroup_count_x,
            VULKAN_OUTPUT_PROJECTION_LOCAL_SIZE_X,
            0,
        )?;

        let total_descriptor_count = embedding_norm_dispatch
            .descriptor_count()
            .checked_add(tied_projection_dispatch.descriptor_count())
            .ok_or(VulkanResidentOutputTransducerRunnerError::DescriptorCountOverflow)?;
        let total_push_constant_byte_count = embedding_norm_dispatch
            .push_constant_byte_count()
            .checked_add(tied_projection_dispatch.push_constant_byte_count())
            .ok_or(VulkanResidentOutputTransducerRunnerError::PushConstantByteCountOverflow)?;

        Ok(Self {
            transducer_id: "output_transducer".to_string(),
            input_signal_id: LFM2_OUTPUT_FRAME_SIGNAL.to_string(),
            logits_byte_capacity: LFM2_LOGITS_BYTES,
            dispatch_count: 2,
            total_descriptor_count,
            total_push_constant_byte_count,
            normalized_frame_buffer,
            logits_buffer,
            embedding_norm_dispatch,
            tied_projection_dispatch,
        })
    }

    pub fn run(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentOutputTransducerRun, VulkanResidentOutputTransducerRunnerError> {
        device.run_resident_kernel_dispatch(&self.embedding_norm_dispatch, &[])?;
        device.run_resident_kernel_dispatch(&self.tied_projection_dispatch, &[])?;
        Ok(VulkanResidentOutputTransducerRun {
            transducer_id: self.transducer_id.clone(),
            input_signal_id: self.input_signal_id.clone(),
            dispatch_count: self.dispatch_count,
            node_ids: vec![
                LFM2_OUTPUT_EMBEDDING_NORM_TRANSDUCER_ID.to_string(),
                LFM2_TIED_OUTPUT_PROJECTION_TRANSDUCER_ID.to_string(),
            ],
            descriptor_counts: vec![
                self.embedding_norm_dispatch.descriptor_count(),
                self.tied_projection_dispatch.descriptor_count(),
            ],
            workgroup_counts_x: vec![
                self.embedding_norm_dispatch.workgroup_count_x(),
                self.tied_projection_dispatch.workgroup_count_x(),
            ],
            push_constant_byte_counts: vec![
                self.embedding_norm_dispatch.push_constant_byte_count(),
                self.tied_projection_dispatch.push_constant_byte_count(),
            ],
            logits_byte_capacity: self.logits_byte_capacity,
        })
    }

    pub fn read_logits_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.logits_buffer.read_bytes(len)
    }

    pub fn logits_buffer(&self) -> &VulkanResidentBuffer {
        &self.logits_buffer
    }

    pub fn read_normalized_frame_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.normalized_frame_buffer.read_bytes(len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentOutputTransducerRun {
    pub transducer_id: String,
    pub input_signal_id: String,
    pub dispatch_count: usize,
    pub node_ids: Vec<String>,
    pub descriptor_counts: Vec<usize>,
    pub workgroup_counts_x: Vec<u32>,
    pub push_constant_byte_counts: Vec<u32>,
    pub logits_byte_capacity: usize,
}

#[derive(Debug)]
pub enum VulkanResidentOutputTransducerRunnerError {
    MissingTransducerParameterBuffer {
        tensor: String,
    },
    InvalidEmbeddingWeight {
        tensor: String,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        byte_capacity: usize,
    },
    InvalidEmbeddingNormWeight {
        tensor: String,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        byte_capacity: usize,
    },
    MissingModelOutputBuffer {
        signal_id: String,
    },
    InvalidInputFrameByteCapacity {
        signal_id: String,
        byte_capacity: usize,
        expected_byte_capacity: usize,
    },
    WorkgroupCountOverflow,
    DescriptorCountOverflow,
    PushConstantByteCountOverflow,
    Vulkan(VulkanError),
}

impl Display for VulkanResidentOutputTransducerRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTransducerParameterBuffer { tensor } => {
                write!(
                    f,
                    "missing transducer parameter buffer for tensor {tensor:?}"
                )
            }
            Self::InvalidEmbeddingWeight {
                tensor,
                dtype,
                shape,
                byte_capacity,
            } => write!(
                f,
                "output projection tensor {tensor:?} has dtype {dtype:?}, shape {shape:?}, and {byte_capacity} bytes"
            ),
            Self::InvalidEmbeddingNormWeight {
                tensor,
                dtype,
                shape,
                byte_capacity,
            } => write!(
                f,
                "output embedding norm tensor {tensor:?} has dtype {dtype:?}, shape {shape:?}, and {byte_capacity} bytes"
            ),
            Self::MissingModelOutputBuffer { signal_id } => {
                write!(f, "missing model output boundary buffer {signal_id:?}")
            }
            Self::InvalidInputFrameByteCapacity {
                signal_id,
                byte_capacity,
                expected_byte_capacity,
            } => write!(
                f,
                "model output boundary buffer {signal_id:?} has {byte_capacity} bytes, expected {expected_byte_capacity}"
            ),
            Self::WorkgroupCountOverflow => {
                f.write_str("output transducer workgroup count overflowed")
            }
            Self::DescriptorCountOverflow => {
                f.write_str("output transducer descriptor count overflowed")
            }
            Self::PushConstantByteCountOverflow => {
                f.write_str("output transducer push constant byte count overflowed")
            }
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentOutputTransducerRunnerError {}

impl From<VulkanError> for VulkanResidentOutputTransducerRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

fn validate_lfm2_output_embedding_weight(
    allocation: &VulkanPermanentParameterBufferAllocation,
) -> Result<(), VulkanResidentOutputTransducerRunnerError> {
    if allocation.parameter.dtype.as_deref() != Some("BF16")
        || allocation.parameter.shape.as_deref() != Some(&[LFM2_VOCAB_SIZE, LFM2_HIDDEN_SIZE])
        || allocation.byte_capacity != LFM2_EMBED_TOKENS_BYTES
    {
        return Err(
            VulkanResidentOutputTransducerRunnerError::InvalidEmbeddingWeight {
                tensor: allocation.parameter.tensor.clone(),
                dtype: allocation.parameter.dtype.clone(),
                shape: allocation.parameter.shape.clone(),
                byte_capacity: allocation.byte_capacity,
            },
        );
    }
    Ok(())
}

fn validate_lfm2_embedding_norm_weight(
    allocation: &VulkanPermanentParameterBufferAllocation,
) -> Result<(), VulkanResidentOutputTransducerRunnerError> {
    if allocation.parameter.dtype.as_deref() != Some("BF16")
        || allocation.parameter.shape.as_deref() != Some(&[LFM2_HIDDEN_SIZE])
        || allocation.byte_capacity != LFM2_FRAME_BYTES
    {
        return Err(
            VulkanResidentOutputTransducerRunnerError::InvalidEmbeddingNormWeight {
                tensor: allocation.parameter.tensor.clone(),
                dtype: allocation.parameter.dtype.clone(),
                shape: allocation.parameter.shape.clone(),
                byte_capacity: allocation.byte_capacity,
            },
        );
    }
    Ok(())
}

pub struct VulkanResidentGreedySamplerRunner {
    pub sampler_id: String,
    pub logits_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
    output_buffer: VulkanResidentBuffer,
    resident_dispatch: VulkanResidentKernelDispatch,
}

impl VulkanResidentGreedySamplerRunner {
    pub fn from_output_transducer(
        device: &VulkanComputeDevice,
        output_transducer: &VulkanResidentOutputTransducerRunner,
        spirv_words: &[u32],
    ) -> Result<Self, VulkanResidentGreedySamplerRunnerError> {
        Self::from_logits_buffer(
            device,
            output_transducer.logits_buffer(),
            output_transducer.logits_byte_capacity,
            spirv_words,
        )
    }

    pub fn from_logits_buffer(
        device: &VulkanComputeDevice,
        logits_buffer: &VulkanResidentBuffer,
        logits_byte_capacity: usize,
        spirv_words: &[u32],
    ) -> Result<Self, VulkanResidentGreedySamplerRunnerError> {
        if logits_byte_capacity != LFM2_LOGITS_BYTES {
            return Err(
                VulkanResidentGreedySamplerRunnerError::InvalidLogitsByteCapacity {
                    byte_capacity: logits_byte_capacity,
                    expected_byte_capacity: LFM2_LOGITS_BYTES,
                },
            );
        }
        let output_buffer = device.create_resident_buffer(LFM2_SAMPLER_OUTPUT_BYTES)?;
        let bindings = [
            VulkanResidentKernelBufferBinding::new(0, logits_buffer, logits_byte_capacity),
            VulkanResidentKernelBufferBinding::new(1, &output_buffer, LFM2_SAMPLER_OUTPUT_BYTES),
        ];
        let resident_dispatch = device.create_resident_kernel_dispatch(
            spirv_words,
            &bindings,
            1,
            VULKAN_GREEDY_SAMPLER_LOCAL_SIZE_X,
            0,
        )?;

        Ok(Self {
            sampler_id: LFM2_GREEDY_SAMPLER_PEDAL_ID.to_string(),
            logits_byte_capacity,
            output_byte_capacity: LFM2_SAMPLER_OUTPUT_BYTES,
            descriptor_count: resident_dispatch.descriptor_count(),
            workgroup_count_x: resident_dispatch.workgroup_count_x(),
            push_constant_byte_count: resident_dispatch.push_constant_byte_count(),
            output_buffer,
            resident_dispatch,
        })
    }

    pub fn run(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentGreedySamplerRun, VulkanResidentGreedySamplerRunnerError> {
        device.run_resident_kernel_dispatch(&self.resident_dispatch, &[])?;
        let output = self.output_buffer.read_bytes(LFM2_SAMPLER_OUTPUT_BYTES)?;
        let token_id = u32::from_le_bytes([output[0], output[1], output[2], output[3]]);
        let selected_logit_bits = u32::from_le_bytes([output[4], output[5], output[6], output[7]]);
        let control_flags = u32::from_le_bytes([output[8], output[9], output[10], output[11]]);
        Ok(VulkanResidentGreedySamplerRun {
            sampler_id: self.sampler_id.clone(),
            token_id,
            selected_logit_bits,
            control_flags,
            descriptor_count: self.descriptor_count,
            workgroup_count_x: self.workgroup_count_x,
            push_constant_byte_count: self.push_constant_byte_count,
        })
    }

    pub fn read_output_bytes(&self) -> Result<Vec<u8>, VulkanError> {
        self.output_buffer.read_bytes(LFM2_SAMPLER_OUTPUT_BYTES)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedySamplerRun {
    pub sampler_id: String,
    pub token_id: u32,
    pub selected_logit_bits: u32,
    pub control_flags: u32,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
}

#[derive(Debug)]
pub enum VulkanResidentGreedySamplerRunnerError {
    InvalidLogitsByteCapacity {
        byte_capacity: usize,
        expected_byte_capacity: usize,
    },
    Vulkan(VulkanError),
}

impl Display for VulkanResidentGreedySamplerRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLogitsByteCapacity {
                byte_capacity,
                expected_byte_capacity,
            } => write!(
                f,
                "greedy sampler logits buffer has {byte_capacity} bytes, expected {expected_byte_capacity}"
            ),
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentGreedySamplerRunnerError {}

impl From<VulkanError> for VulkanResidentGreedySamplerRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

pub struct VulkanResidentSingleTokenTickRunner {
    pub device_id: String,
    pub pedal_count: usize,
    pub dispatch_count: usize,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    pedalboard: VulkanMountedPlacedResidentPedalboardRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
}

impl VulkanResidentSingleTokenTickRunner {
    pub fn new(
        input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
        pedalboard: VulkanMountedPlacedResidentPedalboardRunner,
        output_transducer: VulkanResidentOutputTransducerRunner,
    ) -> Result<Self, VulkanResidentSingleTokenTickRunnerError> {
        let dispatch_count = 1usize
            .checked_add(pedalboard.dispatch_count())
            .and_then(|count| count.checked_add(output_transducer.dispatch_count))
            .ok_or(VulkanResidentSingleTokenTickRunnerError::DispatchCountOverflow)?;
        let total_descriptor_count = input_transducer
            .descriptor_count
            .checked_add(pedalboard.total_descriptor_count)
            .and_then(|count| count.checked_add(output_transducer.total_descriptor_count))
            .ok_or(VulkanResidentSingleTokenTickRunnerError::DescriptorCountOverflow)?;
        let total_push_constant_byte_count = input_transducer
            .push_constant_byte_count
            .checked_add(pedalboard.total_push_constant_byte_count)
            .and_then(|count| count.checked_add(output_transducer.total_push_constant_byte_count))
            .ok_or(VulkanResidentSingleTokenTickRunnerError::PushConstantByteCountOverflow)?;

        Ok(Self {
            device_id: pedalboard.device_id.clone(),
            pedal_count: pedalboard.pedal_count(),
            dispatch_count,
            total_descriptor_count,
            total_push_constant_byte_count,
            input_transducer,
            pedalboard,
            output_transducer,
        })
    }

    pub fn run_token_id_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        token_id: u32,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<VulkanResidentSingleTokenTickRun, VulkanResidentSingleTokenTickRunnerError> {
        let input_run = self.input_transducer.run_token_id(device, token_id)?;
        let pedalboard_run = self.pedalboard.run_with_stream_control(device, control)?;
        let output_run = self.output_transducer.run(device)?;
        Ok(VulkanResidentSingleTokenTickRun {
            device_id: self.device_id.clone(),
            token_id,
            input_run,
            pedalboard_run,
            output_run,
            dispatch_count: self.dispatch_count,
            total_descriptor_count: self.total_descriptor_count,
            total_push_constant_byte_count: self.total_push_constant_byte_count,
        })
    }

    pub fn read_logits_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.output_transducer.read_logits_bytes(len)
    }

    pub fn read_normalized_frame_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.output_transducer.read_normalized_frame_bytes(len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentSingleTokenTickRun {
    pub device_id: String,
    pub token_id: u32,
    pub input_run: VulkanResidentInputEmbeddingTransducerRun,
    pub pedalboard_run: VulkanMountedPlacedResidentPedalboardRun,
    pub output_run: VulkanResidentOutputTransducerRun,
    pub dispatch_count: usize,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
}

#[derive(Debug)]
pub enum VulkanResidentSingleTokenTickRunnerError {
    DispatchCountOverflow,
    DescriptorCountOverflow,
    PushConstantByteCountOverflow,
    InputTransducer(VulkanResidentInputEmbeddingTransducerRunnerError),
    Pedalboard(VulkanMountedPlacedResidentKernelDispatchError),
    OutputTransducer(VulkanResidentOutputTransducerRunnerError),
}

impl Display for VulkanResidentSingleTokenTickRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DispatchCountOverflow => {
                f.write_str("single-token tick dispatch count overflowed")
            }
            Self::DescriptorCountOverflow => {
                f.write_str("single-token tick descriptor count overflowed")
            }
            Self::PushConstantByteCountOverflow => {
                f.write_str("single-token tick push constant byte count overflowed")
            }
            Self::InputTransducer(error) => Display::fmt(error, f),
            Self::Pedalboard(error) => Display::fmt(error, f),
            Self::OutputTransducer(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentSingleTokenTickRunnerError {}

impl From<VulkanResidentInputEmbeddingTransducerRunnerError>
    for VulkanResidentSingleTokenTickRunnerError
{
    fn from(error: VulkanResidentInputEmbeddingTransducerRunnerError) -> Self {
        Self::InputTransducer(error)
    }
}

impl From<VulkanMountedPlacedResidentKernelDispatchError>
    for VulkanResidentSingleTokenTickRunnerError
{
    fn from(error: VulkanMountedPlacedResidentKernelDispatchError) -> Self {
        Self::Pedalboard(error)
    }
}

impl From<VulkanResidentOutputTransducerRunnerError> for VulkanResidentSingleTokenTickRunnerError {
    fn from(error: VulkanResidentOutputTransducerRunnerError) -> Self {
        Self::OutputTransducer(error)
    }
}

pub struct VulkanResidentGreedyFeedbackLoopRunner {
    pub device_id: String,
    pub pedal_count: usize,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
    tick_runner: VulkanResidentSingleTokenTickRunner,
    sampler: VulkanResidentGreedySamplerRunner,
}

impl VulkanResidentGreedyFeedbackLoopRunner {
    pub fn new(
        tick_runner: VulkanResidentSingleTokenTickRunner,
        sampler: VulkanResidentGreedySamplerRunner,
    ) -> Result<Self, VulkanResidentGreedyFeedbackLoopRunnerError> {
        let per_tick_dispatch_count = tick_runner
            .dispatch_count
            .checked_add(1)
            .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::DispatchCountOverflow)?;
        let per_tick_descriptor_count = tick_runner
            .total_descriptor_count
            .checked_add(sampler.descriptor_count)
            .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::DescriptorCountOverflow)?;
        let per_tick_push_constant_byte_count = tick_runner
            .total_push_constant_byte_count
            .checked_add(sampler.push_constant_byte_count)
            .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::PushConstantByteCountOverflow)?;

        Ok(Self {
            device_id: tick_runner.device_id.clone(),
            pedal_count: tick_runner.pedal_count,
            per_tick_dispatch_count,
            per_tick_descriptor_count,
            per_tick_push_constant_byte_count,
            tick_runner,
            sampler,
        })
    }

    pub fn run_bounded(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        max_ticks: usize,
    ) -> Result<VulkanResidentGreedyFeedbackLoopRun, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        if max_ticks == 0 {
            return Err(VulkanResidentGreedyFeedbackLoopRunnerError::ZeroTickBudget);
        }

        let mut input_token_id = initial_token_id;
        let mut tick_runs = Vec::with_capacity(max_ticks);
        let mut sampled_token_ids = Vec::with_capacity(max_ticks);

        for tick_index in 0..max_ticks {
            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(tick_index).map_err(|_| {
                        VulkanResidentGreedyFeedbackLoopRunnerError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::StreamTickOverflow)?;
            let tick_run = self.tick_runner.run_token_id_with_stream_control(
                device,
                input_token_id,
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            )?;
            let sampler_run = self.sampler.run(device)?;
            let sampled_token_id = sampler_run.token_id;
            sampled_token_ids.push(sampled_token_id);
            tick_runs.push(VulkanResidentGreedyFeedbackTickRun {
                stream_tick,
                input_token_id,
                sampled_token_id,
                tick_run,
                sampler_run,
            });
            input_token_id = sampled_token_id;
        }

        Ok(VulkanResidentGreedyFeedbackLoopRun {
            device_id: self.device_id.clone(),
            initial_token_id,
            sampled_token_ids,
            tick_runs,
            per_tick_dispatch_count: self.per_tick_dispatch_count,
            per_tick_descriptor_count: self.per_tick_descriptor_count,
            per_tick_push_constant_byte_count: self.per_tick_push_constant_byte_count,
        })
    }

    pub fn run_prompt_event_bounded(
        &self,
        device: &VulkanComputeDevice,
        prompt_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<VulkanResidentGreedyPromptEventRun, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        if prompt_token_ids.is_empty() {
            return Err(VulkanResidentGreedyFeedbackLoopRunnerError::EmptyPromptEvent);
        }

        let mut external_input_index = 0usize;
        let mut pending_feedback: Option<VulkanResidentPendingPrivateFeedback> = None;
        let mut tick_runs = Vec::new();
        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut remaining_public_outputs = max_new_tokens;
        let mut stop_reason = (max_new_tokens == 0).then(|| "max_new_tokens".to_string());

        while external_input_index < prompt_token_ids.len() || pending_feedback.is_some() {
            let (input_token_id, input_route, input_feedback_depth, input_closes_loop) =
                if external_input_index < prompt_token_ids.len() {
                    let token_id = prompt_token_ids[external_input_index];
                    external_input_index += 1;
                    (
                        token_id,
                        VulkanResidentGreedyPromptEventInputRoute::ExternalInput,
                        0,
                        false,
                    )
                } else {
                    let feedback = pending_feedback.take().ok_or(
                        VulkanResidentGreedyFeedbackLoopRunnerError::MissingPrivateFeedback,
                    )?;
                    (
                        feedback.token_id,
                        VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback,
                        feedback.feedback_depth,
                        feedback.closes_loop_after_processing,
                    )
                };

            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(tick_runs.len()).map_err(|_| {
                        VulkanResidentGreedyFeedbackLoopRunnerError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::StreamTickOverflow)?;
            let tick_run = self.tick_runner.run_token_id_with_stream_control(
                device,
                input_token_id,
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            )?;

            let external_inputs_remaining = prompt_token_ids.len() - external_input_index;
            let should_emit_public_output =
                remaining_public_outputs > 0 && external_inputs_remaining == 0;
            let mut public_output_token_id = None;
            let mut private_feedback_token_id = None;
            let mut private_feedback_closes_loop_after_processing = None;
            let mut sampler_run = None;

            if should_emit_public_output {
                let run = self.sampler.run(device)?;
                let sampled_token_id = run.token_id;
                generated_token_ids.push(sampled_token_id);
                public_output_token_id = Some(sampled_token_id);
                remaining_public_outputs -= 1;

                let close_after_feedback = if eos_token_id == Some(sampled_token_id) {
                    remaining_public_outputs = 0;
                    stop_reason = Some("eos".to_string());
                    true
                } else if remaining_public_outputs == 0 {
                    stop_reason = Some("max_new_tokens".to_string());
                    true
                } else {
                    false
                };
                private_feedback_token_id = Some(sampled_token_id);
                private_feedback_closes_loop_after_processing = Some(close_after_feedback);
                pending_feedback = Some(VulkanResidentPendingPrivateFeedback {
                    token_id: sampled_token_id,
                    feedback_depth: input_feedback_depth.checked_add(1).ok_or(
                        VulkanResidentGreedyFeedbackLoopRunnerError::FeedbackDepthOverflow,
                    )?,
                    closes_loop_after_processing: close_after_feedback,
                });
                sampler_run = Some(run);
            }

            tick_runs.push(VulkanResidentGreedyPromptEventTickRun {
                stream_tick,
                input_token_id,
                input_route,
                input_feedback_depth,
                input_closes_loop_after_processing: input_closes_loop,
                public_output_token_id,
                private_feedback_token_id,
                private_feedback_closes_loop_after_processing,
                tick_run,
                sampler_run,
            });

            if input_closes_loop {
                pending_feedback = None;
            }
        }

        let output_token_ids = prompt_token_ids
            .iter()
            .copied()
            .chain(generated_token_ids.iter().copied())
            .collect();

        Ok(VulkanResidentGreedyPromptEventRun {
            device_id: self.device_id.clone(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            generated_token_ids,
            output_token_ids,
            stop_reason: stop_reason.unwrap_or_else(|| "max_new_tokens".to_string()),
            tick_runs,
            per_tick_dispatch_count: self.per_tick_dispatch_count,
            per_tick_descriptor_count: self.per_tick_descriptor_count,
            per_tick_push_constant_byte_count: self.per_tick_push_constant_byte_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyFeedbackLoopRun {
    pub device_id: String,
    pub initial_token_id: u32,
    pub sampled_token_ids: Vec<u32>,
    pub tick_runs: Vec<VulkanResidentGreedyFeedbackTickRun>,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyFeedbackTickRun {
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub sampled_token_id: u32,
    pub tick_run: VulkanResidentSingleTokenTickRun,
    pub sampler_run: VulkanResidentGreedySamplerRun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentGreedyPromptEventInputRoute {
    ExternalInput,
    PrivateFeedback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyPromptEventRun {
    pub device_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub stop_reason: String,
    pub tick_runs: Vec<VulkanResidentGreedyPromptEventTickRun>,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyPromptEventTickRun {
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub input_route: VulkanResidentGreedyPromptEventInputRoute,
    pub input_feedback_depth: u32,
    pub input_closes_loop_after_processing: bool,
    pub public_output_token_id: Option<u32>,
    pub private_feedback_token_id: Option<u32>,
    pub private_feedback_closes_loop_after_processing: Option<bool>,
    pub tick_run: VulkanResidentSingleTokenTickRun,
    pub sampler_run: Option<VulkanResidentGreedySamplerRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentPendingPrivateFeedback {
    token_id: u32,
    feedback_depth: u32,
    closes_loop_after_processing: bool,
}

#[derive(Debug)]
pub enum VulkanResidentGreedyFeedbackLoopRunnerError {
    ZeroTickBudget,
    EmptyPromptEvent,
    MissingPrivateFeedback,
    StreamTickOverflow,
    DynamicStateCapacityOverflow,
    OutputBudgetOverflow,
    StreamStateCapacityExceeded {
        stream_tick: u64,
        dynamic_state_capacity_activations: usize,
    },
    FeedbackDepthOverflow,
    DispatchCountOverflow,
    DescriptorCountOverflow,
    PushConstantByteCountOverflow,
    Tick(VulkanResidentSingleTokenTickRunnerError),
    Sampler(VulkanResidentGreedySamplerRunnerError),
}

impl Display for VulkanResidentGreedyFeedbackLoopRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTickBudget => {
                f.write_str("greedy feedback loop tick budget must not be zero")
            }
            Self::EmptyPromptEvent => f.write_str("greedy prompt event must contain input"),
            Self::MissingPrivateFeedback => {
                f.write_str("greedy prompt event expected private feedback")
            }
            Self::StreamTickOverflow => f.write_str("greedy feedback loop stream tick overflowed"),
            Self::DynamicStateCapacityOverflow => {
                f.write_str("greedy feedback loop dynamic state capacity overflowed")
            }
            Self::OutputBudgetOverflow => {
                f.write_str("greedy running stream output budget overflowed")
            }
            Self::StreamStateCapacityExceeded {
                stream_tick,
                dynamic_state_capacity_activations,
            } => write!(
                f,
                "greedy feedback loop stream tick {stream_tick} exceeds dynamic state capacity {dynamic_state_capacity_activations}"
            ),
            Self::FeedbackDepthOverflow => {
                f.write_str("greedy feedback loop feedback depth overflowed")
            }
            Self::DispatchCountOverflow => {
                f.write_str("greedy feedback loop dispatch count overflowed")
            }
            Self::DescriptorCountOverflow => {
                f.write_str("greedy feedback loop descriptor count overflowed")
            }
            Self::PushConstantByteCountOverflow => {
                f.write_str("greedy feedback loop push constant byte count overflowed")
            }
            Self::Tick(error) => Display::fmt(error, f),
            Self::Sampler(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentGreedyFeedbackLoopRunnerError {}

impl From<VulkanResidentSingleTokenTickRunnerError>
    for VulkanResidentGreedyFeedbackLoopRunnerError
{
    fn from(error: VulkanResidentSingleTokenTickRunnerError) -> Self {
        Self::Tick(error)
    }
}

impl From<VulkanResidentGreedySamplerRunnerError> for VulkanResidentGreedyFeedbackLoopRunnerError {
    fn from(error: VulkanResidentGreedySamplerRunnerError) -> Self {
        Self::Sampler(error)
    }
}

pub struct VulkanResidentGreedyStreamProcessor {
    pub device_id: String,
    pub pedal_count: usize,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
    pub dynamic_state_capacity_activations: usize,
    _mounted: VulkanMountedPlacedStreamCircuit,
    _transducer_parameter_buffers: VulkanPermanentParameterBuffers,
    loop_runner: VulkanResidentGreedyFeedbackLoopRunner,
}

impl VulkanResidentGreedyStreamProcessor {
    pub fn new(
        mounted: VulkanMountedPlacedStreamCircuit,
        transducer_parameter_buffers: VulkanPermanentParameterBuffers,
        loop_runner: VulkanResidentGreedyFeedbackLoopRunner,
    ) -> Self {
        Self {
            device_id: loop_runner.device_id.clone(),
            pedal_count: loop_runner.pedal_count,
            per_tick_dispatch_count: loop_runner.per_tick_dispatch_count,
            per_tick_descriptor_count: loop_runner.per_tick_descriptor_count,
            per_tick_push_constant_byte_count: loop_runner.per_tick_push_constant_byte_count,
            dynamic_state_capacity_activations: mounted.buffers.dynamic_state_capacity_activations,
            _mounted: mounted,
            _transducer_parameter_buffers: transducer_parameter_buffers,
            loop_runner,
        }
    }

    pub fn run_bounded(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        max_ticks: usize,
    ) -> Result<VulkanResidentGreedyFeedbackLoopRun, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        self.loop_runner.run_bounded(
            device,
            initial_token_id,
            start_stream_tick,
            self.dynamic_state_capacity_activations_u32()?,
            max_ticks,
        )
    }

    pub fn run_prompt_event_bounded(
        &self,
        device: &VulkanComputeDevice,
        prompt_token_ids: &[u32],
        start_stream_tick: u64,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<VulkanResidentGreedyPromptEventRun, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        self.loop_runner.run_prompt_event_bounded(
            device,
            prompt_token_ids,
            start_stream_tick,
            self.dynamic_state_capacity_activations_u32()?,
            max_new_tokens,
            eos_token_id,
        )
    }

    fn dynamic_state_capacity_activations_u32(
        &self,
    ) -> Result<u32, VulkanResidentGreedyFeedbackLoopRunnerError> {
        u32::try_from(self.dynamic_state_capacity_activations)
            .map_err(|_| VulkanResidentGreedyFeedbackLoopRunnerError::DynamicStateCapacityOverflow)
    }

    pub fn into_running_stream(
        self,
        stream_id: impl Into<String>,
    ) -> VulkanResidentGreedyRunningStream {
        VulkanResidentGreedyRunningStream::new(stream_id, self)
    }

    pub fn into_token_stream(self, stream_id: impl Into<String>) -> VulkanResidentTokenStream {
        VulkanResidentTokenStream::new(stream_id, self)
    }
}

pub struct VulkanResidentGreedyRunningStream {
    pub stream_id: String,
    pub next_stream_tick: u64,
    pub remaining_public_outputs: usize,
    pub eos_token_id: Option<u32>,
    pub loop_open: bool,
    pub last_stop_reason: Option<String>,
    processor: VulkanResidentGreedyStreamProcessor,
    external_input_queue: VecDeque<VulkanResidentGreedyExternalInputSignal>,
    private_feedback_queue: VecDeque<VulkanResidentGreedyPrivateFeedbackSignal>,
    public_outputs: Vec<VulkanResidentGreedyPublicOutputSignal>,
    private_feedback_history: Vec<VulkanResidentGreedyPrivateFeedbackSignal>,
    ticks: Vec<VulkanResidentGreedyRunningStreamTick>,
    input_counter: usize,
    public_counter: usize,
    feedback_counter: usize,
}

impl VulkanResidentGreedyRunningStream {
    pub fn new(
        stream_id: impl Into<String>,
        processor: VulkanResidentGreedyStreamProcessor,
    ) -> Self {
        Self {
            stream_id: stream_id.into(),
            next_stream_tick: 0,
            remaining_public_outputs: 0,
            eos_token_id: None,
            loop_open: false,
            last_stop_reason: None,
            processor,
            external_input_queue: VecDeque::new(),
            private_feedback_queue: VecDeque::new(),
            public_outputs: Vec::new(),
            private_feedback_history: Vec::new(),
            ticks: Vec::new(),
            input_counter: 0,
            public_counter: 0,
            feedback_counter: 0,
        }
    }

    pub fn inject_token(
        &mut self,
        token_id: u32,
        origin: impl Into<String>,
    ) -> VulkanResidentGreedyExternalInputSignal {
        let signal = VulkanResidentGreedyExternalInputSignal {
            id: format!("input_{}", self.input_counter),
            token_id,
            origin: origin.into(),
        };
        self.input_counter += 1;
        self.external_input_queue.push_back(signal.clone());
        signal
    }

    pub fn inject_prompt(
        &mut self,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<
        Vec<VulkanResidentGreedyExternalInputSignal>,
        VulkanResidentGreedyFeedbackLoopRunnerError,
    > {
        self.inject_external_tokens(
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
            "external_input",
        )
    }

    pub fn inject_external_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        origin: impl Into<String>,
    ) -> Result<
        Vec<VulkanResidentGreedyExternalInputSignal>,
        VulkanResidentGreedyFeedbackLoopRunnerError,
    > {
        if token_ids.is_empty() {
            return Err(VulkanResidentGreedyFeedbackLoopRunnerError::EmptyPromptEvent);
        }
        let origin = origin.into();

        self.remaining_public_outputs =
            self.remaining_public_outputs
                .checked_add(max_new_tokens)
                .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::OutputBudgetOverflow)?;
        self.eos_token_id = eos_token_id;
        self.loop_open = self.remaining_public_outputs > 0;
        self.last_stop_reason = (max_new_tokens == 0).then(|| "max_new_tokens".to_string());

        Ok(token_ids
            .iter()
            .copied()
            .map(|token_id| self.inject_token(token_id, origin.clone()))
            .collect())
    }

    pub fn continue_loop(
        &mut self,
        additional_public_outputs: usize,
    ) -> Result<(), VulkanResidentGreedyFeedbackLoopRunnerError> {
        self.remaining_public_outputs = self
            .remaining_public_outputs
            .checked_add(additional_public_outputs)
            .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::OutputBudgetOverflow)?;
        if self.remaining_public_outputs > 0 {
            self.loop_open = true;
            self.last_stop_reason = None;
        }
        Ok(())
    }

    pub fn interrupt(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentGreedyStreamControlEvent {
        let reason = reason.into();
        let cleared_private_feedback_ids = self
            .private_feedback_queue
            .iter()
            .map(|signal| signal.id.clone())
            .collect::<Vec<_>>();
        self.private_feedback_queue.clear();
        self.remaining_public_outputs = 0;
        self.loop_open = false;
        self.last_stop_reason = Some(reason.clone());

        VulkanResidentGreedyStreamControlEvent {
            event_type: VulkanResidentGreedyStreamControlEventType::Interrupt,
            reason,
            cleared_private_feedback_ids,
            closing_private_feedback_id: None,
            state_preserved: true,
        }
    }

    pub fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentGreedyStreamControlEvent {
        let reason = reason.into();
        let mut cleared_private_feedback_ids = Vec::new();
        let mut closing_private_feedback_id = None;

        if let Some(mut current) = self.private_feedback_queue.pop_front() {
            closing_private_feedback_id = Some(current.id.clone());
            current.closes_loop_after_processing = true;
            current.stop_reason = Some(reason.clone());
            if let Some(history_signal) = self
                .private_feedback_history
                .iter_mut()
                .find(|signal| signal.id == current.id)
            {
                history_signal.closes_loop_after_processing = true;
                history_signal.stop_reason = Some(reason.clone());
            }
            cleared_private_feedback_ids.extend(
                self.private_feedback_queue
                    .drain(..)
                    .map(|signal| signal.id),
            );
            self.private_feedback_queue.push_front(current);
            self.loop_open = true;
        } else {
            self.loop_open = false;
        }

        self.remaining_public_outputs = 0;
        self.last_stop_reason = Some(reason.clone());

        VulkanResidentGreedyStreamControlEvent {
            event_type: VulkanResidentGreedyStreamControlEventType::StopAfterCurrent,
            reason,
            cleared_private_feedback_ids,
            closing_private_feedback_id,
            state_preserved: true,
        }
    }

    pub fn tick(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentGreedyRunningStreamTick, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        if self.external_input_queue.is_empty() && self.private_feedback_queue.is_empty() {
            let tick = VulkanResidentGreedyRunningStreamTick {
                stream_id: self.stream_id.clone(),
                stream_tick: None,
                status: VulkanResidentGreedyRunningStreamTickStatus::Idle,
                input_signal: None,
                tick_run: None,
                public_output: None,
                private_feedback: None,
                sampler_run: None,
                stop_reason: self.last_stop_reason.clone(),
            };
            self.ticks.push(tick.clone());
            return Ok(tick);
        }

        let stream_tick = self.next_stream_tick;
        self.ensure_stream_tick_capacity(stream_tick)?;
        let input_signal = self
            .next_input_signal()
            .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::MissingPrivateFeedback)?;
        let tick_run = self
            .processor
            .loop_runner
            .tick_runner
            .run_token_id_with_stream_control(
                device,
                input_signal.token_id(),
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations: self
                        .processor
                        .dynamic_state_capacity_activations_u32()?,
                },
            )?;
        self.next_stream_tick = self
            .next_stream_tick
            .checked_add(1)
            .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::StreamTickOverflow)?;

        let should_emit_public_output =
            self.remaining_public_outputs > 0 && self.external_input_queue.is_empty();
        let mut public_output = None;
        let mut private_feedback = None;
        let mut sampler_run = None;

        if should_emit_public_output {
            let run = self.processor.loop_runner.sampler.run(device)?;
            let sampled_token_id = run.token_id;
            self.remaining_public_outputs -= 1;

            let public = VulkanResidentGreedyPublicOutputSignal {
                id: format!("public_{}", self.public_counter),
                token_id: sampled_token_id,
                source_stream_tick: stream_tick,
                sampler_run: run.clone(),
            };
            self.public_counter += 1;
            self.public_outputs.push(public.clone());

            let close_after_feedback = if self.eos_token_id == Some(sampled_token_id) {
                self.remaining_public_outputs = 0;
                self.last_stop_reason = Some("eos".to_string());
                true
            } else if self.remaining_public_outputs == 0 {
                self.last_stop_reason = Some("max_new_tokens".to_string());
                true
            } else {
                false
            };
            let feedback_depth = input_signal
                .feedback_depth()
                .checked_add(1)
                .ok_or(VulkanResidentGreedyFeedbackLoopRunnerError::FeedbackDepthOverflow)?;
            let feedback = VulkanResidentGreedyPrivateFeedbackSignal {
                id: format!("feedback_{}", self.feedback_counter),
                token_id: sampled_token_id,
                source_public_output_id: public.id.clone(),
                feedback_depth,
                closes_loop_after_processing: close_after_feedback,
                stop_reason: self
                    .last_stop_reason
                    .clone()
                    .filter(|_| close_after_feedback),
            };
            self.feedback_counter += 1;
            self.private_feedback_queue.push_back(feedback.clone());
            self.private_feedback_history.push(feedback.clone());

            sampler_run = Some(run);
            public_output = Some(public);
            private_feedback = Some(feedback);
        }

        if input_signal.closes_loop_after_processing() {
            self.loop_open = false;
            self.last_stop_reason = input_signal
                .stop_reason()
                .cloned()
                .or_else(|| self.last_stop_reason.clone())
                .or_else(|| Some("max_new_tokens".to_string()));
        }

        let tick = VulkanResidentGreedyRunningStreamTick {
            stream_id: self.stream_id.clone(),
            stream_tick: Some(stream_tick),
            status: VulkanResidentGreedyRunningStreamTickStatus::Processed,
            input_signal: Some(input_signal),
            tick_run: Some(tick_run),
            public_output,
            private_feedback,
            sampler_run,
            stop_reason: self.last_stop_reason.clone(),
        };
        self.ticks.push(tick.clone());
        Ok(tick)
    }

    pub fn run_until_idle(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<
        Vec<VulkanResidentGreedyRunningStreamTick>,
        VulkanResidentGreedyFeedbackLoopRunnerError,
    > {
        let start = self.ticks.len();
        while !self.external_input_queue.is_empty() || !self.private_feedback_queue.is_empty() {
            self.tick(device)?;
        }
        self.tick(device)?;
        Ok(self.ticks[start..].to_vec())
    }

    pub fn run_prompt(
        &mut self,
        device: &VulkanComputeDevice,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<VulkanResidentGreedyRunningStreamRun, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        let start_public = self.public_outputs.len();
        let start_feedback = self.private_feedback_history.len();
        let start_stream_tick = self.next_stream_tick;
        self.inject_prompt(prompt_token_ids, max_new_tokens, eos_token_id)?;
        let ticks = self.run_until_idle(device)?;
        let public_outputs = self.public_outputs[start_public..].to_vec();
        let private_feedback = self.private_feedback_history[start_feedback..].to_vec();
        let generated_token_ids = public_outputs
            .iter()
            .map(|output| output.token_id)
            .collect::<Vec<_>>();
        let output_token_ids = prompt_token_ids
            .iter()
            .copied()
            .chain(generated_token_ids.iter().copied())
            .collect::<Vec<_>>();

        Ok(VulkanResidentGreedyRunningStreamRun {
            stream_id: self.stream_id.clone(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            generated_token_ids,
            output_token_ids,
            stop_reason: self
                .last_stop_reason
                .clone()
                .unwrap_or_else(|| "max_new_tokens".to_string()),
            start_stream_tick,
            next_stream_tick: self.next_stream_tick,
            ticks,
            public_outputs,
            private_feedback,
        })
    }

    pub fn pending_external_input_count(&self) -> usize {
        self.external_input_queue.len()
    }

    pub fn pending_private_feedback_count(&self) -> usize {
        self.private_feedback_queue.len()
    }

    pub fn public_outputs(&self) -> &[VulkanResidentGreedyPublicOutputSignal] {
        &self.public_outputs
    }

    pub fn private_feedback_history(&self) -> &[VulkanResidentGreedyPrivateFeedbackSignal] {
        &self.private_feedback_history
    }

    pub fn ticks(&self) -> &[VulkanResidentGreedyRunningStreamTick] {
        &self.ticks
    }

    fn next_input_signal(&mut self) -> Option<VulkanResidentGreedyRunningStreamInputSignal> {
        if let Some(signal) = self.external_input_queue.pop_front() {
            return Some(VulkanResidentGreedyRunningStreamInputSignal::External(
                signal,
            ));
        }
        self.private_feedback_queue
            .pop_front()
            .map(VulkanResidentGreedyRunningStreamInputSignal::PrivateFeedback)
    }

    fn ensure_stream_tick_capacity(
        &self,
        stream_tick: u64,
    ) -> Result<(), VulkanResidentGreedyFeedbackLoopRunnerError> {
        if stream_tick >= self.processor.dynamic_state_capacity_activations as u64 {
            return Err(
                VulkanResidentGreedyFeedbackLoopRunnerError::StreamStateCapacityExceeded {
                    stream_tick,
                    dynamic_state_capacity_activations: self
                        .processor
                        .dynamic_state_capacity_activations,
                },
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyExternalInputSignal {
    pub id: String,
    pub token_id: u32,
    pub origin: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyPublicOutputSignal {
    pub id: String,
    pub token_id: u32,
    pub source_stream_tick: u64,
    pub sampler_run: VulkanResidentGreedySamplerRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyPrivateFeedbackSignal {
    pub id: String,
    pub token_id: u32,
    pub source_public_output_id: String,
    pub feedback_depth: u32,
    pub closes_loop_after_processing: bool,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentGreedyStreamControlEventType {
    Interrupt,
    StopAfterCurrent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyStreamControlEvent {
    pub event_type: VulkanResidentGreedyStreamControlEventType,
    pub reason: String,
    pub cleared_private_feedback_ids: Vec<String>,
    pub closing_private_feedback_id: Option<String>,
    pub state_preserved: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanResidentGreedyRunningStreamInputSignal {
    External(VulkanResidentGreedyExternalInputSignal),
    PrivateFeedback(VulkanResidentGreedyPrivateFeedbackSignal),
}

impl VulkanResidentGreedyRunningStreamInputSignal {
    pub fn token_id(&self) -> u32 {
        match self {
            Self::External(signal) => signal.token_id,
            Self::PrivateFeedback(signal) => signal.token_id,
        }
    }

    pub fn route(&self) -> VulkanResidentGreedyPromptEventInputRoute {
        match self {
            Self::External(_) => VulkanResidentGreedyPromptEventInputRoute::ExternalInput,
            Self::PrivateFeedback(_) => VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback,
        }
    }

    pub fn feedback_depth(&self) -> u32 {
        match self {
            Self::External(_) => 0,
            Self::PrivateFeedback(signal) => signal.feedback_depth,
        }
    }

    pub fn closes_loop_after_processing(&self) -> bool {
        match self {
            Self::External(_) => false,
            Self::PrivateFeedback(signal) => signal.closes_loop_after_processing,
        }
    }

    pub fn stop_reason(&self) -> Option<&String> {
        match self {
            Self::External(_) => None,
            Self::PrivateFeedback(signal) => signal.stop_reason.as_ref(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentGreedyRunningStreamTickStatus {
    Processed,
    Idle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyRunningStreamTick {
    pub stream_id: String,
    pub stream_tick: Option<u64>,
    pub status: VulkanResidentGreedyRunningStreamTickStatus,
    pub input_signal: Option<VulkanResidentGreedyRunningStreamInputSignal>,
    pub tick_run: Option<VulkanResidentSingleTokenTickRun>,
    pub public_output: Option<VulkanResidentGreedyPublicOutputSignal>,
    pub private_feedback: Option<VulkanResidentGreedyPrivateFeedbackSignal>,
    pub sampler_run: Option<VulkanResidentGreedySamplerRun>,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentGreedyRunningStreamRun {
    pub stream_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub stop_reason: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub ticks: Vec<VulkanResidentGreedyRunningStreamTick>,
    pub public_outputs: Vec<VulkanResidentGreedyPublicOutputSignal>,
    pub private_feedback: Vec<VulkanResidentGreedyPrivateFeedbackSignal>,
}

pub struct VulkanResidentTokenStream {
    inner: VulkanResidentGreedyRunningStream,
    current_input_event_id: Option<String>,
    current_output_index: usize,
}

impl VulkanResidentTokenStream {
    pub fn new(
        stream_id: impl Into<String>,
        processor: VulkanResidentGreedyStreamProcessor,
    ) -> Self {
        Self {
            inner: VulkanResidentGreedyRunningStream::new(stream_id, processor),
            current_input_event_id: None,
            current_output_index: 0,
        }
    }

    pub fn from_running_stream(stream: VulkanResidentGreedyRunningStream) -> Self {
        Self {
            inner: stream,
            current_input_event_id: None,
            current_output_index: 0,
        }
    }

    pub fn into_inner(self) -> VulkanResidentGreedyRunningStream {
        self.inner
    }

    pub fn stream_id(&self) -> &str {
        &self.inner.stream_id
    }

    pub fn next_stream_tick(&self) -> u64 {
        self.inner.next_stream_tick
    }

    pub fn submit_external_event(
        &mut self,
        device: &VulkanComputeDevice,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenStreamRun, VulkanResidentGreedyFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let queued = self.enqueue_external_event(event)?;
        let event = queued.input_event;
        let pump = self.pump_until_idle(device)?;

        let generated_token_ids = pump
            .output_events
            .iter()
            .map(|output| output.token_id)
            .collect::<Vec<_>>();

        Ok(VulkanResidentTokenStreamRun {
            stream_id: self.inner.stream_id.clone(),
            input_event: event,
            generated_token_ids,
            output_events: pump.output_events,
            stop_reason: self
                .inner
                .last_stop_reason
                .clone()
                .unwrap_or_else(|| "max_new_tokens".to_string()),
            start_stream_tick,
            next_stream_tick: self.inner.next_stream_tick,
            processed_tick_count: pump.processed_tick_count,
            idle_tick_count: pump.idle_tick_count,
        })
    }

    pub fn enqueue_external_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenQueuedInputEvent, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        let start_stream_tick = self.inner.next_stream_tick;
        let enqueued_token_count = event.token_ids.len();
        self.inner.inject_external_tokens(
            &event.token_ids,
            event.max_public_tokens,
            event.eos_token_id,
            event.origin.clone(),
        )?;
        self.current_input_event_id = Some(event.id.clone());
        self.current_output_index = 0;

        Ok(VulkanResidentTokenQueuedInputEvent {
            input_event: event,
            start_stream_tick,
            enqueued_token_count,
        })
    }

    pub fn pump_once(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentTokenStreamTick, VulkanResidentGreedyFeedbackLoopRunnerError> {
        let tick = self.inner.tick(device)?;
        let output_event = tick.public_output.as_ref().map(|output| {
            let output_index = self.current_output_index;
            self.current_output_index += 1;
            VulkanResidentTokenOutputEvent {
                id: output.id.clone(),
                input_event_id: self
                    .current_input_event_id
                    .clone()
                    .unwrap_or_else(|| "feedback_loop".to_string()),
                output_index,
                token_id: output.token_id,
                source_stream_tick: output.source_stream_tick,
            }
        });

        Ok(VulkanResidentTokenStreamTick {
            stream_id: tick.stream_id,
            status: tick.status,
            stream_tick: tick.stream_tick,
            input_token_id: tick.input_signal.as_ref().map(|signal| signal.token_id()),
            input_route: tick.input_signal.as_ref().map(|signal| signal.route()),
            output_event,
            stop_reason: tick.stop_reason,
        })
    }

    pub fn pump_bounded(
        &mut self,
        device: &VulkanComputeDevice,
        max_ticks: usize,
    ) -> Result<VulkanResidentTokenStreamPumpRun, VulkanResidentGreedyFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let mut ticks = Vec::new();
        let mut output_events = Vec::new();
        let mut processed_tick_count = 0usize;
        let mut idle_tick_count = 0usize;
        let mut stop_condition = VulkanResidentTokenStreamPumpStopCondition::TickBudget;

        for _ in 0..max_ticks {
            let tick = self.pump_once(device)?;
            match tick.status {
                VulkanResidentGreedyRunningStreamTickStatus::Processed => {
                    processed_tick_count += 1;
                }
                VulkanResidentGreedyRunningStreamTickStatus::Idle => {
                    idle_tick_count += 1;
                    stop_condition = VulkanResidentTokenStreamPumpStopCondition::Idle;
                }
            }
            if let Some(output_event) = tick.output_event.clone() {
                output_events.push(output_event);
            }
            let is_idle = tick.status == VulkanResidentGreedyRunningStreamTickStatus::Idle;
            ticks.push(tick);
            if is_idle {
                break;
            }
        }

        Ok(VulkanResidentTokenStreamPumpRun {
            stream_id: self.inner.stream_id.clone(),
            start_stream_tick,
            next_stream_tick: self.inner.next_stream_tick,
            stop_condition,
            processed_tick_count,
            idle_tick_count,
            output_events,
            ticks,
            last_stop_reason: self.inner.last_stop_reason.clone(),
        })
    }

    pub fn pump_until_idle(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentTokenStreamPumpRun, VulkanResidentGreedyFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let mut ticks = Vec::new();
        let mut output_events = Vec::new();
        let mut processed_tick_count = 0usize;
        let mut idle_tick_count = 0usize;

        loop {
            let tick = self.pump_once(device)?;
            match tick.status {
                VulkanResidentGreedyRunningStreamTickStatus::Processed => {
                    processed_tick_count += 1;
                }
                VulkanResidentGreedyRunningStreamTickStatus::Idle => {
                    idle_tick_count += 1;
                }
            }
            if let Some(output_event) = tick.output_event.clone() {
                output_events.push(output_event);
            }
            let is_idle = tick.status == VulkanResidentGreedyRunningStreamTickStatus::Idle;
            ticks.push(tick);
            if is_idle {
                break;
            }
        }

        Ok(VulkanResidentTokenStreamPumpRun {
            stream_id: self.inner.stream_id.clone(),
            start_stream_tick,
            next_stream_tick: self.inner.next_stream_tick,
            stop_condition: VulkanResidentTokenStreamPumpStopCondition::Idle,
            processed_tick_count,
            idle_tick_count,
            output_events,
            ticks,
            last_stop_reason: self.inner.last_stop_reason.clone(),
        })
    }

    pub fn interrupt(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentGreedyStreamControlEvent {
        self.inner.interrupt(reason)
    }

    pub fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentGreedyStreamControlEvent {
        self.inner.stop_after_current(reason)
    }

    pub fn snapshot(&self) -> VulkanResidentTokenStreamSnapshot {
        VulkanResidentTokenStreamSnapshot {
            stream_id: self.inner.stream_id.clone(),
            next_stream_tick: self.inner.next_stream_tick,
            loop_open: self.inner.loop_open,
            idle: self.inner.external_input_queue.is_empty()
                && self.inner.private_feedback_queue.is_empty(),
            total_public_outputs: self.inner.public_outputs.len(),
            total_ticks: self.inner.ticks.len(),
            last_stop_reason: self.inner.last_stop_reason.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenInputEvent {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_public_tokens: usize,
    pub eos_token_id: Option<u32>,
    pub origin: String,
}

impl VulkanResidentTokenInputEvent {
    pub fn new(id: impl Into<String>, token_ids: Vec<u32>, max_public_tokens: usize) -> Self {
        Self {
            id: id.into(),
            token_ids,
            max_public_tokens,
            eos_token_id: None,
            origin: "host".to_string(),
        }
    }

    pub fn with_eos_token(mut self, eos_token_id: u32) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = origin.into();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenQueuedInputEvent {
    pub input_event: VulkanResidentTokenInputEvent,
    pub start_stream_tick: u64,
    pub enqueued_token_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenOutputEvent {
    pub id: String,
    pub input_event_id: String,
    pub output_index: usize,
    pub token_id: u32,
    pub source_stream_tick: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamTick {
    pub stream_id: String,
    pub status: VulkanResidentGreedyRunningStreamTickStatus,
    pub stream_tick: Option<u64>,
    pub input_token_id: Option<u32>,
    pub input_route: Option<VulkanResidentGreedyPromptEventInputRoute>,
    pub output_event: Option<VulkanResidentTokenOutputEvent>,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenStreamPumpStopCondition {
    Idle,
    TickBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamPumpRun {
    pub stream_id: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub stop_condition: VulkanResidentTokenStreamPumpStopCondition,
    pub processed_tick_count: usize,
    pub idle_tick_count: usize,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub ticks: Vec<VulkanResidentTokenStreamTick>,
    pub last_stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamRun {
    pub stream_id: String,
    pub input_event: VulkanResidentTokenInputEvent,
    pub generated_token_ids: Vec<u32>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub stop_reason: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub processed_tick_count: usize,
    pub idle_tick_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamSnapshot {
    pub stream_id: String,
    pub next_stream_tick: u64,
    pub loop_open: bool,
    pub idle: bool,
    pub total_public_outputs: usize,
    pub total_ticks: usize,
    pub last_stop_reason: Option<String>,
}

pub struct VulkanResidentTokenRuntime {
    stream: VulkanResidentTokenStream,
    pending_input_events: VecDeque<VulkanResidentTokenInputEvent>,
}

impl VulkanResidentTokenRuntime {
    pub fn new(stream: VulkanResidentTokenStream) -> Self {
        Self {
            stream,
            pending_input_events: VecDeque::new(),
        }
    }

    pub fn from_processor(
        stream_id: impl Into<String>,
        processor: VulkanResidentGreedyStreamProcessor,
    ) -> Self {
        Self::new(processor.into_token_stream(stream_id))
    }

    pub fn stream(&self) -> &VulkanResidentTokenStream {
        &self.stream
    }

    pub fn stream_mut(&mut self) -> &mut VulkanResidentTokenStream {
        &mut self.stream
    }

    pub fn into_stream(self) -> VulkanResidentTokenStream {
        self.stream
    }

    pub fn enqueue_input_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<
        VulkanResidentTokenRuntimeQueuedInputEvent,
        VulkanResidentGreedyFeedbackLoopRunnerError,
    > {
        if event.token_ids.is_empty() {
            return Err(VulkanResidentGreedyFeedbackLoopRunnerError::EmptyPromptEvent);
        }
        self.pending_input_events.push_back(event.clone());
        Ok(VulkanResidentTokenRuntimeQueuedInputEvent {
            input_event: event,
            pending_input_event_count: self.pending_input_events.len(),
        })
    }

    pub fn run_cycle(
        &mut self,
        device: &VulkanComputeDevice,
        max_ticks: usize,
    ) -> Result<VulkanResidentTokenRuntimeCycleRun, VulkanResidentGreedyFeedbackLoopRunnerError>
    {
        let stream_snapshot = self.stream.snapshot();
        let start_stream_tick = stream_snapshot.next_stream_tick;
        let mut remaining_tick_budget = max_ticks;
        let mut queued_input_events = Vec::new();
        let mut pump_runs = Vec::new();
        let mut output_events = Vec::new();
        let mut processed_tick_count = 0usize;
        let mut idle_tick_count = 0usize;
        let mut ticks_used = 0usize;
        let stop_condition;

        if remaining_tick_budget == 0 {
            return Ok(VulkanResidentTokenRuntimeCycleRun {
                stream_id: self.stream.stream_id().to_string(),
                start_stream_tick,
                next_stream_tick: self.stream.next_stream_tick(),
                max_ticks,
                ticks_used,
                stop_condition: VulkanResidentTokenRuntimeCycleStopCondition::TickBudget,
                queued_input_events,
                pump_runs,
                output_events,
                processed_tick_count,
                idle_tick_count,
                pending_input_event_count: self.pending_input_events.len(),
                stream_idle: self.stream.snapshot().idle,
                last_stop_reason: self.stream.snapshot().last_stop_reason,
            });
        }

        loop {
            if self.stream.snapshot().idle {
                if let Some(event) = self.pending_input_events.pop_front() {
                    queued_input_events.push(self.stream.enqueue_external_event(event)?);
                } else {
                    stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::Idle;
                    break;
                }
            }

            if remaining_tick_budget == 0 {
                stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::TickBudget;
                break;
            }

            let pump_run = self.stream.pump_bounded(device, remaining_tick_budget)?;
            let pump_ticks = pump_run.ticks.len();
            ticks_used += pump_ticks;
            remaining_tick_budget = remaining_tick_budget.saturating_sub(pump_ticks);
            processed_tick_count += pump_run.processed_tick_count;
            idle_tick_count += pump_run.idle_tick_count;
            output_events.extend(pump_run.output_events.iter().cloned());
            let pump_stopped_on_budget =
                pump_run.stop_condition == VulkanResidentTokenStreamPumpStopCondition::TickBudget;
            pump_runs.push(pump_run);

            if pump_stopped_on_budget {
                stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::TickBudget;
                break;
            }
        }

        let end_snapshot = self.stream.snapshot();
        let mut stop_condition = stop_condition;
        if stop_condition == VulkanResidentTokenRuntimeCycleStopCondition::Idle
            && (!end_snapshot.idle || !self.pending_input_events.is_empty())
        {
            stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::TickBudget;
        }

        Ok(VulkanResidentTokenRuntimeCycleRun {
            stream_id: end_snapshot.stream_id,
            start_stream_tick,
            next_stream_tick: end_snapshot.next_stream_tick,
            max_ticks,
            ticks_used,
            stop_condition,
            queued_input_events,
            pump_runs,
            output_events,
            processed_tick_count,
            idle_tick_count,
            pending_input_event_count: self.pending_input_events.len(),
            stream_idle: end_snapshot.idle,
            last_stop_reason: end_snapshot.last_stop_reason,
        })
    }

    pub fn snapshot(&self) -> VulkanResidentTokenRuntimeSnapshot {
        let stream = self.stream.snapshot();
        let idle = stream.idle && self.pending_input_events.is_empty();
        VulkanResidentTokenRuntimeSnapshot {
            stream,
            pending_input_event_count: self.pending_input_events.len(),
            idle,
            running: !idle,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeQueuedInputEvent {
    pub input_event: VulkanResidentTokenInputEvent,
    pub pending_input_event_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenRuntimeCycleStopCondition {
    Idle,
    TickBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeCycleRun {
    pub stream_id: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub max_ticks: usize,
    pub ticks_used: usize,
    pub stop_condition: VulkanResidentTokenRuntimeCycleStopCondition,
    pub queued_input_events: Vec<VulkanResidentTokenQueuedInputEvent>,
    pub pump_runs: Vec<VulkanResidentTokenStreamPumpRun>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub processed_tick_count: usize,
    pub idle_tick_count: usize,
    pub pending_input_event_count: usize,
    pub stream_idle: bool,
    pub last_stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSnapshot {
    pub stream: VulkanResidentTokenStreamSnapshot,
    pub pending_input_event_count: usize,
    pub idle: bool,
    pub running: bool,
}

pub struct VulkanResidentTokenRuntimeScheduler {
    runtimes: BTreeMap<String, VulkanResidentTokenRuntime>,
    active_queue: VecDeque<String>,
}

impl VulkanResidentTokenRuntimeScheduler {
    pub fn new() -> Self {
        Self {
            runtimes: BTreeMap::new(),
            active_queue: VecDeque::new(),
        }
    }

    pub fn add_runtime(
        &mut self,
        runtime: VulkanResidentTokenRuntime,
    ) -> Result<(), VulkanResidentTokenRuntimeSchedulerError> {
        let snapshot = runtime.snapshot();
        let stream_id = snapshot.stream.stream_id.clone();
        if self.runtimes.contains_key(&stream_id) {
            return Err(VulkanResidentTokenRuntimeSchedulerError::DuplicateStream(
                stream_id,
            ));
        }
        let running = snapshot.running;
        self.runtimes.insert(stream_id.clone(), runtime);
        if running {
            self.schedule(&stream_id);
        }
        Ok(())
    }

    pub fn has_runtime(&self, stream_id: &str) -> bool {
        self.runtimes.contains_key(stream_id)
    }

    pub fn runtime(&self, stream_id: &str) -> Option<&VulkanResidentTokenRuntime> {
        self.runtimes.get(stream_id)
    }

    pub fn runtime_mut(&mut self, stream_id: &str) -> Option<&mut VulkanResidentTokenRuntime> {
        self.runtimes.get_mut(stream_id)
    }

    pub fn enqueue_input_event(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenRuntimeQueuedInputEvent, VulkanResidentTokenRuntimeSchedulerError>
    {
        let queued = self
            .runtimes
            .get_mut(stream_id)
            .ok_or_else(|| {
                VulkanResidentTokenRuntimeSchedulerError::UnknownStream(stream_id.to_string())
            })?
            .enqueue_input_event(event)?;
        self.schedule(stream_id);
        Ok(queued)
    }

    pub fn run_cycle(
        &mut self,
        device: &VulkanComputeDevice,
        max_runtime_cycles: usize,
        ticks_per_runtime: usize,
    ) -> Result<VulkanResidentTokenRuntimeSchedulerRun, VulkanResidentTokenRuntimeSchedulerError>
    {
        let mut runtime_cycles = Vec::new();
        let mut output_events = Vec::new();

        if max_runtime_cycles == 0 || ticks_per_runtime == 0 {
            return Ok(VulkanResidentTokenRuntimeSchedulerRun {
                max_runtime_cycles,
                ticks_per_runtime,
                stop_condition: if self.active_queue.is_empty() {
                    VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
                } else {
                    VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
                },
                runtime_cycles,
                output_events,
                active_runtime_count: self.active_queue.len(),
                registered_runtime_count: self.runtimes.len(),
            });
        }

        while runtime_cycles.len() < max_runtime_cycles {
            let Some(stream_id) = self.active_queue.pop_front() else {
                break;
            };
            let cycle = self
                .runtimes
                .get_mut(&stream_id)
                .ok_or_else(|| {
                    VulkanResidentTokenRuntimeSchedulerError::UnknownStream(stream_id.clone())
                })?
                .run_cycle(device, ticks_per_runtime)?;
            output_events.extend(cycle.output_events.iter().cloned().map(|output_event| {
                VulkanResidentTokenRuntimeSchedulerOutputEvent {
                    stream_id: stream_id.clone(),
                    output_event,
                }
            }));
            runtime_cycles.push(cycle);

            if self
                .runtimes
                .get(&stream_id)
                .map(|runtime| runtime.snapshot().running)
                .unwrap_or(false)
            {
                self.schedule(&stream_id);
            }
        }

        let stop_condition = if self.active_queue.is_empty() {
            VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
        } else {
            VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
        };

        Ok(VulkanResidentTokenRuntimeSchedulerRun {
            max_runtime_cycles,
            ticks_per_runtime,
            stop_condition,
            runtime_cycles,
            output_events,
            active_runtime_count: self.active_queue.len(),
            registered_runtime_count: self.runtimes.len(),
        })
    }

    pub fn snapshot(&self) -> VulkanResidentTokenRuntimeSchedulerSnapshot {
        let runtimes = self
            .runtimes
            .values()
            .map(VulkanResidentTokenRuntime::snapshot)
            .collect::<Vec<_>>();
        let running = runtimes.iter().any(|runtime| runtime.running);
        VulkanResidentTokenRuntimeSchedulerSnapshot {
            registered_runtime_count: self.runtimes.len(),
            active_runtime_count: self.active_queue.len(),
            idle: !running,
            running,
            runtimes,
        }
    }

    fn schedule(&mut self, stream_id: &str) {
        if !self.active_queue.iter().any(|active| active == stream_id) {
            self.active_queue.push_back(stream_id.to_string());
        }
    }
}

impl Default for VulkanResidentTokenRuntimeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum VulkanResidentTokenRuntimeSchedulerError {
    DuplicateStream(String),
    UnknownStream(String),
    Runtime(VulkanResidentGreedyFeedbackLoopRunnerError),
}

impl Display for VulkanResidentTokenRuntimeSchedulerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStream(stream_id) => {
                write!(
                    f,
                    "resident token runtime stream {stream_id:?} is already registered"
                )
            }
            Self::UnknownStream(stream_id) => {
                write!(
                    f,
                    "resident token runtime stream {stream_id:?} is not registered"
                )
            }
            Self::Runtime(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentTokenRuntimeSchedulerError {}

impl From<VulkanResidentGreedyFeedbackLoopRunnerError>
    for VulkanResidentTokenRuntimeSchedulerError
{
    fn from(error: VulkanResidentGreedyFeedbackLoopRunnerError) -> Self {
        Self::Runtime(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenRuntimeSchedulerStopCondition {
    Idle,
    RuntimeCycleBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSchedulerOutputEvent {
    pub stream_id: String,
    pub output_event: VulkanResidentTokenOutputEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSchedulerRun {
    pub max_runtime_cycles: usize,
    pub ticks_per_runtime: usize,
    pub stop_condition: VulkanResidentTokenRuntimeSchedulerStopCondition,
    pub runtime_cycles: Vec<VulkanResidentTokenRuntimeCycleRun>,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub active_runtime_count: usize,
    pub registered_runtime_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSchedulerSnapshot {
    pub registered_runtime_count: usize,
    pub active_runtime_count: usize,
    pub idle: bool,
    pub running: bool,
    pub runtimes: Vec<VulkanResidentTokenRuntimeSnapshot>,
}

pub struct VulkanMountedPlacedResidentPedalRunner {
    pub pedal_id: String,
    pub dispatches: Vec<VulkanMountedPlacedResidentPedalDispatch>,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
}

impl VulkanMountedPlacedResidentPedalRunner {
    fn from_mounted_bound_plan(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        pedal_id: &str,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        let mut dispatches = Vec::new();
        let mut total_descriptor_count = 0usize;
        let mut total_push_constant_byte_count = 0u32;

        for dispatch in mounted_bound_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.pedal_id == pedal_id)
        {
            let resident_dispatch = mounted.create_resident_kernel_dispatch_for_bound_dispatch(
                device,
                dispatch,
                loaded_manifest,
            )?;
            total_descriptor_count = total_descriptor_count
                .checked_add(resident_dispatch.descriptor_count())
                .ok_or(VulkanMountedPlacedResidentKernelDispatchError::PedalRunnerDescriptorCountOverflow {
                    pedal_id: pedal_id.to_string(),
                })?;
            total_push_constant_byte_count = total_push_constant_byte_count
                .checked_add(resident_dispatch.push_constant_byte_count())
                .ok_or(VulkanMountedPlacedResidentKernelDispatchError::PedalRunnerPushConstantByteCountOverflow {
                    pedal_id: pedal_id.to_string(),
                })?;
            dispatches.push(VulkanMountedPlacedResidentPedalDispatch {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                push_constants: dispatch.push_constants.clone(),
                resident_dispatch,
            });
        }

        if dispatches.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingPedalDispatches {
                    pedal_id: pedal_id.to_string(),
                },
            );
        }

        Ok(Self {
            pedal_id: pedal_id.to_string(),
            dispatches,
            total_descriptor_count,
            total_push_constant_byte_count,
        })
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    pub fn run_zeroed_push_constants(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanMountedPlacedResidentPedalRun, VulkanMountedPlacedResidentKernelDispatchError>
    {
        self.run_with_push_constant_bytes(device, |dispatch| {
            Ok(vec![
                0u8;
                dispatch.resident_dispatch.push_constant_byte_count()
                    as usize
            ])
        })
    }

    pub fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<VulkanMountedPlacedResidentPedalRun, VulkanMountedPlacedResidentKernelDispatchError>
    {
        self.run_with_push_constant_bytes(device, |dispatch| {
            stream_control_push_constant_bytes(&dispatch.push_constants, control)
        })
    }

    fn run_with_push_constant_bytes<F>(
        &self,
        device: &VulkanComputeDevice,
        mut push_constant_bytes_for: F,
    ) -> Result<VulkanMountedPlacedResidentPedalRun, VulkanMountedPlacedResidentKernelDispatchError>
    where
        F: FnMut(
            &VulkanMountedPlacedResidentPedalDispatch,
        ) -> Result<Vec<u8>, VulkanMountedPlacedResidentKernelDispatchError>,
    {
        let mut dispatch_runs = Vec::with_capacity(self.dispatches.len());
        for dispatch in &self.dispatches {
            let push_constants = push_constant_bytes_for(dispatch)?;
            device
                .run_resident_kernel_dispatch(&dispatch.resident_dispatch, &push_constants)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            dispatch_runs.push(VulkanMountedPlacedResidentPedalDispatchRun {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                descriptor_count: dispatch.resident_dispatch.descriptor_count(),
                workgroup_count_x: dispatch.resident_dispatch.workgroup_count_x(),
                push_constant_byte_count: dispatch.resident_dispatch.push_constant_byte_count(),
            });
        }

        Ok(VulkanMountedPlacedResidentPedalRun {
            pedal_id: self.pedal_id.clone(),
            dispatch_runs,
        })
    }
}

pub struct VulkanMountedPlacedResidentPedalDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub resident_dispatch: VulkanResidentKernelDispatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentPedalRun {
    pub pedal_id: String,
    pub dispatch_runs: Vec<VulkanMountedPlacedResidentPedalDispatchRun>,
}

impl VulkanMountedPlacedResidentPedalRun {
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_runs.len()
    }

    pub fn node_ids(&self) -> Vec<&str> {
        self.dispatch_runs
            .iter()
            .map(|dispatch| dispatch.node_id.as_str())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentPedalDispatchRun {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
}

pub struct VulkanMountedPlacedResidentPedalboardRunner {
    pub device_id: String,
    pub pedals: Vec<VulkanMountedPlacedResidentPedalRunner>,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
}

impl VulkanMountedPlacedResidentPedalboardRunner {
    fn from_mounted_bound_plan<I, S>(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        pedal_ids: I,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut pedals = Vec::new();
        let mut total_descriptor_count = 0usize;
        let mut total_push_constant_byte_count = 0u32;

        for pedal_id in pedal_ids {
            let pedal_id = pedal_id.as_ref();
            let runner = VulkanMountedPlacedResidentPedalRunner::from_mounted_bound_plan(
                device,
                mounted,
                mounted_bound_plan,
                pedal_id,
                loaded_manifest,
            )?;
            total_descriptor_count = total_descriptor_count
                .checked_add(runner.total_descriptor_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::PedalboardRunnerDescriptorCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            total_push_constant_byte_count = total_push_constant_byte_count
                .checked_add(runner.total_push_constant_byte_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::PedalboardRunnerPushConstantByteCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            pedals.push(runner);
        }

        if pedals.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingPedalboardPedals {
                    device_id: mounted_bound_plan.device_id.clone(),
                },
            );
        }

        Ok(Self {
            device_id: mounted_bound_plan.device_id.clone(),
            pedals,
            total_descriptor_count,
            total_push_constant_byte_count,
        })
    }

    pub fn pedal_count(&self) -> usize {
        self.pedals.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.pedals
            .iter()
            .map(VulkanMountedPlacedResidentPedalRunner::dispatch_count)
            .sum()
    }

    pub fn pedal_ids(&self) -> Vec<&str> {
        self.pedals
            .iter()
            .map(|pedal| pedal.pedal_id.as_str())
            .collect()
    }

    pub fn run_zeroed_push_constants(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<
        VulkanMountedPlacedResidentPedalboardRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut pedal_runs = Vec::with_capacity(self.pedals.len());
        for pedal in &self.pedals {
            pedal_runs.push(pedal.run_zeroed_push_constants(device)?);
        }

        Ok(VulkanMountedPlacedResidentPedalboardRun {
            device_id: self.device_id.clone(),
            pedal_runs,
        })
    }

    pub fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<
        VulkanMountedPlacedResidentPedalboardRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut pedal_runs = Vec::with_capacity(self.pedals.len());
        for pedal in &self.pedals {
            pedal_runs.push(pedal.run_with_stream_control(device, control)?);
        }

        Ok(VulkanMountedPlacedResidentPedalboardRun {
            device_id: self.device_id.clone(),
            pedal_runs,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentPedalboardRun {
    pub device_id: String,
    pub pedal_runs: Vec<VulkanMountedPlacedResidentPedalRun>,
}

impl VulkanMountedPlacedResidentPedalboardRun {
    pub fn pedal_count(&self) -> usize {
        self.pedal_runs.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.pedal_runs
            .iter()
            .map(VulkanMountedPlacedResidentPedalRun::dispatch_count)
            .sum()
    }

    pub fn pedal_ids(&self) -> Vec<&str> {
        self.pedal_runs
            .iter()
            .map(|pedal| pedal.pedal_id.as_str())
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamControl {
    pub stream_tick: u64,
    pub control_flags: u32,
    pub dynamic_state_capacity_activations: u32,
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
            .map(|state| ((state.pedal_id.as_str(), state.state_id.as_str()), state))
            .collect();
        let activation_index: BTreeMap<_, _> = resident_plan
            .activation_banks
            .iter()
            .flat_map(|bank| {
                bank.slots
                    .iter()
                    .map(move |slot| ((bank.pedal_id.as_str(), slot.slot), slot))
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
        pedal_id: &str,
        node_id: &str,
    ) -> Option<&VulkanDispatchDescriptorResourcePlan> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDispatchDescriptorResourcePlan {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
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
            pedal_id: command.pedal_id.clone(),
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
        pedal_id: String,
        slot: usize,
        byte_capacity: usize,
    },
    StateBuffer {
        pedal_id: String,
        state_id: String,
        state_type: String,
        byte_capacity: usize,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    StateView {
        pedal_id: String,
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
        VulkanKernelDescriptorResource::State { pedal_id, binding } => {
            resolve_state_descriptor_resource(
                command,
                descriptor,
                pedal_id,
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
            pedal_id,
            slot,
            bytes,
        } => {
            let resident = activation_index
                .get(&(pedal_id.as_str(), *slot))
                .ok_or_else(|| {
                    VulkanDescriptorResourcePlanError(format!(
                        "{} descriptor {} activation slot {}.{} is not resident",
                        command.kernel_id, descriptor.binding, pedal_id, slot
                    ))
                })?;
            let byte_capacity = resident.bytes.ok_or_else(|| {
                VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} activation slot {}.{} has unknown byte capacity",
                    command.kernel_id, descriptor.binding, pedal_id, slot
                ))
            })?;
            if *bytes != Some(byte_capacity) {
                return Err(VulkanDescriptorResourcePlanError(format!(
                    "{} descriptor {} activation slot {}.{} byte count {:?} does not match resident {}",
                    command.kernel_id, descriptor.binding, pedal_id, slot, bytes, byte_capacity
                )));
            }
            Ok(VulkanDescriptorResourceAddress::ActivationSlot {
                pedal_id: pedal_id.clone(),
                slot: *slot,
                byte_capacity,
            })
        }
        VulkanSignalResource::StateBuffer {
            pedal_id,
            state_id,
            static_bytes,
            bytes_per_activation,
        } => resolve_signal_state_descriptor_resource(
            command,
            descriptor,
            pedal_id,
            state_id,
            *static_bytes,
            *bytes_per_activation,
            state_index,
            dynamic_state_capacity_activations,
            false,
        ),
        VulkanSignalResource::StateView {
            pedal_id,
            state_id,
            static_bytes,
            bytes_per_activation,
        } => resolve_signal_state_descriptor_resource(
            command,
            descriptor,
            pedal_id,
            state_id,
            *static_bytes,
            *bytes_per_activation,
            state_index,
            dynamic_state_capacity_activations,
            true,
        ),
    }
}

fn resolve_signal_state_descriptor_resource(
    command: &VulkanKernelDispatchCommand,
    descriptor: &VulkanKernelDescriptorBinding,
    pedal_id: &str,
    state_id: &str,
    static_bytes: Option<usize>,
    bytes_per_activation: Option<usize>,
    state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
    dynamic_state_capacity_activations: usize,
    state_view: bool,
) -> Result<VulkanDescriptorResourceAddress, VulkanDescriptorResourcePlanError> {
    let resident = state_index.get(&(pedal_id, state_id)).ok_or_else(|| {
        VulkanDescriptorResourcePlanError(format!(
            "{} descriptor {} state {}.{} is not resident",
            command.kernel_id, descriptor.binding, pedal_id, state_id
        ))
    })?;
    if static_bytes != resident.static_bytes
        || bytes_per_activation != resident.bytes_per_activation
    {
        return Err(VulkanDescriptorResourcePlanError(format!(
            "{} descriptor {} state {}.{} byte shape {:?}/{:?} does not match resident {:?}/{:?}",
            command.kernel_id,
            descriptor.binding,
            pedal_id,
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
            pedal_id: pedal_id.to_string(),
            state_id: state_id.to_string(),
            state_type: resident.state_type.clone(),
            byte_capacity,
            static_bytes: resident.static_bytes,
            bytes_per_activation: resident.bytes_per_activation,
        })
    } else {
        Ok(VulkanDescriptorResourceAddress::StateBuffer {
            pedal_id: pedal_id.to_string(),
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
    pedal_id: &str,
    binding: &VulkanStateBinding,
    state_index: &BTreeMap<(&str, &str), &VulkanResidentStateBuffer>,
    dynamic_state_capacity_activations: usize,
    state_view: bool,
) -> Result<VulkanDescriptorResourceAddress, VulkanDescriptorResourcePlanError> {
    let resident = state_index
        .get(&(pedal_id, binding.state_id.as_str()))
        .ok_or_else(|| {
            VulkanDescriptorResourcePlanError(format!(
                "{} descriptor {} state {}.{} is not resident",
                command.kernel_id, descriptor.binding, pedal_id, binding.state_id
            ))
        })?;
    if binding.state_type != resident.state_type
        || binding.static_bytes != resident.static_bytes
        || binding.bytes_per_activation != resident.bytes_per_activation
    {
        return Err(VulkanDescriptorResourcePlanError(format!(
            "{} descriptor {} state {}.{} binding does not match resident allocation",
            command.kernel_id, descriptor.binding, pedal_id, binding.state_id
        )));
    }
    let byte_capacity =
        descriptor_state_byte_capacity(resident, dynamic_state_capacity_activations)?;
    if state_view {
        Ok(VulkanDescriptorResourceAddress::StateView {
            pedal_id: pedal_id.to_string(),
            state_id: binding.state_id.clone(),
            state_type: binding.state_type.clone(),
            byte_capacity,
            static_bytes: binding.static_bytes,
            bytes_per_activation: binding.bytes_per_activation,
        })
    } else {
        Ok(VulkanDescriptorResourceAddress::StateBuffer {
            pedal_id: pedal_id.to_string(),
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
    let static_bytes = state.static_bytes.unwrap_or(0);
    let dynamic_bytes = match state.bytes_per_activation {
        Some(bytes_per_activation) => {
            if dynamic_state_capacity_activations == 0 {
                return Err(VulkanDescriptorResourcePlanError(format!(
                    "{}.{} requires non-zero dynamic state capacity",
                    state.pedal_id, state.state_id
                )));
            }
            bytes_per_activation
                .checked_mul(dynamic_state_capacity_activations)
                .ok_or_else(|| {
                    VulkanDescriptorResourcePlanError(format!(
                        "{}.{} dynamic state byte capacity overflowed",
                        state.pedal_id, state.state_id
                    ))
                })?
        }
        None => 0,
    };
    let total = static_bytes.checked_add(dynamic_bytes).ok_or_else(|| {
        VulkanDescriptorResourcePlanError(format!(
            "{}.{} state byte capacity overflowed",
            state.pedal_id, state.state_id
        ))
    })?;
    if total == 0 {
        return Err(VulkanDescriptorResourcePlanError(format!(
            "{}.{} has unknown or zero byte capacity",
            state.pedal_id, state.state_id
        )));
    }
    Ok(total)
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

    pub fn link_artifacts(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> VulkanLinkedReusableKernelPlan {
        VulkanLinkedReusableKernelPlan::from_reusable_plan_and_manifest(self, manifest)
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanReusableKernelArtifactManifest {
    pub schema: String,
    pub backend_id: String,
    pub artifacts: Vec<VulkanReusableKernelArtifact>,
}

impl VulkanReusableKernelArtifactManifest {
    pub fn new(artifacts: Vec<VulkanReusableKernelArtifact>) -> Self {
        Self {
            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            artifacts,
        }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn with_artifact(mut self, artifact: VulkanReusableKernelArtifact) -> Self {
        self.artifacts.push(artifact);
        self
    }

    pub fn family_ids(&self) -> Vec<&str> {
        self.artifacts
            .iter()
            .map(|artifact| artifact.family_id.as_str())
            .collect()
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if manifest.schema != VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported reusable kernel manifest schema {:?}",
                    manifest.schema
                ),
            ));
        }
        Ok(manifest)
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn load_artifacts(
        &self,
        artifact_root: impl AsRef<Path>,
    ) -> io::Result<VulkanLoadedReusableKernelArtifactManifest> {
        VulkanLoadedReusableKernelArtifactManifest::from_manifest(self, artifact_root)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanReusableKernelArtifact {
    pub family_id: String,
    pub op: String,
    pub path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

impl VulkanReusableKernelArtifact {
    pub fn from_family(family: &VulkanReusableKernelFamily, path: impl Into<String>) -> Self {
        Self {
            family_id: family.family_id.clone(),
            op: family.op.clone(),
            path: path.into(),
            entry_point: DEFAULT_SPIRV_ENTRY_POINT.to_string(),
            local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
            descriptor_signature: family.descriptor_signature.clone(),
            push_constants: family.push_constants.clone(),
            uses_stream_tick: family.uses_stream_tick,
        }
    }

    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    pub fn with_local_size_x(mut self, local_size_x: u32) -> Self {
        self.local_size_x = local_size_x;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLoadedReusableKernelArtifactManifest {
    pub schema: String,
    pub backend_id: String,
    pub artifacts: Vec<VulkanLoadedReusableKernelArtifact>,
    pub total_word_count: usize,
}

impl VulkanLoadedReusableKernelArtifactManifest {
    pub fn from_manifest(
        manifest: &VulkanReusableKernelArtifactManifest,
        artifact_root: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let artifact_root = artifact_root.as_ref();
        let mut artifacts = Vec::with_capacity(manifest.artifacts.len());
        let mut total_word_count = 0usize;

        for artifact in &manifest.artifacts {
            let resolved_path =
                resolve_reusable_kernel_artifact_path(artifact_root, &artifact.path);
            let words = read_spirv_words(&resolved_path)?;
            total_word_count = total_word_count.checked_add(words.len()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "loaded reusable kernel word count overflowed",
                )
            })?;
            artifacts.push(VulkanLoadedReusableKernelArtifact {
                artifact: artifact.clone(),
                resolved_path,
                words,
            });
        }

        Ok(Self {
            schema: manifest.schema.clone(),
            backend_id: manifest.backend_id.clone(),
            artifacts,
            total_word_count,
        })
    }

    pub fn artifact(&self, family_id: &str) -> Option<&VulkanLoadedReusableKernelArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.artifact.family_id == family_id)
    }

    pub fn family_ids(&self) -> Vec<&str> {
        self.artifacts
            .iter()
            .map(|artifact| artifact.artifact.family_id.as_str())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLoadedReusableKernelArtifact {
    pub artifact: VulkanReusableKernelArtifact,
    pub resolved_path: PathBuf,
    pub words: Vec<u32>,
}

fn resolve_reusable_kernel_artifact_path(artifact_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLinkedReusableKernelPlan {
    pub backend_id: String,
    pub manifest_schema: String,
    pub manifest_backend_id: String,
    pub required_family_count: usize,
    pub linked_family_count: usize,
    pub missing_family_count: usize,
    pub incompatible_family_count: usize,
    pub required_command_count: usize,
    pub linked_command_count: usize,
    pub missing_command_count: usize,
    pub incompatible_command_count: usize,
    pub families: Vec<VulkanLinkedReusableKernelFamily>,
    pub issues: Vec<VulkanReusableKernelLinkIssue>,
}

impl VulkanLinkedReusableKernelPlan {
    pub fn from_reusable_plan_and_manifest(
        reusable_plan: &VulkanReusableKernelPlan,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Self {
        let mut artifacts_by_family_id: BTreeMap<&str, Vec<&VulkanReusableKernelArtifact>> =
            BTreeMap::new();
        for artifact in &manifest.artifacts {
            artifacts_by_family_id
                .entry(artifact.family_id.as_str())
                .or_default()
                .push(artifact);
        }

        let mut families = Vec::with_capacity(reusable_plan.families.len());
        let mut issues = Vec::new();
        let mut linked_family_count = 0usize;
        let mut missing_family_count = 0usize;
        let mut incompatible_family_count = 0usize;
        let mut linked_command_count = 0usize;
        let mut missing_command_count = 0usize;
        let mut incompatible_command_count = 0usize;

        for family in &reusable_plan.families {
            let command_count = family.command_refs.len();
            let artifacts = artifacts_by_family_id
                .get(family.family_id.as_str())
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut family_issues = Vec::new();

            if artifacts.is_empty() {
                family_issues.push(VulkanReusableKernelLinkIssue {
                    family_id: family.family_id.clone(),
                    op: family.op.clone(),
                    problem: VulkanReusableKernelLinkProblem::MissingArtifact,
                });
            } else if artifacts.len() > 1 {
                family_issues.push(VulkanReusableKernelLinkIssue {
                    family_id: family.family_id.clone(),
                    op: family.op.clone(),
                    problem: VulkanReusableKernelLinkProblem::DuplicateArtifact {
                        count: artifacts.len(),
                    },
                });
            }

            let artifact = artifacts.first().copied();
            if let Some(artifact) = artifact {
                family_issues.extend(link_compatibility_issues(family, artifact));
            }

            let (status, artifact_path) = if artifacts.is_empty() {
                missing_family_count += 1;
                missing_command_count += command_count;
                (VulkanReusableKernelLinkStatus::Missing, None)
            } else if family_issues.is_empty() {
                linked_family_count += 1;
                linked_command_count += command_count;
                (
                    VulkanReusableKernelLinkStatus::Linked,
                    artifact.map(|artifact| artifact.path.clone()),
                )
            } else {
                incompatible_family_count += 1;
                incompatible_command_count += command_count;
                (
                    VulkanReusableKernelLinkStatus::Incompatible,
                    artifact.map(|artifact| artifact.path.clone()),
                )
            };

            issues.extend(family_issues.iter().cloned());
            families.push(VulkanLinkedReusableKernelFamily {
                family_id: family.family_id.clone(),
                op: family.op.clone(),
                command_count,
                status,
                artifact_path,
                issues: family_issues,
            });
        }

        Self {
            backend_id: reusable_plan.backend_id.clone(),
            manifest_schema: manifest.schema.clone(),
            manifest_backend_id: manifest.backend_id.clone(),
            required_family_count: reusable_plan.families.len(),
            linked_family_count,
            missing_family_count,
            incompatible_family_count,
            required_command_count: reusable_plan.total_command_count,
            linked_command_count,
            missing_command_count,
            incompatible_command_count,
            families,
            issues,
        }
    }

    pub fn is_fully_linked(&self) -> bool {
        self.missing_family_count == 0
            && self.incompatible_family_count == 0
            && self.linked_command_count == self.required_command_count
    }

    pub fn family(&self, family_id: &str) -> Option<&VulkanLinkedReusableKernelFamily> {
        self.families
            .iter()
            .find(|family| family.family_id == family_id)
    }

    pub fn missing_families(&self) -> Vec<&VulkanLinkedReusableKernelFamily> {
        self.families
            .iter()
            .filter(|family| family.status == VulkanReusableKernelLinkStatus::Missing)
            .collect()
    }

    pub fn incompatible_families(&self) -> Vec<&VulkanLinkedReusableKernelFamily> {
        self.families
            .iter()
            .filter(|family| family.status == VulkanReusableKernelLinkStatus::Incompatible)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLinkedReusableKernelFamily {
    pub family_id: String,
    pub op: String,
    pub command_count: usize,
    pub status: VulkanReusableKernelLinkStatus,
    pub artifact_path: Option<String>,
    pub issues: Vec<VulkanReusableKernelLinkIssue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanReusableKernelLinkStatus {
    Linked,
    Missing,
    Incompatible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelLinkIssue {
    pub family_id: String,
    pub op: String,
    pub problem: VulkanReusableKernelLinkProblem,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanReusableKernelLinkProblem {
    MissingArtifact,
    DuplicateArtifact { count: usize },
    OpMismatch { found: String },
    DescriptorSignatureMismatch,
    PushConstantSignatureMismatch,
    StreamTickUsageMismatch { found: bool },
    EmptySpirvPath,
    UnsupportedEntryPoint { found: String },
    InvalidLocalSizeX { found: u32 },
}

fn link_compatibility_issues(
    family: &VulkanReusableKernelFamily,
    artifact: &VulkanReusableKernelArtifact,
) -> Vec<VulkanReusableKernelLinkIssue> {
    let mut issues = Vec::new();
    let family_id = family.family_id.clone();
    let op = family.op.clone();

    if artifact.op != family.op {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::OpMismatch {
                found: artifact.op.clone(),
            },
        });
    }
    if artifact.descriptor_signature != family.descriptor_signature {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::DescriptorSignatureMismatch,
        });
    }
    if artifact.push_constants != family.push_constants {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::PushConstantSignatureMismatch,
        });
    }
    if artifact.uses_stream_tick != family.uses_stream_tick {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::StreamTickUsageMismatch {
                found: artifact.uses_stream_tick,
            },
        });
    }
    if artifact.path.is_empty() {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::EmptySpirvPath,
        });
    }
    if artifact.entry_point != DEFAULT_SPIRV_ENTRY_POINT {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::UnsupportedEntryPoint {
                found: artifact.entry_point.clone(),
            },
        });
    }
    if artifact.local_size_x == 0 {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id,
            op,
            problem: VulkanReusableKernelLinkProblem::InvalidLocalSizeX {
                found: artifact.local_size_x,
            },
        });
    }

    issues
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPreparedDispatchPlan {
    pub backend_id: String,
    pub reusable_family_count: usize,
    pub dispatches: Vec<VulkanPreparedDispatch>,
    pub total_descriptor_count: usize,
}

impl VulkanPreparedDispatchPlan {
    pub fn from_plans(
        dispatch_plan: &VulkanKernelDispatchPlan,
        reusable_plan: &VulkanReusableKernelPlan,
        descriptor_plan: &VulkanDescriptorResourcePlan,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanPreparedDispatchPlanError> {
        let link_plan = reusable_plan.link_artifacts(manifest);
        if !link_plan.is_fully_linked() {
            return Err(VulkanPreparedDispatchPlanError::Link(link_plan));
        }

        let descriptor_by_dispatch: BTreeMap<usize, &VulkanDispatchDescriptorResourcePlan> =
            descriptor_plan
                .dispatches
                .iter()
                .map(|dispatch| (dispatch.dispatch_index, dispatch))
                .collect();
        let mut family_by_dispatch = BTreeMap::new();
        for family in &reusable_plan.families {
            for command_ref in &family.command_refs {
                family_by_dispatch.insert(command_ref.dispatch_index, family);
            }
        }
        let artifact_by_family: BTreeMap<_, _> = manifest
            .artifacts
            .iter()
            .map(|artifact| (artifact.family_id.as_str(), artifact))
            .collect();

        let mut dispatches = Vec::with_capacity(dispatch_plan.commands.len());
        for command in &dispatch_plan.commands {
            let descriptor_dispatch = descriptor_by_dispatch.get(&command.dispatch_index).ok_or(
                VulkanPreparedDispatchPlanError::MissingDescriptorResources {
                    dispatch_index: command.dispatch_index,
                },
            )?;
            let family = family_by_dispatch.get(&command.dispatch_index).ok_or(
                VulkanPreparedDispatchPlanError::MissingReusableFamily {
                    dispatch_index: command.dispatch_index,
                },
            )?;
            let artifact = artifact_by_family
                .get(family.family_id.as_str())
                .ok_or_else(|| VulkanPreparedDispatchPlanError::MissingLinkedArtifact {
                    family_id: family.family_id.clone(),
                })?;

            dispatches.push(VulkanPreparedDispatch {
                dispatch_index: command.dispatch_index,
                kernel_id: command.kernel_id.clone(),
                pedal_id: command.pedal_id.clone(),
                circuit_id: command.circuit_id.clone(),
                node_index: command.node_index,
                node_id: command.node_id.clone(),
                op: command.op.clone(),
                reusable_family_id: family.family_id.clone(),
                artifact_path: artifact.path.clone(),
                entry_point: artifact.entry_point.clone(),
                local_size_x: artifact.local_size_x,
                descriptors: descriptor_dispatch.descriptors.clone(),
                push_constants: command.push_constants.clone(),
                uses_stream_tick: command.uses_stream_tick,
            });
        }
        let total_descriptor_count = dispatches
            .iter()
            .map(|dispatch| dispatch.descriptors.len())
            .sum();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            reusable_family_count: reusable_plan.total_family_count(),
            dispatches,
            total_descriptor_count,
        })
    }

    pub fn dispatch(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanPreparedDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPreparedDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub artifact_path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptors: Vec<VulkanResolvedDescriptorBinding>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanPreparedDispatchPlanError {
    DescriptorResource(VulkanDescriptorResourcePlanError),
    Link(VulkanLinkedReusableKernelPlan),
    MissingDescriptorResources { dispatch_index: usize },
    MissingReusableFamily { dispatch_index: usize },
    MissingLinkedArtifact { family_id: String },
}

impl Display for VulkanPreparedDispatchPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DescriptorResource(error) => Display::fmt(error, f),
            Self::Link(plan) => write!(
                f,
                "reusable Vulkan kernels are not fully linked: {} missing families, {} incompatible families",
                plan.missing_family_count, plan.incompatible_family_count
            ),
            Self::MissingDescriptorResources { dispatch_index } => write!(
                f,
                "dispatch {dispatch_index} has no resolved descriptor resources"
            ),
            Self::MissingReusableFamily { dispatch_index } => {
                write!(f, "dispatch {dispatch_index} has no reusable kernel family")
            }
            Self::MissingLinkedArtifact { family_id } => {
                write!(f, "reusable family {family_id:?} has no linked artifact")
            }
        }
    }
}

impl Error for VulkanPreparedDispatchPlanError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBoundDispatchPlan {
    pub backend_id: String,
    pub dispatches: Vec<VulkanBoundDispatch>,
    pub total_descriptor_count: usize,
    pub boundary_descriptor_count: usize,
    pub permanent_parameter_descriptor_count: usize,
    pub stream_state_descriptor_count: usize,
    pub activation_slot_descriptor_count: usize,
}

impl VulkanBoundDispatchPlan {
    pub fn from_prepared_plan(
        prepared_plan: &VulkanPreparedDispatchPlan,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        let mut boundary_descriptor_count = 0usize;
        let mut permanent_parameter_descriptor_count = 0usize;
        let mut stream_state_descriptor_count = 0usize;
        let mut activation_slot_descriptor_count = 0usize;
        let mut dispatches = Vec::with_capacity(prepared_plan.dispatches.len());

        for prepared in &prepared_plan.dispatches {
            let mut descriptors = Vec::with_capacity(prepared.descriptors.len());
            for descriptor in &prepared.descriptors {
                let target =
                    VulkanBoundDescriptorTarget::from_resource(prepared, descriptor, buffers)?;
                match target {
                    VulkanBoundDescriptorTarget::BoundaryInput { .. }
                    | VulkanBoundDescriptorTarget::BoundaryOutput { .. } => {
                        boundary_descriptor_count += 1;
                    }
                    VulkanBoundDescriptorTarget::PermanentParameter { .. } => {
                        permanent_parameter_descriptor_count += 1;
                    }
                    VulkanBoundDescriptorTarget::StreamStateBuffer { .. }
                    | VulkanBoundDescriptorTarget::StreamStateView { .. } => {
                        stream_state_descriptor_count += 1;
                    }
                    VulkanBoundDescriptorTarget::ActivationSlot { .. } => {
                        activation_slot_descriptor_count += 1;
                    }
                }
                descriptors.push(VulkanBoundDescriptor {
                    binding: descriptor.binding,
                    usage: descriptor.usage.clone(),
                    name: descriptor.name.clone(),
                    target,
                });
            }

            dispatches.push(VulkanBoundDispatch {
                dispatch_index: prepared.dispatch_index,
                kernel_id: prepared.kernel_id.clone(),
                pedal_id: prepared.pedal_id.clone(),
                circuit_id: prepared.circuit_id.clone(),
                node_index: prepared.node_index,
                node_id: prepared.node_id.clone(),
                op: prepared.op.clone(),
                reusable_family_id: prepared.reusable_family_id.clone(),
                artifact_path: prepared.artifact_path.clone(),
                entry_point: prepared.entry_point.clone(),
                local_size_x: prepared.local_size_x,
                descriptors,
                push_constants: prepared.push_constants.clone(),
                uses_stream_tick: prepared.uses_stream_tick,
            });
        }

        let total_descriptor_count = boundary_descriptor_count
            + permanent_parameter_descriptor_count
            + stream_state_descriptor_count
            + activation_slot_descriptor_count;
        Ok(Self {
            backend_id: prepared_plan.backend_id.clone(),
            dispatches,
            total_descriptor_count,
            boundary_descriptor_count,
            permanent_parameter_descriptor_count,
            stream_state_descriptor_count,
            activation_slot_descriptor_count,
        })
    }

    pub fn dispatch(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanBoundDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedBoundDispatchPlan {
    pub backend_id: String,
    pub device_id: String,
    pub dispatches: Vec<VulkanPlacedBoundDispatch>,
    pub total_descriptor_count: usize,
    pub resident_descriptor_count: usize,
    pub model_boundary_descriptor_count: usize,
    pub local_cable_descriptor_count: usize,
    pub incoming_cable_descriptor_count: usize,
    pub outgoing_cable_descriptor_count: usize,
}

impl VulkanPlacedBoundDispatchPlan {
    pub fn from_bound_plan(
        bound_plan: &VulkanBoundDispatchPlan,
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Self {
        let mut resident_descriptor_count = 0usize;
        let mut model_boundary_descriptor_count = 0usize;
        let mut local_cable_descriptor_count = 0usize;
        let mut incoming_cable_descriptor_count = 0usize;
        let mut outgoing_cable_descriptor_count = 0usize;
        let mut dispatches = Vec::with_capacity(bound_plan.dispatches.len());

        for dispatch in &bound_plan.dispatches {
            let mut descriptors = Vec::with_capacity(dispatch.descriptors.len());
            for descriptor in &dispatch.descriptors {
                let target = VulkanPlacedBoundDescriptorTarget::from_bound_target(
                    &dispatch.pedal_id,
                    &descriptor.target,
                    placed_resident_plan,
                );
                match target {
                    VulkanPlacedBoundDescriptorTarget::Resident { .. } => {
                        resident_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::ModelInput { .. }
                    | VulkanPlacedBoundDescriptorTarget::ModelOutput { .. } => {
                        model_boundary_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::LocalCableInput { .. }
                    | VulkanPlacedBoundDescriptorTarget::LocalCableOutput { .. } => {
                        local_cable_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::IncomingCable { .. } => {
                        incoming_cable_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::OutgoingCable { .. } => {
                        outgoing_cable_descriptor_count += 1;
                    }
                }
                descriptors.push(VulkanPlacedBoundDescriptor {
                    binding: descriptor.binding,
                    usage: descriptor.usage.clone(),
                    name: descriptor.name.clone(),
                    target,
                });
            }

            dispatches.push(VulkanPlacedBoundDispatch {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                pedal_id: dispatch.pedal_id.clone(),
                circuit_id: dispatch.circuit_id.clone(),
                node_index: dispatch.node_index,
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                artifact_path: dispatch.artifact_path.clone(),
                entry_point: dispatch.entry_point.clone(),
                local_size_x: dispatch.local_size_x,
                descriptors,
                push_constants: dispatch.push_constants.clone(),
                uses_stream_tick: dispatch.uses_stream_tick,
            });
        }

        let total_descriptor_count = resident_descriptor_count
            + model_boundary_descriptor_count
            + local_cable_descriptor_count
            + incoming_cable_descriptor_count
            + outgoing_cable_descriptor_count;

        Self {
            backend_id: bound_plan.backend_id.clone(),
            device_id: placed_resident_plan.device_id.clone(),
            dispatches,
            total_descriptor_count,
            resident_descriptor_count,
            model_boundary_descriptor_count,
            local_cable_descriptor_count,
            incoming_cable_descriptor_count,
            outgoing_cable_descriptor_count,
        }
    }

    pub fn dispatch(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanPlacedBoundDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedBoundDispatchPlan {
    pub backend_id: String,
    pub device_id: String,
    pub dispatches: Vec<VulkanMountedPlacedBoundDispatch>,
    pub total_descriptor_count: usize,
    pub resident_descriptor_count: usize,
    pub model_boundary_descriptor_count: usize,
    pub local_cable_descriptor_count: usize,
    pub cable_endpoint_descriptor_count: usize,
    pub incoming_cable_descriptor_count: usize,
    pub outgoing_cable_descriptor_count: usize,
}

impl VulkanMountedPlacedBoundDispatchPlan {
    pub fn from_placed_bound_plan(
        placed_bound_plan: &VulkanPlacedBoundDispatchPlan,
        cable_io: &VulkanPlacedCableIoBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        if placed_bound_plan.device_id != cable_io.plan.device_id {
            return Err(VulkanBoundDispatchPlanError::CableIoDeviceMismatch {
                plan_device_id: placed_bound_plan.device_id.clone(),
                cable_io_device_id: cable_io.plan.device_id.clone(),
            });
        }

        let mut resident_descriptor_count = 0usize;
        let mut model_boundary_descriptor_count = 0usize;
        let mut local_cable_descriptor_count = 0usize;
        let mut cable_endpoint_descriptor_count = 0usize;
        let mut incoming_cable_descriptor_count = 0usize;
        let mut outgoing_cable_descriptor_count = 0usize;
        let mut dispatches = Vec::with_capacity(placed_bound_plan.dispatches.len());

        for dispatch in &placed_bound_plan.dispatches {
            let mut descriptors = Vec::with_capacity(dispatch.descriptors.len());
            for descriptor in &dispatch.descriptors {
                let target = VulkanMountedPlacedBoundDescriptorTarget::from_placed_target(
                    dispatch, descriptor, cable_io,
                )?;
                match target {
                    VulkanMountedPlacedBoundDescriptorTarget::Resident { .. } => {
                        resident_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::ModelInput { .. }
                    | VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { .. } => {
                        model_boundary_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer { .. }
                    | VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer { .. } => {
                        local_cable_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer { .. } => {
                        incoming_cable_descriptor_count += 1;
                        cable_endpoint_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer { .. } => {
                        outgoing_cable_descriptor_count += 1;
                        cable_endpoint_descriptor_count += 1;
                    }
                }
                descriptors.push(VulkanMountedPlacedBoundDescriptor {
                    binding: descriptor.binding,
                    usage: descriptor.usage.clone(),
                    name: descriptor.name.clone(),
                    target,
                });
            }

            dispatches.push(VulkanMountedPlacedBoundDispatch {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                pedal_id: dispatch.pedal_id.clone(),
                circuit_id: dispatch.circuit_id.clone(),
                node_index: dispatch.node_index,
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                artifact_path: dispatch.artifact_path.clone(),
                entry_point: dispatch.entry_point.clone(),
                local_size_x: dispatch.local_size_x,
                descriptors,
                push_constants: dispatch.push_constants.clone(),
                uses_stream_tick: dispatch.uses_stream_tick,
            });
        }

        let total_descriptor_count = resident_descriptor_count
            + model_boundary_descriptor_count
            + local_cable_descriptor_count
            + cable_endpoint_descriptor_count;

        Ok(Self {
            backend_id: placed_bound_plan.backend_id.clone(),
            device_id: placed_bound_plan.device_id.clone(),
            dispatches,
            total_descriptor_count,
            resident_descriptor_count,
            model_boundary_descriptor_count,
            local_cable_descriptor_count,
            cable_endpoint_descriptor_count,
            incoming_cable_descriptor_count,
            outgoing_cable_descriptor_count,
        })
    }

    pub fn dispatch(
        &self,
        pedal_id: &str,
        node_id: &str,
    ) -> Option<&VulkanMountedPlacedBoundDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedBoundDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub artifact_path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptors: Vec<VulkanMountedPlacedBoundDescriptor>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedBoundDescriptor {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub name: String,
    pub target: VulkanMountedPlacedBoundDescriptorTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedBoundDescriptorTarget {
    Resident {
        target: VulkanBoundDescriptorTarget,
    },
    ModelInput {
        signal_id: String,
    },
    ModelOutput {
        signal_id: String,
    },
    LocalCableInputBuffer {
        cable: VulkanPlacedLocalCableBufferBinding,
    },
    LocalCableOutputBuffer {
        cable: VulkanPlacedLocalCableBufferBinding,
    },
    IncomingCableBuffer {
        endpoint: VulkanPlacedCableEndpointBufferBinding,
    },
    OutgoingCableBuffer {
        endpoint: VulkanPlacedCableEndpointBufferBinding,
    },
}

impl VulkanMountedPlacedBoundDescriptorTarget {
    fn from_placed_target(
        dispatch: &VulkanPlacedBoundDispatch,
        descriptor: &VulkanPlacedBoundDescriptor,
        cable_io: &VulkanPlacedCableIoBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        match &descriptor.target {
            VulkanPlacedBoundDescriptorTarget::Resident { target } => Ok(Self::Resident {
                target: target.clone(),
            }),
            VulkanPlacedBoundDescriptorTarget::ModelInput { signal_id } => Ok(Self::ModelInput {
                signal_id: signal_id.clone(),
            }),
            VulkanPlacedBoundDescriptorTarget::ModelOutput { signal_id } => Ok(Self::ModelOutput {
                signal_id: signal_id.clone(),
            }),
            VulkanPlacedBoundDescriptorTarget::LocalCableInput { cable } => {
                Ok(Self::LocalCableInputBuffer {
                    cable: bind_local_cable_buffer(
                        dispatch,
                        descriptor,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
            VulkanPlacedBoundDescriptorTarget::LocalCableOutput { cable } => {
                Ok(Self::LocalCableOutputBuffer {
                    cable: bind_local_cable_buffer(
                        dispatch,
                        descriptor,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
            VulkanPlacedBoundDescriptorTarget::IncomingCable { cable } => {
                Ok(Self::IncomingCableBuffer {
                    endpoint: bind_cable_endpoint_buffer(
                        dispatch,
                        descriptor,
                        VulkanPlacedCableDirection::Incoming,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
            VulkanPlacedBoundDescriptorTarget::OutgoingCable { cable } => {
                Ok(Self::OutgoingCableBuffer {
                    endpoint: bind_cable_endpoint_buffer(
                        dispatch,
                        descriptor,
                        VulkanPlacedCableDirection::Outgoing,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentKernelDispatchReadinessPlan {
    pub backend_id: String,
    pub device_id: String,
    pub dispatches: Vec<VulkanMountedPlacedResidentKernelDispatchReadiness>,
    pub dispatch_count: usize,
    pub instantiable_count: usize,
    pub blocked_count: usize,
    pub missing_loaded_artifact_count: usize,
    pub descriptor_binding_blocked_count: usize,
    pub push_constant_blocked_count: usize,
    pub instantiable_descriptor_count: usize,
}

impl VulkanMountedPlacedResidentKernelDispatchReadinessPlan {
    fn from_mounted_bound_plan(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Self {
        let mut instantiable_count = 0usize;
        let mut blocked_count = 0usize;
        let mut missing_loaded_artifact_count = 0usize;
        let mut descriptor_binding_blocked_count = 0usize;
        let mut push_constant_blocked_count = 0usize;
        let mut instantiable_descriptor_count = 0usize;

        let dispatches = mounted_bound_plan
            .dispatches
            .iter()
            .map(|dispatch| {
                let status = mounted.resident_kernel_dispatch_readiness_for_bound_dispatch(
                    dispatch,
                    loaded_manifest,
                );
                match &status {
                    VulkanMountedPlacedResidentKernelDispatchStatus::Instantiable {
                        descriptor_count,
                        ..
                    } => {
                        instantiable_count += 1;
                        instantiable_descriptor_count += descriptor_count;
                    }
                    VulkanMountedPlacedResidentKernelDispatchStatus::Blocked { error } => {
                        blocked_count += 1;
                        match error {
                            VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                                ..
                            } => missing_loaded_artifact_count += 1,
                            VulkanMountedPlacedResidentKernelDispatchError::UnsupportedPushConstantScalar {
                                ..
                            }
                            | VulkanMountedPlacedResidentKernelDispatchError::UnsupportedPushConstantBinding {
                                ..
                            }
                            | VulkanMountedPlacedResidentKernelDispatchError::PushConstantByteCountOverflow => {
                                push_constant_blocked_count += 1;
                            }
                            _ => descriptor_binding_blocked_count += 1,
                        }
                    }
                }
                VulkanMountedPlacedResidentKernelDispatchReadiness {
                    dispatch_index: dispatch.dispatch_index,
                    kernel_id: dispatch.kernel_id.clone(),
                    pedal_id: dispatch.pedal_id.clone(),
                    node_id: dispatch.node_id.clone(),
                    op: dispatch.op.clone(),
                    reusable_family_id: dispatch.reusable_family_id.clone(),
                    status,
                }
            })
            .collect::<Vec<_>>();

        let dispatch_count = dispatches.len();
        Self {
            backend_id: mounted_bound_plan.backend_id.clone(),
            device_id: mounted_bound_plan.device_id.clone(),
            dispatches,
            dispatch_count,
            instantiable_count,
            blocked_count,
            missing_loaded_artifact_count,
            descriptor_binding_blocked_count,
            push_constant_blocked_count,
            instantiable_descriptor_count,
        }
    }

    pub fn dispatch(
        &self,
        pedal_id: &str,
        node_id: &str,
    ) -> Option<&VulkanMountedPlacedResidentKernelDispatchReadiness> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentKernelDispatchReadiness {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub status: VulkanMountedPlacedResidentKernelDispatchStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedResidentKernelDispatchStatus {
    Instantiable {
        descriptor_count: usize,
        workgroup_count_x: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
    },
    Blocked {
        error: VulkanMountedPlacedResidentKernelDispatchError,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedLocalCableBufferBinding {
    pub buffer_index: usize,
    pub cable: VulkanPlacedLocalCable,
    pub byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableEndpointBufferBinding {
    pub buffer_index: usize,
    pub endpoint: VulkanPlacedCableEndpoint,
    pub byte_capacity: usize,
}

fn bind_local_cable_buffer(
    dispatch: &VulkanPlacedBoundDispatch,
    descriptor: &VulkanPlacedBoundDescriptor,
    cable_index: usize,
    cable_io: &VulkanPlacedCableIoBuffers,
) -> Result<VulkanPlacedLocalCableBufferBinding, VulkanBoundDispatchPlanError> {
    let (buffer_index, allocation) = cable_io.local_buffer(cable_index).ok_or_else(|| {
        VulkanBoundDispatchPlanError::MissingLocalCableBuffer {
            dispatch_index: dispatch.dispatch_index,
            binding: descriptor.binding,
            cable_index,
        }
    })?;
    if allocation.cable.byte_capacity != Some(allocation.byte_capacity) {
        return Err(
            VulkanBoundDispatchPlanError::LocalCableByteCapacityMismatch {
                dispatch_index: dispatch.dispatch_index,
                binding: descriptor.binding,
                cable_index,
                cable_byte_capacity: allocation.cable.byte_capacity,
                mounted_byte_capacity: allocation.byte_capacity,
            },
        );
    }

    Ok(VulkanPlacedLocalCableBufferBinding {
        buffer_index,
        cable: allocation.cable.clone(),
        byte_capacity: allocation.byte_capacity,
    })
}

fn bind_cable_endpoint_buffer(
    dispatch: &VulkanPlacedBoundDispatch,
    descriptor: &VulkanPlacedBoundDescriptor,
    direction: VulkanPlacedCableDirection,
    cable_index: usize,
    cable_io: &VulkanPlacedCableIoBuffers,
) -> Result<VulkanPlacedCableEndpointBufferBinding, VulkanBoundDispatchPlanError> {
    let (buffer_index, allocation) = cable_io.buffer(direction, cable_index).ok_or_else(|| {
        VulkanBoundDispatchPlanError::MissingCableEndpointBuffer {
            dispatch_index: dispatch.dispatch_index,
            binding: descriptor.binding,
            direction,
            cable_index,
        }
    })?;
    if allocation.endpoint.byte_capacity != Some(allocation.byte_capacity) {
        return Err(
            VulkanBoundDispatchPlanError::CableEndpointByteCapacityMismatch {
                dispatch_index: dispatch.dispatch_index,
                binding: descriptor.binding,
                cable_index,
                endpoint_byte_capacity: allocation.endpoint.byte_capacity,
                mounted_byte_capacity: allocation.byte_capacity,
            },
        );
    }

    Ok(VulkanPlacedCableEndpointBufferBinding {
        buffer_index,
        endpoint: allocation.endpoint.clone(),
        byte_capacity: allocation.byte_capacity,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickPlan {
    pub backend_id: String,
    pub device_id: String,
    pub stages: Vec<VulkanMountedPlacedStreamTickStage>,
    pub stage_count: usize,
    pub receive_stage_count: usize,
    pub dispatch_stage_count: usize,
    pub publish_stage_count: usize,
    pub local_cable_read_count: usize,
    pub local_cable_write_count: usize,
    pub incoming_cable_read_count: usize,
    pub outgoing_cable_write_count: usize,
    pub model_input_read_count: usize,
    pub model_output_write_count: usize,
    pub can_execute: bool,
}

impl VulkanMountedPlacedStreamTickPlan {
    pub fn from_mounted_bound_plan(
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
    ) -> Self {
        let mut stages = Vec::new();
        let mut incoming_endpoints =
            BTreeMap::<(usize, usize), VulkanPlacedCableEndpointBufferBinding>::new();
        let mut outgoing_endpoints =
            BTreeMap::<(usize, usize), VulkanPlacedCableEndpointBufferBinding>::new();

        let mut local_cable_read_count = 0usize;
        let mut local_cable_write_count = 0usize;
        let mut incoming_cable_read_count = 0usize;
        let mut outgoing_cable_write_count = 0usize;
        let mut model_input_read_count = 0usize;
        let mut model_output_write_count = 0usize;

        let dispatch_stages = mounted_bound_plan
            .dispatches
            .iter()
            .map(|dispatch| {
                let dispatch_stage =
                    VulkanMountedPlacedStreamTickDispatch::from_bound_dispatch(dispatch);
                local_cable_read_count += dispatch_stage
                    .reads
                    .iter()
                    .filter(|io| {
                        matches!(io, VulkanMountedPlacedStreamTickIo::LocalCableBuffer { .. })
                    })
                    .count();
                local_cable_write_count += dispatch_stage
                    .writes
                    .iter()
                    .filter(|io| {
                        matches!(io, VulkanMountedPlacedStreamTickIo::LocalCableBuffer { .. })
                    })
                    .count();
                incoming_cable_read_count += dispatch_stage
                    .reads
                    .iter()
                    .filter(|io| {
                        matches!(
                            io,
                            VulkanMountedPlacedStreamTickIo::IncomingCableBuffer { .. }
                        )
                    })
                    .count();
                outgoing_cable_write_count += dispatch_stage
                    .writes
                    .iter()
                    .filter(|io| {
                        matches!(
                            io,
                            VulkanMountedPlacedStreamTickIo::OutgoingCableBuffer { .. }
                        )
                    })
                    .count();
                model_input_read_count += dispatch_stage
                    .reads
                    .iter()
                    .filter(|io| matches!(io, VulkanMountedPlacedStreamTickIo::ModelSignal { .. }))
                    .count();
                model_output_write_count += dispatch_stage
                    .writes
                    .iter()
                    .filter(|io| matches!(io, VulkanMountedPlacedStreamTickIo::ModelSignal { .. }))
                    .count();

                for descriptor in &dispatch.descriptors {
                    match &descriptor.target {
                        VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer {
                            endpoint,
                        } => {
                            incoming_endpoints
                                .entry((endpoint.endpoint.cable_index, endpoint.buffer_index))
                                .or_insert_with(|| endpoint.clone());
                        }
                        VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer {
                            endpoint,
                        } => {
                            outgoing_endpoints
                                .entry((endpoint.endpoint.cable_index, endpoint.buffer_index))
                                .or_insert_with(|| endpoint.clone());
                        }
                        _ => {}
                    }
                }

                dispatch_stage
            })
            .collect::<Vec<_>>();

        for endpoint in incoming_endpoints.values() {
            stages.push(VulkanMountedPlacedStreamTickStage::ReceiveCable {
                stage_index: stages.len(),
                cable_index: endpoint.endpoint.cable_index,
                endpoint_id: endpoint.endpoint.endpoint_id.clone(),
                buffer_index: endpoint.buffer_index,
                byte_capacity: endpoint.byte_capacity,
                remote_device_id: endpoint.endpoint.remote_device_id.clone(),
                remote_pedal_id: endpoint.endpoint.remote_pedal_id.clone(),
            });
        }

        for dispatch in dispatch_stages {
            stages.push(VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index: stages.len(),
                dispatch,
            });
        }

        for endpoint in outgoing_endpoints.values() {
            stages.push(VulkanMountedPlacedStreamTickStage::PublishCable {
                stage_index: stages.len(),
                cable_index: endpoint.endpoint.cable_index,
                endpoint_id: endpoint.endpoint.endpoint_id.clone(),
                buffer_index: endpoint.buffer_index,
                byte_capacity: endpoint.byte_capacity,
                remote_device_id: endpoint.endpoint.remote_device_id.clone(),
                remote_pedal_id: endpoint.endpoint.remote_pedal_id.clone(),
            });
        }

        let receive_stage_count = incoming_endpoints.len();
        let publish_stage_count = outgoing_endpoints.len();
        let dispatch_stage_count = mounted_bound_plan.dispatches.len();
        let stage_count = stages.len();

        Self {
            backend_id: mounted_bound_plan.backend_id.clone(),
            device_id: mounted_bound_plan.device_id.clone(),
            stages,
            stage_count,
            receive_stage_count,
            dispatch_stage_count,
            publish_stage_count,
            local_cable_read_count,
            local_cable_write_count,
            incoming_cable_read_count,
            outgoing_cable_write_count,
            model_input_read_count,
            model_output_write_count,
            can_execute: false,
        }
    }

    pub fn advance(&self, stream_tick: u64) -> VulkanMountedPlacedStreamTickRun {
        let mut stages = Vec::with_capacity(self.stages.len());
        let mut blocked = None;
        let mut attempted_stage_count = 0usize;
        let mut completed_stage_count = 0usize;

        for stage in &self.stages {
            let status = if blocked.is_some() {
                VulkanMountedPlacedStreamTickStageStatus::Pending
            } else {
                attempted_stage_count += 1;
                let reason = match stage {
                    VulkanMountedPlacedStreamTickStage::ReceiveCable { .. } => {
                        VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable
                    }
                    VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {
                        VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable
                    }
                    VulkanMountedPlacedStreamTickStage::PublishCable { .. } => {
                        VulkanMountedPlacedStreamTickBlockReason::CablePublishTransportUnavailable
                    }
                };
                blocked = Some((stage.stage_index(), reason.clone()));
                VulkanMountedPlacedStreamTickStageStatus::Blocked { reason }
            };
            if matches!(status, VulkanMountedPlacedStreamTickStageStatus::Completed) {
                completed_stage_count += 1;
            }
            stages.push(VulkanMountedPlacedStreamTickStageRun {
                stage_index: stage.stage_index(),
                stage: stage.clone(),
                status,
            });
        }

        let pending_stage_count = stages
            .iter()
            .filter(|stage| {
                matches!(
                    stage.status,
                    VulkanMountedPlacedStreamTickStageStatus::Pending
                )
            })
            .count();
        let status = blocked
            .map(
                |(stage_index, reason)| VulkanMountedPlacedStreamTickRunStatus::Blocked {
                    stage_index,
                    reason,
                },
            )
            .unwrap_or(VulkanMountedPlacedStreamTickRunStatus::Completed);

        VulkanMountedPlacedStreamTickRun {
            backend_id: self.backend_id.clone(),
            device_id: self.device_id.clone(),
            stream_tick,
            stages,
            planned_stage_count: self.stage_count,
            attempted_stage_count,
            completed_stage_count,
            pending_stage_count,
            status,
            can_execute: self.can_execute,
        }
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickError {
    BoundDispatchPlan(VulkanBoundDispatchPlanError),
}

impl Display for VulkanMountedPlacedStreamTickError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundDispatchPlan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanMountedPlacedStreamTickError {}

impl From<VulkanBoundDispatchPlanError> for VulkanMountedPlacedStreamTickError {
    fn from(error: VulkanBoundDispatchPlanError) -> Self {
        Self::BoundDispatchPlan(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedResidentKernelDispatchError {
    MissingPedalboardPedals {
        device_id: String,
    },
    MissingPedalDispatches {
        pedal_id: String,
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
    MissingOutputDescriptorForWorkgroup {
        dispatch_index: usize,
        op: String,
    },
    MissingOutputBindingForWorkgroup {
        dispatch_index: usize,
        binding: usize,
    },
    InvalidOutputByteLengthForWorkgroup {
        dispatch_index: usize,
        binding: usize,
        byte_len: usize,
    },
    WorkgroupCountOverflow {
        dispatch_index: usize,
        output_element_count: usize,
        local_size_x: u32,
    },
    PedalRunnerDescriptorCountOverflow {
        pedal_id: String,
    },
    PedalRunnerPushConstantByteCountOverflow {
        pedal_id: String,
    },
    PedalboardRunnerDescriptorCountOverflow {
        device_id: String,
    },
    PedalboardRunnerPushConstantByteCountOverflow {
        device_id: String,
    },
    Vulkan(VulkanError),
}

impl Display for VulkanMountedPlacedResidentKernelDispatchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingPedalboardPedals { device_id } => {
                write!(
                    f,
                    "resident pedalboard runner for device {device_id:?} has no pedals"
                )
            }
            Self::MissingPedalDispatches { pedal_id } => {
                write!(f, "pedal {pedal_id:?} has no mounted dispatches")
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
            Self::MissingOutputDescriptorForWorkgroup { dispatch_index, op } => write!(
                f,
                "dispatch {dispatch_index} op {op:?} cannot plan workgroups because it has no output descriptor"
            ),
            Self::MissingOutputBindingForWorkgroup {
                dispatch_index,
                binding,
            } => write!(
                f,
                "dispatch {dispatch_index} cannot plan workgroups because output descriptor binding {binding} is not mounted"
            ),
            Self::InvalidOutputByteLengthForWorkgroup {
                dispatch_index,
                binding,
                byte_len,
            } => write!(
                f,
                "dispatch {dispatch_index} cannot plan workgroups because output descriptor binding {binding} has invalid BF16 byte length {byte_len}"
            ),
            Self::WorkgroupCountOverflow {
                dispatch_index,
                output_element_count,
                local_size_x,
            } => write!(
                f,
                "dispatch {dispatch_index} cannot plan workgroups for {output_element_count} output elements with local_size_x {local_size_x}"
            ),
            Self::PedalRunnerDescriptorCountOverflow { pedal_id } => write!(
                f,
                "resident pedal runner {pedal_id:?} descriptor count overflowed"
            ),
            Self::PedalRunnerPushConstantByteCountOverflow { pedal_id } => write!(
                f,
                "resident pedal runner {pedal_id:?} push-constant byte count overflowed"
            ),
            Self::PedalboardRunnerDescriptorCountOverflow { device_id } => write!(
                f,
                "resident pedalboard runner for device {device_id:?} descriptor count overflowed"
            ),
            Self::PedalboardRunnerPushConstantByteCountOverflow { device_id } => write!(
                f,
                "resident pedalboard runner for device {device_id:?} push-constant byte count overflowed"
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

fn resident_kernel_dispatch_workgroup_count_x(
    dispatch: &VulkanMountedPlacedBoundDispatch,
    buffer_bindings: &[VulkanResidentKernelBufferBinding<'_>],
) -> Result<u32, VulkanMountedPlacedResidentKernelDispatchError> {
    if dispatch.op != "linear" {
        return Ok(1);
    }
    let output_descriptor = dispatch
        .descriptors
        .iter()
        .find(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
        .ok_or_else(|| {
            VulkanMountedPlacedResidentKernelDispatchError::MissingOutputDescriptorForWorkgroup {
                dispatch_index: dispatch.dispatch_index,
                op: dispatch.op.clone(),
            }
        })?;
    let output_binding = u32::try_from(output_descriptor.binding).map_err(|_| {
        VulkanMountedPlacedResidentKernelDispatchError::DescriptorBindingOverflow {
            dispatch_index: dispatch.dispatch_index,
            binding: output_descriptor.binding,
        }
    })?;
    let output_buffer = buffer_bindings
        .iter()
        .find(|binding| binding.binding == output_binding)
        .ok_or_else(|| {
            VulkanMountedPlacedResidentKernelDispatchError::MissingOutputBindingForWorkgroup {
                dispatch_index: dispatch.dispatch_index,
                binding: output_descriptor.binding,
            }
        })?;
    if output_buffer.byte_len == 0 || output_buffer.byte_len % 2 != 0 {
        return Err(
            VulkanMountedPlacedResidentKernelDispatchError::InvalidOutputByteLengthForWorkgroup {
                dispatch_index: dispatch.dispatch_index,
                binding: output_descriptor.binding,
                byte_len: output_buffer.byte_len,
            },
        );
    }
    let output_element_count = output_buffer.byte_len / 2;
    let local_size_x = usize::try_from(dispatch.local_size_x).map_err(|_| {
        VulkanMountedPlacedResidentKernelDispatchError::WorkgroupCountOverflow {
            dispatch_index: dispatch.dispatch_index,
            output_element_count,
            local_size_x: dispatch.local_size_x,
        }
    })?;
    if local_size_x == 0 {
        return Err(
            VulkanMountedPlacedResidentKernelDispatchError::WorkgroupCountOverflow {
                dispatch_index: dispatch.dispatch_index,
                output_element_count,
                local_size_x: dispatch.local_size_x,
            },
        );
    }
    let workgroup_count = output_element_count
        .checked_add(local_size_x - 1)
        .and_then(|rounded| rounded.checked_div(local_size_x))
        .ok_or_else(
            || VulkanMountedPlacedResidentKernelDispatchError::WorkgroupCountOverflow {
                dispatch_index: dispatch.dispatch_index,
                output_element_count,
                local_size_x: dispatch.local_size_x,
            },
        )?;
    u32::try_from(workgroup_count).map_err(|_| {
        VulkanMountedPlacedResidentKernelDispatchError::WorkgroupCountOverflow {
            dispatch_index: dispatch.dispatch_index,
            output_element_count,
            local_size_x: dispatch.local_size_x,
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedBoundDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
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
    LocalCableInput { cable: PedalCablePlacement },
    LocalCableOutput { cable: PedalCablePlacement },
    IncomingCable { cable: PedalCablePlacement },
    OutgoingCable { cable: PedalCablePlacement },
}

impl VulkanPlacedBoundDescriptorTarget {
    fn from_bound_target(
        pedal_id: &str,
        target: &VulkanBoundDescriptorTarget,
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Self {
        match target {
            VulkanBoundDescriptorTarget::BoundaryInput { signal_id } => {
                classify_boundary_input(pedal_id, signal_id, placed_resident_plan)
            }
            VulkanBoundDescriptorTarget::BoundaryOutput { signal_id } => {
                classify_boundary_output(pedal_id, signal_id, placed_resident_plan)
            }
            _ => Self::Resident {
                target: target.clone(),
            },
        }
    }
}

fn classify_boundary_input(
    pedal_id: &str,
    signal_id: &str,
    placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
) -> VulkanPlacedBoundDescriptorTarget {
    if let Some(cable) = placed_resident_plan
        .local_cables
        .iter()
        .find(|cable| {
            cable.destination_pedal_id == pedal_id && cable.destination_port_id == signal_id
        })
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::LocalCableInput { cable };
    }
    if let Some(cable) = placed_resident_plan
        .incoming_cables
        .iter()
        .find(|cable| {
            cable.destination_pedal_id == pedal_id && cable.destination_port_id == signal_id
        })
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::IncomingCable { cable };
    }
    VulkanPlacedBoundDescriptorTarget::ModelInput {
        signal_id: signal_id.to_string(),
    }
}

fn classify_boundary_output(
    pedal_id: &str,
    signal_id: &str,
    placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
) -> VulkanPlacedBoundDescriptorTarget {
    if let Some(cable) = placed_resident_plan
        .local_cables
        .iter()
        .find(|cable| cable.source_pedal_id == pedal_id && cable.source_port_id == signal_id)
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::LocalCableOutput { cable };
    }
    if let Some(cable) = placed_resident_plan
        .outgoing_cables
        .iter()
        .find(|cable| cable.source_pedal_id == pedal_id && cable.source_port_id == signal_id)
        .cloned()
    {
        return VulkanPlacedBoundDescriptorTarget::OutgoingCable { cable };
    }
    VulkanPlacedBoundDescriptorTarget::ModelOutput {
        signal_id: signal_id.to_string(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBoundDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
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
        pedal_id: String,
        circuit_id: String,
        slot: usize,
        byte_capacity: usize,
    },
    StreamStateBuffer {
        buffer_index: usize,
        pedal_id: String,
        state_id: String,
        state_type: String,
        byte_capacity: usize,
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    },
    StreamStateView {
        buffer_index: usize,
        pedal_id: String,
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
                pedal_id,
                slot,
                byte_capacity,
            } => {
                let buffer_index = buffers
                    .activation_slot_buffer_index(pedal_id, *slot)
                    .ok_or_else(
                        || VulkanBoundDispatchPlanError::MissingActivationSlotBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            pedal_id: pedal_id.clone(),
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
                    pedal_id: pedal_id.clone(),
                    circuit_id: buffer.circuit_id.clone(),
                    slot: *slot,
                    byte_capacity: *byte_capacity,
                })
            }
            VulkanDescriptorResourceAddress::StateBuffer {
                pedal_id,
                state_id,
                state_type,
                byte_capacity,
                static_bytes,
                bytes_per_activation,
            } => {
                let buffer_index =
                    buffers
                        .state_buffer_index(pedal_id, state_id)
                        .ok_or_else(|| VulkanBoundDispatchPlanError::MissingStateBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            pedal_id: pedal_id.clone(),
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
                    pedal_id: pedal_id.clone(),
                    state_id: state_id.clone(),
                    state_type: state_type.clone(),
                    byte_capacity: *byte_capacity,
                    static_bytes: *static_bytes,
                    bytes_per_activation: *bytes_per_activation,
                })
            }
            VulkanDescriptorResourceAddress::StateView {
                pedal_id,
                state_id,
                state_type,
                byte_capacity,
                static_bytes,
                bytes_per_activation,
            } => {
                let buffer_index =
                    buffers
                        .state_buffer_index(pedal_id, state_id)
                        .ok_or_else(|| VulkanBoundDispatchPlanError::MissingStateBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            pedal_id: pedal_id.clone(),
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
                    pedal_id: pedal_id.clone(),
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
    CableIoDeviceMismatch {
        plan_device_id: String,
        cable_io_device_id: String,
    },
    MissingStateBuffer {
        dispatch_index: usize,
        binding: usize,
        pedal_id: String,
        state_id: String,
    },
    MissingActivationSlotBuffer {
        dispatch_index: usize,
        binding: usize,
        pedal_id: String,
        slot: usize,
    },
    MissingCableEndpointBuffer {
        dispatch_index: usize,
        binding: usize,
        direction: VulkanPlacedCableDirection,
        cable_index: usize,
    },
    MissingLocalCableBuffer {
        dispatch_index: usize,
        binding: usize,
        cable_index: usize,
    },
    ByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        expected_byte_capacity: usize,
        mounted_byte_capacity: usize,
    },
    LocalCableByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        cable_index: usize,
        cable_byte_capacity: Option<usize>,
        mounted_byte_capacity: usize,
    },
    CableEndpointByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        cable_index: usize,
        endpoint_byte_capacity: Option<usize>,
        mounted_byte_capacity: usize,
    },
}

impl Display for VulkanBoundDispatchPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreparedDispatch(error) => Display::fmt(error, f),
            Self::CableIoDeviceMismatch {
                plan_device_id,
                cable_io_device_id,
            } => write!(
                f,
                "placed bound plan for device {plan_device_id:?} cannot bind cable I/O for device {cable_io_device_id:?}"
            ),
            Self::MissingStateBuffer {
                dispatch_index,
                binding,
                pedal_id,
                state_id,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing stream state buffer {pedal_id}.{state_id}"
            ),
            Self::MissingActivationSlotBuffer {
                dispatch_index,
                binding,
                pedal_id,
                slot,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing activation slot buffer {pedal_id}.slot_{slot}"
            ),
            Self::MissingCableEndpointBuffer {
                dispatch_index,
                binding,
                direction,
                cable_index,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing {direction:?} cable endpoint buffer for cable {cable_index}"
            ),
            Self::MissingLocalCableBuffer {
                dispatch_index,
                binding,
                cable_index,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} references missing local cable buffer for cable {cable_index}"
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
            Self::LocalCableByteCapacityMismatch {
                dispatch_index,
                binding,
                cable_index,
                cable_byte_capacity,
                mounted_byte_capacity,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} local cable {cable_index} expects {cable_byte_capacity:?} bytes but mounted buffer has {mounted_byte_capacity} bytes"
            ),
            Self::CableEndpointByteCapacityMismatch {
                dispatch_index,
                binding,
                cable_index,
                endpoint_byte_capacity,
                mounted_byte_capacity,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} cable {cable_index} endpoint expects {endpoint_byte_capacity:?} bytes but mounted buffer has {mounted_byte_capacity} bytes"
            ),
        }
    }
}

impl Error for VulkanBoundDispatchPlanError {}

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
    hosted_pedals: Option<&BTreeSet<String>>,
) -> Result<BTreeMap<(String, String), VulkanParameterBinding>, VulkanBindingPlanError> {
    let hosts_pedal = |pedal_id: &str| {
        hosted_pedals
            .map(|pedals| pedals.contains(pedal_id))
            .unwrap_or(true)
    };
    let resident_by_tensor: BTreeMap<_, _> = resident_plan
        .permanent_parameters
        .iter()
        .map(|parameter| (parameter.tensor.as_str(), parameter))
        .collect();
    let mut bindings = BTreeMap::new();

    for parameter in &resource_plan.parameters {
        let hosted_uses = parameter
            .uses
            .iter()
            .filter(|use_ref| hosts_pedal(&use_ref.pedal_id))
            .collect::<Vec<_>>();
        if hosted_uses.is_empty() {
            continue;
        }
        let resident = resident_by_tensor
            .get(parameter.tensor.as_str())
            .ok_or_else(|| {
                VulkanBindingPlanError(format!(
                    "resident plan has no permanent parameter for tensor {:?}",
                    parameter.tensor
                ))
            })?;
        for use_ref in hosted_uses {
            let key = (use_ref.pedal_id.clone(), use_ref.param_id.clone());
            let previous = bindings.insert(
                key.clone(),
                VulkanParameterBinding {
                    param_id: use_ref.param_id.clone(),
                    tensor: parameter.tensor.clone(),
                    byte_count: resident.byte_count,
                    shape: resident.shape.clone(),
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
        input_ports: circuit.input_ports.clone(),
        output_ports: circuit.output_ports.clone(),
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

pub const LFM2_DEFAULT_LAST_LAYER_INDEX: usize = 13;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLfm2ResidentGreedyStreamProcessorConfig {
    pub device_id: String,
    pub circuit_index_path: PathBuf,
    pub tensor_index_path: PathBuf,
    pub dynamic_state_capacity_activations: usize,
    pub attention_shader: String,
}

impl VulkanLfm2ResidentGreedyStreamProcessorConfig {
    pub fn default_for_capacity(
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
        let attention_shader = lfm2_default_attention_shader_for_capacity(
            dynamic_state_capacity_activations,
        )
        .ok_or_else(|| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "no default LFM2 attention shader exists for resident capacity {dynamic_state_capacity_activations}; available defaults cover 1..=8 activations"
            ))
        })?;

        Ok(Self {
            device_id: "gpu0".to_string(),
            circuit_index_path: default_lfm2_5_230m_circuit_index_path(),
            tensor_index_path: default_lfm2_5_230m_tensor_index_path(),
            dynamic_state_capacity_activations,
            attention_shader: attention_shader.to_string(),
        })
    }
}

impl Default for VulkanLfm2ResidentGreedyStreamProcessorConfig {
    fn default() -> Self {
        Self {
            device_id: "gpu0".to_string(),
            circuit_index_path: default_lfm2_5_230m_circuit_index_path(),
            tensor_index_path: default_lfm2_5_230m_tensor_index_path(),
            dynamic_state_capacity_activations: 4,
            attention_shader: "gqa_attention_bf16_q16_kv8_d64_cap4.comp".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLfm2ResidentGreedyStreamProcessorBuildError(pub String);

impl Display for VulkanLfm2ResidentGreedyStreamProcessorBuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanLfm2ResidentGreedyStreamProcessorBuildError {}

pub fn default_lfm2_5_230m_circuit_index_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("lowered")
        .join("lfm2_5_230m")
        .join("pedalboard.circuits.json")
}

pub fn default_lfm2_5_230m_tensor_index_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("transpiled")
        .join("lfm2_5_230m")
        .join("tensors.json")
}

pub fn lfm2_default_attention_shader_for_capacity(capacity: usize) -> Option<&'static str> {
    match capacity {
        1..=4 => Some("gqa_attention_bf16_q16_kv8_d64_cap4.comp"),
        5..=8 => Some("gqa_attention_bf16_q16_kv8_d64_cap8.comp"),
        _ => None,
    }
}

pub fn create_default_lfm2_5_230m_resident_greedy_stream_processor(
    device: &VulkanComputeDevice,
    dynamic_state_capacity_activations: usize,
) -> Result<VulkanResidentGreedyStreamProcessor, VulkanLfm2ResidentGreedyStreamProcessorBuildError>
{
    let config = VulkanLfm2ResidentGreedyStreamProcessorConfig::default_for_capacity(
        dynamic_state_capacity_activations,
    )?;
    create_lfm2_resident_greedy_stream_processor_from_config(device, &config)
}

pub fn create_lfm2_resident_greedy_stream_processor_from_config(
    device: &VulkanComputeDevice,
    config: &VulkanLfm2ResidentGreedyStreamProcessorConfig,
) -> Result<VulkanResidentGreedyStreamProcessor, VulkanLfm2ResidentGreedyStreamProcessorBuildError>
{
    if config.dynamic_state_capacity_activations == 0 {
        return Err(VulkanLfm2ResidentGreedyStreamProcessorBuildError(
            "resident dynamic state capacity must be at least 1 activation".to_string(),
        ));
    }

    let (tensor_index, resource_plan, mounted, mounted_bound) =
        lfm2_mount_single_device_stream_circuit(device, config)?;
    let loaded_manifest =
        lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
            &mounted,
            &mounted_bound,
            &config.attention_shader,
        )?;
    let input_transducer_spirv_words =
        compile_required_lfm2_shader("embedding_lookup_bf16_65536x1024.comp")?;
    let embedding_norm_spirv_words = compile_required_lfm2_shader("rms_norm_bf16_serial.comp")?;
    let tied_projection_spirv_words =
        compile_required_lfm2_shader("tied_output_projection_bf16_65536x1024_to_f32.comp")?;
    let sampler_spirv_words = compile_required_lfm2_shader("greedy_sampler_f32_65536.comp")?;

    let transducer_parameter_buffers = lfm2_load_transducer_parameter_buffers(
        device,
        &config.device_id,
        &resource_plan,
        &tensor_index,
    )?;
    let pedal_ids =
        lfm2_prepare_resident_prefix(&mounted, &tensor_index, LFM2_DEFAULT_LAST_LAYER_INDEX)?;
    let input_transducer =
        VulkanResidentInputEmbeddingTransducerRunner::from_mounted_lfm2_token_embedding(
            device,
            &mounted,
            &transducer_parameter_buffers,
            &input_transducer_spirv_words,
        )
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to create LFM2 input token embedding transducer: {error}"
            ))
        })?;
    let pedalboard = mounted
        .create_resident_pedalboard_runner(
            device,
            &mounted_bound,
            pedal_ids.iter().map(String::as_str),
            &loaded_manifest,
        )
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to create LFM2 resident pedalboard runner: {error}"
            ))
        })?;
    let output_transducer =
        VulkanResidentOutputTransducerRunner::from_mounted_lfm2_output_transducer(
            device,
            &mounted,
            &transducer_parameter_buffers,
            &embedding_norm_spirv_words,
            &tied_projection_spirv_words,
        )
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to create LFM2 output transducer: {error}"
            ))
        })?;
    let sampler = VulkanResidentGreedySamplerRunner::from_output_transducer(
        device,
        &output_transducer,
        &sampler_spirv_words,
    )
    .map_err(|error| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to create LFM2 greedy sampler pedal: {error}"
        ))
    })?;
    let tick_runner =
        VulkanResidentSingleTokenTickRunner::new(input_transducer, pedalboard, output_transducer)
            .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to create LFM2 single-token tick runner: {error}"
            ))
        })?;
    let loop_runner =
        VulkanResidentGreedyFeedbackLoopRunner::new(tick_runner, sampler).map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to create LFM2 greedy feedback loop runner: {error}"
            ))
        })?;

    Ok(VulkanResidentGreedyStreamProcessor::new(
        mounted,
        transducer_parameter_buffers,
        loop_runner,
    ))
}

fn lfm2_mount_single_device_stream_circuit(
    device: &VulkanComputeDevice,
    config: &VulkanLfm2ResidentGreedyStreamProcessorConfig,
) -> Result<
    (
        TensorIndex,
        StreamCircuitResourcePlan,
        VulkanMountedPlacedStreamCircuit,
        VulkanMountedPlacedBoundDispatchPlan,
    ),
    VulkanLfm2ResidentGreedyStreamProcessorBuildError,
> {
    let graph = ResolvedLoweredPedalboard::from_index_file(&config.circuit_index_path).map_err(
        |error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to load LFM2 lowered pedalboard {:?}: {error}",
                config.circuit_index_path
            ))
        },
    )?;
    let tensor_index = TensorIndex::from_json_file(&config.tensor_index_path).map_err(|error| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to load LFM2 tensor index {:?}: {error}",
            config.tensor_index_path
        ))
    })?;
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).map_err(
            |error| {
                VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                    "failed to create LFM2 stream execution plan: {error}"
                ))
            },
        )?;
    let resource_plan = StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan)
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to create LFM2 stream resource plan: {error}"
            ))
        })?;
    let placement_spec = StreamCircuitPlacementSpec::new(config.device_id.as_str());
    let placement_plan = graph.placement_plan(&placement_spec).map_err(|error| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to create LFM2 placement plan for {:?}: {error}",
            config.device_id
        ))
    })?;
    let resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        &config.device_id,
        Some(&tensor_index),
        Some(2),
    )
    .map_err(|error| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to create LFM2 Vulkan resident plan for {:?}: {error}",
            config.device_id
        ))
    })?;
    let placed_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, resident)
            .map_err(|error| {
                VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                    "failed to create LFM2 Vulkan placed stream circuit plan: {error}"
                ))
            })?;
    let mounted = VulkanMountedPlacedStreamCircuit::from_placed_plan(
        device,
        placed_plan,
        config.dynamic_state_capacity_activations,
    )
    .map_err(|error| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to mount LFM2 Vulkan stream circuit: {error}"
        ))
    })?;
    let manifest = VulkanReusableKernelArtifactManifest::new(
        mounted
            .placed_plan
            .reusable_kernel_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );
    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&manifest)
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to bind LFM2 Vulkan stream circuit dispatch plan: {error}"
            ))
        })?;

    Ok((tensor_index, resource_plan, mounted, mounted_bound))
}

fn lfm2_load_transducer_parameter_buffers(
    device: &VulkanComputeDevice,
    device_id: &str,
    resource_plan: &StreamCircuitResourcePlan,
    tensor_index: &TensorIndex,
) -> Result<VulkanPermanentParameterBuffers, VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    let transducer_parameter_plan = VulkanPermanentParameterBufferPlan::from_transducer_parameters(
        device_id,
        resource_plan,
        Some(tensor_index),
    )
    .map_err(|error| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to create LFM2 transducer parameter plan: {error}"
        ))
    })?;
    let transducer_parameter_buffers =
        transducer_parameter_plan
            .allocate_buffers(device)
            .map_err(|error| {
                VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                    "failed to allocate LFM2 transducer parameter buffers: {error}"
                ))
            })?;
    transducer_parameter_buffers
        .load_from_tensor_index(tensor_index)
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to load LFM2 transducer parameters: {error}"
            ))
        })?;
    Ok(transducer_parameter_buffers)
}

fn lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    attention_shader: &str,
) -> Result<
    VulkanLoadedReusableKernelArtifactManifest,
    VulkanLfm2ResidentGreedyStreamProcessorBuildError,
> {
    lfm2_loaded_kernel_pack_for_dispatch_shaders(
        mounted,
        mounted_bound,
        &[
            ("layer_00", "operator_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_00",
                "conv_in_projection",
                "linear_bf16_1024x3072.comp",
            ),
            ("layer_00", "split_b_c_x", "split_bf16_3072_to_3x1024.comp"),
            ("layer_00", "input_gate", "multiply_bf16_1024.comp"),
            (
                "layer_00",
                "temporal_memory_update",
                "rolling_state_update_bf16_3x1024.comp",
            ),
            (
                "layer_00",
                "depthwise_temporal_conv",
                "depthwise_conv1d_bf16_3x1024.comp",
            ),
            ("layer_00", "output_gate", "multiply_bf16_1024.comp"),
            (
                "layer_00",
                "conv_out_projection",
                "linear_bf16_1024x1024.comp",
            ),
            ("layer_00", "operator_residual", "add_bf16_1024.comp"),
            ("layer_00", "ffn_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_00",
                "ffn_gate_projection",
                "linear_bf16_1024x2560.comp",
            ),
            (
                "layer_00",
                "ffn_up_projection",
                "linear_bf16_1024x2560.comp",
            ),
            ("layer_00", "ffn_gate_activation", "silu_bf16_2560.comp"),
            ("layer_00", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
            (
                "layer_00",
                "ffn_down_projection",
                "linear_bf16_2560x1024.comp",
            ),
            ("layer_00", "ffn_residual", "add_bf16_1024.comp"),
            ("layer_02", "operator_norm", "rms_norm_bf16_serial.comp"),
            ("layer_02", "q_projection", "linear_bf16_1024x1024.comp"),
            ("layer_02", "k_projection", "linear_bf16_1024x512.comp"),
            ("layer_02", "v_projection", "linear_bf16_1024x512.comp"),
            (
                "layer_02",
                "q_head_norm",
                "rms_norm_per_head_bf16_16x64.comp",
            ),
            (
                "layer_02",
                "k_head_norm",
                "rms_norm_per_head_bf16_8x64.comp",
            ),
            ("layer_02", "q_rope", "rotary_bf16_16x64.comp"),
            ("layer_02", "k_rope", "rotary_bf16_8x64.comp"),
            (
                "layer_02",
                "kv_memory_append",
                "append_kv_state_bf16_8x64.comp",
            ),
            ("layer_02", "attention_read", attention_shader),
            (
                "layer_02",
                "attention_out_projection",
                "linear_bf16_1024x1024.comp",
            ),
            ("layer_02", "operator_residual", "add_bf16_1024.comp"),
            ("layer_02", "ffn_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_02",
                "ffn_gate_projection",
                "linear_bf16_1024x2560.comp",
            ),
            (
                "layer_02",
                "ffn_up_projection",
                "linear_bf16_1024x2560.comp",
            ),
            ("layer_02", "ffn_gate_activation", "silu_bf16_2560.comp"),
            ("layer_02", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
            (
                "layer_02",
                "ffn_down_projection",
                "linear_bf16_2560x1024.comp",
            ),
            ("layer_02", "ffn_residual", "add_bf16_1024.comp"),
        ],
    )
}

fn lfm2_loaded_kernel_pack_for_dispatch_shaders(
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    dispatch_shaders: &[(&str, &str, &str)],
) -> Result<
    VulkanLoadedReusableKernelArtifactManifest,
    VulkanLfm2ResidentGreedyStreamProcessorBuildError,
> {
    let mut loaded_artifacts = Vec::new();
    let mut loaded_families = BTreeSet::new();
    let mut total_word_count = 0usize;

    for (pedal_id, node_id, shader_file) in dispatch_shaders {
        let dispatch = mounted_bound.dispatch(pedal_id, node_id).ok_or_else(|| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "LFM2 mounted dispatch {pedal_id}.{node_id} is missing"
            ))
        })?;
        if !loaded_families.insert(dispatch.reusable_family_id.clone()) {
            continue;
        }
        let spirv_words = compile_required_lfm2_shader(shader_file)?;
        total_word_count = total_word_count
            .checked_add(spirv_words.len())
            .ok_or_else(|| {
                VulkanLfm2ResidentGreedyStreamProcessorBuildError(
                    "LFM2 reusable kernel artifact word count overflowed".to_string(),
                )
            })?;
        let family = mounted
            .placed_plan
            .reusable_kernel_plan
            .family(&dispatch.reusable_family_id)
            .ok_or_else(|| {
                VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                    "LFM2 reusable kernel family {:?} is missing",
                    dispatch.reusable_family_id
                ))
            })?;
        let artifact_path = format!("kernels/{}.spv", dispatch.reusable_family_id);
        loaded_artifacts.push(VulkanLoadedReusableKernelArtifact {
            artifact: VulkanReusableKernelArtifact::from_family(family, artifact_path.clone()),
            resolved_path: PathBuf::from(artifact_path),
            words: spirv_words,
        });
    }

    Ok(VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts: loaded_artifacts,
        total_word_count,
    })
}

fn compile_required_lfm2_shader(
    shader_file: &str,
) -> Result<Vec<u32>, VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    crate::vulkan_compute::compile_shader_words_from_source(shader_file).ok_or_else(|| {
        VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "failed to compile Vulkan shader {shader_file:?}; install glslangValidator or glslc and check runtime-rs/shaders/{shader_file}"
        ))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lfm2LayerKind {
    ShortConv,
    Attention,
}

fn lfm2_layer_kind(
    layer_index: usize,
) -> Result<Lfm2LayerKind, VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    match layer_index {
        0 | 1 | 3 | 5 | 7 | 9 | 11 | 13 => Ok(Lfm2LayerKind::ShortConv),
        2 | 4 | 6 | 8 | 10 | 12 => Ok(Lfm2LayerKind::Attention),
        _ => Err(VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
            "unknown LFM2 layer index {layer_index}"
        ))),
    }
}

fn lfm2_layer_id(layer_index: usize) -> String {
    format!("layer_{layer_index:02}")
}

fn lfm2_prefix_pedal_ids(last_layer_index: usize) -> Vec<String> {
    (0..=last_layer_index).map(lfm2_layer_id).collect()
}

fn lfm2_prepare_resident_prefix(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    last_layer_index: usize,
) -> Result<Vec<String>, VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    for layer_index in 0..=last_layer_index {
        lfm2_load_layer_parameters(mounted, tensor_index, layer_index)?;
    }

    for layer_index in 0..=last_layer_index {
        lfm2_zero_layer_state(mounted, layer_index)?;
    }

    Ok(lfm2_prefix_pedal_ids(last_layer_index))
}

fn lfm2_load_layer_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
) -> Result<(), VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    match lfm2_layer_kind(layer_index)? {
        Lfm2LayerKind::ShortConv => {
            lfm2_load_conv_layer_parameters(mounted, tensor_index, layer_index)
        }
        Lfm2LayerKind::Attention => {
            lfm2_load_attention_layer_parameters(mounted, tensor_index, layer_index)
        }
    }
}

fn lfm2_load_conv_layer_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
) -> Result<(), VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    for suffix in [
        "operator_norm.weight",
        "conv.in_proj.weight",
        "conv.conv.weight",
        "conv.out_proj.weight",
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ] {
        lfm2_load_parameter(mounted, tensor_index, layer_index, suffix)?;
    }
    Ok(())
}

fn lfm2_load_attention_layer_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
) -> Result<(), VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    for suffix in [
        "operator_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_layernorm.weight",
        "self_attn.k_layernorm.weight",
        "self_attn.out_proj.weight",
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ] {
        lfm2_load_parameter(mounted, tensor_index, layer_index, suffix)?;
    }
    Ok(())
}

fn lfm2_load_parameter(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
    suffix: &str,
) -> Result<(), VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    let tensor = format!("model.layers.{layer_index}.{suffix}");
    mounted
        .parameter_buffers
        .load_parameter_from_tensor_index(tensor_index, &tensor)
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to load LFM2 parameter {tensor:?}: {error}"
            ))
        })?;
    Ok(())
}

fn lfm2_zero_layer_state(
    mounted: &VulkanMountedPlacedStreamCircuit,
    layer_index: usize,
) -> Result<(), VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    let pedal_id = lfm2_layer_id(layer_index);
    match lfm2_layer_kind(layer_index)? {
        Lfm2LayerKind::ShortConv => lfm2_zero_state_buffer(mounted, &pedal_id, "temporal_memory"),
        Lfm2LayerKind::Attention => lfm2_zero_state_buffer(mounted, &pedal_id, "kv_memory"),
    }
}

fn lfm2_zero_state_buffer(
    mounted: &VulkanMountedPlacedStreamCircuit,
    pedal_id: &str,
    state_id: &str,
) -> Result<(), VulkanLfm2ResidentGreedyStreamProcessorBuildError> {
    let state = mounted
        .buffers
        .state_buffer(pedal_id, state_id)
        .ok_or_else(|| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "LFM2 state buffer {pedal_id}.{state_id} is missing"
            ))
        })?;
    state
        .buffer
        .write_bytes(&vec![0u8; state.byte_capacity])
        .map_err(|error| {
            VulkanLfm2ResidentGreedyStreamProcessorBuildError(format!(
                "failed to zero LFM2 state buffer {pedal_id}.{state_id}: {error}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::stream_circuit::{ResolvedLoweredPedalboard, StreamCircuitPlacementSpec};
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

    fn mount_lfm2_single_device_stream_circuit(
        device: &VulkanComputeDevice,
    ) -> (
        TensorIndex,
        VulkanMountedPlacedStreamCircuit,
        VulkanReusableKernelArtifactManifest,
        VulkanMountedPlacedBoundDispatchPlan,
    ) {
        mount_lfm2_single_device_stream_circuit_with_capacity(device, 4)
    }

    fn mount_lfm2_single_device_stream_circuit_with_capacity(
        device: &VulkanComputeDevice,
        dynamic_state_capacity_activations: usize,
    ) -> (
        TensorIndex,
        VulkanMountedPlacedStreamCircuit,
        VulkanReusableKernelArtifactManifest,
        VulkanMountedPlacedBoundDispatchPlan,
    ) {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let placement_spec = StreamCircuitPlacementSpec::new("gpu0");
        let placement_plan = graph.placement_plan(&placement_spec).unwrap();
        let resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let placed_plan =
            VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, resident)
                .unwrap();
        let mounted = VulkanMountedPlacedStreamCircuit::from_placed_plan(
            device,
            placed_plan,
            dynamic_state_capacity_activations,
        )
        .unwrap();
        let manifest = VulkanReusableKernelArtifactManifest::new(
            mounted
                .placed_plan
                .reusable_kernel_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );
        let mounted_bound = mounted
            .mounted_placed_bound_dispatch_plan(&manifest)
            .unwrap();
        (tensor_index, mounted, manifest, mounted_bound)
    }

    fn load_layer_00_parameters(
        mounted: &VulkanMountedPlacedStreamCircuit,
        tensor_index: &TensorIndex,
    ) {
        load_lfm2_conv_layer_parameters(mounted, tensor_index, 0);
    }

    fn lfm2_embedding_row_bytes(tensor_index: &TensorIndex, token_id: u32) -> Vec<u8> {
        let metadata = tensor_index.tensors.get(LFM2_EMBED_TOKENS_TENSOR).unwrap();
        let offsets = metadata.data_offsets.as_ref().unwrap();
        let data_start = offsets[0];
        let row_offset = usize::try_from(token_id).unwrap() * LFM2_FRAME_BYTES;
        let absolute_tensor_offset = data_start + row_offset;
        let source_file = metadata.source_file.as_ref().unwrap();
        let data_base = safetensors_data_start(Path::new(source_file)).unwrap();
        let mut file = fs::File::open(source_file).unwrap();
        file.seek(SeekFrom::Start(
            data_base + u64::try_from(absolute_tensor_offset).unwrap(),
        ))
        .unwrap();
        let mut bytes = vec![0u8; LFM2_FRAME_BYTES];
        file.read_exact(&mut bytes).unwrap();
        bytes
    }

    fn load_lfm2_transducer_parameter_buffers(
        device: &VulkanComputeDevice,
        tensor_index: &TensorIndex,
    ) -> VulkanPermanentParameterBuffers {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, tensor_index).unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let transducer_parameter_plan =
            VulkanPermanentParameterBufferPlan::from_transducer_parameters(
                "gpu0",
                &resource_plan,
                Some(tensor_index),
            )
            .unwrap();
        assert_eq!(transducer_parameter_plan.parameter_count, 2);
        assert_eq!(
            transducer_parameter_plan.total_byte_capacity,
            Some(134_219_776)
        );
        assert!(transducer_parameter_plan.unresolved_tensors.is_empty());
        let transducer_parameter_buffers =
            transducer_parameter_plan.allocate_buffers(device).unwrap();
        let loaded = transducer_parameter_buffers
            .load_from_tensor_index(tensor_index)
            .unwrap();
        assert_eq!(loaded.parameter_count, 2);
        assert_eq!(loaded.loaded_count, 2);
        assert_eq!(loaded.total_bytes_loaded, 134_219_776);
        transducer_parameter_buffers
    }

    fn load_lfm2_conv_layer_parameters(
        mounted: &VulkanMountedPlacedStreamCircuit,
        tensor_index: &TensorIndex,
        layer_index: usize,
    ) {
        for suffix in [
            "operator_norm.weight",
            "conv.in_proj.weight",
            "conv.conv.weight",
            "conv.out_proj.weight",
            "ffn_norm.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        ] {
            let tensor = format!("model.layers.{layer_index}.{suffix}");
            mounted
                .parameter_buffers
                .load_parameter_from_tensor_index(tensor_index, &tensor)
                .unwrap();
        }
    }

    fn load_lfm2_attention_layer_parameters(
        mounted: &VulkanMountedPlacedStreamCircuit,
        tensor_index: &TensorIndex,
        layer_index: usize,
    ) {
        for suffix in [
            "operator_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_layernorm.weight",
            "self_attn.k_layernorm.weight",
            "self_attn.out_proj.weight",
            "ffn_norm.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        ] {
            let tensor = format!("model.layers.{layer_index}.{suffix}");
            mounted
                .parameter_buffers
                .load_parameter_from_tensor_index(tensor_index, &tensor)
                .unwrap();
        }
    }

    fn write_layer_00_unit_input_and_zero_state(mounted: &VulkanMountedPlacedStreamCircuit) {
        write_layer_00_constant_input(mounted, [0x80, 0x3f]);
        zero_lfm2_temporal_memory(mounted, "layer_00");
    }

    fn write_layer_00_constant_input(
        mounted: &VulkanMountedPlacedStreamCircuit,
        bf16_little_endian: [u8; 2],
    ) {
        let mut input_frame = Vec::with_capacity(2_048);
        for _ in 0..1024 {
            input_frame.extend_from_slice(&bf16_little_endian);
        }
        mounted
            .boundary_io
            .input_buffer("input_frame")
            .unwrap()
            .buffer
            .write_bytes(&input_frame)
            .unwrap();
    }

    fn write_lfm2_constant_output_frame(
        mounted: &VulkanMountedPlacedStreamCircuit,
        bf16_little_endian: [u8; 2],
    ) {
        let mut output_frame = Vec::with_capacity(LFM2_FRAME_BYTES);
        for _ in 0..LFM2_HIDDEN_SIZE {
            output_frame.extend_from_slice(&bf16_little_endian);
        }
        mounted
            .boundary_io
            .output_buffer(LFM2_OUTPUT_FRAME_SIGNAL)
            .unwrap()
            .buffer
            .write_bytes(&output_frame)
            .unwrap();
    }

    fn zero_lfm2_temporal_memory(mounted: &VulkanMountedPlacedStreamCircuit, pedal_id: &str) {
        let temporal_memory = mounted
            .buffers
            .state_buffer(pedal_id, "temporal_memory")
            .unwrap();
        temporal_memory
            .buffer
            .write_bytes(&vec![0u8; temporal_memory.byte_capacity])
            .unwrap();
    }

    fn zero_lfm2_kv_memory(mounted: &VulkanMountedPlacedStreamCircuit, pedal_id: &str) {
        let kv_memory = mounted.buffers.state_buffer(pedal_id, "kv_memory").unwrap();
        kv_memory
            .buffer
            .write_bytes(&vec![0u8; kv_memory.byte_capacity])
            .unwrap();
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Lfm2LayerKind {
        ShortConv,
        Attention,
    }

    fn lfm2_layer_kind(layer_index: usize) -> Lfm2LayerKind {
        match layer_index {
            0 | 1 | 3 | 5 | 7 | 9 | 11 | 13 => Lfm2LayerKind::ShortConv,
            2 | 4 | 6 | 8 | 10 | 12 => Lfm2LayerKind::Attention,
            _ => panic!("unknown LFM2 layer index {layer_index}"),
        }
    }

    fn lfm2_layer_id(layer_index: usize) -> String {
        format!("layer_{layer_index:02}")
    }

    fn lfm2_prefix_pedal_ids(last_layer_index: usize) -> Vec<String> {
        (0..=last_layer_index).map(lfm2_layer_id).collect()
    }

    fn load_lfm2_layer_parameters(
        mounted: &VulkanMountedPlacedStreamCircuit,
        tensor_index: &TensorIndex,
        layer_index: usize,
    ) {
        match lfm2_layer_kind(layer_index) {
            Lfm2LayerKind::ShortConv => {
                load_lfm2_conv_layer_parameters(mounted, tensor_index, layer_index);
            }
            Lfm2LayerKind::Attention => {
                load_lfm2_attention_layer_parameters(mounted, tensor_index, layer_index);
            }
        }
    }

    fn zero_lfm2_layer_state(mounted: &VulkanMountedPlacedStreamCircuit, layer_index: usize) {
        let pedal_id = lfm2_layer_id(layer_index);
        match lfm2_layer_kind(layer_index) {
            Lfm2LayerKind::ShortConv => {
                zero_lfm2_temporal_memory(mounted, &pedal_id);
            }
            Lfm2LayerKind::Attention => {
                zero_lfm2_kv_memory(mounted, &pedal_id);
            }
        }
    }

    fn prepare_lfm2_resident_prefix(
        mounted: &VulkanMountedPlacedStreamCircuit,
        tensor_index: &TensorIndex,
        last_layer_index: usize,
    ) -> Vec<String> {
        for layer_index in 0..=last_layer_index {
            load_lfm2_layer_parameters(mounted, tensor_index, layer_index);
        }

        write_layer_00_unit_input_and_zero_state(mounted);
        for layer_index in 1..=last_layer_index {
            zero_lfm2_layer_state(mounted, layer_index);
        }

        lfm2_prefix_pedal_ids(last_layer_index)
    }

    fn lfm2_stream_control(
        mounted: &VulkanMountedPlacedStreamCircuit,
        stream_tick: u64,
    ) -> VulkanMountedPlacedStreamControl {
        VulkanMountedPlacedStreamControl {
            stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations: mounted.buffers.dynamic_state_capacity_activations
                as u32,
        }
    }

    fn create_lfm2_resident_prefix_runner(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        pedal_ids: &[String],
    ) -> VulkanMountedPlacedResidentPedalboardRunner {
        mounted
            .create_resident_pedalboard_runner(
                device,
                mounted_bound,
                pedal_ids.iter().map(String::as_str),
                loaded_manifest,
            )
            .unwrap()
    }

    fn assert_lfm2_resident_prefix_runner(
        runner: &VulkanMountedPlacedResidentPedalboardRunner,
        pedal_ids: &[String],
        dispatch_count: usize,
        descriptor_count: usize,
        push_constant_byte_count: u32,
    ) {
        let expected_pedal_ids = pedal_ids.iter().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(runner.device_id, "gpu0");
        assert_eq!(runner.pedal_count(), pedal_ids.len());
        assert_eq!(runner.pedal_ids(), expected_pedal_ids);
        assert_eq!(runner.dispatch_count(), dispatch_count);
        assert_eq!(runner.total_descriptor_count, descriptor_count);
        assert_eq!(
            runner.total_push_constant_byte_count,
            push_constant_byte_count
        );
    }

    fn assert_lfm2_resident_prefix_run(
        run: &VulkanMountedPlacedResidentPedalboardRun,
        pedal_ids: &[String],
        dispatch_count: usize,
    ) {
        let expected_pedal_ids = pedal_ids.iter().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(run.device_id, "gpu0");
        assert_eq!(run.pedal_count(), pedal_ids.len());
        assert_eq!(run.pedal_ids(), expected_pedal_ids);
        assert_eq!(run.dispatch_count(), dispatch_count);
    }

    fn layer_00_level_1_loaded_kernel_pack(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    ) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
        loaded_kernel_pack_for_dispatch_shaders(
            mounted,
            mounted_bound,
            &[
                ("layer_00", "operator_norm", "rms_norm_bf16_serial.comp"),
                (
                    "layer_00",
                    "conv_in_projection",
                    "linear_bf16_1024x3072.comp",
                ),
                ("layer_00", "split_b_c_x", "split_bf16_3072_to_3x1024.comp"),
                ("layer_00", "input_gate", "multiply_bf16_1024.comp"),
                (
                    "layer_00",
                    "temporal_memory_update",
                    "rolling_state_update_bf16_3x1024.comp",
                ),
                (
                    "layer_00",
                    "depthwise_temporal_conv",
                    "depthwise_conv1d_bf16_3x1024.comp",
                ),
                ("layer_00", "output_gate", "multiply_bf16_1024.comp"),
                (
                    "layer_00",
                    "conv_out_projection",
                    "linear_bf16_1024x1024.comp",
                ),
                ("layer_00", "operator_residual", "add_bf16_1024.comp"),
                ("layer_00", "ffn_norm", "rms_norm_bf16_serial.comp"),
                (
                    "layer_00",
                    "ffn_gate_projection",
                    "linear_bf16_1024x2560.comp",
                ),
                (
                    "layer_00",
                    "ffn_up_projection",
                    "linear_bf16_1024x2560.comp",
                ),
                ("layer_00", "ffn_gate_activation", "silu_bf16_2560.comp"),
                ("layer_00", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
                (
                    "layer_00",
                    "ffn_down_projection",
                    "linear_bf16_2560x1024.comp",
                ),
                ("layer_00", "ffn_residual", "add_bf16_1024.comp"),
            ],
        )
    }

    fn lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    ) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
        lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
            mounted,
            mounted_bound,
            "gqa_attention_bf16_q16_kv8_d64_cap4.comp",
        )
    }

    fn lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
        attention_shader: &str,
    ) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
        loaded_kernel_pack_for_dispatch_shaders(
            mounted,
            mounted_bound,
            &[
                ("layer_00", "operator_norm", "rms_norm_bf16_serial.comp"),
                (
                    "layer_00",
                    "conv_in_projection",
                    "linear_bf16_1024x3072.comp",
                ),
                ("layer_00", "split_b_c_x", "split_bf16_3072_to_3x1024.comp"),
                ("layer_00", "input_gate", "multiply_bf16_1024.comp"),
                (
                    "layer_00",
                    "temporal_memory_update",
                    "rolling_state_update_bf16_3x1024.comp",
                ),
                (
                    "layer_00",
                    "depthwise_temporal_conv",
                    "depthwise_conv1d_bf16_3x1024.comp",
                ),
                ("layer_00", "output_gate", "multiply_bf16_1024.comp"),
                (
                    "layer_00",
                    "conv_out_projection",
                    "linear_bf16_1024x1024.comp",
                ),
                ("layer_00", "operator_residual", "add_bf16_1024.comp"),
                ("layer_00", "ffn_norm", "rms_norm_bf16_serial.comp"),
                (
                    "layer_00",
                    "ffn_gate_projection",
                    "linear_bf16_1024x2560.comp",
                ),
                (
                    "layer_00",
                    "ffn_up_projection",
                    "linear_bf16_1024x2560.comp",
                ),
                ("layer_00", "ffn_gate_activation", "silu_bf16_2560.comp"),
                ("layer_00", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
                (
                    "layer_00",
                    "ffn_down_projection",
                    "linear_bf16_2560x1024.comp",
                ),
                ("layer_00", "ffn_residual", "add_bf16_1024.comp"),
                ("layer_02", "operator_norm", "rms_norm_bf16_serial.comp"),
                ("layer_02", "q_projection", "linear_bf16_1024x1024.comp"),
                ("layer_02", "k_projection", "linear_bf16_1024x512.comp"),
                ("layer_02", "v_projection", "linear_bf16_1024x512.comp"),
                (
                    "layer_02",
                    "q_head_norm",
                    "rms_norm_per_head_bf16_16x64.comp",
                ),
                (
                    "layer_02",
                    "k_head_norm",
                    "rms_norm_per_head_bf16_8x64.comp",
                ),
                ("layer_02", "q_rope", "rotary_bf16_16x64.comp"),
                ("layer_02", "k_rope", "rotary_bf16_8x64.comp"),
                (
                    "layer_02",
                    "kv_memory_append",
                    "append_kv_state_bf16_8x64.comp",
                ),
                ("layer_02", "attention_read", attention_shader),
                (
                    "layer_02",
                    "attention_out_projection",
                    "linear_bf16_1024x1024.comp",
                ),
                ("layer_02", "operator_residual", "add_bf16_1024.comp"),
                ("layer_02", "ffn_norm", "rms_norm_bf16_serial.comp"),
                (
                    "layer_02",
                    "ffn_gate_projection",
                    "linear_bf16_1024x2560.comp",
                ),
                (
                    "layer_02",
                    "ffn_up_projection",
                    "linear_bf16_1024x2560.comp",
                ),
                ("layer_02", "ffn_gate_activation", "silu_bf16_2560.comp"),
                ("layer_02", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
                (
                    "layer_02",
                    "ffn_down_projection",
                    "linear_bf16_2560x1024.comp",
                ),
                ("layer_02", "ffn_residual", "add_bf16_1024.comp"),
            ],
        )
    }

    fn loaded_kernel_pack_for_dispatch_shaders(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
        dispatch_shaders: &[(&str, &str, &str)],
    ) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
        let mut loaded_artifacts = Vec::new();
        let mut loaded_families = BTreeSet::new();
        let mut total_word_count = 0usize;

        for (pedal_id, node_id, shader_file) in dispatch_shaders {
            let dispatch = mounted_bound.dispatch(pedal_id, node_id).unwrap();
            if !loaded_families.insert(dispatch.reusable_family_id.clone()) {
                continue;
            }
            let spirv_words =
                crate::vulkan_compute::compile_test_shader_words_from_source(shader_file)?;
            total_word_count = total_word_count.checked_add(spirv_words.len())?;
            let family = mounted
                .placed_plan
                .reusable_kernel_plan
                .family(&dispatch.reusable_family_id)
                .unwrap();
            let artifact_path = format!("kernels/{}.spv", dispatch.reusable_family_id);
            loaded_artifacts.push(VulkanLoadedReusableKernelArtifact {
                artifact: VulkanReusableKernelArtifact::from_family(family, artifact_path.clone()),
                resolved_path: PathBuf::from(artifact_path),
                words: spirv_words,
            });
        }

        Some(VulkanLoadedReusableKernelArtifactManifest {
            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            artifacts: loaded_artifacts,
            total_word_count,
        })
    }

    fn reusable_family_with_kernel<'a>(
        reusable_plan: &'a VulkanReusableKernelPlan,
        kernel_id: &str,
    ) -> &'a VulkanReusableKernelFamily {
        reusable_plan
            .families
            .iter()
            .find(|family| {
                family
                    .command_refs
                    .iter()
                    .any(|command| command.kernel_id == kernel_id)
            })
            .unwrap()
    }

    fn artifact_path_for_family(family: &VulkanReusableKernelFamily) -> String {
        format!("kernels/{}.spv", family.family_id)
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
    fn placed_resident_plan_hosts_only_the_pedals_assigned_to_a_device() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let placement_spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "cpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_03", "lan:worker-a");
        let placement_plan = graph.placement_plan(&placement_spec).unwrap();

        let gpu0 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let gpu1 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu1",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let cpu0 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "cpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let lan = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "lan:worker-a",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        assert_eq!(gpu0.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(gpu0.device_id, "gpu0");
        assert_eq!(gpu0.hosted_pedal_ids.len(), 11);
        assert!(gpu0.hosts_pedal("layer_00"));
        assert!(!gpu0.hosts_pedal("layer_02"));
        assert_eq!(gpu0.resident_plan.circuit_count, 11);
        assert_eq!(gpu0.resident_plan.permanent_parameters.len(), 103);
        assert_eq!(gpu0.resident_plan.stream_state_buffers.len(), 11);
        assert_eq!(gpu0.resident_plan.activation_banks.len(), 11);
        assert_eq!(gpu0.resident_plan.state_view_signal_count, 16);
        assert_eq!(gpu0.signal_element_bytes, Some(2));
        assert_eq!(gpu0.local_cables.len(), 9);
        assert_eq!(gpu0.incoming_cables.len(), 1);
        assert_eq!(gpu0.outgoing_cables.len(), 1);
        assert_eq!(gpu0.incoming_cables[0].source_pedal_id, "layer_03");
        assert_eq!(gpu0.incoming_cables[0].destination_pedal_id, "layer_04");
        assert_eq!(gpu0.outgoing_cables[0].source_pedal_id, "layer_00");
        assert_eq!(gpu0.outgoing_cables[0].destination_pedal_id, "layer_01");

        let gpu0_cable_io = VulkanPlacedCableIoPlan::from_placed_resident_plan(&gpu0).unwrap();
        assert_eq!(gpu0_cable_io.device_id, "gpu0");
        assert_eq!(gpu0_cable_io.local_cable_count, 9);
        assert_eq!(gpu0_cable_io.total_endpoint_count, 2);
        assert_eq!(gpu0_cable_io.total_buffer_count, 11);
        assert_eq!(gpu0_cable_io.incoming_endpoint_count, 1);
        assert_eq!(gpu0_cable_io.outgoing_endpoint_count, 1);
        assert_eq!(gpu0_cable_io.total_byte_capacity, Some(22_528));
        let gpu0_local = &gpu0_cable_io.local_cables[0];
        assert_eq!(gpu0_local.cable_id, "cable_4_local");
        assert_eq!(gpu0_local.source_pedal_id, "layer_04");
        assert_eq!(gpu0_local.destination_pedal_id, "layer_05");
        assert_eq!(gpu0_local.byte_capacity, Some(2_048));

        assert_eq!(gpu1.hosted_pedal_ids, vec!["layer_02".to_string()]);
        assert_eq!(gpu1.resident_plan.circuit_count, 1);
        assert_eq!(gpu1.resident_plan.permanent_parameters.len(), 11);
        assert_eq!(gpu1.resident_plan.stream_state_buffers.len(), 1);
        assert_eq!(gpu1.resident_plan.state_view_signal_count, 2);
        assert_eq!(gpu1.incoming_cables[0].source_pedal_id, "layer_01");
        assert_eq!(gpu1.outgoing_cables[0].destination_pedal_id, "layer_03");
        let gpu1_cable_io = VulkanPlacedCableIoPlan::from_placed_resident_plan(&gpu1).unwrap();
        assert_eq!(gpu1_cable_io.device_id, "gpu1");
        assert_eq!(gpu1_cable_io.local_cable_count, 0);
        assert_eq!(gpu1_cable_io.total_endpoint_count, 2);
        assert_eq!(gpu1_cable_io.total_buffer_count, 2);
        assert_eq!(gpu1_cable_io.total_byte_capacity, Some(4_096));
        assert_eq!(gpu1_cable_io.unresolved_byte_cables, Vec::<usize>::new());
        let gpu1_incoming = gpu1_cable_io
            .endpoint(VulkanPlacedCableDirection::Incoming, 1)
            .unwrap();
        assert_eq!(gpu1_incoming.endpoint_id, "cable_1_in");
        assert_eq!(gpu1_incoming.signal, "frame");
        assert_eq!(gpu1_incoming.shape, vec![1024]);
        assert_eq!(gpu1_incoming.element_count, 1024);
        assert_eq!(gpu1_incoming.byte_capacity, Some(2_048));
        assert_eq!(gpu1_incoming.local_device_id, "gpu1");
        assert_eq!(gpu1_incoming.remote_device_id, "cpu0");
        assert_eq!(gpu1_incoming.local_pedal_id, "layer_02");
        assert_eq!(gpu1_incoming.remote_pedal_id, "layer_01");
        assert_eq!(gpu1_incoming.local_port_id, "input_frame");
        assert_eq!(gpu1_incoming.remote_port_id, "output_frame");
        let gpu1_outgoing = gpu1_cable_io
            .endpoint(VulkanPlacedCableDirection::Outgoing, 2)
            .unwrap();
        assert_eq!(gpu1_outgoing.endpoint_id, "cable_2_out");
        assert_eq!(gpu1_outgoing.byte_capacity, Some(2_048));
        assert_eq!(gpu1_outgoing.local_device_id, "gpu1");
        assert_eq!(gpu1_outgoing.remote_device_id, "lan:worker-a");
        assert_eq!(gpu1_outgoing.local_pedal_id, "layer_02");
        assert_eq!(gpu1_outgoing.remote_pedal_id, "layer_03");

        assert_eq!(cpu0.hosted_pedal_ids, vec!["layer_01".to_string()]);
        assert_eq!(cpu0.resident_plan.permanent_parameters.len(), 8);
        assert_eq!(cpu0.resident_plan.state_view_signal_count, 1);
        assert_eq!(lan.hosted_pedal_ids, vec!["layer_03".to_string()]);
        assert_eq!(lan.resident_plan.permanent_parameters.len(), 8);
        assert_eq!(lan.resident_plan.state_view_signal_count, 1);
    }

    #[test]
    fn placed_stream_circuit_plan_dispatches_only_hosted_pedals() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let placement_spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "cpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_03", "lan:worker-a");
        let placement_plan = graph.placement_plan(&placement_spec).unwrap();
        let gpu0_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let gpu1_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu1",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        let gpu0_plan = VulkanPlacedStreamCircuitPlan::from_plans(
            &execution_plan,
            &resource_plan,
            gpu0_resident,
        )
        .unwrap();
        let gpu1_plan = VulkanPlacedStreamCircuitPlan::from_plans(
            &execution_plan,
            &resource_plan,
            gpu1_resident,
        )
        .unwrap();

        assert_eq!(gpu0_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(gpu0_plan.device_id, "gpu0");
        assert_eq!(gpu0_plan.binding_plan.circuits.len(), 11);
        assert_eq!(gpu0_plan.binding_plan.total_node_count(), 191);
        assert_eq!(gpu0_plan.kernel_interface_plan.total_kernel_count(), 191);
        assert_eq!(gpu0_plan.dispatch_plan.total_dispatch_count(), 191);
        assert!(gpu0_plan.binding_plan.circuit("layer_00").is_some());
        assert!(gpu0_plan.binding_plan.circuit("layer_04").is_some());
        assert!(gpu0_plan.binding_plan.circuit("layer_01").is_none());
        assert!(gpu0_plan.binding_plan.circuit("layer_02").is_none());
        assert!(
            gpu0_plan
                .dispatch_plan
                .command("layer_02", "kv_memory_append")
                .is_none()
        );
        assert_eq!(
            gpu0_plan
                .dispatch_plan
                .command("layer_04", "operator_norm")
                .map(|command| command.dispatch_index),
            Some(16)
        );

        assert_eq!(gpu1_plan.device_id, "gpu1");
        assert_eq!(gpu1_plan.binding_plan.circuits.len(), 1);
        assert_eq!(gpu1_plan.binding_plan.total_node_count(), 19);
        assert_eq!(gpu1_plan.dispatch_plan.total_dispatch_count(), 19);
        assert_eq!(
            gpu1_plan
                .dispatch_plan
                .command("layer_02", "operator_norm")
                .map(|command| command.dispatch_index),
            Some(0)
        );
        assert_eq!(
            gpu1_plan
                .dispatch_plan
                .command("layer_02", "kv_memory_append")
                .map(|command| command.dispatch_index),
            Some(8)
        );
        assert!(
            gpu1_plan
                .dispatch_plan
                .command("layer_00", "operator_norm")
                .is_none()
        );

        let gpu1_descriptors = VulkanDescriptorResourcePlan::from_plans(
            &gpu1_plan.dispatch_plan,
            &gpu1_plan.placed_resident_plan.resident_plan,
            4,
        )
        .unwrap();
        assert_eq!(gpu1_descriptors.dispatches.len(), 19);
        let kv_append = gpu1_descriptors
            .dispatch("layer_02", "kv_memory_append")
            .unwrap();
        assert_eq!(kv_append.descriptors.len(), 9);
        assert!(matches!(
            kv_append.descriptors[2].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                ref pedal_id,
                ref state_id,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
    }

    #[test]
    fn resident_pedal_runner_executes_layer_00_end_to_end() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping layer_00 resident pedal runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = layer_00_level_1_loaded_kernel_pack(&mounted, &mounted_bound)
        else {
            eprintln!("skipping layer_00 resident pedal runner: no GLSL to SPIR-V compiler found");
            return;
        };
        load_layer_00_parameters(&mounted, &tensor_index);
        write_layer_00_unit_input_and_zero_state(&mounted);

        let runner = mounted
            .create_resident_pedal_runner(&device, &mounted_bound, "layer_00", &loaded_manifest)
            .unwrap();
        assert_eq!(runner.pedal_id, "layer_00");
        assert_eq!(runner.dispatch_count(), 16);
        assert_eq!(runner.total_descriptor_count, 52);
        assert_eq!(runner.total_push_constant_byte_count, 256);

        let run = runner
            .run_with_stream_control(
                &device,
                VulkanMountedPlacedStreamControl {
                    stream_tick: 7,
                    control_flags: 0,
                    dynamic_state_capacity_activations: mounted
                        .buffers
                        .dynamic_state_capacity_activations
                        as u32,
                },
            )
            .unwrap();
        assert_eq!(run.pedal_id, "layer_00");
        assert_eq!(run.dispatch_count(), 16);
        assert_eq!(
            run.node_ids(),
            vec![
                "operator_norm",
                "conv_in_projection",
                "split_b_c_x",
                "input_gate",
                "temporal_memory_update",
                "depthwise_temporal_conv",
                "output_gate",
                "conv_out_projection",
                "operator_residual",
                "ffn_norm",
                "ffn_gate_projection",
                "ffn_up_projection",
                "ffn_gate_activation",
                "ffn_gate_multiply",
                "ffn_down_projection",
                "ffn_residual",
            ]
        );

        let final_residual_dispatch = mounted_bound.dispatch("layer_00", "ffn_residual").unwrap();
        let final_residual_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(final_residual_dispatch)
            .unwrap();
        assert_eq!(
            final_residual_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x86, 0x3f, 0x82, 0x3f, 0x81, 0x3f, 0x7e, 0x3f, 0x83, 0x3f, 0x83, 0x3f, 0x83, 0x3f,
                0x83, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_input_transducer_feeds_layer_00_from_token_embedding() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident input transducer: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = layer_00_level_1_loaded_kernel_pack(&mounted, &mounted_bound)
        else {
            eprintln!("skipping resident input transducer: no GLSL to SPIR-V compiler found");
            return;
        };
        let Some(input_transducer_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "embedding_lookup_bf16_65536x1024.comp",
            )
        else {
            eprintln!("skipping resident input transducer: no GLSL to SPIR-V compiler found");
            return;
        };
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let transducer_parameter_plan =
            VulkanPermanentParameterBufferPlan::from_transducer_parameters(
                "gpu0",
                &resource_plan,
                Some(&tensor_index),
            )
            .unwrap();
        assert_eq!(transducer_parameter_plan.parameter_count, 2);
        assert_eq!(
            transducer_parameter_plan.total_byte_capacity,
            Some(134_219_776)
        );
        assert!(transducer_parameter_plan.unresolved_tensors.is_empty());
        let embed_tokens = transducer_parameter_plan
            .parameters
            .iter()
            .find(|parameter| parameter.tensor == LFM2_EMBED_TOKENS_TENSOR)
            .unwrap();
        assert_eq!(embed_tokens.use_count, 2);
        assert_eq!(embed_tokens.byte_capacity, Some(LFM2_EMBED_TOKENS_BYTES));
        let transducer_parameter_buffers =
            transducer_parameter_plan.allocate_buffers(&device).unwrap();
        assert_eq!(
            transducer_parameter_buffers.total_byte_capacity,
            134_219_776
        );
        assert!(
            transducer_parameter_buffers
                .parameter_buffer("model.embedding_norm.weight")
                .is_some()
        );
        let loaded_embedding = transducer_parameter_buffers
            .load_parameter_from_tensor_index(&tensor_index, LFM2_EMBED_TOKENS_TENSOR)
            .unwrap();
        assert_eq!(loaded_embedding.byte_count, LFM2_EMBED_TOKENS_BYTES);
        let token_id = 1u32;

        let input_transducer_runner =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_lfm2_token_embedding(
                &device,
                &mounted,
                &transducer_parameter_buffers,
                &input_transducer_spirv_words,
            )
            .unwrap();
        assert_eq!(
            input_transducer_runner.transducer_id,
            LFM2_TOKEN_EMBEDDING_TRANSDUCER_ID
        );
        assert_eq!(
            input_transducer_runner.parameter_tensor,
            LFM2_EMBED_TOKENS_TENSOR
        );
        assert_eq!(
            input_transducer_runner.output_signal_id,
            LFM2_INPUT_FRAME_SIGNAL
        );
        assert_eq!(input_transducer_runner.descriptor_count, 2);
        assert_eq!(input_transducer_runner.workgroup_count_x, 2);
        assert_eq!(input_transducer_runner.push_constant_byte_count, 4);

        let transducer_run = input_transducer_runner
            .run_token_id(&device, token_id)
            .unwrap();
        assert_eq!(
            transducer_run.transducer_id,
            LFM2_TOKEN_EMBEDDING_TRANSDUCER_ID
        );
        assert_eq!(transducer_run.token_id, token_id);
        assert_eq!(transducer_run.output_signal_id, LFM2_INPUT_FRAME_SIGNAL);
        assert_eq!(transducer_run.dispatch_count, 1);
        assert_eq!(transducer_run.descriptor_count, 2);
        assert_eq!(transducer_run.workgroup_count_x, 2);
        assert_eq!(transducer_run.push_constant_byte_count, 4);
        let input_frame = mounted
            .boundary_io
            .input_buffer(LFM2_INPUT_FRAME_SIGNAL)
            .unwrap();
        assert_eq!(
            input_frame.buffer.read_bytes(LFM2_FRAME_BYTES).unwrap(),
            lfm2_embedding_row_bytes(&tensor_index, token_id)
        );

        load_layer_00_parameters(&mounted, &tensor_index);
        zero_lfm2_temporal_memory(&mounted, "layer_00");
        let runner = mounted
            .create_resident_pedal_runner(&device, &mounted_bound, "layer_00", &loaded_manifest)
            .unwrap();
        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_eq!(run.pedal_id, "layer_00");
        assert_eq!(run.dispatch_count(), 16);

        let final_residual_dispatch = mounted_bound.dispatch("layer_00", "ffn_residual").unwrap();
        let final_residual_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(final_residual_dispatch)
            .unwrap();
        assert_eq!(
            final_residual_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x84, 0x3c, 0x09, 0x3c, 0xbf, 0x3b, 0x90, 0xbc, 0x30, 0x3b, 0xc6, 0xba, 0x8e, 0x3b,
                0x34, 0xbb,
            ]
        );
    }

    #[test]
    fn resident_output_transducer_projects_output_frame_to_logits() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident output transducer: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, _mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(embedding_norm_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "rms_norm_bf16_serial.comp",
            )
        else {
            eprintln!("skipping resident output transducer: no GLSL to SPIR-V compiler found");
            return;
        };
        let Some(tied_projection_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "tied_output_projection_bf16_65536x1024_to_f32.comp",
            )
        else {
            eprintln!("skipping resident output transducer: no GLSL to SPIR-V compiler found");
            return;
        };
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
                .unwrap();
        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
        let transducer_parameter_plan =
            VulkanPermanentParameterBufferPlan::from_transducer_parameters(
                "gpu0",
                &resource_plan,
                Some(&tensor_index),
            )
            .unwrap();
        let transducer_parameter_buffers =
            transducer_parameter_plan.allocate_buffers(&device).unwrap();
        let loaded_transducers = transducer_parameter_buffers
            .load_from_tensor_index(&tensor_index)
            .unwrap();
        assert_eq!(loaded_transducers.parameter_count, 2);
        assert_eq!(loaded_transducers.loaded_count, 2);
        assert_eq!(loaded_transducers.total_bytes_loaded, 134_219_776);

        write_lfm2_constant_output_frame(&mounted, [0x80, 0x3f]);
        let runner = VulkanResidentOutputTransducerRunner::from_mounted_lfm2_output_transducer(
            &device,
            &mounted,
            &transducer_parameter_buffers,
            &embedding_norm_spirv_words,
            &tied_projection_spirv_words,
        )
        .unwrap();
        assert_eq!(runner.transducer_id, "output_transducer");
        assert_eq!(runner.input_signal_id, LFM2_OUTPUT_FRAME_SIGNAL);
        assert_eq!(runner.logits_byte_capacity, LFM2_LOGITS_BYTES);
        assert_eq!(runner.dispatch_count, 2);
        assert_eq!(runner.total_descriptor_count, 6);
        assert_eq!(runner.total_push_constant_byte_count, 0);

        let run = runner.run(&device).unwrap();
        assert_eq!(run.transducer_id, "output_transducer");
        assert_eq!(run.input_signal_id, LFM2_OUTPUT_FRAME_SIGNAL);
        assert_eq!(run.dispatch_count, 2);
        assert_eq!(
            run.node_ids,
            vec![
                LFM2_OUTPUT_EMBEDDING_NORM_TRANSDUCER_ID.to_string(),
                LFM2_TIED_OUTPUT_PROJECTION_TRANSDUCER_ID.to_string(),
            ]
        );
        assert_eq!(run.descriptor_counts, vec![3, 3]);
        assert_eq!(run.workgroup_counts_x, vec![1, 1024]);
        assert_eq!(run.push_constant_byte_counts, vec![0, 0]);
        assert_eq!(run.logits_byte_capacity, LFM2_LOGITS_BYTES);

        assert_eq!(
            runner.read_normalized_frame_bytes(16).unwrap(),
            vec![
                0x1a, 0x40, 0x5b, 0x40, 0x56, 0x40, 0x58, 0x40, 0x59, 0x40, 0x55, 0x40, 0x4e, 0x40,
                0x3c, 0x40,
            ]
        );
        assert_eq!(
            runner.read_logits_bytes(16).unwrap(),
            vec![
                0x8e, 0x09, 0xa6, 0x3f, 0x10, 0x7a, 0x4f, 0xbd, 0x22, 0xee, 0x7a, 0xc0, 0xdd, 0x90,
                0xa1, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_greedy_sampler_selects_largest_logit() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy sampler: {error}");
                return;
            }
        };
        let Some(sampler_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "greedy_sampler_f32_65536.comp",
            )
        else {
            eprintln!("skipping resident greedy sampler: no GLSL to SPIR-V compiler found");
            return;
        };

        let logits_buffer = device.create_resident_buffer(LFM2_LOGITS_BYTES).unwrap();
        let mut logits = vec![0u8; LFM2_LOGITS_BYTES];
        let token_7 = 7usize;
        let token_1024 = 1_024usize;
        logits[(token_7 * 4)..((token_7 + 1) * 4)].copy_from_slice(&3.5f32.to_le_bytes());
        logits[(token_1024 * 4)..((token_1024 + 1) * 4)].copy_from_slice(&9.25f32.to_le_bytes());
        logits_buffer.write_bytes(&logits).unwrap();

        let runner = VulkanResidentGreedySamplerRunner::from_logits_buffer(
            &device,
            &logits_buffer,
            LFM2_LOGITS_BYTES,
            &sampler_spirv_words,
        )
        .unwrap();
        assert_eq!(runner.sampler_id, LFM2_GREEDY_SAMPLER_PEDAL_ID);
        assert_eq!(runner.logits_byte_capacity, LFM2_LOGITS_BYTES);
        assert_eq!(runner.output_byte_capacity, LFM2_SAMPLER_OUTPUT_BYTES);
        assert_eq!(runner.descriptor_count, 2);
        assert_eq!(runner.workgroup_count_x, 1);
        assert_eq!(runner.push_constant_byte_count, 0);

        let run = runner.run(&device).unwrap();
        assert_eq!(run.sampler_id, LFM2_GREEDY_SAMPLER_PEDAL_ID);
        assert_eq!(run.token_id, token_1024 as u32);
        assert_eq!(run.selected_logit_bits, 9.25f32.to_bits());
        assert_eq!(run.control_flags, 0);
        assert_eq!(run.descriptor_count, 2);
        assert_eq!(run.workgroup_count_x, 1);
        assert_eq!(run.push_constant_byte_count, 0);
        assert_eq!(
            runner.read_output_bytes().unwrap(),
            vec![
                0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x14, 0x41, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn resident_single_token_tick_runs_input_board_and_output_to_logits() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident single-token tick runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!(
                "skipping resident single-token tick runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };
        let Some(input_transducer_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "embedding_lookup_bf16_65536x1024.comp",
            )
        else {
            eprintln!(
                "skipping resident single-token tick runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };
        let Some(embedding_norm_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "rms_norm_bf16_serial.comp",
            )
        else {
            eprintln!(
                "skipping resident single-token tick runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };
        let Some(tied_projection_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "tied_output_projection_bf16_65536x1024_to_f32.comp",
            )
        else {
            eprintln!(
                "skipping resident single-token tick runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };

        let transducer_parameter_buffers =
            load_lfm2_transducer_parameter_buffers(&device, &tensor_index);
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 13);
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_lfm2_token_embedding(
                &device,
                &mounted,
                &transducer_parameter_buffers,
                &input_transducer_spirv_words,
            )
            .unwrap();
        let pedalboard = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_lfm2_output_transducer(
                &device,
                &mounted,
                &transducer_parameter_buffers,
                &embedding_norm_spirv_words,
                &tied_projection_spirv_words,
            )
            .unwrap();
        let runner = VulkanResidentSingleTokenTickRunner::new(
            input_transducer,
            pedalboard,
            output_transducer,
        )
        .unwrap();
        assert_eq!(runner.device_id, "gpu0");
        assert_eq!(runner.pedal_count, 14);
        assert_eq!(runner.dispatch_count, 245);
        assert_eq!(runner.total_descriptor_count, 802);
        assert_eq!(runner.total_push_constant_byte_count, 3_876);

        let token_id = 1u32;
        let run = runner
            .run_token_id_with_stream_control(&device, token_id, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_eq!(run.device_id, "gpu0");
        assert_eq!(run.token_id, token_id);
        assert_eq!(run.dispatch_count, 245);
        assert_eq!(run.total_descriptor_count, 802);
        assert_eq!(run.total_push_constant_byte_count, 3_876);
        assert_eq!(run.input_run.dispatch_count, 1);
        assert_eq!(run.input_run.output_signal_id, LFM2_INPUT_FRAME_SIGNAL);
        assert_lfm2_resident_prefix_run(&run.pedalboard_run, &pedal_ids, 242);
        assert_eq!(run.output_run.dispatch_count, 2);
        assert_eq!(run.output_run.logits_byte_capacity, LFM2_LOGITS_BYTES);

        let input_frame = mounted
            .boundary_io
            .input_buffer(LFM2_INPUT_FRAME_SIGNAL)
            .unwrap();
        assert_eq!(
            input_frame.buffer.read_bytes(LFM2_FRAME_BYTES).unwrap(),
            lfm2_embedding_row_bytes(&tensor_index, token_id)
        );
        assert_eq!(
            runner.read_normalized_frame_bytes(16).unwrap(),
            vec![
                0xb5, 0xbf, 0xee, 0xbf, 0x51, 0x3f, 0x99, 0xbf, 0x35, 0xbf, 0xc4, 0xbe, 0x7a, 0x3f,
                0x94, 0xbf,
            ]
        );
        assert_eq!(
            runner.read_logits_bytes(16).unwrap(),
            vec![
                0xa3, 0xc8, 0x12, 0xc0, 0xf7, 0xc1, 0x9d, 0x41, 0x84, 0x92, 0x6a, 0x41, 0x16, 0x9c,
                0x17, 0xc0,
            ]
        );
    }

    fn create_lfm2_resident_greedy_stream_processor(
        device: &VulkanComputeDevice,
        skip_label: &str,
    ) -> Option<VulkanResidentGreedyStreamProcessor> {
        create_lfm2_resident_greedy_stream_processor_with_capacity(
            device,
            skip_label,
            4,
            "gqa_attention_bf16_q16_kv8_d64_cap4.comp",
        )
    }

    fn create_lfm2_resident_greedy_stream_processor_with_capacity(
        device: &VulkanComputeDevice,
        skip_label: &str,
        dynamic_state_capacity_activations: usize,
        attention_shader: &str,
    ) -> Option<VulkanResidentGreedyStreamProcessor> {
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit_with_capacity(
                device,
                dynamic_state_capacity_activations,
            );
        let Some(loaded_manifest) =
            lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
                &mounted,
                &mounted_bound,
                attention_shader,
            )
        else {
            eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
            return None;
        };
        let Some(input_transducer_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "embedding_lookup_bf16_65536x1024.comp",
            )
        else {
            eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
            return None;
        };
        let Some(embedding_norm_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "rms_norm_bf16_serial.comp",
            )
        else {
            eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
            return None;
        };
        let Some(tied_projection_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "tied_output_projection_bf16_65536x1024_to_f32.comp",
            )
        else {
            eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
            return None;
        };
        let Some(sampler_spirv_words) =
            crate::vulkan_compute::compile_test_shader_words_from_source(
                "greedy_sampler_f32_65536.comp",
            )
        else {
            eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
            return None;
        };

        let transducer_parameter_buffers =
            load_lfm2_transducer_parameter_buffers(device, &tensor_index);
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 13);
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_lfm2_token_embedding(
                device,
                &mounted,
                &transducer_parameter_buffers,
                &input_transducer_spirv_words,
            )
            .unwrap();
        let pedalboard = create_lfm2_resident_prefix_runner(
            device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_lfm2_output_transducer(
                device,
                &mounted,
                &transducer_parameter_buffers,
                &embedding_norm_spirv_words,
                &tied_projection_spirv_words,
            )
            .unwrap();
        let sampler = VulkanResidentGreedySamplerRunner::from_output_transducer(
            device,
            &output_transducer,
            &sampler_spirv_words,
        )
        .unwrap();
        let tick_runner = VulkanResidentSingleTokenTickRunner::new(
            input_transducer,
            pedalboard,
            output_transducer,
        )
        .unwrap();
        let loop_runner =
            VulkanResidentGreedyFeedbackLoopRunner::new(tick_runner, sampler).unwrap();
        Some(VulkanResidentGreedyStreamProcessor::new(
            mounted,
            transducer_parameter_buffers,
            loop_runner,
        ))
    }

    #[test]
    fn resident_greedy_feedback_loop_runs_two_ticks() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy feedback loop: {error}");
                return;
            }
        };
        let Some(processor) =
            create_lfm2_resident_greedy_stream_processor(&device, "resident greedy feedback loop")
        else {
            return;
        };
        assert_eq!(processor.device_id, "gpu0");
        assert_eq!(processor.pedal_count, 14);
        assert_eq!(processor.per_tick_dispatch_count, 246);
        assert_eq!(processor.per_tick_descriptor_count, 804);
        assert_eq!(processor.per_tick_push_constant_byte_count, 3_876);
        assert_eq!(processor.dynamic_state_capacity_activations, 4);

        let run = processor.run_bounded(&device, 1, 0, 2).unwrap();
        assert_eq!(run.device_id, "gpu0");
        assert_eq!(run.initial_token_id, 1);
        assert_eq!(run.tick_runs.len(), 2);
        assert_eq!(run.per_tick_dispatch_count, 246);
        assert_eq!(run.per_tick_descriptor_count, 804);
        assert_eq!(run.per_tick_push_constant_byte_count, 3_876);
        assert_eq!(run.tick_runs[0].stream_tick, 0);
        assert_eq!(run.tick_runs[0].input_token_id, 1);
        assert_eq!(run.tick_runs[1].stream_tick, 1);
        assert_eq!(
            run.tick_runs[1].input_token_id,
            run.tick_runs[0].sampled_token_id
        );
        assert_eq!(run.tick_runs[0].tick_run.dispatch_count, 245);
        assert_eq!(run.tick_runs[0].sampler_run.descriptor_count, 2);
        assert_eq!(run.tick_runs[1].tick_run.dispatch_count, 245);
        assert_eq!(run.tick_runs[1].sampler_run.descriptor_count, 2);
        assert_eq!(run.sampled_token_ids, vec![1, 1]);
        assert_eq!(run.tick_runs[0].sampler_run.token_id, 1);
        assert_eq!(run.tick_runs[1].sampler_run.token_id, 1);
        assert_eq!(
            run.tick_runs
                .iter()
                .map(|tick| tick.sampler_run.selected_logit_bits)
                .collect::<Vec<_>>(),
            vec![1_100_857_847, 1_101_580_110]
        );
    }

    #[test]
    fn resident_greedy_prompt_event_drains_external_input_before_feedback() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy prompt event: {error}");
                return;
            }
        };
        let Some(processor) =
            create_lfm2_resident_greedy_stream_processor(&device, "resident greedy prompt event")
        else {
            return;
        };

        let run = processor
            .run_prompt_event_bounded(&device, &[1, 36_309], 0, 1, None)
            .unwrap();

        assert_eq!(run.device_id, "gpu0");
        assert_eq!(run.prompt_token_ids, vec![1, 36_309]);
        assert_eq!(run.generated_token_ids.len(), 1);
        assert_eq!(
            run.output_token_ids,
            vec![1, 36_309, run.generated_token_ids[0]]
        );
        assert_eq!(run.stop_reason, "max_new_tokens");
        assert_eq!(run.tick_runs.len(), 3);
        assert_eq!(run.per_tick_dispatch_count, 246);
        assert_eq!(run.per_tick_descriptor_count, 804);
        assert_eq!(run.per_tick_push_constant_byte_count, 3_876);

        assert_eq!(run.tick_runs[0].stream_tick, 0);
        assert_eq!(run.tick_runs[0].input_token_id, 1);
        assert_eq!(
            run.tick_runs[0].input_route,
            VulkanResidentGreedyPromptEventInputRoute::ExternalInput
        );
        assert_eq!(run.tick_runs[0].public_output_token_id, None);
        assert_eq!(run.tick_runs[0].private_feedback_token_id, None);
        assert!(run.tick_runs[0].sampler_run.is_none());

        assert_eq!(run.tick_runs[1].stream_tick, 1);
        assert_eq!(run.tick_runs[1].input_token_id, 36_309);
        assert_eq!(
            run.tick_runs[1].input_route,
            VulkanResidentGreedyPromptEventInputRoute::ExternalInput
        );
        assert_eq!(
            run.tick_runs[1].public_output_token_id,
            Some(run.generated_token_ids[0])
        );
        assert_eq!(
            run.tick_runs[1].private_feedback_token_id,
            Some(run.generated_token_ids[0])
        );
        assert_eq!(
            run.tick_runs[1].private_feedback_closes_loop_after_processing,
            Some(true)
        );
        assert_eq!(
            run.tick_runs[1].sampler_run.as_ref().unwrap().token_id,
            run.generated_token_ids[0]
        );

        assert_eq!(run.tick_runs[2].stream_tick, 2);
        assert_eq!(run.tick_runs[2].input_token_id, run.generated_token_ids[0]);
        assert_eq!(
            run.tick_runs[2].input_route,
            VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback
        );
        assert_eq!(run.tick_runs[2].input_feedback_depth, 1);
        assert!(run.tick_runs[2].input_closes_loop_after_processing);
        assert_eq!(run.tick_runs[2].public_output_token_id, None);
        assert_eq!(run.tick_runs[2].private_feedback_token_id, None);
        assert!(run.tick_runs[2].sampler_run.is_none());
    }

    #[test]
    fn resident_greedy_running_stream_accepts_later_input_without_resetting_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy running stream: {error}");
                return;
            }
        };
        let Some(processor) =
            create_lfm2_resident_greedy_stream_processor(&device, "resident greedy running stream")
        else {
            return;
        };
        let mut stream = processor.into_running_stream("stream_0");
        assert_eq!(stream.stream_id, "stream_0");
        assert_eq!(stream.next_stream_tick, 0);
        assert_eq!(stream.pending_external_input_count(), 0);
        assert_eq!(stream.pending_private_feedback_count(), 0);

        let first = stream.run_prompt(&device, &[1], 1, None).unwrap();
        assert_eq!(first.stream_id, "stream_0");
        assert_eq!(first.prompt_token_ids, vec![1]);
        assert_eq!(first.generated_token_ids.len(), 1);
        assert_eq!(
            first.output_token_ids,
            vec![1, first.generated_token_ids[0]]
        );
        assert_eq!(first.stop_reason, "max_new_tokens");
        assert_eq!(first.start_stream_tick, 0);
        assert_eq!(first.next_stream_tick, 2);
        assert_eq!(first.ticks.len(), 3);
        assert_eq!(
            first.ticks[0].status,
            VulkanResidentGreedyRunningStreamTickStatus::Processed
        );
        assert_eq!(first.ticks[0].stream_tick, Some(0));
        assert_eq!(
            first.ticks[0].input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::ExternalInput
        );
        assert_eq!(
            first.ticks[0].public_output.as_ref().unwrap().token_id,
            first.generated_token_ids[0]
        );
        assert_eq!(first.ticks[1].stream_tick, Some(1));
        assert_eq!(
            first.ticks[1].input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback
        );
        assert_eq!(
            first.ticks[1].input_signal.as_ref().unwrap().token_id(),
            first.generated_token_ids[0]
        );
        assert_eq!(
            first.ticks[2].status,
            VulkanResidentGreedyRunningStreamTickStatus::Idle
        );
        assert_eq!(first.ticks[2].stream_tick, None);
        assert_eq!(stream.next_stream_tick, 2);
        assert_eq!(stream.public_outputs().len(), 1);
        assert_eq!(stream.private_feedback_history().len(), 1);
        assert_eq!(stream.pending_external_input_count(), 0);
        assert_eq!(stream.pending_private_feedback_count(), 0);

        let second = stream.run_prompt(&device, &[36_309], 1, None).unwrap();
        assert_eq!(second.prompt_token_ids, vec![36_309]);
        assert_eq!(second.generated_token_ids.len(), 1);
        assert_eq!(
            second.output_token_ids,
            vec![36_309, second.generated_token_ids[0]]
        );
        assert_eq!(second.stop_reason, "max_new_tokens");
        assert_eq!(second.start_stream_tick, 2);
        assert_eq!(second.next_stream_tick, 4);
        assert_eq!(second.ticks.len(), 3);
        assert_eq!(second.ticks[0].stream_tick, Some(2));
        assert_eq!(
            second.ticks[0].input_signal.as_ref().unwrap().token_id(),
            36_309
        );
        assert_eq!(
            second.ticks[0].input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::ExternalInput
        );
        assert_eq!(second.ticks[1].stream_tick, Some(3));
        assert_eq!(
            second.ticks[1].input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback
        );
        assert_eq!(
            second.ticks[2].status,
            VulkanResidentGreedyRunningStreamTickStatus::Idle
        );
        assert_eq!(second.ticks[2].stream_tick, None);
        assert_eq!(stream.next_stream_tick, 4);
        assert_eq!(stream.public_outputs().len(), 2);
        assert_eq!(stream.private_feedback_history().len(), 2);
        assert_eq!(stream.ticks().len(), 6);
        assert!(!stream.loop_open);
        assert_eq!(stream.last_stop_reason.as_deref(), Some("max_new_tokens"));

        stream.inject_prompt(&[1], 0, None).unwrap();
        let error = stream.tick(&device).unwrap_err();
        assert!(matches!(
            error,
            VulkanResidentGreedyFeedbackLoopRunnerError::StreamStateCapacityExceeded {
                stream_tick: 4,
                dynamic_state_capacity_activations: 4,
            }
        ));
        assert_eq!(stream.pending_external_input_count(), 1);
        assert_eq!(stream.next_stream_tick, 4);
    }

    #[test]
    fn resident_greedy_running_stream_uses_configured_capacity() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy running stream capacity: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor_with_capacity(
            &device,
            "resident greedy running stream capacity",
            8,
            "gqa_attention_bf16_q16_kv8_d64_cap8.comp",
        ) else {
            return;
        };
        assert_eq!(processor.dynamic_state_capacity_activations, 8);

        let mut stream = processor.into_running_stream("stream_0");
        let run = stream.run_prompt(&device, &[1], 7, None).unwrap();
        assert_eq!(run.prompt_token_ids, vec![1]);
        assert_eq!(run.generated_token_ids.len(), 7);
        assert_eq!(run.output_token_ids.len(), 8);
        assert_eq!(run.stop_reason, "max_new_tokens");
        assert_eq!(run.start_stream_tick, 0);
        assert_eq!(run.next_stream_tick, 8);
        assert_eq!(stream.next_stream_tick, 8);
        assert_eq!(stream.public_outputs().len(), 7);
        assert_eq!(stream.private_feedback_history().len(), 7);
        assert_eq!(run.ticks.len(), 9);
        assert_eq!(run.ticks[0].stream_tick, Some(0));
        assert_eq!(run.ticks[7].stream_tick, Some(7));
        assert_eq!(
            run.ticks[7].input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback
        );
        assert_eq!(
            run.ticks[8].status,
            VulkanResidentGreedyRunningStreamTickStatus::Idle
        );
        assert_eq!(run.ticks[8].stream_tick, None);

        stream.inject_prompt(&[36_309], 0, None).unwrap();
        let error = stream.tick(&device).unwrap_err();
        assert!(matches!(
            error,
            VulkanResidentGreedyFeedbackLoopRunnerError::StreamStateCapacityExceeded {
                stream_tick: 8,
                dynamic_state_capacity_activations: 8,
            }
        ));
        assert_eq!(stream.pending_external_input_count(), 1);
        assert_eq!(stream.next_stream_tick, 8);
    }

    #[test]
    fn resident_token_stream_api_accepts_external_events_and_emits_public_events() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident token stream API: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor_with_capacity(
            &device,
            "resident token stream API",
            8,
            "gqa_attention_bf16_q16_kv8_d64_cap8.comp",
        ) else {
            return;
        };
        let mut stream = processor.into_token_stream("host_stream_0");
        assert_eq!(stream.stream_id(), "host_stream_0");
        assert_eq!(stream.next_stream_tick(), 0);

        let first_event =
            VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host");
        let first = stream
            .submit_external_event(&device, first_event.clone())
            .unwrap();
        assert_eq!(first.stream_id, "host_stream_0");
        assert_eq!(first.input_event, first_event);
        assert_eq!(first.generated_token_ids.len(), 3);
        assert_eq!(first.output_events.len(), 3);
        assert_eq!(first.stop_reason, "max_new_tokens");
        assert_eq!(first.start_stream_tick, 0);
        assert_eq!(first.next_stream_tick, 4);
        assert_eq!(first.processed_tick_count, 4);
        assert_eq!(first.idle_tick_count, 1);
        assert_eq!(
            first
                .output_events
                .iter()
                .map(|event| event.input_event_id.as_str())
                .collect::<Vec<_>>(),
            vec!["event_0", "event_0", "event_0"]
        );
        assert_eq!(
            first
                .output_events
                .iter()
                .map(|event| event.output_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            first
                .output_events
                .iter()
                .map(|event| event.source_stream_tick)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );

        let second_event =
            VulkanResidentTokenInputEvent::new("event_1", vec![36_309], 1).with_origin("test_host");
        let second = stream
            .submit_external_event(&device, second_event.clone())
            .unwrap();
        assert_eq!(second.input_event, second_event);
        assert_eq!(second.generated_token_ids.len(), 1);
        assert_eq!(second.output_events.len(), 1);
        assert_eq!(second.output_events[0].input_event_id, "event_1");
        assert_eq!(second.output_events[0].output_index, 0);
        assert_eq!(second.output_events[0].source_stream_tick, 4);
        assert_eq!(second.start_stream_tick, 4);
        assert_eq!(second.next_stream_tick, 6);
        assert_eq!(second.processed_tick_count, 2);
        assert_eq!(second.idle_tick_count, 1);

        let snapshot = stream.snapshot();
        assert_eq!(snapshot.stream_id, "host_stream_0");
        assert_eq!(snapshot.next_stream_tick, 6);
        assert!(!snapshot.loop_open);
        assert!(snapshot.idle);
        assert_eq!(snapshot.total_public_outputs, 4);
        assert_eq!(snapshot.total_ticks, 8);
        assert_eq!(snapshot.last_stop_reason.as_deref(), Some("max_new_tokens"));
    }

    #[test]
    fn resident_token_stream_can_be_pumped_one_tick_at_a_time() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident token stream pump: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor_with_capacity(
            &device,
            "resident token stream pump",
            8,
            "gqa_attention_bf16_q16_kv8_d64_cap8.comp",
        ) else {
            return;
        };
        let mut stream = processor.into_token_stream("host_stream_0");
        let event =
            VulkanResidentTokenInputEvent::new("event_0", vec![1], 2).with_origin("test_host");
        let queued = stream.enqueue_external_event(event.clone()).unwrap();
        assert_eq!(queued.input_event, event);
        assert_eq!(queued.start_stream_tick, 0);
        assert_eq!(queued.enqueued_token_count, 1);
        assert!(!stream.snapshot().idle);

        let first = stream.pump_once(&device).unwrap();
        assert_eq!(first.stream_id, "host_stream_0");
        assert_eq!(
            first.status,
            VulkanResidentGreedyRunningStreamTickStatus::Processed
        );
        assert_eq!(first.stream_tick, Some(0));
        assert_eq!(first.input_token_id, Some(1));
        assert_eq!(
            first.input_route,
            Some(VulkanResidentGreedyPromptEventInputRoute::ExternalInput)
        );
        assert_eq!(
            first.output_event.as_ref().unwrap().input_event_id,
            "event_0"
        );
        assert_eq!(first.output_event.as_ref().unwrap().output_index, 0);
        assert_eq!(first.output_event.as_ref().unwrap().source_stream_tick, 0);

        let second = stream.pump_once(&device).unwrap();
        assert_eq!(second.stream_tick, Some(1));
        assert_eq!(
            second.input_route,
            Some(VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback)
        );
        assert_eq!(
            second.output_event.as_ref().unwrap().input_event_id,
            "event_0"
        );
        assert_eq!(second.output_event.as_ref().unwrap().output_index, 1);
        assert_eq!(second.output_event.as_ref().unwrap().source_stream_tick, 1);

        let closing = stream.pump_once(&device).unwrap();
        assert_eq!(closing.stream_tick, Some(2));
        assert_eq!(
            closing.input_route,
            Some(VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback)
        );
        assert!(closing.output_event.is_none());
        assert_eq!(closing.stop_reason.as_deref(), Some("max_new_tokens"));

        let idle = stream.pump_once(&device).unwrap();
        assert_eq!(
            idle.status,
            VulkanResidentGreedyRunningStreamTickStatus::Idle
        );
        assert_eq!(idle.stream_tick, None);
        assert!(idle.output_event.is_none());
        assert_eq!(idle.stop_reason.as_deref(), Some("max_new_tokens"));

        let snapshot = stream.snapshot();
        assert_eq!(snapshot.next_stream_tick, 3);
        assert!(snapshot.idle);
        assert_eq!(snapshot.total_public_outputs, 2);
        assert_eq!(snapshot.total_ticks, 4);
    }

    #[test]
    fn resident_token_stream_can_pump_bounded_runtime_cycles() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident token stream bounded pump: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor_with_capacity(
            &device,
            "resident token stream bounded pump",
            8,
            "gqa_attention_bf16_q16_kv8_d64_cap8.comp",
        ) else {
            return;
        };
        let mut stream = processor.into_token_stream("host_stream_0");
        stream
            .enqueue_external_event(
                VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
            )
            .unwrap();

        let first_cycle = stream.pump_bounded(&device, 2).unwrap();
        assert_eq!(first_cycle.stream_id, "host_stream_0");
        assert_eq!(first_cycle.start_stream_tick, 0);
        assert_eq!(first_cycle.next_stream_tick, 2);
        assert_eq!(
            first_cycle.stop_condition,
            VulkanResidentTokenStreamPumpStopCondition::TickBudget
        );
        assert_eq!(first_cycle.processed_tick_count, 2);
        assert_eq!(first_cycle.idle_tick_count, 0);
        assert_eq!(first_cycle.output_events.len(), 2);
        assert_eq!(first_cycle.ticks.len(), 2);
        assert_eq!(first_cycle.ticks[0].stream_tick, Some(0));
        assert_eq!(first_cycle.ticks[1].stream_tick, Some(1));
        assert_eq!(
            first_cycle
                .output_events
                .iter()
                .map(|event| event.output_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(stream.snapshot().next_stream_tick, 2);
        assert!(!stream.snapshot().idle);

        let second_cycle = stream.pump_bounded(&device, 3).unwrap();
        assert_eq!(second_cycle.start_stream_tick, 2);
        assert_eq!(second_cycle.next_stream_tick, 4);
        assert_eq!(
            second_cycle.stop_condition,
            VulkanResidentTokenStreamPumpStopCondition::Idle
        );
        assert_eq!(second_cycle.processed_tick_count, 2);
        assert_eq!(second_cycle.idle_tick_count, 1);
        assert_eq!(second_cycle.output_events.len(), 1);
        assert_eq!(second_cycle.output_events[0].output_index, 2);
        assert_eq!(second_cycle.output_events[0].source_stream_tick, 2);
        assert_eq!(second_cycle.ticks.len(), 3);
        assert_eq!(second_cycle.ticks[0].stream_tick, Some(2));
        assert_eq!(second_cycle.ticks[1].stream_tick, Some(3));
        assert_eq!(second_cycle.ticks[2].stream_tick, None);
        assert_eq!(
            second_cycle.last_stop_reason.as_deref(),
            Some("max_new_tokens")
        );

        let snapshot = stream.snapshot();
        assert_eq!(snapshot.next_stream_tick, 4);
        assert!(snapshot.idle);
        assert_eq!(snapshot.total_public_outputs, 3);
        assert_eq!(snapshot.total_ticks, 5);

        let no_budget = stream.pump_bounded(&device, 0).unwrap();
        assert_eq!(
            no_budget.stop_condition,
            VulkanResidentTokenStreamPumpStopCondition::TickBudget
        );
        assert_eq!(no_budget.processed_tick_count, 0);
        assert_eq!(no_budget.idle_tick_count, 0);
        assert!(no_budget.output_events.is_empty());
        assert!(no_budget.ticks.is_empty());
        assert_eq!(no_budget.start_stream_tick, 4);
        assert_eq!(no_budget.next_stream_tick, 4);
    }

    #[test]
    fn resident_token_runtime_queues_events_and_runs_bounded_cycles() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident token runtime cycle: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor_with_capacity(
            &device,
            "resident token runtime cycle",
            8,
            "gqa_attention_bf16_q16_kv8_d64_cap8.comp",
        ) else {
            return;
        };
        let mut runtime = VulkanResidentTokenRuntime::from_processor("runtime_stream_0", processor);
        let initial = runtime.snapshot();
        assert!(initial.idle);
        assert!(!initial.running);
        assert!(initial.stream.idle);
        assert_eq!(initial.pending_input_event_count, 0);

        let queued_first = runtime
            .enqueue_input_event(
                VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
            )
            .unwrap();
        assert_eq!(queued_first.pending_input_event_count, 1);
        let queued_second = runtime
            .enqueue_input_event(
                VulkanResidentTokenInputEvent::new("event_1", vec![36_309], 1)
                    .with_origin("test_host"),
            )
            .unwrap();
        assert_eq!(queued_second.pending_input_event_count, 2);
        let queued_snapshot = runtime.snapshot();
        assert!(!queued_snapshot.idle);
        assert!(queued_snapshot.running);
        assert!(queued_snapshot.stream.idle);
        assert_eq!(queued_snapshot.pending_input_event_count, 2);

        let no_budget = runtime.run_cycle(&device, 0).unwrap();
        assert_eq!(
            no_budget.stop_condition,
            VulkanResidentTokenRuntimeCycleStopCondition::TickBudget
        );
        assert_eq!(no_budget.ticks_used, 0);
        assert_eq!(no_budget.pending_input_event_count, 2);
        assert!(no_budget.stream_idle);

        let first_cycle = runtime.run_cycle(&device, 2).unwrap();
        assert_eq!(first_cycle.stream_id, "runtime_stream_0");
        assert_eq!(first_cycle.start_stream_tick, 0);
        assert_eq!(first_cycle.next_stream_tick, 2);
        assert_eq!(first_cycle.max_ticks, 2);
        assert_eq!(first_cycle.ticks_used, 2);
        assert_eq!(
            first_cycle.stop_condition,
            VulkanResidentTokenRuntimeCycleStopCondition::TickBudget
        );
        assert_eq!(first_cycle.queued_input_events.len(), 1);
        assert_eq!(first_cycle.queued_input_events[0].input_event.id, "event_0");
        assert_eq!(first_cycle.pending_input_event_count, 1);
        assert!(!first_cycle.stream_idle);
        assert_eq!(first_cycle.processed_tick_count, 2);
        assert_eq!(first_cycle.idle_tick_count, 0);
        assert_eq!(first_cycle.output_events.len(), 2);
        assert_eq!(
            first_cycle
                .output_events
                .iter()
                .map(|event| (event.input_event_id.as_str(), event.output_index))
                .collect::<Vec<_>>(),
            vec![("event_0", 0), ("event_0", 1)]
        );

        let second_cycle = runtime.run_cycle(&device, 4).unwrap();
        assert_eq!(second_cycle.start_stream_tick, 2);
        assert_eq!(second_cycle.next_stream_tick, 5);
        assert_eq!(second_cycle.ticks_used, 4);
        assert_eq!(
            second_cycle.stop_condition,
            VulkanResidentTokenRuntimeCycleStopCondition::TickBudget
        );
        assert_eq!(second_cycle.queued_input_events.len(), 1);
        assert_eq!(
            second_cycle.queued_input_events[0].input_event.id,
            "event_1"
        );
        assert_eq!(second_cycle.pending_input_event_count, 0);
        assert!(!second_cycle.stream_idle);
        assert_eq!(second_cycle.processed_tick_count, 3);
        assert_eq!(second_cycle.idle_tick_count, 1);
        assert_eq!(second_cycle.output_events.len(), 2);
        assert_eq!(
            second_cycle
                .output_events
                .iter()
                .map(|event| (event.input_event_id.as_str(), event.output_index))
                .collect::<Vec<_>>(),
            vec![("event_0", 2), ("event_1", 0)]
        );
        assert_eq!(second_cycle.output_events[0].source_stream_tick, 2);
        assert_eq!(second_cycle.output_events[1].source_stream_tick, 4);

        let final_cycle = runtime.run_cycle(&device, 3).unwrap();
        assert_eq!(final_cycle.start_stream_tick, 5);
        assert_eq!(final_cycle.next_stream_tick, 6);
        assert_eq!(
            final_cycle.stop_condition,
            VulkanResidentTokenRuntimeCycleStopCondition::Idle
        );
        assert_eq!(final_cycle.ticks_used, 2);
        assert_eq!(final_cycle.processed_tick_count, 1);
        assert_eq!(final_cycle.idle_tick_count, 1);
        assert!(final_cycle.output_events.is_empty());
        assert_eq!(final_cycle.pending_input_event_count, 0);
        assert!(final_cycle.stream_idle);

        let idle_cycle = runtime.run_cycle(&device, 3).unwrap();
        assert_eq!(
            idle_cycle.stop_condition,
            VulkanResidentTokenRuntimeCycleStopCondition::Idle
        );
        assert_eq!(idle_cycle.ticks_used, 0);
        assert_eq!(idle_cycle.pending_input_event_count, 0);
        assert!(idle_cycle.stream_idle);

        let final_snapshot = runtime.snapshot();
        assert!(final_snapshot.idle);
        assert!(!final_snapshot.running);
        assert_eq!(final_snapshot.stream.next_stream_tick, 6);
        assert_eq!(final_snapshot.stream.total_public_outputs, 4);
        assert_eq!(final_snapshot.pending_input_event_count, 0);
    }

    #[test]
    fn resident_token_runtime_scheduler_round_robins_registered_runtime() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident token runtime scheduler: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor_with_capacity(
            &device,
            "resident token runtime scheduler",
            8,
            "gqa_attention_bf16_q16_kv8_d64_cap8.comp",
        ) else {
            return;
        };
        let runtime = VulkanResidentTokenRuntime::from_processor("scheduler_stream_0", processor);
        let mut scheduler = VulkanResidentTokenRuntimeScheduler::new();

        let initial = scheduler.snapshot();
        assert_eq!(initial.registered_runtime_count, 0);
        assert_eq!(initial.active_runtime_count, 0);
        assert!(initial.idle);
        assert!(!initial.running);
        assert!(initial.runtimes.is_empty());

        scheduler.add_runtime(runtime).unwrap();
        assert!(scheduler.has_runtime("scheduler_stream_0"));
        let registered = scheduler.snapshot();
        assert_eq!(registered.registered_runtime_count, 1);
        assert_eq!(registered.active_runtime_count, 0);
        assert!(registered.idle);
        assert!(!registered.running);
        assert_eq!(registered.runtimes.len(), 1);
        assert_eq!(
            registered.runtimes[0].stream.stream_id,
            "scheduler_stream_0"
        );

        let queued_first = scheduler
            .enqueue_input_event(
                "scheduler_stream_0",
                VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
            )
            .unwrap();
        assert_eq!(queued_first.pending_input_event_count, 1);
        let queued_second = scheduler
            .enqueue_input_event(
                "scheduler_stream_0",
                VulkanResidentTokenInputEvent::new("event_1", vec![36_309], 1)
                    .with_origin("test_host"),
            )
            .unwrap();
        assert_eq!(queued_second.pending_input_event_count, 2);
        let queued = scheduler.snapshot();
        assert_eq!(queued.registered_runtime_count, 1);
        assert_eq!(queued.active_runtime_count, 1);
        assert!(!queued.idle);
        assert!(queued.running);
        assert_eq!(queued.runtimes[0].pending_input_event_count, 2);
        assert!(queued.runtimes[0].stream.idle);

        let no_budget = scheduler.run_cycle(&device, 0, 2).unwrap();
        assert_eq!(
            no_budget.stop_condition,
            VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
        );
        assert!(no_budget.runtime_cycles.is_empty());
        assert!(no_budget.output_events.is_empty());
        assert_eq!(no_budget.active_runtime_count, 1);
        assert_eq!(no_budget.registered_runtime_count, 1);

        let first = scheduler.run_cycle(&device, 1, 2).unwrap();
        assert_eq!(
            first.stop_condition,
            VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
        );
        assert_eq!(first.runtime_cycles.len(), 1);
        assert_eq!(first.runtime_cycles[0].stream_id, "scheduler_stream_0");
        assert_eq!(first.runtime_cycles[0].start_stream_tick, 0);
        assert_eq!(first.runtime_cycles[0].next_stream_tick, 2);
        assert_eq!(first.runtime_cycles[0].pending_input_event_count, 1);
        assert_eq!(first.output_events.len(), 2);
        assert!(
            first
                .output_events
                .iter()
                .all(|event| event.stream_id == "scheduler_stream_0")
        );
        assert_eq!(
            first
                .output_events
                .iter()
                .map(|event| {
                    (
                        event.output_event.input_event_id.as_str(),
                        event.output_event.output_index,
                    )
                })
                .collect::<Vec<_>>(),
            vec![("event_0", 0), ("event_0", 1)]
        );
        assert_eq!(first.active_runtime_count, 1);
        assert_eq!(first.registered_runtime_count, 1);

        let second = scheduler.run_cycle(&device, 1, 4).unwrap();
        assert_eq!(
            second.stop_condition,
            VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
        );
        assert_eq!(second.runtime_cycles.len(), 1);
        assert_eq!(second.runtime_cycles[0].stream_id, "scheduler_stream_0");
        assert_eq!(second.runtime_cycles[0].start_stream_tick, 2);
        assert_eq!(second.runtime_cycles[0].next_stream_tick, 5);
        assert_eq!(second.runtime_cycles[0].pending_input_event_count, 0);
        assert_eq!(second.output_events.len(), 2);
        assert!(
            second
                .output_events
                .iter()
                .all(|event| event.stream_id == "scheduler_stream_0")
        );
        assert_eq!(
            second
                .output_events
                .iter()
                .map(|event| {
                    (
                        event.output_event.input_event_id.as_str(),
                        event.output_event.output_index,
                    )
                })
                .collect::<Vec<_>>(),
            vec![("event_0", 2), ("event_1", 0)]
        );
        assert_eq!(second.active_runtime_count, 1);
        assert_eq!(second.registered_runtime_count, 1);

        let final_run = scheduler.run_cycle(&device, 1, 3).unwrap();
        assert_eq!(
            final_run.stop_condition,
            VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
        );
        assert_eq!(final_run.runtime_cycles.len(), 1);
        assert_eq!(final_run.runtime_cycles[0].stream_id, "scheduler_stream_0");
        assert_eq!(final_run.runtime_cycles[0].start_stream_tick, 5);
        assert_eq!(final_run.runtime_cycles[0].next_stream_tick, 6);
        assert_eq!(
            final_run.runtime_cycles[0].stop_condition,
            VulkanResidentTokenRuntimeCycleStopCondition::Idle
        );
        assert!(final_run.output_events.is_empty());
        assert_eq!(final_run.active_runtime_count, 0);
        assert_eq!(final_run.registered_runtime_count, 1);

        let idle_run = scheduler.run_cycle(&device, 1, 3).unwrap();
        assert_eq!(
            idle_run.stop_condition,
            VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
        );
        assert!(idle_run.runtime_cycles.is_empty());
        assert!(idle_run.output_events.is_empty());
        assert_eq!(idle_run.active_runtime_count, 0);
        assert_eq!(idle_run.registered_runtime_count, 1);

        let final_snapshot = scheduler.snapshot();
        assert_eq!(final_snapshot.registered_runtime_count, 1);
        assert_eq!(final_snapshot.active_runtime_count, 0);
        assert!(final_snapshot.idle);
        assert!(!final_snapshot.running);
        assert_eq!(final_snapshot.runtimes[0].stream.next_stream_tick, 6);
        assert_eq!(final_snapshot.runtimes[0].stream.total_public_outputs, 4);
        assert_eq!(final_snapshot.runtimes[0].pending_input_event_count, 0);
    }

    #[test]
    fn resident_greedy_running_stream_interrupt_clears_feedback_without_resetting_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy running stream interrupt: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor(
            &device,
            "resident greedy running stream interrupt",
        ) else {
            return;
        };
        let mut stream = processor.into_running_stream("stream_0");

        stream.inject_prompt(&[1], 3, None).unwrap();
        let first_tick = stream.tick(&device).unwrap();
        assert_eq!(first_tick.stream_tick, Some(0));
        assert_eq!(
            first_tick.input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::ExternalInput
        );
        assert!(first_tick.public_output.is_some());
        assert!(first_tick.private_feedback.is_some());
        assert_eq!(stream.remaining_public_outputs, 2);
        assert_eq!(stream.pending_private_feedback_count(), 1);
        assert_eq!(stream.next_stream_tick, 1);

        let event = stream.interrupt("user_interrupt");
        assert_eq!(
            event.event_type,
            VulkanResidentGreedyStreamControlEventType::Interrupt
        );
        assert_eq!(event.reason, "user_interrupt");
        assert_eq!(event.cleared_private_feedback_ids, vec!["feedback_0"]);
        assert_eq!(event.closing_private_feedback_id, None);
        assert!(event.state_preserved);
        assert_eq!(stream.pending_private_feedback_count(), 0);
        assert_eq!(stream.remaining_public_outputs, 0);
        assert!(!stream.loop_open);
        assert_eq!(stream.last_stop_reason.as_deref(), Some("user_interrupt"));

        let idle = stream.tick(&device).unwrap();
        assert_eq!(
            idle.status,
            VulkanResidentGreedyRunningStreamTickStatus::Idle
        );
        assert_eq!(idle.stream_tick, None);
        assert_eq!(stream.next_stream_tick, 1);

        let resumed = stream.run_prompt(&device, &[36_309], 1, None).unwrap();
        assert_eq!(resumed.start_stream_tick, 1);
        assert_eq!(resumed.next_stream_tick, 3);
        assert_eq!(resumed.prompt_token_ids, vec![36_309]);
        assert_eq!(resumed.generated_token_ids.len(), 1);
        assert_eq!(stream.next_stream_tick, 3);
        assert_eq!(stream.public_outputs().len(), 2);
        assert_eq!(stream.private_feedback_history().len(), 2);
    }

    #[test]
    fn resident_greedy_running_stream_stop_after_current_processes_one_feedback_then_idles() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident greedy running stream stop-after-current: {error}");
                return;
            }
        };
        let Some(processor) = create_lfm2_resident_greedy_stream_processor(
            &device,
            "resident greedy running stream stop-after-current",
        ) else {
            return;
        };
        let mut stream = processor.into_running_stream("stream_0");

        stream.inject_prompt(&[1], 3, None).unwrap();
        let first_tick = stream.tick(&device).unwrap();
        assert_eq!(first_tick.stream_tick, Some(0));
        assert!(first_tick.public_output.is_some());
        assert_eq!(
            first_tick
                .private_feedback
                .as_ref()
                .unwrap()
                .closes_loop_after_processing,
            false
        );
        assert_eq!(stream.pending_private_feedback_count(), 1);
        assert_eq!(stream.remaining_public_outputs, 2);

        let event = stream.stop_after_current("user_stop");
        assert_eq!(
            event.event_type,
            VulkanResidentGreedyStreamControlEventType::StopAfterCurrent
        );
        assert_eq!(event.reason, "user_stop");
        assert_eq!(
            event.closing_private_feedback_id.as_deref(),
            Some("feedback_0")
        );
        assert!(event.cleared_private_feedback_ids.is_empty());
        assert!(event.state_preserved);
        assert_eq!(stream.pending_private_feedback_count(), 1);
        assert_eq!(stream.remaining_public_outputs, 0);
        assert!(stream.loop_open);
        assert_eq!(
            stream.private_feedback_history()[0].closes_loop_after_processing,
            true
        );
        assert_eq!(
            stream.private_feedback_history()[0].stop_reason.as_deref(),
            Some("user_stop")
        );

        let closing_tick = stream.tick(&device).unwrap();
        assert_eq!(closing_tick.stream_tick, Some(1));
        assert_eq!(
            closing_tick.input_signal.as_ref().unwrap().route(),
            VulkanResidentGreedyPromptEventInputRoute::PrivateFeedback
        );
        assert!(
            closing_tick
                .input_signal
                .as_ref()
                .unwrap()
                .closes_loop_after_processing()
        );
        assert!(closing_tick.public_output.is_none());
        assert!(closing_tick.private_feedback.is_none());
        assert_eq!(closing_tick.stop_reason.as_deref(), Some("user_stop"));
        assert!(!stream.loop_open);
        assert_eq!(stream.last_stop_reason.as_deref(), Some("user_stop"));

        let idle = stream.tick(&device).unwrap();
        assert_eq!(
            idle.status,
            VulkanResidentGreedyRunningStreamTickStatus::Idle
        );
        assert_eq!(idle.stream_tick, None);
        assert_eq!(stream.next_stream_tick, 2);
        assert_eq!(stream.pending_private_feedback_count(), 0);
        assert_eq!(stream.public_outputs().len(), 1);
        assert_eq!(stream.private_feedback_history().len(), 1);
    }

    #[test]
    fn resident_pedalboard_runner_executes_layer_00_to_layer_01_over_local_cable() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident pedalboard runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = layer_00_level_1_loaded_kernel_pack(&mounted, &mounted_bound)
        else {
            eprintln!("skipping resident pedalboard runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 1);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 32, 104, 512);

        let run = runner.run_zeroed_push_constants(&device).unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 32);

        let layer_00_output_dispatch = mounted_bound.dispatch("layer_00", "ffn_residual").unwrap();
        let layer_00_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_00_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_00_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x86, 0x3f, 0x82, 0x3f, 0x81, 0x3f, 0x7e, 0x3f, 0x83, 0x3f, 0x83, 0x3f, 0x83, 0x3f,
                0x83, 0x3f,
            ]
        );

        let layer_01_output_dispatch = mounted_bound.dispatch("layer_01", "ffn_residual").unwrap();
        let layer_01_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_01_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_01_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x86, 0x3f, 0x84, 0x3f, 0x80, 0x3f, 0x7f, 0x3f, 0x83, 0x3f, 0x84, 0x3f, 0x88, 0x3f,
                0x83, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_attention_layer_02_with_per_pedal_kv_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident attention pedalboard runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!(
                "skipping resident attention pedalboard runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 2);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 51, 167, 816);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 51);

        let kv_memory = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap();
        assert_ne!(kv_memory.buffer.read_bytes(16).unwrap(), vec![0; 16]);

        let layer_02_output_dispatch = mounted_bound.dispatch("layer_02", "ffn_residual").unwrap();
        let layer_02_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_02_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_02_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x8b, 0x3f, 0x7e, 0x3f, 0x87, 0x3f, 0x6b, 0x3f, 0x71, 0x3f, 0x87, 0x3f, 0x8a, 0x3f,
                0x7e, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_attention_pedal_reuses_kv_state_across_stream_ticks() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident multi-tick attention runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!(
                "skipping resident multi-tick attention runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 2);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        let layer_02_runner = mounted
            .create_resident_pedal_runner(&device, &mounted_bound, "layer_02", &loaded_manifest)
            .unwrap();
        let dynamic_state_capacity_activations =
            mounted.buffers.dynamic_state_capacity_activations as u32;

        runner
            .run_with_stream_control(
                &device,
                VulkanMountedPlacedStreamControl {
                    stream_tick: 0,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            )
            .unwrap();
        let kv_memory = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap();
        let kv_after_tick_0 = kv_memory.buffer.read_bytes(2_064).unwrap();
        let tick_0_slot_0 = kv_after_tick_0[0..16].to_vec();
        assert_ne!(tick_0_slot_0, vec![0; 16]);
        assert_eq!(&kv_after_tick_0[2_048..2_064], &[0u8; 16]);

        write_layer_00_constant_input(&mounted, [0x00, 0x3f]);
        runner
            .run_with_stream_control(
                &device,
                VulkanMountedPlacedStreamControl {
                    stream_tick: 1,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            )
            .unwrap();

        let layer_02_output_dispatch = mounted_bound.dispatch("layer_02", "ffn_residual").unwrap();
        let layer_02_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_02_output_dispatch)
            .unwrap();
        let historical_output = layer_02_output_bindings[2]
            .buffer
            .read_bytes(2_048)
            .unwrap();
        let kv_after_tick_1 = kv_memory.buffer.read_bytes(4_112).unwrap();
        assert_eq!(&kv_after_tick_1[0..16], tick_0_slot_0.as_slice());
        assert_ne!(&kv_after_tick_1[2_048..2_064], &[0u8; 16]);
        assert_ne!(&kv_after_tick_1[2_048..2_064], tick_0_slot_0.as_slice());

        zero_lfm2_kv_memory(&mounted, "layer_02");
        layer_02_runner
            .run_with_stream_control(
                &device,
                VulkanMountedPlacedStreamControl {
                    stream_tick: 1,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            )
            .unwrap();
        let no_history_output = layer_02_output_bindings[2]
            .buffer
            .read_bytes(2_048)
            .unwrap();
        let kv_after_no_history = kv_memory.buffer.read_bytes(4_112).unwrap();
        assert_eq!(&kv_after_no_history[0..16], &[0u8; 16]);
        assert_ne!(&kv_after_no_history[2_048..2_064], &[0u8; 16]);
        assert_ne!(historical_output, no_history_output);
    }

    #[test]
    fn resident_pedalboard_runner_executes_attention_output_into_next_conv_layer() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_03 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_03 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 3);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 67, 219, 1072);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 67);

        let layer_03_output_dispatch = mounted_bound.dispatch("layer_03", "ffn_residual").unwrap();
        let layer_03_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_03_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_03_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x89, 0x3f, 0x73, 0x3f, 0x86, 0x3f, 0x6c, 0x3f, 0x6f, 0x3f, 0x88, 0x3f, 0x88, 0x3f,
                0x83, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_second_attention_layer_with_independent_kv_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_04 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_04 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 4);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 86, 282, 1376);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 86);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);

        let layer_04_output_dispatch = mounted_bound.dispatch("layer_04", "ffn_residual").unwrap();
        let layer_04_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_04_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_04_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x8c, 0x3f, 0x62, 0x3f, 0x88, 0x3f, 0x62, 0x3f, 0x73, 0x3f, 0x85, 0x3f, 0x89, 0x3f,
                0x84, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_second_attention_output_into_next_conv_layer() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_05 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_05 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 5);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 102, 334, 1632);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 102);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);

        let layer_05_output_dispatch = mounted_bound.dispatch("layer_05", "ffn_residual").unwrap();
        let layer_05_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_05_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_05_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x8a, 0x3f, 0x61, 0x3f, 0x86, 0x3f, 0x61, 0x3f, 0x74, 0x3f, 0x85, 0x3f, 0x8a, 0x3f,
                0x83, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_third_attention_layer_with_independent_kv_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_06 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_06 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 6);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 121, 397, 1936);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 121);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_06_kv);

        let layer_06_output_dispatch = mounted_bound.dispatch("layer_06", "ffn_residual").unwrap();
        let layer_06_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_06_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_06_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x8b, 0x3f, 0x5f, 0x3f, 0x80, 0x3f, 0x6a, 0x3f, 0x7a, 0x3f, 0x8d, 0x3f, 0x88, 0x3f,
                0x80, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_third_attention_output_into_next_conv_layer() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_07 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_07 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 7);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 137, 449, 2192);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 137);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_06_kv);

        let layer_07_output_dispatch = mounted_bound.dispatch("layer_07", "ffn_residual").unwrap();
        let layer_07_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_07_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_07_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x8a, 0x3f, 0x62, 0x3f, 0x7f, 0x3f, 0x6f, 0x3f, 0x7c, 0x3f, 0x8a, 0x3f, 0x8a, 0x3f,
                0x7f, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_fourth_attention_layer_with_independent_kv_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_08 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_08 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 8);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 156, 512, 2496);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 156);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_08_kv = mounted
            .buffers
            .state_buffer("layer_08", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_08_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_02_kv, layer_08_kv);
        assert_ne!(layer_04_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_08_kv);
        assert_ne!(layer_06_kv, layer_08_kv);

        let layer_08_output_dispatch = mounted_bound.dispatch("layer_08", "ffn_residual").unwrap();
        let layer_08_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_08_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_08_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x97, 0x3f, 0x63, 0x3f, 0x86, 0x3f, 0x61, 0x3f, 0x69, 0x3f, 0x8d, 0x3f, 0x83, 0x3f,
                0x71, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_fourth_attention_output_into_next_conv_layer() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_09 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_09 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 9);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 172, 564, 2752);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 172);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_08_kv = mounted
            .buffers
            .state_buffer("layer_08", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_08_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_02_kv, layer_08_kv);
        assert_ne!(layer_04_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_08_kv);
        assert_ne!(layer_06_kv, layer_08_kv);

        let layer_09_output_dispatch = mounted_bound.dispatch("layer_09", "ffn_residual").unwrap();
        let layer_09_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_09_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_09_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x95, 0x3f, 0x5c, 0x3f, 0x83, 0x3f, 0x63, 0x3f, 0x78, 0x3f, 0x8e, 0x3f, 0x82, 0x3f,
                0x7b, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_fifth_attention_layer_with_independent_kv_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_10 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_10 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 10);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 191, 627, 3056);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 191);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_08_kv = mounted
            .buffers
            .state_buffer("layer_08", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_10_kv = mounted
            .buffers
            .state_buffer("layer_10", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_08_kv, vec![0; 16]);
        assert_ne!(layer_10_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_02_kv, layer_08_kv);
        assert_ne!(layer_02_kv, layer_10_kv);
        assert_ne!(layer_04_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_08_kv);
        assert_ne!(layer_04_kv, layer_10_kv);
        assert_ne!(layer_06_kv, layer_08_kv);
        assert_ne!(layer_06_kv, layer_10_kv);
        assert_ne!(layer_08_kv, layer_10_kv);

        let layer_10_output_dispatch = mounted_bound.dispatch("layer_10", "ffn_residual").unwrap();
        let layer_10_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_10_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_10_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x94, 0x3f, 0x53, 0x3f, 0x85, 0x3f, 0x46, 0x3f, 0x90, 0x3f, 0x94, 0x3f, 0x87, 0x3f,
                0x63, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_fifth_attention_output_into_next_conv_layer() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_11 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_11 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 11);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 207, 679, 3312);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 207);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_08_kv = mounted
            .buffers
            .state_buffer("layer_08", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_10_kv = mounted
            .buffers
            .state_buffer("layer_10", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_08_kv, vec![0; 16]);
        assert_ne!(layer_10_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_02_kv, layer_08_kv);
        assert_ne!(layer_02_kv, layer_10_kv);
        assert_ne!(layer_04_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_08_kv);
        assert_ne!(layer_04_kv, layer_10_kv);
        assert_ne!(layer_06_kv, layer_08_kv);
        assert_ne!(layer_06_kv, layer_10_kv);
        assert_ne!(layer_08_kv, layer_10_kv);

        let layer_11_output_dispatch = mounted_bound.dispatch("layer_11", "ffn_residual").unwrap();
        let layer_11_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_11_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_11_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x95, 0x3f, 0x4b, 0x3f, 0x86, 0x3f, 0x30, 0x3f, 0x93, 0x3f, 0x9c, 0x3f, 0x82, 0x3f,
                0x64, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_final_attention_layer_with_independent_kv_state() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident layer_12 prefix runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!("skipping resident layer_12 prefix runner: no GLSL to SPIR-V compiler found");
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 12);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 226, 742, 3616);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 226);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_08_kv = mounted
            .buffers
            .state_buffer("layer_08", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_10_kv = mounted
            .buffers
            .state_buffer("layer_10", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_12_kv = mounted
            .buffers
            .state_buffer("layer_12", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_08_kv, vec![0; 16]);
        assert_ne!(layer_10_kv, vec![0; 16]);
        assert_ne!(layer_12_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_02_kv, layer_08_kv);
        assert_ne!(layer_02_kv, layer_10_kv);
        assert_ne!(layer_02_kv, layer_12_kv);
        assert_ne!(layer_04_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_08_kv);
        assert_ne!(layer_04_kv, layer_10_kv);
        assert_ne!(layer_04_kv, layer_12_kv);
        assert_ne!(layer_06_kv, layer_08_kv);
        assert_ne!(layer_06_kv, layer_10_kv);
        assert_ne!(layer_06_kv, layer_12_kv);
        assert_ne!(layer_08_kv, layer_10_kv);
        assert_ne!(layer_08_kv, layer_12_kv);
        assert_ne!(layer_10_kv, layer_12_kv);

        let layer_12_output_dispatch = mounted_bound.dispatch("layer_12", "ffn_residual").unwrap();
        let layer_12_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_12_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_12_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x98, 0x3f, 0x4a, 0x3f, 0x6f, 0x3f, 0x1d, 0x3f, 0x8f, 0x3f, 0x9c, 0x3f, 0x37, 0x3f,
                0x6d, 0x3f,
            ]
        );
    }

    #[test]
    fn resident_pedalboard_runner_executes_full_lfm2_layer_stack() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping resident full LFM2 layer-stack runner: {error}");
                return;
            }
        };
        let (tensor_index, mounted, _manifest, mounted_bound) =
            mount_lfm2_single_device_stream_circuit(&device);
        let Some(loaded_manifest) = lfm2_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        ) else {
            eprintln!(
                "skipping resident full LFM2 layer-stack runner: no GLSL to SPIR-V compiler found"
            );
            return;
        };
        let pedal_ids = prepare_lfm2_resident_prefix(&mounted, &tensor_index, 13);

        let runner = create_lfm2_resident_prefix_runner(
            &device,
            &mounted,
            &mounted_bound,
            &loaded_manifest,
            &pedal_ids,
        );
        assert_lfm2_resident_prefix_runner(&runner, &pedal_ids, 242, 794, 3872);

        let run = runner
            .run_with_stream_control(&device, lfm2_stream_control(&mounted, 0))
            .unwrap();
        assert_lfm2_resident_prefix_run(&run, &pedal_ids, 242);

        let layer_02_kv = mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_04_kv = mounted
            .buffers
            .state_buffer("layer_04", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_06_kv = mounted
            .buffers
            .state_buffer("layer_06", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_08_kv = mounted
            .buffers
            .state_buffer("layer_08", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_10_kv = mounted
            .buffers
            .state_buffer("layer_10", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        let layer_12_kv = mounted
            .buffers
            .state_buffer("layer_12", "kv_memory")
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap();
        assert_ne!(layer_02_kv, vec![0; 16]);
        assert_ne!(layer_04_kv, vec![0; 16]);
        assert_ne!(layer_06_kv, vec![0; 16]);
        assert_ne!(layer_08_kv, vec![0; 16]);
        assert_ne!(layer_10_kv, vec![0; 16]);
        assert_ne!(layer_12_kv, vec![0; 16]);
        assert_ne!(layer_02_kv, layer_04_kv);
        assert_ne!(layer_02_kv, layer_06_kv);
        assert_ne!(layer_02_kv, layer_08_kv);
        assert_ne!(layer_02_kv, layer_10_kv);
        assert_ne!(layer_02_kv, layer_12_kv);
        assert_ne!(layer_04_kv, layer_06_kv);
        assert_ne!(layer_04_kv, layer_08_kv);
        assert_ne!(layer_04_kv, layer_10_kv);
        assert_ne!(layer_04_kv, layer_12_kv);
        assert_ne!(layer_06_kv, layer_08_kv);
        assert_ne!(layer_06_kv, layer_10_kv);
        assert_ne!(layer_06_kv, layer_12_kv);
        assert_ne!(layer_08_kv, layer_10_kv);
        assert_ne!(layer_08_kv, layer_12_kv);
        assert_ne!(layer_10_kv, layer_12_kv);

        let layer_13_output_dispatch = mounted_bound.dispatch("layer_13", "ffn_residual").unwrap();
        let layer_13_output_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(layer_13_output_dispatch)
            .unwrap();
        assert_eq!(
            layer_13_output_bindings[2].buffer.read_bytes(16).unwrap(),
            vec![
                0x84, 0x3f, 0x48, 0x3f, 0x82, 0x3f, 0x10, 0x3f, 0x92, 0x3f, 0x9f, 0x3f, 0x16, 0x3f,
                0x63, 0x3f,
            ]
        );
    }

    #[test]
    fn mounted_single_device_stream_circuit_binds_local_cable_buffers() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping single-device placed Vulkan stream-circuit mount: {error}");
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
        let placement_spec = StreamCircuitPlacementSpec::new("gpu0");
        let placement_plan = graph.placement_plan(&placement_spec).unwrap();
        let resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();

        assert_eq!(resident.hosted_pedal_ids.len(), 14);
        assert_eq!(resident.local_cables.len(), 13);
        assert_eq!(resident.incoming_cables.len(), 0);
        assert_eq!(resident.outgoing_cables.len(), 0);

        let placed_plan =
            VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, resident)
                .unwrap();
        let mounted =
            VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, placed_plan, 4).unwrap();

        assert_eq!(mounted.device_id(), "gpu0");
        assert!(!mounted.can_execute());
        assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 14);
        assert_eq!(
            mounted.placed_plan.dispatch_plan.total_dispatch_count(),
            242
        );
        assert_eq!(mounted.parameter_buffers.plan.device_id, "gpu0");
        assert_eq!(mounted.parameter_buffers.plan.parameter_count, 130);
        assert_eq!(
            mounted.parameter_buffers.plan.total_byte_capacity,
            Some(325_166_592)
        );
        assert!(mounted.parameter_buffers.plan.unresolved_tensors.is_empty());
        assert_eq!(mounted.parameter_buffers.total_byte_capacity, 325_166_592);
        let operator_norm_weight = mounted
            .parameter_buffers
            .parameter_buffer("model.layers.0.operator_norm.weight")
            .unwrap();
        assert_eq!(
            operator_norm_weight.parameter.dtype.as_deref(),
            Some("BF16")
        );
        assert_eq!(operator_norm_weight.parameter.shape, Some(vec![1024]));
        assert_eq!(operator_norm_weight.byte_capacity, 2_048);
        operator_norm_weight
            .buffer
            .write_bytes(&[21, 22, 23, 24])
            .unwrap();
        assert_eq!(
            operator_norm_weight.buffer.read_bytes(4).unwrap(),
            vec![21, 22, 23, 24]
        );
        let operator_norm_metadata = tensor_index
            .tensors
            .get("model.layers.0.operator_norm.weight")
            .unwrap();
        let operator_norm_source_available = operator_norm_metadata
            .source_file
            .as_ref()
            .map(|source_file| Path::new(source_file).exists())
            .unwrap_or(false);
        if operator_norm_source_available {
            let loaded_weight = mounted
                .parameter_buffers
                .load_parameter_from_tensor_index(
                    &tensor_index,
                    "model.layers.0.operator_norm.weight",
                )
                .unwrap();
            assert_eq!(loaded_weight.tensor, "model.layers.0.operator_norm.weight");
            assert_eq!(loaded_weight.data_start, 158_345_216);
            assert_eq!(loaded_weight.data_end, 158_347_264);
            assert_eq!(loaded_weight.byte_count, 2_048);
            assert_eq!(
                operator_norm_weight.buffer.read_bytes(16).unwrap(),
                vec![
                    0xc6, 0x3e, 0xb9, 0x3e, 0xba, 0x3e, 0xba, 0x3e, 0xc2, 0x3e, 0xba, 0x3e, 0xbe,
                    0x3e, 0x12, 0x3f,
                ]
            );
        } else {
            eprintln!(
                "skipping real safetensors parameter load: source file for model.layers.0.operator_norm.weight is unavailable"
            );
        }
        assert_eq!(mounted.boundary_io.plan.device_id, "gpu0");
        assert_eq!(mounted.boundary_io.plan.input_count, 1);
        assert_eq!(mounted.boundary_io.plan.output_count, 1);
        assert_eq!(mounted.boundary_io.plan.total_buffer_count, 2);
        assert_eq!(mounted.boundary_io.plan.total_byte_capacity, Some(4_096));
        assert_eq!(mounted.boundary_io.total_byte_capacity, 4_096);
        let model_input = mounted.boundary_io.input_buffer("input_frame").unwrap();
        assert_eq!(model_input.boundary.pedal_id, "layer_00");
        assert_eq!(model_input.boundary.port_id, "input_frame");
        assert_eq!(model_input.boundary.shape, vec![1024]);
        assert_eq!(model_input.byte_capacity, 2_048);
        model_input.buffer.write_bytes(&[1, 2, 3, 4]).unwrap();
        assert_eq!(model_input.buffer.read_bytes(4).unwrap(), vec![1, 2, 3, 4]);
        let model_output = mounted.boundary_io.output_buffer("output_frame").unwrap();
        assert_eq!(model_output.boundary.pedal_id, "layer_13");
        assert_eq!(model_output.boundary.port_id, "output_frame");
        assert_eq!(model_output.byte_capacity, 2_048);
        assert_eq!(mounted.cable_io.plan.local_cable_count, 13);
        assert_eq!(mounted.cable_io.plan.total_endpoint_count, 0);
        assert_eq!(mounted.cable_io.plan.total_buffer_count, 13);
        assert_eq!(mounted.cable_io.plan.total_byte_capacity, Some(26_624));
        assert_eq!(mounted.cable_io.local_buffers.len(), 13);
        assert_eq!(mounted.cable_io.incoming_buffers.len(), 0);
        assert_eq!(mounted.cable_io.outgoing_buffers.len(), 0);
        assert_eq!(mounted.cable_io.total_byte_capacity, 26_624);
        let first_local_cable = mounted.cable_io.local_cable_buffer(0).unwrap();
        assert_eq!(first_local_cable.cable.cable_id, "cable_0_local");
        assert_eq!(first_local_cable.cable.source_pedal_id, "layer_00");
        assert_eq!(first_local_cable.cable.destination_pedal_id, "layer_01");
        assert_eq!(first_local_cable.cable.byte_capacity, Some(2_048));
        assert_eq!(first_local_cable.byte_capacity, 2_048);
        assert_eq!(first_local_cable.buffer.byte_capacity(), 2_048);
        first_local_cable
            .buffer
            .write_bytes(&[11, 12, 13, 14])
            .unwrap();
        assert_eq!(
            first_local_cable.buffer.read_bytes(4).unwrap(),
            vec![11, 12, 13, 14]
        );

        let manifest = VulkanReusableKernelArtifactManifest::new(
            mounted
                .placed_plan
                .reusable_kernel_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );
        let placed_bound = mounted.placed_bound_dispatch_plan(&manifest).unwrap();
        assert_eq!(placed_bound.device_id, "gpu0");
        assert_eq!(placed_bound.dispatches.len(), 242);
        assert_eq!(placed_bound.model_boundary_descriptor_count, 3);
        assert_eq!(placed_bound.local_cable_descriptor_count, 39);
        assert_eq!(placed_bound.incoming_cable_descriptor_count, 0);
        assert_eq!(placed_bound.outgoing_cable_descriptor_count, 0);

        let mounted_bound = mounted
            .mounted_placed_bound_dispatch_plan(&manifest)
            .unwrap();
        assert_eq!(mounted_bound.device_id, "gpu0");
        assert_eq!(mounted_bound.dispatches.len(), 242);
        assert_eq!(
            mounted_bound.total_descriptor_count,
            placed_bound.total_descriptor_count
        );
        assert_eq!(mounted_bound.model_boundary_descriptor_count, 3);
        assert_eq!(mounted_bound.local_cable_descriptor_count, 39);
        assert_eq!(mounted_bound.cable_endpoint_descriptor_count, 0);
        assert_eq!(mounted_bound.incoming_cable_descriptor_count, 0);
        assert_eq!(mounted_bound.outgoing_cable_descriptor_count, 0);

        let tick_plan = mounted.stream_tick_plan(&manifest).unwrap();
        assert_eq!(tick_plan.device_id, "gpu0");
        assert!(!tick_plan.can_execute);
        assert_eq!(tick_plan.stage_count, 242);
        assert_eq!(tick_plan.receive_stage_count, 0);
        assert_eq!(tick_plan.dispatch_stage_count, 242);
        assert_eq!(tick_plan.publish_stage_count, 0);
        assert_eq!(tick_plan.local_cable_read_count, 26);
        assert_eq!(tick_plan.local_cable_write_count, 13);
        assert_eq!(tick_plan.incoming_cable_read_count, 0);
        assert_eq!(tick_plan.outgoing_cable_write_count, 0);
        assert_eq!(tick_plan.model_input_read_count, 2);
        assert_eq!(tick_plan.model_output_write_count, 1);
        assert_eq!(
            tick_plan.stages[0],
            VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index: 0,
                dispatch: VulkanMountedPlacedStreamTickDispatch {
                    dispatch_index: 0,
                    kernel_id: "layer_00.operator_norm".to_string(),
                    pedal_id: "layer_00".to_string(),
                    node_id: "operator_norm".to_string(),
                    op: "rms_norm".to_string(),
                    descriptor_count: mounted_bound
                        .dispatch("layer_00", "operator_norm")
                        .unwrap()
                        .descriptors
                        .len(),
                    resident_descriptor_count: 2,
                    reads: vec![VulkanMountedPlacedStreamTickIo::ModelSignal {
                        signal_id: "input_frame".to_string(),
                    }],
                    writes: vec![],
                },
            }
        );
        let tick_run = mounted.advance_stream_tick(&manifest, 42).unwrap();
        assert_eq!(tick_run.device_id, "gpu0");
        assert_eq!(tick_run.stream_tick, 42);
        assert!(!tick_run.can_execute);
        assert_eq!(tick_run.planned_stage_count, 242);
        assert_eq!(tick_run.attempted_stage_count, 1);
        assert_eq!(tick_run.completed_stage_count, 0);
        assert_eq!(tick_run.pending_stage_count, 241);
        assert_eq!(
            tick_run.status,
            VulkanMountedPlacedStreamTickRunStatus::Blocked {
                stage_index: 0,
                reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
            }
        );
        assert_eq!(tick_run.stages[0].stage, tick_plan.stages[0]);
        assert_eq!(
            tick_run.stages[0].status,
            VulkanMountedPlacedStreamTickStageStatus::Blocked {
                reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
            }
        );
        assert_eq!(
            tick_run.stages[1].status,
            VulkanMountedPlacedStreamTickStageStatus::Pending
        );
        let operator_norm_dispatch = mounted_bound.dispatch("layer_00", "operator_norm").unwrap();
        let operator_norm_family_id = operator_norm_dispatch.reusable_family_id.as_str();
        let operator_norm_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(operator_norm_dispatch)
            .unwrap();
        assert_eq!(
            operator_norm_bindings.len(),
            operator_norm_dispatch.descriptors.len()
        );
        assert_eq!(operator_norm_bindings[0].binding, 0);
        assert_eq!(operator_norm_bindings[0].byte_len, 2_048);
        assert_eq!(operator_norm_bindings[1].binding, 1);
        assert_eq!(operator_norm_bindings[1].byte_len, 5_120);
        assert_eq!(operator_norm_bindings[2].binding, 2);
        assert_eq!(operator_norm_bindings[2].byte_len, 2_048);

        let empty_loaded_manifest = VulkanLoadedReusableKernelArtifactManifest {
            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            artifacts: Vec::new(),
            total_word_count: 0,
        };
        let empty_readiness = mounted
            .resident_kernel_dispatch_readiness_plan(&manifest, &empty_loaded_manifest)
            .unwrap();
        assert_eq!(empty_readiness.device_id, "gpu0");
        assert_eq!(empty_readiness.dispatch_count, 242);
        assert_eq!(empty_readiness.instantiable_count, 0);
        assert_eq!(empty_readiness.blocked_count, 242);
        assert_eq!(empty_readiness.missing_loaded_artifact_count, 242);
        assert_eq!(empty_readiness.descriptor_binding_blocked_count, 0);
        assert_eq!(empty_readiness.push_constant_blocked_count, 0);
        assert_eq!(empty_readiness.instantiable_descriptor_count, 0);
        assert!(matches!(
            empty_readiness
                .dispatch("layer_00", "operator_norm")
                .unwrap()
                .status,
            VulkanMountedPlacedResidentKernelDispatchStatus::Blocked {
                error:
                    VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                        dispatch_index: 0,
                        ref family_id,
                    },
            } if family_id == operator_norm_family_id
        ));

        let rms_norm_family = mounted
            .placed_plan
            .reusable_kernel_plan
            .family(operator_norm_family_id)
            .unwrap();
        assert_eq!(operator_norm_family_id, "rms_norm.signature_1");
        let rms_norm_loaded_manifest = VulkanLoadedReusableKernelArtifactManifest {
            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            total_word_count: 2,
            artifacts: vec![VulkanLoadedReusableKernelArtifact {
                artifact: VulkanReusableKernelArtifact::from_family(
                    rms_norm_family,
                    format!("kernels/{operator_norm_family_id}.spv"),
                ),
                resolved_path: PathBuf::from(format!("kernels/{operator_norm_family_id}.spv")),
                words: vec![0x0723_0203, 0],
            }],
        };
        let rms_norm_readiness = mounted
            .resident_kernel_dispatch_readiness_plan(&manifest, &rms_norm_loaded_manifest)
            .unwrap();
        assert_eq!(
            rms_norm_readiness.instantiable_count,
            rms_norm_family.command_refs.len()
        );
        assert_eq!(
            rms_norm_readiness.blocked_count,
            rms_norm_readiness.dispatch_count - rms_norm_family.command_refs.len()
        );
        assert_eq!(
            rms_norm_readiness.missing_loaded_artifact_count,
            rms_norm_readiness.blocked_count
        );
        assert_eq!(rms_norm_readiness.descriptor_binding_blocked_count, 0);
        assert_eq!(rms_norm_readiness.push_constant_blocked_count, 0);
        assert!(matches!(
            rms_norm_readiness
                .dispatch("layer_00", "operator_norm")
                .unwrap()
                .status,
            VulkanMountedPlacedResidentKernelDispatchStatus::Instantiable {
                descriptor_count: 3,
                workgroup_count_x: 1,
                local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
                push_constant_byte_count: 16,
            }
        ));
        if operator_norm_source_available {
            if let Some(spirv_words) = crate::vulkan_compute::compile_test_shader_words_from_source(
                "rms_norm_bf16_serial.comp",
            ) {
                let rms_norm_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                    schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                    backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                    total_word_count: spirv_words.len(),
                    artifacts: vec![VulkanLoadedReusableKernelArtifact {
                        artifact: VulkanReusableKernelArtifact::from_family(
                            rms_norm_family,
                            format!("kernels/{operator_norm_family_id}.spv"),
                        ),
                        resolved_path: PathBuf::from(format!(
                            "kernels/{operator_norm_family_id}.spv"
                        )),
                        words: spirv_words,
                    }],
                };
                let resident_dispatch = mounted
                    .create_resident_kernel_dispatch_for_bound_dispatch(
                        &device,
                        operator_norm_dispatch,
                        &rms_norm_kernel_manifest,
                    )
                    .unwrap();
                let mut input_frame = Vec::with_capacity(2_048);
                for _ in 0..1024 {
                    input_frame.extend_from_slice(&[0x80, 0x3f]);
                }
                model_input.buffer.write_bytes(&input_frame).unwrap();

                device
                    .run_resident_kernel_dispatch(&resident_dispatch, &[0u8; 16])
                    .unwrap();

                assert_eq!(
                    operator_norm_bindings[1].buffer.read_bytes(16).unwrap(),
                    vec![
                        0xc6, 0x3e, 0xb9, 0x3e, 0xba, 0x3e, 0xba, 0x3e, 0xc2, 0x3e, 0xba, 0x3e,
                        0xbe, 0x3e, 0x12, 0x3f,
                    ]
                );

                mounted
                    .parameter_buffers
                    .load_parameter_from_tensor_index(
                        &tensor_index,
                        "model.layers.0.conv.in_proj.weight",
                    )
                    .unwrap();
                if let Some(linear_spirv_words) =
                    crate::vulkan_compute::compile_test_shader_words_from_source(
                        "linear_bf16_1024x3072.comp",
                    )
                {
                    let conv_in_dispatch = mounted_bound
                        .dispatch("layer_00", "conv_in_projection")
                        .unwrap();
                    let conv_in_bindings = mounted
                        .resident_kernel_buffer_bindings_for_bound_dispatch(conv_in_dispatch)
                        .unwrap();
                    assert_eq!(conv_in_bindings[0].byte_len, 5_120);
                    assert_eq!(conv_in_bindings[1].byte_len, 6_144);
                    assert_eq!(conv_in_bindings[2].byte_len, 6_291_456);
                    let linear_family = mounted
                        .placed_plan
                        .reusable_kernel_plan
                        .family(&conv_in_dispatch.reusable_family_id)
                        .unwrap();
                    assert_eq!(linear_family.op, "linear");
                    assert_eq!(linear_family.command_refs.len(), 8);
                    let linear_artifact_path = artifact_path_for_family(linear_family);
                    let linear_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                        total_word_count: linear_spirv_words.len(),
                        artifacts: vec![VulkanLoadedReusableKernelArtifact {
                            artifact: VulkanReusableKernelArtifact::from_family(
                                linear_family,
                                linear_artifact_path.clone(),
                            ),
                            resolved_path: PathBuf::from(linear_artifact_path),
                            words: linear_spirv_words,
                        }],
                    };
                    let linear_dispatch = mounted
                        .create_resident_kernel_dispatch_for_bound_dispatch(
                            &device,
                            conv_in_dispatch,
                            &linear_kernel_manifest,
                        )
                        .unwrap();
                    assert_eq!(linear_dispatch.workgroup_count_x(), 48);

                    device
                        .run_resident_kernel_dispatch(&linear_dispatch, &[0u8; 16])
                        .unwrap();

                    assert_eq!(
                        conv_in_bindings[1].buffer.read_bytes(16).unwrap(),
                        vec![
                            0xc7, 0x3e, 0x74, 0xbe, 0x7f, 0x3e, 0x97, 0x3e, 0x5a, 0xbe, 0xd2, 0xbe,
                            0xab, 0xbe, 0xc5, 0xbd,
                        ]
                    );

                    if let Some(split_spirv_words) =
                        crate::vulkan_compute::compile_test_shader_words_from_source(
                            "split_bf16_3072_to_3x1024.comp",
                        )
                    {
                        let split_dispatch =
                            mounted_bound.dispatch("layer_00", "split_b_c_x").unwrap();
                        assert_eq!(split_dispatch.reusable_family_id, "split");
                        let split_bindings = mounted
                            .resident_kernel_buffer_bindings_for_bound_dispatch(split_dispatch)
                            .unwrap();
                        assert_eq!(split_bindings[0].byte_len, 6_144);
                        assert_eq!(split_bindings[1].byte_len, 5_120);
                        assert_eq!(split_bindings[2].byte_len, 5_120);
                        assert_eq!(split_bindings[3].byte_len, 5_120);
                        let split_family = mounted
                            .placed_plan
                            .reusable_kernel_plan
                            .family(&split_dispatch.reusable_family_id)
                            .unwrap();
                        let split_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                            total_word_count: split_spirv_words.len(),
                            artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                artifact: VulkanReusableKernelArtifact::from_family(
                                    split_family,
                                    "kernels/split.spv",
                                ),
                                resolved_path: PathBuf::from("kernels/split.spv"),
                                words: split_spirv_words,
                            }],
                        };
                        let split_resident_dispatch = mounted
                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                &device,
                                split_dispatch,
                                &split_kernel_manifest,
                            )
                            .unwrap();
                        assert_eq!(split_resident_dispatch.workgroup_count_x(), 1);

                        device
                            .run_resident_kernel_dispatch(&split_resident_dispatch, &[0u8; 16])
                            .unwrap();

                        assert_eq!(
                            split_bindings[1].buffer.read_bytes(16).unwrap(),
                            vec![
                                0xc7, 0x3e, 0x74, 0xbe, 0x7f, 0x3e, 0x97, 0x3e, 0x5a, 0xbe, 0xd2,
                                0xbe, 0xab, 0xbe, 0xc5, 0xbd,
                            ]
                        );
                        assert_eq!(
                            split_bindings[2].buffer.read_bytes(16).unwrap(),
                            vec![
                                0x04, 0xbf, 0x91, 0x3e, 0x9c, 0x3e, 0xd8, 0xbe, 0x9d, 0x3d, 0xe1,
                                0xbc, 0x87, 0x3d, 0x15, 0x3f,
                            ]
                        );
                        assert_eq!(
                            split_bindings[3].buffer.read_bytes(16).unwrap(),
                            vec![
                                0x16, 0xbe, 0xeb, 0xbe, 0x8c, 0xbc, 0xc3, 0x3d, 0x4d, 0xbf, 0x63,
                                0xbb, 0x40, 0xbe, 0x48, 0xbf,
                            ]
                        );

                        if let Some(multiply_spirv_words) =
                            crate::vulkan_compute::compile_test_shader_words_from_source(
                                "multiply_bf16_1024.comp",
                            )
                        {
                            let multiply_dispatch =
                                mounted_bound.dispatch("layer_00", "input_gate").unwrap();
                            assert_eq!(
                                multiply_dispatch.reusable_family_id,
                                "multiply.signature_1"
                            );
                            let multiply_bindings = mounted
                                .resident_kernel_buffer_bindings_for_bound_dispatch(
                                    multiply_dispatch,
                                )
                                .unwrap();
                            assert_eq!(multiply_bindings[0].byte_len, 5_120);
                            assert_eq!(multiply_bindings[1].byte_len, 5_120);
                            assert_eq!(multiply_bindings[2].byte_len, 6_144);
                            let multiply_family = mounted
                                .placed_plan
                                .reusable_kernel_plan
                                .family(&multiply_dispatch.reusable_family_id)
                                .unwrap();
                            let multiply_kernel_manifest =
                                VulkanLoadedReusableKernelArtifactManifest {
                                    schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                        .to_string(),
                                    backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                                    total_word_count: multiply_spirv_words.len(),
                                    artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                        artifact: VulkanReusableKernelArtifact::from_family(
                                            multiply_family,
                                            "kernels/multiply.signature_1.spv",
                                        ),
                                        resolved_path: PathBuf::from(
                                            "kernels/multiply.signature_1.spv",
                                        ),
                                        words: multiply_spirv_words,
                                    }],
                                };
                            let multiply_resident_dispatch = mounted
                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                    &device,
                                    multiply_dispatch,
                                    &multiply_kernel_manifest,
                                )
                                .unwrap();
                            assert_eq!(multiply_resident_dispatch.workgroup_count_x(), 1);

                            device
                                .run_resident_kernel_dispatch(
                                    &multiply_resident_dispatch,
                                    &[0u8; 16],
                                )
                                .unwrap();

                            let gated_x_first_16 = vec![
                                0x69, 0xbd, 0xe0, 0x3d, 0x8b, 0xbb, 0xe6, 0x3c, 0x2f, 0x3e, 0xba,
                                0x3a, 0x80, 0x3d, 0x9a, 0x3d,
                            ];
                            assert_eq!(
                                multiply_bindings[2].buffer.read_bytes(16).unwrap(),
                                gated_x_first_16.clone()
                            );

                            if let Some(rolling_spirv_words) =
                                crate::vulkan_compute::compile_test_shader_words_from_source(
                                    "rolling_state_update_bf16_3x1024.comp",
                                )
                            {
                                let rolling_dispatch = mounted_bound
                                    .dispatch("layer_00", "temporal_memory_update")
                                    .unwrap();
                                assert_eq!(
                                    rolling_dispatch.reusable_family_id,
                                    "rolling_state_update"
                                );
                                let rolling_bindings = mounted
                                    .resident_kernel_buffer_bindings_for_bound_dispatch(
                                        rolling_dispatch,
                                    )
                                    .unwrap();
                                assert_eq!(
                                    rolling_bindings
                                        .iter()
                                        .map(|binding| binding.byte_len)
                                        .collect::<Vec<_>>(),
                                    vec![6_144, 6_144, 6_144, 6_144, 6_144, 6_144]
                                );
                                let zero_temporal_memory = vec![0u8; 6_144];
                                rolling_bindings[3]
                                    .buffer
                                    .write_bytes(&zero_temporal_memory)
                                    .unwrap();
                                rolling_bindings[4]
                                    .buffer
                                    .write_bytes(&zero_temporal_memory)
                                    .unwrap();
                                let rolling_family = mounted
                                    .placed_plan
                                    .reusable_kernel_plan
                                    .family(&rolling_dispatch.reusable_family_id)
                                    .unwrap();
                                let rolling_kernel_manifest =
                                    VulkanLoadedReusableKernelArtifactManifest {
                                        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                            .to_string(),
                                        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                                        total_word_count: rolling_spirv_words.len(),
                                        artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                            artifact: VulkanReusableKernelArtifact::from_family(
                                                rolling_family,
                                                "kernels/rolling_state_update.spv",
                                            ),
                                            resolved_path: PathBuf::from(
                                                "kernels/rolling_state_update.spv",
                                            ),
                                            words: rolling_spirv_words,
                                        }],
                                    };
                                let rolling_resident_dispatch = mounted
                                    .create_resident_kernel_dispatch_for_bound_dispatch(
                                        &device,
                                        rolling_dispatch,
                                        &rolling_kernel_manifest,
                                    )
                                    .unwrap();
                                assert_eq!(rolling_resident_dispatch.workgroup_count_x(), 1);

                                device
                                    .run_resident_kernel_dispatch(
                                        &rolling_resident_dispatch,
                                        &[0u8; 16],
                                    )
                                    .unwrap();

                                let temporal_window =
                                    rolling_bindings[2].buffer.read_bytes(6_144).unwrap();
                                assert!(
                                    temporal_window[..4_096].iter().all(|byte| *byte == 0),
                                    "first two temporal frames should be empty after a zero-state first tick"
                                );
                                assert_eq!(
                                    &temporal_window[4_096..4_112],
                                    gated_x_first_16.as_slice()
                                );
                                assert_eq!(
                                    rolling_bindings[4].buffer.read_bytes(6_144).unwrap(),
                                    temporal_window
                                );

                                mounted
                                    .parameter_buffers
                                    .load_parameter_from_tensor_index(
                                        &tensor_index,
                                        "model.layers.0.conv.conv.weight",
                                    )
                                    .unwrap();
                                if let Some(depthwise_spirv_words) =
                                    crate::vulkan_compute::compile_test_shader_words_from_source(
                                        "depthwise_conv1d_bf16_3x1024.comp",
                                    )
                                {
                                    let depthwise_dispatch = mounted_bound
                                        .dispatch("layer_00", "depthwise_temporal_conv")
                                        .unwrap();
                                    assert_eq!(
                                        depthwise_dispatch.reusable_family_id,
                                        "depthwise_conv1d"
                                    );
                                    let depthwise_bindings = mounted
                                        .resident_kernel_buffer_bindings_for_bound_dispatch(
                                            depthwise_dispatch,
                                        )
                                        .unwrap();
                                    assert_eq!(depthwise_bindings.len(), 4);
                                    assert_eq!(depthwise_bindings[0].byte_len, 6_144);
                                    assert!(depthwise_bindings[1].byte_len >= 2_048);
                                    assert_eq!(depthwise_bindings[2].byte_len, 6_144);
                                    assert_eq!(depthwise_bindings[3].byte_len, 6_144);
                                    let depthwise_family = mounted
                                        .placed_plan
                                        .reusable_kernel_plan
                                        .family(&depthwise_dispatch.reusable_family_id)
                                        .unwrap();
                                    let depthwise_kernel_manifest =
                                        VulkanLoadedReusableKernelArtifactManifest {
                                            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                .to_string(),
                                            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                .to_string(),
                                            total_word_count: depthwise_spirv_words.len(),
                                            artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                                artifact: VulkanReusableKernelArtifact::from_family(
                                                    depthwise_family,
                                                    "kernels/depthwise_conv1d.spv",
                                                ),
                                                resolved_path: PathBuf::from(
                                                    "kernels/depthwise_conv1d.spv",
                                                ),
                                                words: depthwise_spirv_words,
                                            }],
                                        };
                                    let depthwise_resident_dispatch = mounted
                                        .create_resident_kernel_dispatch_for_bound_dispatch(
                                            &device,
                                            depthwise_dispatch,
                                            &depthwise_kernel_manifest,
                                        )
                                        .unwrap();
                                    assert_eq!(depthwise_resident_dispatch.workgroup_count_x(), 1);

                                    device
                                        .run_resident_kernel_dispatch(
                                            &depthwise_resident_dispatch,
                                            &[0u8; 16],
                                        )
                                        .unwrap();

                                    assert_eq!(
                                        depthwise_bindings[1].buffer.read_bytes(16).unwrap(),
                                        vec![
                                            0x20, 0x3c, 0xb1, 0xba, 0x17, 0x38, 0x6b, 0x38, 0x5b,
                                            0xb9, 0x82, 0x37, 0x6c, 0xb8, 0x8a, 0xba,
                                        ]
                                    );

                                    if let Some(output_gate_spirv_words) =
                                        crate::vulkan_compute::compile_test_shader_words_from_source(
                                            "multiply_bf16_1024.comp",
                                        )
                                    {
                                        let output_gate_dispatch = mounted_bound
                                            .dispatch("layer_00", "output_gate")
                                            .unwrap();
                                        assert_eq!(output_gate_dispatch.op, "multiply");
                                        let output_gate_bindings = mounted
                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                output_gate_dispatch,
                                            )
                                            .unwrap();
                                        assert_eq!(output_gate_bindings.len(), 3);
                                        assert!(output_gate_bindings[0].byte_len >= 2_048);
                                        assert!(output_gate_bindings[1].byte_len >= 2_048);
                                        assert!(output_gate_bindings[2].byte_len >= 2_048);
                                        let output_gate_family = mounted
                                            .placed_plan
                                            .reusable_kernel_plan
                                            .family(&output_gate_dispatch.reusable_family_id)
                                            .unwrap();
                                        let output_gate_artifact_path = format!(
                                            "kernels/{}.spv",
                                            output_gate_dispatch.reusable_family_id
                                        );
                                        let output_gate_kernel_manifest =
                                            VulkanLoadedReusableKernelArtifactManifest {
                                                schema:
                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                        .to_string(),
                                                backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                    .to_string(),
                                                total_word_count: output_gate_spirv_words.len(),
                                                artifacts:
                                                    vec![VulkanLoadedReusableKernelArtifact {
                                                    artifact:
                                                        VulkanReusableKernelArtifact::from_family(
                                                            output_gate_family,
                                                            output_gate_artifact_path.clone(),
                                                        ),
                                                    resolved_path: PathBuf::from(
                                                        output_gate_artifact_path,
                                                    ),
                                                    words: output_gate_spirv_words,
                                                }],
                                            };
                                        let output_gate_resident_dispatch = mounted
                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                &device,
                                                output_gate_dispatch,
                                                &output_gate_kernel_manifest,
                                            )
                                            .unwrap();
                                        assert_eq!(
                                            output_gate_resident_dispatch.workgroup_count_x(),
                                            1
                                        );

                                        device
                                            .run_resident_kernel_dispatch(
                                                &output_gate_resident_dispatch,
                                                &[0u8; 16],
                                            )
                                            .unwrap();

                                        assert_eq!(
                                            output_gate_bindings[2].buffer.read_bytes(16).unwrap(),
                                            vec![
                                                0xa5, 0xbb, 0xc9, 0xb9, 0x38, 0x37, 0xc6, 0xb7,
                                                0x86, 0xb7, 0xe5, 0xb4, 0x79, 0xb6, 0x21, 0xba,
                                            ]
                                        );

                                        mounted
                                            .parameter_buffers
                                            .load_parameter_from_tensor_index(
                                                &tensor_index,
                                                "model.layers.0.conv.out_proj.weight",
                                            )
                                            .unwrap();
                                        if let Some(conv_out_projection_spirv_words) =
                                            crate::vulkan_compute::compile_test_shader_words_from_source(
                                                "linear_bf16_1024x1024.comp",
                                            )
                                        {
                                            let conv_out_projection_dispatch = mounted_bound
                                                .dispatch("layer_00", "conv_out_projection")
                                                .unwrap();
                                            assert_eq!(
                                                conv_out_projection_dispatch.op,
                                                "linear"
                                            );
                                            let conv_out_projection_bindings = mounted
                                                .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                    conv_out_projection_dispatch,
                                                )
                                                .unwrap();
                                            assert_eq!(conv_out_projection_bindings.len(), 3);
                                            assert!(
                                                conv_out_projection_bindings[0].byte_len >= 2_048
                                            );
                                            assert!(
                                                conv_out_projection_bindings[1].byte_len >= 2_048
                                            );
                                            assert_eq!(
                                                conv_out_projection_bindings[2].byte_len,
                                                2_097_152
                                            );
                                            let conv_out_projection_family = mounted
                                                .placed_plan
                                                .reusable_kernel_plan
                                                .family(
                                                    &conv_out_projection_dispatch
                                                        .reusable_family_id,
                                                )
                                                .unwrap();
                                            let conv_out_projection_artifact_path = format!(
                                                "kernels/{}.spv",
                                                conv_out_projection_dispatch.reusable_family_id
                                            );
                                            let conv_out_projection_kernel_manifest =
                                                VulkanLoadedReusableKernelArtifactManifest {
                                                    schema:
                                                        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                            .to_string(),
                                                    backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                        .to_string(),
                                                    total_word_count:
                                                        conv_out_projection_spirv_words.len(),
                                                    artifacts: vec![
                                                        VulkanLoadedReusableKernelArtifact {
                                                            artifact:
                                                                VulkanReusableKernelArtifact::from_family(
                                                                    conv_out_projection_family,
                                                                    conv_out_projection_artifact_path.clone(),
                                                                ),
                                                            resolved_path: PathBuf::from(
                                                                conv_out_projection_artifact_path,
                                                            ),
                                                            words: conv_out_projection_spirv_words,
                                                        },
                                                    ],
                                                };
                                            let conv_out_projection_resident_dispatch = mounted
                                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                                    &device,
                                                    conv_out_projection_dispatch,
                                                    &conv_out_projection_kernel_manifest,
                                                )
                                                .unwrap();
                                            assert!(
                                                conv_out_projection_resident_dispatch
                                                    .workgroup_count_x()
                                                    >= 8
                                            );

                                            device
                                                .run_resident_kernel_dispatch(
                                                    &conv_out_projection_resident_dispatch,
                                                    &[0u8; 16],
                                                )
                                                .unwrap();

                                            assert_eq!(
                                                conv_out_projection_bindings[1]
                                                    .buffer
                                                    .read_bytes(16)
                                                    .unwrap(),
                                                vec![
                                                    0x2f, 0xb9, 0xe4, 0xb9, 0xa3, 0xb9, 0x0c,
                                                    0xb9, 0x4d, 0xba, 0x82, 0xb9, 0xfd, 0x39,
                                                    0x26, 0x3a,
                                                ]
                                            );

                                            if let Some(residual_spirv_words) =
                                                crate::vulkan_compute::compile_test_shader_words_from_source(
                                                    "add_bf16_1024.comp",
                                                )
                                            {
                                                let residual_dispatch = mounted_bound
                                                    .dispatch("layer_00", "operator_residual")
                                                    .unwrap();
                                                assert_eq!(
                                                    residual_dispatch.op,
                                                    "residual_add"
                                                );
                                                let residual_bindings = mounted
                                                    .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                        residual_dispatch,
                                                    )
                                                    .unwrap();
                                                assert_eq!(residual_bindings.len(), 3);
                                                assert!(residual_bindings[0].byte_len >= 2_048);
                                                assert!(residual_bindings[1].byte_len >= 2_048);
                                                assert!(residual_bindings[2].byte_len >= 2_048);
                                                let residual_family = mounted
                                                    .placed_plan
                                                    .reusable_kernel_plan
                                                    .family(
                                                        &residual_dispatch.reusable_family_id,
                                                    )
                                                    .unwrap();
                                                let residual_artifact_path = format!(
                                                    "kernels/{}.spv",
                                                    residual_dispatch.reusable_family_id
                                                );
                                                let residual_kernel_manifest =
                                                    VulkanLoadedReusableKernelArtifactManifest {
                                                        schema:
                                                            VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                .to_string(),
                                                        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                            .to_string(),
                                                        total_word_count: residual_spirv_words.len(),
                                                        artifacts: vec![
                                                            VulkanLoadedReusableKernelArtifact {
                                                                artifact:
                                                                    VulkanReusableKernelArtifact::from_family(
                                                                        residual_family,
                                                                        residual_artifact_path.clone(),
                                                                    ),
                                                                resolved_path: PathBuf::from(
                                                                    residual_artifact_path,
                                                                ),
                                                                words: residual_spirv_words,
                                                            },
                                                        ],
                                                    };
                                                let residual_resident_dispatch = mounted
                                                    .create_resident_kernel_dispatch_for_bound_dispatch(
                                                        &device,
                                                        residual_dispatch,
                                                        &residual_kernel_manifest,
                                                    )
                                                    .unwrap();
                                                assert_eq!(
                                                    residual_resident_dispatch.workgroup_count_x(),
                                                    1
                                                );

                                                device
                                                    .run_resident_kernel_dispatch(
                                                        &residual_resident_dispatch,
                                                        &[0u8; 16],
                                                    )
                                                    .unwrap();

                                                let residual_output = residual_bindings[2]
                                                    .buffer
                                                    .read_bytes(2_048)
                                                    .unwrap();
                                                assert_eq!(
                                                    &residual_output[..16],
                                                    &[
                                                        0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f,
                                                    ]
                                                );
                                                assert_eq!(
                                                    &residual_output[588..604],
                                                    &[
                                                        0x7e, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f,
                                                    ]
                                                );

                                                mounted
                                                    .parameter_buffers
                                                    .load_parameter_from_tensor_index(
                                                        &tensor_index,
                                                        "model.layers.0.ffn_norm.weight",
                                                    )
                                                    .unwrap();
                                                if let Some(ffn_norm_spirv_words) =
                                                    crate::vulkan_compute::compile_test_shader_words_from_source(
                                                        "rms_norm_bf16_serial.comp",
                                                    )
                                                {
                                                    let ffn_norm_dispatch = mounted_bound
                                                        .dispatch("layer_00", "ffn_norm")
                                                        .unwrap();
                                                    assert_eq!(ffn_norm_dispatch.op, "rms_norm");
                                                    let ffn_norm_bindings = mounted
                                                        .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                            ffn_norm_dispatch,
                                                        )
                                                        .unwrap();
                                                    assert_eq!(ffn_norm_bindings.len(), 3);
                                                    assert!(ffn_norm_bindings[0].byte_len >= 2_048);
                                                    assert!(ffn_norm_bindings[1].byte_len >= 2_048);
                                                    assert_eq!(ffn_norm_bindings[2].byte_len, 2_048);
                                                    let ffn_norm_family = mounted
                                                        .placed_plan
                                                        .reusable_kernel_plan
                                                        .family(
                                                            &ffn_norm_dispatch.reusable_family_id,
                                                        )
                                                        .unwrap();
                                                    let ffn_norm_artifact_path = format!(
                                                        "kernels/{}.spv",
                                                        ffn_norm_dispatch.reusable_family_id
                                                    );
                                                    let ffn_norm_kernel_manifest =
                                                        VulkanLoadedReusableKernelArtifactManifest {
                                                            schema:
                                                                VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                    .to_string(),
                                                            backend_id:
                                                                VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                    .to_string(),
                                                            total_word_count:
                                                                ffn_norm_spirv_words.len(),
                                                            artifacts: vec![
                                                                VulkanLoadedReusableKernelArtifact {
                                                                    artifact:
                                                                        VulkanReusableKernelArtifact::from_family(
                                                                            ffn_norm_family,
                                                                            ffn_norm_artifact_path.clone(),
                                                                        ),
                                                                    resolved_path: PathBuf::from(
                                                                        ffn_norm_artifact_path,
                                                                    ),
                                                                    words: ffn_norm_spirv_words,
                                                                },
                                                            ],
                                                        };
                                                    let ffn_norm_resident_dispatch = mounted
                                                        .create_resident_kernel_dispatch_for_bound_dispatch(
                                                            &device,
                                                            ffn_norm_dispatch,
                                                            &ffn_norm_kernel_manifest,
                                                        )
                                                        .unwrap();
                                                    assert_eq!(
                                                        ffn_norm_resident_dispatch
                                                            .workgroup_count_x(),
                                                        1
                                                    );

                                                    device
                                                        .run_resident_kernel_dispatch(
                                                            &ffn_norm_resident_dispatch,
                                                            &[0u8; 16],
                                                        )
                                                        .unwrap();

                                                    assert_eq!(
                                                        ffn_norm_bindings[1]
                                                            .buffer
                                                            .read_bytes(16)
                                                            .unwrap(),
                                                        vec![
                                                            0x6b, 0x3e, 0x6e, 0x3e, 0x69, 0x3e,
                                                            0x6e, 0x3e, 0x78, 0x3e, 0x6e, 0x3e,
                                                            0x79, 0x3e, 0x99, 0x3e,
                                                        ]
                                                    );

                                                    if let Some(ffn_projection_spirv_words) =
                                                        crate::vulkan_compute::compile_test_shader_words_from_source(
                                                            "linear_bf16_1024x2560.comp",
                                                        )
                                                    {
                                                        mounted
                                                            .parameter_buffers
                                                            .load_parameter_from_tensor_index(
                                                                &tensor_index,
                                                                "model.layers.0.feed_forward.w1.weight",
                                                            )
                                                            .unwrap();
                                                        let ffn_gate_dispatch = mounted_bound
                                                            .dispatch(
                                                                "layer_00",
                                                                "ffn_gate_projection",
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_gate_dispatch.op, "linear");
                                                        let ffn_gate_bindings = mounted
                                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                ffn_gate_dispatch,
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_gate_bindings.len(), 3);
                                                        assert!(
                                                            ffn_gate_bindings[0].byte_len >= 2_048
                                                        );
                                                        assert_eq!(
                                                            ffn_gate_bindings[1].byte_len,
                                                            5_120
                                                        );
                                                        assert_eq!(
                                                            ffn_gate_bindings[2].byte_len,
                                                            5_242_880
                                                        );
                                                        let ffn_gate_family = mounted
                                                            .placed_plan
                                                            .reusable_kernel_plan
                                                            .family(
                                                                &ffn_gate_dispatch
                                                                    .reusable_family_id,
                                                            )
                                                            .unwrap();
                                                        let ffn_gate_artifact_path = format!(
                                                            "kernels/{}.spv",
                                                            ffn_gate_dispatch.reusable_family_id
                                                        );
                                                        let ffn_gate_kernel_manifest =
                                                            VulkanLoadedReusableKernelArtifactManifest {
                                                                schema:
                                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                        .to_string(),
                                                                backend_id:
                                                                    VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                        .to_string(),
                                                                total_word_count:
                                                                    ffn_projection_spirv_words
                                                                        .len(),
                                                                artifacts: vec![
                                                                    VulkanLoadedReusableKernelArtifact {
                                                                        artifact:
                                                                            VulkanReusableKernelArtifact::from_family(
                                                                                ffn_gate_family,
                                                                                ffn_gate_artifact_path.clone(),
                                                                            ),
                                                                        resolved_path:
                                                                            PathBuf::from(
                                                                                ffn_gate_artifact_path,
                                                                            ),
                                                                        words:
                                                                            ffn_projection_spirv_words
                                                                                .clone(),
                                                                    },
                                                                ],
                                                            };
                                                        let ffn_gate_resident_dispatch = mounted
                                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                &device,
                                                                ffn_gate_dispatch,
                                                                &ffn_gate_kernel_manifest,
                                                            )
                                                            .unwrap();
                                                        assert!(
                                                            ffn_gate_resident_dispatch
                                                                .workgroup_count_x()
                                                                >= 20
                                                        );

                                                        device
                                                            .run_resident_kernel_dispatch(
                                                                &ffn_gate_resident_dispatch,
                                                                &[0u8; 16],
                                                            )
                                                            .unwrap();

                                                        assert_eq!(
                                                            ffn_gate_bindings[1]
                                                                .buffer
                                                                .read_bytes(16)
                                                                .unwrap(),
                                                            vec![
                                                                0x0a, 0x3d, 0x16, 0x3e, 0xea,
                                                                0x3d, 0x7c, 0x3e, 0x88, 0x3e,
                                                                0x07, 0x3e, 0x4a, 0x3e, 0x38,
                                                                0x3d,
                                                            ]
                                                        );

                                                        mounted
                                                            .parameter_buffers
                                                            .load_parameter_from_tensor_index(
                                                                &tensor_index,
                                                                "model.layers.0.feed_forward.w3.weight",
                                                            )
                                                            .unwrap();
                                                        let ffn_up_dispatch = mounted_bound
                                                            .dispatch(
                                                                "layer_00",
                                                                "ffn_up_projection",
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_up_dispatch.op, "linear");
                                                        let ffn_up_bindings = mounted
                                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                ffn_up_dispatch,
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_up_bindings.len(), 3);
                                                        assert!(
                                                            ffn_up_bindings[0].byte_len >= 2_048
                                                        );
                                                        assert_eq!(
                                                            ffn_up_bindings[1].byte_len,
                                                            5_120
                                                        );
                                                        assert_eq!(
                                                            ffn_up_bindings[2].byte_len,
                                                            5_242_880
                                                        );
                                                        let ffn_up_family = mounted
                                                            .placed_plan
                                                            .reusable_kernel_plan
                                                            .family(
                                                                &ffn_up_dispatch
                                                                    .reusable_family_id,
                                                            )
                                                            .unwrap();
                                                        let ffn_up_artifact_path = format!(
                                                            "kernels/{}.spv",
                                                            ffn_up_dispatch.reusable_family_id
                                                        );
                                                        let ffn_up_kernel_manifest =
                                                            VulkanLoadedReusableKernelArtifactManifest {
                                                                schema:
                                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                        .to_string(),
                                                                backend_id:
                                                                    VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                        .to_string(),
                                                                total_word_count:
                                                                    ffn_projection_spirv_words
                                                                        .len(),
                                                                artifacts: vec![
                                                                    VulkanLoadedReusableKernelArtifact {
                                                                        artifact:
                                                                            VulkanReusableKernelArtifact::from_family(
                                                                                ffn_up_family,
                                                                                ffn_up_artifact_path.clone(),
                                                                            ),
                                                                        resolved_path:
                                                                            PathBuf::from(
                                                                                ffn_up_artifact_path,
                                                                            ),
                                                                        words:
                                                                            ffn_projection_spirv_words,
                                                                    },
                                                                ],
                                                            };
                                                        let ffn_up_resident_dispatch = mounted
                                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                &device,
                                                                ffn_up_dispatch,
                                                                &ffn_up_kernel_manifest,
                                                            )
                                                            .unwrap();
                                                        assert!(
                                                            ffn_up_resident_dispatch
                                                                .workgroup_count_x()
                                                                >= 20
                                                        );

                                                        device
                                                            .run_resident_kernel_dispatch(
                                                                &ffn_up_resident_dispatch,
                                                                &[0u8; 16],
                                                            )
                                                            .unwrap();

                                                        assert_eq!(
                                                            ffn_up_bindings[1]
                                                                .buffer
                                                                .read_bytes(16)
                                                                .unwrap(),
                                                            vec![
                                                                0x35, 0xbe, 0xe6, 0xbe, 0x5d,
                                                                0xbe, 0x1d, 0x3e, 0x2a, 0xbe,
                                                                0x8b, 0x3c, 0x5e, 0x3e, 0xb1,
                                                                0xbe,
                                                            ]
                                                        );

                                                        if let Some(silu_spirv_words) =
                                                            crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                "silu_bf16_2560.comp",
                                                            )
                                                        {
                                                            let silu_dispatch = mounted_bound
                                                                .dispatch(
                                                                    "layer_00",
                                                                    "ffn_gate_activation",
                                                                )
                                                                .unwrap();
                                                            assert_eq!(silu_dispatch.op, "silu");
                                                            let silu_bindings = mounted
                                                                .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                    silu_dispatch,
                                                                )
                                                                .unwrap();
                                                            assert_eq!(silu_bindings.len(), 2);
                                                            assert_eq!(
                                                                silu_bindings[0].byte_len,
                                                                5_120
                                                            );
                                                            assert_eq!(
                                                                silu_bindings[1].byte_len,
                                                                5_120
                                                            );
                                                            let silu_family = mounted
                                                                .placed_plan
                                                                .reusable_kernel_plan
                                                                .family(
                                                                    &silu_dispatch
                                                                        .reusable_family_id,
                                                                )
                                                                .unwrap();
                                                            let silu_artifact_path = format!(
                                                                "kernels/{}.spv",
                                                                silu_dispatch.reusable_family_id
                                                            );
                                                            let silu_kernel_manifest =
                                                                VulkanLoadedReusableKernelArtifactManifest {
                                                                    schema:
                                                                        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                            .to_string(),
                                                                    backend_id:
                                                                        VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                            .to_string(),
                                                                    total_word_count:
                                                                        silu_spirv_words.len(),
                                                                    artifacts: vec![
                                                                        VulkanLoadedReusableKernelArtifact {
                                                                            artifact:
                                                                                VulkanReusableKernelArtifact::from_family(
                                                                                    silu_family,
                                                                                    silu_artifact_path.clone(),
                                                                                ),
                                                                            resolved_path:
                                                                                PathBuf::from(
                                                                                    silu_artifact_path,
                                                                                ),
                                                                            words:
                                                                                silu_spirv_words,
                                                                        },
                                                                    ],
                                                                };
                                                            let silu_resident_dispatch = mounted
                                                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                    &device,
                                                                    silu_dispatch,
                                                                    &silu_kernel_manifest,
                                                                )
                                                                .unwrap();
                                                            assert_eq!(
                                                                silu_resident_dispatch
                                                                    .workgroup_count_x(),
                                                                1
                                                            );

                                                            device
                                                                .run_resident_kernel_dispatch(
                                                                    &silu_resident_dispatch,
                                                                    &[0u8; 16],
                                                                )
                                                                .unwrap();

                                                            assert_eq!(
                                                                silu_bindings[1]
                                                                    .buffer
                                                                    .read_bytes(16)
                                                                    .unwrap(),
                                                                vec![
                                                                    0x8c, 0x3c, 0xa1, 0x3d, 0x77,
                                                                    0x3d, 0x0d, 0x3e, 0x1a, 0x3e,
                                                                    0x90, 0x3d, 0xde, 0x3d, 0xbc,
                                                                    0x3c,
                                                                ]
                                                            );

                                                            if let Some(ffn_multiply_spirv_words) =
                                                                crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                    "multiply_bf16_2560.comp",
                                                                )
                                                            {
                                                                let ffn_multiply_dispatch =
                                                                    mounted_bound
                                                                        .dispatch(
                                                                            "layer_00",
                                                                            "ffn_gate_multiply",
                                                                        )
                                                                        .unwrap();
                                                                assert_eq!(
                                                                    ffn_multiply_dispatch.op,
                                                                    "multiply"
                                                                );
                                                                let ffn_multiply_bindings = mounted
                                                                    .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                        ffn_multiply_dispatch,
                                                                    )
                                                                    .unwrap();
                                                                assert_eq!(
                                                                    ffn_multiply_bindings.len(),
                                                                    3
                                                                );
                                                                assert_eq!(
                                                                    ffn_multiply_bindings[0]
                                                                        .byte_len,
                                                                    5_120
                                                                );
                                                                assert_eq!(
                                                                    ffn_multiply_bindings[1]
                                                                        .byte_len,
                                                                    5_120
                                                                );
                                                                assert_eq!(
                                                                    ffn_multiply_bindings[2]
                                                                        .byte_len,
                                                                    5_120
                                                                );
                                                                let ffn_multiply_family = mounted
                                                                    .placed_plan
                                                                    .reusable_kernel_plan
                                                                    .family(
                                                                        &ffn_multiply_dispatch
                                                                            .reusable_family_id,
                                                                    )
                                                                    .unwrap();
                                                                let ffn_multiply_artifact_path =
                                                                    format!(
                                                                        "kernels/{}.spv",
                                                                        ffn_multiply_dispatch
                                                                            .reusable_family_id
                                                                    );
                                                                let ffn_multiply_kernel_manifest =
                                                                    VulkanLoadedReusableKernelArtifactManifest {
                                                                        schema:
                                                                            VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                                .to_string(),
                                                                        backend_id:
                                                                            VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                                .to_string(),
                                                                        total_word_count:
                                                                            ffn_multiply_spirv_words
                                                                                .len(),
                                                                        artifacts: vec![
                                                                            VulkanLoadedReusableKernelArtifact {
                                                                                artifact:
                                                                                    VulkanReusableKernelArtifact::from_family(
                                                                                        ffn_multiply_family,
                                                                                        ffn_multiply_artifact_path.clone(),
                                                                                    ),
                                                                                resolved_path:
                                                                                    PathBuf::from(
                                                                                        ffn_multiply_artifact_path,
                                                                                    ),
                                                                                words:
                                                                                    ffn_multiply_spirv_words,
                                                                            },
                                                                        ],
                                                                    };
                                                                let ffn_multiply_resident_dispatch =
                                                                    mounted
                                                                        .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                            &device,
                                                                            ffn_multiply_dispatch,
                                                                            &ffn_multiply_kernel_manifest,
                                                                        )
                                                                        .unwrap();
                                                                assert_eq!(
                                                                    ffn_multiply_resident_dispatch
                                                                        .workgroup_count_x(),
                                                                    1
                                                                );

                                                                device
                                                                    .run_resident_kernel_dispatch(
                                                                        &ffn_multiply_resident_dispatch,
                                                                        &[0u8; 16],
                                                                    )
                                                                    .unwrap();

                                                                assert_eq!(
                                                                    ffn_multiply_bindings[2]
                                                                        .buffer
                                                                        .read_bytes(16)
                                                                        .unwrap(),
                                                                    vec![
                                                                        0x46, 0xbb, 0x11, 0xbd,
                                                                        0x55, 0xbc, 0xad, 0x3c,
                                                                        0xcd, 0xbc, 0x9c, 0x3a,
                                                                        0xc1, 0x3c, 0x02, 0xbc,
                                                                    ]
                                                                );

                                                                mounted
                                                                    .parameter_buffers
                                                                    .load_parameter_from_tensor_index(
                                                                        &tensor_index,
                                                                        "model.layers.0.feed_forward.w2.weight",
                                                                    )
                                                                    .unwrap();
                                                                if let Some(ffn_down_spirv_words) =
                                                                    crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                        "linear_bf16_2560x1024.comp",
                                                                    )
                                                                {
                                                                    let ffn_down_dispatch =
                                                                        mounted_bound
                                                                            .dispatch(
                                                                                "layer_00",
                                                                                "ffn_down_projection",
                                                                            )
                                                                            .unwrap();
                                                                    assert_eq!(
                                                                        ffn_down_dispatch.op,
                                                                        "linear"
                                                                    );
                                                                    let ffn_down_bindings = mounted
                                                                        .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                            ffn_down_dispatch,
                                                                        )
                                                                        .unwrap();
                                                                    assert_eq!(
                                                                        ffn_down_bindings.len(),
                                                                        3
                                                                    );
                                                                    assert_eq!(
                                                                        ffn_down_bindings[0]
                                                                            .byte_len,
                                                                        5_120
                                                                    );
                                                                    assert!(
                                                                        ffn_down_bindings[1]
                                                                            .byte_len
                                                                            >= 2_048
                                                                    );
                                                                    assert_eq!(
                                                                        ffn_down_bindings[2]
                                                                            .byte_len,
                                                                        5_242_880
                                                                    );
                                                                    let ffn_down_family = mounted
                                                                        .placed_plan
                                                                        .reusable_kernel_plan
                                                                        .family(
                                                                            &ffn_down_dispatch
                                                                                .reusable_family_id,
                                                                        )
                                                                        .unwrap();
                                                                    let ffn_down_artifact_path =
                                                                        format!(
                                                                            "kernels/{}.spv",
                                                                            ffn_down_dispatch
                                                                                .reusable_family_id
                                                                        );
                                                                    let ffn_down_kernel_manifest =
                                                                        VulkanLoadedReusableKernelArtifactManifest {
                                                                            schema:
                                                                                VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                                    .to_string(),
                                                                            backend_id:
                                                                                VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                                    .to_string(),
                                                                            total_word_count:
                                                                                ffn_down_spirv_words
                                                                                    .len(),
                                                                            artifacts: vec![
                                                                                VulkanLoadedReusableKernelArtifact {
                                                                                    artifact:
                                                                                        VulkanReusableKernelArtifact::from_family(
                                                                                            ffn_down_family,
                                                                                            ffn_down_artifact_path.clone(),
                                                                                        ),
                                                                                    resolved_path:
                                                                                        PathBuf::from(
                                                                                            ffn_down_artifact_path,
                                                                                        ),
                                                                                    words:
                                                                                        ffn_down_spirv_words,
                                                                                },
                                                                            ],
                                                                        };
                                                                    let ffn_down_resident_dispatch =
                                                                        mounted
                                                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                                &device,
                                                                                ffn_down_dispatch,
                                                                                &ffn_down_kernel_manifest,
                                                                            )
                                                                            .unwrap();
                                                                    assert!(
                                                                        ffn_down_resident_dispatch
                                                                            .workgroup_count_x()
                                                                            >= 8
                                                                    );

                                                                    device
                                                                        .run_resident_kernel_dispatch(
                                                                            &ffn_down_resident_dispatch,
                                                                            &[0u8; 16],
                                                                        )
                                                                        .unwrap();

                                                                    assert_eq!(
                                                                        ffn_down_bindings[1]
                                                                            .buffer
                                                                            .read_bytes(16)
                                                                            .unwrap(),
                                                                        vec![
                                                                            0x37, 0x3d, 0x80,
                                                                            0x3c, 0x06, 0x3c,
                                                                            0x1d, 0xbc, 0xc2,
                                                                            0x3c, 0xac, 0x3c,
                                                                            0xc2, 0x3c, 0xa2,
                                                                            0x3c,
                                                                        ]
                                                                    );

                                                                    if let Some(final_residual_spirv_words) =
                                                                        crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                            "add_bf16_1024.comp",
                                                                        )
                                                                    {
                                                                        let final_residual_dispatch =
                                                                            mounted_bound
                                                                                .dispatch(
                                                                                    "layer_00",
                                                                                    "ffn_residual",
                                                                                )
                                                                                .unwrap();
                                                                        assert_eq!(
                                                                            final_residual_dispatch.op,
                                                                            "residual_add"
                                                                        );
                                                                        let final_residual_bindings = mounted
                                                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                                final_residual_dispatch,
                                                                            )
                                                                            .unwrap();
                                                                        assert_eq!(
                                                                            final_residual_bindings
                                                                                .len(),
                                                                            3
                                                                        );
                                                                        assert!(
                                                                            final_residual_bindings[0]
                                                                                .byte_len
                                                                                >= 2_048
                                                                        );
                                                                        assert!(
                                                                            final_residual_bindings[1]
                                                                                .byte_len
                                                                                >= 2_048
                                                                        );
                                                                        assert!(
                                                                            final_residual_bindings[2]
                                                                                .byte_len
                                                                                >= 2_048
                                                                        );
                                                                        let final_residual_family =
                                                                            mounted
                                                                                .placed_plan
                                                                                .reusable_kernel_plan
                                                                                .family(
                                                                                    &final_residual_dispatch
                                                                                        .reusable_family_id,
                                                                                )
                                                                                .unwrap();
                                                                        let final_residual_artifact_path =
                                                                            format!(
                                                                                "kernels/{}.spv",
                                                                                final_residual_dispatch
                                                                                    .reusable_family_id
                                                                            );
                                                                        let final_residual_kernel_manifest =
                                                                            VulkanLoadedReusableKernelArtifactManifest {
                                                                                schema:
                                                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                                        .to_string(),
                                                                                backend_id:
                                                                                    VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                                        .to_string(),
                                                                                total_word_count:
                                                                                    final_residual_spirv_words
                                                                                        .len(),
                                                                                artifacts: vec![
                                                                                    VulkanLoadedReusableKernelArtifact {
                                                                                        artifact:
                                                                                            VulkanReusableKernelArtifact::from_family(
                                                                                                final_residual_family,
                                                                                                final_residual_artifact_path.clone(),
                                                                                            ),
                                                                                        resolved_path:
                                                                                            PathBuf::from(
                                                                                                final_residual_artifact_path,
                                                                                            ),
                                                                                        words:
                                                                                            final_residual_spirv_words,
                                                                                    },
                                                                                ],
                                                                            };
                                                                        let final_residual_resident_dispatch =
                                                                            mounted
                                                                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                                    &device,
                                                                                    final_residual_dispatch,
                                                                                    &final_residual_kernel_manifest,
                                                                                )
                                                                                .unwrap();
                                                                        assert_eq!(
                                                                            final_residual_resident_dispatch
                                                                                .workgroup_count_x(),
                                                                            1
                                                                        );

                                                                        device
                                                                            .run_resident_kernel_dispatch(
                                                                                &final_residual_resident_dispatch,
                                                                                &[0u8; 16],
                                                                            )
                                                                            .unwrap();

                                                                        assert_eq!(
                                                                            final_residual_bindings[2]
                                                                                .buffer
                                                                                .read_bytes(16)
                                                                                .unwrap(),
                                                                            vec![
                                                                                0x86, 0x3f,
                                                                                0x82, 0x3f,
                                                                                0x81, 0x3f,
                                                                                0x7e, 0x3f,
                                                                                0x83, 0x3f,
                                                                                0x83, 0x3f,
                                                                                0x83, 0x3f,
                                                                                0x83, 0x3f,
                                                                            ]
                                                                        );
                                                                    } else {
                                                                        eprintln!(
                                                                            "skipping BF16 final residual Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                                        );
                                                                    }
                                                                } else {
                                                                    eprintln!(
                                                                        "skipping BF16 FFN down projection Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                                    );
                                                                }
                                                            } else {
                                                                eprintln!(
                                                                    "skipping BF16 FFN multiply Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                                );
                                                            }
                                                        } else {
                                                            eprintln!(
                                                                "skipping BF16 SiLU Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                            );
                                                        }
                                                    } else {
                                                        eprintln!(
                                                            "skipping BF16 FFN projection Vulkan dispatches: no GLSL to SPIR-V compiler found"
                                                        );
                                                    }
                                                } else {
                                                    eprintln!(
                                                        "skipping BF16 FFN RMSNorm Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                    );
                                                }
                                            } else {
                                                eprintln!(
                                                    "skipping BF16 operator residual Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                );
                                            }
                                        } else {
                                            eprintln!(
                                                "skipping BF16 conv out projection Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                            );
                                        }
                                    } else {
                                        eprintln!(
                                            "skipping BF16 output gate Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                        );
                                    }
                                } else {
                                    eprintln!(
                                        "skipping BF16 depthwise conv Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                    );
                                }
                            } else {
                                eprintln!(
                                    "skipping BF16 rolling state Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                );
                            }
                        } else {
                            eprintln!(
                                "skipping BF16 multiply Vulkan dispatch: no GLSL to SPIR-V compiler found"
                            );
                        }
                    } else {
                        eprintln!(
                            "skipping BF16 split Vulkan dispatch: no GLSL to SPIR-V compiler found"
                        );
                    }
                } else {
                    eprintln!(
                        "skipping linear BF16 Vulkan dispatch: no GLSL to SPIR-V compiler found"
                    );
                }
            } else {
                eprintln!(
                    "skipping serial RMSNorm Vulkan dispatch: no GLSL to SPIR-V compiler found"
                );
            }
        }

        assert_eq!(
            mounted_bound
                .dispatch("layer_00", "operator_norm")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::ModelInput {
                signal_id: "input_frame".to_string(),
            }
        );
        assert_eq!(
            mounted_bound
                .dispatch("layer_00", "ffn_residual")
                .unwrap()
                .descriptors
                .last()
                .unwrap()
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer {
                cable: VulkanPlacedLocalCableBufferBinding {
                    buffer_index: 0,
                    cable: mounted
                        .cable_io
                        .local_cable_buffer(0)
                        .unwrap()
                        .cable
                        .clone(),
                    byte_capacity: 2_048,
                },
            }
        );
        assert_eq!(
            mounted_bound
                .dispatch("layer_01", "operator_norm")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer {
                cable: VulkanPlacedLocalCableBufferBinding {
                    buffer_index: 0,
                    cable: mounted
                        .cable_io
                        .local_cable_buffer(0)
                        .unwrap()
                        .cable
                        .clone(),
                    byte_capacity: 2_048,
                },
            }
        );
        assert_eq!(
            mounted_bound
                .dispatch("layer_01", "operator_residual")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer {
                cable: VulkanPlacedLocalCableBufferBinding {
                    buffer_index: 0,
                    cable: mounted
                        .cable_io
                        .local_cable_buffer(0)
                        .unwrap()
                        .cable
                        .clone(),
                    byte_capacity: 2_048,
                },
            }
        );
        assert_eq!(
            mounted_bound
                .dispatch("layer_13", "ffn_residual")
                .unwrap()
                .descriptors
                .last()
                .unwrap()
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::ModelOutput {
                signal_id: "output_frame".to_string(),
            }
        );
        let layer_01_norm_tick = match &tick_plan.stages[16] {
            VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } => dispatch,
            stage => panic!("expected layer_01 operator_norm dispatch, got {stage:?}"),
        };
        assert_eq!(layer_01_norm_tick.pedal_id, "layer_01");
        assert_eq!(layer_01_norm_tick.node_id, "operator_norm");
        assert_eq!(
            layer_01_norm_tick.reads,
            vec![VulkanMountedPlacedStreamTickIo::LocalCableBuffer {
                cable_index: 0,
                buffer_index: 0,
                byte_capacity: 2_048,
            }]
        );
    }

    #[test]
    fn mounted_placed_stream_circuit_binds_only_local_device_slice() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping placed Vulkan stream-circuit mount: {error}");
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
        let placement_spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "cpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_03", "lan:worker-a");
        let placement_plan = graph.placement_plan(&placement_spec).unwrap();
        let gpu1_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu1",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
        let gpu1_plan = VulkanPlacedStreamCircuitPlan::from_plans(
            &execution_plan,
            &resource_plan,
            gpu1_resident,
        )
        .unwrap();

        let mounted =
            VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, gpu1_plan, 4).unwrap();

        assert_eq!(mounted.device_id(), "gpu1");
        assert!(!mounted.can_execute());
        assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 1);
        assert_eq!(mounted.placed_plan.dispatch_plan.total_dispatch_count(), 19);
        assert_eq!(mounted.parameter_buffers.plan.device_id, "gpu1");
        assert_eq!(mounted.parameter_buffers.plan.parameter_count, 11);
        assert_eq!(
            mounted.parameter_buffers.plan.total_byte_capacity,
            mounted
                .placed_plan
                .placed_resident_plan
                .resident_plan
                .permanent_parameter_bytes
        );
        assert_eq!(
            Some(mounted.parameter_buffers.total_byte_capacity),
            mounted.parameter_buffers.plan.total_byte_capacity
        );
        assert!(mounted.parameter_buffers.plan.unresolved_tensors.is_empty());
        assert_eq!(mounted.boundary_io.plan.device_id, "gpu1");
        assert_eq!(mounted.boundary_io.plan.input_count, 0);
        assert_eq!(mounted.boundary_io.plan.output_count, 0);
        assert_eq!(mounted.boundary_io.plan.total_buffer_count, 0);
        assert_eq!(mounted.boundary_io.plan.total_byte_capacity, Some(0));
        assert_eq!(mounted.boundary_io.total_byte_capacity, 0);
        assert_eq!(mounted.buffers.state_buffers.len(), 1);
        assert_eq!(mounted.buffers.activation_slot_buffers.len(), 4);
        assert_eq!(mounted.buffers.total_byte_capacity, 25_600);
        assert_eq!(mounted.cable_io.plan.device_id, "gpu1");
        assert_eq!(mounted.cable_io.plan.total_endpoint_count, 2);
        assert_eq!(mounted.cable_io.plan.total_byte_capacity, Some(4_096));
        assert_eq!(mounted.cable_io.incoming_buffers.len(), 1);
        assert_eq!(mounted.cable_io.outgoing_buffers.len(), 1);
        assert_eq!(mounted.cable_io.total_byte_capacity, 4_096);
        let incoming_cable = mounted.cable_io.incoming_buffer(1).unwrap();
        assert_eq!(
            incoming_cable.endpoint.direction,
            VulkanPlacedCableDirection::Incoming
        );
        assert_eq!(incoming_cable.endpoint.local_pedal_id, "layer_02");
        assert_eq!(incoming_cable.endpoint.remote_pedal_id, "layer_01");
        assert_eq!(incoming_cable.byte_capacity, 2_048);
        assert_eq!(incoming_cable.buffer.byte_capacity(), 2_048);
        incoming_cable.buffer.write_bytes(&[7, 8, 9, 10]).unwrap();
        assert_eq!(
            incoming_cable.buffer.read_bytes(4).unwrap(),
            vec![7, 8, 9, 10]
        );
        let outgoing_cable = mounted.cable_io.outgoing_buffer(2).unwrap();
        assert_eq!(
            outgoing_cable.endpoint.direction,
            VulkanPlacedCableDirection::Outgoing
        );
        assert_eq!(outgoing_cable.endpoint.local_pedal_id, "layer_02");
        assert_eq!(outgoing_cable.endpoint.remote_pedal_id, "layer_03");
        assert_eq!(outgoing_cable.byte_capacity, 2_048);
        assert_eq!(outgoing_cable.buffer.byte_capacity(), 2_048);
        assert_eq!(
            mounted
                .buffers
                .state_buffer("layer_02", "kv_memory")
                .map(|buffer| buffer.byte_capacity),
            Some(8_192)
        );

        let descriptor_plan = mounted.descriptor_resource_plan().unwrap();
        assert_eq!(descriptor_plan.dispatches.len(), 19);
        assert!(
            descriptor_plan
                .dispatch("layer_00", "operator_norm")
                .is_none()
        );
        assert!(
            descriptor_plan
                .dispatch("layer_02", "kv_memory_append")
                .is_some()
        );

        let manifest = VulkanReusableKernelArtifactManifest::new(
            mounted
                .placed_plan
                .reusable_kernel_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );
        let prepared = mounted.prepared_dispatch_plan(&manifest).unwrap();
        assert_eq!(prepared.dispatches.len(), 19);
        assert_eq!(
            prepared
                .dispatch("layer_02", "kv_memory_append")
                .map(|dispatch| dispatch.artifact_path.as_str()),
            Some("kernels/append_state_update.spv")
        );
        let bound = mounted.bound_dispatch_plan(&manifest).unwrap();
        assert_eq!(bound.dispatches.len(), 19);
        assert_eq!(
            bound.total_descriptor_count,
            prepared.total_descriptor_count
        );
        assert!(bound.boundary_descriptor_count > 0);
        assert!(bound.permanent_parameter_descriptor_count > 0);
        assert!(bound.stream_state_descriptor_count > 0);
        assert!(bound.activation_slot_descriptor_count > 0);

        let placed_bound = mounted.placed_bound_dispatch_plan(&manifest).unwrap();
        assert_eq!(placed_bound.device_id, "gpu1");
        assert_eq!(placed_bound.dispatches.len(), 19);
        assert_eq!(
            placed_bound.total_descriptor_count,
            bound.total_descriptor_count
        );
        assert_eq!(placed_bound.model_boundary_descriptor_count, 0);
        assert_eq!(placed_bound.local_cable_descriptor_count, 0);
        assert_eq!(placed_bound.incoming_cable_descriptor_count, 2);
        assert_eq!(placed_bound.outgoing_cable_descriptor_count, 1);
        assert_eq!(
            placed_bound
                .dispatch("layer_02", "operator_norm")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanPlacedBoundDescriptorTarget::IncomingCable {
                cable: mounted.placed_plan.placed_resident_plan.incoming_cables[0].clone(),
            }
        );
        assert_eq!(
            placed_bound
                .dispatch("layer_02", "operator_residual")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanPlacedBoundDescriptorTarget::IncomingCable {
                cable: mounted.placed_plan.placed_resident_plan.incoming_cables[0].clone(),
            }
        );
        assert_eq!(
            placed_bound
                .dispatch("layer_02", "ffn_residual")
                .unwrap()
                .descriptors
                .last()
                .unwrap()
                .target,
            VulkanPlacedBoundDescriptorTarget::OutgoingCable {
                cable: mounted.placed_plan.placed_resident_plan.outgoing_cables[0].clone(),
            }
        );

        let mounted_bound = mounted
            .mounted_placed_bound_dispatch_plan(&manifest)
            .unwrap();
        assert_eq!(mounted_bound.device_id, "gpu1");
        assert_eq!(mounted_bound.dispatches.len(), 19);
        assert_eq!(
            mounted_bound.total_descriptor_count,
            placed_bound.total_descriptor_count
        );
        assert_eq!(
            mounted_bound.resident_descriptor_count,
            placed_bound.resident_descriptor_count
        );
        assert_eq!(mounted_bound.model_boundary_descriptor_count, 0);
        assert_eq!(mounted_bound.local_cable_descriptor_count, 0);
        assert_eq!(mounted_bound.cable_endpoint_descriptor_count, 3);
        assert_eq!(mounted_bound.incoming_cable_descriptor_count, 2);
        assert_eq!(mounted_bound.outgoing_cable_descriptor_count, 1);
        assert_eq!(
            mounted_bound
                .dispatch("layer_02", "operator_norm")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer {
                endpoint: VulkanPlacedCableEndpointBufferBinding {
                    buffer_index: 0,
                    endpoint: mounted
                        .cable_io
                        .incoming_buffer(1)
                        .unwrap()
                        .endpoint
                        .clone(),
                    byte_capacity: 2_048,
                },
            }
        );
        assert_eq!(
            mounted_bound
                .dispatch("layer_02", "operator_residual")
                .unwrap()
                .descriptors[0]
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer {
                endpoint: VulkanPlacedCableEndpointBufferBinding {
                    buffer_index: 0,
                    endpoint: mounted
                        .cable_io
                        .incoming_buffer(1)
                        .unwrap()
                        .endpoint
                        .clone(),
                    byte_capacity: 2_048,
                },
            }
        );
        assert_eq!(
            mounted_bound
                .dispatch("layer_02", "ffn_residual")
                .unwrap()
                .descriptors
                .last()
                .unwrap()
                .target,
            VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer {
                endpoint: VulkanPlacedCableEndpointBufferBinding {
                    buffer_index: 0,
                    endpoint: mounted
                        .cable_io
                        .outgoing_buffer(2)
                        .unwrap()
                        .endpoint
                        .clone(),
                    byte_capacity: 2_048,
                },
            }
        );

        let tick_plan = mounted.stream_tick_plan(&manifest).unwrap();
        assert_eq!(tick_plan.device_id, "gpu1");
        assert!(!tick_plan.can_execute);
        assert_eq!(tick_plan.stage_count, 21);
        assert_eq!(tick_plan.receive_stage_count, 1);
        assert_eq!(tick_plan.dispatch_stage_count, 19);
        assert_eq!(tick_plan.publish_stage_count, 1);
        assert_eq!(tick_plan.local_cable_read_count, 0);
        assert_eq!(tick_plan.local_cable_write_count, 0);
        assert_eq!(tick_plan.incoming_cable_read_count, 2);
        assert_eq!(tick_plan.outgoing_cable_write_count, 1);
        assert_eq!(tick_plan.model_input_read_count, 0);
        assert_eq!(tick_plan.model_output_write_count, 0);
        assert_eq!(
            tick_plan.stages[0],
            VulkanMountedPlacedStreamTickStage::ReceiveCable {
                stage_index: 0,
                cable_index: 1,
                endpoint_id: "cable_1_in".to_string(),
                buffer_index: 0,
                byte_capacity: 2_048,
                remote_device_id: "cpu0".to_string(),
                remote_pedal_id: "layer_01".to_string(),
            }
        );
        assert_eq!(
            tick_plan.stages[1],
            VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index: 1,
                dispatch: VulkanMountedPlacedStreamTickDispatch {
                    dispatch_index: 0,
                    kernel_id: "layer_02.operator_norm".to_string(),
                    pedal_id: "layer_02".to_string(),
                    node_id: "operator_norm".to_string(),
                    op: "rms_norm".to_string(),
                    descriptor_count: mounted_bound
                        .dispatch("layer_02", "operator_norm")
                        .unwrap()
                        .descriptors
                        .len(),
                    resident_descriptor_count: 2,
                    reads: vec![VulkanMountedPlacedStreamTickIo::IncomingCableBuffer {
                        cable_index: 1,
                        buffer_index: 0,
                        byte_capacity: 2_048,
                    }],
                    writes: vec![],
                },
            }
        );
        assert_eq!(
            tick_plan.stages[20],
            VulkanMountedPlacedStreamTickStage::PublishCable {
                stage_index: 20,
                cable_index: 2,
                endpoint_id: "cable_2_out".to_string(),
                buffer_index: 0,
                byte_capacity: 2_048,
                remote_device_id: "lan:worker-a".to_string(),
                remote_pedal_id: "layer_03".to_string(),
            }
        );
        let tick_run = mounted.advance_stream_tick(&manifest, 7).unwrap();
        assert_eq!(tick_run.device_id, "gpu1");
        assert_eq!(tick_run.stream_tick, 7);
        assert!(!tick_run.can_execute);
        assert_eq!(tick_run.planned_stage_count, 21);
        assert_eq!(tick_run.attempted_stage_count, 1);
        assert_eq!(tick_run.completed_stage_count, 0);
        assert_eq!(tick_run.pending_stage_count, 20);
        assert_eq!(
            tick_run.status,
            VulkanMountedPlacedStreamTickRunStatus::Blocked {
                stage_index: 0,
                reason: VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable,
            }
        );
        assert_eq!(tick_run.stages[0].stage, tick_plan.stages[0]);
        assert_eq!(
            tick_run.stages[0].status,
            VulkanMountedPlacedStreamTickStageStatus::Blocked {
                reason: VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable,
            }
        );
        assert_eq!(
            tick_run.stages[1].status,
            VulkanMountedPlacedStreamTickStageStatus::Pending
        );
        assert_eq!(
            tick_run.stages[20].status,
            VulkanMountedPlacedStreamTickStageStatus::Pending
        );

        let kv_append = bound.dispatch("layer_02", "kv_memory_append").unwrap();
        assert!(matches!(
            kv_append.descriptors[2].target,
            VulkanBoundDescriptorTarget::StreamStateBuffer {
                ref pedal_id,
                ref state_id,
                buffer_index: 0,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
        let mounted_kv_append = mounted_bound
            .dispatch("layer_02", "kv_memory_append")
            .unwrap();
        let kv_bindings = mounted
            .resident_kernel_buffer_bindings_for_bound_dispatch(mounted_kv_append)
            .unwrap();
        assert_eq!(kv_bindings.len(), mounted_kv_append.descriptors.len());
        assert_eq!(kv_bindings[0].binding, 0);
        assert_eq!(kv_bindings[0].byte_len, 2_048);
        assert_eq!(kv_bindings[2].binding, 2);
        assert_eq!(kv_bindings[2].byte_len, 8_192);

        if let Some(spirv_words) = crate::vulkan_compute::compile_test_shader_words() {
            let family = mounted
                .placed_plan
                .reusable_kernel_plan
                .family(&mounted_kv_append.reusable_family_id)
                .unwrap();
            let loaded_manifest = VulkanLoadedReusableKernelArtifactManifest {
                schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                total_word_count: spirv_words.len(),
                artifacts: vec![VulkanLoadedReusableKernelArtifact {
                    artifact: VulkanReusableKernelArtifact::from_family(
                        family,
                        "kernels/append_state_update.spv",
                    ),
                    resolved_path: PathBuf::from("kernels/append_state_update.spv"),
                    words: spirv_words,
                }],
            };
            let resident_dispatch = mounted
                .create_resident_kernel_dispatch_for_bound_dispatch(
                    &device,
                    mounted_kv_append,
                    &loaded_manifest,
                )
                .unwrap();

            assert_eq!(resident_dispatch.descriptor_count(), kv_bindings.len());
            assert_eq!(resident_dispatch.workgroup_count_x(), 1);
            assert_eq!(resident_dispatch.push_constant_byte_count(), 16);
        } else {
            eprintln!(
                "skipping resident kernel dispatch handle smoke: no GLSL to SPIR-V compiler found"
            );
        }
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
                shape: Some(vec![3072, 1024]),
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
    fn stream_control_push_constants_follow_kernel_abi_order() {
        let push_constants =
            VulkanKernelStreamMetadata::for_op("rotary_position_embedding").push_constants();
        let bytes = stream_control_push_constant_bytes(
            &push_constants,
            VulkanMountedPlacedStreamControl {
                stream_tick: 42,
                control_flags: 7,
                dynamic_state_capacity_activations: 4,
            },
        )
        .unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(&42u64.to_le_bytes());
        expected.extend_from_slice(&7u32.to_le_bytes());
        expected.extend_from_slice(&4u32.to_le_bytes());
        assert_eq!(bytes, expected);
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
    fn descriptor_resource_plan_resolves_lfm2_dispatch_patch_bay() {
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

        let descriptor_plan =
            VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();

        assert_eq!(descriptor_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(descriptor_plan.dynamic_state_capacity_activations, 4);
        assert_eq!(descriptor_plan.dispatches.len(), 242);
        assert_eq!(descriptor_plan.total_descriptor_count, 794);

        let first = descriptor_plan
            .dispatch("layer_00", "operator_norm")
            .unwrap();
        assert_eq!(first.dispatch_index, 0);
        assert_eq!(first.descriptors.len(), 3);
        assert_eq!(
            first.descriptors[0].resource,
            VulkanDescriptorResourceAddress::BoundaryInput {
                signal_id: "input_frame".to_string(),
            }
        );
        assert_eq!(
            first.descriptors[1].resource,
            VulkanDescriptorResourceAddress::ActivationSlot {
                pedal_id: "layer_00".to_string(),
                slot: 0,
                byte_capacity: 5120,
            }
        );
        assert_eq!(
            first.descriptors[2].resource,
            VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: "operator_norm".to_string(),
                tensor: "model.layers.0.operator_norm.weight".to_string(),
                byte_count: Some(2048),
            }
        );

        let kv_append = descriptor_plan
            .dispatch("layer_02", "kv_memory_append")
            .unwrap();
        assert_eq!(kv_append.descriptors.len(), 9);
        assert_eq!(
            kv_append.descriptors[2].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                state_type: "append_only_attention_memory".to_string(),
                byte_capacity: 8192,
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );
        assert_eq!(
            kv_append.descriptors[6].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                state_type: "append_only_attention_memory".to_string(),
                byte_capacity: 8192,
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );
        assert_eq!(
            kv_append.descriptors[7].resource,
            VulkanDescriptorResourceAddress::StateView {
                pedal_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                state_type: "append_only_attention_memory".to_string(),
                byte_capacity: 8192,
                static_bytes: None,
                bytes_per_activation: Some(2048),
            }
        );

        let last = descriptor_plan
            .dispatch("layer_13", "ffn_residual")
            .unwrap();
        assert_eq!(
            last.descriptors.last().unwrap().resource,
            VulkanDescriptorResourceAddress::BoundaryOutput {
                signal_id: "output_frame".to_string(),
            }
        );
    }

    #[test]
    fn descriptor_resource_plan_requires_dynamic_capacity_for_kv_state() {
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

        let error = VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 0)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("layer_02.kv_memory requires non-zero dynamic state capacity")
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
        assert_eq!(reusable_plan.total_family_count(), 26);
        assert_eq!(reusable_plan.reusable_family_count(), 26);
        assert_eq!(reusable_plan.families_for_op("rms_norm").len(), 4);
        assert_eq!(reusable_plan.families_for_op("linear").len(), 6);

        let linear = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
        assert_eq!(linear.op, "linear");
        assert_eq!(linear.command_refs.len(), 8);
        assert!(!linear.uses_stream_tick);
        assert_eq!(
            linear.descriptor_signature,
            vec![
                VulkanKernelDescriptorSlotSignature {
                    binding: 0,
                    usage: VulkanKernelDescriptorUsage::InputSignal,
                    resource_class: VulkanKernelDescriptorResourceClass::SignalBuffer,
                    byte_capacity: Some(5_120),
                    shape: None,
                },
                VulkanKernelDescriptorSlotSignature {
                    binding: 1,
                    usage: VulkanKernelDescriptorUsage::OutputSignal,
                    resource_class: VulkanKernelDescriptorResourceClass::SignalBuffer,
                    byte_capacity: Some(6_144),
                    shape: None,
                },
                VulkanKernelDescriptorSlotSignature {
                    binding: 2,
                    usage: VulkanKernelDescriptorUsage::Parameter,
                    resource_class: VulkanKernelDescriptorResourceClass::ParameterBuffer,
                    byte_capacity: Some(6_291_456),
                    shape: Some(vec![3072, 1024]),
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
            "layer_13.conv_in_projection"
        );

        let rope = reusable_plan.family("rotary_position_embedding").unwrap();
        assert_eq!(rope.command_refs.len(), 6);
        assert_eq!(
            reusable_plan
                .families_for_op("rotary_position_embedding")
                .iter()
                .map(|family| family.command_refs.len())
                .sum::<usize>(),
            12
        );
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
        let conv_in_family =
            reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
        let conv_in_family_id = conv_in_family.family_id.as_str();

        let empty = reusable_plan.coverage_report(std::iter::empty::<&str>());
        assert!(!empty.all_available());
        assert_eq!(empty.required_family_count, 26);
        assert_eq!(empty.available_family_count, 0);
        assert_eq!(empty.missing_family_count, 26);
        assert_eq!(empty.required_command_count, 242);
        assert_eq!(empty.covered_command_count, 0);
        assert_eq!(empty.missing_command_count, 242);
        assert!(
            empty
                .missing_families()
                .iter()
                .any(|family| family.family_id == conv_in_family_id && family.command_count == 8)
        );

        let partial_family_ids = [conv_in_family_id, "rms_norm.signature_1"];
        let partial_covered_command_count = partial_family_ids
            .iter()
            .map(|family_id| reusable_plan.family(family_id).unwrap().command_refs.len())
            .sum::<usize>();
        let partial = reusable_plan.coverage_report(partial_family_ids);
        assert!(!partial.all_available());
        assert_eq!(partial.available_family_count, 2);
        assert_eq!(partial.missing_family_count, 24);
        assert_eq!(partial.covered_command_count, partial_covered_command_count);
        assert_eq!(
            partial.missing_command_count,
            242 - partial_covered_command_count
        );
        assert_eq!(partial.missing_families().len(), 24);

        let full = reusable_plan.coverage_report(
            reusable_plan
                .families
                .iter()
                .map(|family| family.family_id.as_str()),
        );
        assert!(full.all_available());
        assert_eq!(full.available_family_count, 26);
        assert_eq!(full.missing_family_count, 0);
        assert_eq!(full.covered_command_count, 242);
        assert_eq!(full.missing_command_count, 0);
    }

    #[test]
    fn reusable_kernel_artifact_manifest_links_lfm2_kernel_families() {
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
        let conv_in_family =
            reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
        let conv_in_family_id = conv_in_family.family_id.as_str();
        let conv_in_artifact_path = artifact_path_for_family(conv_in_family);
        let manifest = VulkanReusableKernelArtifactManifest::new(
            reusable_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );

        let link_plan = reusable_plan.link_artifacts(&manifest);

        assert_eq!(
            manifest.schema,
            VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
        );
        assert_eq!(manifest.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(manifest.artifacts.len(), 26);
        assert!(link_plan.is_fully_linked());
        assert_eq!(link_plan.required_family_count, 26);
        assert_eq!(link_plan.linked_family_count, 26);
        assert_eq!(link_plan.missing_family_count, 0);
        assert_eq!(link_plan.incompatible_family_count, 0);
        assert_eq!(link_plan.required_command_count, 242);
        assert_eq!(link_plan.linked_command_count, 242);
        assert_eq!(link_plan.missing_command_count, 0);
        assert_eq!(link_plan.incompatible_command_count, 0);
        assert!(link_plan.issues.is_empty());

        let linear = link_plan.family(conv_in_family_id).unwrap();
        assert_eq!(linear.status, VulkanReusableKernelLinkStatus::Linked);
        assert_eq!(linear.command_count, 8);
        assert_eq!(
            linear.artifact_path.as_deref(),
            Some(conv_in_artifact_path.as_str())
        );

        let manifest_path = std::env::temp_dir().join(format!(
            "llmoop-reusable-kernel-manifest-{}.json",
            std::process::id()
        ));
        manifest.write_json_file(&manifest_path).unwrap();
        let read = VulkanReusableKernelArtifactManifest::from_json_file(&manifest_path).unwrap();
        std::fs::remove_file(&manifest_path).unwrap();
        assert_eq!(read, manifest);
        assert_eq!(read.family_ids().len(), 26);

        let artifact_root = std::env::temp_dir().join(format!(
            "llmoop-reusable-kernel-artifacts-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(artifact_root.join("kernels")).unwrap();
        for (index, artifact) in manifest.artifacts.iter().enumerate() {
            crate::vulkan::write_spirv_words(
                artifact_root.join(&artifact.path),
                &[0x0723_0203, index as u32],
            )
            .unwrap();
        }

        let loaded = manifest.load_artifacts(&artifact_root).unwrap();

        assert_eq!(
            loaded.schema,
            VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
        );
        assert_eq!(loaded.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(loaded.artifacts.len(), 26);
        assert_eq!(loaded.family_ids().len(), 26);
        assert_eq!(loaded.total_word_count, 52);
        let loaded_linear = loaded.artifact(conv_in_family_id).unwrap();
        assert_eq!(loaded_linear.artifact.family_id, conv_in_family_id);
        assert_eq!(
            loaded_linear.resolved_path,
            artifact_root.join(&conv_in_artifact_path)
        );
        assert_eq!(loaded_linear.words[0], 0x0723_0203);
        std::fs::remove_dir_all(&artifact_root).unwrap();
    }

    #[test]
    fn reusable_kernel_link_plan_reports_partial_and_incompatible_artifacts() {
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
        let linear = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
        let linear_family_id = linear.family_id.as_str();
        let linear_artifact_path = artifact_path_for_family(linear);

        let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
            VulkanReusableKernelArtifact::from_family(linear, linear_artifact_path),
        );
        let partial_link = reusable_plan.link_artifacts(&partial_manifest);

        assert!(!partial_link.is_fully_linked());
        assert_eq!(partial_link.linked_family_count, 1);
        assert_eq!(partial_link.missing_family_count, 25);
        assert_eq!(partial_link.incompatible_family_count, 0);
        assert_eq!(partial_link.linked_command_count, 8);
        assert_eq!(partial_link.missing_command_count, 242 - 8);
        assert_eq!(
            partial_link.family(linear_family_id).unwrap().status,
            VulkanReusableKernelLinkStatus::Linked
        );
        assert!(
            partial_link
                .missing_families()
                .iter()
                .any(|family| family.family_id == "append_state_update")
        );

        let mut bad_linear = VulkanReusableKernelArtifact::from_family(linear, "")
            .with_entry_point("not_main")
            .with_local_size_x(0);
        bad_linear.op = "multiply".to_string();
        bad_linear.descriptor_signature.pop();
        let incompatible_manifest =
            VulkanReusableKernelArtifactManifest::empty().with_artifact(bad_linear);
        let incompatible_link = reusable_plan.link_artifacts(&incompatible_manifest);

        assert!(!incompatible_link.is_fully_linked());
        assert_eq!(incompatible_link.linked_family_count, 0);
        assert_eq!(incompatible_link.missing_family_count, 25);
        assert_eq!(incompatible_link.incompatible_family_count, 1);
        assert_eq!(incompatible_link.incompatible_command_count, 8);
        assert_eq!(incompatible_link.missing_command_count, 242 - 8);
        let linear_link = incompatible_link.family(linear_family_id).unwrap();
        assert_eq!(
            linear_link.status,
            VulkanReusableKernelLinkStatus::Incompatible
        );
        assert!(linear_link.issues.iter().any(|issue| matches!(
            issue.problem,
            VulkanReusableKernelLinkProblem::OpMismatch { .. }
        )));
        assert!(linear_link.issues.iter().any(|issue| matches!(
            issue.problem,
            VulkanReusableKernelLinkProblem::DescriptorSignatureMismatch
        )));
        assert!(linear_link.issues.iter().any(|issue| matches!(
            issue.problem,
            VulkanReusableKernelLinkProblem::EmptySpirvPath
        )));
        assert!(linear_link.issues.iter().any(|issue| matches!(
            issue.problem,
            VulkanReusableKernelLinkProblem::UnsupportedEntryPoint { .. }
        )));
        assert!(linear_link.issues.iter().any(|issue| matches!(
            issue.problem,
            VulkanReusableKernelLinkProblem::InvalidLocalSizeX { .. }
        )));
        assert_eq!(incompatible_link.incompatible_families().len(), 1);
    }

    #[test]
    fn prepared_dispatch_plan_links_artifacts_to_descriptor_resources() {
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
        let conv_in_family =
            reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
        let conv_in_family_id = conv_in_family.family_id.as_str();
        let conv_in_artifact_path = artifact_path_for_family(conv_in_family);
        let descriptor_plan =
            VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
        let manifest = VulkanReusableKernelArtifactManifest::new(
            reusable_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );

        let prepared = VulkanPreparedDispatchPlan::from_plans(
            &dispatch_plan,
            &reusable_plan,
            &descriptor_plan,
            &manifest,
        )
        .unwrap();

        assert_eq!(prepared.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(prepared.reusable_family_count, 26);
        assert_eq!(prepared.dispatches.len(), 242);
        assert_eq!(prepared.total_descriptor_count, 794);

        let first = prepared.dispatch("layer_00", "operator_norm").unwrap();
        assert_eq!(first.dispatch_index, 0);
        assert_eq!(first.kernel_id, "layer_00.operator_norm");
        assert_eq!(first.reusable_family_id, "rms_norm.signature_1");
        assert_eq!(first.artifact_path, "kernels/rms_norm.signature_1.spv");
        assert_eq!(first.entry_point, DEFAULT_SPIRV_ENTRY_POINT);
        assert_eq!(first.local_size_x, DEFAULT_COMPUTE_LOCAL_SIZE_X);
        assert_eq!(first.descriptors.len(), 3);

        let linear = prepared.dispatch("layer_00", "conv_in_projection").unwrap();
        assert_eq!(linear.dispatch_index, 1);
        assert_eq!(linear.reusable_family_id, conv_in_family_id);
        assert_eq!(linear.artifact_path, conv_in_artifact_path);
        assert_eq!(linear.descriptors.len(), 3);

        let kv_append = prepared.dispatch("layer_02", "kv_memory_append").unwrap();
        assert_eq!(kv_append.dispatch_index, 40);
        assert_eq!(kv_append.reusable_family_id, "append_state_update");
        assert_eq!(kv_append.artifact_path, "kernels/append_state_update.spv");
        assert!(kv_append.uses_stream_tick);
        assert_eq!(kv_append.descriptors.len(), 9);
        assert!(matches!(
            kv_append.descriptors[2].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                ref pedal_id,
                ref state_id,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
        assert!(matches!(
            kv_append.descriptors[6].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                ref pedal_id,
                ref state_id,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
    }

    #[test]
    fn prepared_dispatch_plan_rejects_unlinked_reusable_kernels() {
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
        let descriptor_plan =
            VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
        let linear = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
        let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
            VulkanReusableKernelArtifact::from_family(linear, artifact_path_for_family(linear)),
        );

        let error = VulkanPreparedDispatchPlan::from_plans(
            &dispatch_plan,
            &reusable_plan,
            &descriptor_plan,
            &partial_manifest,
        )
        .unwrap_err();

        let VulkanPreparedDispatchPlanError::Link(link_plan) = error else {
            panic!("expected reusable kernel link failure");
        };
        assert_eq!(link_plan.linked_family_count, 1);
        assert_eq!(link_plan.missing_family_count, 25);
        assert_eq!(link_plan.linked_command_count, 8);
        assert_eq!(link_plan.missing_command_count, 242 - 8);
        assert!(
            link_plan
                .family("append_state_update")
                .is_some_and(|family| family.status == VulkanReusableKernelLinkStatus::Missing)
        );
    }

    #[test]
    fn bound_dispatch_plan_maps_prepared_descriptors_to_mounted_stream_buffers() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan stream-circuit binding: {error}");
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
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            &execution_plan,
            &resource_plan,
            &resident_plan,
        )
        .unwrap();
        let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
        let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
        let descriptor_plan =
            VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
        let manifest = VulkanReusableKernelArtifactManifest::new(
            reusable_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );
        let prepared = VulkanPreparedDispatchPlan::from_plans(
            &dispatch_plan,
            &reusable_plan,
            &descriptor_plan,
            &manifest,
        )
        .unwrap();
        let buffers = resident_plan.allocate_stream_buffers(&device, 4).unwrap();

        let bound = VulkanBoundDispatchPlan::from_prepared_plan(&prepared, &buffers).unwrap();

        assert_eq!(bound.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
        assert_eq!(bound.dispatches.len(), 242);
        assert_eq!(bound.total_descriptor_count, 794);
        assert_eq!(bound.boundary_descriptor_count, 42);
        assert_eq!(bound.permanent_parameter_descriptor_count, 130);
        assert_eq!(bound.stream_state_descriptor_count, 122);
        assert_eq!(bound.activation_slot_descriptor_count, 500);
        assert_eq!(
            bound.boundary_descriptor_count
                + bound.permanent_parameter_descriptor_count
                + bound.stream_state_descriptor_count
                + bound.activation_slot_descriptor_count,
            bound.total_descriptor_count
        );

        let first = bound.dispatch("layer_00", "operator_norm").unwrap();
        assert_eq!(first.dispatch_index, 0);
        assert_eq!(
            first.descriptors[0].target,
            VulkanBoundDescriptorTarget::BoundaryInput {
                signal_id: "input_frame".to_string(),
            }
        );
        assert_eq!(
            first.descriptors[1].target,
            VulkanBoundDescriptorTarget::ActivationSlot {
                buffer_index: buffers.activation_slot_buffer_index("layer_00", 0).unwrap(),
                pedal_id: "layer_00".to_string(),
                circuit_id: "layer_00_exact_lfm2_conv_circuit_v1".to_string(),
                slot: 0,
                byte_capacity: 5120,
            }
        );
        assert_eq!(
            first.descriptors[2].target,
            VulkanBoundDescriptorTarget::PermanentParameter {
                param_id: "operator_norm".to_string(),
                tensor: "model.layers.0.operator_norm.weight".to_string(),
                byte_count: Some(2048),
            }
        );

        let kv_append = bound.dispatch("layer_02", "kv_memory_append").unwrap();
        assert!(matches!(
            kv_append.descriptors[2].target,
            VulkanBoundDescriptorTarget::StreamStateBuffer {
                ref pedal_id,
                ref state_id,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
        assert!(matches!(
            kv_append.descriptors[6].target,
            VulkanBoundDescriptorTarget::StreamStateBuffer {
                ref pedal_id,
                ref state_id,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
        assert!(matches!(
            kv_append.descriptors[7].target,
            VulkanBoundDescriptorTarget::StreamStateView {
                ref pedal_id,
                ref state_id,
                byte_capacity: 8192,
                ..
            } if pedal_id == "layer_02" && state_id == "kv_memory"
        ));
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
        assert_eq!(mounted.reusable_kernel_plan.total_family_count(), 26);
        assert_eq!(mounted.reusable_kernel_plan.total_command_count, 242);
        let empty_coverage = mounted.reusable_kernel_coverage_report(std::iter::empty::<&str>());
        assert!(!empty_coverage.all_available());
        assert_eq!(empty_coverage.missing_family_count, 26);
        assert_eq!(empty_coverage.missing_command_count, 242);
        let empty_link =
            mounted.link_reusable_kernels(&VulkanReusableKernelArtifactManifest::empty());
        assert!(!empty_link.is_fully_linked());
        assert_eq!(empty_link.missing_family_count, 26);
        assert_eq!(empty_link.missing_command_count, 242);
        let descriptor_plan = mounted.descriptor_resource_plan().unwrap();
        assert_eq!(descriptor_plan.total_descriptor_count, 794);
        assert_eq!(descriptor_plan.dynamic_state_capacity_activations, 4);
        let manifest = VulkanReusableKernelArtifactManifest::new(
            mounted
                .reusable_kernel_plan
                .families
                .iter()
                .map(|family| {
                    VulkanReusableKernelArtifact::from_family(
                        family,
                        format!("kernels/{}.spv", family.family_id),
                    )
                })
                .collect(),
        );
        let prepared = mounted.prepared_dispatch_plan(&manifest).unwrap();
        assert_eq!(prepared.dispatches.len(), 242);
        assert_eq!(prepared.total_descriptor_count, 794);
        let bound = mounted.bound_dispatch_plan(&manifest).unwrap();
        assert_eq!(bound.dispatches.len(), 242);
        assert_eq!(bound.total_descriptor_count, 794);
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
