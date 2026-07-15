use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::stream_circuit::{CableTransport, PedalCablePlacement, StreamCircuitPlacementPlan};
use crate::stream_plan::{
    CircuitActivationPlan, PlannedNode, SignalProducer, SignalStorage, StreamCircuitExecutionPlan,
    StreamCircuitResourcePlan, TensorIndex,
};
use crate::vulkan::{DEFAULT_COMPUTE_LOCAL_SIZE_X, DEFAULT_SPIRV_ENTRY_POINT};
use crate::vulkan_compute::{VulkanComputeDevice, VulkanError, VulkanResidentBuffer};

pub const VULKAN_STREAM_CIRCUIT_BACKEND_ID: &str = "vulkan_stream_circuit_ir";
pub const VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA: &str =
    "llmoop.vulkan_reusable_kernel_artifacts.v1";

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
    pub endpoints: Vec<VulkanPlacedCableEndpoint>,
    pub incoming_endpoint_count: usize,
    pub outgoing_endpoint_count: usize,
    pub total_endpoint_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_byte_cables: Vec<usize>,
}

impl VulkanPlacedCableIoPlan {
    pub fn from_placed_resident_plan(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
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

        let incoming_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedCableDirection::Incoming)
            .count();
        let outgoing_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedCableDirection::Outgoing)
            .count();
        let unresolved_byte_cables = endpoints
            .iter()
            .filter(|endpoint| endpoint.byte_capacity.is_none())
            .map(|endpoint| endpoint.cable_index)
            .collect::<Vec<_>>();
        let total_byte_capacity =
            endpoints.iter().try_fold(Some(0usize), |total, endpoint| {
                match (total, endpoint.byte_capacity) {
                    (Some(total), Some(bytes)) => Some(total.checked_add(bytes).ok_or_else(|| {
                        VulkanPlacedCableIoPlanError(
                            "placed cable endpoint byte capacity overflowed".to_string(),
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
        let mut incoming_buffers = Vec::with_capacity(self.incoming_endpoint_count);
        let mut outgoing_buffers = Vec::with_capacity(self.outgoing_endpoint_count);
        let mut total_byte_capacity = 0usize;

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
            incoming_buffers,
            outgoing_buffers,
            total_byte_capacity,
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
        let byte_capacity = match signal_element_bytes {
            Some(bytes_per_element) => Some(
                element_count
                    .checked_mul(bytes_per_element)
                    .ok_or_else(|| {
                        VulkanPlacedCableIoPlanError(format!(
                            "cable {} byte capacity overflowed",
                            cable.cable_index
                        ))
                    })?,
            ),
            None => None,
        };

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanPlacedCableDirection {
    Incoming,
    Outgoing,
}

pub struct VulkanPlacedCableIoBuffers {
    pub plan: VulkanPlacedCableIoPlan,
    pub incoming_buffers: Vec<VulkanPlacedCableBufferAllocation>,
    pub outgoing_buffers: Vec<VulkanPlacedCableBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPlacedCableIoBuffers {
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
    CableIo(VulkanPlacedCableIoPlanError),
    Vulkan(VulkanError),
}

impl Display for VulkanStreamCircuitMountError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(error) => Display::fmt(error, f),
            Self::CableIo(error) => Display::fmt(error, f),
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

impl From<VulkanPlacedCableIoPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanPlacedCableIoPlanError) -> Self {
        Self::CableIo(error)
    }
}

impl From<VulkanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

pub struct VulkanMountedPlacedStreamCircuit {
    pub placed_plan: VulkanPlacedStreamCircuitPlan,
    pub buffers: VulkanStreamCircuitStreamBuffers,
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
        let cable_io_plan =
            VulkanPlacedCableIoPlan::from_placed_resident_plan(&placed_plan.placed_resident_plan)?;
        let cable_io = cable_io_plan.allocate_buffers(device)?;
        Ok(Self {
            placed_plan,
            buffers,
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
                    VulkanMountedPlacedBoundDescriptorTarget::LocalCableInput { .. }
                    | VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutput { .. } => {
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
    LocalCableInput {
        cable: PedalCablePlacement,
    },
    LocalCableOutput {
        cable: PedalCablePlacement,
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
                Ok(Self::LocalCableInput {
                    cable: cable.clone(),
                })
            }
            VulkanPlacedBoundDescriptorTarget::LocalCableOutput { cable } => {
                Ok(Self::LocalCableOutput {
                    cable: cable.clone(),
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
pub struct VulkanPlacedCableEndpointBufferBinding {
    pub buffer_index: usize,
    pub endpoint: VulkanPlacedCableEndpoint,
    pub byte_capacity: usize,
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
    ByteCapacityMismatch {
        dispatch_index: usize,
        binding: usize,
        expected_byte_capacity: usize,
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
            Self::ByteCapacityMismatch {
                dispatch_index,
                binding,
                expected_byte_capacity,
                mounted_byte_capacity,
            } => write!(
                f,
                "dispatch {dispatch_index} descriptor {binding} expects {expected_byte_capacity} bytes but mounted buffer has {mounted_byte_capacity} bytes"
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
        assert_eq!(gpu0_cable_io.total_endpoint_count, 2);
        assert_eq!(gpu0_cable_io.incoming_endpoint_count, 1);
        assert_eq!(gpu0_cable_io.outgoing_endpoint_count, 1);
        assert_eq!(gpu0_cable_io.total_byte_capacity, Some(4_096));

        assert_eq!(gpu1.hosted_pedal_ids, vec!["layer_02".to_string()]);
        assert_eq!(gpu1.resident_plan.circuit_count, 1);
        assert_eq!(gpu1.resident_plan.permanent_parameters.len(), 11);
        assert_eq!(gpu1.resident_plan.stream_state_buffers.len(), 1);
        assert_eq!(gpu1.resident_plan.state_view_signal_count, 2);
        assert_eq!(gpu1.incoming_cables[0].source_pedal_id, "layer_01");
        assert_eq!(gpu1.outgoing_cables[0].destination_pedal_id, "layer_03");
        let gpu1_cable_io = VulkanPlacedCableIoPlan::from_placed_resident_plan(&gpu1).unwrap();
        assert_eq!(gpu1_cable_io.device_id, "gpu1");
        assert_eq!(gpu1_cable_io.total_endpoint_count, 2);
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
        assert_eq!(manifest.artifacts.len(), 12);
        assert!(link_plan.is_fully_linked());
        assert_eq!(link_plan.required_family_count, 12);
        assert_eq!(link_plan.linked_family_count, 12);
        assert_eq!(link_plan.missing_family_count, 0);
        assert_eq!(link_plan.incompatible_family_count, 0);
        assert_eq!(link_plan.required_command_count, 242);
        assert_eq!(link_plan.linked_command_count, 242);
        assert_eq!(link_plan.missing_command_count, 0);
        assert_eq!(link_plan.incompatible_command_count, 0);
        assert!(link_plan.issues.is_empty());

        let linear = link_plan.family("linear").unwrap();
        assert_eq!(linear.status, VulkanReusableKernelLinkStatus::Linked);
        assert_eq!(linear.command_count, 82);
        assert_eq!(linear.artifact_path.as_deref(), Some("kernels/linear.spv"));

        let manifest_path = std::env::temp_dir().join(format!(
            "llmoop-reusable-kernel-manifest-{}.json",
            std::process::id()
        ));
        manifest.write_json_file(&manifest_path).unwrap();
        let read = VulkanReusableKernelArtifactManifest::from_json_file(&manifest_path).unwrap();
        std::fs::remove_file(&manifest_path).unwrap();
        assert_eq!(read, manifest);
        assert_eq!(read.family_ids().len(), 12);
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
        let linear = reusable_plan.family("linear").unwrap();

        let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
            VulkanReusableKernelArtifact::from_family(linear, "kernels/linear.spv"),
        );
        let partial_link = reusable_plan.link_artifacts(&partial_manifest);

        assert!(!partial_link.is_fully_linked());
        assert_eq!(partial_link.linked_family_count, 1);
        assert_eq!(partial_link.missing_family_count, 11);
        assert_eq!(partial_link.incompatible_family_count, 0);
        assert_eq!(partial_link.linked_command_count, 82);
        assert_eq!(partial_link.missing_command_count, 242 - 82);
        assert_eq!(
            partial_link.family("linear").unwrap().status,
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
        assert_eq!(incompatible_link.missing_family_count, 11);
        assert_eq!(incompatible_link.incompatible_family_count, 1);
        assert_eq!(incompatible_link.incompatible_command_count, 82);
        assert_eq!(incompatible_link.missing_command_count, 242 - 82);
        let linear_link = incompatible_link.family("linear").unwrap();
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
        assert_eq!(prepared.reusable_family_count, 12);
        assert_eq!(prepared.dispatches.len(), 242);
        assert_eq!(prepared.total_descriptor_count, 794);

        let first = prepared.dispatch("layer_00", "operator_norm").unwrap();
        assert_eq!(first.dispatch_index, 0);
        assert_eq!(first.kernel_id, "layer_00.operator_norm");
        assert_eq!(first.reusable_family_id, "rms_norm");
        assert_eq!(first.artifact_path, "kernels/rms_norm.spv");
        assert_eq!(first.entry_point, DEFAULT_SPIRV_ENTRY_POINT);
        assert_eq!(first.local_size_x, DEFAULT_COMPUTE_LOCAL_SIZE_X);
        assert_eq!(first.descriptors.len(), 3);

        let linear = prepared.dispatch("layer_00", "conv_in_projection").unwrap();
        assert_eq!(linear.dispatch_index, 1);
        assert_eq!(linear.reusable_family_id, "linear");
        assert_eq!(linear.artifact_path, "kernels/linear.spv");
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
        let linear = reusable_plan.family("linear").unwrap();
        let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
            VulkanReusableKernelArtifact::from_family(linear, "kernels/linear.spv"),
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
        assert_eq!(link_plan.missing_family_count, 11);
        assert_eq!(link_plan.linked_command_count, 82);
        assert_eq!(link_plan.missing_command_count, 242 - 82);
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
        assert_eq!(mounted.reusable_kernel_plan.total_family_count(), 12);
        assert_eq!(mounted.reusable_kernel_plan.total_command_count, 242);
        let empty_coverage = mounted.reusable_kernel_coverage_report(std::iter::empty::<&str>());
        assert!(!empty_coverage.all_available());
        assert_eq!(empty_coverage.missing_family_count, 12);
        assert_eq!(empty_coverage.missing_command_count, 242);
        let empty_link =
            mounted.link_reusable_kernels(&VulkanReusableKernelArtifactManifest::empty());
        assert!(!empty_link.is_fully_linked());
        assert_eq!(empty_link.missing_family_count, 12);
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
