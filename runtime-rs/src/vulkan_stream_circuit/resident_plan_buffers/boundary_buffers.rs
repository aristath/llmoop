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
