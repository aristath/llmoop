pub struct VulkanResidentInputEmbeddingTransducerRunner {
    pub transducer_id: String,
    pub parameter_tensor: String,
    pub output_signal_id: String,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
    stream_control_buffer: Arc<VulkanResidentBuffer>,
    resident_dispatch: VulkanResidentKernelDispatch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentInputEmbeddingTransducerSpec {
    pub transducer_id: String,
    pub parameter_tensor: String,
    pub parameter_dtype: String,
    pub parameter_shape: Vec<usize>,
    pub parameter_byte_capacity: usize,
    pub output_signal_id: String,
    pub output_frame_byte_capacity: usize,
    pub output_frame_word_count: usize,
    pub local_size_x: u32,
}

impl VulkanResidentInputEmbeddingTransducerRunner {
    pub fn from_mounted_token_embedding(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transducer_parameter_buffers: &VulkanPermanentParameterBuffers,
        spirv_words: &[u32],
        spec: &VulkanResidentInputEmbeddingTransducerSpec,
    ) -> Result<Self, VulkanResidentInputEmbeddingTransducerRunnerError> {
        let embedding_weight = transducer_parameter_buffers
            .parameter_buffer(&spec.parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                    tensor: spec.parameter_tensor.clone(),
                }
            })?;
        Self::from_mounted_token_embedding_with_parameter_allocation(
            device,
            mounted,
            embedding_weight,
            spirv_words,
            spec,
        )
    }

    fn from_mounted_token_embedding_with_parameter_allocation(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        embedding_weight: &VulkanPermanentParameterBufferAllocation,
        spirv_words: &[u32],
        spec: &VulkanResidentInputEmbeddingTransducerSpec,
    ) -> Result<Self, VulkanResidentInputEmbeddingTransducerRunnerError> {
        validate_input_embedding_weight(embedding_weight, spec)?;

        let output_frame = mounted
            .boundary_io
            .input_buffer(&spec.output_signal_id)
            .ok_or_else(|| {
                VulkanResidentInputEmbeddingTransducerRunnerError::MissingModelInputBuffer {
                    signal_id: spec.output_signal_id.clone(),
                }
            })?;
        if output_frame.byte_capacity != spec.output_frame_byte_capacity {
            return Err(
                VulkanResidentInputEmbeddingTransducerRunnerError::InvalidOutputFrameByteCapacity {
                    signal_id: spec.output_signal_id.clone(),
                    byte_capacity: output_frame.byte_capacity,
                    expected_byte_capacity: spec.output_frame_byte_capacity,
                },
            );
        }

        let bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                &embedding_weight.buffer,
                embedding_weight.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                1,
                &output_frame.buffer,
                output_frame.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Write),
            VulkanResidentKernelBufferBinding::new(
                2,
                &mounted.stream_control_buffer,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        ];
        let workgroup_count_x = u32::try_from(
            spec.output_frame_word_count
                .div_ceil(spec.local_size_x as usize),
        )
        .map_err(|_| VulkanResidentInputEmbeddingTransducerRunnerError::WorkgroupCountOverflow)?;
        let resident_dispatch = device.create_resident_kernel_dispatch(
            spirv_words,
            &bindings,
            workgroup_count_x,
            spec.local_size_x,
            0,
        )?;

        Ok(Self {
            transducer_id: spec.transducer_id.clone(),
            parameter_tensor: spec.parameter_tensor.clone(),
            output_signal_id: spec.output_signal_id.clone(),
            descriptor_count: resident_dispatch.descriptor_count(),
            workgroup_count_x: resident_dispatch.workgroup_count_x(),
            push_constant_byte_count: resident_dispatch.push_constant_byte_count(),
            stream_control_buffer: mounted.stream_control_buffer.clone(),
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
        let run = self.prepare_token_id(token_id)?;
        device.run_resident_kernel_dispatch(&self.resident_dispatch, &[])?;
        Ok(run)
    }

    fn prepare_token_id(
        &self,
        token_id: u32,
    ) -> Result<
        VulkanResidentInputEmbeddingTransducerRun,
        VulkanResidentInputEmbeddingTransducerRunnerError,
    > {
        self.prepare_token_id_only(token_id)?;
        Ok(self.completed_run(token_id))
    }

    fn prepare_token_id_only(
        &self,
        token_id: u32,
    ) -> Result<(), VulkanResidentInputEmbeddingTransducerRunnerError> {
        self.stream_control_buffer
            .write_bytes(&token_id.to_le_bytes())
            .map_err(VulkanResidentInputEmbeddingTransducerRunnerError::Vulkan)
    }

    fn completed_run(&self, token_id: u32) -> VulkanResidentInputEmbeddingTransducerRun {
        VulkanResidentInputEmbeddingTransducerRun {
            transducer_id: self.transducer_id.clone(),
            token_id,
            output_signal_id: self.output_signal_id.clone(),
            dispatch_count: 1,
            descriptor_count: self.descriptor_count,
            workgroup_count_x: self.workgroup_count_x,
            push_constant_byte_count: self.push_constant_byte_count,
        }
    }
}

struct VulkanResidentBatchedInputEmbeddingRunner {
    batch_capacity: usize,
    token_ids_buffer: VulkanResidentBuffer,
    dispatch: VulkanResidentKernelDispatch,
    sequence: VulkanResidentKernelSequence,
}

impl VulkanResidentBatchedInputEmbeddingRunner {
    fn new(
        device: &VulkanComputeDevice,
        batch_capacity: usize,
        embedding_weight: &VulkanPermanentParameterBufferAllocation,
        output_frames: &VulkanResidentBuffer,
        spirv_words: &[u32],
        spec: &VulkanResidentInputEmbeddingTransducerSpec,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if batch_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        validate_input_embedding_weight(embedding_weight, spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let output_byte_capacity = spec
            .output_frame_byte_capacity
            .checked_mul(batch_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched input embedding output capacity overflowed".to_string(),
                ))
            })?;
        if output_frames.byte_capacity() < output_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched input embedding output has {} bytes, requires {output_byte_capacity}",
                    output_frames.byte_capacity()
                )),
            ));
        }
        let token_byte_capacity = batch_capacity
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched input embedding token capacity overflowed".to_string(),
                ))
            })?;
        let mut token_ids_buffer = device
            .create_host_visible_resident_buffer(token_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        token_ids_buffer
            .persistently_map()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                &embedding_weight.buffer,
                embedding_weight.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(1, output_frames, output_byte_capacity)
                .with_access(VulkanResidentKernelBufferAccess::Write),
            VulkanResidentKernelBufferBinding::new(2, &token_ids_buffer, token_byte_capacity)
                .with_access(VulkanResidentKernelBufferAccess::Read),
        ];
        let workgroup_count_x = u32::try_from(
            spec.output_frame_word_count
                .div_ceil(spec.local_size_x as usize),
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                VulkanResidentInputEmbeddingTransducerRunnerError::WorkgroupCountOverflow,
            )
        })?;
        let workgroup_count_y = u32::try_from(batch_capacity).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched input embedding width exceeds u32".to_string(),
            ))
        })?;
        let dispatch = device
            .create_resident_kernel_dispatch_2d(
                spirv_words,
                &bindings,
                workgroup_count_x,
                workgroup_count_y,
                spec.local_size_x,
                std::mem::size_of::<u32>() as u32,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(Self {
            batch_capacity,
            token_ids_buffer,
            dispatch,
            sequence: device
                .create_resident_kernel_sequence()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
        })
    }

    fn run(
        &self,
        device: &VulkanComputeDevice,
        token_ids: &[u32],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if token_ids.is_empty() || token_ids.len() > self.batch_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched input embedding capacity {} cannot process {} tokens",
                    self.batch_capacity,
                    token_ids.len()
                )),
            ));
        }
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(token_ids));
        for token_id in token_ids {
            bytes.extend_from_slice(&token_id.to_le_bytes());
        }
        self.token_ids_buffer
            .write_bytes(&bytes)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let batch_width = u32::try_from(token_ids.len()).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched input embedding width exceeds u32".to_string(),
            ))
        })?;
        device
            .run_resident_kernel_sequence(
                &self.sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    &self.dispatch,
                    &batch_width.to_le_bytes(),
                )],
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
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

fn validate_input_embedding_weight(
    allocation: &VulkanPermanentParameterBufferAllocation,
    spec: &VulkanResidentInputEmbeddingTransducerSpec,
) -> Result<(), VulkanResidentInputEmbeddingTransducerRunnerError> {
    if allocation.parameter.dtype.as_deref() != Some(spec.parameter_dtype.as_str())
        || allocation.parameter.shape.as_deref() != Some(spec.parameter_shape.as_slice())
        || allocation.byte_capacity != spec.parameter_byte_capacity
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

