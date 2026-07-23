pub struct VulkanResidentSingleTokenTickRunner {
    pub device_id: String,
    pub pedal_count: usize,
    pub dispatch_count: usize,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    pedalboard: VulkanMountedPlacedResidentPedalboardRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
    stream_control_buffer: Arc<VulkanResidentBuffer>,
    completed_pedalboard_run: Arc<VulkanMountedPlacedResidentPedalboardRun>,
    completed_output_run: Arc<VulkanResidentOutputTransducerRun>,
    sequence: VulkanResidentKernelSequence,
    feedback_sequence: VulkanResidentKernelSequence,
}

struct VulkanResidentSingleTokenTickExecution<'a> {
    input_token_is_resident: bool,
    emit_output: bool,
    input_tracking_dispatches: &'a [VulkanResidentKernelDispatch],
    tail_dispatches: &'a [VulkanResidentKernelDispatch],
}

impl VulkanResidentSingleTokenTickRunner {
    pub fn new(
        device: &VulkanComputeDevice,
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
        let sequence = device.create_resident_kernel_sequence()?;
        let feedback_sequence = device.create_resident_kernel_sequence()?;
        let stream_control_buffer = input_transducer.stream_control_buffer.clone();
        let completed_pedalboard_run = Arc::new(pedalboard.completed_sequence_run());
        let completed_output_run = Arc::new(output_transducer.completed_run());

        Ok(Self {
            device_id: pedalboard.device_id.clone(),
            pedal_count: pedalboard.pedal_count(),
            dispatch_count,
            total_descriptor_count,
            total_push_constant_byte_count,
            input_transducer,
            pedalboard,
            output_transducer,
            stream_control_buffer,
            completed_pedalboard_run,
            completed_output_run,
            sequence,
            feedback_sequence,
        })
    }

    pub fn run_token_id_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        token_id: u32,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<VulkanResidentSingleTokenTickRun, VulkanResidentSingleTokenTickRunnerError> {
        self.run_token_id_with_stream_control_and_tail(
            device,
            token_id,
            control,
            VulkanResidentSingleTokenTickExecution {
                input_token_is_resident: false,
                emit_output: true,
                input_tracking_dispatches: &[],
                tail_dispatches: &[],
            },
        )
    }

    fn run_token_id_with_stream_control_and_tail(
        &self,
        device: &VulkanComputeDevice,
        token_id: u32,
        control: VulkanMountedPlacedStreamControl,
        execution: VulkanResidentSingleTokenTickExecution<'_>,
    ) -> Result<VulkanResidentSingleTokenTickRun, VulkanResidentSingleTokenTickRunnerError> {
        if !execution.input_token_is_resident {
            self.stream_control_buffer
                .write_bytes(&stream_control_bytes(token_id, control))?;
        }
        let mut pedal_push_constants = Vec::with_capacity(self.pedalboard.dispatch_count());
        for pedal in &self.pedalboard.pedals {
            for dispatch in &pedal.dispatches {
                pedal_push_constants.push(stream_control_push_constant_bytes(
                    &dispatch.push_constants,
                    control,
                )?);
            }
        }

        let mut sequence_steps = Vec::with_capacity(
            self.dispatch_count
                + execution.input_tracking_dispatches.len()
                + execution.tail_dispatches.len(),
        );
        sequence_steps.push(VulkanResidentKernelSequenceStep::new(
            &self.input_transducer.resident_dispatch,
            &[],
        ));
        sequence_steps.extend(
            execution
                .input_tracking_dispatches
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        let mut pedal_push_constant_index = 0usize;
        for pedal in &self.pedalboard.pedals {
            for dispatch in &pedal.dispatches {
                sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                    &dispatch.resident_dispatch,
                    &pedal_push_constants[pedal_push_constant_index],
                ));
                pedal_push_constant_index += 1;
            }
        }
        if execution.emit_output {
            sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.embedding_norm_dispatch,
                &[],
            ));
            sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.tied_projection_dispatch,
                &[],
            ));
        }
        for tail_dispatch in execution.tail_dispatches {
            sequence_steps.push(VulkanResidentKernelSequenceStep::new(tail_dispatch, &[]));
        }

        let execution_start = Instant::now();
        device.run_resident_kernel_sequence(&self.sequence, &sequence_steps)?;
        let execution_time_ns =
            u64::try_from(execution_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let input_run = self.input_transducer.completed_run(token_id);
        let dispatch_count = if execution.emit_output {
            self.dispatch_count
        } else {
            self.dispatch_count - self.output_transducer.dispatch_count
        };
        let total_descriptor_count = if execution.emit_output {
            self.total_descriptor_count
        } else {
            self.total_descriptor_count - self.output_transducer.total_descriptor_count
        };
        let total_push_constant_byte_count = if execution.emit_output {
            self.total_push_constant_byte_count
        } else {
            self.total_push_constant_byte_count
                - self.output_transducer.total_push_constant_byte_count
        };
        Ok(VulkanResidentSingleTokenTickRun {
            device_id: self.device_id.clone(),
            token_id,
            input_run,
            pedalboard_run: self.completed_pedalboard_run.clone(),
            output_run: execution
                .emit_output
                .then(|| self.completed_output_run.clone()),
            dispatch_count,
            total_descriptor_count,
            total_push_constant_byte_count,
            execution_time_ns,
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
    pub pedalboard_run: Arc<VulkanMountedPlacedResidentPedalboardRun>,
    pub output_run: Option<Arc<VulkanResidentOutputTransducerRun>>,
    pub dispatch_count: usize,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    pub execution_time_ns: u64,
}

#[derive(Debug)]
pub enum VulkanResidentSingleTokenTickRunnerError {
    DispatchCountOverflow,
    DescriptorCountOverflow,
    PushConstantByteCountOverflow,
    Vulkan(VulkanError),
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
            Self::Vulkan(error) => Display::fmt(error, f),
            Self::InputTransducer(error) => Display::fmt(error, f),
            Self::Pedalboard(error) => Display::fmt(error, f),
            Self::OutputTransducer(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentSingleTokenTickRunnerError {}

impl From<VulkanError> for VulkanResidentSingleTokenTickRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

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

