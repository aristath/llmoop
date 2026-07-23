pub struct VulkanResidentFeedbackLoopRunner {
    pub device_id: String,
    pub pedal_count: usize,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
    tick_runner: VulkanResidentSingleTokenTickRunner,
    sampler: VulkanResidentSamplerRunner,
}

impl VulkanResidentFeedbackLoopRunner {
    pub fn new(
        tick_runner: VulkanResidentSingleTokenTickRunner,
        sampler: VulkanResidentSamplerRunner,
    ) -> Result<Self, VulkanResidentFeedbackLoopRunnerError> {
        let per_tick_dispatch_count = tick_runner
            .dispatch_count
            .checked_add(sampler.dispatch_count)
            .ok_or(VulkanResidentFeedbackLoopRunnerError::DispatchCountOverflow)?;
        let per_tick_descriptor_count = tick_runner
            .total_descriptor_count
            .checked_add(sampler.descriptor_count)
            .ok_or(VulkanResidentFeedbackLoopRunnerError::DescriptorCountOverflow)?;
        let per_tick_push_constant_byte_count = tick_runner
            .total_push_constant_byte_count
            .checked_add(sampler.push_constant_byte_count)
            .ok_or(VulkanResidentFeedbackLoopRunnerError::PushConstantByteCountOverflow)?;

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
    ) -> Result<VulkanResidentFeedbackLoopRun, VulkanResidentFeedbackLoopRunnerError> {
        if max_ticks == 0 {
            return Err(VulkanResidentFeedbackLoopRunnerError::ZeroTickBudget);
        }

        let mut input_token_id = initial_token_id;
        let mut tick_runs = Vec::with_capacity(max_ticks);
        let mut sampled_token_ids = Vec::with_capacity(max_ticks);
        let sampler_dispatches = self.sampler.resident_dispatches();

        for tick_index in 0..max_ticks {
            let stream_tick = start_stream_tick
                .checked_add(
                    u64::try_from(tick_index)
                        .map_err(|_| VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?,
                )
                .ok_or(VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?;
            let tick_run = self.tick_runner.run_token_id_with_stream_control_and_tail(
                device,
                input_token_id,
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
                VulkanResidentSingleTokenTickExecution {
                    input_token_is_resident: tick_index != 0,
                    emit_output: true,
                    input_tracking_dispatches: self.sampler.input_tracking_dispatches(),
                    tail_dispatches: sampler_dispatches,
                },
            )?;
            let sampler_run = self.sampler.completed_run()?;
            let sampled_token_id = sampler_run.token_id;
            sampled_token_ids.push(sampled_token_id);
            tick_runs.push(VulkanResidentFeedbackTickRun {
                stream_tick,
                input_token_id,
                sampled_token_id,
                tick_run,
                sampler_run,
            });
            input_token_id = sampled_token_id;
        }

        Ok(VulkanResidentFeedbackLoopRun {
            device_id: self.device_id.clone(),
            initial_token_id,
            sampled_token_ids,
            tick_runs,
            per_tick_dispatch_count: self.per_tick_dispatch_count,
            per_tick_descriptor_count: self.per_tick_descriptor_count,
            per_tick_push_constant_byte_count: self.per_tick_push_constant_byte_count,
        })
    }

    fn run_resident_feedback_cycle(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        tick_count: usize,
        buffers: &VulkanStreamCircuitStreamBuffers,
        snapshots: &VulkanResidentStateTransactionBank,
    ) -> Result<VulkanResidentFeedbackLoopRun, VulkanResidentFeedbackLoopRunnerError> {
        if tick_count == 0 {
            return Err(VulkanResidentFeedbackLoopRunnerError::ZeroTickBudget);
        }
        if tick_count > snapshots.cycle_width
            || tick_count > self.sampler.history_capacity_activations
        {
            return Err(
                VulkanResidentFeedbackLoopRunnerError::FeedbackCycleWidthExceeded {
                    requested: tick_count,
                    snapshot_capacity: snapshots.cycle_width,
                    sampler_history_capacity: self.sampler.history_capacity_activations,
                },
            );
        }
        if self.per_tick_push_constant_byte_count != 0 {
            return Err(
                VulkanResidentFeedbackLoopRunnerError::FeedbackCyclePushConstants {
                    byte_count: self.per_tick_push_constant_byte_count,
                },
            );
        }

        let steps_per_tick = self.per_tick_dispatch_count;
        let total_steps = steps_per_tick
            .checked_mul(tick_count)
            .ok_or(VulkanResidentFeedbackLoopRunnerError::DispatchCountOverflow)?;
        let mut sequence_steps = Vec::with_capacity(total_steps);
        for _ in 0..tick_count {
            sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                &self.tick_runner.input_transducer.resident_dispatch,
                &[],
            ));
            sequence_steps.extend(
                self.sampler
                    .input_tracking_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
            for pedal in &self.tick_runner.pedalboard.pedals {
                for dispatch in &pedal.dispatches {
                    sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                        &dispatch.resident_dispatch,
                        &[],
                    ));
                }
            }
            sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                &self.tick_runner.output_transducer.embedding_norm_dispatch,
                &[],
            ));
            sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                &self.tick_runner.output_transducer.tied_projection_dispatch,
                &[],
            ));
            sequence_steps.extend(
                self.sampler
                    .resident_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
        }
        debug_assert_eq!(sequence_steps.len(), total_steps);
        let snapshot_copies = snapshots.copies_for_cycle(buffers, steps_per_tick, tick_count)?;

        let execution_start = Instant::now();
        device.run_resident_kernel_sequence_with_snapshot_copies(
            &self.tick_runner.feedback_sequence,
            &sequence_steps,
            &snapshot_copies,
        )?;
        let execution_time_ns =
            u64::try_from(execution_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let per_tick_execution_time_ns = execution_time_ns / tick_count as u64;

        let mut input_token_id = initial_token_id;
        let mut sampled_token_ids = Vec::with_capacity(tick_count);
        let mut tick_runs = Vec::with_capacity(tick_count);
        for tick_index in 0..tick_count {
            let stream_tick = start_stream_tick
                .checked_add(
                    u64::try_from(tick_index)
                        .map_err(|_| VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?,
                )
                .ok_or(VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?;
            let sampler_run = self.sampler.completed_run_at(stream_tick)?;
            let sampled_token_id = sampler_run.token_id;
            let tick_run = VulkanResidentSingleTokenTickRun {
                device_id: self.tick_runner.device_id.clone(),
                token_id: input_token_id,
                input_run: self
                    .tick_runner
                    .input_transducer
                    .completed_run(input_token_id),
                pedalboard_run: self.tick_runner.completed_pedalboard_run.clone(),
                output_run: Some(self.tick_runner.completed_output_run.clone()),
                dispatch_count: self.tick_runner.dispatch_count,
                total_descriptor_count: self.tick_runner.total_descriptor_count,
                total_push_constant_byte_count: self.tick_runner.total_push_constant_byte_count,
                execution_time_ns: per_tick_execution_time_ns,
            };
            sampled_token_ids.push(sampled_token_id);
            tick_runs.push(VulkanResidentFeedbackTickRun {
                stream_tick,
                input_token_id,
                sampled_token_id,
                tick_run,
                sampler_run,
            });
            input_token_id = sampled_token_id;
        }

        Ok(VulkanResidentFeedbackLoopRun {
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
    ) -> Result<VulkanResidentPromptEventRun, VulkanResidentFeedbackLoopRunnerError> {
        if prompt_token_ids.is_empty() {
            return Err(VulkanResidentFeedbackLoopRunnerError::EmptyPromptEvent);
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
                        VulkanResidentPromptEventInputRoute::ExternalInput,
                        0,
                        false,
                    )
                } else {
                    let feedback = pending_feedback
                        .take()
                        .ok_or(VulkanResidentFeedbackLoopRunnerError::MissingPrivateFeedback)?;
                    (
                        feedback.token_id,
                        VulkanResidentPromptEventInputRoute::PrivateFeedback,
                        feedback.feedback_depth,
                        feedback.closes_loop_after_processing,
                    )
                };

            let stream_tick = start_stream_tick
                .checked_add(
                    u64::try_from(tick_runs.len())
                        .map_err(|_| VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?,
                )
                .ok_or(VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?;
            let external_inputs_remaining = prompt_token_ids.len() - external_input_index;
            let should_emit_public_output =
                remaining_public_outputs > 0 && external_inputs_remaining == 0;
            let tail_dispatches = if should_emit_public_output {
                self.sampler.resident_dispatches()
            } else {
                &[]
            };
            let tick_run = self.tick_runner.run_token_id_with_stream_control_and_tail(
                device,
                input_token_id,
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
                VulkanResidentSingleTokenTickExecution {
                    input_token_is_resident: matches!(
                        input_route,
                        VulkanResidentPromptEventInputRoute::PrivateFeedback
                    ),
                    emit_output: should_emit_public_output,
                    input_tracking_dispatches: self.sampler.input_tracking_dispatches(),
                    tail_dispatches,
                },
            )?;

            let mut public_output_token_id = None;
            let mut private_feedback_token_id = None;
            let mut private_feedback_closes_loop_after_processing = None;
            let mut sampler_run = None;

            if should_emit_public_output {
                let run = self.sampler.completed_run()?;
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
                    feedback_depth: input_feedback_depth
                        .checked_add(1)
                        .ok_or(VulkanResidentFeedbackLoopRunnerError::FeedbackDepthOverflow)?,
                    closes_loop_after_processing: close_after_feedback,
                });
                sampler_run = Some(run);
            }

            tick_runs.push(VulkanResidentPromptEventTickRun {
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

        Ok(VulkanResidentPromptEventRun {
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
pub struct VulkanResidentFeedbackLoopRun {
    pub device_id: String,
    pub initial_token_id: u32,
    pub sampled_token_ids: Vec<u32>,
    pub tick_runs: Vec<VulkanResidentFeedbackTickRun>,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentFeedbackTickRun {
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub sampled_token_id: u32,
    pub tick_run: VulkanResidentSingleTokenTickRun,
    pub sampler_run: VulkanResidentSamplerRun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentPromptEventInputRoute {
    ExternalInput,
    PrivateFeedback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentPromptEventRun {
    pub device_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub stop_reason: String,
    pub tick_runs: Vec<VulkanResidentPromptEventTickRun>,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentPromptEventTickRun {
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub input_route: VulkanResidentPromptEventInputRoute,
    pub input_feedback_depth: u32,
    pub input_closes_loop_after_processing: bool,
    pub public_output_token_id: Option<u32>,
    pub private_feedback_token_id: Option<u32>,
    pub private_feedback_closes_loop_after_processing: Option<bool>,
    pub tick_run: VulkanResidentSingleTokenTickRun,
    pub sampler_run: Option<VulkanResidentSamplerRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentPendingPrivateFeedback {
    token_id: u32,
    feedback_depth: u32,
    closes_loop_after_processing: bool,
}

#[derive(Debug)]
pub enum VulkanResidentFeedbackLoopRunnerError {
    ZeroTickBudget,
    EmptyPromptEvent,
    MissingPrivateFeedback,
    StreamTickOverflow,
    DynamicStateCapacityOverflow,
    OutputBudgetOverflow,
    FeedbackDepthOverflow,
    DispatchCountOverflow,
    DescriptorCountOverflow,
    PushConstantByteCountOverflow,
    FeedbackCycleWidthExceeded {
        requested: usize,
        snapshot_capacity: usize,
        sampler_history_capacity: usize,
    },
    FeedbackCyclePushConstants {
        byte_count: u32,
    },
    Tick(VulkanResidentSingleTokenTickRunnerError),
    Sampler(VulkanResidentSamplerRunnerError),
    Vulkan(VulkanError),
}

impl Display for VulkanResidentFeedbackLoopRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTickBudget => f.write_str("feedback loop tick budget must not be zero"),
            Self::EmptyPromptEvent => f.write_str("prompt event must contain input"),
            Self::MissingPrivateFeedback => f.write_str("prompt event expected private feedback"),
            Self::StreamTickOverflow => f.write_str("feedback loop stream tick overflowed"),
            Self::DynamicStateCapacityOverflow => {
                f.write_str("feedback loop dynamic state capacity overflowed")
            }
            Self::OutputBudgetOverflow => f.write_str("running stream output budget overflowed"),
            Self::FeedbackDepthOverflow => f.write_str("feedback loop feedback depth overflowed"),
            Self::DispatchCountOverflow => f.write_str("feedback loop dispatch count overflowed"),
            Self::DescriptorCountOverflow => {
                f.write_str("feedback loop descriptor count overflowed")
            }
            Self::PushConstantByteCountOverflow => {
                f.write_str("feedback loop push constant byte count overflowed")
            }
            Self::FeedbackCycleWidthExceeded {
                requested,
                snapshot_capacity,
                sampler_history_capacity,
            } => write!(
                f,
                "resident feedback cycle requests {requested} ticks, snapshot capacity is {snapshot_capacity} and sampler history capacity is {sampler_history_capacity}"
            ),
            Self::FeedbackCyclePushConstants { byte_count } => write!(
                f,
                "resident feedback cycle requires buffer-backed control, but the compiled tick ABI still declares {byte_count} push-constant bytes"
            ),
            Self::Tick(error) => Display::fmt(error, f),
            Self::Sampler(error) => Display::fmt(error, f),
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentFeedbackLoopRunnerError {}

impl From<VulkanResidentSingleTokenTickRunnerError> for VulkanResidentFeedbackLoopRunnerError {
    fn from(error: VulkanResidentSingleTokenTickRunnerError) -> Self {
        Self::Tick(error)
    }
}

impl From<VulkanResidentSamplerRunnerError> for VulkanResidentFeedbackLoopRunnerError {
    fn from(error: VulkanResidentSamplerRunnerError) -> Self {
        Self::Sampler(error)
    }
}

impl From<VulkanError> for VulkanResidentFeedbackLoopRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}
