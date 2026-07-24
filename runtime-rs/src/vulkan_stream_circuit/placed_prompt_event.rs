#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedSingleTokenTickRun {
    pub input_device_id: String,
    pub output_device_id: String,
    pub token_id: u32,
    pub stream_tick: u64,
    pub input_run: VulkanResidentInputEmbeddingTransducerRun,
    pub placed_run: VulkanMountedPlacedResidentInProcessStreamTickRun,
    pub output_run: Option<VulkanResidentOutputTransducerRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedSingleTokenSampleRun {
    pub tick_run: VulkanResidentInProcessPlacedSingleTokenTickRun,
    pub sampler_run: VulkanResidentSamplerRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedFeedbackLoopRun {
    pub input_device_id: String,
    pub output_device_id: String,
    pub initial_token_id: u32,
    pub sampled_token_ids: Vec<u32>,
    pub tick_runs: Vec<VulkanResidentInProcessPlacedFeedbackTickRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedFeedbackTickRun {
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub sampled_token_id: u32,
    pub tick_run: VulkanResidentInProcessPlacedSingleTokenSampleRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEventRun {
    pub input_device_id: String,
    pub output_device_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub stop_reason: String,
    pub tick_count: usize,
    pub scheduler_turn_count: usize,
    pub completed_stage_count: usize,
    pub transport_stats: VulkanPlacedEdgeTransportStats,
    pub output_source_stream_ticks: Vec<u64>,
    pub speculative_decode: VulkanSpeculativeDecodeStats,
    pub resident_feedback: VulkanResidentFeedbackExecutionStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentFeedbackExecutionStats {
    pub window_count: usize,
    pub planned_tick_count: usize,
    pub submitted_tick_count: usize,
    pub executed_tick_count: usize,
    pub retained_tick_count: usize,
    pub sampled_tick_count: usize,
    pub discarded_tick_count: usize,
    pub template_record_count: usize,
    pub template_replay_count: usize,
}

impl VulkanResidentFeedbackExecutionStats {
    fn record_window(
        &mut self,
        planned_tick_count: usize,
        executed_tick_count: usize,
        sampled_tick_count: usize,
        template_replayed: bool,
    ) {
        self.window_count = self.window_count.saturating_add(1);
        self.planned_tick_count = self.planned_tick_count.saturating_add(planned_tick_count);
        self.submitted_tick_count = self
            .submitted_tick_count
            .saturating_add(planned_tick_count);
        self.executed_tick_count = self
            .executed_tick_count
            .saturating_add(executed_tick_count);
        self.retained_tick_count = self
            .retained_tick_count
            .saturating_add(executed_tick_count);
        self.sampled_tick_count = self.sampled_tick_count.saturating_add(sampled_tick_count);
        self.discarded_tick_count = self
            .discarded_tick_count
            .saturating_add(planned_tick_count.saturating_sub(executed_tick_count));
        if template_replayed {
            self.template_replay_count = self.template_replay_count.saturating_add(1);
        } else {
            self.template_record_count = self.template_record_count.saturating_add(1);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentInProcessPlacedActivePromptEvent {
    input_event: VulkanResidentTokenInputEvent,
    start_stream_tick: u64,
    next_external_input_index: usize,
    pending_feedback: Option<VulkanResidentPendingPrivateFeedback>,
    remaining_public_outputs: usize,
    stop_reason: Option<String>,
    terminated: bool,
    tick_count: usize,
    scheduler_turn_count: usize,
    completed_stage_count: usize,
    transport_stats: VulkanPlacedEdgeTransportStats,
    generated_token_ids: Vec<u32>,
    output_source_stream_ticks: Vec<u64>,
    output_events: Vec<VulkanResidentTokenOutputEvent>,
    speculative_decode: VulkanSpeculativeDecodeStats,
    resident_feedback: VulkanResidentFeedbackExecutionStats,
}

impl VulkanResidentInProcessPlacedActivePromptEvent {
    fn new(
        input_event: VulkanResidentTokenInputEvent,
        start_stream_tick: u64,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if input_event.token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::EmptyPromptEvent);
        }
        let remaining_public_outputs = input_event.max_public_tokens;
        let stop_reason = (remaining_public_outputs == 0).then(|| "max_new_tokens".to_string());
        Ok(Self {
            generated_token_ids: Vec::with_capacity(remaining_public_outputs),
            output_source_stream_ticks: Vec::with_capacity(remaining_public_outputs),
            output_events: Vec::with_capacity(remaining_public_outputs),
            resident_feedback: VulkanResidentFeedbackExecutionStats::default(),
            input_event,
            start_stream_tick,
            next_external_input_index: 0,
            pending_feedback: None,
            remaining_public_outputs,
            stop_reason,
            terminated: false,
            tick_count: 0,
            scheduler_turn_count: 0,
            completed_stage_count: 0,
            transport_stats: VulkanPlacedEdgeTransportStats::default(),
            speculative_decode: VulkanSpeculativeDecodeStats::default(),
        })
    }

    fn next_activation(&self) -> Option<VulkanResidentInProcessPlacedPromptActivation> {
        if self.terminated {
            return None;
        }
        let (
            input_token_id,
            input_feedback_depth,
            input_closes_loop_after_processing,
            input_is_feedback,
        ) = if self.next_external_input_index < self.input_event.token_ids.len() {
            (
                self.input_event.token_ids[self.next_external_input_index],
                0,
                false,
                false,
            )
        } else {
            let feedback = self.pending_feedback.as_ref()?;
            (
                feedback.token_id,
                feedback.feedback_depth,
                feedback.closes_loop_after_processing,
                true,
            )
        };
        let external_inputs_remaining = self.input_event.token_ids.len()
            - self.next_external_input_index
            - usize::from(!input_is_feedback);
        Some(VulkanResidentInProcessPlacedPromptActivation {
            input_token_id,
            input_feedback_depth,
            input_closes_loop_after_processing,
            input_is_feedback,
            should_emit_public_output: self.remaining_public_outputs > 0
                && external_inputs_remaining == 0,
        })
    }

    fn resident_feedback_window_tick_count(&self, mounted_window_width: usize) -> usize {
        if mounted_window_width < 2 || self.remaining_public_outputs < 2 {
            return 0;
        }
        let Some(activation) = self.next_activation() else {
            return 0;
        };
        if !activation.input_is_feedback
            || activation.input_closes_loop_after_processing
            || !activation.should_emit_public_output
        {
            return 0;
        }
        mounted_window_width.min(self.remaining_public_outputs)
    }

    fn complete_activation(
        &mut self,
        activation: &VulkanResidentInProcessPlacedPromptActivation,
        stream_tick: u64,
        scheduler_turn_count: usize,
        completed_stage_count: usize,
        transport_stats: &VulkanPlacedEdgeTransportStats,
        sampled_token_id: Option<u32>,
    ) -> Result<Option<VulkanResidentTokenOutputEvent>, VulkanResidentInProcessPlacedRuntimeError>
    {
        if activation.input_is_feedback {
            self.pending_feedback
                .take()
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
        } else {
            self.next_external_input_index = self.next_external_input_index.saturating_add(1);
        }

        self.tick_count = self.tick_count.saturating_add(1);
        self.scheduler_turn_count = self
            .scheduler_turn_count
            .saturating_add(scheduler_turn_count);
        self.completed_stage_count = self
            .completed_stage_count
            .saturating_add(completed_stage_count);
        self.transport_stats.accumulate(transport_stats);

        let output_event = if activation.should_emit_public_output {
            let sampled_token_id = sampled_token_id
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingFusedSamplerRun)?;
            let output_index = self.generated_token_ids.len();
            let output_event = VulkanResidentTokenOutputEvent {
                id: format!("{}.{}", self.input_event.id, output_index),
                input_event_id: self.input_event.id.clone(),
                output_index,
                token_id: sampled_token_id,
                source_stream_tick: stream_tick,
            };
            self.generated_token_ids.push(sampled_token_id);
            self.output_source_stream_ticks.push(stream_tick);
            self.output_events.push(output_event.clone());
            self.remaining_public_outputs -= 1;

            let closes_loop_after_processing =
                if self.input_event.stop_token_ids.contains(&sampled_token_id) {
                    self.remaining_public_outputs = 0;
                    self.stop_reason = Some("eos".to_string());
                    true
                } else if self.remaining_public_outputs == 0 {
                    self.stop_reason = Some("max_new_tokens".to_string());
                    true
                } else {
                    false
                };
            self.pending_feedback = Some(VulkanResidentPendingPrivateFeedback {
                token_id: sampled_token_id,
                feedback_depth: activation
                    .input_feedback_depth
                    .checked_add(1)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::FeedbackDepthOverflow)?,
                closes_loop_after_processing,
            });
            Some(output_event)
        } else {
            None
        };

        if activation.input_closes_loop_after_processing {
            self.pending_feedback = None;
        }
        Ok(output_event)
    }

    fn is_complete(&self) -> bool {
        self.terminated
            || (self.next_external_input_index == self.input_event.token_ids.len()
                && self.pending_feedback.is_none())
    }

    fn interrupt(&mut self, reason: impl Into<String>) -> VulkanResidentStreamControlEvent {
        let reason = reason.into();
        let cleared_private_feedback_ids = self
            .pending_feedback
            .take()
            .map(|_| self.pending_feedback_id())
            .into_iter()
            .collect();
        self.remaining_public_outputs = 0;
        self.stop_reason = Some(reason.clone());
        self.terminated = true;
        VulkanResidentStreamControlEvent {
            event_type: VulkanResidentStreamControlEventType::Interrupt,
            reason,
            cleared_private_feedback_ids,
            closing_private_feedback_id: None,
            state_preserved: true,
        }
    }

    fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentStreamControlEvent {
        let reason = reason.into();
        let closing_private_feedback_id = self
            .pending_feedback
            .as_ref()
            .map(|_| self.pending_feedback_id());
        if let Some(feedback) = &mut self.pending_feedback {
            feedback.closes_loop_after_processing = true;
        }
        self.remaining_public_outputs = 0;
        self.stop_reason = Some(reason.clone());
        VulkanResidentStreamControlEvent {
            event_type: VulkanResidentStreamControlEventType::StopAfterCurrent,
            reason,
            cleared_private_feedback_ids: Vec::new(),
            closing_private_feedback_id,
            state_preserved: true,
        }
    }

    fn pending_feedback_id(&self) -> String {
        format!(
            "{}.feedback.{}",
            self.input_event.id,
            self.generated_token_ids.len().saturating_sub(1)
        )
    }

    fn into_event_run(
        self,
        input_device_id: String,
        output_device_id: String,
    ) -> VulkanResidentInProcessPlacedPromptEventRun {
        let output_token_ids = self.input_event.token_ids[..self.next_external_input_index]
            .iter()
            .copied()
            .chain(self.generated_token_ids.iter().copied())
            .collect();
        VulkanResidentInProcessPlacedPromptEventRun {
            input_device_id,
            output_device_id,
            prompt_token_ids: self.input_event.token_ids,
            generated_token_ids: self.generated_token_ids,
            output_token_ids,
            stop_reason: self
                .stop_reason
                .unwrap_or_else(|| "max_new_tokens".to_string()),
            tick_count: self.tick_count,
            scheduler_turn_count: self.scheduler_turn_count,
            completed_stage_count: self.completed_stage_count,
            transport_stats: self.transport_stats,
            output_source_stream_ticks: self.output_source_stream_ticks,
            speculative_decode: self.speculative_decode,
            resident_feedback: self.resident_feedback,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentInProcessPlacedPromptActivation {
    input_token_id: u32,
    input_feedback_depth: u32,
    input_closes_loop_after_processing: bool,
    input_is_feedback: bool,
    should_emit_public_output: bool,
}
