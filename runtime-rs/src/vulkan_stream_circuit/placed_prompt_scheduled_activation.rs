enum VulkanResidentInProcessPlacedScheduledActivationStart {
    Complete(VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun),
    Pending,
}

struct VulkanResidentInProcessPlacedPendingSchedulerActivation {
    activation_id: u64,
    input_event_id: String,
    start_stream_tick: u64,
    state_bindings: Vec<VulkanResidentScheduledActivationStateBinding>,
    output_events: Vec<VulkanResidentTokenOutputEvent>,
    generated_token_ids: Vec<u32>,
    max_tokens: usize,
    generated_tokens: usize,
    window: VulkanResidentInProcessPlacedPendingStreamFeedbackWindow,
}

impl VulkanResidentInProcessPlacedPromptStream {
    fn begin_runtime_scheduler_activation<F>(
        &mut self,
        activation: &RuntimeStreamActivation,
        on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedScheduledActivationStart,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "stream already has a submitted scheduler activation",
            ));
        }
        let RuntimeStreamActivationKind::DecodeFeedback { max_tokens, .. } = activation.kind
        else {
            return self
                .run_runtime_scheduler_activation_with_output(activation, on_output_event)
                .map(VulkanResidentInProcessPlacedScheduledActivationStart::Complete);
        };
        if self.speculative_draft_tokens > 0 && self.processor.speculative_decoder_count() > 0 {
            return self
                .run_runtime_scheduler_activation_with_output(activation, on_output_event)
                .map(VulkanResidentInProcessPlacedScheduledActivationStart::Complete);
        }
        let state_bindings = self.scheduler_activation_state_bindings(activation)?;
        let Some(window) = self.submit_resident_feedback_window_limited(max_tokens)? else {
            return self
                .run_runtime_scheduler_activation_with_state_bindings(
                    activation,
                    state_bindings,
                    on_output_event,
                )
                .map(VulkanResidentInProcessPlacedScheduledActivationStart::Complete);
        };
        self.active_input_event
            .as_mut()
            .expect("submitted scheduler feedback requires an active input event")
            .resident_feedback
            .record_asynchronous_submission();
        self.pending_scheduler_activation = Some(
            VulkanResidentInProcessPlacedPendingSchedulerActivation {
                activation_id: activation.id,
                input_event_id: activation.input_event_id.clone(),
                start_stream_tick: self.next_stream_tick(),
                state_bindings,
                output_events: Vec::new(),
                generated_token_ids: Vec::new(),
                max_tokens,
                generated_tokens: 0,
                window,
            },
        );
        Ok(VulkanResidentInProcessPlacedScheduledActivationStart::Pending)
    }

    fn pending_runtime_scheduler_activation_id(&self) -> Option<u64> {
        self.pending_scheduler_activation
            .as_ref()
            .map(|pending| pending.activation_id)
    }

    fn wait_pending_runtime_scheduler_activation_for(
        &mut self,
        timeout_ns: u64,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        let completed = self
            .pending_scheduler_activation
            .as_ref()
            .map(|pending| self.wait_resident_feedback_window_for(&pending.window, timeout_ns))
            .transpose()
            .map(Option::unwrap_or_default)?;
        if let Some(active) = self.active_input_event.as_mut() {
            active
                .resident_feedback
                .record_bounded_wait(completed);
        }
        Ok(completed)
    }

    fn poll_runtime_scheduler_activation_with_output<F>(
        &mut self,
        mut on_output_event: F,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if self.pending_scheduler_activation.is_none() {
            return Ok(None);
        }
        self.active_input_event
            .as_mut()
            .expect("pending scheduler feedback requires an active input event")
            .resident_feedback
            .record_completion_poll();
        let window_is_complete = {
            let pending = self
                .pending_scheduler_activation
                .as_ref()
                .expect("pending scheduler activation was checked");
            self.resident_feedback_window_is_complete(&pending.window)?
        };
        if !window_is_complete {
            return Ok(None);
        }

        let mut pending = self
            .pending_scheduler_activation
            .take()
            .expect("pending scheduler activation disappeared after completion");
        let generated_before = self.active_generated_token_count();
        let mut window_output_events = Vec::new();
        let completion = self.complete_submitted_resident_feedback_window(
            pending.window,
            &mut |output_event| window_output_events.push(output_event),
        )?;
        let remaining_before = pending
            .max_tokens
            .saturating_sub(pending.generated_tokens);
        let generated_delta = self.scheduled_feedback_generated_delta(
            generated_before,
            remaining_before,
            "resident feedback window emitted more tokens than scheduled",
        )?;
        pending.generated_tokens = pending.generated_tokens.saturating_add(generated_delta);
        pending
            .generated_token_ids
            .extend(window_output_events.iter().map(|event| event.token_id));
        for output_event in &window_output_events {
            on_output_event(output_event.clone());
        }
        pending.output_events.extend(window_output_events);

        let mut completed_input_run = self.complete_active_input_event_if_complete()?;
        let can_continue = completed_input_run.is_none()
            && completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_NONE
            && completion.executed_tick_count > 0
            && pending.generated_tokens < pending.max_tokens;
        if can_continue {
            let remaining = pending.max_tokens - pending.generated_tokens;
            if let Some(window) = self.submit_resident_feedback_window_limited(remaining)? {
                self.active_input_event
                    .as_mut()
                    .expect("submitted scheduler feedback requires an active input event")
                    .resident_feedback
                    .record_asynchronous_submission();
                pending.window = window;
                self.pending_scheduler_activation = Some(pending);
                return Ok(None);
            }

            let mut trailing_output_events = Vec::new();
            completed_input_run = self.run_scheduled_feedback_window_with_output(
                &pending.input_event_id,
                remaining,
                &mut |output_event| trailing_output_events.push(output_event),
            )?;
            pending
                .generated_token_ids
                .extend(trailing_output_events.iter().map(|event| event.token_id));
            for output_event in &trailing_output_events {
                on_output_event(output_event.clone());
            }
            pending.output_events.extend(trailing_output_events);
        }
        if completed_input_run.is_none() {
            completed_input_run = self.close_scheduled_loop_if_exhausted()?;
        }

        let outcome = RuntimeStreamActivationOutcome::generated_tokens(
            pending.generated_token_ids.clone(),
            completed_input_run.is_none(),
        );
        Ok(Some(
            VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun {
                activation_id: pending.activation_id,
                input_event_id: pending.input_event_id,
                start_stream_tick: pending.start_stream_tick,
                next_stream_tick: self.next_stream_tick(),
                state_bindings: pending.state_bindings,
                output_events: pending.output_events,
                generated_token_ids: pending.generated_token_ids,
                outcome,
                completed_input_run,
            },
        ))
    }

    pub fn run_runtime_scheduler_activation_with_output<F>(
        &mut self,
        activation: &RuntimeStreamActivation,
        on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let state_bindings = self.scheduler_activation_state_bindings(activation)?;
        self.run_runtime_scheduler_activation_with_state_bindings(
            activation,
            state_bindings,
            on_output_event,
        )
    }

    fn run_runtime_scheduler_activation_with_state_bindings<F>(
        &mut self,
        activation: &RuntimeStreamActivation,
        state_bindings: Vec<VulkanResidentScheduledActivationStateBinding>,
        mut on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let mut output_events = Vec::new();
        let mut generated_token_ids = Vec::new();
        let start_stream_tick = self.next_stream_tick();
        let mut capture_output_event = |event: VulkanResidentTokenOutputEvent| {
            generated_token_ids.push(event.token_id);
            output_events.push(event.clone());
            on_output_event(event);
        };

        let completed_input_run = match &activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { token_ids, .. } => self
                .run_scheduled_prefill_chunk_with_output(
                    activation,
                    token_ids,
                    &mut capture_output_event,
                )?,
            RuntimeStreamActivationKind::DecodeFeedback { max_tokens, .. } => self
                .run_scheduled_feedback_window_with_output(
                    &activation.input_event_id,
                    *max_tokens,
                    &mut capture_output_event,
                )?,
        };

        let outcome = if matches!(activation.kind, RuntimeStreamActivationKind::PrefillChunk { .. })
            && generated_token_ids.is_empty()
        {
            RuntimeStreamActivationOutcome::prefill_complete()
        } else {
            RuntimeStreamActivationOutcome::generated_tokens(
                generated_token_ids.clone(),
                completed_input_run.is_none(),
            )
        };

        Ok(
            VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun {
                activation_id: activation.id,
                input_event_id: activation.input_event_id.clone(),
                start_stream_tick,
                next_stream_tick: self.next_stream_tick(),
                state_bindings,
                output_events,
                generated_token_ids,
                outcome,
                completed_input_run,
            },
        )
    }

    fn run_scheduled_prefill_chunk_with_output<F>(
        &mut self,
        scheduler_activation: &RuntimeStreamActivation,
        token_ids: &[u32],
        on_output_event: &mut F,
    ) -> Result<Option<VulkanResidentInProcessPlacedSubmittedInputRun>, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let mut completed_input_run = None;
        let mut processed = 0usize;
        while processed < token_ids.len() {
            let before_index = self
                .active_input_event
                .as_ref()
                .map(|event| event.next_external_input_index);
            let remaining = token_ids.len() - processed;
            let (ran_block, completed_run) =
                self.run_temporal_external_input_block_limited_with_output(remaining, on_output_event)?;
            if ran_block {
                let after_index = self
                    .active_input_event
                    .as_ref()
                    .map(|event| event.next_external_input_index)
                    .or_else(|| before_index.map(|index| index + remaining));
                let processed_delta = before_index
                    .zip(after_index)
                    .map(|(before, after)| after.saturating_sub(before))
                    .unwrap_or(remaining);
                if processed_delta == 0 || processed_delta > remaining {
                    return Err(placed_scheduler_divergence(
                        "temporal prefill block did not advance by the scheduled prompt window",
                    ));
                }
                processed += processed_delta;
                if let Some(completed_run) = completed_run {
                    completed_input_run = Some(completed_run);
                    break;
                }
                continue;
            }

            let run = self
                .run_next_activation()?
                .ok_or_else(|| placed_scheduler_divergence("scheduled prefill had no backend activation"))?;
            if run.input_event_id != scheduler_activation.input_event_id {
                return Err(placed_scheduler_divergence(
                    "scheduled prefill ran a different input event",
                ));
            }
            if run.input_is_feedback {
                return Err(placed_scheduler_divergence(
                    "scheduled prefill reached private feedback before consuming prompt input",
                ));
            }
            if run.input_token_id != token_ids[processed] {
                return Err(placed_scheduler_divergence(
                    "scheduled prefill token diverged from backend input token",
                ));
            }
            if let Some(output_event) = run.output_event {
                on_output_event(output_event);
            }
            processed += 1;
            if let Some(completed_run) = run.completed_input_run {
                completed_input_run = Some(completed_run);
                break;
            }
        }
        if completed_input_run.is_none() {
            completed_input_run = self.close_scheduled_loop_if_exhausted()?;
        }
        Ok(completed_input_run)
    }

    fn run_scheduled_feedback_window_with_output<F>(
        &mut self,
        scheduler_input_event_id: &str,
        max_tokens: usize,
        on_output_event: &mut F,
    ) -> Result<Option<VulkanResidentInProcessPlacedSubmittedInputRun>, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let mut completed_input_run = None;
        let mut generated = 0usize;
        while generated < max_tokens {
            let remaining = max_tokens - generated;
            let generated_before = self.active_generated_token_count();
            if self.run_speculative_feedback_window_limited_with_output(
                remaining,
                on_output_event,
            )? {
                let generated_delta = self.scheduled_feedback_generated_delta(
                    generated_before,
                    remaining,
                    "speculative feedback window emitted more tokens than scheduled",
                )?;
                if generated_delta == 0 {
                    break;
                }
                generated += generated_delta;
                if let Some(completed_run) = self.complete_active_input_event_if_complete()? {
                    completed_input_run = Some(completed_run);
                    break;
                }
                continue;
            }

            if self.run_resident_feedback_window_limited_with_output(remaining, on_output_event)? {
                let generated_delta = self.scheduled_feedback_generated_delta(
                    generated_before,
                    remaining,
                    "resident feedback window emitted more tokens than scheduled",
                )?;
                if generated_delta == 0 {
                    break;
                }
                generated += generated_delta;
                if let Some(completed_run) = self.complete_active_input_event_if_complete()? {
                    completed_input_run = Some(completed_run);
                    break;
                }
                continue;
            }

            let run = self
                .run_next_activation()?
                .ok_or_else(|| placed_scheduler_divergence("scheduled feedback had no backend activation"))?;
            if run.input_event_id != scheduler_input_event_id {
                return Err(placed_scheduler_divergence(
                    "scheduled feedback ran a different input event",
                ));
            }
            if !run.input_is_feedback {
                return Err(placed_scheduler_divergence(
                    "scheduled feedback reached external input",
                ));
            }
            if let Some(output_event) = run.output_event {
                on_output_event(output_event);
                generated += 1;
            }
            if let Some(completed_run) = run.completed_input_run {
                completed_input_run = Some(completed_run);
                break;
            }
            if generated == max_tokens {
                break;
            }
        }
        if completed_input_run.is_none() {
            completed_input_run = self.close_scheduled_loop_if_exhausted()?;
        }
        Ok(completed_input_run)
    }

    fn active_generated_token_count(&self) -> usize {
        self.active_input_event
            .as_ref()
            .map(|event| event.generated_token_ids.len())
            .unwrap_or(0)
    }

    fn scheduled_feedback_generated_delta(
        &self,
        generated_before: usize,
        remaining: usize,
        overflow_message: &'static str,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let generated_after = self.active_generated_token_count();
        let generated_delta = generated_after.saturating_sub(generated_before);
        if generated_delta > remaining {
            return Err(placed_scheduler_divergence(overflow_message));
        }
        Ok(generated_delta)
    }

    fn complete_active_input_event_if_complete(
        &mut self,
    ) -> Result<Option<VulkanResidentInProcessPlacedSubmittedInputRun>, VulkanResidentInProcessPlacedRuntimeError>
    {
        self.active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
            .then(|| self.complete_active_input_event())
            .transpose()
    }

    fn close_scheduled_loop_if_exhausted(
        &mut self,
    ) -> Result<Option<VulkanResidentInProcessPlacedSubmittedInputRun>, VulkanResidentInProcessPlacedRuntimeError>
    {
        let should_close = self
            .active_input_event
            .as_ref()
            .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
            .is_some_and(|activation| {
                activation.input_is_feedback
                    && activation.input_closes_loop_after_processing
                    && !activation.should_emit_public_output
            });
        if !should_close {
            return self.complete_active_input_event_if_complete();
        }
        if let Some(completed_input_run) = self
            .run_next_activation()?
            .and_then(|run| run.completed_input_run)
        {
            return Ok(Some(completed_input_run));
        }
        self.complete_active_input_event_if_complete()
    }

    fn scheduler_activation_state_bindings(
        &mut self,
        activation: &RuntimeStreamActivation,
    ) -> Result<
        Vec<VulkanResidentScheduledActivationStateBinding>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut bindings = Vec::new();
        for reservation in &activation.state_reservations {
            bindings.extend(self.scheduler_state_reservation_bindings(reservation)?);
        }
        Ok(bindings)
    }

    fn scheduler_state_reservation_bindings(
        &mut self,
        reservation: &RuntimeStreamStateReservation,
    ) -> Result<
        Vec<VulkanResidentScheduledActivationStateBinding>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut bindings = Vec::with_capacity(reservation.slots.len());
        for slot in &reservation.slots {
            let state = self.package.resident_state_buffer(&slot.key).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "scheduled transient state {}.{} is not resident in package {:?}",
                        slot.key.node_instance_id, slot.key.state_id, self.package.package_id
                    )),
                )
            })?;
            let page_binding = self
                .transient_state_pages
                .bind_slot(
                    state,
                    self.package.dynamic_state_capacity_activations,
                    slot,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            bindings.push(VulkanResidentScheduledActivationStateBinding {
                key: page_binding.key,
                logical_activation_index: page_binding.logical_activation_index,
                transient_block_id: page_binding.transient_block_id,
                transient_block_activation_offset: page_binding.transient_block_activation_offset,
                transient_block_activation_capacity: page_binding
                    .transient_block_activation_capacity,
                resident_page_index: page_binding.resident_page_index,
                resident_activation_offset: page_binding.resident_activation_offset,
                resident_byte_offset: page_binding.resident_byte_offset,
                bytes_per_activation: page_binding.bytes_per_activation,
            });
        }
        Ok(bindings)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun {
    pub activation_id: u64,
    pub input_event_id: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub state_bindings: Vec<VulkanResidentScheduledActivationStateBinding>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub outcome: RuntimeStreamActivationOutcome,
    pub completed_input_run: Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentScheduledActivationStateBinding {
    pub key: TransientStateKey,
    pub logical_activation_index: usize,
    pub transient_block_id: TransientStateBlockId,
    pub transient_block_activation_offset: usize,
    pub transient_block_activation_capacity: usize,
    pub resident_page_index: usize,
    pub resident_activation_offset: usize,
    pub resident_byte_offset: usize,
    pub bytes_per_activation: usize,
}
