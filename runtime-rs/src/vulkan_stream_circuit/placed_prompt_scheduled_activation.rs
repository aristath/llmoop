impl VulkanResidentInProcessPlacedPromptStream {
    pub fn run_runtime_scheduler_activation_with_output<F>(
        &mut self,
        activation: &RuntimeStreamActivation,
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
        let state_bindings = self.scheduler_activation_state_bindings(activation)?;
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
                    activation,
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
        scheduler_activation: &RuntimeStreamActivation,
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
            if run.input_event_id != scheduler_activation.input_event_id {
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
