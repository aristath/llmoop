pub struct VulkanResidentInProcessPlacedPromptStream {
    package: Arc<VulkanResidentInProcessPlacedModelPackage>,
    processor: VulkanResidentInProcessPlacedStreamProcessor,
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    session: VulkanResidentInProcessPlacedPromptSession,
    active_input_event: Option<VulkanResidentInProcessPlacedActivePromptEvent>,
    pending_input_events: VecDeque<VulkanResidentTokenInputEvent>,
    speculative_draft_tokens: usize,
}

impl VulkanResidentInProcessPlacedPromptStream {
    pub fn new(
        package: Arc<VulkanResidentInProcessPlacedModelPackage>,
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_package_devices_and_session(package, devices, random_seed, 0)
    }

    pub fn from_runtime_model_for_bound_devices(
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        random_seed: u32,
        speculative_draft_tokens: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_bound_devices_with_sampler_config(
            devices,
            manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
            random_seed,
            speculative_draft_tokens,
            VulkanResidentSamplerRuntimeConfig::default(),
        )
    }

    pub fn from_runtime_model_for_bound_devices_with_sampler_config(
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        mut runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        random_seed: u32,
        speculative_draft_tokens: usize,
        sampler_config: VulkanResidentSamplerRuntimeConfig,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        runtime_model.package.sampler.spec = sampler_config
            .apply_to(&runtime_model.package.sampler.spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let package = Arc::new(
            VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
                &devices,
                manifest_dir,
                runtime_model,
                dynamic_state_capacity_activations,
                speculative_draft_tokens > 0,
            )?,
        );
        let mut stream = Self::new(package, devices, random_seed)?;
        stream.speculative_draft_tokens = speculative_draft_tokens;
        Ok(stream)
    }

    pub fn from_package_devices_and_session(
        package: Arc<VulkanResidentInProcessPlacedModelPackage>,
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
        start_stream_tick: u64,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        for device_id in &package.device_ids {
            if !devices.contains_key(device_id) {
                return Err(
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.clone(),
                    },
                );
            }
        }
        let processor = package.create_stream_processor_for_bound_devices(&devices, random_seed)?;
        let session = processor.prompt_session_from_stream_tick(start_stream_tick);
        Ok(Self {
            package,
            processor,
            devices,
            session,
            active_input_event: None,
            pending_input_events: VecDeque::new(),
            speculative_draft_tokens: 0,
        })
    }

    pub fn package(&self) -> &VulkanResidentInProcessPlacedModelPackage {
        &self.package
    }

    pub fn session(&self) -> &VulkanResidentInProcessPlacedPromptSession {
        &self.session
    }

    pub fn devices(&self) -> &BTreeMap<String, Rc<VulkanComputeDevice>> {
        &self.devices
    }

    pub fn remount_model_preserving_state(
        &mut self,
        package: Arc<VulkanResidentInProcessPlacedModelPackage>,
        random_seed: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let processor = package.create_stream_processor_inheriting_state_for_bound_devices(
            &self.devices,
            random_seed,
            &self.processor,
        )?;
        self.session.transport = VulkanInProcessPlacedEdgeTransport::new();
        self.package = package;
        self.processor = processor;
        Ok(())
    }

    pub fn next_stream_tick(&self) -> u64 {
        self.session.next_stream_tick
    }

    pub fn completed_prompt_event_count(&self) -> usize {
        self.session.completed_prompt_event_count
    }

    pub fn pending_input_event_count(&self) -> usize {
        self.pending_input_events.len() + usize::from(self.active_input_event.is_some())
    }

    pub fn is_idle(&self) -> bool {
        self.active_input_event.is_none() && self.pending_input_events.is_empty()
    }

    pub fn enqueue_input_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> VulkanResidentInProcessPlacedQueuedInputEvent {
        self.pending_input_events.push_back(event.clone());
        VulkanResidentInProcessPlacedQueuedInputEvent {
            input_event: event,
            pending_input_event_count: self.pending_input_event_count(),
            next_stream_tick: self.next_stream_tick(),
        }
    }

    fn activate_next_input_event(
        &mut self,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        if self.active_input_event.is_some() {
            return Ok(true);
        }
        let Some(input_event) = self.pending_input_events.pop_front() else {
            return Ok(false);
        };
        self.active_input_event = Some(VulkanResidentInProcessPlacedActivePromptEvent::new(
            input_event,
            self.session.next_stream_tick,
        )?);
        Ok(true)
    }

    fn run_temporal_external_input_block_with_output<F>(
        &mut self,
        on_output_event: &mut F,
    ) -> Result<
        (bool, Option<VulkanResidentInProcessPlacedSubmittedInputRun>),
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        self.run_temporal_external_input_block_limited_with_output(usize::MAX, on_output_event)
    }

    fn run_temporal_external_input_block_limited_with_output<F>(
        &mut self,
        max_external_inputs: usize,
        on_output_event: &mut F,
    ) -> Result<
        (bool, Option<VulkanResidentInProcessPlacedSubmittedInputRun>),
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if !self.activate_next_input_event()? {
            return Ok((false, None));
        }
        if max_external_inputs < 2 {
            return Ok((false, None));
        }
        let active = self
            .active_input_event
            .as_ref()
            .expect("temporal block requires an active input event");
        let external_input_count = active
            .input_event
            .token_ids
            .len()
            .saturating_sub(active.next_external_input_index);
        if external_input_count < 2 || active.pending_feedback.is_some() {
            return Ok((false, None));
        }
        let block_width = self.processor.temporal_block_width(external_input_count)?;
        if block_width < 2 {
            return Ok((false, None));
        }
        if block_width > max_external_inputs {
            return Ok((false, None));
        }
        let block_start_index = active.next_external_input_index;
        let block_end_index = block_start_index + block_width;
        let input_token_ids =
            active.input_event.token_ids[block_start_index..block_end_index].to_vec();
        let sample_last = block_end_index == active.input_event.token_ids.len()
            && active.remaining_public_outputs > 0;
        let start_stream_tick = self.session.next_stream_tick;
        let block_run = self.processor.run_temporal_prompt_block(
            &self.devices,
            &input_token_ids,
            start_stream_tick,
            sample_last,
        )?;

        for (block_index, input_token_id) in input_token_ids.iter().enumerate() {
            let stream_tick = self.session.next_stream_tick;
            let activation = self
                .active_input_event
                .as_ref()
                .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
            if activation.input_is_feedback || activation.input_token_id != *input_token_id {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "temporal block diverged from the external input queue".to_string(),
                    ),
                ));
            }
            let transport_stats = if block_index + 1 == block_width {
                &block_run.transport_stats
            } else {
                &VulkanPlacedEdgeTransportStats::default()
            };
            let sampled_token_id = (block_index + 1 == block_width)
                .then_some(block_run.sampled_token_id)
                .flatten();
            let output_event = self
                .active_input_event
                .as_mut()
                .expect("temporal block requires an active input event")
                .complete_activation(
                    &activation,
                    stream_tick,
                    block_run.scheduler_turn_count_per_tick,
                    block_run.completed_stage_count_per_tick,
                    transport_stats,
                    sampled_token_id,
                )?;
            self.session.next_stream_tick = stream_tick
                .checked_add(1)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            if let Some(output_event) = output_event {
                on_output_event(output_event);
            }
        }

        let completed_input_run = self
            .active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
            .then(|| self.complete_active_input_event())
            .transpose()?;
        Ok((true, completed_input_run))
    }

    pub fn run_next_activation(
        &mut self,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedPromptStreamActivationRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        if !self.activate_next_input_event()? {
            return Ok(None);
        }

        let activation = self
            .active_input_event
            .as_ref()
            .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
        let input_event_id = self
            .active_input_event
            .as_ref()
            .expect("active prompt event was initialized")
            .input_event
            .id
            .clone();
        let stream_tick = self.session.next_stream_tick;
        let tail = if activation.should_emit_public_output {
            VulkanResidentPlacedTokenTickTail::Sample
        } else if self.processor.speculative_decoder_count() > 0 {
            VulkanResidentPlacedTokenTickTail::Hidden
        } else {
            VulkanResidentPlacedTokenTickTail::None
        };
        self.processor.prepare_token_input(placed_token_input(
            activation.input_token_id,
            &self.processor.model.input_device_id,
            &self.processor.model.output_device_id,
            activation.input_is_feedback,
        ))?;
        let placed_run = self
            .processor
            .execute_prepared_token_id_stream_tick_on_bound_devices_in_process_with_transport(
                &self.devices,
                &mut self.session.transport,
                stream_tick,
                tail,
            )?;
        let sampled_token_id = if activation.should_emit_public_output {
            Some(
                self.processor
                    .sampler
                    .completed_token_id()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?,
            )
        } else {
            None
        };
        self.processor
            .synchronize_speculative_decoders_after_target_tick(
                &self.devices,
                activation.input_token_id,
                stream_tick,
            )?;
        let output_event = self
            .active_input_event
            .as_mut()
            .expect("active prompt event was initialized")
            .complete_activation(
                &activation,
                stream_tick,
                placed_run.scheduler_turn_count,
                placed_run.completed_stage_delta,
                &placed_run.transport_stats,
                sampled_token_id,
            )?;
        self.session.next_stream_tick = stream_tick
            .checked_add(1)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;

        let completed_input_run = if self
            .active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
        {
            Some(self.complete_active_input_event()?)
        } else {
            None
        };

        Ok(Some(
            VulkanResidentInProcessPlacedPromptStreamActivationRun {
                input_event_id,
                stream_tick,
                input_token_id: activation.input_token_id,
                input_is_feedback: activation.input_is_feedback,
                output_event,
                completed_input_run,
            },
        ))
    }

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
        let mut capture_output_event = |event: VulkanResidentTokenOutputEvent| {
            generated_token_ids.push(event.token_id);
            output_events.push(event.clone());
            on_output_event(event);
        };

        let completed_input_run = match &activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { token_ids, .. } => {
                self.run_scheduled_prefill_chunk_with_output(
                    activation,
                    token_ids,
                    &mut capture_output_event,
                )?
            }
            RuntimeStreamActivationKind::DecodeFeedback { max_tokens, .. } => {
                self.run_scheduled_feedback_window_with_output(
                    activation,
                    *max_tokens,
                    &mut capture_output_event,
                )?
            }
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
            let generated_before = self
                .active_input_event
                .as_ref()
                .map(|event| event.generated_token_ids.len())
                .unwrap_or(0);
            if self.run_resident_feedback_window_limited_with_output(remaining, on_output_event)? {
                let generated_after = self
                    .active_input_event
                    .as_ref()
                    .map(|event| event.generated_token_ids.len())
                    .unwrap_or(generated_before);
                let generated_delta = generated_after.saturating_sub(generated_before);
                if generated_delta == 0 {
                    break;
                }
                generated += generated_delta;
                if generated_delta > remaining {
                    return Err(placed_scheduler_divergence(
                        "resident feedback window emitted more tokens than scheduled",
                    ));
                }
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
        let _ = self.run_next_activation()?;
        self.complete_active_input_event_if_complete()
    }

    pub fn interrupt(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptStreamControlRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let reason = reason.into();
        let control_event = if let Some(active_input_event) = &mut self.active_input_event {
            active_input_event.interrupt(reason)
        } else {
            VulkanResidentStreamControlEvent {
                event_type: VulkanResidentStreamControlEventType::Interrupt,
                reason,
                cleared_private_feedback_ids: Vec::new(),
                closing_private_feedback_id: None,
                state_preserved: true,
            }
        };
        let completed_input_run = self
            .active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
            .then(|| self.complete_active_input_event())
            .transpose()?;
        Ok(VulkanResidentInProcessPlacedPromptStreamControlRun {
            control_event,
            completed_input_run,
        })
    }

    pub fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentInProcessPlacedPromptStreamControlRun {
        let reason = reason.into();
        let control_event = if let Some(active_input_event) = &mut self.active_input_event {
            active_input_event.stop_after_current(reason)
        } else {
            VulkanResidentStreamControlEvent {
                event_type: VulkanResidentStreamControlEventType::StopAfterCurrent,
                reason,
                cleared_private_feedback_ids: Vec::new(),
                closing_private_feedback_id: None,
                state_preserved: true,
            }
        };
        VulkanResidentInProcessPlacedPromptStreamControlRun {
            control_event,
            completed_input_run: None,
        }
    }

    pub fn run_next_queued_input_event(
        &mut self,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.run_next_queued_input_event_with_output(|_| {})
    }

    pub fn run_next_queued_input_event_with_output<F>(
        &mut self,
        mut on_output_event: F,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if self.is_idle() {
            return Ok(None);
        }
        loop {
            let (ran_temporal_block, completed_input_run) =
                self.run_temporal_external_input_block_with_output(&mut on_output_event)?;
            if let Some(completed_input_run) = completed_input_run {
                return Ok(Some(completed_input_run));
            }
            if ran_temporal_block {
                continue;
            }
            if self.run_speculative_feedback_window_with_output(&mut on_output_event)? {
                if self
                    .active_input_event
                    .as_ref()
                    .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
                {
                    return Ok(Some(self.complete_active_input_event()?));
                }
                continue;
            }
            if self.run_resident_feedback_window_with_output(&mut on_output_event)? {
                if self
                    .active_input_event
                    .as_ref()
                    .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
                {
                    return Ok(Some(self.complete_active_input_event()?));
                }
                continue;
            }
            let activation = self
                .run_next_activation()?
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
            if let Some(output_event) = activation.output_event {
                on_output_event(output_event);
            }
            if let Some(completed_input_run) = activation.completed_input_run {
                return Ok(Some(completed_input_run));
            }
        }
    }

    fn run_speculative_feedback_window_with_output<F>(
        &mut self,
        on_output_event: &mut F,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if self.speculative_draft_tokens == 0 || self.processor.speculative_decoder_count() == 0 {
            return Ok(false);
        }
        let Some(active) = self.active_input_event.as_ref() else {
            return Ok(false);
        };
        let Some(activation) = active.next_activation() else {
            return Ok(false);
        };
        if !activation.input_is_feedback
            || activation.input_closes_loop_after_processing
            || !activation.should_emit_public_output
            || active.remaining_public_outputs < 2
        {
            return Ok(false);
        }
        let draft_token_count = self
            .speculative_draft_tokens
            .min(active.remaining_public_outputs - 1);
        let stop_token_ids = active
            .input_event
            .stop_token_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let start_stream_tick = self.session.next_stream_tick;
        let cycle = self.processor.run_speculative_cycle_on_bound_devices(
            &self.devices,
            activation.input_token_id,
            start_stream_tick,
            draft_token_count,
            &stop_token_ids,
        )?;
        self.active_input_event
            .as_mut()
            .expect("speculative feedback cycle requires an active input event")
            .speculative_decode
            .record_cycle(&cycle);
        for sampled_token_id in cycle.verification.emitted_token_ids {
            let stream_tick = self.session.next_stream_tick;
            let output_event = {
                let active = self
                    .active_input_event
                    .as_mut()
                    .expect("speculative feedback cycle requires an active input event");
                let activation = active
                    .next_activation()
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
                active.complete_activation(
                    &activation,
                    stream_tick,
                    0,
                    0,
                    &VulkanPlacedEdgeTransportStats::default(),
                    Some(sampled_token_id),
                )?
            };
            self.session.next_stream_tick = stream_tick
                .checked_add(1)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            if let Some(output_event) = output_event {
                on_output_event(output_event);
            }
        }
        Ok(true)
    }

    pub fn submit_input_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<
        VulkanResidentInProcessPlacedSubmittedInputRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.ensure_idle_for_immediate_input_event()?;
        self.enqueue_input_event(event);
        self.run_next_queued_input_event()?
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::EmptyPromptEvent)
    }

    pub fn run_queued_input_events_until_idle(
        &mut self,
    ) -> Result<VulkanResidentInProcessPlacedInputQueueRun, VulkanResidentInProcessPlacedRuntimeError>
    {
        let start_stream_tick = self.next_stream_tick();
        let mut submitted_runs = Vec::new();
        while let Some(submitted_run) = self.run_next_queued_input_event()? {
            submitted_runs.push(submitted_run);
        }
        let next_stream_tick = self.next_stream_tick();
        let output_events = submitted_runs
            .iter()
            .flat_map(|submitted_run| submitted_run.output_events.iter().cloned())
            .collect::<Vec<_>>();
        let generated_token_ids = output_events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>();
        let tick_count = submitted_runs
            .iter()
            .map(|submitted_run| submitted_run.session_run.tick_count)
            .sum::<usize>();

        Ok(VulkanResidentInProcessPlacedInputQueueRun {
            start_stream_tick,
            next_stream_tick,
            submitted_runs,
            output_events,
            generated_token_ids,
            tick_count,
            pending_input_event_count: self.pending_input_event_count(),
        })
    }

    pub fn submit_input_events_until_idle<I>(
        &mut self,
        events: I,
    ) -> Result<VulkanResidentInProcessPlacedInputQueueRun, VulkanResidentInProcessPlacedRuntimeError>
    where
        I: IntoIterator<Item = VulkanResidentTokenInputEvent>,
    {
        for event in events {
            self.enqueue_input_event(event);
        }
        self.run_queued_input_events_until_idle()
    }

    fn ensure_idle_for_immediate_input_event(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if self.is_idle() {
            Ok(())
        } else {
            Err(VulkanResidentInProcessPlacedRuntimeError::PromptStreamBusy)
        }
    }

    fn run_resident_feedback_window_with_output<F>(
        &mut self,
        on_output_event: &mut F,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        self.run_resident_feedback_window_limited_with_output(usize::MAX, on_output_event)
    }

    fn run_resident_feedback_window_limited_with_output<F>(
        &mut self,
        max_feedback_ticks: usize,
        on_output_event: &mut F,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let processor = &self.processor;
        let devices = &self.devices;
        let active_input_event = &mut self.active_input_event;
        let session = &mut self.session;
        let window_width = processor.resident_feedback_window_width();
        let mut remaining_feedback_ticks = max_feedback_ticks;
        let mut submission_replay = None;
        let mut ran_window = false;
        loop {
            let mut tick_count = active_input_event
                .as_ref()
                .map(|event| event.resident_feedback_window_tick_count(window_width))
                .unwrap_or(0);
            tick_count = tick_count.min(remaining_feedback_ticks);
            if tick_count == 0 {
                break;
            }
            if tick_count < 2 {
                break;
            }
            let tick_delta = u64::try_from(tick_count)
                .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            session
                .next_stream_tick
                .checked_add(tick_delta)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let feedback_depth_delta = u32::try_from(tick_count)
                .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::FeedbackDepthOverflow)?;
            active_input_event
                .as_ref()
                .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?
                .input_feedback_depth
                .checked_add(feedback_depth_delta)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::FeedbackDepthOverflow)?;
            let start_stream_tick = session.next_stream_tick;
            let mut restore_after_tick = None;
            let replay_slot = (tick_count == window_width).then_some(&mut submission_replay);
            processor.run_resident_feedback_window(
                devices,
                start_stream_tick,
                tick_count,
                replay_slot,
                |tick_index, sampled_token_id, scheduler_turn_count, completed_stage_count| {
                    if restore_after_tick.is_some() {
                        return Ok(());
                    }
                    let stream_tick = session.next_stream_tick;
                    let (output_event, closes_loop) = {
                        let active_input_event = active_input_event
                            .as_mut()
                            .expect("resident feedback window requires an active input event");
                        let activation = active_input_event.next_activation().ok_or(
                            VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback,
                        )?;
                        let closes_loop = activation.input_closes_loop_after_processing;
                        let output_event = active_input_event.complete_activation(
                            &activation,
                            stream_tick,
                            scheduler_turn_count,
                            completed_stage_count,
                            &VulkanPlacedEdgeTransportStats::default(),
                            activation
                                .should_emit_public_output
                                .then_some(sampled_token_id),
                        )?;
                        (output_event, closes_loop)
                    };
                    session.next_stream_tick = stream_tick
                        .checked_add(1)
                        .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                    if let Some(output_event) = output_event {
                        on_output_event(output_event);
                    }
                    if closes_loop {
                        restore_after_tick = Some(tick_index);
                    }
                    Ok(())
                },
            )?;
            ran_window = true;
            remaining_feedback_ticks = remaining_feedback_ticks.saturating_sub(tick_count);
            if remaining_feedback_ticks == 0 {
                break;
            }
            if let Some(tick_index) = restore_after_tick {
                processor.restore_resident_feedback_state_after_tick(tick_index)?;
                break;
            }
        }
        Ok(ran_window)
    }

    fn complete_active_input_event(
        &mut self,
    ) -> Result<
        VulkanResidentInProcessPlacedSubmittedInputRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let active_input_event = self
            .active_input_event
            .take()
            .expect("completed prompt event was active");
        debug_assert!(active_input_event.is_complete());
        let input_event = active_input_event.input_event.clone();
        let output_events = active_input_event.output_events.clone();
        let generated_token_ids = active_input_event.generated_token_ids.clone();
        let start_stream_tick = active_input_event.start_stream_tick;
        let event_run = active_input_event.into_event_run(
            self.package.input_device_id.clone(),
            self.package.output_device_id.clone(),
        );
        let session_run = self
            .session
            .complete_prompt_event(start_stream_tick, event_run)?;
        Ok(VulkanResidentInProcessPlacedSubmittedInputRun {
            input_event,
            pending_input_event_count: self.pending_input_event_count(),
            session_run,
            output_events,
            generated_token_ids,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedQueuedInputEvent {
    pub input_event: VulkanResidentTokenInputEvent,
    pub pending_input_event_count: usize,
    pub next_stream_tick: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedSubmittedInputRun {
    pub input_event: VulkanResidentTokenInputEvent,
    pub pending_input_event_count: usize,
    pub session_run: VulkanResidentInProcessPlacedPromptSessionRun,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptStreamActivationRun {
    pub input_event_id: String,
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub input_is_feedback: bool,
    pub output_event: Option<VulkanResidentTokenOutputEvent>,
    pub completed_input_run: Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun {
    pub activation_id: u64,
    pub input_event_id: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub outcome: RuntimeStreamActivationOutcome,
    pub completed_input_run: Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptStreamControlRun {
    pub control_event: VulkanResidentStreamControlEvent,
    pub completed_input_run: Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedInputQueueRun {
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub submitted_runs: Vec<VulkanResidentInProcessPlacedSubmittedInputRun>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub tick_count: usize,
    pub pending_input_event_count: usize,
}

fn placed_scheduler_divergence(message: impl Into<String>) -> VulkanResidentInProcessPlacedRuntimeError {
    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
        "placed stream diverged from runtime scheduler: {}",
        message.into()
    )))
}
