pub struct VulkanResidentInProcessPlacedPromptStream {
    package: Arc<VulkanResidentInProcessPlacedModelPackage>,
    processor: VulkanResidentInProcessPlacedStreamProcessor,
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    session: VulkanResidentInProcessPlacedPromptSession,
    transient_state_pages: VulkanResidentTransientStatePageTable,
    active_input_event: Option<VulkanResidentInProcessPlacedActivePromptEvent>,
    pending_input_events: VecDeque<VulkanResidentTokenInputEvent>,
    speculative_draft_tokens: usize,
    resident_feedback_submission_replay: Option<VulkanResidentPlacedFeedbackSubmissionReplay>,
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
            transient_state_pages: VulkanResidentTransientStatePageTable::default(),
            active_input_event: None,
            pending_input_events: VecDeque::new(),
            speculative_draft_tokens: 0,
            resident_feedback_submission_replay: None,
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
        self.resident_feedback_submission_replay = None;
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
            if self.run_speculative_feedback_window_limited_with_output(
                usize::MAX,
                &mut on_output_event,
            )? {
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

    fn run_speculative_feedback_window_limited_with_output<F>(
        &mut self,
        max_public_outputs: usize,
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
            || max_public_outputs < 2
        {
            return Ok(false);
        }
        let draft_token_count = self
            .speculative_draft_tokens
            .min(active.remaining_public_outputs - 1)
            .min(max_public_outputs - 1);
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
        let submission_replay = &mut self.resident_feedback_submission_replay;
        let window_width = processor.resident_feedback_window_width();
        let mut remaining_feedback_ticks = max_feedback_ticks;
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
            let replay_slot = (tick_count == window_width
                && processor
                    .resident_feedback_loop
                    .as_ref()
                    .is_some_and(|feedback_loop| feedback_loop.replayable))
            .then_some(&mut *submission_replay);
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
