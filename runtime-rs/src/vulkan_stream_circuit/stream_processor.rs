pub struct VulkanResidentStreamProcessor {
    pub device_id: String,
    pub component_count: usize,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
    pub dynamic_state_capacity_activations: usize,
    _mounted: VulkanMountedPlacedStreamCircuit,
    _transducer_parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    loop_runner: VulkanResidentFeedbackLoopRunner,
    static_state_snapshots: VulkanResidentStateTransactionBank,
    backend_loop_window: usize,
}

impl VulkanResidentStreamProcessor {
    pub fn new(
        device: &VulkanComputeDevice,
        mounted: VulkanMountedPlacedStreamCircuit,
        transducer_parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
        loop_runner: VulkanResidentFeedbackLoopRunner,
    ) -> Result<Self, VulkanError> {
        let backend_loop_window = backend_loop_window_for_static_state_bytes(
            total_static_state_bytes(&mounted.buffers)?,
            loop_runner.sampler.history_capacity_activations,
            backend_loop_transaction_budget_bytes(device),
        );
        let static_state_snapshots =
            VulkanResidentStateTransactionBank::new(device, &mounted.buffers, backend_loop_window)?;
        Ok(Self {
            device_id: loop_runner.device_id.clone(),
            component_count: loop_runner.component_count,
            per_tick_dispatch_count: loop_runner.per_tick_dispatch_count,
            per_tick_descriptor_count: loop_runner.per_tick_descriptor_count,
            per_tick_push_constant_byte_count: loop_runner.per_tick_push_constant_byte_count,
            dynamic_state_capacity_activations: mounted.buffers.dynamic_state_capacity_activations,
            _mounted: mounted,
            _transducer_parameter_buffers: transducer_parameter_buffers,
            loop_runner,
            static_state_snapshots,
            backend_loop_window,
        })
    }

    pub fn run_bounded(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        max_ticks: usize,
    ) -> Result<VulkanResidentFeedbackLoopRun, VulkanResidentFeedbackLoopRunnerError> {
        self.loop_runner.run_bounded(
            device,
            initial_token_id,
            start_stream_tick,
            self.dynamic_state_capacity_activations_u32()?,
            max_ticks,
        )
    }

    pub fn run_prompt_event_bounded(
        &self,
        device: &VulkanComputeDevice,
        prompt_token_ids: &[u32],
        start_stream_tick: u64,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<VulkanResidentPromptEventRun, VulkanResidentFeedbackLoopRunnerError> {
        self.loop_runner.run_prompt_event_bounded(
            device,
            prompt_token_ids,
            start_stream_tick,
            self.dynamic_state_capacity_activations_u32()?,
            max_new_tokens,
            eos_token_id,
        )
    }

    fn dynamic_state_capacity_activations_u32(
        &self,
    ) -> Result<u32, VulkanResidentFeedbackLoopRunnerError> {
        u32::try_from(self.dynamic_state_capacity_activations)
            .map_err(|_| VulkanResidentFeedbackLoopRunnerError::DynamicStateCapacityOverflow)
    }

    pub fn into_running_stream(self, stream_id: impl Into<String>) -> VulkanResidentRunningStream {
        VulkanResidentRunningStream::new(stream_id, self)
    }

    pub fn into_token_stream(self, stream_id: impl Into<String>) -> VulkanResidentTokenStream {
        VulkanResidentTokenStream::new(stream_id, self)
    }

    pub fn backend_loop_window(&self) -> usize {
        self.backend_loop_window
    }
}

fn total_static_state_bytes(
    buffers: &VulkanStreamCircuitStreamBuffers,
) -> Result<usize, VulkanError> {
    buffers
        .state_buffers
        .iter()
        .filter_map(|state| state.static_byte_capacity)
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| VulkanError("static state snapshot size overflowed".to_string()))
}

fn backend_loop_window_for_static_state_bytes(
    static_state_bytes: usize,
    sampler_history_capacity: usize,
    transaction_budget_bytes: usize,
) -> usize {
    let snapshot_limited = transaction_budget_bytes
        .checked_div(static_state_bytes)
        .unwrap_or(VULKAN_BACKEND_LOOP_MAX_WINDOW)
        .max(1);
    VULKAN_BACKEND_LOOP_MAX_WINDOW
        .min(snapshot_limited)
        .min(sampler_history_capacity.max(1))
}

fn backend_loop_transaction_budget_bytes(device: &VulkanComputeDevice) -> usize {
    let device_local_memory_bytes =
        usize::try_from(device.device_local_memory_bytes()).unwrap_or(usize::MAX);
    device_local_memory_bytes
        .checked_div(VULKAN_BACKEND_LOOP_TRANSACTION_HEAP_FRACTION_DIVISOR)
        .unwrap_or(usize::MAX)
        .max(VULKAN_BACKEND_LOOP_MIN_TRANSACTION_BUDGET_BYTES)
}

pub struct VulkanResidentRunningStream {
    pub stream_id: String,
    pub next_stream_tick: u64,
    pub remaining_public_outputs: usize,
    pub eos_token_id: Option<u32>,
    pub stop_token_ids: Vec<u32>,
    pub loop_open: bool,
    pub last_stop_reason: Option<String>,
    processor: VulkanResidentStreamProcessor,
    external_input_queue: VecDeque<VulkanResidentExternalInputSignal>,
    private_feedback_queue: VecDeque<VulkanResidentPrivateFeedbackSignal>,
    public_outputs: Vec<VulkanResidentPublicOutputSignal>,
    private_feedback_history: Vec<VulkanResidentPrivateFeedbackSignal>,
    ticks: Vec<VulkanResidentRunningStreamTick>,
    input_counter: usize,
    public_counter: usize,
    feedback_counter: usize,
}

impl VulkanResidentRunningStream {
    pub fn new(stream_id: impl Into<String>, processor: VulkanResidentStreamProcessor) -> Self {
        Self {
            stream_id: stream_id.into(),
            next_stream_tick: 0,
            remaining_public_outputs: 0,
            eos_token_id: None,
            stop_token_ids: Vec::new(),
            loop_open: false,
            last_stop_reason: None,
            processor,
            external_input_queue: VecDeque::new(),
            private_feedback_queue: VecDeque::new(),
            public_outputs: Vec::new(),
            private_feedback_history: Vec::new(),
            ticks: Vec::new(),
            input_counter: 0,
            public_counter: 0,
            feedback_counter: 0,
        }
    }

    pub fn inject_token(
        &mut self,
        token_id: u32,
        origin: impl Into<String>,
    ) -> VulkanResidentExternalInputSignal {
        let signal = VulkanResidentExternalInputSignal {
            id: format!("input_{}", self.input_counter),
            token_id,
            origin: origin.into(),
        };
        self.input_counter += 1;
        self.external_input_queue.push_back(signal.clone());
        signal
    }

    pub fn inject_prompt(
        &mut self,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<VulkanResidentExternalInputSignal>, VulkanResidentFeedbackLoopRunnerError> {
        self.inject_external_tokens(
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
            "external_input",
        )
    }

    pub fn inject_external_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        origin: impl Into<String>,
    ) -> Result<Vec<VulkanResidentExternalInputSignal>, VulkanResidentFeedbackLoopRunnerError> {
        self.inject_external_tokens_with_stop_tokens(
            token_ids,
            max_new_tokens,
            eos_token_id.into_iter().collect(),
            origin,
        )
    }

    pub fn inject_external_tokens_with_stop_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
        stop_token_ids: Vec<u32>,
        origin: impl Into<String>,
    ) -> Result<Vec<VulkanResidentExternalInputSignal>, VulkanResidentFeedbackLoopRunnerError> {
        if token_ids.is_empty() {
            return Err(VulkanResidentFeedbackLoopRunnerError::EmptyPromptEvent);
        }
        let origin = origin.into();

        self.remaining_public_outputs =
            self.remaining_public_outputs
                .checked_add(max_new_tokens)
                .ok_or(VulkanResidentFeedbackLoopRunnerError::OutputBudgetOverflow)?;
        self.eos_token_id = stop_token_ids.first().copied();
        self.stop_token_ids = stop_token_ids;
        self.loop_open = self.remaining_public_outputs > 0;
        self.last_stop_reason = (max_new_tokens == 0).then(|| "max_new_tokens".to_string());

        Ok(token_ids
            .iter()
            .copied()
            .map(|token_id| self.inject_token(token_id, origin.clone()))
            .collect())
    }

    pub fn continue_loop(
        &mut self,
        additional_public_outputs: usize,
    ) -> Result<(), VulkanResidentFeedbackLoopRunnerError> {
        self.remaining_public_outputs = self
            .remaining_public_outputs
            .checked_add(additional_public_outputs)
            .ok_or(VulkanResidentFeedbackLoopRunnerError::OutputBudgetOverflow)?;
        if self.remaining_public_outputs > 0 {
            self.loop_open = true;
            self.last_stop_reason = None;
        }
        Ok(())
    }

    pub fn interrupt(&mut self, reason: impl Into<String>) -> VulkanResidentStreamControlEvent {
        let reason = reason.into();
        let cleared_private_feedback_ids = self
            .private_feedback_queue
            .iter()
            .map(|signal| signal.id.clone())
            .collect::<Vec<_>>();
        self.private_feedback_queue.clear();
        self.remaining_public_outputs = 0;
        self.loop_open = false;
        self.last_stop_reason = Some(reason.clone());

        VulkanResidentStreamControlEvent {
            event_type: VulkanResidentStreamControlEventType::Interrupt,
            reason,
            cleared_private_feedback_ids,
            closing_private_feedback_id: None,
            state_preserved: true,
        }
    }

    pub fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentStreamControlEvent {
        let reason = reason.into();
        let mut cleared_private_feedback_ids = Vec::new();
        let mut closing_private_feedback_id = None;

        if let Some(mut current) = self.private_feedback_queue.pop_front() {
            closing_private_feedback_id = Some(current.id.clone());
            current.closes_loop_after_processing = true;
            current.stop_reason = Some(reason.clone());
            if let Some(history_signal) = self
                .private_feedback_history
                .iter_mut()
                .find(|signal| signal.id == current.id)
            {
                history_signal.closes_loop_after_processing = true;
                history_signal.stop_reason = Some(reason.clone());
            }
            cleared_private_feedback_ids.extend(
                self.private_feedback_queue
                    .drain(..)
                    .map(|signal| signal.id),
            );
            self.private_feedback_queue.push_front(current);
            self.loop_open = true;
        } else {
            self.loop_open = false;
        }

        self.remaining_public_outputs = 0;
        self.last_stop_reason = Some(reason.clone());

        VulkanResidentStreamControlEvent {
            event_type: VulkanResidentStreamControlEventType::StopAfterCurrent,
            reason,
            cleared_private_feedback_ids,
            closing_private_feedback_id,
            state_preserved: true,
        }
    }

    pub fn tick(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentRunningStreamTick, VulkanResidentFeedbackLoopRunnerError> {
        if self.external_input_queue.is_empty() && self.private_feedback_queue.is_empty() {
            let tick = VulkanResidentRunningStreamTick {
                stream_id: self.stream_id.clone(),
                stream_tick: None,
                status: VulkanResidentRunningStreamTickStatus::Idle,
                input_signal: None,
                tick_run: None,
                public_output: None,
                private_feedback: None,
                sampler_run: None,
                stop_reason: self.last_stop_reason.clone(),
            };
            self.ticks.push(tick.clone());
            return Ok(tick);
        }

        let stream_tick = self.next_stream_tick;
        let input_signal = self
            .next_input_signal()
            .ok_or(VulkanResidentFeedbackLoopRunnerError::MissingPrivateFeedback)?;
        let should_emit_public_output =
            self.remaining_public_outputs > 0 && self.external_input_queue.is_empty();
        let tail_dispatches = if should_emit_public_output {
            self.processor.loop_runner.sampler.resident_dispatches()
        } else {
            &[]
        };
        let tick_run = self
            .processor
            .loop_runner
            .tick_runner
            .run_token_id_with_stream_control_and_tail(
                device,
                input_signal.token_id(),
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations: self
                        .processor
                        .dynamic_state_capacity_activations_u32()?,
                },
                VulkanResidentSingleTokenTickExecution {
                    input_token_is_resident: matches!(
                        &input_signal,
                        VulkanResidentRunningStreamInputSignal::PrivateFeedback(_)
                    ),
                    emit_output: should_emit_public_output,
                    input_tracking_dispatches: self
                        .processor
                        .loop_runner
                        .sampler
                        .input_tracking_dispatches(),
                    tail_dispatches,
                },
            )?;
        self.next_stream_tick = self
            .next_stream_tick
            .checked_add(1)
            .ok_or(VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?;

        let mut public_output = None;
        let mut private_feedback = None;
        let mut sampler_run = None;

        if should_emit_public_output {
            let run = self.processor.loop_runner.sampler.completed_run()?;
            let sampled_token_id = run.token_id;
            self.remaining_public_outputs -= 1;

            let public = VulkanResidentPublicOutputSignal {
                id: format!("public_{}", self.public_counter),
                token_id: sampled_token_id,
                source_stream_tick: stream_tick,
                sampler_run: run.clone(),
            };
            self.public_counter += 1;
            self.public_outputs.push(public.clone());

            let close_after_feedback = if self.stop_token_ids.contains(&sampled_token_id) {
                self.remaining_public_outputs = 0;
                self.last_stop_reason = Some("eos".to_string());
                true
            } else if self.remaining_public_outputs == 0 {
                self.last_stop_reason = Some("max_new_tokens".to_string());
                true
            } else {
                false
            };
            let feedback_depth = input_signal
                .feedback_depth()
                .checked_add(1)
                .ok_or(VulkanResidentFeedbackLoopRunnerError::FeedbackDepthOverflow)?;
            let feedback = VulkanResidentPrivateFeedbackSignal {
                id: format!("feedback_{}", self.feedback_counter),
                token_id: sampled_token_id,
                source_public_output_id: public.id.clone(),
                feedback_depth,
                closes_loop_after_processing: close_after_feedback,
                stop_reason: self
                    .last_stop_reason
                    .clone()
                    .filter(|_| close_after_feedback),
            };
            self.feedback_counter += 1;
            self.private_feedback_queue.push_back(feedback.clone());
            self.private_feedback_history.push(feedback.clone());

            sampler_run = Some(run);
            public_output = Some(public);
            private_feedback = Some(feedback);
        }

        if input_signal.closes_loop_after_processing() {
            self.loop_open = false;
            self.last_stop_reason = input_signal
                .stop_reason()
                .cloned()
                .or_else(|| self.last_stop_reason.clone())
                .or_else(|| Some("max_new_tokens".to_string()));
        }

        let tick = VulkanResidentRunningStreamTick {
            stream_id: self.stream_id.clone(),
            stream_tick: Some(stream_tick),
            status: VulkanResidentRunningStreamTickStatus::Processed,
            input_signal: Some(input_signal),
            tick_run: Some(tick_run),
            public_output,
            private_feedback,
            sampler_run,
            stop_reason: self.last_stop_reason.clone(),
        };
        self.ticks.push(tick.clone());
        Ok(tick)
    }

    fn feedback_cycle_tick_count(&self, max_ticks: usize) -> usize {
        if max_ticks < 2
            || !self.external_input_queue.is_empty()
            || self.private_feedback_queue.len() != 1
            || self.remaining_public_outputs < 2
            || self
                .private_feedback_queue
                .front()
                .is_some_and(|feedback| feedback.closes_loop_after_processing)
        {
            return 0;
        }
        max_ticks
            .min(self.remaining_public_outputs)
            .min(self.processor.static_state_snapshots.cycle_width)
            .min(
                self.processor
                    .loop_runner
                    .sampler
                    .history_capacity_activations,
            )
    }

    fn drive_backend_loop_window(
        &mut self,
        device: &VulkanComputeDevice,
        max_ticks: usize,
    ) -> Result<Vec<VulkanResidentRunningStreamTick>, VulkanResidentFeedbackLoopRunnerError> {
        let tick_count = self.feedback_cycle_tick_count(max_ticks);
        if tick_count < 2 {
            return Ok(vec![self.tick(device)?]);
        }
        let initial_input = self
            .next_input_signal()
            .ok_or(VulkanResidentFeedbackLoopRunnerError::MissingPrivateFeedback)?;
        let initial_token_id = initial_input.token_id();
        let start_stream_tick = self.next_stream_tick;
        let cycle = self.processor.loop_runner.run_resident_feedback_cycle(
            device,
            initial_token_id,
            start_stream_tick,
            tick_count,
            &self.processor._mounted.buffers,
            &self.processor.static_state_snapshots,
        )?;

        let mut input_signal = Some(initial_input);
        let mut feedback_to_queue = None;
        let mut restore_after_tick = None;
        let mut ticks = Vec::with_capacity(tick_count);
        let cycle_tick_count = cycle.tick_runs.len();

        for (tick_index, cycle_tick) in cycle.tick_runs.into_iter().enumerate() {
            let input = input_signal
                .take()
                .ok_or(VulkanResidentFeedbackLoopRunnerError::MissingPrivateFeedback)?;
            self.next_stream_tick = self
                .next_stream_tick
                .checked_add(1)
                .ok_or(VulkanResidentFeedbackLoopRunnerError::StreamTickOverflow)?;

            if input.closes_loop_after_processing() {
                self.loop_open = false;
                self.last_stop_reason = input
                    .stop_reason()
                    .cloned()
                    .or_else(|| self.last_stop_reason.clone())
                    .or_else(|| Some("eos".to_string()));
                let tick = VulkanResidentRunningStreamTick {
                    stream_id: self.stream_id.clone(),
                    stream_tick: Some(cycle_tick.stream_tick),
                    status: VulkanResidentRunningStreamTickStatus::Processed,
                    input_signal: Some(input),
                    tick_run: Some(cycle_tick.tick_run),
                    public_output: None,
                    private_feedback: None,
                    sampler_run: None,
                    stop_reason: self.last_stop_reason.clone(),
                };
                self.ticks.push(tick.clone());
                ticks.push(tick);
                restore_after_tick = Some(tick_index);
                break;
            }

            let run = cycle_tick.sampler_run;
            let sampled_token_id = run.token_id;
            self.remaining_public_outputs -= 1;
            let public = VulkanResidentPublicOutputSignal {
                id: format!("public_{}", self.public_counter),
                token_id: sampled_token_id,
                source_stream_tick: cycle_tick.stream_tick,
                sampler_run: run.clone(),
            };
            self.public_counter += 1;
            self.public_outputs.push(public.clone());

            let sampled_eos = self.stop_token_ids.contains(&sampled_token_id);
            let close_after_feedback = if sampled_eos {
                self.remaining_public_outputs = 0;
                self.last_stop_reason = Some("eos".to_string());
                true
            } else if self.remaining_public_outputs == 0 {
                self.last_stop_reason = Some("max_new_tokens".to_string());
                true
            } else {
                false
            };
            let feedback_depth = input
                .feedback_depth()
                .checked_add(1)
                .ok_or(VulkanResidentFeedbackLoopRunnerError::FeedbackDepthOverflow)?;
            let feedback = VulkanResidentPrivateFeedbackSignal {
                id: format!("feedback_{}", self.feedback_counter),
                token_id: sampled_token_id,
                source_public_output_id: public.id.clone(),
                feedback_depth,
                closes_loop_after_processing: close_after_feedback,
                stop_reason: self
                    .last_stop_reason
                    .clone()
                    .filter(|_| close_after_feedback),
            };
            self.feedback_counter += 1;
            self.private_feedback_history.push(feedback.clone());

            let tick = VulkanResidentRunningStreamTick {
                stream_id: self.stream_id.clone(),
                stream_tick: Some(cycle_tick.stream_tick),
                status: VulkanResidentRunningStreamTickStatus::Processed,
                input_signal: Some(input),
                tick_run: Some(cycle_tick.tick_run),
                public_output: Some(public),
                private_feedback: Some(feedback.clone()),
                sampler_run: Some(run),
                stop_reason: self.last_stop_reason.clone(),
            };
            self.ticks.push(tick.clone());
            ticks.push(tick);

            if close_after_feedback && (!sampled_eos || tick_index + 1 == cycle_tick_count) {
                feedback_to_queue = Some(feedback);
                break;
            }
            input_signal = Some(VulkanResidentRunningStreamInputSignal::PrivateFeedback(
                feedback,
            ));
            if tick_index + 1 == cycle_tick_count {
                let VulkanResidentRunningStreamInputSignal::PrivateFeedback(feedback) =
                    input_signal.take().unwrap()
                else {
                    unreachable!("resident feedback cycle must end with private feedback")
                };
                feedback_to_queue = Some(feedback);
            }
        }

        if let Some(feedback) = feedback_to_queue {
            self.private_feedback_queue.push_back(feedback);
        }
        if let Some(tick_index) = restore_after_tick {
            self.processor
                .static_state_snapshots
                .commit_prefix(&self.processor._mounted.buffers, tick_index + 1)?;
        }
        Ok(ticks)
    }

    pub fn run_until_idle(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<Vec<VulkanResidentRunningStreamTick>, VulkanResidentFeedbackLoopRunnerError> {
        let start = self.ticks.len();
        while !self.external_input_queue.is_empty() || !self.private_feedback_queue.is_empty() {
            self.drive_backend_loop_window(device, self.processor.backend_loop_window)?;
        }
        self.tick(device)?;
        Ok(self.ticks[start..].to_vec())
    }

    pub fn run_prompt(
        &mut self,
        device: &VulkanComputeDevice,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<VulkanResidentRunningStreamRun, VulkanResidentFeedbackLoopRunnerError> {
        let start_public = self.public_outputs.len();
        let start_feedback = self.private_feedback_history.len();
        let start_stream_tick = self.next_stream_tick;
        self.inject_prompt(prompt_token_ids, max_new_tokens, eos_token_id)?;
        let ticks = self.run_until_idle(device)?;
        let public_outputs = self.public_outputs[start_public..].to_vec();
        let private_feedback = self.private_feedback_history[start_feedback..].to_vec();
        let generated_token_ids = public_outputs
            .iter()
            .map(|output| output.token_id)
            .collect::<Vec<_>>();
        let output_token_ids = prompt_token_ids
            .iter()
            .copied()
            .chain(generated_token_ids.iter().copied())
            .collect::<Vec<_>>();

        Ok(VulkanResidentRunningStreamRun {
            stream_id: self.stream_id.clone(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            generated_token_ids,
            output_token_ids,
            stop_reason: self
                .last_stop_reason
                .clone()
                .unwrap_or_else(|| "max_new_tokens".to_string()),
            start_stream_tick,
            next_stream_tick: self.next_stream_tick,
            ticks,
            public_outputs,
            private_feedback,
        })
    }

    pub fn pending_external_input_count(&self) -> usize {
        self.external_input_queue.len()
    }

    pub fn pending_private_feedback_count(&self) -> usize {
        self.private_feedback_queue.len()
    }

    pub fn public_outputs(&self) -> &[VulkanResidentPublicOutputSignal] {
        &self.public_outputs
    }

    pub fn private_feedback_history(&self) -> &[VulkanResidentPrivateFeedbackSignal] {
        &self.private_feedback_history
    }

    pub fn ticks(&self) -> &[VulkanResidentRunningStreamTick] {
        &self.ticks
    }

    fn next_input_signal(&mut self) -> Option<VulkanResidentRunningStreamInputSignal> {
        if let Some(signal) = self.external_input_queue.pop_front() {
            return Some(VulkanResidentRunningStreamInputSignal::External(signal));
        }
        self.private_feedback_queue
            .pop_front()
            .map(VulkanResidentRunningStreamInputSignal::PrivateFeedback)
    }
}
