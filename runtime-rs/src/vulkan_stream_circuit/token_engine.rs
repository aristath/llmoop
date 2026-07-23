pub struct VulkanResidentTokenEngine {
    scheduler: VulkanResidentTokenRuntimeScheduler,
    streams: BTreeMap<String, VulkanResidentTokenEngineStream>,
    models: BTreeMap<String, Arc<dyn VulkanResidentTokenModelPackage>>,
    device: VulkanComputeDevice,
}

pub trait VulkanResidentTokenModelPackage {
    fn device_id(&self) -> &str;

    fn dynamic_state_capacity_activations(&self) -> usize;

    fn permanent_parameter_count(&self) -> usize;

    fn permanent_parameter_bytes(&self) -> usize;

    fn transducer_parameter_count(&self) -> usize;

    fn transducer_parameter_bytes(&self) -> usize;

    fn reusable_kernel_word_count(&self) -> usize;

    fn create_stream_processor(
        &self,
        device: &VulkanComputeDevice,
        random_seed: u32,
    ) -> Result<VulkanResidentStreamProcessor, VulkanResidentTokenModelPackageError>;

    fn snapshot(
        &self,
        model_id: String,
        registered_stream_count: usize,
    ) -> VulkanResidentTokenEngineModelSnapshot {
        VulkanResidentTokenEngineModelSnapshot {
            model_id,
            device_id: self.device_id().to_string(),
            dynamic_state_capacity_activations: self.dynamic_state_capacity_activations(),
            permanent_parameter_count: self.permanent_parameter_count(),
            permanent_parameter_bytes: self.permanent_parameter_bytes(),
            transducer_parameter_count: self.transducer_parameter_count(),
            transducer_parameter_bytes: self.transducer_parameter_bytes(),
            reusable_kernel_word_count: self.reusable_kernel_word_count(),
            registered_stream_count,
        }
    }
}

impl VulkanResidentTokenEngine {
    pub fn new(device: VulkanComputeDevice) -> Self {
        Self {
            scheduler: VulkanResidentTokenRuntimeScheduler::new(),
            streams: BTreeMap::new(),
            models: BTreeMap::new(),
            device,
        }
    }

    pub fn from_processor(
        device: VulkanComputeDevice,
        stream_id: impl Into<String>,
        processor: VulkanResidentStreamProcessor,
    ) -> Result<Self, VulkanResidentTokenEngineError> {
        let mut engine = Self::new(device);
        engine.add_processor(stream_id, processor)?;
        Ok(engine)
    }

    pub fn add_processor(
        &mut self,
        stream_id: impl Into<String>,
        processor: VulkanResidentStreamProcessor,
    ) -> Result<VulkanResidentTokenEngineStream, VulkanResidentTokenEngineError> {
        self.add_processor_with_residency(
            stream_id,
            processor,
            None,
            VulkanResidentTokenEngineStreamResidency::OwnedProcessor,
        )
    }

    pub fn add_model_package<M>(
        &mut self,
        model_id: impl Into<String>,
        model: M,
    ) -> Result<VulkanResidentTokenEngineModelSnapshot, VulkanResidentTokenEngineError>
    where
        M: VulkanResidentTokenModelPackage + 'static,
    {
        self.add_model_package_arc(model_id, Arc::new(model))
    }

    fn add_model_package_arc(
        &mut self,
        model_id: impl Into<String>,
        model: Arc<dyn VulkanResidentTokenModelPackage>,
    ) -> Result<VulkanResidentTokenEngineModelSnapshot, VulkanResidentTokenEngineError> {
        let model_id = model_id.into();
        if self.models.contains_key(&model_id) {
            return Err(VulkanResidentTokenEngineError::DuplicateModel(model_id));
        }
        let snapshot = model.snapshot(model_id.clone(), 0);
        self.models.insert(model_id, model);
        Ok(snapshot)
    }

    pub fn create_stream_from_model(
        &mut self,
        model_id: &str,
        stream_id: impl Into<String>,
        random_seed: u32,
    ) -> Result<VulkanResidentTokenEngineStream, VulkanResidentTokenEngineError> {
        let model = self
            .models
            .get(model_id)
            .ok_or_else(|| VulkanResidentTokenEngineError::UnknownModel(model_id.to_string()))?
            .clone();
        let processor = model.create_stream_processor(&self.device, random_seed)?;
        self.add_processor_with_residency(
            stream_id,
            processor,
            Some(model_id.to_string()),
            VulkanResidentTokenEngineStreamResidency::SharedModel,
        )
    }

    fn add_processor_with_residency(
        &mut self,
        stream_id: impl Into<String>,
        processor: VulkanResidentStreamProcessor,
        model_id: Option<String>,
        residency: VulkanResidentTokenEngineStreamResidency,
    ) -> Result<VulkanResidentTokenEngineStream, VulkanResidentTokenEngineError> {
        let stream_id = stream_id.into();
        let stream = VulkanResidentTokenEngineStream {
            stream_id: stream_id.clone(),
            model_id,
            residency,
            device_id: processor.device_id.clone(),
            pedal_count: processor.pedal_count,
            per_tick_dispatch_count: processor.per_tick_dispatch_count,
            per_tick_descriptor_count: processor.per_tick_descriptor_count,
            per_tick_push_constant_byte_count: processor.per_tick_push_constant_byte_count,
            dynamic_state_capacity_activations: processor.dynamic_state_capacity_activations,
        };
        let runtime = VulkanResidentTokenRuntime::from_processor(&stream_id, processor);
        self.scheduler.add_runtime(runtime)?;
        self.streams.insert(stream_id, stream.clone());
        Ok(stream)
    }

    pub fn device(&self) -> &VulkanComputeDevice {
        &self.device
    }

    pub fn scheduler(&self) -> &VulkanResidentTokenRuntimeScheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut VulkanResidentTokenRuntimeScheduler {
        &mut self.scheduler
    }

    pub fn stream(&self, stream_id: &str) -> Option<&VulkanResidentTokenEngineStream> {
        self.streams.get(stream_id)
    }

    pub fn runtime_snapshot(&self, stream_id: &str) -> Option<VulkanResidentTokenRuntimeSnapshot> {
        self.scheduler
            .runtime(stream_id)
            .map(VulkanResidentTokenRuntime::snapshot)
    }

    pub fn enqueue_input_event(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenRuntimeQueuedInputEvent, VulkanResidentTokenEngineError> {
        Ok(self.scheduler.enqueue_input_event(stream_id, event)?)
    }

    pub fn run_cycle(
        &mut self,
        max_runtime_cycles: usize,
        ticks_per_runtime: usize,
    ) -> Result<VulkanResidentTokenRuntimeSchedulerRun, VulkanResidentTokenEngineError> {
        Ok(self
            .scheduler
            .run_cycle(&self.device, max_runtime_cycles, ticks_per_runtime)?)
    }

    pub fn run_until_idle(
        &mut self,
        max_scheduler_turns: usize,
        max_runtime_cycles_per_turn: usize,
        ticks_per_runtime: usize,
    ) -> Result<VulkanResidentTokenEngineRun, VulkanResidentTokenEngineError> {
        self.run_until_idle_with_budget(VulkanResidentTokenEngineRunBudget {
            max_scheduler_turns,
            max_runtime_cycles_per_turn,
            ticks_per_runtime,
        })
    }

    pub fn run_until_idle_with_budget(
        &mut self,
        budget: VulkanResidentTokenEngineRunBudget,
    ) -> Result<VulkanResidentTokenEngineRun, VulkanResidentTokenEngineError> {
        let start_snapshot = self.snapshot();
        let mut scheduler_runs = Vec::new();
        let mut output_events = Vec::new();
        let mut runtime_cycle_count = 0usize;

        if budget.max_scheduler_turns == 0
            || budget.max_runtime_cycles_per_turn == 0
            || budget.ticks_per_runtime == 0
        {
            let end_snapshot = self.snapshot();
            let stop_condition = if end_snapshot.scheduler.running {
                VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget
            } else {
                VulkanResidentTokenEngineRunStopCondition::Idle
            };
            return Ok(VulkanResidentTokenEngineRun {
                max_scheduler_turns: budget.max_scheduler_turns,
                max_runtime_cycles_per_turn: budget.max_runtime_cycles_per_turn,
                ticks_per_runtime: budget.ticks_per_runtime,
                stop_condition,
                scheduler_runs,
                output_events,
                runtime_cycle_count,
                start_snapshot,
                end_snapshot,
            });
        }

        while scheduler_runs.len() < budget.max_scheduler_turns && self.snapshot().scheduler.running
        {
            let scheduler_run =
                self.run_cycle(budget.max_runtime_cycles_per_turn, budget.ticks_per_runtime)?;
            let produced_runtime_cycles = scheduler_run.runtime_cycles.len();
            runtime_cycle_count = runtime_cycle_count
                .checked_add(produced_runtime_cycles)
                .ok_or(VulkanResidentTokenEngineError::RunCycleCountOverflow)?;
            output_events.extend(scheduler_run.output_events.iter().cloned());
            let still_running = self.snapshot().scheduler.running;
            if produced_runtime_cycles == 0 && still_running {
                return Err(VulkanResidentTokenEngineError::RunStalled);
            }
            scheduler_runs.push(scheduler_run);
        }

        let end_snapshot = self.snapshot();
        let stop_condition = if end_snapshot.scheduler.running {
            VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget
        } else {
            VulkanResidentTokenEngineRunStopCondition::Idle
        };

        Ok(VulkanResidentTokenEngineRun {
            max_scheduler_turns: budget.max_scheduler_turns,
            max_runtime_cycles_per_turn: budget.max_runtime_cycles_per_turn,
            ticks_per_runtime: budget.ticks_per_runtime,
            stop_condition,
            scheduler_runs,
            output_events,
            runtime_cycle_count,
            start_snapshot,
            end_snapshot,
        })
    }

    pub fn submit_input_event_until_idle(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
        budget: VulkanResidentTokenEngineRunBudget,
    ) -> Result<VulkanResidentTokenEngineSubmittedInputRun, VulkanResidentTokenEngineError> {
        let input_event_id = event.id.clone();
        let queued_input_event = self.enqueue_input_event(stream_id, event)?;
        let run = self.run_until_idle_with_budget(budget)?;
        let output_events = run
            .output_events
            .iter()
            .filter(|event| {
                event.stream_id == stream_id && event.output_event.input_event_id == input_event_id
            })
            .cloned()
            .collect::<Vec<_>>();
        let generated_token_ids = output_events
            .iter()
            .map(|event| event.output_event.token_id)
            .collect();

        Ok(VulkanResidentTokenEngineSubmittedInputRun {
            stream_id: stream_id.to_string(),
            input_event_id,
            queued_input_event,
            run,
            output_events,
            generated_token_ids,
        })
    }

    pub fn submit_tokens_until_idle(
        &mut self,
        stream_id: &str,
        input_event_id: impl Into<String>,
        token_ids: Vec<u32>,
        max_public_tokens: usize,
        origin: impl Into<String>,
        budget: VulkanResidentTokenEngineRunBudget,
    ) -> Result<VulkanResidentTokenEngineSubmittedInputRun, VulkanResidentTokenEngineError> {
        self.submit_input_event_until_idle(
            stream_id,
            VulkanResidentTokenInputEvent::new(input_event_id, token_ids, max_public_tokens)
                .with_origin(origin),
            budget,
        )
    }

    pub fn submit_text_until_idle<C>(
        &mut self,
        request: VulkanResidentTokenEngineTextInputRequest,
        budget: VulkanResidentTokenEngineRunBudget,
        codec: &C,
    ) -> Result<VulkanResidentTokenEngineSubmittedTextRun, VulkanResidentTokenEngineError>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let VulkanResidentTokenEngineTextInputRequest {
            stream_id,
            input_event_id,
            input_text,
            max_public_tokens,
            origin,
        } = request;
        let encoded_token_ids = codec.encode_text(&input_text)?;
        let submitted_tokens = self.submit_tokens_until_idle(
            &stream_id,
            input_event_id.clone(),
            encoded_token_ids.clone(),
            max_public_tokens,
            origin,
            budget,
        )?;
        let generated_text = codec.decode_tokens(&submitted_tokens.generated_token_ids)?;

        Ok(VulkanResidentTokenEngineSubmittedTextRun {
            stream_id,
            input_event_id,
            input_text,
            encoded_token_ids,
            generated_text,
            submitted_tokens,
        })
    }

    pub fn submit_live_text_turn_until_idle<C>(
        &mut self,
        request: VulkanResidentTokenEngineTextInputRequest,
        budget: VulkanResidentTokenEngineRunBudget,
        codec: &C,
    ) -> Result<VulkanResidentTokenEngineLiveTextTurnRun, VulkanResidentTokenEngineError>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let VulkanResidentTokenEngineTextInputRequest {
            stream_id,
            input_event_id,
            input_text,
            max_public_tokens,
            origin,
        } = request;
        let queued_input_event = self.enqueue_text_input_event(
            &stream_id,
            input_event_id.clone(),
            input_text,
            max_public_tokens,
            origin,
            codec,
        )?;
        let mut cycles = Vec::new();
        let mut output_events = Vec::new();
        let mut runtime_cycle_count = 0usize;

        if budget.max_scheduler_turns != 0
            && budget.max_runtime_cycles_per_turn != 0
            && budget.ticks_per_runtime != 0
        {
            while cycles.len() < budget.max_scheduler_turns && self.snapshot().scheduler.running {
                let cycle = self.run_text_cycle(
                    budget.max_runtime_cycles_per_turn,
                    budget.ticks_per_runtime,
                    codec,
                )?;
                let produced_runtime_cycles = cycle.scheduler_run.runtime_cycles.len();
                runtime_cycle_count = runtime_cycle_count
                    .checked_add(produced_runtime_cycles)
                    .ok_or(VulkanResidentTokenEngineError::RunCycleCountOverflow)?;
                output_events.extend(
                    cycle
                        .output_events
                        .iter()
                        .filter(|event| {
                            event.stream_id == stream_id && event.input_event_id == input_event_id
                        })
                        .cloned(),
                );
                let still_running = self.snapshot().scheduler.running;
                if produced_runtime_cycles == 0 && still_running {
                    return Err(VulkanResidentTokenEngineError::RunStalled);
                }
                cycles.push(cycle);
            }
        }

        let generated_token_ids = output_events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>();
        let generated_text = codec.decode_tokens(&generated_token_ids)?;
        let mut output_token_ids = queued_input_event.encoded_token_ids.clone();
        output_token_ids.extend(generated_token_ids.iter().copied());
        let output_text = codec.decode_tokens(&output_token_ids)?;
        let stop_condition = if self.snapshot().scheduler.running {
            VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget
        } else {
            VulkanResidentTokenEngineRunStopCondition::Idle
        };

        Ok(VulkanResidentTokenEngineLiveTextTurnRun {
            stream_id,
            input_event_id,
            queued_input_event,
            cycles,
            output_events,
            generated_token_ids,
            generated_text,
            output_text,
            stop_condition,
            runtime_cycle_count,
        })
    }

    pub fn submit_live_text_batch_until_idle<C>(
        &mut self,
        requests: Vec<VulkanResidentTokenEngineTextInputRequest>,
        budget: VulkanResidentTokenEngineRunBudget,
        codec: &C,
    ) -> Result<VulkanResidentTokenEngineLiveTextBatchRun, VulkanResidentTokenEngineError>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let mut queued_input_events = Vec::with_capacity(requests.len());
        for request in requests {
            queued_input_events.push(self.enqueue_text_input_event(
                &request.stream_id,
                request.input_event_id,
                request.input_text,
                request.max_public_tokens,
                request.origin,
                codec,
            )?);
        }

        let mut cycles = Vec::new();
        let mut output_events = Vec::new();
        let mut runtime_cycle_count = 0usize;

        if budget.max_scheduler_turns != 0
            && budget.max_runtime_cycles_per_turn != 0
            && budget.ticks_per_runtime != 0
        {
            while cycles.len() < budget.max_scheduler_turns && self.snapshot().scheduler.running {
                let cycle = self.run_text_cycle(
                    budget.max_runtime_cycles_per_turn,
                    budget.ticks_per_runtime,
                    codec,
                )?;
                let produced_runtime_cycles = cycle.scheduler_run.runtime_cycles.len();
                runtime_cycle_count = runtime_cycle_count
                    .checked_add(produced_runtime_cycles)
                    .ok_or(VulkanResidentTokenEngineError::RunCycleCountOverflow)?;
                output_events.extend(cycle.output_events.iter().cloned());
                let still_running = self.snapshot().scheduler.running;
                if produced_runtime_cycles == 0 && still_running {
                    return Err(VulkanResidentTokenEngineError::RunStalled);
                }
                cycles.push(cycle);
            }
        }

        let generated_token_ids = output_events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>();
        let generated_text = codec.decode_tokens(&generated_token_ids)?;
        let stop_condition = if self.snapshot().scheduler.running {
            VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget
        } else {
            VulkanResidentTokenEngineRunStopCondition::Idle
        };

        Ok(VulkanResidentTokenEngineLiveTextBatchRun {
            queued_input_events,
            cycles,
            output_events,
            generated_token_ids,
            generated_text,
            stop_condition,
            runtime_cycle_count,
        })
    }

    pub fn enqueue_text_input_event<C>(
        &mut self,
        stream_id: &str,
        input_event_id: impl Into<String>,
        input_text: impl Into<String>,
        max_public_tokens: usize,
        origin: impl Into<String>,
        codec: &C,
    ) -> Result<VulkanResidentTokenEngineQueuedTextInputEvent, VulkanResidentTokenEngineError>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let input_event_id = input_event_id.into();
        let input_text = input_text.into();
        let encoded_token_ids = codec.encode_text(&input_text)?;
        let queued_input_event = self.enqueue_input_event(
            stream_id,
            VulkanResidentTokenInputEvent::new(
                input_event_id.clone(),
                encoded_token_ids.clone(),
                max_public_tokens,
            )
            .with_origin(origin),
        )?;

        Ok(VulkanResidentTokenEngineQueuedTextInputEvent {
            stream_id: stream_id.to_string(),
            input_event_id,
            input_text,
            encoded_token_ids,
            queued_input_event,
        })
    }

    pub fn run_text_cycle<C>(
        &mut self,
        max_runtime_cycles: usize,
        ticks_per_runtime: usize,
        codec: &C,
    ) -> Result<VulkanResidentTokenEngineTextCycleRun, VulkanResidentTokenEngineError>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let scheduler_run = self.run_cycle(max_runtime_cycles, ticks_per_runtime)?;
        let output_events = scheduler_run
            .output_events
            .iter()
            .map(|event| {
                let token_id = event.output_event.token_id;
                Ok(VulkanResidentTokenEngineTextOutputEvent {
                    stream_id: event.stream_id.clone(),
                    input_event_id: event.output_event.input_event_id.clone(),
                    output_index: event.output_event.output_index,
                    token_id,
                    text: codec.decode_tokens(&[token_id])?,
                    source_stream_tick: event.output_event.source_stream_tick,
                })
            })
            .collect::<Result<Vec<_>, VulkanResidentTokenEngineError>>()?;
        let generated_token_ids = output_events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>();
        let generated_text = codec.decode_tokens(&generated_token_ids)?;

        Ok(VulkanResidentTokenEngineTextCycleRun {
            scheduler_run,
            output_events,
            generated_token_ids,
            generated_text,
        })
    }

    pub fn snapshot(&self) -> VulkanResidentTokenEngineSnapshot {
        let mut stream_counts_by_model = BTreeMap::new();
        for stream in self.streams.values() {
            if let Some(model_id) = &stream.model_id {
                *stream_counts_by_model
                    .entry(model_id.clone())
                    .or_insert(0usize) += 1;
            }
        }
        let models = self
            .models
            .iter()
            .map(|(model_id, model)| {
                model.snapshot(
                    model_id.clone(),
                    stream_counts_by_model.get(model_id).copied().unwrap_or(0),
                )
            })
            .collect();
        VulkanResidentTokenEngineSnapshot {
            device_name: self.device.device_name().to_string(),
            scheduler: self.scheduler.snapshot(),
            streams: self.streams.values().cloned().collect(),
            models,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineStream {
    pub stream_id: String,
    pub model_id: Option<String>,
    pub residency: VulkanResidentTokenEngineStreamResidency,
    pub device_id: String,
    pub pedal_count: usize,
    pub per_tick_dispatch_count: usize,
    pub per_tick_descriptor_count: usize,
    pub per_tick_push_constant_byte_count: u32,
    pub dynamic_state_capacity_activations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenEngineStreamResidency {
    OwnedProcessor,
    SharedModel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineModelSnapshot {
    pub model_id: String,
    pub device_id: String,
    pub dynamic_state_capacity_activations: usize,
    pub permanent_parameter_count: usize,
    pub permanent_parameter_bytes: usize,
    pub transducer_parameter_count: usize,
    pub transducer_parameter_bytes: usize,
    pub reusable_kernel_word_count: usize,
    pub registered_stream_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineSnapshot {
    pub device_name: String,
    pub scheduler: VulkanResidentTokenRuntimeSchedulerSnapshot,
    pub streams: Vec<VulkanResidentTokenEngineStream>,
    pub models: Vec<VulkanResidentTokenEngineModelSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenModelPackageError {
    message: String,
}

impl VulkanResidentTokenModelPackageError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for VulkanResidentTokenModelPackageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for VulkanResidentTokenModelPackageError {}

