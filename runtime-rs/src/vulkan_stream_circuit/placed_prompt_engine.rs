#[derive(Default)]
pub struct VulkanResidentInProcessPlacedPromptEngine {
    streams: BTreeMap<String, VulkanResidentInProcessPlacedPromptStream>,
    runtime_scheduler: RuntimeStreamScheduler,
}

impl VulkanResidentInProcessPlacedPromptEngine {
    pub fn new() -> Self {
        Self {
            streams: BTreeMap::new(),
            runtime_scheduler: RuntimeStreamScheduler::new(),
        }
    }

    pub fn add_stream(
        &mut self,
        stream_id: impl Into<String>,
        stream: VulkanResidentInProcessPlacedPromptStream,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineStreamSnapshot,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let stream_id = stream_id.into();
        if self.streams.contains_key(&stream_id) {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::DuplicateStream(stream_id));
        }
        let state_declarations = stream
            .package()
            .transient_state_declarations()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?
            .into_iter()
            .map(|declaration| (declaration.key, declaration.block_shape));
        let execution_class_id = stream.package().stream_execution_class_id();
        self.runtime_scheduler
            .add_stream_with_state_declarations_and_execution_class(
                stream_id.clone(),
                execution_class_id,
                state_declarations,
            )?;
        let snapshot = placed_prompt_engine_stream_snapshot(&stream_id, &stream);
        self.streams.insert(stream_id, stream);
        Ok(snapshot)
    }

    pub fn stream(&self, stream_id: &str) -> Option<&VulkanResidentInProcessPlacedPromptStream> {
        self.streams.get(stream_id)
    }

    pub fn enqueue_input_event(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineQueuedInputEvent,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let queued_input_event = {
            let stream = self.streams.get_mut(stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.to_string(),
                }
            })?;
            stream.enqueue_input_event(event)
        };
        self.runtime_scheduler.enqueue_input_event(
            stream_id,
            RuntimeStreamInputEvent::new(
                queued_input_event.input_event.id.clone(),
                queued_input_event.input_event.token_ids.clone(),
                queued_input_event.input_event.max_public_tokens,
            ),
        )?;
        Ok(VulkanResidentInProcessPlacedPromptEngineQueuedInputEvent {
            stream_id: stream_id.to_string(),
            queued_input_event,
        })
    }

    pub fn interrupt_stream(
        &mut self,
        stream_id: &str,
        reason: impl Into<String>,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineControlRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let (stream_control_run, stream_still_active) = {
            let stream = self.streams.get_mut(stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.to_string(),
                }
            })?;
            let stream_control_run = stream.interrupt(reason)?;
            (stream_control_run, !stream.is_idle())
        };
        if !stream_still_active {
            self.runtime_scheduler
                .interrupt_stream(stream_id, "placed prompt stream interrupt")?;
        }
        Ok(VulkanResidentInProcessPlacedPromptEngineControlRun {
            stream_id: stream_id.to_string(),
            stream_control_run,
        })
    }

    pub fn stop_stream_after_current(
        &mut self,
        stream_id: &str,
        reason: impl Into<String>,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineControlRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let (stream_control_run, stream_still_active) = {
            let stream = self.streams.get_mut(stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.to_string(),
                }
            })?;
            let stream_control_run = stream.stop_after_current(reason);
            (stream_control_run, !stream.is_idle())
        };
        if stream_still_active {
            self.runtime_scheduler.close_stream_after_current(stream_id)?;
        }
        Ok(VulkanResidentInProcessPlacedPromptEngineControlRun {
            stream_id: stream_id.to_string(),
            stream_control_run,
        })
    }

    pub fn submit_input_event_until_idle(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        self.submit_input_event_until_idle_with_output(stream_id, event, |_| {})
    }

    pub fn submit_input_event_until_idle_with_output<F>(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
        on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    {
        let input_event_id = event.id.clone();
        let queued_input_event = self.enqueue_input_event(stream_id, event)?;
        let engine_run = self.run_until_idle_bounded_with_output(usize::MAX, on_output_event)?;
        let output_events = engine_run
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
            .collect::<Vec<_>>();

        Ok(VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun {
            stream_id: stream_id.to_string(),
            input_event_id,
            queued_input_event,
            engine_run,
            output_events,
            generated_token_ids,
        })
    }

    pub fn submit_input_events_until_idle_bounded<I>(
        &mut self,
        requests: I,
        max_input_events: usize,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineBatchRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        I: IntoIterator<Item = VulkanResidentInProcessPlacedPromptEngineInputRequest>,
    {
        let mut queued_input_events = Vec::new();
        for request in requests {
            queued_input_events
                .push(self.enqueue_input_event(&request.stream_id, request.input_event)?);
        }
        let engine_run = self.run_until_idle_bounded(max_input_events)?;
        let output_events = engine_run.output_events.clone();
        let generated_token_ids = engine_run.generated_token_ids.clone();

        Ok(VulkanResidentInProcessPlacedPromptEngineBatchRun {
            queued_input_events,
            engine_run,
            output_events,
            generated_token_ids,
        })
    }

    pub fn run_until_idle_bounded(
        &mut self,
        max_input_events: usize,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        self.run_until_idle_bounded_with_output(max_input_events, |_| {})
    }

    pub fn run_until_idle_bounded_with_output<F>(
        &mut self,
        max_input_events: usize,
        mut on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    {
        let start_snapshot = self.snapshot();
        let mut input_runs = Vec::new();
        let mut output_events = Vec::new();
        let scheduler_activation_capacity = self.streams.len().max(1);
        let scheduler_budget = RuntimeStreamSchedulerBudget::new(
            scheduler_activation_capacity,
            VULKAN_BACKEND_LOOP_MAX_WINDOW,
            scheduler_activation_capacity.saturating_mul(VULKAN_BACKEND_LOOP_MAX_WINDOW),
        )
        .with_max_decode_tokens_per_activation(VULKAN_BACKEND_LOOP_MAX_WINDOW);

        while input_runs.len() < max_input_events {
            let scheduler_step = self
                .runtime_scheduler
                .schedule_batch_step(scheduler_budget.clone())?;
            if scheduler_step.batches.is_empty() {
                break;
            }

            for batch in scheduler_step.batches {
                for activation in batch.activations {
                    let stream_id = activation.stream_id.clone();
                    let stream = self.streams.get_mut(&stream_id).ok_or_else(|| {
                        VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                            stream_id: stream_id.clone(),
                        }
                    })?;
                    let callback_stream_id = stream_id.clone();
                    let scheduled_run = stream
                        .run_runtime_scheduler_activation_with_output(&activation, |output_event| {
                            on_output_event(VulkanResidentTokenRuntimeSchedulerOutputEvent {
                                stream_id: callback_stream_id.clone(),
                                output_event,
                            });
                        })?;
                    self.runtime_scheduler
                        .complete_activation(activation.id, scheduled_run.outcome.clone())?;

                    let stream_output_events = placed_prompt_engine_output_events_for(
                        &stream_id,
                        &scheduled_run.output_events,
                    );
                    output_events.extend(stream_output_events.iter().cloned());
                    if let Some(submitted_run) = scheduled_run.completed_input_run {
                        let stream_generated_token_ids = submitted_run.generated_token_ids.clone();
                        input_runs.push(VulkanResidentInProcessPlacedPromptEngineInputRun {
                            stream_id,
                            submitted_run,
                            output_events: stream_output_events,
                            generated_token_ids: stream_generated_token_ids,
                        });
                    }
                }
            }
        }

        let end_snapshot = self.snapshot();
        let generated_token_ids = output_events
            .iter()
            .map(|event| event.output_event.token_id)
            .collect::<Vec<_>>();
        let stop_condition = if end_snapshot.idle {
            VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
        } else {
            VulkanResidentInProcessPlacedPromptEngineRunStopCondition::InputEventBudget
        };

        Ok(VulkanResidentInProcessPlacedPromptEngineRun {
            max_input_events,
            stop_condition,
            processed_input_event_count: input_runs.len(),
            input_runs,
            output_events,
            generated_token_ids,
            start_snapshot,
            end_snapshot,
        })
    }

    pub fn snapshot(&self) -> VulkanResidentInProcessPlacedPromptEngineSnapshot {
        let streams = self
            .streams
            .iter()
            .map(|(stream_id, stream)| placed_prompt_engine_stream_snapshot(stream_id, stream))
            .collect::<Vec<_>>();
        let idle = streams.iter().all(|stream| stream.idle);
        let active_stream_ids = streams
            .iter()
            .filter(|stream| !stream.idle)
            .map(|stream| stream.stream_id.clone())
            .collect::<Vec<_>>();
        VulkanResidentInProcessPlacedPromptEngineSnapshot {
            stream_count: streams.len(),
            active_stream_count: active_stream_ids.len(),
            active_stream_ids,
            idle,
            streams,
        }
    }

}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineInputRequest {
    pub stream_id: String,
    pub input_event: VulkanResidentTokenInputEvent,
}

impl VulkanResidentInProcessPlacedPromptEngineInputRequest {
    pub fn new(stream_id: impl Into<String>, input_event: VulkanResidentTokenInputEvent) -> Self {
        Self {
            stream_id: stream_id.into(),
            input_event,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineQueuedInputEvent {
    pub stream_id: String,
    pub queued_input_event: VulkanResidentInProcessPlacedQueuedInputEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun {
    pub stream_id: String,
    pub input_event_id: String,
    pub queued_input_event: VulkanResidentInProcessPlacedPromptEngineQueuedInputEvent,
    pub engine_run: VulkanResidentInProcessPlacedPromptEngineRun,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineControlRun {
    pub stream_id: String,
    pub stream_control_run: VulkanResidentInProcessPlacedPromptStreamControlRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineBatchRun {
    pub queued_input_events: Vec<VulkanResidentInProcessPlacedPromptEngineQueuedInputEvent>,
    pub engine_run: VulkanResidentInProcessPlacedPromptEngineRun,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineInputRun {
    pub stream_id: String,
    pub submitted_run: VulkanResidentInProcessPlacedSubmittedInputRun,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentInProcessPlacedPromptEngineRunStopCondition {
    Idle,
    InputEventBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineRun {
    pub max_input_events: usize,
    pub stop_condition: VulkanResidentInProcessPlacedPromptEngineRunStopCondition,
    pub processed_input_event_count: usize,
    pub input_runs: Vec<VulkanResidentInProcessPlacedPromptEngineInputRun>,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub start_snapshot: VulkanResidentInProcessPlacedPromptEngineSnapshot,
    pub end_snapshot: VulkanResidentInProcessPlacedPromptEngineSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineSnapshot {
    pub stream_count: usize,
    pub active_stream_count: usize,
    pub active_stream_ids: Vec<String>,
    pub idle: bool,
    pub streams: Vec<VulkanResidentInProcessPlacedPromptEngineStreamSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineStreamSnapshot {
    pub stream_id: String,
    pub input_device_id: String,
    pub output_device_id: String,
    pub device_ids: Vec<String>,
    pub hosted_component_count: usize,
    pub context_window_activations: usize,
    pub pending_input_event_count: usize,
    pub next_stream_tick: u64,
    pub completed_prompt_event_count: usize,
    pub idle: bool,
}

#[derive(Debug)]
pub enum VulkanResidentInProcessPlacedPromptEngineError {
    DuplicateStream(String),
    UnknownStream { stream_id: String },
    Stream(VulkanResidentInProcessPlacedRuntimeError),
    Scheduler(RuntimeStreamSchedulerError),
}

impl Display for VulkanResidentInProcessPlacedPromptEngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStream(stream_id) => {
                write!(
                    f,
                    "placed prompt engine stream {stream_id:?} already exists"
                )
            }
            Self::UnknownStream { stream_id } => {
                write!(
                    f,
                    "placed prompt engine stream {stream_id:?} does not exist"
                )
            }
            Self::Stream(error) => write!(f, "{error}"),
            Self::Scheduler(error) => write!(f, "{error}"),
        }
    }
}

impl Error for VulkanResidentInProcessPlacedPromptEngineError {}

impl From<VulkanResidentInProcessPlacedRuntimeError>
    for VulkanResidentInProcessPlacedPromptEngineError
{
    fn from(error: VulkanResidentInProcessPlacedRuntimeError) -> Self {
        Self::Stream(error)
    }
}

impl From<RuntimeStreamSchedulerError> for VulkanResidentInProcessPlacedPromptEngineError {
    fn from(error: RuntimeStreamSchedulerError) -> Self {
        Self::Scheduler(error)
    }
}

fn placed_prompt_engine_output_events_for(
    stream_id: &str,
    output_events: &[VulkanResidentTokenOutputEvent],
) -> Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent> {
    output_events
        .iter()
        .cloned()
        .map(
            |output_event| VulkanResidentTokenRuntimeSchedulerOutputEvent {
                stream_id: stream_id.to_string(),
                output_event,
            },
        )
        .collect()
}

fn placed_prompt_engine_stream_snapshot(
    stream_id: &str,
    stream: &VulkanResidentInProcessPlacedPromptStream,
) -> VulkanResidentInProcessPlacedPromptEngineStreamSnapshot {
    VulkanResidentInProcessPlacedPromptEngineStreamSnapshot {
        stream_id: stream_id.to_string(),
        input_device_id: stream.package().input_device_id.clone(),
        output_device_id: stream.package().output_device_id.clone(),
        device_ids: stream.package().device_ids.clone(),
        hosted_component_count: stream.package().hosted_component_count,
        context_window_activations: stream.package().dynamic_state_capacity_activations,
        pending_input_event_count: stream.pending_input_event_count(),
        next_stream_tick: stream.next_stream_tick(),
        completed_prompt_event_count: stream.completed_prompt_event_count(),
        idle: stream.is_idle(),
    }
}
