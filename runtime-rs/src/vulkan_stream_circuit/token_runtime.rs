pub struct VulkanResidentTokenRuntime {
    stream: VulkanResidentTokenStream,
    pending_input_events: VecDeque<VulkanResidentTokenInputEvent>,
}

impl VulkanResidentTokenRuntime {
    pub fn new(stream: VulkanResidentTokenStream) -> Self {
        Self {
            stream,
            pending_input_events: VecDeque::new(),
        }
    }

    pub fn from_processor(
        stream_id: impl Into<String>,
        processor: VulkanResidentStreamProcessor,
    ) -> Self {
        Self::new(processor.into_token_stream(stream_id))
    }

    pub fn stream(&self) -> &VulkanResidentTokenStream {
        &self.stream
    }

    pub fn stream_mut(&mut self) -> &mut VulkanResidentTokenStream {
        &mut self.stream
    }

    pub fn into_stream(self) -> VulkanResidentTokenStream {
        self.stream
    }

    pub fn enqueue_input_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenRuntimeQueuedInputEvent, VulkanResidentFeedbackLoopRunnerError>
    {
        if event.token_ids.is_empty() {
            return Err(VulkanResidentFeedbackLoopRunnerError::EmptyPromptEvent);
        }
        self.pending_input_events.push_back(event.clone());
        Ok(VulkanResidentTokenRuntimeQueuedInputEvent {
            input_event: event,
            pending_input_event_count: self.pending_input_events.len(),
        })
    }

    pub fn run_cycle(
        &mut self,
        device: &VulkanComputeDevice,
        max_ticks: usize,
    ) -> Result<VulkanResidentTokenRuntimeCycleRun, VulkanResidentFeedbackLoopRunnerError> {
        let stream_snapshot = self.stream.snapshot();
        let start_stream_tick = stream_snapshot.next_stream_tick;
        let mut remaining_tick_budget = max_ticks;
        let mut queued_input_events = Vec::new();
        let mut pump_runs = Vec::new();
        let mut output_events = Vec::new();
        let mut processed_tick_count = 0usize;
        let mut idle_tick_count = 0usize;
        let mut ticks_used = 0usize;
        let stop_condition;

        if remaining_tick_budget == 0 {
            return Ok(VulkanResidentTokenRuntimeCycleRun {
                stream_id: self.stream.stream_id().to_string(),
                start_stream_tick,
                next_stream_tick: self.stream.next_stream_tick(),
                max_ticks,
                ticks_used,
                stop_condition: VulkanResidentTokenRuntimeCycleStopCondition::TickBudget,
                queued_input_events,
                pump_runs,
                output_events,
                processed_tick_count,
                idle_tick_count,
                pending_input_event_count: self.pending_input_events.len(),
                stream_idle: self.stream.snapshot().idle,
                last_stop_reason: self.stream.snapshot().last_stop_reason,
            });
        }

        loop {
            if self.stream.snapshot().idle {
                if let Some(event) = self.pending_input_events.pop_front() {
                    queued_input_events.push(self.stream.enqueue_external_event(event)?);
                } else {
                    stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::Idle;
                    break;
                }
            }

            if remaining_tick_budget == 0 {
                stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::TickBudget;
                break;
            }

            let pump_run = self.stream.pump_bounded(device, remaining_tick_budget)?;
            let pump_ticks = pump_run.ticks.len();
            ticks_used += pump_ticks;
            remaining_tick_budget = remaining_tick_budget.saturating_sub(pump_ticks);
            processed_tick_count += pump_run.processed_tick_count;
            idle_tick_count += pump_run.idle_tick_count;
            output_events.extend(pump_run.output_events.iter().cloned());
            let pump_stopped_on_budget =
                pump_run.stop_condition == VulkanResidentTokenStreamPumpStopCondition::TickBudget;
            pump_runs.push(pump_run);

            if pump_stopped_on_budget {
                stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::TickBudget;
                break;
            }
        }

        let end_snapshot = self.stream.snapshot();
        let mut stop_condition = stop_condition;
        if stop_condition == VulkanResidentTokenRuntimeCycleStopCondition::Idle
            && (!end_snapshot.idle || !self.pending_input_events.is_empty())
        {
            stop_condition = VulkanResidentTokenRuntimeCycleStopCondition::TickBudget;
        }

        Ok(VulkanResidentTokenRuntimeCycleRun {
            stream_id: end_snapshot.stream_id,
            start_stream_tick,
            next_stream_tick: end_snapshot.next_stream_tick,
            max_ticks,
            ticks_used,
            stop_condition,
            queued_input_events,
            pump_runs,
            output_events,
            processed_tick_count,
            idle_tick_count,
            pending_input_event_count: self.pending_input_events.len(),
            stream_idle: end_snapshot.idle,
            last_stop_reason: end_snapshot.last_stop_reason,
        })
    }

    pub fn snapshot(&self) -> VulkanResidentTokenRuntimeSnapshot {
        let stream = self.stream.snapshot();
        let idle = stream.idle && self.pending_input_events.is_empty();
        VulkanResidentTokenRuntimeSnapshot {
            stream,
            pending_input_event_count: self.pending_input_events.len(),
            idle,
            running: !idle,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeQueuedInputEvent {
    pub input_event: VulkanResidentTokenInputEvent,
    pub pending_input_event_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenRuntimeCycleStopCondition {
    Idle,
    TickBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeCycleRun {
    pub stream_id: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub max_ticks: usize,
    pub ticks_used: usize,
    pub stop_condition: VulkanResidentTokenRuntimeCycleStopCondition,
    pub queued_input_events: Vec<VulkanResidentTokenQueuedInputEvent>,
    pub pump_runs: Vec<VulkanResidentTokenStreamPumpRun>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub processed_tick_count: usize,
    pub idle_tick_count: usize,
    pub pending_input_event_count: usize,
    pub stream_idle: bool,
    pub last_stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSnapshot {
    pub stream: VulkanResidentTokenStreamSnapshot,
    pub pending_input_event_count: usize,
    pub idle: bool,
    pub running: bool,
}

pub struct VulkanResidentTokenRuntimeScheduler {
    runtimes: BTreeMap<String, VulkanResidentTokenRuntime>,
    active_queue: VecDeque<String>,
}

impl VulkanResidentTokenRuntimeScheduler {
    pub fn new() -> Self {
        Self {
            runtimes: BTreeMap::new(),
            active_queue: VecDeque::new(),
        }
    }

    pub fn add_runtime(
        &mut self,
        runtime: VulkanResidentTokenRuntime,
    ) -> Result<(), VulkanResidentTokenRuntimeSchedulerError> {
        let snapshot = runtime.snapshot();
        let stream_id = snapshot.stream.stream_id.clone();
        if self.runtimes.contains_key(&stream_id) {
            return Err(VulkanResidentTokenRuntimeSchedulerError::DuplicateStream(
                stream_id,
            ));
        }
        let running = snapshot.running;
        self.runtimes.insert(stream_id.clone(), runtime);
        if running {
            self.schedule(&stream_id);
        }
        Ok(())
    }

    pub fn has_runtime(&self, stream_id: &str) -> bool {
        self.runtimes.contains_key(stream_id)
    }

    pub fn runtime(&self, stream_id: &str) -> Option<&VulkanResidentTokenRuntime> {
        self.runtimes.get(stream_id)
    }

    pub fn runtime_mut(&mut self, stream_id: &str) -> Option<&mut VulkanResidentTokenRuntime> {
        self.runtimes.get_mut(stream_id)
    }

    pub fn enqueue_input_event(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenRuntimeQueuedInputEvent, VulkanResidentTokenRuntimeSchedulerError>
    {
        let queued = self
            .runtimes
            .get_mut(stream_id)
            .ok_or_else(|| {
                VulkanResidentTokenRuntimeSchedulerError::UnknownStream(stream_id.to_string())
            })?
            .enqueue_input_event(event)?;
        self.schedule(stream_id);
        Ok(queued)
    }

    pub fn run_cycle(
        &mut self,
        device: &VulkanComputeDevice,
        max_runtime_cycles: usize,
        ticks_per_runtime: usize,
    ) -> Result<VulkanResidentTokenRuntimeSchedulerRun, VulkanResidentTokenRuntimeSchedulerError>
    {
        let mut runtime_cycles = Vec::new();
        let mut output_events = Vec::new();

        if max_runtime_cycles == 0 || ticks_per_runtime == 0 {
            return Ok(VulkanResidentTokenRuntimeSchedulerRun {
                max_runtime_cycles,
                ticks_per_runtime,
                stop_condition: if self.active_queue.is_empty() {
                    VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
                } else {
                    VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
                },
                runtime_cycles,
                output_events,
                active_runtime_count: self.active_queue.len(),
                registered_runtime_count: self.runtimes.len(),
            });
        }

        while runtime_cycles.len() < max_runtime_cycles {
            let Some(stream_id) = self.active_queue.pop_front() else {
                break;
            };
            let cycle = self
                .runtimes
                .get_mut(&stream_id)
                .ok_or_else(|| {
                    VulkanResidentTokenRuntimeSchedulerError::UnknownStream(stream_id.clone())
                })?
                .run_cycle(device, ticks_per_runtime)?;
            output_events.extend(cycle.output_events.iter().cloned().map(|output_event| {
                VulkanResidentTokenRuntimeSchedulerOutputEvent {
                    stream_id: stream_id.clone(),
                    output_event,
                }
            }));
            runtime_cycles.push(cycle);

            if self
                .runtimes
                .get(&stream_id)
                .map(|runtime| runtime.snapshot().running)
                .unwrap_or(false)
            {
                self.schedule(&stream_id);
            }
        }

        let stop_condition = if self.active_queue.is_empty() {
            VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
        } else {
            VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
        };

        Ok(VulkanResidentTokenRuntimeSchedulerRun {
            max_runtime_cycles,
            ticks_per_runtime,
            stop_condition,
            runtime_cycles,
            output_events,
            active_runtime_count: self.active_queue.len(),
            registered_runtime_count: self.runtimes.len(),
        })
    }

    pub fn snapshot(&self) -> VulkanResidentTokenRuntimeSchedulerSnapshot {
        let runtimes = self
            .runtimes
            .values()
            .map(VulkanResidentTokenRuntime::snapshot)
            .collect::<Vec<_>>();
        let running = runtimes.iter().any(|runtime| runtime.running);
        VulkanResidentTokenRuntimeSchedulerSnapshot {
            registered_runtime_count: self.runtimes.len(),
            active_runtime_count: self.active_queue.len(),
            idle: !running,
            running,
            runtimes,
        }
    }

    fn schedule(&mut self, stream_id: &str) {
        if !self.active_queue.iter().any(|active| active == stream_id) {
            self.active_queue.push_back(stream_id.to_string());
        }
    }
}

impl Default for VulkanResidentTokenRuntimeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum VulkanResidentTokenRuntimeSchedulerError {
    DuplicateStream(String),
    UnknownStream(String),
    Runtime(VulkanResidentFeedbackLoopRunnerError),
}

impl Display for VulkanResidentTokenRuntimeSchedulerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStream(stream_id) => {
                write!(
                    f,
                    "resident token runtime stream {stream_id:?} is already registered"
                )
            }
            Self::UnknownStream(stream_id) => {
                write!(
                    f,
                    "resident token runtime stream {stream_id:?} is not registered"
                )
            }
            Self::Runtime(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentTokenRuntimeSchedulerError {}

impl From<VulkanResidentFeedbackLoopRunnerError> for VulkanResidentTokenRuntimeSchedulerError {
    fn from(error: VulkanResidentFeedbackLoopRunnerError) -> Self {
        Self::Runtime(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenRuntimeSchedulerStopCondition {
    Idle,
    RuntimeCycleBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSchedulerOutputEvent {
    pub stream_id: String,
    pub output_event: VulkanResidentTokenOutputEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSchedulerRun {
    pub max_runtime_cycles: usize,
    pub ticks_per_runtime: usize,
    pub stop_condition: VulkanResidentTokenRuntimeSchedulerStopCondition,
    pub runtime_cycles: Vec<VulkanResidentTokenRuntimeCycleRun>,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub active_runtime_count: usize,
    pub registered_runtime_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenRuntimeSchedulerSnapshot {
    pub registered_runtime_count: usize,
    pub active_runtime_count: usize,
    pub idle: bool,
    pub running: bool,
    pub runtimes: Vec<VulkanResidentTokenRuntimeSnapshot>,
}

