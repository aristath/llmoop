#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentExternalInputSignal {
    pub id: String,
    pub token_id: u32,
    pub origin: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentPublicOutputSignal {
    pub id: String,
    pub token_id: u32,
    pub source_stream_tick: u64,
    pub sampler_run: VulkanResidentSamplerRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentPrivateFeedbackSignal {
    pub id: String,
    pub token_id: u32,
    pub source_public_output_id: String,
    pub feedback_depth: u32,
    pub closes_loop_after_processing: bool,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentStreamControlEventType {
    Interrupt,
    StopAfterCurrent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentStreamControlEvent {
    pub event_type: VulkanResidentStreamControlEventType,
    pub reason: String,
    pub cleared_private_feedback_ids: Vec<String>,
    pub closing_private_feedback_id: Option<String>,
    pub state_preserved: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanResidentRunningStreamInputSignal {
    External(VulkanResidentExternalInputSignal),
    PrivateFeedback(VulkanResidentPrivateFeedbackSignal),
}

impl VulkanResidentRunningStreamInputSignal {
    pub fn token_id(&self) -> u32 {
        match self {
            Self::External(signal) => signal.token_id,
            Self::PrivateFeedback(signal) => signal.token_id,
        }
    }

    pub fn route(&self) -> VulkanResidentPromptEventInputRoute {
        match self {
            Self::External(_) => VulkanResidentPromptEventInputRoute::ExternalInput,
            Self::PrivateFeedback(_) => VulkanResidentPromptEventInputRoute::PrivateFeedback,
        }
    }

    pub fn feedback_depth(&self) -> u32 {
        match self {
            Self::External(_) => 0,
            Self::PrivateFeedback(signal) => signal.feedback_depth,
        }
    }

    pub fn closes_loop_after_processing(&self) -> bool {
        match self {
            Self::External(_) => false,
            Self::PrivateFeedback(signal) => signal.closes_loop_after_processing,
        }
    }

    pub fn stop_reason(&self) -> Option<&String> {
        match self {
            Self::External(_) => None,
            Self::PrivateFeedback(signal) => signal.stop_reason.as_ref(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentRunningStreamTickStatus {
    Processed,
    Idle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentRunningStreamTick {
    pub stream_id: String,
    pub stream_tick: Option<u64>,
    pub status: VulkanResidentRunningStreamTickStatus,
    pub input_signal: Option<VulkanResidentRunningStreamInputSignal>,
    pub tick_run: Option<VulkanResidentSingleTokenTickRun>,
    pub public_output: Option<VulkanResidentPublicOutputSignal>,
    pub private_feedback: Option<VulkanResidentPrivateFeedbackSignal>,
    pub sampler_run: Option<VulkanResidentSamplerRun>,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentRunningStreamRun {
    pub stream_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub stop_reason: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub ticks: Vec<VulkanResidentRunningStreamTick>,
    pub public_outputs: Vec<VulkanResidentPublicOutputSignal>,
    pub private_feedback: Vec<VulkanResidentPrivateFeedbackSignal>,
}

pub struct VulkanResidentTokenStream {
    inner: VulkanResidentRunningStream,
    current_input_event_id: Option<String>,
    current_output_index: usize,
}

impl VulkanResidentTokenStream {
    pub fn new(stream_id: impl Into<String>, processor: VulkanResidentStreamProcessor) -> Self {
        Self {
            inner: VulkanResidentRunningStream::new(stream_id, processor),
            current_input_event_id: None,
            current_output_index: 0,
        }
    }

    pub fn from_running_stream(stream: VulkanResidentRunningStream) -> Self {
        Self {
            inner: stream,
            current_input_event_id: None,
            current_output_index: 0,
        }
    }

    pub fn into_inner(self) -> VulkanResidentRunningStream {
        self.inner
    }

    pub fn stream_id(&self) -> &str {
        &self.inner.stream_id
    }

    pub fn next_stream_tick(&self) -> u64 {
        self.inner.next_stream_tick
    }

    pub fn submit_external_event(
        &mut self,
        device: &VulkanComputeDevice,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenStreamRun, VulkanResidentFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let queued = self.enqueue_external_event(event)?;
        let event = queued.input_event;
        let pump = self.pump_until_idle(device)?;

        let generated_token_ids = pump
            .output_events
            .iter()
            .map(|output| output.token_id)
            .collect::<Vec<_>>();

        Ok(VulkanResidentTokenStreamRun {
            stream_id: self.inner.stream_id.clone(),
            input_event: event,
            generated_token_ids,
            output_events: pump.output_events,
            stop_reason: self
                .inner
                .last_stop_reason
                .clone()
                .unwrap_or_else(|| "max_new_tokens".to_string()),
            start_stream_tick,
            next_stream_tick: self.inner.next_stream_tick,
            processed_tick_count: pump.processed_tick_count,
            idle_tick_count: pump.idle_tick_count,
        })
    }

    pub fn enqueue_external_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<VulkanResidentTokenQueuedInputEvent, VulkanResidentFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let enqueued_token_count = event.token_ids.len();
        self.inner.inject_external_tokens_with_stop_tokens(
            &event.token_ids,
            event.max_public_tokens,
            event.stop_token_ids.clone(),
            event.origin.clone(),
        )?;
        self.current_input_event_id = Some(event.id.clone());
        self.current_output_index = 0;

        Ok(VulkanResidentTokenQueuedInputEvent {
            input_event: event,
            start_stream_tick,
            enqueued_token_count,
        })
    }

    pub fn pump_once(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentTokenStreamTick, VulkanResidentFeedbackLoopRunnerError> {
        let tick = self.inner.tick(device)?;
        Ok(self.public_tick_from_running_tick(tick))
    }

    fn public_tick_from_running_tick(
        &mut self,
        tick: VulkanResidentRunningStreamTick,
    ) -> VulkanResidentTokenStreamTick {
        let output_event = tick.public_output.as_ref().map(|output| {
            let output_index = self.current_output_index;
            self.current_output_index += 1;
            VulkanResidentTokenOutputEvent {
                id: output.id.clone(),
                input_event_id: self
                    .current_input_event_id
                    .clone()
                    .unwrap_or_else(|| "feedback_loop".to_string()),
                output_index,
                token_id: output.token_id,
                source_stream_tick: output.source_stream_tick,
            }
        });

        VulkanResidentTokenStreamTick {
            stream_id: tick.stream_id,
            status: tick.status,
            stream_tick: tick.stream_tick,
            input_token_id: tick.input_signal.as_ref().map(|signal| signal.token_id()),
            input_route: tick.input_signal.as_ref().map(|signal| signal.route()),
            output_event,
            stop_reason: tick.stop_reason,
        }
    }

    pub fn pump_bounded(
        &mut self,
        device: &VulkanComputeDevice,
        max_ticks: usize,
    ) -> Result<VulkanResidentTokenStreamPumpRun, VulkanResidentFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let mut ticks = Vec::new();
        let mut output_events = Vec::new();
        let mut processed_tick_count = 0usize;
        let mut idle_tick_count = 0usize;
        let mut stop_condition = VulkanResidentTokenStreamPumpStopCondition::TickBudget;

        while ticks.len() < max_ticks {
            let remaining_ticks = max_ticks - ticks.len();
            let running_ticks = self
                .inner
                .drive_backend_loop_window(device, remaining_ticks)?;
            let mut reached_idle = false;
            for running_tick in running_ticks {
                let tick = self.public_tick_from_running_tick(running_tick);
                match tick.status {
                    VulkanResidentRunningStreamTickStatus::Processed => {
                        processed_tick_count += 1;
                    }
                    VulkanResidentRunningStreamTickStatus::Idle => {
                        idle_tick_count += 1;
                        stop_condition = VulkanResidentTokenStreamPumpStopCondition::Idle;
                    }
                }
                if let Some(output_event) = tick.output_event.clone() {
                    output_events.push(output_event);
                }
                reached_idle = tick.status == VulkanResidentRunningStreamTickStatus::Idle;
                ticks.push(tick);
                if reached_idle || ticks.len() == max_ticks {
                    break;
                }
            }
            if reached_idle {
                break;
            }
        }

        Ok(VulkanResidentTokenStreamPumpRun {
            stream_id: self.inner.stream_id.clone(),
            start_stream_tick,
            next_stream_tick: self.inner.next_stream_tick,
            stop_condition,
            processed_tick_count,
            idle_tick_count,
            output_events,
            ticks,
            last_stop_reason: self.inner.last_stop_reason.clone(),
        })
    }

    pub fn pump_until_idle(
        &mut self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentTokenStreamPumpRun, VulkanResidentFeedbackLoopRunnerError> {
        let start_stream_tick = self.inner.next_stream_tick;
        let mut ticks = Vec::new();
        let mut output_events = Vec::new();
        let mut processed_tick_count = 0usize;
        let mut idle_tick_count = 0usize;

        loop {
            let window = self.inner.processor.backend_loop_window;
            let run = self.pump_bounded(device, window)?;
            processed_tick_count += run.processed_tick_count;
            idle_tick_count += run.idle_tick_count;
            output_events.extend(run.output_events);
            ticks.extend(run.ticks);
            if run.stop_condition == VulkanResidentTokenStreamPumpStopCondition::Idle {
                break;
            }
        }

        Ok(VulkanResidentTokenStreamPumpRun {
            stream_id: self.inner.stream_id.clone(),
            start_stream_tick,
            next_stream_tick: self.inner.next_stream_tick,
            stop_condition: VulkanResidentTokenStreamPumpStopCondition::Idle,
            processed_tick_count,
            idle_tick_count,
            output_events,
            ticks,
            last_stop_reason: self.inner.last_stop_reason.clone(),
        })
    }

    pub fn interrupt(&mut self, reason: impl Into<String>) -> VulkanResidentStreamControlEvent {
        self.inner.interrupt(reason)
    }

    pub fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentStreamControlEvent {
        self.inner.stop_after_current(reason)
    }

    pub fn snapshot(&self) -> VulkanResidentTokenStreamSnapshot {
        VulkanResidentTokenStreamSnapshot {
            stream_id: self.inner.stream_id.clone(),
            next_stream_tick: self.inner.next_stream_tick,
            loop_open: self.inner.loop_open,
            idle: self.inner.external_input_queue.is_empty()
                && self.inner.private_feedback_queue.is_empty(),
            total_public_outputs: self.inner.public_outputs.len(),
            total_ticks: self.inner.ticks.len(),
            last_stop_reason: self.inner.last_stop_reason.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenInputEvent {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_public_tokens: usize,
    pub eos_token_id: Option<u32>,
    pub stop_token_ids: Vec<u32>,
    pub origin: String,
}

impl VulkanResidentTokenInputEvent {
    pub fn new(id: impl Into<String>, token_ids: Vec<u32>, max_public_tokens: usize) -> Self {
        Self {
            id: id.into(),
            token_ids,
            max_public_tokens,
            eos_token_id: None,
            stop_token_ids: Vec::new(),
            origin: "host".to_string(),
        }
    }

    pub fn with_eos_token(mut self, eos_token_id: u32) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self.stop_token_ids = vec![eos_token_id];
        self
    }

    pub fn with_stop_tokens(mut self, stop_token_ids: Vec<u32>) -> Self {
        self.eos_token_id = stop_token_ids.last().copied();
        self.stop_token_ids = stop_token_ids;
        self
    }

    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = origin.into();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenQueuedInputEvent {
    pub input_event: VulkanResidentTokenInputEvent,
    pub start_stream_tick: u64,
    pub enqueued_token_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenOutputEvent {
    pub id: String,
    pub input_event_id: String,
    pub output_index: usize,
    pub token_id: u32,
    pub source_stream_tick: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamTick {
    pub stream_id: String,
    pub status: VulkanResidentRunningStreamTickStatus,
    pub stream_tick: Option<u64>,
    pub input_token_id: Option<u32>,
    pub input_route: Option<VulkanResidentPromptEventInputRoute>,
    pub output_event: Option<VulkanResidentTokenOutputEvent>,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenStreamPumpStopCondition {
    Idle,
    TickBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamPumpRun {
    pub stream_id: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub stop_condition: VulkanResidentTokenStreamPumpStopCondition,
    pub processed_tick_count: usize,
    pub idle_tick_count: usize,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub ticks: Vec<VulkanResidentTokenStreamTick>,
    pub last_stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamRun {
    pub stream_id: String,
    pub input_event: VulkanResidentTokenInputEvent,
    pub generated_token_ids: Vec<u32>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub stop_reason: String,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub processed_tick_count: usize,
    pub idle_tick_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenStreamSnapshot {
    pub stream_id: String,
    pub next_stream_tick: u64,
    pub loop_open: bool,
    pub idle: bool,
    pub total_public_outputs: usize,
    pub total_ticks: usize,
    pub last_stop_reason: Option<String>,
}

