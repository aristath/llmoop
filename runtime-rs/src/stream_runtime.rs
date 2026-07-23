use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::stream_state::{
    TransientStateArena, TransientStateArenaSnapshot, TransientStateBlockShape,
    TransientStateError, TransientStateKey, TransientStateSlot, TransientStateTable,
    TransientStateTableSnapshot,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeStreamStatus {
    Idle,
    Active,
    Interrupted,
    Closing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamInputEvent {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_public_tokens: usize,
}

impl RuntimeStreamInputEvent {
    pub fn new(
        id: impl Into<String>,
        token_ids: impl Into<Vec<u32>>,
        max_public_tokens: usize,
    ) -> Self {
        Self {
            id: id.into(),
            token_ids: token_ids.into(),
            max_public_tokens,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeStreamActivationKind {
    PrefillChunk {
        token_offset: usize,
        token_ids: Vec<u32>,
    },
    DecodeFeedback {
        feedback_depth: usize,
        max_tokens: usize,
    },
}

impl RuntimeStreamActivationKind {
    pub fn work_units(&self) -> usize {
        match self {
            Self::PrefillChunk { token_ids, .. } => token_ids.len(),
            Self::DecodeFeedback { max_tokens, .. } => *max_tokens,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamStateReservation {
    pub key: TransientStateKey,
    pub slots: Vec<TransientStateSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamActivation {
    pub id: u64,
    pub stream_id: String,
    pub execution_class_id: String,
    pub input_event_id: String,
    pub kind: RuntimeStreamActivationKind,
    pub state_reservations: Vec<RuntimeStreamStateReservation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamActivationOutcome {
    pub generated_token_ids: Vec<u32>,
    pub continue_generation: bool,
}

impl RuntimeStreamActivationOutcome {
    pub fn prefill_complete() -> Self {
        Self {
            generated_token_ids: Vec::new(),
            continue_generation: true,
        }
    }

    pub fn generated(token_id: u32, continue_generation: bool) -> Self {
        Self::generated_tokens([token_id], continue_generation)
    }

    pub fn generated_tokens(
        token_ids: impl IntoIterator<Item = u32>,
        continue_generation: bool,
    ) -> Self {
        Self {
            generated_token_ids: token_ids.into_iter().collect(),
            continue_generation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamSchedulerBudget {
    pub max_activations: usize,
    pub max_prefill_tokens_per_activation: usize,
    pub max_decode_tokens_per_activation: usize,
    pub max_work_units: usize,
}

impl RuntimeStreamSchedulerBudget {
    pub fn new(
        max_activations: usize,
        max_prefill_tokens_per_activation: usize,
        max_work_units: usize,
    ) -> Self {
        Self {
            max_activations,
            max_prefill_tokens_per_activation,
            max_decode_tokens_per_activation: 1,
            max_work_units,
        }
    }

    pub fn with_max_decode_tokens_per_activation(mut self, max_tokens: usize) -> Self {
        self.max_decode_tokens_per_activation = max_tokens;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamSchedulerStep {
    pub activations: Vec<RuntimeStreamActivation>,
    pub exhausted_activation_budget: bool,
    pub exhausted_work_budget: bool,
    pub idle: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeStreamActivationBatchKind {
    PrefillChunk {
        execution_class_id: String,
        token_count: usize,
    },
    DecodeFeedback {
        execution_class_id: String,
        max_tokens: usize,
    },
}

impl RuntimeStreamActivationBatchKind {
    pub fn for_activation(activation: &RuntimeStreamActivation) -> Self {
        match &activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { token_ids, .. } => Self::PrefillChunk {
                execution_class_id: activation.execution_class_id.clone(),
                token_count: token_ids.len(),
            },
            RuntimeStreamActivationKind::DecodeFeedback { max_tokens, .. } => {
                Self::DecodeFeedback {
                    execution_class_id: activation.execution_class_id.clone(),
                    max_tokens: *max_tokens,
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamActivationBatch {
    pub kind: RuntimeStreamActivationBatchKind,
    pub activations: Vec<RuntimeStreamActivation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamSchedulerBatchStep {
    pub batches: Vec<RuntimeStreamActivationBatch>,
    pub exhausted_activation_budget: bool,
    pub exhausted_work_budget: bool,
    pub idle: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeStreamSchedulerRunStopCondition {
    Idle,
    StepBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamCompletedActivation {
    pub activation: RuntimeStreamActivation,
    pub outcome: RuntimeStreamActivationOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamBatchActivationOutcome {
    pub activation_id: u64,
    pub outcome: RuntimeStreamActivationOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamSchedulerRun {
    pub stop_condition: RuntimeStreamSchedulerRunStopCondition,
    pub max_steps: usize,
    pub scheduled_steps: usize,
    pub completed_activations: Vec<RuntimeStreamCompletedActivation>,
    pub final_snapshot: RuntimeStreamSchedulerSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamSnapshot {
    pub stream_id: String,
    pub execution_class_id: String,
    pub status: RuntimeStreamStatus,
    pub queued_input_event_count: usize,
    pub current_input_event_id: Option<String>,
    pub in_flight_activation_count: usize,
    pub completed_input_event_count: usize,
    pub scheduled_activation_count: usize,
    pub generated_token_count: usize,
    pub transient_state_entry_count: usize,
    pub transient_state_block_count: usize,
    pub transient_state_logical_activation_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStreamSchedulerSnapshot {
    pub stream_count: usize,
    pub active_stream_count: usize,
    pub in_flight_activation_count: usize,
    pub transient_state_arena: TransientStateArenaSnapshot,
    pub streams: Vec<RuntimeStreamSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStreamSchedulerError(pub String);

impl Display for RuntimeStreamSchedulerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RuntimeStreamSchedulerError {}

impl From<TransientStateError> for RuntimeStreamSchedulerError {
    fn from(error: TransientStateError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeStreamCurrentEvent {
    event: RuntimeStreamInputEvent,
    next_prompt_token_index: usize,
    generated_token_count: usize,
    next_feedback_depth: usize,
}

impl RuntimeStreamCurrentEvent {
    fn new(event: RuntimeStreamInputEvent) -> Self {
        Self {
            event,
            next_prompt_token_index: 0,
            generated_token_count: 0,
            next_feedback_depth: 0,
        }
    }

    fn prompt_done(&self) -> bool {
        self.next_prompt_token_index >= self.event.token_ids.len()
    }

    fn generation_done(&self) -> bool {
        self.generated_token_count >= self.event.max_public_tokens
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeStreamState {
    stream_id: String,
    execution_class_id: String,
    status: RuntimeStreamStatus,
    closing_after_current: bool,
    queued_input_events: VecDeque<RuntimeStreamInputEvent>,
    current_event: Option<RuntimeStreamCurrentEvent>,
    in_flight_activation_ids: Vec<u64>,
    completed_input_event_count: usize,
    scheduled_activation_count: usize,
    generated_token_count: usize,
    transient_state_table: TransientStateTable,
}

impl RuntimeStreamState {
    fn new(
        stream_id: impl Into<String>,
        execution_class_id: impl Into<String>,
    ) -> Result<Self, RuntimeStreamSchedulerError> {
        let stream_id = stream_id.into();
        let execution_class_id = execution_class_id.into();
        Self::with_transient_state_table(
            stream_id.clone(),
            execution_class_id,
            TransientStateTable::new(stream_id)?,
        )
    }

    fn with_transient_state_table(
        stream_id: impl Into<String>,
        execution_class_id: impl Into<String>,
        transient_state_table: TransientStateTable,
    ) -> Result<Self, RuntimeStreamSchedulerError> {
        let stream_id = stream_id.into();
        let execution_class_id = execution_class_id.into();
        if stream_id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "stream id must not be empty".to_string(),
            ));
        }
        if execution_class_id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "stream execution class id must not be empty".to_string(),
            ));
        }
        if transient_state_table.stream_id() != stream_id {
            return Err(RuntimeStreamSchedulerError(format!(
                "transient state table stream id {:?} does not match runtime stream id {:?}",
                transient_state_table.stream_id(),
                stream_id
            )));
        }
        Ok(Self {
            stream_id,
            execution_class_id,
            status: RuntimeStreamStatus::Idle,
            closing_after_current: false,
            queued_input_events: VecDeque::new(),
            current_event: None,
            in_flight_activation_ids: Vec::new(),
            completed_input_event_count: 0,
            scheduled_activation_count: 0,
            generated_token_count: 0,
            transient_state_table,
        })
    }

    fn has_in_flight_work(&self) -> bool {
        !self.in_flight_activation_ids.is_empty()
    }

    fn has_schedulable_running_work(&self) -> bool {
        self.current_event.is_some() && !self.has_in_flight_work()
    }

    fn has_schedulable_waiting_work(&self) -> bool {
        self.current_event.is_none()
            && !self.queued_input_events.is_empty()
            && !self.has_in_flight_work()
            && self.status != RuntimeStreamStatus::Closing
            && self.status != RuntimeStreamStatus::Interrupted
    }

    fn has_pending_work(&self) -> bool {
        self.current_event.is_some()
            || !self.queued_input_events.is_empty()
            || self.has_in_flight_work()
    }

    fn refresh_status(&mut self) {
        if self.status == RuntimeStreamStatus::Interrupted {
            return;
        }
        if self.closing_after_current {
            self.status = RuntimeStreamStatus::Closing;
        } else if self.has_pending_work() {
            self.status = RuntimeStreamStatus::Active;
        } else {
            self.status = RuntimeStreamStatus::Idle;
        }
    }

    fn snapshot(&self) -> RuntimeStreamSnapshot {
        let transient_state = self.transient_state_table.snapshot();
        RuntimeStreamSnapshot {
            stream_id: self.stream_id.clone(),
            execution_class_id: self.execution_class_id.clone(),
            status: self.status,
            queued_input_event_count: self.queued_input_events.len(),
            current_input_event_id: self
                .current_event
                .as_ref()
                .map(|event| event.event.id.clone()),
            in_flight_activation_count: self.in_flight_activation_ids.len(),
            completed_input_event_count: self.completed_input_event_count,
            scheduled_activation_count: self.scheduled_activation_count,
            generated_token_count: self.generated_token_count,
            transient_state_entry_count: transient_state.entry_count,
            transient_state_block_count: transient_state.block_count,
            transient_state_logical_activation_count: transient_state.logical_activation_count,
        }
    }
}

#[derive(Default)]
pub struct RuntimeStreamScheduler {
    streams: BTreeMap<String, RuntimeStreamState>,
    active_queue: VecDeque<String>,
    in_flight: BTreeMap<u64, RuntimeStreamActivation>,
    transient_state_arena: TransientStateArena,
    next_activation_id: u64,
}

impl RuntimeStreamScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_stream_with_execution_class(
        &mut self,
        stream_id: impl Into<String>,
        execution_class_id: impl Into<String>,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        self.add_stream_with_state_declarations_and_execution_class(
            stream_id,
            execution_class_id,
            [],
        )
    }

    pub fn add_stream(
        &mut self,
        stream_id: impl Into<String>,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        self.add_stream_with_execution_class(stream_id, "default")
    }

    pub fn add_stream_with_state_declarations<I>(
        &mut self,
        stream_id: impl Into<String>,
        state_declarations: I,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError>
    where
        I: IntoIterator<Item = (TransientStateKey, TransientStateBlockShape)>,
    {
        self.add_stream_with_state_declarations_and_execution_class(
            stream_id,
            "default",
            state_declarations,
        )
    }

    pub fn add_stream_with_state_declarations_and_execution_class<I>(
        &mut self,
        stream_id: impl Into<String>,
        execution_class_id: impl Into<String>,
        state_declarations: I,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError>
    where
        I: IntoIterator<Item = (TransientStateKey, TransientStateBlockShape)>,
    {
        let stream_id = stream_id.into();
        let execution_class_id = execution_class_id.into();
        if stream_id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "stream id must not be empty".to_string(),
            ));
        }
        if execution_class_id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "stream execution class id must not be empty".to_string(),
            ));
        }
        if self.streams.contains_key(&stream_id) {
            return Err(RuntimeStreamSchedulerError(format!(
                "stream {stream_id:?} already exists"
            )));
        }
        let mut stream = RuntimeStreamState::new(stream_id.clone(), execution_class_id)?;
        for (key, shape) in state_declarations {
            stream.transient_state_table.declare_state(key, shape)?;
        }
        let snapshot = stream.snapshot();
        self.streams.insert(stream_id, stream);
        Ok(snapshot)
    }

    pub fn declare_stream_state(
        &mut self,
        stream_id: &str,
        key: TransientStateKey,
        shape: TransientStateBlockShape,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        let stream = self.stream_mut(stream_id)?;
        stream.transient_state_table.declare_state(key, shape)?;
        Ok(stream.snapshot())
    }

    pub fn transient_state_arena_snapshot(
        &self,
    ) -> Result<TransientStateArenaSnapshot, RuntimeStreamSchedulerError> {
        Ok(self.transient_state_arena.snapshot()?)
    }

    pub fn stream_transient_state_snapshot(
        &self,
        stream_id: &str,
    ) -> Result<TransientStateTableSnapshot, RuntimeStreamSchedulerError> {
        Ok(self.stream(stream_id)?.transient_state_table.snapshot())
    }

    pub fn fork_stream_transient_state(
        &mut self,
        source_stream_id: &str,
        target_stream_id: impl Into<String>,
        execution_class_id: impl Into<String>,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        let target_stream_id = target_stream_id.into();
        let execution_class_id = execution_class_id.into();
        if target_stream_id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "forked stream id must not be empty".to_string(),
            ));
        }
        if execution_class_id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "forked stream execution class id must not be empty".to_string(),
            ));
        }
        if self.streams.contains_key(&target_stream_id) {
            return Err(RuntimeStreamSchedulerError(format!(
                "stream {target_stream_id:?} already exists"
            )));
        }

        let source_table = self.stream(source_stream_id)?.transient_state_table.clone();
        let forked_table = source_table.fork(&mut self.transient_state_arena, &target_stream_id)?;
        let stream = RuntimeStreamState::with_transient_state_table(
            target_stream_id.clone(),
            execution_class_id,
            forked_table,
        )?;
        let snapshot = stream.snapshot();
        self.streams.insert(target_stream_id, stream);
        Ok(snapshot)
    }

    pub fn share_stream_state(
        &mut self,
        target_stream_id: &str,
        source_stream_id: &str,
        key: &TransientStateKey,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        let source_table = self.stream(source_stream_id)?.transient_state_table.clone();
        let arena = &mut self.transient_state_arena;
        let target = self.streams.get_mut(target_stream_id).ok_or_else(|| {
            RuntimeStreamSchedulerError(format!("unknown stream {target_stream_id:?}"))
        })?;
        target
            .transient_state_table
            .share_state_from(arena, &source_table, key)?;
        Ok(target.snapshot())
    }

    pub fn enqueue_input_event(
        &mut self,
        stream_id: &str,
        event: RuntimeStreamInputEvent,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        if event.id.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "input event id must not be empty".to_string(),
            ));
        }
        if event.token_ids.is_empty() {
            return Err(RuntimeStreamSchedulerError(format!(
                "input event {} has no tokens",
                event.id
            )));
        }
        let stream = self.stream_mut(stream_id)?;
        if stream.status == RuntimeStreamStatus::Closing {
            return Err(RuntimeStreamSchedulerError(format!(
                "stream {stream_id:?} is closing"
            )));
        }
        if stream.status == RuntimeStreamStatus::Interrupted {
            return Err(RuntimeStreamSchedulerError(format!(
                "stream {stream_id:?} is interrupted"
            )));
        }
        stream.queued_input_events.push_back(event);
        stream.refresh_status();
        let snapshot = stream.snapshot();
        self.activate_stream(stream_id);
        Ok(snapshot)
    }

    pub fn interrupt_stream(
        &mut self,
        stream_id: &str,
        reason: impl Into<String>,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        let reason = reason.into();
        if reason.is_empty() {
            return Err(RuntimeStreamSchedulerError(
                "interrupt reason must not be empty".to_string(),
            ));
        }
        let in_flight_ids = {
            let stream = self.stream_mut(stream_id)?;
            stream.queued_input_events.clear();
            stream.current_event = None;
            stream.in_flight_activation_ids.clone()
        };
        for activation_id in in_flight_ids {
            self.in_flight.remove(&activation_id);
        }
        let snapshot = {
            let arena = &mut self.transient_state_arena;
            let stream = self.streams.get_mut(stream_id).ok_or_else(|| {
                RuntimeStreamSchedulerError(format!("unknown stream {stream_id:?}"))
            })?;
            stream.in_flight_activation_ids.clear();
            stream.transient_state_table.reset_all(arena)?;
            stream.status = RuntimeStreamStatus::Interrupted;
            stream.snapshot()
        };
        self.active_queue.retain(|candidate| candidate != stream_id);
        Ok(snapshot)
    }

    pub fn close_stream_after_current(
        &mut self,
        stream_id: &str,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        let stream = self.stream_mut(stream_id)?;
        stream.closing_after_current = true;
        stream.queued_input_events.clear();
        stream.refresh_status();
        if stream.has_pending_work() {
            self.activate_stream(stream_id);
        }
        Ok(self.stream(stream_id)?.snapshot())
    }

    pub fn schedule_step(
        &mut self,
        budget: RuntimeStreamSchedulerBudget,
    ) -> Result<RuntimeStreamSchedulerStep, RuntimeStreamSchedulerError> {
        if budget.max_activations == 0 || budget.max_work_units == 0 {
            return Ok(RuntimeStreamSchedulerStep {
                activations: Vec::new(),
                exhausted_activation_budget: budget.max_activations == 0,
                exhausted_work_budget: budget.max_work_units == 0,
                idle: self.active_stream_count() == 0,
            });
        }
        if budget.max_prefill_tokens_per_activation == 0 {
            return Err(RuntimeStreamSchedulerError(
                "max_prefill_tokens_per_activation must be positive".to_string(),
            ));
        }
        if budget.max_decode_tokens_per_activation == 0 {
            return Err(RuntimeStreamSchedulerError(
                "max_decode_tokens_per_activation must be positive".to_string(),
            ));
        }

        self.refresh_active_queue();
        let mut activations = Vec::new();
        let mut consumed_work_units = 0usize;

        while activations.len() < budget.max_activations
            && consumed_work_units < budget.max_work_units
        {
            let Some(stream_id) = self.next_schedulable_stream_id() else {
                break;
            };
            let remaining_work_units = budget.max_work_units - consumed_work_units;
            let Some(activation) = self.prepare_activation(
                &stream_id,
                budget
                    .max_prefill_tokens_per_activation
                    .min(remaining_work_units),
                budget
                    .max_decode_tokens_per_activation
                    .min(remaining_work_units),
            )?
            else {
                continue;
            };
            consumed_work_units = consumed_work_units.saturating_add(activation.kind.work_units());
            activations.push(activation);
        }

        Ok(RuntimeStreamSchedulerStep {
            exhausted_activation_budget: activations.len() == budget.max_activations,
            exhausted_work_budget: consumed_work_units == budget.max_work_units,
            idle: self.active_stream_count() == 0 && activations.is_empty(),
            activations,
        })
    }

    pub fn schedule_batch_step(
        &mut self,
        budget: RuntimeStreamSchedulerBudget,
    ) -> Result<RuntimeStreamSchedulerBatchStep, RuntimeStreamSchedulerError> {
        let step = self.schedule_step(budget)?;
        Ok(RuntimeStreamSchedulerBatchStep {
            batches: group_compatible_stream_activations(step.activations),
            exhausted_activation_budget: step.exhausted_activation_budget,
            exhausted_work_budget: step.exhausted_work_budget,
            idle: step.idle,
        })
    }

    pub fn run_until_idle_with<F>(
        &mut self,
        budget: RuntimeStreamSchedulerBudget,
        max_steps: usize,
        mut execute_activation: F,
    ) -> Result<RuntimeStreamSchedulerRun, RuntimeStreamSchedulerError>
    where
        F: FnMut(
            &RuntimeStreamActivation,
        ) -> Result<RuntimeStreamActivationOutcome, RuntimeStreamSchedulerError>,
    {
        let mut completed_activations = Vec::new();
        let mut scheduled_steps = 0usize;

        while scheduled_steps < max_steps {
            let step = self.schedule_step(budget.clone())?;
            if step.activations.is_empty() && step.idle {
                return Ok(RuntimeStreamSchedulerRun {
                    stop_condition: RuntimeStreamSchedulerRunStopCondition::Idle,
                    max_steps,
                    scheduled_steps,
                    completed_activations,
                    final_snapshot: self.snapshot(),
                });
            }
            if step.activations.is_empty() {
                break;
            }
            scheduled_steps = scheduled_steps.saturating_add(1);
            for activation in step.activations {
                let outcome = execute_activation(&activation)?;
                self.complete_activation(activation.id, outcome.clone())?;
                completed_activations.push(RuntimeStreamCompletedActivation {
                    activation,
                    outcome,
                });
            }
        }

        let stop_condition = if self.snapshot().active_stream_count == 0
            && self.snapshot().in_flight_activation_count == 0
        {
            RuntimeStreamSchedulerRunStopCondition::Idle
        } else {
            RuntimeStreamSchedulerRunStopCondition::StepBudget
        };
        Ok(RuntimeStreamSchedulerRun {
            stop_condition,
            max_steps,
            scheduled_steps,
            completed_activations,
            final_snapshot: self.snapshot(),
        })
    }

    pub fn run_batches_until_idle_with<F>(
        &mut self,
        budget: RuntimeStreamSchedulerBudget,
        max_steps: usize,
        mut execute_batch: F,
    ) -> Result<RuntimeStreamSchedulerRun, RuntimeStreamSchedulerError>
    where
        F: FnMut(
            &RuntimeStreamActivationBatch,
        )
            -> Result<Vec<RuntimeStreamBatchActivationOutcome>, RuntimeStreamSchedulerError>,
    {
        let mut completed_activations = Vec::new();
        let mut scheduled_steps = 0usize;

        while scheduled_steps < max_steps {
            let step = self.schedule_batch_step(budget.clone())?;
            if step.batches.is_empty() && step.idle {
                return Ok(RuntimeStreamSchedulerRun {
                    stop_condition: RuntimeStreamSchedulerRunStopCondition::Idle,
                    max_steps,
                    scheduled_steps,
                    completed_activations,
                    final_snapshot: self.snapshot(),
                });
            }
            if step.batches.is_empty() {
                break;
            }
            scheduled_steps = scheduled_steps.saturating_add(1);
            for batch in step.batches {
                let mut outcomes = execute_batch(&batch)?;
                for activation in batch.activations {
                    let outcome_index = outcomes
                        .iter()
                        .position(|outcome| outcome.activation_id == activation.id)
                        .ok_or_else(|| {
                            RuntimeStreamSchedulerError(format!(
                                "batch executor did not return an outcome for activation {}",
                                activation.id
                            ))
                        })?;
                    let outcome = outcomes.remove(outcome_index).outcome;
                    self.complete_activation(activation.id, outcome.clone())?;
                    completed_activations.push(RuntimeStreamCompletedActivation {
                        activation,
                        outcome,
                    });
                }
                if !outcomes.is_empty() {
                    return Err(RuntimeStreamSchedulerError(
                        "batch executor returned outcomes for unknown activations".to_string(),
                    ));
                }
            }
        }

        let snapshot = self.snapshot();
        let stop_condition =
            if snapshot.active_stream_count == 0 && snapshot.in_flight_activation_count == 0 {
                RuntimeStreamSchedulerRunStopCondition::Idle
            } else {
                RuntimeStreamSchedulerRunStopCondition::StepBudget
            };
        Ok(RuntimeStreamSchedulerRun {
            stop_condition,
            max_steps,
            scheduled_steps,
            completed_activations,
            final_snapshot: snapshot,
        })
    }

    pub fn complete_activation(
        &mut self,
        activation_id: u64,
        outcome: RuntimeStreamActivationOutcome,
    ) -> Result<RuntimeStreamSnapshot, RuntimeStreamSchedulerError> {
        let activation = self.in_flight.remove(&activation_id).ok_or_else(|| {
            RuntimeStreamSchedulerError(format!("unknown in-flight activation {activation_id}"))
        })?;
        let stream_id = activation.stream_id.clone();
        let stream = self.stream_mut(&stream_id)?;
        stream
            .in_flight_activation_ids
            .retain(|candidate| *candidate != activation_id);
        let current = stream.current_event.as_mut().ok_or_else(|| {
            RuntimeStreamSchedulerError(format!(
                "stream {stream_id:?} has no current event for activation {activation_id}"
            ))
        })?;

        match activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { token_ids, .. } => {
                current.next_prompt_token_index = current
                    .next_prompt_token_index
                    .saturating_add(token_ids.len());
                if !outcome.generated_token_ids.is_empty() {
                    if !current.prompt_done() {
                        return Err(RuntimeStreamSchedulerError(format!(
                            "prefill activation {activation_id} emitted public tokens before the prompt was complete"
                        )));
                    }
                    let generated_token_count = outcome.generated_token_ids.len();
                    let new_generated_token_count = current
                        .generated_token_count
                        .checked_add(generated_token_count)
                        .ok_or_else(|| {
                            RuntimeStreamSchedulerError(format!(
                                "stream {stream_id:?} generated token count overflowed"
                            ))
                        })?;
                    if new_generated_token_count > current.event.max_public_tokens {
                        return Err(RuntimeStreamSchedulerError(format!(
                            "prefill activation {activation_id} exceeded input event public output budget"
                        )));
                    }
                    let new_feedback_depth = current
                        .next_feedback_depth
                        .checked_add(generated_token_count)
                        .ok_or_else(|| {
                            RuntimeStreamSchedulerError(format!(
                                "stream {stream_id:?} feedback depth overflowed"
                            ))
                        })?;
                    stream.generated_token_count = stream
                        .generated_token_count
                        .checked_add(generated_token_count)
                        .ok_or_else(|| {
                            RuntimeStreamSchedulerError(format!(
                                "stream {stream_id:?} total generated token count overflowed"
                            ))
                        })?;
                    current.generated_token_count = new_generated_token_count;
                    current.next_feedback_depth = new_feedback_depth;
                }
                if !outcome.continue_generation {
                    current.generated_token_count = current.event.max_public_tokens;
                }
            }
            RuntimeStreamActivationKind::DecodeFeedback { max_tokens, .. } => {
                if outcome.generated_token_ids.is_empty() {
                    return Err(RuntimeStreamSchedulerError(format!(
                        "decode activation {activation_id} completed without generated tokens"
                    )));
                }
                if outcome.generated_token_ids.len() > max_tokens {
                    return Err(RuntimeStreamSchedulerError(format!(
                        "decode activation {activation_id} generated {} tokens, exceeding its window of {max_tokens}",
                        outcome.generated_token_ids.len()
                    )));
                }
                let generated_token_count = outcome.generated_token_ids.len();
                let new_generated_token_count = current
                    .generated_token_count
                    .checked_add(generated_token_count)
                    .ok_or_else(|| {
                        RuntimeStreamSchedulerError(format!(
                            "stream {stream_id:?} generated token count overflowed"
                        ))
                    })?;
                if new_generated_token_count > current.event.max_public_tokens {
                    return Err(RuntimeStreamSchedulerError(format!(
                        "decode activation {activation_id} exceeded input event public output budget"
                    )));
                }
                let new_feedback_depth = current
                    .next_feedback_depth
                    .checked_add(generated_token_count)
                    .ok_or_else(|| {
                        RuntimeStreamSchedulerError(format!(
                            "stream {stream_id:?} feedback depth overflowed"
                        ))
                    })?;
                stream.generated_token_count = stream
                    .generated_token_count
                    .checked_add(generated_token_count)
                    .ok_or_else(|| {
                        RuntimeStreamSchedulerError(format!(
                            "stream {stream_id:?} total generated token count overflowed"
                        ))
                    })?;
                current.generated_token_count = new_generated_token_count;
                current.next_feedback_depth = new_feedback_depth;
                if !outcome.continue_generation {
                    current.generated_token_count = current.event.max_public_tokens;
                }
            }
        }

        if current.prompt_done() && current.generation_done() {
            stream.current_event = None;
            stream.completed_input_event_count =
                stream.completed_input_event_count.saturating_add(1);
            if stream.closing_after_current {
                stream.closing_after_current = false;
                stream.status = RuntimeStreamStatus::Idle;
            }
        }
        stream.refresh_status();
        let snapshot = stream.snapshot();
        if stream.has_schedulable_running_work() || stream.has_schedulable_waiting_work() {
            self.activate_stream(&stream_id);
        }
        Ok(snapshot)
    }

    pub fn snapshot(&self) -> RuntimeStreamSchedulerSnapshot {
        RuntimeStreamSchedulerSnapshot {
            stream_count: self.streams.len(),
            active_stream_count: self.active_stream_count(),
            in_flight_activation_count: self.in_flight.len(),
            transient_state_arena: self
                .transient_state_arena
                .snapshot()
                .expect("validated transient state block shapes remain snapshot-safe"),
            streams: self
                .streams
                .values()
                .map(RuntimeStreamState::snapshot)
                .collect(),
        }
    }

    fn prepare_activation(
        &mut self,
        stream_id: &str,
        max_prefill_tokens: usize,
        max_decode_tokens: usize,
    ) -> Result<Option<RuntimeStreamActivation>, RuntimeStreamSchedulerError> {
        let activation_id = self.next_activation_id;
        self.next_activation_id = self
            .next_activation_id
            .checked_add(1)
            .ok_or_else(|| RuntimeStreamSchedulerError("activation id overflow".to_string()))?;

        let (execution_class_id, input_event_id, kind, state_keys) = {
            let stream = self.stream_mut(stream_id)?;
            if stream.has_in_flight_work() {
                return Ok(None);
            }
            if stream.current_event.is_none() {
                let Some(event) = stream.queued_input_events.pop_front() else {
                    stream.refresh_status();
                    return Ok(None);
                };
                stream.current_event = Some(RuntimeStreamCurrentEvent::new(event));
            }
            let current = stream
                .current_event
                .as_ref()
                .expect("current event was just installed");
            let input_event_id = current.event.id.clone();
            let kind = if !current.prompt_done() {
                let token_offset = current.next_prompt_token_index;
                let token_limit =
                    (token_offset + max_prefill_tokens).min(current.event.token_ids.len());
                RuntimeStreamActivationKind::PrefillChunk {
                    token_offset,
                    token_ids: current.event.token_ids[token_offset..token_limit].to_vec(),
                }
            } else if !current.generation_done() {
                let max_tokens = max_decode_tokens.min(
                    current
                        .event
                        .max_public_tokens
                        .saturating_sub(current.generated_token_count),
                );
                RuntimeStreamActivationKind::DecodeFeedback {
                    feedback_depth: current.next_feedback_depth,
                    max_tokens,
                }
            } else {
                return Ok(None);
            };
            (
                stream.execution_class_id.clone(),
                input_event_id,
                kind,
                stream.transient_state_table.state_keys(),
            )
        };
        let state_reservations =
            self.reserve_activation_state(stream_id, &state_keys, kind.work_units())?;
        let activation = RuntimeStreamActivation {
            id: activation_id,
            stream_id: stream_id.to_string(),
            execution_class_id,
            input_event_id,
            kind,
            state_reservations,
        };
        let stream = self.stream_mut(stream_id)?;
        stream.in_flight_activation_ids.push(activation_id);
        stream.scheduled_activation_count = stream.scheduled_activation_count.saturating_add(1);
        stream.refresh_status();
        self.in_flight.insert(activation_id, activation.clone());
        Ok(Some(activation))
    }

    fn reserve_activation_state(
        &mut self,
        stream_id: &str,
        state_keys: &[TransientStateKey],
        work_units: usize,
    ) -> Result<Vec<RuntimeStreamStateReservation>, RuntimeStreamSchedulerError> {
        if state_keys.is_empty() || work_units == 0 {
            return Ok(Vec::new());
        }
        let arena = &mut self.transient_state_arena;
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| RuntimeStreamSchedulerError(format!("unknown stream {stream_id:?}")))?;
        state_keys
            .iter()
            .map(|key| {
                Ok(RuntimeStreamStateReservation {
                    key: key.clone(),
                    slots: stream
                        .transient_state_table
                        .append_activations(arena, key, work_units)?,
                })
            })
            .collect()
    }

    fn next_schedulable_stream_id(&mut self) -> Option<String> {
        self.refresh_active_queue();
        let running_index = self.active_queue.iter().position(|stream_id| {
            self.streams
                .get(stream_id)
                .is_some_and(RuntimeStreamState::has_schedulable_running_work)
        });
        if let Some(index) = running_index {
            return self.active_queue.remove(index);
        }
        let waiting_index = self.active_queue.iter().position(|stream_id| {
            self.streams
                .get(stream_id)
                .is_some_and(RuntimeStreamState::has_schedulable_waiting_work)
        });
        waiting_index.and_then(|index| self.active_queue.remove(index))
    }

    fn refresh_active_queue(&mut self) {
        let active_ids = self
            .streams
            .iter_mut()
            .filter_map(|(stream_id, stream)| {
                stream.refresh_status();
                (stream.has_schedulable_running_work() || stream.has_schedulable_waiting_work())
                    .then_some(stream_id.clone())
            })
            .collect::<Vec<_>>();
        self.active_queue
            .retain(|stream_id| active_ids.iter().any(|active| active == stream_id));
        for stream_id in active_ids {
            self.activate_stream(&stream_id);
        }
    }

    fn activate_stream(&mut self, stream_id: &str) {
        if !self
            .active_queue
            .iter()
            .any(|candidate| candidate == stream_id)
        {
            self.active_queue.push_back(stream_id.to_string());
        }
    }

    fn active_stream_count(&self) -> usize {
        self.streams
            .values()
            .filter(|stream| stream.status == RuntimeStreamStatus::Active)
            .count()
    }

    fn stream(&self, stream_id: &str) -> Result<&RuntimeStreamState, RuntimeStreamSchedulerError> {
        self.streams
            .get(stream_id)
            .ok_or_else(|| RuntimeStreamSchedulerError(format!("unknown stream {stream_id:?}")))
    }

    fn stream_mut(
        &mut self,
        stream_id: &str,
    ) -> Result<&mut RuntimeStreamState, RuntimeStreamSchedulerError> {
        self.streams
            .get_mut(stream_id)
            .ok_or_else(|| RuntimeStreamSchedulerError(format!("unknown stream {stream_id:?}")))
    }
}

fn group_compatible_stream_activations(
    activations: Vec<RuntimeStreamActivation>,
) -> Vec<RuntimeStreamActivationBatch> {
    let mut batches: Vec<RuntimeStreamActivationBatch> = Vec::new();
    for activation in activations {
        let kind = RuntimeStreamActivationBatchKind::for_activation(&activation);
        if let Some(batch) = batches.iter_mut().find(|batch| batch.kind == kind) {
            batch.activations.push(activation);
        } else {
            batches.push(RuntimeStreamActivationBatch {
                kind,
                activations: vec![activation],
            });
        }
    }
    batches
}

#[cfg(test)]
#[path = "stream_runtime_tests.rs"]
mod stream_runtime_tests;
