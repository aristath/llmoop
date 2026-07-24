const VULKAN_PENDING_ACTIVATION_CONTROL_WAIT_NS: u64 = 10_000_000;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanResidentInProcessPlacedPromptEngineStreamHistory {
    committed_state_token_ids: Vec<u32>,
    pending_feedback_token_ids: VecDeque<u32>,
}

pub struct VulkanResidentInProcessPlacedPromptEngine {
    streams: BTreeMap<String, VulkanResidentInProcessPlacedPromptStream>,
    runtime_scheduler: RuntimeStreamScheduler,
    stream_histories:
        BTreeMap<String, VulkanResidentInProcessPlacedPromptEngineStreamHistory>,
    resident_prefix_state_cache: VulkanResidentPlacedPrefixStateCache,
    latest_prefix_checkpoint_by_stream: BTreeMap<String, RuntimePrefixStateCacheKey>,
    multi_stream_batch_runners:
        BTreeMap<VulkanResidentInProcessPlacedPromptEngineBatchKey, VulkanResidentPlacedMultiStreamBatchRunner>,
    pending_wait_group_cursor: usize,
}

impl Default for VulkanResidentInProcessPlacedPromptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VulkanResidentInProcessPlacedPromptEngine {
    fn drop(&mut self) {
        // Batch runners own descriptor sets that remain bound to buffers owned
        // by the streams. Destroy those descriptors before stream teardown.
        self.multi_stream_batch_runners.clear();
    }
}

impl VulkanResidentInProcessPlacedPromptEngine {
    pub fn new() -> Self {
        Self {
            streams: BTreeMap::new(),
            runtime_scheduler: RuntimeStreamScheduler::with_prefix_state_cache_capacity(1),
            stream_histories: BTreeMap::new(),
            resident_prefix_state_cache: VulkanResidentPlacedPrefixStateCache::default(),
            latest_prefix_checkpoint_by_stream: BTreeMap::new(),
            multi_stream_batch_runners: BTreeMap::new(),
            pending_wait_group_cursor: 0,
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
        let evicted = self
            .runtime_scheduler
            .set_prefix_state_cache_capacity(self.streams.len().saturating_add(1))?;
        self.resident_prefix_state_cache.evict_keys(&evicted);
        self.latest_prefix_checkpoint_by_stream
            .retain(|_, key| !evicted.contains(key));
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
        self.stream_histories.insert(stream_id.clone(), Default::default());
        self.streams.insert(stream_id, stream);
        Ok(snapshot)
    }

    pub fn stream(&self, stream_id: &str) -> Option<&VulkanResidentInProcessPlacedPromptStream> {
        self.streams.get(stream_id)
    }

    pub fn fork_stream(
        &mut self,
        source_stream_id: &str,
        target_stream_id: impl Into<String>,
        random_seed: u32,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineStreamSnapshot,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let target_stream_id = target_stream_id.into();
        if self.streams.contains_key(&target_stream_id) {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::DuplicateStream(
                target_stream_id,
            ));
        }
        let source = self.streams.get(source_stream_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: source_stream_id.to_string(),
            }
        })?;
        let forked = source.fork_preserving_state(random_seed)?;
        let source_history = self
            .stream_histories
            .get(source_stream_id)
            .cloned()
            .ok_or_else(|| VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: source_stream_id.to_string(),
            })?;
        let evicted = self
            .runtime_scheduler
            .set_prefix_state_cache_capacity(self.streams.len().saturating_add(1))?;
        self.resident_prefix_state_cache.evict_keys(&evicted);
        self.latest_prefix_checkpoint_by_stream
            .retain(|_, key| !evicted.contains(key));
        let execution_class_id = forked.package().stream_execution_class_id();
        self.runtime_scheduler.fork_stream_transient_state(
            source_stream_id,
            &target_stream_id,
            execution_class_id,
        )?;
        let snapshot = placed_prompt_engine_stream_snapshot(&target_stream_id, &forked);
        self.stream_histories
            .insert(target_stream_id.clone(), source_history);
        self.streams.insert(target_stream_id, forked);
        Ok(snapshot)
    }

    pub fn reset_stream_transient_state(
        &mut self,
        stream_id: &str,
    ) -> Result<usize, VulkanResidentInProcessPlacedPromptEngineError> {
        let stream = self.streams.get_mut(stream_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.to_string(),
            }
        })?;
        let zeroed = stream.reset_transient_state()?;
        self.runtime_scheduler
            .reset_stream_transient_state(stream_id)?;
        self.stream_histories.insert(stream_id.to_string(), Default::default());
        Ok(zeroed)
    }

    pub fn remove_stream(
        &mut self,
        stream_id: &str,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineStreamSnapshot,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let stream = self.streams.get(stream_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.to_string(),
            }
        })?;
        if !stream.is_idle() || stream.pending_scheduler_activation.is_some() {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(
                    "cannot remove a placed prompt stream while work is pending",
                ),
            ));
        }
        self.multi_stream_batch_runners
            .retain(|key, _| !key.stream_ids.iter().any(|candidate| candidate == stream_id));
        self.runtime_scheduler.remove_stream(stream_id)?;
        let stream = self
            .streams
            .remove(stream_id)
            .expect("validated placed prompt stream exists");
        self.stream_histories.remove(stream_id);
        self.latest_prefix_checkpoint_by_stream.remove(stream_id);
        let evicted = self
            .runtime_scheduler
            .set_prefix_state_cache_capacity(self.streams.len())?;
        self.resident_prefix_state_cache.evict_keys(&evicted);
        self.latest_prefix_checkpoint_by_stream
            .retain(|_, key| !evicted.contains(key));
        Ok(placed_prompt_engine_stream_snapshot(stream_id, &stream))
    }

    pub fn enqueue_input_event(
        &mut self,
        stream_id: &str,
        mut event: VulkanResidentTokenInputEvent,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineQueuedInputEvent,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let original_token_count = event.token_ids.len();
        let mut reused_prefix_token_count = 0usize;
        let can_restore_prefix = self
            .streams
            .get(stream_id)
            .is_some_and(|stream| stream.is_idle() && stream.next_stream_tick() == 0)
            && self
                .stream_histories
                .get(stream_id)
                .is_some_and(|history| history.committed_state_token_ids.is_empty())
            && event.token_ids.len() > 1;
        if can_restore_prefix {
            let stream = self.streams.get(stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.to_string(),
                }
            })?;
            let state_keys = stream
                .package()
                .transient_state_declarations()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?
                .into_iter()
                .map(|declaration| declaration.key)
                .collect::<Vec<_>>();
            let runtime_graph_id = stream.package().runtime_execution_identity.clone();
            let runtime_modifier_bytes = prefix_cache_runtime_modifier_bytes(stream)?;
            let restorable_token_ids = &event.token_ids[..event.token_ids.len() - 1];
            let restored_key = self.runtime_scheduler.restore_longest_stream_prefix_state(
                stream_id,
                runtime_graph_id,
                restorable_token_ids,
                &runtime_modifier_bytes,
                state_keys,
            )?;
            if let Some(key) = restored_key {
                let restore_result = self.resident_prefix_state_cache.restore(
                    &key,
                    self.streams.get_mut(stream_id).expect("stream was validated"),
                );
                if let Err(error) = restore_result {
                    self.runtime_scheduler
                        .reset_stream_transient_state(stream_id)?;
                    self.streams
                        .get_mut(stream_id)
                        .expect("stream was validated")
                        .reset_transient_state()?;
                    return Err(error.into());
                }
                reused_prefix_token_count = key.token_count;
                self.stream_histories
                    .get_mut(stream_id)
                    .expect("stream history was registered")
                    .committed_state_token_ids = key.token_ids;
                event.token_ids.drain(..reused_prefix_token_count);
            } else {
                self.resident_prefix_state_cache.record_miss();
            }
        }

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
            original_token_count,
            reused_prefix_token_count,
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
        let mut prefill_activation_count = 0usize;
        let mut decode_activation_count = 0usize;
        let mut prefill_time_ns = 0u64;
        let mut decode_time_ns = 0u64;
        let mut scheduler_step_count = 0usize;
        let mut activation_batch_count = 0usize;
        let mut prefill_activation_batch_count = 0usize;
        let mut decode_activation_batch_count = 0usize;
        let mut max_activation_batch_width = 0usize;
        let mut physical_multi_stream_batch_count = 0usize;
        let mut max_physical_multi_stream_batch_width = 0usize;
        let mut max_pending_activation_count = 0usize;
        let mut pending_activations =
            BTreeMap::<u64, VulkanResidentInProcessPlacedPromptEnginePendingActivation>::new();
        let scheduler_activation_capacity = self.streams.len().max(1);

        while input_runs.len() < max_input_events {
            let completed_pending = self.poll_pending_scheduler_activations_with_output(
                &mut pending_activations,
                &mut on_output_event,
            )?;
            for completed in completed_pending {
                decode_activation_count = decode_activation_count.saturating_add(1);
                decode_time_ns = decode_time_ns.saturating_add(completed.activation_time_ns);
                output_events.extend(completed.output_events);
                if let Some(input_run) = completed.input_run {
                    input_runs.push(input_run);
                }
            }
            if input_runs.len() >= max_input_events {
                break;
            }

            let scheduler_snapshot = self.runtime_scheduler.snapshot();
            let activation_work_width = if self
                .has_schedulable_physical_multi_stream_batch(&scheduler_snapshot)
            {
                1
            } else {
                VULKAN_BACKEND_LOOP_MAX_WINDOW
            };
            let scheduler_budget = RuntimeStreamSchedulerBudget::new(
                scheduler_activation_capacity,
                activation_work_width,
                scheduler_activation_capacity.saturating_mul(activation_work_width),
            )
            .with_max_decode_tokens_per_activation(activation_work_width);
            let scheduler_step = self
                .runtime_scheduler
                .schedule_batch_step(scheduler_budget)?;
            if scheduler_step.batches.is_empty() {
                if pending_activations.is_empty() {
                    break;
                }
                self.wait_for_pending_scheduler_activation(
                    &pending_activations,
                    VULKAN_PENDING_ACTIVATION_CONTROL_WAIT_NS,
                )?;
                continue;
            }
            scheduler_step_count = scheduler_step_count.saturating_add(1);
            activation_batch_count =
                activation_batch_count.saturating_add(scheduler_step.batches.len());

            for batch in scheduler_step.batches {
                let batch_run = self.run_scheduler_activation_batch_with_output(
                    batch,
                    &mut pending_activations,
                    &mut on_output_event,
                )?;
                max_pending_activation_count =
                    max_pending_activation_count.max(pending_activations.len());
                if batch_run.prefill_activation_count > 0 {
                    prefill_activation_batch_count =
                        prefill_activation_batch_count.saturating_add(1);
                }
                if batch_run.decode_activation_count > 0 {
                    decode_activation_batch_count =
                        decode_activation_batch_count.saturating_add(1);
                }
                max_activation_batch_width =
                    max_activation_batch_width.max(batch_run.batch_width);
                if batch_run.physical_multi_stream_batch {
                    physical_multi_stream_batch_count =
                        physical_multi_stream_batch_count.saturating_add(1);
                    max_physical_multi_stream_batch_width =
                        max_physical_multi_stream_batch_width.max(batch_run.batch_width);
                }
                prefill_activation_count =
                    prefill_activation_count.saturating_add(batch_run.prefill_activation_count);
                decode_activation_count =
                    decode_activation_count.saturating_add(batch_run.decode_activation_count);
                prefill_time_ns = prefill_time_ns.saturating_add(batch_run.prefill_time_ns);
                decode_time_ns = decode_time_ns.saturating_add(batch_run.decode_time_ns);
                output_events.extend(batch_run.output_events);
                input_runs.extend(batch_run.input_runs);
            }
        }

        while !pending_activations.is_empty() {
            let completed_pending = self.poll_pending_scheduler_activations_with_output(
                &mut pending_activations,
                &mut on_output_event,
            )?;
            if completed_pending.is_empty() {
                self.wait_for_pending_scheduler_activation(
                    &pending_activations,
                    VULKAN_PENDING_ACTIVATION_CONTROL_WAIT_NS,
                )?;
                continue;
            }
            for completed in completed_pending {
                decode_activation_count = decode_activation_count.saturating_add(1);
                decode_time_ns = decode_time_ns.saturating_add(completed.activation_time_ns);
                output_events.extend(completed.output_events);
                if let Some(input_run) = completed.input_run {
                    input_runs.push(input_run);
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
            scheduler_step_count,
            activation_batch_count,
            prefill_activation_batch_count,
            decode_activation_batch_count,
            max_activation_batch_width,
            physical_multi_stream_batch_count,
            max_physical_multi_stream_batch_width,
            max_pending_activation_count,
            prefill_activation_count,
            decode_activation_count,
            prefill_time_ns,
            decode_time_ns,
            prefix_state_cache: self.resident_prefix_state_cache.stats(),
            start_snapshot,
            end_snapshot,
        })
    }

    fn run_scheduler_activation_batch_with_output<F>(
        &mut self,
        batch: RuntimeStreamActivationBatch,
        pending_activations: &mut BTreeMap<
            u64,
            VulkanResidentInProcessPlacedPromptEnginePendingActivation,
        >,
        on_output_event: &mut F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineScheduledBatchRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    {
        if let Some(run) =
            self.try_run_multi_stream_activation_batch_with_output(&batch, on_output_event)?
        {
            return Ok(run);
        }
        let batch_width = batch.activations.len();
        let mut input_runs = Vec::new();
        let mut output_events = Vec::new();
        let mut prefill_activation_count = 0usize;
        let mut decode_activation_count = 0usize;
        let mut prefill_time_ns = 0u64;
        let mut decode_time_ns = 0u64;

        for activation in batch.activations {
            let activation_is_prefill =
                matches!(activation.kind, RuntimeStreamActivationKind::PrefillChunk { .. });
            let stream_id = activation.stream_id.clone();
            let stream = self.streams.get_mut(&stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.clone(),
                }
            })?;
            let callback_stream_id = stream_id.clone();
            let activation_start = Instant::now();
            let activation_start_result =
                stream.begin_runtime_scheduler_activation(&activation, |output_event| {
                    on_output_event(VulkanResidentTokenRuntimeSchedulerOutputEvent {
                        stream_id: callback_stream_id.clone(),
                        output_event,
                    });
                })?;
            let VulkanResidentInProcessPlacedScheduledActivationStart::Complete(scheduled_run) =
                activation_start_result
            else {
                if pending_activations
                    .insert(
                        activation.id,
                        VulkanResidentInProcessPlacedPromptEnginePendingActivation {
                            stream_id,
                            activation_started: activation_start,
                            activation: activation.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                        placed_scheduler_divergence(format!(
                            "scheduler activation {} was submitted twice",
                            activation.id
                        )),
                    ));
                }
                continue;
            };
            let activation_time_ns =
                u64::try_from(activation_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            if activation_is_prefill {
                prefill_activation_count = prefill_activation_count.saturating_add(1);
                prefill_time_ns = prefill_time_ns.saturating_add(activation_time_ns);
            } else {
                decode_activation_count = decode_activation_count.saturating_add(1);
                decode_time_ns = decode_time_ns.saturating_add(activation_time_ns);
            }
            self.runtime_scheduler
                .complete_activation(activation.id, scheduled_run.outcome.clone())?;
            self.record_completed_activation_state(&activation, &scheduled_run)?;

            let stream_output_events =
                placed_prompt_engine_output_events_for(&stream_id, &scheduled_run.output_events);
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

        Ok(VulkanResidentInProcessPlacedPromptEngineScheduledBatchRun {
            batch_width,
            physical_multi_stream_batch: false,
            input_runs,
            output_events,
            prefill_activation_count,
            decode_activation_count,
            prefill_time_ns,
            decode_time_ns,
        })
    }

    fn try_run_multi_stream_activation_batch_with_output<F>(
        &mut self,
        batch: &RuntimeStreamActivationBatch,
        on_output_event: &mut F,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedPromptEngineScheduledBatchRun>,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    {
        if batch.activations.len() < 2
            || batch
                .activations
                .iter()
                .any(|activation| activation.kind.work_units() != 1)
        {
            return Ok(None);
        }
        let stream_ids = batch
            .activations
            .iter()
            .map(|activation| activation.stream_id.clone())
            .collect::<Vec<_>>();
        let first_stream = self
            .streams
            .get(&stream_ids[0])
            .ok_or_else(|| VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_ids[0].clone(),
            })?;
        let first_devices = first_stream.devices.clone();
        for stream_id in &stream_ids[1..] {
            let stream = self.streams.get(stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.clone(),
                }
            })?;
            if !placed_prompt_streams_share_physical_batch_contract(first_stream, stream) {
                return Ok(None);
            }
        }
        let key = VulkanResidentInProcessPlacedPromptEngineBatchKey {
            execution_class_id: batch.activations[0].execution_class_id.clone(),
            stream_ids: stream_ids.clone(),
        };
        if !self.multi_stream_batch_runners.contains_key(&key) {
            let processors = stream_ids
                .iter()
                .map(|stream_id| {
                    self.streams
                        .get(stream_id)
                        .map(|stream| &stream.processor)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                                stream_id: stream_id.clone(),
                            }
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let runner =
                VulkanResidentPlacedMultiStreamBatchRunner::new(&first_devices, &processors)?;
            self.multi_stream_batch_runners.insert(key.clone(), runner);
        }

        let batch_started = Instant::now();
        let mut prepared_lanes = Vec::with_capacity(batch.activations.len());
        for activation in &batch.activations {
            let stream = self.streams.get_mut(&activation.stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: activation.stream_id.clone(),
                }
            })?;
            prepared_lanes.push(stream.prepare_runtime_scheduler_batch_lane(activation)?);
        }
        let input_token_ids = prepared_lanes
            .iter()
            .map(|lane| lane.input_token_id)
            .collect::<Vec<_>>();
        let stream_ticks = prepared_lanes
            .iter()
            .map(|lane| lane.start_stream_tick)
            .collect::<Vec<_>>();
        let device_run = {
            let processors = stream_ids
                .iter()
                .map(|stream_id| {
                    self.streams
                        .get(stream_id)
                        .map(|stream| &stream.processor)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                                stream_id: stream_id.clone(),
                            }
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.multi_stream_batch_runners
                .get(&key)
                .expect("multi-stream batch runner was inserted")
                .run(
                    &first_devices,
                    &processors,
                    &input_token_ids,
                    &stream_ticks,
                )?
        };
        let batch_time_ns =
            u64::try_from(batch_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let mut input_runs = Vec::new();
        let mut output_events = Vec::new();
        for ((activation, prepared), sampled_token_id) in batch
            .activations
            .iter()
            .zip(prepared_lanes)
            .zip(device_run.sampled_token_ids)
        {
            let stream_id = activation.stream_id.clone();
            let callback_stream_id = stream_id.clone();
            let scheduled_run = self
                .streams
                .get_mut(&stream_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                        stream_id: stream_id.clone(),
                    }
                })?
                .complete_runtime_scheduler_batch_lane(
                    prepared,
                    sampled_token_id,
                    device_run.scheduler_turn_count_per_tick,
                    device_run.completed_stage_count_per_tick,
                    |output_event| {
                        on_output_event(VulkanResidentTokenRuntimeSchedulerOutputEvent {
                            stream_id: callback_stream_id.clone(),
                            output_event,
                        });
                    },
                )?;
            self.runtime_scheduler
                .complete_activation(activation.id, scheduled_run.outcome.clone())?;
            self.record_completed_activation_state(activation, &scheduled_run)?;
            let stream_output_events =
                placed_prompt_engine_output_events_for(&stream_id, &scheduled_run.output_events);
            output_events.extend(stream_output_events.iter().cloned());
            if let Some(submitted_run) = scheduled_run.completed_input_run {
                let generated_token_ids = submitted_run.generated_token_ids.clone();
                input_runs.push(VulkanResidentInProcessPlacedPromptEngineInputRun {
                    stream_id,
                    submitted_run,
                    output_events: stream_output_events,
                    generated_token_ids,
                });
            }
        }
        let (prefill_activation_count, decode_activation_count, prefill_time_ns, decode_time_ns) =
            match batch.kind {
                RuntimeStreamActivationBatchKind::PrefillChunk { .. } => {
                    (batch.activations.len(), 0, batch_time_ns, 0)
                }
                RuntimeStreamActivationBatchKind::DecodeFeedback { .. } => {
                    (0, batch.activations.len(), 0, batch_time_ns)
                }
            };
        Ok(Some(
            VulkanResidentInProcessPlacedPromptEngineScheduledBatchRun {
                batch_width: batch.activations.len(),
                physical_multi_stream_batch: true,
                input_runs,
                output_events,
                prefill_activation_count,
                decode_activation_count,
                prefill_time_ns,
                decode_time_ns,
            },
        ))
    }

    fn has_schedulable_physical_multi_stream_batch(
        &self,
        snapshot: &RuntimeStreamSchedulerSnapshot,
    ) -> bool {
        let schedulable = snapshot
            .streams
            .iter()
            .filter(|stream| {
                stream.status == RuntimeStreamStatus::Active
                    && stream.in_flight_activation_count == 0
            })
            .filter_map(|stream| self.streams.get(&stream.stream_id))
            .collect::<Vec<_>>();
        schedulable.iter().enumerate().any(|(index, stream)| {
            schedulable[index + 1..].iter().any(|candidate| {
                placed_prompt_streams_share_physical_batch_contract(stream, candidate)
            })
        })
    }

    fn record_completed_activation_state(
        &mut self,
        activation: &RuntimeStreamActivation,
        scheduled_run: &VulkanResidentInProcessPlacedPromptStreamScheduledActivationRun,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        let processed_count = scheduled_run
            .outcome
            .processed_state_activation_count
            .unwrap_or_else(|| activation.kind.work_units());
        let history = self
            .stream_histories
            .get_mut(&activation.stream_id)
            .ok_or_else(|| VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: activation.stream_id.clone(),
            })?;
        match &activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { token_ids, .. } => {
                let processed_prompt_count = processed_count.min(token_ids.len());
                history
                    .committed_state_token_ids
                    .extend_from_slice(&token_ids[..processed_prompt_count]);
                history
                    .pending_feedback_token_ids
                    .extend(scheduled_run.generated_token_ids.iter().copied());
                for _ in processed_prompt_count..processed_count {
                    let token_id =
                        history.pending_feedback_token_ids.pop_front().ok_or_else(|| {
                            VulkanResidentInProcessPlacedPromptEngineError::Stream(
                                placed_scheduler_divergence(format!(
                                    "prefill activation {} closed with no feedback token",
                                    activation.id
                                )),
                            )
                        })?;
                    history.committed_state_token_ids.push(token_id);
                }
            }
            RuntimeStreamActivationKind::DecodeFeedback { .. } => {
                history
                    .pending_feedback_token_ids
                    .extend(scheduled_run.generated_token_ids.iter().copied());
                for _ in 0..processed_count {
                    let token_id = history.pending_feedback_token_ids.pop_front().ok_or_else(|| {
                        VulkanResidentInProcessPlacedPromptEngineError::Stream(
                            placed_scheduler_divergence(format!(
                                "decode activation {} executed an input tick with no feedback token",
                                activation.id
                            )),
                        )
                    })?;
                    history.committed_state_token_ids.push(token_id);
                }
            }
        }

        if let RuntimeStreamActivationKind::PrefillChunk {
            remaining_prompt_token_count,
            ..
        } = activation.kind
        {
            self.capture_aligned_prefix_checkpoint(
                &activation.stream_id,
                remaining_prompt_token_count,
            )?;
        }
        Ok(())
    }

    fn capture_aligned_prefix_checkpoint(
        &mut self,
        stream_id: &str,
        remaining_prompt_token_count: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        let state = self
            .runtime_scheduler
            .stream_transient_state_snapshot(stream_id)?;
        let append_entries = state
            .entries
            .iter()
            .filter(|entry| entry.shape.retention == TransientStateRetention::Append)
            .collect::<Vec<_>>();
        if append_entries.is_empty()
            || append_entries.iter().any(|entry| {
                entry.logical_activation_count == 0
                    || entry.logical_activation_count % entry.shape.activation_capacity != 0
            })
        {
            return Ok(());
        }
        let append_alignment =
            append_entries
                .iter()
                .try_fold(1usize, |alignment, entry| {
                    let divisor =
                        greatest_common_divisor(alignment, entry.shape.activation_capacity);
                    alignment
                        .checked_div(divisor)
                        .and_then(|value| value.checked_mul(entry.shape.activation_capacity))
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedPromptEngineError::Stream(
                                placed_scheduler_divergence(
                                    "prefix checkpoint state-page alignment overflowed",
                                ),
                            )
                        })
                })?;
        if remaining_prompt_token_count >= append_alignment {
            return Ok(());
        }
        let token_ids = self
            .stream_histories
            .get(stream_id)
            .ok_or_else(|| VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.to_string(),
            })?
            .committed_state_token_ids
            .clone();
        if token_ids.is_empty()
            || append_entries
                .iter()
                .any(|entry| entry.logical_activation_count != token_ids.len())
        {
            return Ok(());
        }
        let stream = self.streams.get(stream_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.to_string(),
            }
        })?;
        let runtime_graph_id = stream.package().runtime_execution_identity.clone();
        let runtime_modifier_bytes = prefix_cache_runtime_modifier_bytes(stream)?;
        let key = RuntimePrefixStateCacheKey::from_token_prefix(
            stream.package().stream_execution_class_id(),
            runtime_graph_id,
            &token_ids,
            &runtime_modifier_bytes,
            state.entries.iter().map(|entry| entry.key.clone()),
        )
        .map_err(RuntimeStreamSchedulerError::from)?;

        let prepared_entry = self.resident_prefix_state_cache.prepare_capture(
            key.clone(),
            self.streams.get(stream_id).expect("stream was validated"),
            &state,
        )?;
        let cache_insert = self
            .runtime_scheduler
            .cache_stream_prefix_state(stream_id, key.clone())?;
        self.resident_prefix_state_cache
            .install(prepared_entry, &cache_insert);
        if let Some(previous) = self.latest_prefix_checkpoint_by_stream.get(stream_id)
            && previous != &key
            && !cache_insert.evicted_keys.contains(previous)
        {
            let previous = previous.clone();
            self.runtime_scheduler.evict_prefix_state(&previous)?;
            self.resident_prefix_state_cache.evict(&previous);
        }
        self.latest_prefix_checkpoint_by_stream
            .retain(|_, checkpoint| !cache_insert.evicted_keys.contains(checkpoint));
        self.latest_prefix_checkpoint_by_stream
            .insert(stream_id.to_string(), key);
        Ok(())
    }

    fn poll_pending_scheduler_activations_with_output<F>(
        &mut self,
        pending_activations: &mut BTreeMap<
            u64,
            VulkanResidentInProcessPlacedPromptEnginePendingActivation,
        >,
        on_output_event: &mut F,
    ) -> Result<
        Vec<VulkanResidentInProcessPlacedPromptEngineCompletedPendingActivation>,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    {
        let pending_ids = pending_activations.keys().copied().collect::<Vec<_>>();
        let mut completed = Vec::new();
        for activation_id in pending_ids {
            let pending = pending_activations.get(&activation_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::Stream(
                    placed_scheduler_divergence(format!(
                        "pending scheduler activation {activation_id} disappeared"
                    )),
                )
            })?;
            let stream_id = pending.stream_id.clone();
            let scheduled_run = {
                let stream = self.streams.get_mut(&stream_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                        stream_id: stream_id.clone(),
                    }
                })?;
                if stream.pending_runtime_scheduler_activation_id() != Some(activation_id) {
                    return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                        placed_scheduler_divergence(format!(
                            "stream {stream_id:?} does not own pending scheduler activation {activation_id}"
                        )),
                    ));
                }
                let callback_stream_id = stream_id.clone();
                stream.poll_runtime_scheduler_activation_with_output(|output_event| {
                    on_output_event(VulkanResidentTokenRuntimeSchedulerOutputEvent {
                        stream_id: callback_stream_id.clone(),
                        output_event,
                    });
                })?
            };
            let Some(scheduled_run) = scheduled_run else {
                continue;
            };
            let pending = pending_activations
                .remove(&activation_id)
                .expect("completed pending activation was present");
            let activation_time_ns =
                u64::try_from(pending.activation_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX);
            self.runtime_scheduler
                .complete_activation(activation_id, scheduled_run.outcome.clone())?;
            self.record_completed_activation_state(&pending.activation, &scheduled_run)?;
            let output_events =
                placed_prompt_engine_output_events_for(&stream_id, &scheduled_run.output_events);
            let input_run = scheduled_run.completed_input_run.map(|submitted_run| {
                let generated_token_ids = submitted_run.generated_token_ids.clone();
                VulkanResidentInProcessPlacedPromptEngineInputRun {
                    stream_id: stream_id.clone(),
                    submitted_run,
                    output_events: output_events.clone(),
                    generated_token_ids,
                }
            });
            completed.push(
                VulkanResidentInProcessPlacedPromptEngineCompletedPendingActivation {
                    input_run,
                    output_events,
                    activation_time_ns,
                },
            );
        }
        Ok(completed)
    }

    fn wait_for_pending_scheduler_activation(
        &mut self,
        pending_activations: &BTreeMap<
            u64,
            VulkanResidentInProcessPlacedPromptEnginePendingActivation,
        >,
        timeout_ns: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        if pending_activations.is_empty() {
            return Ok(());
        }
        let mut groups = Vec::<(
            Rc<VulkanComputeDevice>,
            Vec<(
                String,
                VulkanResidentInProcessPlacedPendingSchedulerWaitTarget<'_>,
            )>,
        )>::new();
        for (activation_id, pending) in pending_activations {
            let stream = self.streams.get(&pending.stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: pending.stream_id.clone(),
                }
            })?;
            if stream.pending_runtime_scheduler_activation_id() != Some(*activation_id) {
                return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                    placed_scheduler_divergence(format!(
                        "stream {:?} does not own pending scheduler activation {activation_id}",
                        pending.stream_id
                    )),
                ));
            }
            let target = stream
                .pending_runtime_scheduler_wait_target()?
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedPromptEngineError::Stream(
                        placed_scheduler_divergence(format!(
                            "stream {:?} has no completion target for pending scheduler activation {activation_id}",
                            pending.stream_id
                        )),
                    )
                })?;
            if let Some((_, group)) = groups
                .iter_mut()
                .find(|(device, _)| Rc::ptr_eq(device, &target.device))
            {
                group.push((pending.stream_id.clone(), target));
            } else {
                groups.push((
                    Rc::clone(&target.device),
                    vec![(pending.stream_id.clone(), target)],
                ));
            }
        }
        let selected_group_index = self.pending_wait_group_cursor % groups.len();
        let (device, group) = &groups[selected_group_index];
        let points = group
            .iter()
            .map(|(_, target)| target.point)
            .collect::<Vec<_>>();
        device
            .wait_any_timeline_semaphore_points_for(&points, timeout_ns)
            .map_err(|error| {
                VulkanResidentInProcessPlacedPromptEngineError::Stream(
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(error),
                )
            })?;
        let completion_by_stream = group
            .iter()
            .map(|(stream_id, target)| {
                target
                    .device
                    .timeline_semaphore_point_is_complete(target.point)
                    .map(|completed| (stream_id.clone(), completed))
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedPromptEngineError::Stream(
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(error),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next_wait_group_cursor = selected_group_index.saturating_add(1);
        drop(groups);
        self.pending_wait_group_cursor = next_wait_group_cursor;
        for (stream_id, completed) in completion_by_stream {
            self.streams
                .get_mut(&stream_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                        stream_id: stream_id.clone(),
                    }
                })?
                .record_pending_runtime_scheduler_wait(completed);
        }
        Ok(())
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
            prefix_state_cache: self.resident_prefix_state_cache.stats(),
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
    pub original_token_count: usize,
    pub reused_prefix_token_count: usize,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentInProcessPlacedPromptEngineScheduledBatchRun {
    batch_width: usize,
    physical_multi_stream_batch: bool,
    input_runs: Vec<VulkanResidentInProcessPlacedPromptEngineInputRun>,
    output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    prefill_activation_count: usize,
    decode_activation_count: usize,
    prefill_time_ns: u64,
    decode_time_ns: u64,
}

struct VulkanResidentInProcessPlacedPromptEnginePendingActivation {
    stream_id: String,
    activation_started: Instant,
    activation: RuntimeStreamActivation,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanResidentInProcessPlacedPromptEngineBatchKey {
    execution_class_id: String,
    stream_ids: Vec<String>,
}

struct VulkanResidentInProcessPlacedPromptEngineCompletedPendingActivation {
    input_run: Option<VulkanResidentInProcessPlacedPromptEngineInputRun>,
    output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    activation_time_ns: u64,
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
    pub scheduler_step_count: usize,
    pub activation_batch_count: usize,
    pub prefill_activation_batch_count: usize,
    pub decode_activation_batch_count: usize,
    pub max_activation_batch_width: usize,
    pub physical_multi_stream_batch_count: usize,
    pub max_physical_multi_stream_batch_width: usize,
    pub max_pending_activation_count: usize,
    pub prefill_activation_count: usize,
    pub decode_activation_count: usize,
    pub prefill_time_ns: u64,
    pub decode_time_ns: u64,
    pub prefix_state_cache: VulkanResidentPlacedPrefixStateCacheStats,
    pub start_snapshot: VulkanResidentInProcessPlacedPromptEngineSnapshot,
    pub end_snapshot: VulkanResidentInProcessPlacedPromptEngineSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptEngineSnapshot {
    pub stream_count: usize,
    pub active_stream_count: usize,
    pub active_stream_ids: Vec<String>,
    pub idle: bool,
    pub prefix_state_cache: VulkanResidentPlacedPrefixStateCacheStats,
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

fn prefix_cache_runtime_modifier_bytes(
    stream: &VulkanResidentInProcessPlacedPromptStream,
) -> Result<Vec<u8>, VulkanResidentInProcessPlacedPromptEngineError> {
    serde_json::to_vec(&serde_json::json!({
        "schema": "nerve.prefix_state_runtime_modifiers.v1",
        "sampler": stream.package.sampler_spec,
        "speculative_draft_tokens": stream.speculative_draft_tokens,
    }))
    .map_err(|error| {
        VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to serialize prefix-state runtime modifiers: {error}"
            )),
        )
        .into()
    })
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn placed_prompt_streams_share_physical_batch_contract(
    first: &VulkanResidentInProcessPlacedPromptStream,
    second: &VulkanResidentInProcessPlacedPromptStream,
) -> bool {
    Arc::ptr_eq(&first.package, &second.package)
        && first.speculative_draft_tokens == 0
        && second.speculative_draft_tokens == 0
        && first.processor.speculative_decoder_count() == 0
        && second.processor.speculative_decoder_count() == 0
        && first.devices.len() == second.devices.len()
        && second.devices.iter().all(|(device_id, device)| {
            first
                .devices
                .get(device_id)
                .is_some_and(|first_device| Rc::ptr_eq(first_device, device))
        })
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
