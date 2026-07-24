impl VulkanResidentInProcessPlacedStreamProcessor {
    pub fn device(
        &self,
        device_id: &str,
    ) -> Option<&VulkanResidentInProcessPlacedStreamProcessorDevice> {
        self.device_slices
            .iter()
            .find(|slice| slice.device_id == device_id)
    }

    fn prepare_token_input(
        &self,
        input: VulkanResidentPlacedTokenInput,
    ) -> Result<VulkanResidentInputEmbeddingTransducerRun, VulkanResidentInProcessPlacedRuntimeError>
    {
        let token_id = input.token_id();
        match input {
            VulkanResidentPlacedTokenInput::HostSupplied(_) => self
                .input_transducer
                .prepare_token_id(token_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer),
            VulkanResidentPlacedTokenInput::ResidentFeedback(_) => {
                Ok(self.input_transducer.completed_run(token_id))
            }
            VulkanResidentPlacedTokenInput::EdgeFeedback(_) => {
                Ok(self.input_transducer.completed_run(token_id))
            }
        }
    }

    fn resident_feedback_next_window_tick_count(&self) -> usize {
        if !self.speculative_decoders.is_empty() {
            return 0;
        }
        self.resident_feedback_loop
            .as_ref()
            .map(|feedback_loop| feedback_loop.window_policy.next_tick_count())
            .unwrap_or(0)
    }

    fn mount_resident_feedback_submission_template(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        feedback_synchronization: Option<&VulkanResidentPlacedFeedbackTimelineSynchronization>,
        output_synchronization: &VulkanResidentPlacedOutputTimelineSynchronization,
    ) -> Result<
        (VulkanResidentQueueSubmissionTemplate, Vec<u64>),
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedEdgeTransport::new();
        let submission_batch = VulkanResidentQueueSubmissionBatch::new();
        let mut output_timeline_values = Vec::with_capacity(tick_count);
        for tick_index in 0..tick_count {
            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(tick_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let mut slices = SmallVec::<
                [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>; 4],
            >::with_capacity(self.device_slices.len());
            for slice in &self.device_slices {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                let mut dispatch_extensions =
                    VulkanMountedPlacedResidentStreamTickDispatchExtensions {
                        sequence_variant: VulkanResidentPlacedTokenTickTail::Sample
                            .sequence_variant(),
                        ..Default::default()
                    };
                if slice.device_id == self.model.input_device_id {
                    dispatch_extensions
                        .prefix_dispatches
                        .push(&self.input_transducer.resident_dispatch);
                }
                if slice.device_id == self.model.output_device_id {
                    dispatch_extensions
                        .prefix_dispatches
                        .extend(self.sampler.input_tracking_dispatches());
                }
                if slice.device_id == self.model.output_device_id {
                    dispatch_extensions
                        .suffix_dispatches
                        .push(&self.output_transducer.embedding_norm_dispatch);
                    dispatch_extensions
                        .suffix_dispatches
                        .push(&self.output_transducer.tied_projection_dispatch);
                    dispatch_extensions
                        .suffix_dispatches
                        .extend(self.sampler.resident_dispatches());
                    dispatch_extensions
                        .suffix_dispatches
                        .push(self.sampler.feedback_control_dispatch());
                }
                slices.push(
                    VulkanMountedPlacedResidentInProcessStreamTickSlice::new_with_dispatch_extensions(
                        device,
                        &slice.mounted,
                        &slice.resident_execution_plan,
                        dispatch_extensions,
                        stream_tick,
                    ),
                );
            }
            let completes_window = tick_index + 1 == tick_count;
            let feedback_turn = feedback_synchronization
                .map(|synchronization| {
                    synchronization
                        .prepare_turn(&self.model.input_device_id, &self.model.output_device_id)
                })
                .transpose()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let output_turn = output_synchronization
                .prepare_turn(&self.model.output_device_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            output_timeline_values.push(output_turn.value);
            let run = run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
                &mut slices,
                &mut transport,
                &self.activation_schedule,
                Some(&self.distributed_dispatch_runners),
                Some(&self.edge_synchronizations),
                VulkanPlacedSubmissionContext {
                    policy: VulkanPlacedSubmissionPolicy {
                        write_stream_control: false,
                        signal_completion: completes_window,
                        wait_for_completion: false,
                        feedback_lane: Some(tick_index),
                    },
                    state_transactions: None,
                    feedback_turn,
                    output_turn: Some(output_turn),
                    submission_batch: Some(&submission_batch),
                },
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Tick)?;
            if run.status != VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed {
                return Err(VulkanResidentInProcessPlacedRuntimeError::IncompleteTick(
                    run.status,
                ));
            }
        }
        let queued_submission_count = submission_batch.pending_submission_count();
        let submission_template = submission_batch
            .mount()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        debug_assert_eq!(
            submission_template.submission_count(),
            queued_submission_count
        );
        Ok((submission_template, output_timeline_values))
    }

    fn advance_resident_feedback_submission_replay(
        &self,
        feedback_synchronization: Option<&VulkanResidentPlacedFeedbackTimelineSynchronization>,
        output_synchronization: &VulkanResidentPlacedOutputTimelineSynchronization,
        tick_count: usize,
    ) -> Result<Vec<u64>, VulkanResidentInProcessPlacedRuntimeError> {
        // Feedback eligibility requires a completed, bridged traversal of the
        // same graph for every tick. Each feedback edge, remote edge, and
        // distributed dispatch therefore advances once per replayed tick, so
        // the mounted queue template can use one uniform timeline offset.
        if let Some(feedback_synchronization) = feedback_synchronization {
            feedback_synchronization
                .advance_replayed_turns(tick_count)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        self.edge_synchronizations
            .advance_replayed_dependencies(tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.distributed_dispatch_runners
            .advance_replayed_dependency_values(tick_count)
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Tick(
                    VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                )
            })?;
        output_synchronization
            .reserve_replayed_turns(tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn run_resident_feedback_window<F>(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        stop_token_ids: &[u32],
        mut submission_replay: Option<&mut Option<VulkanResidentPlacedFeedbackSubmissionReplay>>,
        mut on_sampled_token: F,
    ) -> Result<
        VulkanResidentFeedbackControlCompletion,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(usize, u32, usize, usize, bool)
            -> Result<(), VulkanResidentInProcessPlacedRuntimeError>,
    {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed resident feedback loop is not mounted".to_string(),
            ))
        })?;
        if tick_count < 2 || tick_count > feedback_loop.window_policy.maximum_tick_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "placed resident feedback window requests {tick_count} ticks, mounted width is {}",
                    feedback_loop.window_policy.maximum_tick_count
                )),
            ));
        }
        feedback_loop
            .control
            .arm(tick_count, stop_token_ids)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut template_replayed = false;
        let output_timeline_values =
            if let Some(replay) = submission_replay
                .as_deref_mut()
                .and_then(Option::as_mut)
                .filter(|replay| replay.tick_count == tick_count)
            {
                template_replayed = true;
                replay
                    .validate_tick_count(tick_count)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let output_timeline_values = self.advance_resident_feedback_submission_replay(
                    feedback_loop.feedback_synchronization.as_deref(),
                    &feedback_loop.output_synchronization,
                    tick_count,
                )?;
                replay
                    .submit_next(tick_count)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                output_timeline_values
            } else {
                let (submission_template, output_timeline_values) = self
                    .mount_resident_feedback_submission_template(
                        devices,
                        start_stream_tick,
                        tick_count,
                        feedback_loop.feedback_synchronization.as_deref(),
                        &feedback_loop.output_synchronization,
                    )?;
                submission_template
                    .submit_with_timeline_value_offset(0)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                if let Some(replay_slot) = submission_replay {
                    *replay_slot = Some(
                        VulkanResidentPlacedFeedbackSubmissionReplay::new(
                            submission_template,
                            tick_count,
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                }
                output_timeline_values
            };
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        let terminal_output_value = output_timeline_values.last().copied().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "resident feedback window has no output timeline value".to_string(),
            ))
        })?;
        feedback_loop
            .output_synchronization
            .wait_for_turn(output_device, terminal_output_value)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        // The output timeline signal is recorded after the terminal output
        // slice. Every upstream slice, distributed shard, and transfer is a
        // semaphore dependency of that slice, so this one wait is the graph
        // completion proof. Waiting every slice fence again only serialized
        // host control on already-completed work.
        let mut completion = feedback_loop
            .control
            .completion()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        completion.template_replayed = template_replayed;
        if completion.executed_tick_count == 0
            || completion.executed_tick_count > tick_count
            || completion.sampled_tick_count == 0
            || completion.sampled_tick_count > completion.executed_tick_count
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "resident feedback control completed {} model ticks and {} sampled ticks for a {tick_count}-tick window",
                    completion.executed_tick_count, completion.sampled_tick_count
                )),
            ));
        }
        if !matches!(
            completion.stop_reason,
            VULKAN_FEEDBACK_STOP_REASON_NONE
                | VULKAN_FEEDBACK_STOP_REASON_EOS
                | VULKAN_FEEDBACK_STOP_REASON_CANCELLED
        ) || (completion.executed_tick_count < tick_count
            && completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_NONE)
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "resident feedback control reported invalid stop reason {} after {} of {tick_count} ticks",
                    completion.stop_reason, completion.executed_tick_count
                )),
            ));
        }
        if completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_CANCELLED {
            feedback_loop.control.acknowledge_cancellation();
        }
        for tick_index in 0..completion.sampled_tick_count {
            let stream_tick = start_stream_tick
                .checked_add(u64::try_from(tick_index).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                })?)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let sampled_token_id = self
                .sampler
                .completed_run_at(stream_tick)
                .map(|run| run.token_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
            on_sampled_token(
                tick_index,
                sampled_token_id,
                feedback_loop.scheduler_turn_count_per_tick,
                feedback_loop.completed_stage_count_per_tick,
                completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_CANCELLED
                    && tick_index + 1 == completion.sampled_tick_count,
            )?;
        }
        Ok(completion)
    }

}
