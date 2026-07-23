impl VulkanResidentInProcessPlacedStreamProcessor {
    pub fn run_feedback_bounded_in_process(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        max_ticks: usize,
    ) -> Result<
        VulkanResidentInProcessPlacedFeedbackLoopRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        if max_ticks == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }

        let mut input_token_id = initial_token_id;
        let mut tick_runs = Vec::with_capacity(max_ticks);
        let mut sampled_token_ids = Vec::with_capacity(max_ticks);
        let mut transport = VulkanInProcessPlacedCableTransport::new();

        for tick_index in 0..max_ticks {
            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(tick_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let (tick_run, sampler_run) = self
                .run_prepared_token_id_stream_tick_in_process_with_transport(
                    device,
                    &mut transport,
                    placed_token_input(
                        input_token_id,
                        &self.model.input_device_id,
                        &self.model.output_device_id,
                        tick_index != 0,
                    ),
                    stream_tick,
                    VulkanResidentPlacedTokenTickTail::Sample,
                )?;
            let tick_run = VulkanResidentInProcessPlacedSingleTokenSampleRun {
                tick_run,
                sampler_run: sampler_run
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingFusedSamplerRun)?,
            };
            let sampled_token_id = tick_run.sampler_run.token_id;
            sampled_token_ids.push(sampled_token_id);
            tick_runs.push(VulkanResidentInProcessPlacedFeedbackTickRun {
                stream_tick,
                input_token_id,
                sampled_token_id,
                tick_run,
            });
            input_token_id = sampled_token_id;
        }

        Ok(VulkanResidentInProcessPlacedFeedbackLoopRun {
            input_device_id: self.model.input_device_id.clone(),
            output_device_id: self.model.output_device_id.clone(),
            initial_token_id,
            sampled_token_ids,
            tick_runs,
        })
    }

    pub fn run_speculative_cycle_on_bound_devices(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        initial_token_id: u32,
        start_stream_tick: u64,
        draft_token_count: usize,
        stop_token_ids: &BTreeSet<u32>,
    ) -> Result<VulkanSpeculativeCycleRun, VulkanResidentInProcessPlacedRuntimeError> {
        if draft_token_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let decoder = self.speculative_decoders.first().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident model package has no mounted speculative decoder",
                ),
            )
        })?;
        let target_tick_count = draft_token_count
            .checked_add(1)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        self.ensure_verification_state_transactions(devices, target_tick_count)?;
        let draft_device = devices.get(&decoder.device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: decoder.device_id.clone(),
            }
        })?;

        decoder.capture_baseline()?;
        self.capture_verification_baseline()?;
        self.sampler
            .capture_token_state()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let run = (|| {
            let draft_start = Instant::now();
            let mut draft_token_ids = Vec::with_capacity(draft_token_count);
            let mut draft_input_token_id = initial_token_id;
            for draft_index in 0..draft_token_count {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(draft_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                let token_id = decoder.run_draft_step(
                    draft_device.as_ref(),
                    draft_input_token_id,
                    stream_tick,
                    draft_index,
                )?;
                draft_token_ids.push(token_id);
                draft_input_token_id = token_id;
            }
            let draft_time_ns = u64::try_from(draft_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

            let target_inputs = std::iter::once(initial_token_id)
                .chain(draft_token_ids.iter().copied())
                .collect::<Vec<_>>();
            let target_start = Instant::now();
            let target_token_ids =
                self.run_batched_target_candidates(devices, &target_inputs, start_stream_tick)?;
            let target_verification_time_ns =
                u64::try_from(target_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let mut verification =
                verify_speculative_token_prefix(&draft_token_ids, &target_token_ids)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            truncate_speculative_verification_at_stop(&mut verification, stop_token_ids);
            self.commit_verification_prefix(verification.committed_target_tick_count)?;
            let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: self.model.output_device_id.clone(),
                }
            })?;
            self.sampler
                .record_input_tokens(
                    output_device,
                    &target_inputs[..verification.committed_target_tick_count],
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
            let catch_up_start = Instant::now();
            decoder.restore_baseline()?;
            let batched_output = self.batched_output_projection.borrow();
            let normalized_target_frames = &batched_output
                .as_ref()
                .expect("speculative target output batch was initialized")
                .normalized_frames_buffer;
            for (catch_up_index, input_token_id) in std::iter::once(initial_token_id)
                .chain(draft_token_ids.iter().copied())
                .take(verification.committed_target_tick_count)
                .enumerate()
            {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(catch_up_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                decoder.run_catch_up_step(
                    draft_device.as_ref(),
                    input_token_id,
                    stream_tick,
                    catch_up_index,
                    verification.committed_target_tick_count,
                    normalized_target_frames,
                    self.model
                        .output_transducer_spec
                        .normalized_frame_byte_capacity,
                )?;
            }
            let draft_catch_up_time_ns =
                u64::try_from(catch_up_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

            Ok(VulkanSpeculativeCycleRun {
                decoder_id: decoder.id.clone(),
                initial_token_id,
                start_stream_tick,
                draft_token_ids,
                target_token_ids,
                verification,
                draft_time_ns,
                target_verification_time_ns,
                draft_catch_up_time_ns,
            })
        })();
        if run.is_err() {
            let _ = decoder.restore_baseline();
            let _ = self.restore_verification_baseline();
            let _ = self.sampler.restore_token_state();
        }
        run
    }

    pub fn prompt_session_from_stream_tick(
        &self,
        start_stream_tick: u64,
    ) -> VulkanResidentInProcessPlacedPromptSession {
        VulkanResidentInProcessPlacedPromptSession::new(start_stream_tick)
    }
}
