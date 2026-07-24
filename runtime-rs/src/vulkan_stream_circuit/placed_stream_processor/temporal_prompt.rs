impl VulkanResidentInProcessPlacedStreamProcessor {
    fn linear_pipeline_device_indices(
        &self,
    ) -> Result<Vec<usize>, VulkanResidentInProcessPlacedRuntimeError> {
        let mut current = self
            .device_slices
            .iter()
            .position(|slice| slice.device_id == self.model.input_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "placed package {:?} has no input pipeline device {:?}",
                    self.model.package_id, self.model.input_device_id
                )))
            })?;
        let output = self
            .device_slices
            .iter()
            .position(|slice| slice.device_id == self.model.output_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "placed package {:?} has no output pipeline device {:?}",
                    self.model.package_id, self.model.output_device_id
                )))
            })?;
        let mut ordered = Vec::with_capacity(self.device_slices.len());
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current) {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError("placed verification pipeline contains a device cycle".to_string()),
                ));
            }
            ordered.push(current);
            if current == output {
                break;
            }
            let outgoing = &self.device_slices[current]
                .mounted
                .edge_io
                .outgoing_buffers;
            let [outgoing] = outgoing.as_slice() else {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "placed verification pipeline device {:?} has {} outgoing activation edges; expected one",
                        self.device_slices[current].device_id,
                        outgoing.len()
                    )),
                ));
            };
            current = self
                .device_slices
                .iter()
                .position(|slice| {
                    slice.device_id == outgoing.endpoint.remote_device_id
                        && slice
                            .mounted
                            .edge_io
                            .incoming_buffer(outgoing.endpoint.edge_index)
                            .is_some()
                })
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "placed verification pipeline edge {} has no mounted destination {:?}",
                        outgoing.endpoint.edge_index, outgoing.endpoint.remote_device_id
                    )))
                })?;
        }
        if ordered.len() != self.device_slices.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "placed verification pipeline reaches {} of {} mounted devices",
                    ordered.len(),
                    self.device_slices.len()
                )),
            ));
        }
        Ok(ordered)
    }

    fn temporal_block_width(
        &self,
        available_token_count: usize,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        const SIGNAL_MEMORY_BUDGET_PER_DEVICE: usize = 256 * 1024 * 1024;
        const RECORDED_DISPATCH_BUDGET_PER_SUBMISSION: usize = 65_536;

        if available_token_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let mut width = available_token_count;
        for slice in &self.device_slices {
            let (_, signal_buffer_plan) =
                component_batch_signal_buffer_plan(&slice.mounted, &slice.mounted_bound.dispatches)?;
            let signal_bytes_per_lane =
                signal_buffer_plan
                    .iter()
                    .try_fold(0usize, |total, allocation| {
                        total
                            .checked_add(allocation.frame_byte_capacity)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "temporal signal byte count overflowed".to_string(),
                                ))
                            })
                    })?;
            if let Some(memory_width) =
                SIGNAL_MEMORY_BUDGET_PER_DEVICE.checked_div(signal_bytes_per_lane)
            {
                width = width.min(memory_width.max(1));
            }

            if let Some(causal_scan_tile_width) = slice
                .package_slice
                .batch_kernels
                .iter()
                .filter(|artifact| {
                    artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
                })
                .map(|artifact| artifact.lane_tile_width)
                .min()
            {
                width = width.min(causal_scan_tile_width);
            }

            let mut scalar_dispatches_per_lane_by_component = BTreeMap::<&str, usize>::new();
            for dispatch in &slice.mounted_bound.dispatches {
                if !slice.package_slice.batch_kernels.iter().any(|artifact| {
                    artifact.component_id == dispatch.component_id && artifact.node_id == dispatch.node_id
                }) {
                    *scalar_dispatches_per_lane_by_component
                        .entry(&dispatch.component_id)
                        .or_default() += 1;
                }
            }
            let scalar_dispatches_per_lane = scalar_dispatches_per_lane_by_component
                .values()
                .copied()
                .max()
                .unwrap_or_default();
            if let Some(dispatch_width) =
                RECORDED_DISPATCH_BUDGET_PER_SUBMISSION.checked_div(scalar_dispatches_per_lane)
            {
                width = width.min(dispatch_width.max(1));
            }
        }
        Ok(width.max(1))
    }

    fn ensure_temporal_block_execution(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        block_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if self
            .temporal_block_execution
            .borrow()
            .as_ref()
            .is_some_and(|runner| runner.execution_graph.lane_capacity >= block_width)
        {
            return Ok(());
        }
        let pipeline = self.linear_pipeline_device_indices()?;
        let first_device_index = *pipeline.first().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "temporal pipeline is empty".to_string(),
            ))
        })?;
        let execution_graph = VulkanResidentPlacedComponentBatchRunner::new(
            devices,
            &self.device_slices,
            &self.execution_quantum_calibrators,
            block_width,
            VulkanComponentBatchExecutionMode::CausalSequence,
            &self.model.distributed_execution_plan,
            &self.model.distributed_parameter_buffers,
        )?;
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
        let embedding_weight = self
            .model
            .input_transducer_parameter_buffers
            .parameter_buffer(&self.model.input_transducer_spec.parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                    VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: self.model.input_transducer_spec.parameter_tensor.clone(),
                    },
                )
            })?;
        let input_signal = execution_graph.slice(first_device_index)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelInput(
                self.model.input_transducer_spec.output_signal_id.clone(),
            ),
        )?;
        let input_embedding = VulkanResidentBatchedInputEmbeddingRunner::new(
            input_device,
            block_width,
            embedding_weight,
            &input_signal.buffer,
            &self.model.input_transducer_batch_spirv_words,
            &self.model.input_transducer_spec,
        )?;
        let scalar_input = self.device_slices[first_device_index]
            .mounted
            .boundary_io
            .input_buffer(&self.model.input_transducer_spec.output_signal_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "temporal input device has no boundary {:?}",
                    self.model.input_transducer_spec.output_signal_id
                )))
            })?;
        let input_frame_copies = (0..block_width)
            .map(|frame_index| {
                let source_offset = input_signal
                    .frame_byte_capacity
                    .checked_mul(frame_index)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "temporal input frame offset overflowed".to_string(),
                        ))
                    })?;
                let copy = VulkanResidentBufferRangeCopy::new(
                    &input_signal.buffer,
                    &scalar_input.buffer,
                    source_offset,
                    0,
                    input_signal.frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                input_device
                    .create_resident_buffer_copy_batch(&[copy])
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let last_device_index = *pipeline.last().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "temporal pipeline is empty".to_string(),
            ))
        })?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        let output_signal = execution_graph.slice(last_device_index)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelOutput(
                self.model.output_transducer_spec.input_signal_id.clone(),
            ),
        )?;
        let scalar_output = self.device_slices[last_device_index]
            .mounted
            .boundary_io
            .output_buffer(&self.model.output_transducer_spec.input_signal_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "temporal output device has no boundary {:?}",
                    self.model.output_transducer_spec.input_signal_id
                )))
            })?;
        let output_frame_copies = (0..block_width)
            .map(|frame_index| {
                let source_offset = output_signal
                    .frame_byte_capacity
                    .checked_mul(frame_index)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "temporal output frame offset overflowed".to_string(),
                        ))
                    })?;
                let copy = VulkanResidentBufferRangeCopy::new(
                    &output_signal.buffer,
                    &scalar_output.buffer,
                    source_offset,
                    0,
                    output_signal.frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                output_device
                    .create_resident_buffer_copy_batch(&[copy])
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<Vec<_>, _>>()?;
        *self.temporal_block_execution.borrow_mut() =
            Some(VulkanResidentPlacedTemporalBlockRunner {
                execution_graph,
                input_embedding,
                input_frame_copies,
                output_frame_copies,
                pipeline,
            });
        Ok(())
    }

    fn run_temporal_prompt_block(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        sample_last: bool,
    ) -> Result<VulkanResidentTemporalBlockRun, VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let tick_count = u64::try_from(input_token_ids.len())
            .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        let end_stream_tick = start_stream_tick
            .checked_add(tick_count - 1)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        self.ensure_temporal_block_execution(devices, input_token_ids.len())?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "temporal context capacity exceeds u32".to_string(),
                ))
            })?;
        let runner_guard = self.temporal_block_execution.borrow();
        let runner = runner_guard.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "temporal block execution is not mounted".to_string(),
            ))
        })?;
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
        runner.input_embedding.run(input_device, input_token_ids)?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        self.sampler
            .record_input_tokens(output_device, input_token_ids)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;

        let mut transport_stats = VulkanPlacedEdgeTransportStats::default();
        for (pipeline_index, device_index) in runner.pipeline.iter().copied().enumerate() {
            let slice = &self.device_slices[device_index];
            runner.execution_graph.run_causal_sequence(
                devices,
                device_index,
                &slice.device_id,
                &slice.mounted,
                input_token_ids,
                start_stream_tick,
                capacity,
            )?;
            if let Some(next_device_index) = runner.pipeline.get(pipeline_index + 1).copied() {
                let [outgoing] = slice.mounted.edge_io.outgoing_buffers.as_slice() else {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "temporal pipeline device {:?} has {} outgoing edges; expected one",
                            slice.device_id,
                            slice.mounted.edge_io.outgoing_buffers.len()
                        )),
                    ));
                };
                runner.execution_graph.transfer_edge(
                    device_index,
                    next_device_index,
                    outgoing.endpoint.edge_index,
                )?;
                let transferred_bytes =
                    outgoing.byte_capacity.saturating_mul(input_token_ids.len());
                transport_stats.direct_copy_count =
                    transport_stats.direct_copy_count.saturating_add(1);
                transport_stats.direct_copy_byte_count = transport_stats
                    .direct_copy_byte_count
                    .saturating_add(transferred_bytes);
                transport_stats.direct_receive_count =
                    transport_stats.direct_receive_count.saturating_add(1);
                transport_stats.direct_receive_byte_count = transport_stats
                    .direct_receive_byte_count
                    .saturating_add(transferred_bytes);
            }
        }

        if !self.speculative_decoders.is_empty() {
            let last_device_index = *runner.pipeline.last().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "temporal pipeline is empty".to_string(),
                ))
            })?;
            let output_slice = &self.device_slices[last_device_index];
            let output_device = devices.get(&output_slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: output_slice.device_id.clone(),
                }
            })?;
            for (frame_index, input_token_id) in input_token_ids.iter().copied().enumerate() {
                runner.input_frame_copies[frame_index]
                    .run()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                runner.output_frame_copies[frame_index]
                    .run()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(frame_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                output_device
                    .run_resident_kernel_dispatch(
                        &self.output_transducer.embedding_norm_dispatch,
                        &[],
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                self.synchronize_speculative_decoders_after_target_tick(
                    devices,
                    input_token_id,
                    stream_tick,
                )?;
            }
        }

        let sampled_token_id = if sample_last {
            let last_device_index = *runner.pipeline.last().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "temporal pipeline is empty".to_string(),
                ))
            })?;
            let output_slice = &self.device_slices[last_device_index];
            let output_device = devices.get(&output_slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: output_slice.device_id.clone(),
                }
            })?;
            runner
                .output_frame_copies
                .get(input_token_ids.len() - 1)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "temporal output frame copy is not mounted".to_string(),
                    ))
                })?
                .run()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            output_slice
                .mounted
                .stream_control_buffer
                .write_bytes_at(
                    VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                    &stream_control_metadata_bytes(VulkanMountedPlacedStreamControl {
                        stream_tick: end_stream_tick,
                        control_flags: 0,
                        dynamic_state_capacity_activations: capacity,
                    }),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.output_transducer
                .run(output_device)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
            Some(
                self.sampler
                    .run(output_device)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?
                    .token_id,
            )
        } else {
            None
        };

        Ok(VulkanResidentTemporalBlockRun {
            sampled_token_id,
            scheduler_turn_count_per_tick: self.activation_schedule.turns.len(),
            completed_stage_count_per_tick: self
                .device_slices
                .iter()
                .map(|slice| slice.dispatch_count)
                .sum(),
            transport_stats,
        })
    }

    fn run_batched_target_candidates(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let pipeline = self.linear_pipeline_device_indices()?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "placed verification context capacity exceeds u32".to_string(),
                ))
            })?;
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
        self.verification_input_embedding
            .borrow()
            .as_ref()
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "verification input embedding is not mounted".to_string(),
                ))
            })?
            .run(input_device, input_token_ids)?;

        {
            let transaction_guard = self.verification_state_transactions.borrow();
            let transactions = transaction_guard.as_ref().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "verification state transaction is not mounted".to_string(),
                ))
            })?;
            let batch_execution_guard = self.component_batch_execution.borrow();
            let batch_execution = batch_execution_guard.as_ref().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "component batch execution is not mounted".to_string(),
                ))
            })?;
            for (pipeline_index, device_index) in pipeline.iter().copied().enumerate() {
                let slice = &self.device_slices[device_index];
                let transaction = transactions.get(device_index).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "verification state transaction has no device index {device_index}"
                    )))
                })?;
                batch_execution.run_independent_candidates(
                    devices,
                    device_index,
                    &slice.device_id,
                    &slice.mounted,
                    transaction,
                    input_token_ids,
                    start_stream_tick,
                    capacity,
                )?;
                if let Some(next_device_index) = pipeline.get(pipeline_index + 1).copied() {
                    let [outgoing] = slice.mounted.edge_io.outgoing_buffers.as_slice() else {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "placed verification device {:?} has {} outgoing edges; expected one",
                                slice.device_id,
                                slice.mounted.edge_io.outgoing_buffers.len()
                            )),
                        ));
                    };
                    batch_execution.transfer_edge(
                        device_index,
                        next_device_index,
                        outgoing.endpoint.edge_index,
                    )?;
                }
            }
        }

        let output_device_index = *pipeline.last().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed verification pipeline is empty".to_string(),
            ))
        })?;
        let output_slice = &self.device_slices[output_device_index];
        let output_device = devices.get(&output_slice.device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: output_slice.device_id.clone(),
            }
        })?;
        let batch_capacity_is_insufficient = self
            .batched_output_projection
            .borrow()
            .as_ref()
            .is_none_or(|runner| runner.batch_capacity < input_token_ids.len());
        if batch_capacity_is_insufficient {
            let norm_weight = self
                .model
                .output_transducer_parameter_buffers
                .parameter_buffer(&self.model.output_transducer_spec.norm_parameter_tensor)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                        VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                            tensor: self.model.output_transducer_spec.norm_parameter_tensor.clone(),
                        },
                    )
                })?;
            let projection_weight = self
                .model
                .output_transducer_parameter_buffers
                .parameter_buffer(&self.model.output_transducer_spec.projection_parameter_tensor)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                        VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                            tensor: self
                                .model
                                .output_transducer_spec
                                .projection_parameter_tensor
                                .clone(),
                        },
                    )
                })?;
            let projection_scale = projection_scale_parameter_buffer(
                &self.model.output_transducer_parameter_buffers,
                &self.model.output_transducer_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
            let batch_execution = self.component_batch_execution.borrow();
            let raw_output = batch_execution
                .as_ref()
                .expect("verification component batch was initialized")
                .slice(output_device_index)?
                .signal_buffer(&VulkanComponentBatchSignalKey::ModelOutput(
                    self.model.output_transducer_spec.input_signal_id.clone(),
                ))?;
            let runner = VulkanResidentBatchedOutputProjectionRunner::new(
                output_device,
                input_token_ids.len(),
                self.model.embedding_norm_batch_lane_tile_width,
                self.model.projection_batch_lane_tile_width,
                &raw_output.buffer,
                norm_weight,
                projection_weight,
                projection_scale,
                &self.model.embedding_norm_batch_spirv_words,
                &self.model.tied_projection_batch_spirv_words,
                &self.model.output_transducer_spec,
                &self.sampler,
                &self.model.sampler_kernels,
                &self.model.sampler_spec,
            )?;
            drop(batch_execution);
            *self.batched_output_projection.borrow_mut() = Some(runner);
        }
        let batch_runner = self.batched_output_projection.borrow();
        let batch_runner = batch_runner
            .as_ref()
            .expect("batched output projection runner was initialized");
        batch_runner.project(output_device, input_token_ids.len())?;
        batch_runner.sample_batch(output_device, input_token_ids, start_stream_tick, capacity)?;
        let mut target_token_ids = Vec::with_capacity(input_token_ids.len());
        for batch_index in 0..input_token_ids.len() {
            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(batch_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            target_token_ids.push(
                self.sampler
                    .completed_run_at(stream_tick)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?
                    .token_id,
            );
        }
        Ok(target_token_ids)
    }

}
