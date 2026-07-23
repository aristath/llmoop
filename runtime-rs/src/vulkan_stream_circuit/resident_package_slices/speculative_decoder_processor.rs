impl VulkanResidentSpeculativeDecoderProcessor {
    #[allow(clippy::too_many_arguments)]
    fn from_model(
        device: &VulkanComputeDevice,
        model: &VulkanResidentSpeculativeDecoderModelPackage,
        target_hidden: &VulkanResidentBuffer,
        target_output_parameters: &VulkanPermanentParameterBuffers,
        sampler_kernels: &[VulkanResidentSamplerKernelArtifact],
        sampler_spec: &VulkanResidentSamplerSpec,
        random_seed: u32,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let mounted = model
            .device_slice
            .create_mounted_stream_circuit(device)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        mounted.buffers.zero_state_buffers().map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to zero speculative decoder {:?} state: {error}",
                    model.id
                )),
            )
        })?;
        mounted
            .buffers
            .apply_clone_state_policies()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to initialize speculative decoder {:?} cloned state: {error}",
                        model.id
                    )),
                )
            })?;
        let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
        let mounted_bound = mounted
            .mounted_placed_bound_dispatch_plan(&reusable_manifest)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BoundDispatchPlan)?;
        let tick_plan = VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(&mounted_bound);
        let execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan::from_tick_plan(
            device,
            &mounted,
            &mounted_bound,
            model.device_slice.loaded_manifest(),
            tick_plan,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        if execution_plan.distributed_dispatch_count != 0
            || execution_plan.tick_plan.receive_stage_count != 0
            || execution_plan.tick_plan.publish_stage_count != 0
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative decoder {:?} did not compile to one device-resident circuit",
                    model.id
                )),
            ));
        }

        let adapter = &model.package.input_adapter;
        let hidden_input = mounted
            .boundary_io
            .input_buffer(&adapter.target_hidden_signal_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "speculative decoder {:?} has no hidden input {:?}",
                        model.id, adapter.target_hidden_signal_id
                    )),
                )
            })?;
        let input_embedding_weight = model
            .parameter(
                target_output_parameters,
                &model.input_embedding_spec.parameter_tensor,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                    VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: model.input_embedding_spec.parameter_tensor.clone(),
                    },
                )
            })?;
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding_with_parameter_allocation(
                device,
                &mounted,
                input_embedding_weight,
                &model.input_embedding_spirv_words,
                &model.input_embedding_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let output_spec = model
            .output_transducer_spec(model.package.output_transducer.input_signal_id.clone())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let norm_weight = model
            .parameter(target_output_parameters, &output_spec.norm_parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                    VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: output_spec.norm_parameter_tensor.clone(),
                    },
                )
            })?;
        let projection_weight = model
            .parameter(
                target_output_parameters,
                &output_spec.projection_parameter_tensor,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                    VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: output_spec.projection_parameter_tensor.clone(),
                    },
                )
            })?;
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_output_transducer_with_parameter_allocations(
                device,
                &mounted,
                norm_weight,
                projection_weight,
                &model.output_norm_spirv_words,
                &model.output_projection_spirv_words,
                &output_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let sampler = VulkanResidentSamplerRunner::from_output_transducer_with_spec(
            device,
            &mounted,
            &output_transducer,
            sampler_kernels,
            sampler_spec,
            random_seed,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let target_hidden_copy = device
            .create_resident_buffer_copy(
                target_hidden,
                &hidden_input.buffer,
                adapter.target_hidden_byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let recursive_hidden_copy = device
            .create_resident_buffer_copy(
                output_transducer.normalized_frame_buffer(),
                &hidden_input.buffer,
                adapter.target_hidden_byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let pending_target_hidden = device
            .create_resident_buffer(adapter.target_hidden_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        pending_target_hidden
            .write_bytes(&vec![0u8; adapter.target_hidden_byte_capacity])
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let pending_hidden_input_copy = device
            .create_resident_buffer_copy(
                &pending_target_hidden,
                &hidden_input.buffer,
                adapter.target_hidden_byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let update_pending_hidden_copy = device
            .create_resident_buffer_copy(
                target_hidden,
                &pending_target_hidden,
                adapter.target_hidden_byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let restore_target_hidden_copy = device
            .create_resident_buffer_copy(
                &pending_target_hidden,
                target_hidden,
                adapter.target_hidden_byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let state_transaction =
            VulkanResidentStateTransactionBank::new_transactional(device, &mounted.buffers, 1)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let draft_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let state_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        Ok(Self {
            id: model.id.clone(),
            device_id: model.device_id.clone(),
            mounted,
            execution_plan,
            input_transducer,
            output_transducer,
            sampler,
            draft_sequence,
            state_sequence,
            hidden_input_signal_id: adapter.target_hidden_signal_id.clone(),
            target_hidden_copy,
            recursive_hidden_copy,
            pending_hidden_input_copy,
            update_pending_hidden_copy,
            restore_target_hidden_copy,
            pending_target_hidden,
            state_transaction,
        })
    }

    fn capture_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.state_transaction
            .capture_baseline(&self.mounted.buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.sampler
            .capture_token_state()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    fn restore_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.restore_target_hidden_copy
            .run(self.restore_target_hidden_copy.byte_len())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        self.state_transaction
            .restore_baseline(&self.mounted.buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.sampler
            .restore_token_state()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    fn run_draft_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
        draft_index: usize,
    ) -> Result<u32, VulkanResidentInProcessPlacedRuntimeError> {
        let hidden_source = if draft_index == 0 {
            VulkanDraftHiddenSource::Target
        } else {
            VulkanDraftHiddenSource::Recursive
        };
        self.run_composed_step(
            device,
            &self.draft_sequence,
            input_token_id,
            stream_tick,
            hidden_source,
            true,
        )?
        .map(|sampled| sampled.token_id)
        .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingFusedSamplerRun)
    }

    fn run_state_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
        hidden_source: VulkanDraftHiddenSource,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.run_composed_step(
            device,
            &self.state_sequence,
            input_token_id,
            stream_tick,
            hidden_source,
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_composed_step(
        &self,
        device: &VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        input_token_id: u32,
        stream_tick: u64,
        hidden_source: VulkanDraftHiddenSource,
        include_output: bool,
    ) -> Result<Option<VulkanResidentSamplerRun>, VulkanResidentInProcessPlacedRuntimeError> {
        let hidden_copy = match hidden_source {
            VulkanDraftHiddenSource::Target => &self.target_hidden_copy,
            VulkanDraftHiddenSource::Recursive => &self.recursive_hidden_copy,
            VulkanDraftHiddenSource::PendingTarget => &self.pending_hidden_input_copy,
        };
        self.run_composed_step_with_input_copies(
            device,
            sequence,
            input_token_id,
            stream_tick,
            &[VulkanResidentKernelSequenceInputCopy::new(hidden_copy)],
            include_output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_composed_step_with_input_copies(
        &self,
        device: &VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        input_token_id: u32,
        stream_tick: u64,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        include_output: bool,
    ) -> Result<Option<VulkanResidentSamplerRun>, VulkanResidentInProcessPlacedRuntimeError> {
        self.input_transducer
            .prepare_token_id_only(input_token_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let dynamic_state_capacity_activations = u32::try_from(
            self.mounted.buffers.dynamic_state_capacity_activations,
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "speculative decoder dynamic state capacity exceeds u32".to_string(),
            ))
        })?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations,
        };
        self.mounted
            .stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let decoder_dispatches = self
            .execution_plan
            .dispatch_segments
            .iter()
            .flat_map(|segment| segment.dispatches.iter())
            .collect::<Vec<_>>();
        let decoder_push_constants = decoder_dispatches
            .iter()
            .map(|dispatch| stream_control_push_constant_bytes(&dispatch.push_constants, control))
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        let output_dispatch_count = if include_output {
            2usize
                .checked_add(self.sampler.resident_dispatches().len())
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative decoder composed dispatch count overflowed".to_string(),
                    ))
                })?
        } else {
            0
        };
        let mut steps = Vec::with_capacity(
            1usize
                .checked_add(self.sampler.input_tracking_dispatches().len())
                .and_then(|count| count.checked_add(decoder_dispatches.len()))
                .and_then(|count| count.checked_add(output_dispatch_count))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative decoder composed dispatch count overflowed".to_string(),
                    ))
                })?,
        );
        steps.push(VulkanResidentKernelSequenceStep::new(
            &self.input_transducer.resident_dispatch,
            &[],
        ));
        steps.extend(
            self.sampler
                .input_tracking_dispatches()
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        steps.extend(decoder_dispatches.iter().zip(&decoder_push_constants).map(
            |(dispatch, push_constants)| {
                VulkanResidentKernelSequenceStep::new(&dispatch.resident_dispatch, push_constants)
            },
        ));
        if include_output {
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.embedding_norm_dispatch,
                &[],
            ));
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.tied_projection_dispatch,
                &[],
            ));
            steps.extend(
                self.sampler
                    .resident_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
        }
        device
            .run_resident_kernel_sequence_with_input_copies(sequence, input_copies, &steps)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        include_output
            .then(|| self.sampler.completed_run())
            .transpose()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_catch_up_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
        catch_up_index: usize,
        committed_tick_count: usize,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if committed_tick_count == 0 || catch_up_index >= committed_tick_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up index {catch_up_index} is outside committed width {committed_tick_count}"
                )),
            ));
        }
        let hidden_input = self
            .mounted
            .boundary_io
            .input_buffer(&self.hidden_input_signal_id)
            .expect("validated speculative hidden input must remain mounted");
        let hidden_range = (catch_up_index > 0)
            .then(|| {
                let source_offset = frame_byte_capacity
                    .checked_mul(catch_up_index - 1)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "speculative catch-up hidden offset overflowed".to_string(),
                        ))
                    })?;
                VulkanResidentBufferRangeCopy::new(
                    normalized_target_frames,
                    &hidden_input.buffer,
                    source_offset,
                    0,
                    frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .transpose()?;
        let commit_range = (catch_up_index + 1 == committed_tick_count)
            .then(|| {
                let source_offset = frame_byte_capacity
                    .checked_mul(committed_tick_count - 1)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "speculative catch-up commit offset overflowed".to_string(),
                        ))
                    })?;
                VulkanResidentBufferRangeCopy::new(
                    normalized_target_frames,
                    &self.pending_target_hidden,
                    source_offset,
                    0,
                    frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .transpose()?;
        let mut input_copies = Vec::with_capacity(1);
        if let Some(hidden_range) = hidden_range {
            input_copies.push(VulkanResidentKernelSequenceInputCopy::from_range(
                hidden_range,
            ));
        } else {
            input_copies.push(VulkanResidentKernelSequenceInputCopy::new(
                &self.pending_hidden_input_copy,
            ));
        }
        self.run_composed_step_with_input_copies(
            device,
            &self.state_sequence,
            input_token_id,
            stream_tick,
            &input_copies,
            false,
        )?;
        if let Some(commit_range) = commit_range {
            device
                .create_resident_buffer_copy_batch(&[commit_range])
                .and_then(|copy| copy.run())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

    fn commit_target_hidden(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.update_pending_hidden_copy
            .run(self.update_pending_hidden_copy.byte_len())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)
    }
}

