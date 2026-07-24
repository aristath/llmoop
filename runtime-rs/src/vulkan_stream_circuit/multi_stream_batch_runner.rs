struct VulkanResidentPlacedMultiStreamBatchRunner {
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    output_projection: VulkanResidentBatchedOutputProjectionRunner,
    pipeline: Vec<usize>,
    dynamic_state_capacity_activations: u32,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
}

struct VulkanResidentPlacedMultiStreamBatchRun {
    sampled_token_ids: Vec<u32>,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
}

impl VulkanResidentPlacedMultiStreamBatchRunner {
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        processors: &[&VulkanResidentInProcessPlacedStreamProcessor],
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let first = processors.first().copied().ok_or(
            VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget,
        )?;
        if processors.iter().copied().any(|processor| {
            !Arc::ptr_eq(&processor.model, &first.model)
                || processor.model.dynamic_state_capacity_activations
                    != first.model.dynamic_state_capacity_activations
                || !processor.speculative_decoders.is_empty()
        }) {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "multi-stream batch lanes must share one mounted package, context capacity, and non-speculative execution contract"
                        .to_string(),
                ),
            ));
        }
        let pipeline = first.linear_pipeline_device_indices()?;
        let first_device_index = *pipeline.first().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "multi-stream batch pipeline is empty".to_string(),
            ))
        })?;
        let execution_graph =
            VulkanResidentPlacedComponentBatchRunner::new_for_independent_streams(
                devices, processors,
            )?;
        let input_device = devices.get(&first.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: first.model.input_device_id.clone(),
            }
        })?;
        let embedding_weight = first
            .model
            .input_transducer_parameter_buffers
            .parameter_buffer(&first.model.input_transducer_spec.parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                    VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: first.model.input_transducer_spec.parameter_tensor.clone(),
                    },
                )
            })?;
        let input_signal = execution_graph.slice(first_device_index)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelInput(
                first.model.input_transducer_spec.output_signal_id.clone(),
            ),
        )?;
        let input_embedding = VulkanResidentBatchedInputEmbeddingRunner::new(
            input_device,
            processors.len(),
            embedding_weight,
            &input_signal.buffer,
            &first.model.input_transducer_batch_spirv_words,
            &first.model.input_transducer_spec,
        )?;

        let last_device_index = *pipeline.last().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "multi-stream batch pipeline is empty".to_string(),
            ))
        })?;
        let output_device = devices.get(&first.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: first.model.output_device_id.clone(),
            }
        })?;
        let output_signal = execution_graph.slice(last_device_index)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelOutput(
                first.model.output_transducer_spec.input_signal_id.clone(),
            ),
        )?;
        let norm_weight = first
            .model
            .output_transducer_parameter_buffers
            .parameter_buffer(&first.model.output_transducer_spec.norm_parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                    VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: first
                            .model
                            .output_transducer_spec
                            .norm_parameter_tensor
                            .clone(),
                    },
                )
            })?;
        let projection_weight = first
            .model
            .output_transducer_parameter_buffers
            .parameter_buffer(
                &first
                    .model
                    .output_transducer_spec
                    .projection_parameter_tensor,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                    VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: first
                            .model
                            .output_transducer_spec
                            .projection_parameter_tensor
                            .clone(),
                    },
                )
            })?;
        let projection_scale = projection_scale_parameter_buffer(
            &first.model.output_transducer_parameter_buffers,
            &first.model.output_transducer_spec,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let sampler_lanes = processors
            .iter()
            .map(|processor| &processor.sampler)
            .collect::<Vec<_>>();
        let output_projection =
            VulkanResidentBatchedOutputProjectionRunner::new_for_sampler_lanes(
                output_device,
                first.model.embedding_norm_batch_lane_tile_width,
                first.model.projection_batch_lane_tile_width,
                &output_signal.buffer,
                norm_weight,
                projection_weight,
                projection_scale,
                &first.model.embedding_norm_batch_spirv_words,
                &first.model.tied_projection_batch_spirv_words,
                &first.model.output_transducer_spec,
                &sampler_lanes,
                &first.model.sampler_kernels,
                &first.model.sampler_spec,
            )?;
        let dynamic_state_capacity_activations =
            u32::try_from(first.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "multi-stream batch context capacity exceeds u32".to_string(),
                ))
            })?;
        Ok(Self {
            execution_graph,
            input_embedding,
            output_projection,
            pipeline,
            dynamic_state_capacity_activations,
            scheduler_turn_count_per_tick: first.activation_schedule.turns.len(),
            completed_stage_count_per_tick: first
                .device_slices
                .iter()
                .map(|slice| slice.dispatch_count)
                .sum(),
        })
    }

    fn run(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        processors: &[&VulkanResidentInProcessPlacedStreamProcessor],
        input_token_ids: &[u32],
        stream_ticks: &[u64],
    ) -> Result<VulkanResidentPlacedMultiStreamBatchRun, VulkanResidentInProcessPlacedRuntimeError>
    {
        let batch_width = processors.len();
        if input_token_ids.len() != batch_width || stream_ticks.len() != batch_width {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "multi-stream batch has {batch_width} processors, {} input tokens, and {} stream ticks",
                    input_token_ids.len(),
                    stream_ticks.len()
                )),
            ));
        }
        let first = processors.first().copied().ok_or(
            VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget,
        )?;
        let input_device = devices.get(&first.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: first.model.input_device_id.clone(),
            }
        })?;
        self.input_embedding.run(input_device, input_token_ids)?;
        for (pipeline_index, device_index) in self.pipeline.iter().copied().enumerate() {
            let slice = &first.device_slices[device_index];
            self.execution_graph.run_independent_streams(
                devices,
                device_index,
                &slice.device_id,
                &slice.mounted,
                input_token_ids,
                stream_ticks,
                self.dynamic_state_capacity_activations,
            )?;
            if let Some(next_device_index) = self.pipeline.get(pipeline_index + 1).copied() {
                let [outgoing] = slice.mounted.edge_io.outgoing_buffers.as_slice() else {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "multi-stream batch device {:?} has {} outgoing edges; expected one",
                            slice.device_id,
                            slice.mounted.edge_io.outgoing_buffers.len()
                        )),
                    ));
                };
                self.execution_graph.transfer_edge(
                    device_index,
                    next_device_index,
                    outgoing.endpoint.edge_index,
                )?;
            }
        }

        let output_device = devices.get(&first.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: first.model.output_device_id.clone(),
            }
        })?;
        for (processor, input_token_id) in processors.iter().zip(input_token_ids) {
            processor
                .sampler
                .record_input_tokens(output_device, std::slice::from_ref(input_token_id))
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        }
        self.output_projection.project(output_device, batch_width)?;
        let capacities = vec![self.dynamic_state_capacity_activations; batch_width];
        self.output_projection.sample_independent_streams(
            output_device,
            input_token_ids,
            stream_ticks,
            &capacities,
        )?;
        let sampled_token_ids = processors
            .iter()
            .zip(stream_ticks)
            .map(|(processor, stream_tick)| {
                processor
                    .sampler
                    .completed_run_at(*stream_tick)
                    .map(|run| run.token_id)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VulkanResidentPlacedMultiStreamBatchRun {
            sampled_token_ids,
            scheduler_turn_count_per_tick: self.scheduler_turn_count_per_tick,
            completed_stage_count_per_tick: self.completed_stage_count_per_tick,
        })
    }
}
