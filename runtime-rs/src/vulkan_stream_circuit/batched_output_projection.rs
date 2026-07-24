struct VulkanResidentBatchedOutputProjectionRunner {
    batch_capacity: usize,
    normalized_frames_buffer: VulkanResidentBuffer,
    _batched_logits_buffer: VulkanResidentBuffer,
    norm_dispatch: VulkanResidentKernelDispatch,
    projection_dispatch: VulkanResidentKernelDispatch,
    projection_sequence: VulkanResidentKernelSequence,
    sampler_views: Vec<VulkanResidentSamplerLogitsView>,
}

impl VulkanResidentBatchedOutputProjectionRunner {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &VulkanComputeDevice,
        batch_capacity: usize,
        norm_batch_lane_tile_width: u32,
        batch_lane_tile_width: u32,
        raw_frames_buffer: &VulkanResidentBuffer,
        norm_weight: &VulkanPermanentParameterBufferAllocation,
        projection_weight: &VulkanPermanentParameterBufferAllocation,
        projection_scale: Option<&VulkanPermanentParameterBufferAllocation>,
        norm_spirv_words: &[u32],
        projection_spirv_words: &[u32],
        output_spec: &VulkanResidentOutputTransducerSpec,
        sampler: &VulkanResidentSamplerRunner,
        sampler_kernels: &[VulkanResidentSamplerKernelArtifact],
        sampler_spec: &VulkanResidentSamplerSpec,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let sampler_lanes = vec![sampler; batch_capacity];
        Self::new_for_sampler_lanes(
            device,
            norm_batch_lane_tile_width,
            batch_lane_tile_width,
            raw_frames_buffer,
            norm_weight,
            projection_weight,
            projection_scale,
            norm_spirv_words,
            projection_spirv_words,
            output_spec,
            &sampler_lanes,
            sampler_kernels,
            sampler_spec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_for_sampler_lanes(
        device: &VulkanComputeDevice,
        norm_batch_lane_tile_width: u32,
        batch_lane_tile_width: u32,
        raw_frames_buffer: &VulkanResidentBuffer,
        norm_weight: &VulkanPermanentParameterBufferAllocation,
        projection_weight: &VulkanPermanentParameterBufferAllocation,
        projection_scale: Option<&VulkanPermanentParameterBufferAllocation>,
        norm_spirv_words: &[u32],
        projection_spirv_words: &[u32],
        output_spec: &VulkanResidentOutputTransducerSpec,
        sampler_lanes: &[&VulkanResidentSamplerRunner],
        sampler_kernels: &[VulkanResidentSamplerKernelArtifact],
        sampler_spec: &VulkanResidentSamplerSpec,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let batch_capacity = sampler_lanes.len();
        if batch_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let norm_tile_width = usize::try_from(norm_batch_lane_tile_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched output norm lane tile width exceeds usize".to_string(),
            ))
        })?;
        if norm_tile_width == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("batched output norm lane tile width is zero".to_string()),
            ));
        }
        let tile_width = usize::try_from(batch_lane_tile_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched output projection lane tile width exceeds usize".to_string(),
            ))
        })?;
        if tile_width == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("batched output projection lane tile width is zero".to_string()),
            ));
        }
        validate_output_projection_weight(projection_weight, output_spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        validate_output_projection_scale(projection_scale, output_spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        validate_output_embedding_norm_weight(norm_weight, output_spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let normalized_frames_byte_capacity = output_spec
            .normalized_frame_byte_capacity
            .checked_mul(batch_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched normalized frame capacity overflowed".to_string(),
                ))
            })?;
        let batched_logits_byte_capacity = output_spec
            .logits_byte_capacity
            .checked_mul(batch_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched logits capacity overflowed".to_string(),
                ))
            })?;
        let normalized_frames_buffer = device
            .create_resident_buffer(normalized_frames_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let batched_logits_buffer = device
            .create_resident_buffer(batched_logits_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        if raw_frames_buffer.byte_capacity() < normalized_frames_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched raw output buffer has {} bytes, requires {normalized_frames_byte_capacity}",
                    raw_frames_buffer.byte_capacity()
                )),
            ));
        }
        let norm_workgroup_count_y = batch_capacity
            .checked_add(norm_tile_width - 1)
            .map(|width| width / norm_tile_width)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched output norm workgroup count overflowed".to_string(),
                ))
            })?;
        let norm_bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                raw_frames_buffer,
                normalized_frames_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                1,
                &normalized_frames_buffer,
                normalized_frames_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Write),
            VulkanResidentKernelBufferBinding::new(
                2,
                &norm_weight.buffer,
                norm_weight.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        ];
        let norm_dispatch = device
            .create_resident_kernel_dispatch_2d(
                norm_spirv_words,
                &norm_bindings,
                1,
                norm_workgroup_count_y,
                output_spec.norm_local_size_x,
                std::mem::size_of::<u32>() as u32,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let workgroup_count_y = batch_capacity
            .checked_add(tile_width - 1)
            .map(|width| width / tile_width)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched output projection workgroup count overflowed".to_string(),
                ))
            })?;
        let mut bindings = vec![
            VulkanResidentKernelBufferBinding::new(
                0,
                &normalized_frames_buffer,
                normalized_frames_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                1,
                &projection_weight.buffer,
                projection_weight.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                2,
                &batched_logits_buffer,
                batched_logits_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Write),
        ];
        if let Some(scale) = projection_scale {
            bindings.push(
                VulkanResidentKernelBufferBinding::new(3, &scale.buffer, scale.byte_capacity)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        let projection_dispatch = device
            .create_resident_kernel_dispatch_2d(
                projection_spirv_words,
                &bindings,
                output_spec.projection_workgroup_count_x,
                workgroup_count_y,
                output_spec.projection_local_size_x,
                std::mem::size_of::<u32>() as u32,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut sampler_views = Vec::with_capacity(batch_capacity);
        for (batch_index, sampler) in sampler_lanes.iter().copied().enumerate() {
            let logits_byte_offset = output_spec
                .logits_byte_capacity
                .checked_mul(batch_index)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "batched sampler logits offset overflowed".to_string(),
                    ))
                })?;
            sampler_views.push(
                sampler
                    .create_logits_view(
                        device,
                        &batched_logits_buffer,
                        logits_byte_offset,
                        sampler_kernels,
                        sampler_spec,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?,
            );
        }
        Ok(Self {
            batch_capacity,
            normalized_frames_buffer,
            _batched_logits_buffer: batched_logits_buffer,
            norm_dispatch,
            projection_dispatch,
            projection_sequence: device
                .create_resident_kernel_sequence()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            sampler_views,
        })
    }

    fn project(
        &self,
        device: &VulkanComputeDevice,
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if batch_width == 0 || batch_width > self.batch_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched output projection capacity {} cannot process {} frames",
                    self.batch_capacity, batch_width
                )),
            ));
        }
        let batch_width = u32::try_from(batch_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched output projection width exceeds u32".to_string(),
            ))
        })?;
        device
            .run_resident_kernel_sequence(
                &self.projection_sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(
                        &self.norm_dispatch,
                        &batch_width.to_le_bytes(),
                    ),
                    VulkanResidentKernelSequenceStep::new(
                        &self.projection_dispatch,
                        &batch_width.to_le_bytes(),
                    ),
                ],
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn sample_batch(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let batch_width = input_token_ids.len();
        let stream_ticks =
            consecutive_component_batch_stream_ticks(start_stream_tick, batch_width)?;
        let dynamic_state_capacities =
            vec![dynamic_state_capacity_activations; batch_width];
        let token_prefixes = (0..batch_width)
            .map(|batch_index| &input_token_ids[..=batch_index])
            .collect::<Vec<_>>();
        self.sample_lanes(
            device,
            &token_prefixes,
            &stream_ticks,
            &dynamic_state_capacities,
        )
    }

    fn sample_independent_streams(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        stream_ticks: &[u64],
        dynamic_state_capacities: &[u32],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let token_prefixes = vec![&[][..]; input_token_ids.len()];
        self.sample_lanes(
            device,
            &token_prefixes,
            stream_ticks,
            dynamic_state_capacities,
        )
    }

    fn sample_lanes(
        &self,
        device: &VulkanComputeDevice,
        token_prefixes: &[&[u32]],
        stream_ticks: &[u64],
        dynamic_state_capacities: &[u32],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let batch_width = token_prefixes.len();
        if batch_width == 0 || batch_width > self.sampler_views.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched output projection has {} sampler lanes, cannot sample {batch_width}",
                    self.sampler_views.len()
                )),
            ));
        }
        if stream_ticks.len() != batch_width
            || dynamic_state_capacities.len() != batch_width
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched sampler has {batch_width} token lanes, {} stream ticks, and {} state capacities",
                    stream_ticks.len(),
                    dynamic_state_capacities.len()
                )),
            ));
        }
        let submission_batch = VulkanResidentQueueSubmissionBatch::new();
        for (batch_index, view) in self.sampler_views.iter().take(batch_width).enumerate() {
            view.prepare_token_state(device, token_prefixes[batch_index])
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
            view.prepare_stream_tick(
                stream_ticks[batch_index],
                dynamic_state_capacities[batch_index],
            )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            view.record(device)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            submission_batch
                .enqueue_recorded_sequence(
                    device,
                    &view.sequence,
                    &[],
                    &[],
                    batch_index + 1 == batch_width,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        submission_batch
            .mount()
            .and_then(|template| template.submit_with_timeline_value_offset(0).map(|_| ()))
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        device
            .wait_resident_kernel_sequence(&self.sampler_views[batch_width - 1].sequence)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }
}
