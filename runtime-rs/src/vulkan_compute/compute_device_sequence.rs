impl VulkanComputeDevice {
    pub fn run_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.run_resident_kernel_sequence_with_snapshot_copies(sequence, steps, &[])
    }

    pub fn run_recorded_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_recorded_resident_kernel_sequence(sequence)?;
        self.wait_resident_kernel_sequence(sequence)
    }

    pub fn submit_recorded_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(sequence, &[], &[])
    }

    pub fn submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
        &self,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        self.submit_command_buffer_with_timeline_semaphores(
            sequence.command_buffer,
            Some(sequence.completion_fence),
            wait_points,
            signal_points,
            "resident kernel sequence",
        )
    }

    pub fn submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
        &self,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        self.submit_command_buffer_with_timeline_semaphores(
            sequence.command_buffer,
            None,
            wait_points,
            signal_points,
            "resident kernel sequence",
        )
    }

    fn submit_resident_kernel_sequence_and_wait(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_resident_kernel_sequence(sequence)?;
        self.wait_resident_kernel_sequence(sequence)
    }

    fn submit_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_command_buffer_with_timeline_semaphores(
            sequence.command_buffer,
            Some(sequence.completion_fence),
            &[],
            &[],
            "resident kernel sequence",
        )
    }

    fn submit_command_buffer_with_timeline_semaphores(
        &self,
        command_buffer: vk::CommandBuffer,
        completion_fence: Option<vk::Fence>,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        label: &str,
    ) -> Result<(), VulkanError> {
        for point in wait_points.iter().chain(signal_points) {
            self.validate_local_timeline_semaphore(point.semaphore)?;
        }
        let wait_infos = wait_points
            .iter()
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.semaphore.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        let signal_infos = signal_points
            .iter()
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.semaphore.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        unsafe {
            if let Some(completion_fence) = completion_fence {
                self.device
                    .reset_fences(&[completion_fence])
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to reset {label} completion fence: {error:?}"
                        ))
                    })?;
            }
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
            let submit_info = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .command_buffer_infos(&command_buffers)
                .signal_semaphore_infos(&signal_infos)];
            self.device
                .queue_submit2(
                    self.queue,
                    &submit_info,
                    completion_fence.unwrap_or(vk::Fence::null()),
                )
                .map_err(|error| VulkanError(format!("failed to submit {label}: {error:?}")))?;
            RESIDENT_SEQUENCE_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

}

impl VulkanResidentQueueSubmitter {
    fn submit_prepared_resident_queue_batch(
        &self,
        submissions: &[VulkanPreparedResidentQueueSubmission],
        timeline_value_offset: u64,
    ) -> Result<(), VulkanError> {
        if submissions.is_empty() {
            return Ok(());
        }
        let wait_infos = submissions
            .iter()
            .map(|submission| {
                submission
                    .wait_points
                    .iter()
                    .map(|(semaphore, value)| {
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*semaphore)
                            .value(
                                offset_timeline_value(*value, timeline_value_offset)
                                    .expect("resident submission template offsets were validated"),
                            )
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let command_infos =
            submissions
                .iter()
                .map(|submission| {
                    [vk::CommandBufferSubmitInfo::default()
                        .command_buffer(submission.command_buffer)]
                })
                .collect::<Vec<_>>();
        let signal_infos = submissions
            .iter()
            .map(|submission| {
                submission
                    .signal_points
                    .iter()
                    .map(|(semaphore, value)| {
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*semaphore)
                            .value(
                                offset_timeline_value(*value, timeline_value_offset)
                                    .expect("resident submission template offsets were validated"),
                            )
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let submit_infos = (0..submissions.len())
            .map(|index| {
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait_infos[index])
                    .command_buffer_infos(&command_infos[index])
                    .signal_semaphore_infos(&signal_infos[index])
            })
            .collect::<Vec<_>>();
        let mut completion_fences = Vec::new();
        for fence in submissions
            .iter()
            .filter_map(|submission| submission.completion_fence)
        {
            if !completion_fences.contains(&fence) {
                completion_fences.push(fence);
            }
        }
        unsafe {
            if !completion_fences.is_empty() {
                self.device
                    .reset_fences(&completion_fences)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to reset resident queue batch completion fences: {error:?}"
                        ))
                    })?;
            }
            let batch_fence = if completion_fences.len() == 1 {
                completion_fences[0]
            } else {
                vk::Fence::null()
            };
            self.device
                .queue_submit2(self.queue, &submit_infos, batch_fence)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to submit resident queue batch containing {} commands: {error:?}",
                        submissions.len()
                    ))
                })?;
            RESIDENT_QUEUE_BATCH_SUBMITS.fetch_add(1, Ordering::Relaxed);
            RESIDENT_QUEUE_BATCH_COMMANDS.fetch_add(
                u64::try_from(submissions.len()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            if completion_fences.len() > 1 {
                let completion_submit = [vk::SubmitInfo2::default()];
                for fence in completion_fences {
                    self.device
                        .queue_submit2(self.queue, &completion_submit, fence)
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to submit resident queue batch completion fence: {error:?}"
                            ))
                        })?;
                    RESIDENT_QUEUE_BATCH_SUBMITS.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }
}

impl VulkanComputeDevice {
    pub fn wait_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .wait_for_fences(&[sequence.completion_fence], true, u64::MAX)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed waiting for resident kernel sequence: {error:?}"
                    ))
                })?;
            RESIDENT_SEQUENCE_FENCE_WAITS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn run_resident_kernel_sequence_with_snapshot_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, &[], steps, snapshot_copies, true)
    }

    pub fn run_resident_kernel_sequence_with_input_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, input_copies, steps, &[], true)
    }

    pub fn record_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, &[], steps, &[], false)
    }

    pub fn record_resident_kernel_sequence_with_snapshot_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, &[], steps, snapshot_copies, false)
    }

    fn prepare_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        execute: bool,
    ) -> Result<(), VulkanError> {
        if steps.is_empty() {
            return Err(VulkanError(
                "resident kernel sequence must contain at least one dispatch".to_string(),
            ));
        }
        for (step_index, step) in steps.iter().enumerate() {
            if step.dispatch.pipeline_key.push_constant_byte_count
                != step.push_constants.len() as u32
            {
                return Err(VulkanError(format!(
                    "resident kernel sequence step {step_index} expects {} push-constant bytes, got {}",
                    step.dispatch.pipeline_key.push_constant_byte_count,
                    step.push_constants.len()
                )));
            }
        }
        if let Some(copy) = snapshot_copies
            .iter()
            .find(|copy| copy.after_step_index >= steps.len())
        {
            return Err(VulkanError(format!(
                "resident snapshot follows step {}, but sequence contains {} steps",
                copy.after_step_index,
                steps.len()
            )));
        }

        unsafe {
            RESIDENT_SEQUENCE_PREPARE_CALLS.fetch_add(1, Ordering::Relaxed);
            let profiling_enabled = execute && std::env::var_os("NERVE_VK_PERF_LOGGER").is_some();
            let command_buffer_matches = !profiling_enabled
                && sequence
                    .recorded_input_copies
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == input_copies.len()
                            && recorded
                                .iter()
                                .zip(input_copies)
                                .all(|(recorded, copy)| *recorded == copy.recorded())
                    })
                && sequence
                    .recorded_steps
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == steps.len()
                            && recorded.iter().zip(steps).all(|(recorded, step)| {
                                recorded.pipeline == step.dispatch.pipeline
                                    && recorded.descriptor_set == step.dispatch.descriptor_set
                                    && recorded.workgroup_count_x == step.dispatch.workgroup_count_x
                                    && recorded.workgroup_count_y == step.dispatch.workgroup_count_y
                                    && recorded.base_workgroup_z == step.dispatch.base_workgroup_z
                                    && recorded.push_constants == step.push_constants
                            })
                    })
                && sequence
                    .recorded_snapshot_copies
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == snapshot_copies.len()
                            && recorded
                                .iter()
                                .zip(snapshot_copies)
                                .all(|(recorded, copy)| *recorded == copy.recorded())
                    });
            if command_buffer_matches {
                RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
            } else {
                RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
            }
            let host_start = profiling_enabled.then(Instant::now);
            let query_count = u32::try_from(steps.len() + 1).map_err(|_| {
                VulkanError("resident kernel timestamp count overflowed".to_string())
            })?;
            let query_pool = if profiling_enabled {
                let query_pool_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(query_count);
                Some(
                    self.device
                        .create_query_pool(&query_pool_info, None)
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to create resident kernel timestamp pool: {error:?}"
                            ))
                        })?,
                )
            } else {
                None
            };

            if !command_buffer_matches {
                self.device
                    .reset_command_buffer(
                        sequence.command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to reset resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;

                let command_begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                self.device
                    .begin_command_buffer(sequence.command_buffer, &command_begin)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to begin resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;
            }

            if !command_buffer_matches && let Some(query_pool) = query_pool {
                self.device.cmd_reset_query_pool(
                    sequence.command_buffer,
                    query_pool,
                    0,
                    query_count,
                );
                self.device.cmd_write_timestamp(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }

            if !command_buffer_matches {
                if input_copies.is_empty() {
                    // A resident sequence is an independently submitted circuit unit. Its
                    // inputs may have been produced by the host, a transfer, or an earlier
                    // compute sequence on this queue, so establish the full producer-to-
                    // consumer dependency at the sequence boundary.
                    let input_visibility_barrier = [vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::HOST_WRITE
                                | vk::AccessFlags::TRANSFER_WRITE
                                | vk::AccessFlags::SHADER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )];
                    self.device.cmd_pipeline_barrier(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::HOST
                            | vk::PipelineStageFlags::TRANSFER
                            | vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &input_visibility_barrier,
                        &[],
                        &[],
                    );
                } else {
                    let input_to_transfer = [vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::HOST_WRITE
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::TRANSFER_WRITE,
                        )
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                    self.device.cmd_pipeline_barrier(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::HOST
                            | vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &input_to_transfer,
                        &[],
                        &[],
                    );
                    for (copy_index, input_copy) in input_copies.iter().enumerate() {
                        if copy_index != 0 {
                            let transfer_order = [vk::MemoryBarrier::default()
                                .src_access_mask(
                                    vk::AccessFlags::TRANSFER_READ
                                        | vk::AccessFlags::TRANSFER_WRITE,
                                )
                                .dst_access_mask(
                                    vk::AccessFlags::TRANSFER_READ
                                        | vk::AccessFlags::TRANSFER_WRITE,
                                )];
                            self.device.cmd_pipeline_barrier(
                                sequence.command_buffer,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::DependencyFlags::empty(),
                                &transfer_order,
                                &[],
                                &[],
                            );
                        }
                        let regions = [vk::BufferCopy {
                            src_offset: input_copy.source_offset(),
                            dst_offset: input_copy.destination_offset(),
                            size: input_copy.byte_len(),
                        }];
                        self.device.cmd_copy_buffer(
                            sequence.command_buffer,
                            input_copy.source(),
                            input_copy.destination(),
                            &regions,
                        );
                    }
                    let transfer_to_compute = [vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )];
                    self.device.cmd_pipeline_barrier(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &transfer_to_compute,
                        &[],
                        &[],
                    );
                }
            }

            let mut pending_buffer_accesses = Vec::<VulkanResidentKernelBufferAccessRecord>::new();
            if !command_buffer_matches {
                for (step_index, step) in steps.iter().enumerate() {
                    let dependencies = take_resident_kernel_buffer_dependencies(
                        &mut pending_buffer_accesses,
                        &step.dispatch.buffer_accesses,
                    );
                    if !dependencies.is_empty() {
                        let buffer_barriers = dependencies
                            .iter()
                            .map(|dependency| {
                                vk::BufferMemoryBarrier::default()
                                    .src_access_mask(
                                        vk::AccessFlags::SHADER_READ
                                            | vk::AccessFlags::SHADER_WRITE,
                                    )
                                    .dst_access_mask(
                                        vk::AccessFlags::SHADER_READ
                                            | vk::AccessFlags::SHADER_WRITE,
                                    )
                                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .buffer(dependency.buffer)
                                    .offset(0)
                                    .size(vk::WHOLE_SIZE)
                            })
                            .collect::<Vec<_>>();
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &buffer_barriers,
                            &[],
                        );
                    }

                    self.device.cmd_bind_pipeline(
                        sequence.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        step.dispatch.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        sequence.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        step.dispatch.pipeline_layout,
                        0,
                        &[step.dispatch.descriptor_set],
                        &[],
                    );
                    if !step.push_constants.is_empty() {
                        self.device.cmd_push_constants(
                            sequence.command_buffer,
                            step.dispatch.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            step.push_constants,
                        );
                    }
                    if step.dispatch.base_workgroup_z == 0 {
                        self.device.cmd_dispatch(
                            sequence.command_buffer,
                            step.dispatch.workgroup_count_x,
                            step.dispatch.workgroup_count_y,
                            1,
                        );
                    } else {
                        self.device.cmd_dispatch_base(
                            sequence.command_buffer,
                            0,
                            0,
                            step.dispatch.base_workgroup_z,
                            step.dispatch.workgroup_count_x,
                            step.dispatch.workgroup_count_y,
                            1,
                        );
                    }
                    merge_resident_kernel_buffer_accesses(
                        &mut pending_buffer_accesses,
                        &step.dispatch.buffer_accesses,
                    );
                    if let Some(query_pool) = query_pool {
                        self.device.cmd_write_timestamp(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query_pool,
                            u32::try_from(step_index + 1).map_err(|_| {
                                VulkanError(
                                    "resident kernel timestamp index overflowed".to_string(),
                                )
                            })?,
                        );
                    }

                    let step_snapshot_copies = snapshot_copies
                        .iter()
                        .filter(|copy| copy.after_step_index == step_index)
                        .collect::<Vec<_>>();
                    if !step_snapshot_copies.is_empty() {
                        let compute_to_transfer = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &compute_to_transfer,
                            &[],
                            &[],
                        );
                        for copy in step_snapshot_copies {
                            let regions = [vk::BufferCopy {
                                src_offset: copy.source_offset,
                                dst_offset: copy.destination_offset,
                                size: copy.byte_len,
                            }];
                            self.device.cmd_copy_buffer(
                                sequence.command_buffer,
                                copy.source.buffer,
                                copy.destination.buffer,
                                &regions,
                            );
                        }
                        let transfer_to_compute = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &transfer_to_compute,
                            &[],
                            &[],
                        );
                        pending_buffer_accesses.clear();
                    }
                }

                let host_visibility_barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                self.device.cmd_pipeline_barrier(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &host_visibility_barrier,
                    &[],
                    &[],
                );

                self.device
                    .end_command_buffer(sequence.command_buffer)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to end resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;

                if profiling_enabled {
                    *sequence.recorded_input_copies.borrow_mut() = None;
                    *sequence.recorded_steps.borrow_mut() = None;
                    *sequence.recorded_snapshot_copies.borrow_mut() = None;
                } else {
                    *sequence.recorded_input_copies.borrow_mut() = Some(
                        input_copies
                            .iter()
                            .copied()
                            .map(VulkanResidentKernelSequenceInputCopy::recorded)
                            .collect(),
                    );
                    *sequence.recorded_steps.borrow_mut() = Some(
                        steps
                            .iter()
                            .map(|step| VulkanResidentKernelRecordedStep {
                                pipeline: step.dispatch.pipeline,
                                descriptor_set: step.dispatch.descriptor_set,
                                workgroup_count_x: step.dispatch.workgroup_count_x,
                                workgroup_count_y: step.dispatch.workgroup_count_y,
                                base_workgroup_z: step.dispatch.base_workgroup_z,
                                push_constants: step.push_constants.to_vec(),
                            })
                            .collect(),
                    );
                    *sequence.recorded_snapshot_copies.borrow_mut() = Some(
                        snapshot_copies
                            .iter()
                            .copied()
                            .map(VulkanResidentKernelSequenceSnapshotCopy::recorded)
                            .collect(),
                    );
                }
            }

            if !execute {
                return Ok(());
            }

            self.submit_resident_kernel_sequence_and_wait(sequence)?;
            let host_submit_wait_ns = host_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();

            if let Some(query_pool) = query_pool {
                let mut timestamps = vec![0u64; query_count as usize];
                let result = self.device.get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                );
                self.device.destroy_query_pool(query_pool, None);
                result.map_err(|error| {
                    VulkanError(format!(
                        "failed to read resident kernel timestamps: {error:?}"
                    ))
                })?;
                print_resident_kernel_timestamp_summary(
                    steps,
                    &timestamps,
                    sequence.timestamp_period_ns,
                    host_submit_wait_ns,
                );
            }

            Ok(())
        }
    }
}
