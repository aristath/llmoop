/// A maximal sequence of dispatch stages that can execute without crossing a
/// cable transport boundary. The sequence keeps its pipelines, descriptors,
/// command buffer, and fence resident for the lifetime of the mounted model.
pub struct VulkanMountedPlacedResidentDispatchSegmentRunner {
    pub start_stage_index: usize,
    pub end_stage_index: usize,
    pub dispatch_count: usize,
    dispatches: Vec<VulkanMountedPlacedResidentPedalDispatch>,
    stream_control_buffer: Arc<VulkanResidentBuffer>,
    sequences: RefCell<BTreeMap<u8, VulkanResidentKernelSequence>>,
    feedback_sequences: RefCell<Vec<VulkanResidentKernelSequence>>,
}

impl VulkanMountedPlacedResidentDispatchSegmentRunner {
    fn from_dispatch_stages(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        stages: &[VulkanMountedPlacedStreamTickStage],
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        let start_stage_index = stages
            .first()
            .map(VulkanMountedPlacedStreamTickStage::stage_index)
            .ok_or_else(|| {
                VulkanMountedPlacedResidentKernelDispatchError::EmptyDispatchSegment {
                    device_id: mounted.device_id().to_string(),
                }
            })?;
        let end_stage_index = stages
            .last()
            .map(VulkanMountedPlacedStreamTickStage::stage_index)
            .and_then(|index| index.checked_add(1))
            .ok_or_else(|| {
                VulkanMountedPlacedResidentKernelDispatchError::DispatchSegmentStageOverflow {
                    device_id: mounted.device_id().to_string(),
                }
            })?;
        let mut dispatches = Vec::with_capacity(stages.len());

        for stage in stages {
            let VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index,
                dispatch,
            } = stage
            else {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::NonDispatchStageInSegment {
                        device_id: mounted.device_id().to_string(),
                        stage_index: stage.stage_index(),
                    },
                );
            };
            let bound_dispatch = mounted_bound_plan
                .dispatches
                .iter()
                .find(|bound| bound.dispatch_index == dispatch.dispatch_index)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingSegmentDispatch {
                        device_id: mounted.device_id().to_string(),
                        stage_index: *stage_index,
                        dispatch_index: dispatch.dispatch_index,
                    }
                })?;
            let resident_dispatch = mounted.create_resident_kernel_dispatch_for_bound_dispatch(
                device,
                bound_dispatch,
                loaded_manifest,
            )?;
            dispatches.push(VulkanMountedPlacedResidentPedalDispatch {
                dispatch_index: bound_dispatch.dispatch_index,
                kernel_id: bound_dispatch.kernel_id.clone(),
                pedal_id: bound_dispatch.pedal_id.clone(),
                node_id: bound_dispatch.node_id.clone(),
                op: bound_dispatch.op.clone(),
                reusable_family_id: bound_dispatch.reusable_family_id.clone(),
                push_constants: bound_dispatch.push_constants.clone(),
                resident_dispatch,
            });
        }

        Ok(Self {
            start_stage_index,
            end_stage_index,
            dispatch_count: dispatches.len(),
            dispatches,
            stream_control_buffer: mounted.stream_control_buffer.clone(),
            sequences: RefCell::new(BTreeMap::from([(
                0,
                device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?,
            )])),
            feedback_sequences: RefCell::new(Vec::new()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_with_stream_control_and_timeline_semaphores<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        sequence_variant: u8,
        feedback_lane: Option<usize>,
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if let Some(feedback_lane) = feedback_lane {
            while self.feedback_sequences.borrow().len() <= feedback_lane {
                let sequence = device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
                self.feedback_sequences.borrow_mut().push(sequence);
            }
            let feedback_sequences = self.feedback_sequences.borrow();
            let sequence = feedback_sequences
                .get(feedback_lane)
                .expect("resident feedback sequence lane was initialized");
            return self.record_and_submit_sequence(
                device,
                sequence,
                control,
                prefix_dispatches,
                suffix_dispatches,
                snapshot_copies,
                wait_points,
                signal_points,
                signal_completion,
                submission_batch,
            );
        }
        debug_assert!(snapshot_copies.is_empty());
        if !self.sequences.borrow().contains_key(&sequence_variant) {
            let sequence = device
                .create_resident_kernel_sequence()
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            self.sequences
                .borrow_mut()
                .entry(sequence_variant)
                .or_insert(sequence);
        }
        let sequences = self.sequences.borrow();
        let sequence = sequences
            .get(&sequence_variant)
            .expect("resident sequence variant was initialized");
        self.record_and_submit_sequence(
            device,
            sequence,
            control,
            prefix_dispatches,
            suffix_dispatches,
            snapshot_copies,
            wait_points,
            signal_points,
            signal_completion,
            submission_batch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_and_submit_sequence<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if self.can_replay_recorded_commands(sequence, prefix_dispatches, suffix_dispatches) {
            return self.submit_recorded_sequence(
                device,
                sequence,
                wait_points,
                signal_points,
                signal_completion,
                submission_batch,
            );
        }
        let push_constants = self
            .dispatches
            .iter()
            .map(|dispatch| stream_control_push_constant_bytes(&dispatch.push_constants, control))
            .collect::<Result<Vec<_>, _>>()?;
        let mut steps = Vec::with_capacity(
            prefix_dispatches.len() + self.dispatches.len() + suffix_dispatches.len(),
        );
        steps.extend(
            prefix_dispatches
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        steps.extend(self.dispatches.iter().zip(&push_constants).map(
            |(dispatch, push_constants)| {
                VulkanResidentKernelSequenceStep::new(&dispatch.resident_dispatch, push_constants)
            },
        ));
        steps.extend(
            suffix_dispatches
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        if snapshot_copies.is_empty() {
            device.record_resident_kernel_sequence(sequence, &steps)
        } else {
            device.record_resident_kernel_sequence_with_snapshot_copies(
                sequence,
                &steps,
                snapshot_copies,
            )
        }
        .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        self.submit_recorded_sequence(
            device,
            sequence,
            wait_points,
            signal_points,
            signal_completion,
            submission_batch,
        )
    }

    fn submit_recorded_sequence<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if let Some(submission_batch) = submission_batch {
            submission_batch.enqueue_recorded_sequence(
                device,
                sequence,
                wait_points,
                signal_points,
                signal_completion,
            )
        } else if signal_completion {
            device.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                sequence,
                wait_points,
                signal_points,
            )
        } else {
            device.submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                sequence,
                wait_points,
                signal_points,
            )
        }
        .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }

    fn can_replay_recorded_commands(
        &self,
        sequence: &VulkanResidentKernelSequence,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
    ) -> bool {
        std::env::var_os("NERVE_VK_PERF_LOGGER").is_none()
            && sequence.has_recorded_commands()
            && self
                .dispatches
                .iter()
                .all(|dispatch| dispatch.push_constants.is_empty())
            && prefix_dispatches
                .iter()
                .all(|dispatch| dispatch.push_constant_byte_count() == 0)
            && suffix_dispatches
                .iter()
                .all(|dispatch| dispatch.push_constant_byte_count() == 0)
    }

    fn wait_submitted(
        &self,
        device: &VulkanComputeDevice,
        sequence_variant: u8,
        feedback_lane: Option<usize>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if let Some(feedback_lane) = feedback_lane {
            let feedback_sequences = self.feedback_sequences.borrow();
            let sequence = feedback_sequences
                .get(feedback_lane)
                .expect("submitted resident feedback sequence lane must remain mounted");
            return device
                .wait_resident_kernel_sequence(sequence)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan);
        }
        let sequences = self.sequences.borrow();
        let sequence = sequences
            .get(&sequence_variant)
            .expect("submitted resident sequence variant must remain mounted");
        device
            .wait_resident_kernel_sequence(sequence)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }

    fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        sequence_variant: u8,
        capture_execution_trace: bool,
    ) -> Result<
        Vec<VulkanMountedPlacedResidentPedalRun>,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        self.stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;

        if !self.sequences.borrow().contains_key(&sequence_variant) {
            let sequence = device
                .create_resident_kernel_sequence()
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            self.sequences
                .borrow_mut()
                .entry(sequence_variant)
                .or_insert(sequence);
        }
        let sequences = self.sequences.borrow();
        let sequence = sequences
            .get(&sequence_variant)
            .expect("resident sequence variant was initialized");

        let execution_start = capture_execution_trace.then(Instant::now);
        if self.can_replay_recorded_commands(sequence, prefix_dispatches, suffix_dispatches) {
            device
                .run_recorded_resident_kernel_sequence(sequence)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            return Ok(execution_start
                .map(|start| {
                    let execution_time_ns =
                        u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    self.completed_pedal_runs(execution_time_ns)
                })
                .unwrap_or_default());
        }

        let push_constants = self
            .dispatches
            .iter()
            .map(|dispatch| stream_control_push_constant_bytes(&dispatch.push_constants, control))
            .collect::<Result<Vec<_>, _>>()?;
        let mut steps = Vec::with_capacity(
            prefix_dispatches.len() + self.dispatches.len() + suffix_dispatches.len(),
        );
        steps.extend(
            prefix_dispatches
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        steps.extend(self.dispatches.iter().zip(&push_constants).map(
            |(dispatch, push_constants)| {
                VulkanResidentKernelSequenceStep::new(&dispatch.resident_dispatch, push_constants)
            },
        ));
        steps.extend(
            suffix_dispatches
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        device
            .run_resident_kernel_sequence(sequence, &steps)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        Ok(execution_start
            .map(|start| {
                let execution_time_ns =
                    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.completed_pedal_runs(execution_time_ns)
            })
            .unwrap_or_default())
    }

    fn completed_pedal_runs(
        &self,
        execution_time_ns: u64,
    ) -> Vec<VulkanMountedPlacedResidentPedalRun> {
        let mut pedal_runs = Vec::<VulkanMountedPlacedResidentPedalRun>::new();
        for (dispatch_offset, dispatch) in self.dispatches.iter().enumerate() {
            let dispatch_run = VulkanMountedPlacedResidentPedalDispatchRun {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                descriptor_count: dispatch.resident_dispatch.descriptor_count(),
                workgroup_count_x: dispatch.resident_dispatch.workgroup_count_x(),
                push_constant_byte_count: dispatch.resident_dispatch.push_constant_byte_count(),
                // A composed segment has one measurable execution boundary.
                run_time_ns: if dispatch_offset == 0 {
                    execution_time_ns
                } else {
                    0
                },
            };
            if let Some(pedal_run) = pedal_runs
                .last_mut()
                .filter(|run| run.pedal_id == dispatch.pedal_id)
            {
                pedal_run.dispatch_runs.push(dispatch_run);
            } else {
                pedal_runs.push(VulkanMountedPlacedResidentPedalRun {
                    pedal_id: dispatch.pedal_id.clone(),
                    dispatch_runs: vec![dispatch_run],
                });
            }
        }
        pedal_runs
    }
}
