struct VulkanResidentComponentBatchSliceRunner {
    lane_capacity: usize,
    signal_buffers: Vec<VulkanComponentBatchSignalBuffer>,
    signal_buffer_indices: BTreeMap<VulkanComponentBatchSignalKey, usize>,
    stream_control_buffers: Vec<VulkanResidentBuffer>,
    steps: Vec<VulkanComponentBatchDispatchStep>,
    execution_units: Vec<VulkanComponentBatchExecutionUnit>,
    sequences: Vec<VulkanResidentKernelSequence>,
    quantum_calibrator: Rc<RefCell<RuntimeExecutionQuantumCalibrator>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionUnit {
    LocalComponent {
        component_id: String,
        step_start: usize,
        step_end: usize,
    },
    DistributedDispatch {
        dispatch_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchDispatchSpan {
    component_id: String,
    dispatch_index: usize,
    step_start: usize,
    step_end: usize,
    distributed: bool,
}

fn finish_component_batch_local_execution_unit(
    execution_units: &mut Vec<VulkanComponentBatchExecutionUnit>,
    component_id: &str,
    step_start: usize,
    step_end: usize,
) {
    if step_start < step_end {
        execution_units.push(VulkanComponentBatchExecutionUnit::LocalComponent {
            component_id: component_id.to_string(),
            step_start,
            step_end,
        });
    }
}

fn component_batch_execution_units(
    dispatch_spans: &[VulkanComponentBatchDispatchSpan],
) -> Result<Vec<VulkanComponentBatchExecutionUnit>, VulkanError> {
    let mut execution_units = Vec::new();
    let mut current_component_id = None::<&str>;
    let mut local_step_start = 0usize;
    let mut expected_step_start = 0usize;
    let mut previous_dispatch_index = None;
    for span in dispatch_spans {
        if span.step_start != expected_step_start {
            return Err(VulkanError(format!(
                "component batch dispatch {} starts at step {}, expected {expected_step_start}",
                span.dispatch_index, span.step_start
            )));
        }
        if previous_dispatch_index.is_some_and(|previous| previous >= span.dispatch_index) {
            return Err(VulkanError(format!(
                "component batch dispatch indices are not strictly increasing at {}",
                span.dispatch_index
            )));
        }
        if span.distributed && span.step_end != span.step_start {
            return Err(VulkanError(format!(
                "distributed component batch dispatch {} owns local steps {}..{}",
                span.dispatch_index, span.step_start, span.step_end
            )));
        }
        if !span.distributed && span.step_end <= span.step_start {
            return Err(VulkanError(format!(
                "local component batch dispatch {} has no executable steps",
                span.dispatch_index
            )));
        }
        previous_dispatch_index = Some(span.dispatch_index);
        expected_step_start = span.step_end;
        if span.distributed {
            if let Some(component_id) = current_component_id.take() {
                finish_component_batch_local_execution_unit(
                    &mut execution_units,
                    component_id,
                    local_step_start,
                    span.step_start,
                );
            }
            execution_units.push(VulkanComponentBatchExecutionUnit::DistributedDispatch {
                dispatch_index: span.dispatch_index,
            });
            continue;
        }
        if current_component_id != Some(span.component_id.as_str()) {
            if let Some(component_id) = current_component_id {
                finish_component_batch_local_execution_unit(
                    &mut execution_units,
                    component_id,
                    local_step_start,
                    span.step_start,
                );
            }
            current_component_id = Some(&span.component_id);
            local_step_start = span.step_start;
        }
    }
    if let (Some(component_id), Some(last_span)) = (current_component_id, dispatch_spans.last()) {
        finish_component_batch_local_execution_unit(
            &mut execution_units,
            component_id,
            local_step_start,
            last_span.step_end,
        );
    }
    Ok(execution_units)
}

fn component_batch_execution_units_for_distributed_groups(
    dispatch_spans: &[VulkanComponentBatchDispatchSpan],
    distributed_group_leaders: &BTreeSet<usize>,
) -> Result<Vec<VulkanComponentBatchExecutionUnit>, VulkanError> {
    let mut execution_units = component_batch_execution_units(dispatch_spans)?;
    execution_units.retain(|unit| match unit {
        VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index } => {
            distributed_group_leaders.contains(dispatch_index)
        }
        VulkanComponentBatchExecutionUnit::LocalComponent { .. } => true,
    });
    Ok(execution_units)
}

impl VulkanResidentComponentBatchSliceRunner {
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device: &VulkanComputeDevice,
        slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        distributed_execution_plan: &VulkanDistributedExecutionPlan,
        quantum_calibrator: Rc<RefCell<RuntimeExecutionQuantumCalibrator>>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if lane_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("component batch lane capacity is zero".to_string()),
            ));
        }
        let (signal_buffer_indices, signal_buffer_plan) =
            component_batch_signal_buffer_plan(&slice.mounted, &slice.mounted_bound.dispatches)?;
        let mut shared_device_ids_by_buffer = BTreeMap::<usize, BTreeSet<String>>::new();
        for dispatch in distributed_execution_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.owner_device_id == slice.device_id)
        {
            for activation in std::iter::once(&dispatch.input_activation)
                .chain(&dispatch.auxiliary_input_activations)
                .chain(std::iter::once(&dispatch.output_activation))
            {
                let key = VulkanComponentBatchSignalKey::Activation {
                    component_id: activation.component_id.clone(),
                    signal_id: activation.signal_id.clone(),
                };
                let buffer_index = *signal_buffer_indices.get(&key).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed component batch has no signal buffer for {key:?}"
                    )))
                })?;
                shared_device_ids_by_buffer
                    .entry(buffer_index)
                    .or_default()
                    .extend(dispatch.shards.iter().map(|shard| shard.device_id.clone()));
            }
        }
        let mut signal_buffers = Vec::<VulkanComponentBatchSignalBuffer>::new();
        for (buffer_index, allocation) in signal_buffer_plan.into_iter().enumerate() {
            let byte_capacity = allocation
                .frame_byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "component batch signal capacity overflowed".to_string(),
                    ))
                })?;
            let shared_device_ids = shared_device_ids_by_buffer.get(&buffer_index);
            let (mut buffer, shared_device_buffers) =
                if let Some(shared_device_ids) = shared_device_ids {
                    if !shared_device_ids.contains(&slice.device_id) {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "distributed component batch buffer {buffer_index} omits owner {:?}",
                                slice.device_id
                            )),
                        ));
                    }
                    let peers = shared_device_ids
                        .iter()
                        .filter(|device_id| *device_id != &slice.device_id)
                        .map(|device_id| {
                            devices
                                .get(device_id)
                                .map(|device| device.as_ref())
                                .ok_or_else(|| {
                                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                        device_id: device_id.clone(),
                                    }
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let shared_allocation = device
                        .create_shared_host_allocation(&peers, byte_capacity)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    let mut shared_device_buffers = BTreeMap::new();
                    for device_id in shared_device_ids {
                        let import_device = devices.get(device_id).ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                device_id: device_id.clone(),
                            }
                        })?;
                        let imported = Arc::new(
                            import_device
                                .import_shared_host_buffer(Arc::clone(&shared_allocation))
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                        shared_device_buffers.insert(device_id.clone(), imported);
                    }
                    let owner_buffer = Arc::clone(
                        shared_device_buffers
                            .get(&slice.device_id)
                            .expect("validated distributed batch owner was imported"),
                    );
                    (owner_buffer, shared_device_buffers)
                } else {
                    let buffer = if allocation.host_visible {
                        // Cross-device edges are the one place where the batch must be
                        // host-addressable. The edge still moves once per device boundary,
                        // as one contiguous frame batch.
                        device.create_host_visible_resident_buffer(byte_capacity)
                    } else {
                        device.create_resident_buffer(byte_capacity)
                    }
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    (Arc::new(buffer), BTreeMap::new())
                };
            if allocation.host_visible {
                Arc::get_mut(&mut buffer)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "host-visible component batch edge buffer is unexpectedly shared"
                                .to_string(),
                        ))
                    })?
                    .persistently_map()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            signal_buffers.push(VulkanComponentBatchSignalBuffer {
                frame_byte_capacity: allocation.frame_byte_capacity,
                buffer,
                shared_device_buffers,
            });
        }

        let stream_control_buffers = (0..lane_capacity)
            .map(|_| {
                let mut buffer = device
                    .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                buffer
                    .persistently_map()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                Ok::<_, VulkanResidentInProcessPlacedRuntimeError>(buffer)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut steps = Vec::new();
        let mut dispatch_spans = Vec::with_capacity(slice.mounted_bound.dispatches.len());
        for dispatch in &slice.mounted_bound.dispatches {
            let dispatch_step_start = steps.len();
            let commits_state = component_batch_descriptors_commit_state(
                dispatch
                    .descriptors
                    .iter()
                    .map(|descriptor| &descriptor.usage),
            );
            if distributed_execution_plan
                .dispatches
                .iter()
                .any(|distributed| {
                    distributed.owner_device_id == slice.device_id
                        && distributed.dispatch_index == dispatch.dispatch_index
                })
            {
                dispatch_spans.push(VulkanComponentBatchDispatchSpan {
                    component_id: dispatch.component_id.clone(),
                    dispatch_index: dispatch.dispatch_index,
                    step_start: dispatch_step_start,
                    step_end: dispatch_step_start,
                    distributed: true,
                });
                continue;
            }
            let batch_artifact = select_component_batch_kernel_artifact(
                &slice.package_slice.batch_kernels,
                &dispatch.component_id,
                &dispatch.node_id,
                execution_mode,
                lane_capacity,
            );
            if let Some(batch_artifact) = batch_artifact {
                if batch_artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
                    && lane_capacity > batch_artifact.lane_tile_width
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "causal scan kernel {}.{} cannot execute {lane_capacity} lanes with tile width {}",
                            dispatch.component_id, dispatch.node_id, batch_artifact.lane_tile_width
                        )),
                    ));
                }
                if !dispatch.push_constants.is_empty() {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "component batch kernel {}.{} requires model-specific scalar values",
                            dispatch.component_id, dispatch.node_id
                        )),
                    ));
                }
                if batch_artifact.batch_mode == VulkanResidentComponentKernelBatchMode::WeightShared
                    && dispatch.descriptors.iter().any(|descriptor| {
                        matches!(
                            descriptor.usage,
                            VulkanKernelDescriptorUsage::StateRead
                                | VulkanKernelDescriptorUsage::StateWrite
                                | VulkanKernelDescriptorUsage::StateView
                        )
                    })
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "weight-shared component batch kernel {}.{} is not stateless",
                            dispatch.component_id, dispatch.node_id
                        )),
                    ));
                }
                let bindings = component_batch_bindings(
                    &slice.mounted,
                    dispatch,
                    &signal_buffers,
                    &signal_buffer_indices,
                    None,
                    None,
                )?;
                let workgroup_count_y = match batch_artifact.batch_mode {
                    VulkanResidentComponentKernelBatchMode::WeightShared => u32::try_from(
                        lane_capacity
                            .checked_add(batch_artifact.lane_tile_width - 1)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "component batch workgroup count overflowed".to_string(),
                                ))
                            })?
                            / batch_artifact.lane_tile_width,
                    )
                    .map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "component batch workgroup count exceeds u32".to_string(),
                        ))
                    })?,
                    VulkanResidentComponentKernelBatchMode::CausalScan => 1,
                    VulkanResidentComponentKernelBatchMode::SerialLanes => {
                        unreachable!("serial-lane kernels do not have component batch artifacts")
                    }
                };
                for stage in &batch_artifact.stages {
                    let batch_control_byte_count = batch_stage_control_byte_count(stage);
                    let resident = device
                        .create_resident_kernel_dispatch_2d_labeled(
                            &stage.spirv_words,
                            &bindings,
                            stage.workgroup_count_x,
                            workgroup_count_y,
                            stage.local_size_x,
                            batch_control_byte_count,
                            Some(vulkan_dispatch_semantic_label(dispatch, Some("batch"))),
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    steps.push(VulkanComponentBatchDispatchStep {
                        dispatch: resident,
                        batch_control_byte_count,
                        push_constants: Vec::new(),
                        lane_index: None,
                        commits_state,
                        snapshot_state_buffer_indices: BTreeSet::new(),
                    });
                }
                dispatch_spans.push(VulkanComponentBatchDispatchSpan {
                    component_id: dispatch.component_id.clone(),
                    dispatch_index: dispatch.dispatch_index,
                    step_start: dispatch_step_start,
                    step_end: steps.len(),
                    distributed: false,
                });
                continue;
            }

            let artifact = slice
                .package_slice
                .loaded_manifest
                .artifact(&dispatch.reusable_family_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "component batch scalar kernel {}.{} has no loaded artifact",
                        dispatch.component_id, dispatch.node_id
                    )))
                })?;
            let snapshot_state_buffer_indices = dispatch
                .descriptors
                .iter()
                .filter(|descriptor| {
                    matches!(
                        descriptor.usage,
                        VulkanKernelDescriptorUsage::StateWrite
                            | VulkanKernelDescriptorUsage::StateView
                    )
                })
                .filter_map(|descriptor| match &descriptor.target {
                    VulkanMountedPlacedBoundDescriptorTarget::Resident {
                        target:
                            VulkanBoundDescriptorTarget::StreamStateBuffer {
                                buffer_index,
                                static_bytes: Some(_),
                                ..
                            }
                            | VulkanBoundDescriptorTarget::StreamStateView {
                                buffer_index,
                                static_bytes: Some(_),
                                ..
                            },
                    } => Some(*buffer_index),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            for (lane_index, stream_control_buffer) in stream_control_buffers.iter().enumerate() {
                let bindings = component_batch_bindings(
                    &slice.mounted,
                    dispatch,
                    &signal_buffers,
                    &signal_buffer_indices,
                    Some(lane_index),
                    dispatch.uses_stream_tick.then_some(stream_control_buffer),
                )?;
                let resident = device
                    .create_resident_kernel_dispatch_labeled(
                        &artifact.words,
                        &bindings,
                        artifact.artifact.workgroup_count_x,
                        artifact.artifact.local_size_x,
                        push_constant_byte_count(&dispatch.push_constants).map_err(|error| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!("invalid component batch push constants: {error}"),
                            ))
                        })?,
                        Some(vulkan_dispatch_semantic_label(
                            dispatch,
                            Some(&format!("lane={lane_index}")),
                        )),
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                steps.push(VulkanComponentBatchDispatchStep {
                    dispatch: resident,
                    batch_control_byte_count: 0,
                    push_constants: dispatch.push_constants.clone(),
                    lane_index: Some(lane_index),
                    commits_state,
                    snapshot_state_buffer_indices: snapshot_state_buffer_indices.clone(),
                });
            }
            dispatch_spans.push(VulkanComponentBatchDispatchSpan {
                component_id: dispatch.component_id.clone(),
                dispatch_index: dispatch.dispatch_index,
                step_start: dispatch_step_start,
                step_end: steps.len(),
                distributed: false,
            });
        }
        let distributed_group_leaders = distributed_execution_plan
            .dispatch_groups
            .iter()
            .filter(|group| group.owner_device_id == slice.device_id)
            .map(|group| group.leader().dispatch_index)
            .collect::<BTreeSet<_>>();
        let execution_units = component_batch_execution_units_for_distributed_groups(
            &dispatch_spans,
            &distributed_group_leaders,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        let sequences = (0..execution_units
            .iter()
            .filter(|unit| matches!(unit, VulkanComponentBatchExecutionUnit::LocalComponent { .. }))
            .count())
            .map(|_| {
                device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            lane_capacity,
            signal_buffers,
            signal_buffer_indices,
            stream_control_buffers,
            steps,
            execution_units,
            sequences,
            quantum_calibrator,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_independent_candidates(
        &self,
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transaction: &VulkanResidentStateTransactionBank,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        run_distributed: impl FnMut(
            usize,
            &[u8],
        ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError>,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.run(
            device,
            mounted,
            VulkanComponentBatchStateSemantics::IndependentCandidates(transaction),
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            run_distributed,
        )
    }

    fn run_causal_sequence(
        &self,
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        run_distributed: impl FnMut(
            usize,
            &[u8],
        ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError>,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.run(
            device,
            mounted,
            VulkanComponentBatchStateSemantics::CausalSequence,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            run_distributed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        state_semantics: VulkanComponentBatchStateSemantics<'_>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        mut run_distributed: impl FnMut(
            usize,
            &[u8],
        )
            -> Result<(), VulkanResidentInProcessPlacedRuntimeError>,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let batch_width = input_token_ids.len();
        if batch_width == 0 || batch_width > self.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch tile width {} cannot execute {batch_width} lanes",
                    self.lane_capacity
                )),
            ));
        }
        let lane_controls = component_batch_lane_stream_control_bytes(
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
        )?;
        for (stream_control_buffer, control_bytes) in
            self.stream_control_buffers.iter().zip(&lane_controls)
        {
            stream_control_buffer
                .write_bytes(control_bytes)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }

        let batch_width_u32 = u32::try_from(batch_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "component batch width exceeds u32".to_string(),
            ))
        })?;
        let batch_control = component_batch_control_bytes(
            batch_width_u32,
            start_stream_tick,
            dynamic_state_capacity_activations,
        );
        let mut sequence_index = 0usize;
        let mut local_submission_batch = VulkanResidentQueueSubmissionBatch::new();
        for (unit_index, unit) in self.execution_units.iter().enumerate() {
            match unit {
                VulkanComponentBatchExecutionUnit::LocalComponent {
                    component_id,
                    step_start,
                    step_end,
                } => {
                    let flush_after_segment = self
                        .execution_units
                        .get(unit_index + 1)
                        .is_none_or(|next| {
                            !matches!(
                                next,
                                VulkanComponentBatchExecutionUnit::LocalComponent { .. }
                            )
                        });
                    self.run_segment(
                        device,
                        mounted,
                        state_semantics,
                        batch_width,
                        start_stream_tick,
                        dynamic_state_capacity_activations,
                        &batch_control,
                        component_id,
                        sequence_index,
                        *step_start,
                        *step_end,
                        Some(&local_submission_batch),
                        flush_after_segment,
                    )?;
                    sequence_index += 1;
                    if flush_after_segment {
                        self.submit_and_wait_local_batch(
                            std::mem::replace(
                                &mut local_submission_batch,
                                VulkanResidentQueueSubmissionBatch::new(),
                            ),
                        )?;
                    }
                }
                VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index } => {
                    run_distributed(*dispatch_index, &batch_control)?;
                }
            }
        }
        self.submit_and_wait_local_batch(
            local_submission_batch,
        )?;
        debug_assert_eq!(sequence_index, self.sequences.len());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_segment<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        state_semantics: VulkanComponentBatchStateSemantics<'_>,
        batch_width: usize,
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        batch_control: &[u8; VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY as usize],
        component_id: &str,
        segment_index: usize,
        step_start: usize,
        step_end: usize,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
        signal_completion: bool,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let mut push_constant_storage = Vec::<Vec<u8>>::new();
        let mut active_steps = Vec::<&VulkanComponentBatchDispatchStep>::new();
        for step in &self.steps[step_start..step_end] {
            if step.lane_index.is_some_and(|lane| lane >= batch_width) {
                continue;
            }
            let push_constants = if let Some(lane_index) = step.lane_index {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(lane_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                stream_control_push_constant_bytes(
                    &step.push_constants,
                    VulkanMountedPlacedStreamControl {
                        stream_tick,
                        control_flags: 0,
                        dynamic_state_capacity_activations,
                    },
                )
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "invalid component batch stream control: {error}"
                    )))
                })?
            } else {
                component_batch_push_constant_bytes(step.batch_control_byte_count, batch_control)?
            };
            push_constant_storage.push(push_constants);
            active_steps.push(step);
        }
        if active_steps.is_empty() {
            return Ok(());
        }
        let sequence_steps = active_steps
            .iter()
            .zip(&push_constant_storage)
            .map(|(step, push_constants)| {
                VulkanResidentKernelSequenceStep::new(&step.dispatch, push_constants)
            })
            .collect::<Vec<_>>();
        let mut snapshot_copies = Vec::new();
        if let VulkanComponentBatchStateSemantics::IndependentCandidates(transaction) =
            state_semantics
        {
            for (step_index, step) in active_steps.iter().enumerate() {
                let Some(lane_index) = step.lane_index else {
                    continue;
                };
                if step.snapshot_state_buffer_indices.is_empty() {
                    continue;
                }
                snapshot_copies.extend(
                    transaction
                        .copies_for_state_buffers(
                            &mounted.buffers,
                            step_index,
                            lane_index,
                            &step.snapshot_state_buffer_indices,
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
            }
        }
        if let Some(submission_batch) = submission_batch {
            let execution_cost = RuntimeExecutionCost::new(
                active_steps.iter().fold(0u64, |total, step| {
                    total.saturating_add(step.dispatch.estimated_work_units())
                }),
                active_steps.iter().fold(0u64, |total, step| {
                    total.saturating_add(step.dispatch.estimated_memory_bytes())
                }),
                u64::try_from(active_steps.len()).unwrap_or(u64::MAX),
            );
            let mut execution_region = RuntimeExecutionRegion::new(
                format!("{component_id}:{step_start}..{step_end}"),
                component_id,
                execution_cost,
            );
            execution_region.kernel_families = active_steps
                .iter()
                .map(|step| step.dispatch.execution_family())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            execution_region.commits_state_after = active_steps
                .iter()
                .any(|step| step.commits_state);
            device
                .record_resident_kernel_sequence_with_snapshot_copies(
                    &self.sequences[segment_index],
                    &sequence_steps,
                    &snapshot_copies,
                )
                .and_then(|_| {
                    submission_batch.enqueue_recorded_sequence_with_execution_region(
                        device,
                        &self.sequences[segment_index],
                        &[],
                        &[],
                        signal_completion,
                        Some(execution_region),
                    )
                })
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        } else {
            device
                .run_resident_kernel_sequence_with_snapshot_copies(
                    &self.sequences[segment_index],
                    &sequence_steps,
                    &snapshot_copies,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        }
    }

    fn submit_and_wait_local_batch<'a>(
        &'a self,
        submission_batch: VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if submission_batch.pending_submission_count() == 0 {
            return Ok(());
        }
        let template = {
            let calibrator = self.quantum_calibrator.borrow();
            submission_batch
                .mount_calibrated(&calibrator)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
        };
        let measurements = template
            .submit_calibrated_quanta_and_wait(0)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut calibrator = self.quantum_calibrator.borrow_mut();
        for measurement in measurements {
            calibrator.observe_quantum(
                measurement.cost,
                &measurement.kernel_families,
                measurement.duration_ns,
            );
            record_vulkan_execution_quantum_measurement(&measurement);
        }
        Ok(())
    }

    fn signal_buffer(
        &self,
        key: &VulkanComponentBatchSignalKey,
    ) -> Result<&VulkanComponentBatchSignalBuffer, VulkanResidentInProcessPlacedRuntimeError> {
        self.signal_buffer_indices
            .get(key)
            .and_then(|index| self.signal_buffers.get(*index))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch has no signal buffer {key:?}"
                )))
            })
    }

    fn distributed_signal_buffer(
        &self,
        key: &VulkanComponentBatchSignalKey,
        device_id: &str,
    ) -> Result<&Arc<VulkanResidentBuffer>, VulkanResidentInProcessPlacedRuntimeError> {
        let allocation = self.signal_buffer_indices.get(key).and_then(|index| {
            self.signal_buffers
                .get(*index)
                .and_then(|buffer| buffer.shared_device_buffers.get(device_id))
        });
        allocation.ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "distributed component batch signal {key:?} is not imported on {device_id:?}"
            )))
        })
    }
}
