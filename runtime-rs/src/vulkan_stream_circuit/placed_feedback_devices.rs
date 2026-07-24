pub struct VulkanResidentInProcessPlacedStreamProcessorDevice {
    pub device_id: String,
    pub hosted_component_count: usize,
    pub incoming_edge_count: usize,
    pub outgoing_edge_count: usize,
    pub dispatch_count: usize,
    package_slice: Arc<VulkanResidentModelPackageDeviceSlice>,
    mounted: VulkanMountedPlacedStreamCircuit,
    mounted_bound: VulkanMountedPlacedBoundDispatchPlan,
    resident_execution_plan: VulkanMountedPlacedResidentStreamTickExecutionPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentInProcessPlacedFeedbackLoopEligibility {
    device_slice_count: usize,
    every_slice_has_terminal_segment: bool,
    distributed_dispatches_are_bridged: bool,
    has_dynamic_push_constants: bool,
    window_width: usize,
    sampler_history_capacity: usize,
}

impl VulkanResidentInProcessPlacedFeedbackLoopEligibility {
    fn disabled_reason(self) -> Option<&'static str> {
        if self.device_slice_count == 0 {
            Some("no_device_slices")
        } else if !self.every_slice_has_terminal_segment {
            Some("missing_terminal_segment")
        } else if !self.distributed_dispatches_are_bridged {
            Some("unbridged_distributed_dispatch")
        } else {
            None
        }
    }

    fn window_width(self) -> Option<usize> {
        if self.disabled_reason().is_some() {
            return None;
        }
        let width = self.window_width.min(self.sampler_history_capacity.max(1));
        (width >= 2).then_some(width)
    }
}

struct VulkanResidentInProcessPlacedFeedbackLoop {
    feedback_synchronization: Option<Box<VulkanResidentPlacedFeedbackTimelineSynchronization>>,
    output_synchronization: Box<VulkanResidentPlacedOutputTimelineSynchronization>,
    control: VulkanResidentFeedbackControlPlane,
    window_policy: VulkanResidentFeedbackWindowPolicy,
    replayable: bool,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
}

const VULKAN_RESIDENT_FEEDBACK_TARGET_CONTROL_LATENCY_NS: u64 = 250_000_000;

/// Learns how many already-recorded ticks may be submitted before returning to
/// a host control boundary. The width is an execution/responsiveness decision,
/// never a limit on how many tokens an input event may generate.
#[derive(Debug)]
struct VulkanResidentFeedbackWindowPolicy {
    maximum_tick_count: usize,
    next_tick_count: Cell<usize>,
    estimated_tick_time_ns: Cell<Option<u64>>,
}

impl VulkanResidentFeedbackWindowPolicy {
    fn new(maximum_tick_count: usize) -> Self {
        debug_assert!(maximum_tick_count >= 2);
        Self {
            maximum_tick_count,
            next_tick_count: Cell::new(2),
            estimated_tick_time_ns: Cell::new(None),
        }
    }

    fn next_tick_count(&self) -> usize {
        self.next_tick_count.get()
    }

    fn observe_completed_window(
        &self,
        planned_tick_count: usize,
        executed_tick_count: usize,
        elapsed_time_ns: u64,
        stopped: bool,
    ) {
        if stopped
            || planned_tick_count != executed_tick_count
            || executed_tick_count == 0
            || elapsed_time_ns == 0
        {
            return;
        }
        // Interrupted and predicated windows are deliberately excluded: their
        // elapsed time does not describe the cost of the submitted shape.
        let observed_tick_time_ns =
            elapsed_time_ns.div_ceil(u64::try_from(executed_tick_count).unwrap_or(u64::MAX));
        let estimated_tick_time_ns = self
            .estimated_tick_time_ns
            .get()
            .map(|previous| {
                previous
                    .saturating_mul(3)
                    .saturating_add(observed_tick_time_ns)
                    .div_ceil(4)
            })
            .unwrap_or(observed_tick_time_ns)
            .max(1);
        self.estimated_tick_time_ns
            .set(Some(estimated_tick_time_ns));
        let responsive_tick_count =
            VULKAN_RESIDENT_FEEDBACK_TARGET_CONTROL_LATENCY_NS / estimated_tick_time_ns;
        self.next_tick_count.set(
            usize::try_from(responsive_tick_count)
                .unwrap_or(usize::MAX)
                .clamp(2, self.maximum_tick_count),
        );
    }
}

struct VulkanResidentPlacedFeedbackMount<'a> {
    input_transducer: &'a VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: &'a VulkanResidentOutputTransducerRunner,
    sampler: &'a VulkanResidentSamplerRunner,
    control: VulkanResidentFeedbackControlPlane,
}

struct VulkanResidentPlacedFeedbackTimelineSynchronization {
    output_signal: VulkanTimelineSemaphore,
    input_wait: VulkanTimelineSemaphore,
    next_value: Cell<u64>,
    pending_value: Cell<Option<u64>>,
}

#[derive(Clone, Copy)]
struct VulkanPlacedFeedbackTimelineTurn<'a> {
    input_device_id: &'a str,
    output_device_id: &'a str,
    input_wait: Option<VulkanTimelineSemaphorePoint<'a>>,
    output_signal: VulkanTimelineSemaphorePoint<'a>,
}

struct VulkanResidentPlacedOutputTimelineSynchronization {
    signal: VulkanTimelineSemaphore,
    next_value: Cell<u64>,
}

#[derive(Clone, Copy)]
struct VulkanPlacedOutputTimelineTurn<'a> {
    output_device_id: &'a str,
    signal: VulkanTimelineSemaphorePoint<'a>,
    value: u64,
}

struct VulkanResidentPlacedFeedbackSubmissionReplay {
    template: VulkanResidentQueueSubmissionTemplate,
    tick_count: usize,
    next_timeline_value_offset: u64,
}

impl VulkanResidentPlacedFeedbackTimelineSynchronization {
    fn new(
        input_device: &VulkanComputeDevice,
        output_device: &VulkanComputeDevice,
    ) -> Result<Option<Self>, VulkanError> {
        if input_device.shares_logical_device_with(output_device) {
            return Ok(None);
        }
        if !input_device.supports_opaque_fd_timeline_semaphores()
            || !output_device.supports_opaque_fd_timeline_semaphores()
        {
            return Err(VulkanError(
                "cross-device resident feedback requires persistent opaque-file timeline semaphores"
                    .to_string(),
            ));
        }
        let output_signal = output_device.create_opaque_fd_exportable_timeline_semaphore(0)?;
        let input_wait = input_device.create_timeline_semaphore(0)?;
        input_device.import_timeline_semaphore_opaque_fd(
            &input_wait,
            output_device.export_timeline_semaphore_opaque_fd(&output_signal)?,
        )?;
        Ok(Some(Self {
            output_signal,
            input_wait,
            next_value: Cell::new(1),
            pending_value: Cell::new(None),
        }))
    }

    fn prepare_turn<'a>(
        &'a self,
        input_device_id: &'a str,
        output_device_id: &'a str,
    ) -> Result<VulkanPlacedFeedbackTimelineTurn<'a>, VulkanError> {
        let value = self.next_value.get();
        self.next_value.set(value.checked_add(1).ok_or_else(|| {
            VulkanError("resident feedback timeline semaphore exhausted its values".to_string())
        })?);
        let input_wait = self
            .pending_value
            .replace(Some(value))
            .map(|pending| VulkanTimelineSemaphorePoint::new(&self.input_wait, pending));
        Ok(VulkanPlacedFeedbackTimelineTurn {
            input_device_id,
            output_device_id,
            input_wait,
            output_signal: VulkanTimelineSemaphorePoint::new(&self.output_signal, value),
        })
    }

    fn advance_replayed_turns(&self, count: usize) -> Result<(), VulkanError> {
        let count = u64::try_from(count)
            .map_err(|_| VulkanError("resident feedback replay width exceeds u64".to_string()))?;
        if count == 0 {
            return Err(VulkanError(
                "resident feedback replay width must not be zero".to_string(),
            ));
        }
        let first_value = self.next_value.get();
        let expected_pending = first_value.checked_sub(1).ok_or_else(|| {
            VulkanError("resident feedback replay has no preceding timeline value".to_string())
        })?;
        if self.pending_value.get() != Some(expected_pending) {
            return Err(VulkanError(format!(
                "resident feedback replay expected pending timeline value {expected_pending}, found {:?}",
                self.pending_value.get()
            )));
        }
        let next_value = first_value.checked_add(count).ok_or_else(|| {
            VulkanError("resident feedback replay exhausted timeline values".to_string())
        })?;
        self.next_value.set(next_value);
        self.pending_value.set(Some(next_value - 1));
        Ok(())
    }
}

impl VulkanResidentPlacedOutputTimelineSynchronization {
    fn new(output_device: &VulkanComputeDevice) -> Result<Self, VulkanError> {
        Ok(Self {
            signal: output_device.create_timeline_semaphore(0)?,
            next_value: Cell::new(1),
        })
    }

    fn prepare_turn<'a>(
        &'a self,
        output_device_id: &'a str,
    ) -> Result<VulkanPlacedOutputTimelineTurn<'a>, VulkanError> {
        let value = self.next_value.get();
        self.next_value.set(value.checked_add(1).ok_or_else(|| {
            VulkanError("resident output timeline semaphore exhausted its values".to_string())
        })?);
        Ok(VulkanPlacedOutputTimelineTurn {
            output_device_id,
            signal: VulkanTimelineSemaphorePoint::new(&self.signal, value),
            value,
        })
    }

    fn wait_for_turn(
        &self,
        output_device: &VulkanComputeDevice,
        value: u64,
    ) -> Result<(), VulkanError> {
        output_device.wait_timeline_semaphore_value(&self.signal, value)
    }

    fn reserve_replayed_turns(&self, count: usize) -> Result<Vec<u64>, VulkanError> {
        let count = u64::try_from(count)
            .map_err(|_| VulkanError("resident output replay width exceeds u64".to_string()))?;
        if count == 0 {
            return Err(VulkanError(
                "resident output replay width must not be zero".to_string(),
            ));
        }
        let first_value = self.next_value.get();
        let next_value = first_value.checked_add(count).ok_or_else(|| {
            VulkanError("resident output replay exhausted timeline values".to_string())
        })?;
        self.next_value.set(next_value);
        Ok((first_value..next_value).collect())
    }
}

impl VulkanResidentPlacedFeedbackSubmissionReplay {
    fn new(
        template: VulkanResidentQueueSubmissionTemplate,
        tick_count: usize,
    ) -> Result<Self, VulkanError> {
        let next_timeline_value_offset = u64::try_from(tick_count)
            .map_err(|_| VulkanError("resident feedback replay width exceeds u64".to_string()))?;
        Ok(Self {
            template,
            tick_count,
            next_timeline_value_offset,
        })
    }

    fn validate_tick_count(&self, tick_count: usize) -> Result<(), VulkanError> {
        if tick_count != self.tick_count {
            return Err(VulkanError(format!(
                "resident feedback replay was mounted for {} ticks, received {tick_count}",
                self.tick_count
            )));
        }
        Ok(())
    }

    fn submit_next(&mut self, tick_count: usize) -> Result<usize, VulkanError> {
        self.validate_tick_count(tick_count)?;
        let offset = self.next_timeline_value_offset;
        let next_offset = offset
            .checked_add(u64::try_from(tick_count).map_err(|_| {
                VulkanError("resident feedback replay width exceeds u64".to_string())
            })?)
            .ok_or_else(|| {
                VulkanError("resident feedback replay offset exhausted u64".to_string())
            })?;
        let submitted = self.template.submit_with_timeline_value_offset(offset)?;
        self.next_timeline_value_offset = next_offset;
        Ok(submitted)
    }
}

fn apply_placed_clone_state_policies(
    devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    initialized: &BTreeSet<(String, String)>,
) -> Result<usize, VulkanError> {
    let mut state_index = BTreeMap::<(String, String), (usize, usize)>::new();
    let mut states = Vec::new();
    for (device_index, device) in devices.iter().enumerate() {
        for (state_index_on_device, state) in
            device.mounted.buffers.state_buffers.iter().enumerate()
        {
            let key = (state.component_id.clone(), state.state_id.clone());
            if state_index
                .insert(key.clone(), (device_index, state_index_on_device))
                .is_some()
            {
                return Err(VulkanError(format!(
                    "duplicate placed state buffer {}.{}",
                    key.0, key.1
                )));
            }
            states.push((key, state.clone_from.clone()));
        }
    }
    let copies = ordered_clone_state_copies(states, initialized)?;
    let mut total_copied = 0usize;
    for (target_id, source_id) in copies {
        let (target_device_index, target_state_index) = state_index
            .get(&target_id)
            .copied()
            .expect("clone target was indexed from resident states");
        let (source_device_index, source_state_index) = state_index
            .get(&source_id)
            .copied()
            .expect("planned clone source must exist");
        let target =
            &devices[target_device_index].mounted.buffers.state_buffers[target_state_index];
        let source =
            &devices[source_device_index].mounted.buffers.state_buffers[source_state_index];
        validate_state_buffer_copy(target, source)?;
        let bytes = source.buffer.read_bytes(source.byte_capacity)?;
        target.buffer.write_bytes(&bytes)?;
        total_copied = total_copied
            .checked_add(bytes.len())
            .ok_or_else(|| VulkanError("placed clone state byte count overflowed".to_string()))?;
    }
    Ok(total_copied)
}

fn inherit_matching_placed_stream_state(
    target_devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    source_devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
) -> Result<(usize, BTreeSet<(String, String)>), VulkanError> {
    let source_by_id = source_devices
        .iter()
        .flat_map(|device| device.mounted.buffers.state_buffers.iter())
        .map(|state| ((state.component_id.as_str(), state.state_id.as_str()), state))
        .collect::<BTreeMap<_, _>>();
    let mut copied = BTreeSet::new();
    let mut total_copied = 0usize;
    for target in target_devices
        .iter()
        .flat_map(|device| device.mounted.buffers.state_buffers.iter())
    {
        let key = (target.component_id.as_str(), target.state_id.as_str());
        let Some(source) = source_by_id.get(&key) else {
            continue;
        };
        validate_state_buffer_copy(target, source)?;
        let bytes = source.buffer.read_bytes(source.byte_capacity)?;
        target.buffer.write_bytes(&bytes)?;
        total_copied = total_copied.checked_add(bytes.len()).ok_or_else(|| {
            VulkanError("inherited placed state byte count overflowed".to_string())
        })?;
        copied.insert((target.component_id.clone(), target.state_id.clone()));
    }
    Ok((total_copied, copied))
}

impl VulkanResidentInProcessPlacedFeedbackLoop {
    fn new_if_supported<'a, F, E>(
        model: &VulkanResidentInProcessPlacedModelPackage,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        activation_schedule: &VulkanMountedPlacedResidentInProcessSchedule,
        mount: VulkanResidentPlacedFeedbackMount<'_>,
        device_for: &F,
    ) -> Result<Option<Self>, VulkanError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let VulkanResidentPlacedFeedbackMount {
            input_transducer,
            output_transducer,
            sampler,
            control,
        } = mount;
        let has_dynamic_push_constants = input_transducer
            .resident_dispatch
            .push_constant_byte_count()
            != 0
            || output_transducer
                .embedding_norm_dispatch
                .push_constant_byte_count()
                != 0
            || output_transducer
                .tied_projection_dispatch
                .push_constant_byte_count()
                != 0
            || sampler
                .resident_dispatches()
                .iter()
                .any(|dispatch| dispatch.push_constant_byte_count() != 0)
            || device_slices
                .iter()
                .flat_map(|slice| &slice.resident_execution_plan.dispatch_segments)
                .flat_map(|segment| &segment.dispatches)
                .any(|dispatch| {
                    dispatch.resident_dispatch.push_constant_byte_count() != 0
                        && dispatch.push_constants.as_slice()
                            != [VulkanKernelScalarBinding {
                                name: "expert_start".to_string(),
                                scalar_type: "u32".to_string(),
                                source: VulkanKernelScalarSource::PushConstant,
                            }]
                });
        let window_width = VULKAN_BACKEND_LOOP_MAX_WINDOW
            .min(sampler.history_capacity_activations.max(1));
        let eligibility = VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            device_slice_count: device_slices.len(),
            every_slice_has_terminal_segment: device_slices
                .iter()
                .all(|slice| !slice.resident_execution_plan.dispatch_segments.is_empty()),
            distributed_dispatches_are_bridged: device_slices.iter().all(|slice| {
                slice
                    .resident_execution_plan
                    .distributed_dispatch_dependencies
                    .values()
                    .all(|dependency| {
                        dependency.has_owner_producer && dependency.has_owner_continuation
                    })
            }),
            has_dynamic_push_constants,
            window_width,
            sampler_history_capacity: sampler.history_capacity_activations,
        };
        let Some(window_width) = eligibility.window_width() else {
            return Ok(None);
        };
        let input_device = device_for(&model.input_device_id).map_err(|error| {
            VulkanError(format!("feedback input device resolution failed: {error}"))
        })?;
        let output_device = device_for(&model.output_device_id).map_err(|error| {
            VulkanError(format!("feedback output device resolution failed: {error}"))
        })?;
        let feedback_synchronization =
            VulkanResidentPlacedFeedbackTimelineSynchronization::new(input_device, output_device)?
                .map(Box::new);
        let output_synchronization = Box::new(
            VulkanResidentPlacedOutputTimelineSynchronization::new(output_device)?,
        );
        let completed_stage_count_per_tick =
            device_slices.iter().try_fold(0usize, |total, slice| {
                total
                    .checked_add(slice.resident_execution_plan.tick_plan.stage_count)
                    .ok_or_else(|| {
                        VulkanError("placed feedback stage count overflowed".to_string())
                    })
            })?;
        Ok(Some(Self {
            feedback_synchronization,
            output_synchronization,
            control,
            window_policy: VulkanResidentFeedbackWindowPolicy::new(window_width),
            replayable: !has_dynamic_push_constants,
            scheduler_turn_count_per_tick: activation_schedule.turns.len(),
            completed_stage_count_per_tick,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanResidentPlacedTokenTickTail {
    None,
    Hidden,
    Logits,
    Sample,
}

impl VulkanResidentPlacedTokenTickTail {
    fn sequence_variant(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Hidden => 1,
            Self::Logits => 2,
            Self::Sample => 3,
        }
    }

    fn produces_logits(self) -> bool {
        matches!(self, Self::Logits | Self::Sample)
    }
}

fn placed_token_input(
    token_id: u32,
    input_device_id: &str,
    output_device_id: &str,
    input_is_feedback: bool,
) -> VulkanResidentPlacedTokenInput {
    if !input_is_feedback {
        VulkanResidentPlacedTokenInput::HostSupplied(token_id)
    } else if input_device_id == output_device_id {
        VulkanResidentPlacedTokenInput::ResidentFeedback(token_id)
    } else {
        VulkanResidentPlacedTokenInput::EdgeFeedback(token_id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanResidentPlacedTokenInput {
    HostSupplied(u32),
    ResidentFeedback(u32),
    EdgeFeedback(u32),
}

impl VulkanResidentPlacedTokenInput {
    fn token_id(self) -> u32 {
        match self {
            Self::HostSupplied(token_id)
            | Self::ResidentFeedback(token_id)
            | Self::EdgeFeedback(token_id) => token_id,
        }
    }
}

fn pair_placed_edge_endpoints(
    plans: &[VulkanPlacedEdgeIoPlan],
) -> Result<Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>, VulkanError> {
    let mut incoming_by_key = BTreeMap::new();
    for plan in plans {
        for endpoint in plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
        {
            let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(endpoint);
            if incoming_by_key
                .insert(key.clone(), endpoint.clone())
                .is_some()
            {
                return Err(VulkanError(format!(
                    "placed execution_graph repeats incoming edge endpoint {key:?}"
                )));
            }
        }
    }

    let mut pairs = Vec::with_capacity(incoming_by_key.len());
    let mut outgoing_keys = BTreeSet::new();
    for plan in plans {
        for outgoing in plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
        {
            let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(outgoing);
            if !outgoing_keys.insert(key.clone()) {
                return Err(VulkanError(format!(
                    "placed execution_graph repeats outgoing edge endpoint {key:?}"
                )));
            }
            let incoming = incoming_by_key.remove(&key).ok_or_else(|| {
                VulkanError(format!(
                    "placed execution_graph has no incoming endpoint for edge {key:?}"
                ))
            })?;
            let outgoing_byte_capacity = outgoing.byte_capacity.ok_or_else(|| {
                VulkanError(format!("outgoing edge {key:?} has unknown byte capacity"))
            })?;
            let incoming_byte_capacity = incoming.byte_capacity.ok_or_else(|| {
                VulkanError(format!("incoming edge {key:?} has unknown byte capacity"))
            })?;
            if outgoing_byte_capacity != incoming_byte_capacity {
                return Err(VulkanError(format!(
                    "placed edge {key:?} has outgoing capacity {outgoing_byte_capacity} and incoming capacity {incoming_byte_capacity}"
                )));
            }
            pairs.push((outgoing.clone(), incoming));
        }
    }
    if let Some(key) = incoming_by_key.keys().next() {
        return Err(VulkanError(format!(
            "placed execution_graph has no outgoing endpoint for edge {key:?}"
        )));
    }
    Ok(pairs)
}

struct VulkanPlacedDeviceLinks {
    endpoint_overrides: BTreeMap<String, Vec<VulkanPlacedEdgeEndpointBufferOverride>>,
    synchronizations: VulkanPlacedEdgeTimelineSynchronizations,
    stream_control_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Default)]
struct VulkanPlacedEdgeTimelineSynchronizations {
    edges: BTreeMap<VulkanPlacedEdgePacketKey, VulkanPlacedEdgeTimelineSynchronization>,
}

struct VulkanPlacedEdgeTimelineSynchronization {
    source_signal: VulkanTimelineSemaphore,
    destination_wait: VulkanTimelineSemaphore,
    next_value: Cell<u64>,
    pending_value: Cell<Option<u64>>,
}

impl VulkanPlacedEdgeTimelineSynchronizations {
    fn advance_replayed_dependencies(&self, count: usize) -> Result<(), VulkanError> {
        let count = u64::try_from(count)
            .map_err(|_| VulkanError("placed edge replay width exceeds u64".to_string()))?;
        for (key, synchronization) in &self.edges {
            if synchronization.pending_value.get().is_some() {
                return Err(VulkanError(format!(
                    "cross-device edge {key:?} cannot replay with an unconsumed timeline dependency"
                )));
            }
            synchronization
                .next_value
                .get()
                .checked_add(count)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "cross-device edge {key:?} exhausts its timeline values during replay"
                    ))
                })?;
        }
        for synchronization in self.edges.values() {
            synchronization.next_value.set(
                synchronization
                    .next_value
                    .get()
                    .checked_add(count)
                    .expect("placed edge replay advance was validated"),
            );
        }
        Ok(())
    }

    fn prepare_source_signal<'a>(
        &'a self,
        endpoint: &VulkanPlacedEdgeEndpoint,
    ) -> Result<Option<VulkanTimelineSemaphorePoint<'a>>, VulkanError> {
        let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(endpoint);
        let Some(synchronization) = self.edges.get(&key) else {
            return Ok(None);
        };
        if synchronization.pending_value.get().is_some() {
            return Err(VulkanError(format!(
                "cross-device edge {key:?} already has an unconsumed timeline dependency"
            )));
        }
        let value = synchronization.next_value.get();
        let next = value.checked_add(1).ok_or_else(|| {
            VulkanError(format!(
                "cross-device edge {key:?} exhausted its timeline semaphore values"
            ))
        })?;
        synchronization.next_value.set(next);
        synchronization.pending_value.set(Some(value));
        Ok(Some(VulkanTimelineSemaphorePoint::new(
            &synchronization.source_signal,
            value,
        )))
    }

    fn take_destination_wait<'a>(
        &'a self,
        endpoint: &VulkanPlacedEdgeEndpoint,
    ) -> Result<Option<VulkanTimelineSemaphorePoint<'a>>, VulkanError> {
        let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(endpoint);
        let Some(synchronization) = self.edges.get(&key) else {
            return Ok(None);
        };
        let value = synchronization.pending_value.take().ok_or_else(|| {
            VulkanError(format!(
                "cross-device edge {key:?} has no queued timeline dependency"
            ))
        })?;
        Ok(Some(VulkanTimelineSemaphorePoint::new(
            &synchronization.destination_wait,
            value,
        )))
    }

    fn has_pending_dependencies(&self) -> bool {
        self.edges
            .values()
            .any(|synchronization| synchronization.pending_value.get().is_some())
    }
}

fn create_placed_device_links<'a, F>(
    device_slices: &[Arc<VulkanResidentModelPackageDeviceSlice>],
    device_for: &F,
) -> Result<VulkanPlacedDeviceLinks, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    let plans = device_slices
        .iter()
        .map(|slice| {
            VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
                &slice.placed_plan.placed_resident_plan,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "failed to plan shared edge endpoints for {:?}: {error}",
                    slice.device_id
                )))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let edge_pairs = pair_placed_edge_endpoints(&plans)
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

    let mut endpoint_overrides =
        BTreeMap::<String, Vec<VulkanPlacedEdgeEndpointBufferOverride>>::new();
    let mut synchronizations = BTreeMap::new();
    for (outgoing, incoming) in edge_pairs {
        let outgoing_byte_capacity = outgoing
            .byte_capacity
            .expect("paired outgoing edge capacity was validated");
        let source_device = device_for(&outgoing.local_device_id)?;
        let destination_device = device_for(&incoming.local_device_id)?;
        let devices_share_queue = source_device.shares_logical_device_with(destination_device);
        let (outgoing_buffer, incoming_buffer) = if devices_share_queue {
            let buffer = Arc::new(
                source_device
                    .create_resident_buffer(outgoing_byte_capacity)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            );
            (buffer.clone(), buffer)
        } else {
            let allocation = source_device
                .create_shared_host_allocation(&[destination_device], outgoing_byte_capacity)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let outgoing_buffer = Arc::new(
                source_device
                    .import_shared_host_buffer(Arc::clone(&allocation))
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            );
            let incoming_buffer = Arc::new(
                destination_device
                    .import_shared_host_buffer(allocation)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            );
            (outgoing_buffer, incoming_buffer)
        };
        if !devices_share_queue {
            if !source_device.supports_opaque_fd_timeline_semaphores()
                || !destination_device.supports_opaque_fd_timeline_semaphores()
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "cross-device edge {:?} requires persistent opaque-file timeline semaphores",
                        VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing)
                    )),
                ));
            }
            let source_signal = source_device
                .create_opaque_fd_exportable_timeline_semaphore(0)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let destination_wait = destination_device
                .create_timeline_semaphore(0)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            destination_device
                .import_timeline_semaphore_opaque_fd(
                    &destination_wait,
                    source_device
                        .export_timeline_semaphore_opaque_fd(&source_signal)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing);
            if synchronizations
                .insert(
                    key.clone(),
                    VulkanPlacedEdgeTimelineSynchronization {
                        source_signal,
                        destination_wait,
                        next_value: Cell::new(1),
                        pending_value: Cell::new(None),
                    },
                )
                .is_some()
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "cross-device edge synchronization repeats {key:?}"
                    )),
                ));
            }
        }
        endpoint_overrides
            .entry(outgoing.local_device_id.clone())
            .or_default()
            .push(VulkanPlacedEdgeEndpointBufferOverride {
                direction: VulkanPlacedEdgeDirection::Outgoing,
                edge_index: outgoing.edge_index,
                buffer: outgoing_buffer,
            });
        endpoint_overrides
            .entry(incoming.local_device_id.clone())
            .or_default()
            .push(VulkanPlacedEdgeEndpointBufferOverride {
                direction: VulkanPlacedEdgeDirection::Incoming,
                edge_index: incoming.edge_index,
                buffer: incoming_buffer,
            });
    }
    let mut unique_devices = Vec::<(&VulkanComputeDevice, Vec<String>)>::new();
    for slice in device_slices {
        let device = device_for(&slice.device_id)?;
        if let Some((_, device_ids)) = unique_devices
            .iter_mut()
            .find(|(candidate, _)| candidate.shares_logical_device_with(device))
        {
            device_ids.push(slice.device_id.clone());
        } else {
            unique_devices.push((device, vec![slice.device_id.clone()]));
        }
    }
    let mut stream_control_buffers = BTreeMap::new();
    if let Some((owner_device, _)) = unique_devices.first() {
        let buffers = if unique_devices.len() == 1 {
            let mut buffer = owner_device
                .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            buffer
                .persistently_map()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            vec![Arc::new(buffer)]
        } else {
            let peers = unique_devices
                .iter()
                .skip(1)
                .map(|(device, _)| *device)
                .collect::<Vec<_>>();
            let allocation = owner_device
                .create_shared_host_allocation(&peers, VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            unique_devices
                .iter()
                .map(|(device, _)| {
                    device
                        .import_shared_host_buffer(Arc::clone(&allocation))
                        .map(Arc::new)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        buffers[0]
            .write_bytes(&[0; VULKAN_STREAM_CONTROL_BYTE_CAPACITY])
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        for ((_, device_ids), buffer) in unique_devices.iter().zip(buffers) {
            for device_id in device_ids {
                stream_control_buffers.insert(device_id.clone(), buffer.clone());
            }
        }
    }
    Ok(VulkanPlacedDeviceLinks {
        endpoint_overrides,
        synchronizations: VulkanPlacedEdgeTimelineSynchronizations {
            edges: synchronizations,
        },
        stream_control_buffers,
    })
}
