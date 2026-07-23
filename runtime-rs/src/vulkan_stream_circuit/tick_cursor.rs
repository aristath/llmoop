#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentStreamTickCursor {
    pub tick_plan: Arc<VulkanMountedPlacedStreamTickPlan>,
    pub stream_tick: u64,
    pub next_stage_index: usize,
    pub completed_stage_count: usize,
    pub pedal_runs: Vec<VulkanMountedPlacedResidentPedalRun>,
    capture_execution_trace: bool,
    last_blocked: Option<(usize, VulkanMountedPlacedStreamTickBlockReason)>,
}

impl VulkanMountedPlacedResidentStreamTickCursor {
    pub fn new(tick_plan: VulkanMountedPlacedStreamTickPlan, stream_tick: u64) -> Self {
        Self::new_shared(Arc::new(tick_plan), stream_tick, true)
    }

    fn new_shared(
        tick_plan: Arc<VulkanMountedPlacedStreamTickPlan>,
        stream_tick: u64,
        capture_execution_trace: bool,
    ) -> Self {
        Self {
            tick_plan,
            stream_tick,
            next_stage_index: 0,
            completed_stage_count: 0,
            pedal_runs: Vec::new(),
            capture_execution_trace,
            last_blocked: None,
        }
    }

    pub fn is_completed(&self) -> bool {
        self.next_stage_index == self.tick_plan.stages.len()
    }

    pub fn advance_with_resident_pedals_and_in_process_transport(
        &mut self,
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        transport: &mut VulkanInProcessPlacedCableTransport,
    ) -> Result<
        VulkanMountedPlacedResidentStreamTickCursorAdvance,
        VulkanMountedPlacedResidentStreamTickError,
    > {
        let execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan::from_tick_plan(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            self.tick_plan.as_ref().clone(),
        )
        .map_err(VulkanMountedPlacedResidentStreamTickError::Dispatch)?;
        let dispatch_extensions =
            VulkanMountedPlacedResidentStreamTickDispatchExtensions::default();
        self.advance_with_resident_execution_plan_and_in_process_transport(
            device,
            mounted,
            &execution_plan,
            &dispatch_extensions,
            transport,
        )
    }

    pub fn advance_with_resident_execution_plan_and_in_process_transport(
        &mut self,
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        execution_plan: &VulkanMountedPlacedResidentStreamTickExecutionPlan,
        dispatch_extensions: &VulkanMountedPlacedResidentStreamTickDispatchExtensions<'_>,
        transport: &mut VulkanInProcessPlacedCableTransport,
    ) -> Result<
        VulkanMountedPlacedResidentStreamTickCursorAdvance,
        VulkanMountedPlacedResidentStreamTickError,
    > {
        if self.tick_plan.device_id != mounted.device_id() {
            return Err(VulkanMountedPlacedResidentStreamTickError::DeviceMismatch {
                plan_device_id: self.tick_plan.device_id.clone(),
                mounted_device_id: mounted.device_id().to_string(),
            });
        }
        if self.tick_plan.device_id != execution_plan.tick_plan.device_id {
            return Err(
                VulkanMountedPlacedResidentStreamTickError::BoundPlanDeviceMismatch {
                    plan_device_id: self.tick_plan.device_id.clone(),
                    bound_plan_device_id: execution_plan.tick_plan.device_id.clone(),
                },
            );
        }

        let dynamic_state_capacity_activations =
            u32::try_from(mounted.buffers.dynamic_state_capacity_activations).map_err(|_| {
                VulkanMountedPlacedResidentStreamTickError::DynamicStateCapacityOverflow {
                    capacity: mounted.buffers.dynamic_state_capacity_activations,
                }
            })?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick: self.stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations,
        };

        self.last_blocked = None;
        let completed_before = self.completed_stage_count;

        while self.next_stage_index < self.tick_plan.stages.len() {
            let stage = &self.tick_plan.stages[self.next_stage_index];
            match stage {
                VulkanMountedPlacedStreamTickStage::ReceiveCable { cable_index, .. } => {
                    match transport.receive_incoming_cable(mounted, *cable_index) {
                        Ok(_) => self.complete_current_stage(),
                        Err(VulkanPlacedCableTransportError::MissingPacket { .. }) => {
                            let reason =
                                VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable;
                            self.last_blocked = Some((stage.stage_index(), reason));
                            return Ok(self.advance_report(completed_before));
                        }
                        Err(error) => {
                            return Err(VulkanMountedPlacedResidentStreamTickError::Transport(
                                error,
                            ));
                        }
                    }
                }
                VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {
                    if let Some(dispatch) =
                        execution_plan.distributed_dispatch_at_stage(self.next_stage_index)
                    {
                        self.last_blocked = Some((
                            stage.stage_index(),
                            VulkanMountedPlacedStreamTickBlockReason::DistributedDispatchPending {
                                dispatch_index: dispatch.dispatch_index,
                            },
                        ));
                        return Ok(self.advance_report(completed_before));
                    }
                    let segment = execution_plan
                        .segment_starting_at(self.next_stage_index)
                        .ok_or_else(|| {
                            VulkanMountedPlacedResidentStreamTickError::Dispatch(
                                VulkanMountedPlacedResidentKernelDispatchError::MissingDispatchSegment {
                                    device_id: mounted.device_id().to_string(),
                                    stage_index: self.next_stage_index,
                                },
                            )
                        })?;
                    self.pedal_runs.extend(
                        segment
                            .run_with_stream_control(
                                device,
                                control,
                                if execution_plan.first_dispatch_segment_stage_index()
                                    == Some(segment.start_stage_index)
                                {
                                    &dispatch_extensions.prefix_dispatches
                                } else {
                                    &[]
                                },
                                if execution_plan.last_dispatch_segment_stage_index()
                                    == Some(segment.start_stage_index)
                                {
                                    &dispatch_extensions.suffix_dispatches
                                } else {
                                    &[]
                                },
                                dispatch_extensions.sequence_variant,
                                self.capture_execution_trace,
                            )
                            .map_err(VulkanMountedPlacedResidentStreamTickError::Dispatch)?,
                    );
                    while self.next_stage_index < segment.end_stage_index {
                        self.complete_current_stage();
                    }
                }
                VulkanMountedPlacedStreamTickStage::PublishCable { cable_index, .. } => {
                    transport
                        .publish_outgoing_cable(mounted, *cable_index)
                        .map_err(VulkanMountedPlacedResidentStreamTickError::Transport)?;
                    self.complete_current_stage();
                }
            }
        }

        Ok(self.advance_report(completed_before))
    }

    fn pending_distributed_dispatch<'a>(
        &self,
        execution_plan: &'a VulkanMountedPlacedResidentStreamTickExecutionPlan,
    ) -> Option<&'a VulkanMountedPlacedStreamTickDispatch> {
        execution_plan.distributed_dispatch_at_stage(self.next_stage_index)
    }

    fn complete_pending_distributed_dispatch(
        &mut self,
        execution_plan: &VulkanMountedPlacedResidentStreamTickExecutionPlan,
        dispatch_index: usize,
    ) -> Result<usize, VulkanMountedPlacedResidentStreamTickError> {
        let group = execution_plan
            .distributed_dispatch_group_at_stage(self.next_stage_index)
            .ok_or_else(|| {
                VulkanMountedPlacedResidentStreamTickError::Dispatch(
                    VulkanMountedPlacedResidentKernelDispatchError::MissingDispatchSegment {
                        device_id: execution_plan.tick_plan.device_id.clone(),
                        stage_index: self.next_stage_index,
                    },
                )
            })?;
        if group.leader().dispatch_index != dispatch_index {
            return Err(VulkanMountedPlacedResidentStreamTickError::Dispatch(
                VulkanMountedPlacedResidentKernelDispatchError::DistributedDispatchMismatch {
                    device_id: execution_plan.tick_plan.device_id.clone(),
                    stage_index: self.next_stage_index,
                    expected_dispatch_index: group.leader().dispatch_index,
                    completed_dispatch_index: dispatch_index,
                },
            ));
        }
        self.last_blocked = None;
        for dispatch in &group.dispatches {
            let current = self
                .tick_plan
                .stages
                .get(self.next_stage_index)
                .and_then(|stage| match stage {
                    VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } => Some(dispatch),
                    _ => None,
                })
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentStreamTickError::Dispatch(
                        VulkanMountedPlacedResidentKernelDispatchError::MissingDispatchSegment {
                            device_id: execution_plan.tick_plan.device_id.clone(),
                            stage_index: self.next_stage_index,
                        },
                    )
                })?;
            if current.dispatch_index != dispatch.dispatch_index {
                return Err(VulkanMountedPlacedResidentStreamTickError::Dispatch(
                    VulkanMountedPlacedResidentKernelDispatchError::DistributedDispatchMismatch {
                        device_id: execution_plan.tick_plan.device_id.clone(),
                        stage_index: self.next_stage_index,
                        expected_dispatch_index: dispatch.dispatch_index,
                        completed_dispatch_index: current.dispatch_index,
                    },
                ));
            }
            self.complete_current_stage();
        }
        Ok(group.dispatches.len())
    }

    fn complete_current_stage(&mut self) {
        self.completed_stage_count += 1;
        self.next_stage_index += 1;
    }

    fn advance_report(
        &self,
        completed_before: usize,
    ) -> VulkanMountedPlacedResidentStreamTickCursorAdvance {
        VulkanMountedPlacedResidentStreamTickCursorAdvance {
            completed_stage_delta: self.completed_stage_count - completed_before,
            completed: self.is_completed(),
        }
    }

    pub fn snapshot(&self) -> VulkanMountedPlacedResidentStreamTickRun {
        let stages = if self.capture_execution_trace {
            let mut stages = Vec::with_capacity(self.tick_plan.stages.len());
            for (index, stage) in self.tick_plan.stages.iter().enumerate() {
                let status = if index < self.next_stage_index {
                    VulkanMountedPlacedStreamTickStageStatus::Completed
                } else if self
                    .last_blocked
                    .as_ref()
                    .is_some_and(|(stage_index, _)| *stage_index == index)
                {
                    VulkanMountedPlacedStreamTickStageStatus::Blocked {
                        reason: self.last_blocked.as_ref().unwrap().1.clone(),
                    }
                } else {
                    VulkanMountedPlacedStreamTickStageStatus::Pending
                };
                stages.push(VulkanMountedPlacedStreamTickStageRun {
                    stage_index: stage.stage_index(),
                    stage: stage.clone(),
                    status,
                });
            }
            stages
        } else {
            Vec::new()
        };
        let blocked_stage_count = usize::from(self.last_blocked.is_some());
        let pending_stage_count = self.tick_plan.stage_count.saturating_sub(
            self.completed_stage_count
                .saturating_add(blocked_stage_count),
        );
        let status = self
            .last_blocked
            .clone()
            .map(
                |(stage_index, reason)| VulkanMountedPlacedStreamTickRunStatus::Blocked {
                    stage_index,
                    reason,
                },
            )
            .unwrap_or_else(|| {
                if self.is_completed() {
                    VulkanMountedPlacedStreamTickRunStatus::Completed
                } else {
                    VulkanMountedPlacedStreamTickRunStatus::Blocked {
                        stage_index: self.next_stage_index,
                        reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
                    }
                }
            });
        let attempted_stage_count =
            self.completed_stage_count + usize::from(self.last_blocked.is_some());

        VulkanMountedPlacedResidentStreamTickRun {
            tick_run: VulkanMountedPlacedStreamTickRun {
                backend_id: self.tick_plan.backend_id.clone(),
                device_id: self.tick_plan.device_id.clone(),
                stream_tick: self.stream_tick,
                stages,
                planned_stage_count: self.tick_plan.stage_count,
                attempted_stage_count,
                completed_stage_count: self.completed_stage_count,
                pending_stage_count,
                status,
                can_execute: true,
            },
            pedalboard_run: if !self.capture_execution_trace || self.pedal_runs.is_empty() {
                None
            } else {
                Some(VulkanMountedPlacedResidentPedalboardRun {
                    device_id: self.tick_plan.device_id.clone(),
                    pedal_runs: self.pedal_runs.clone(),
                })
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentStreamTickCursorAdvance {
    pub completed_stage_delta: usize,
    pub completed: bool,
}

#[derive(Default)]
pub struct VulkanMountedPlacedResidentStreamTickDispatchExtensions<'a> {
    pub prefix_dispatches: SmallVec<[&'a VulkanResidentKernelDispatch; 1]>,
    pub suffix_dispatches: SmallVec<[&'a VulkanResidentKernelDispatch; 3]>,
    sequence_variant: u8,
}

pub struct VulkanMountedPlacedResidentInProcessStreamTickSlice<'a> {
    pub device: &'a VulkanComputeDevice,
    pub mounted: &'a VulkanMountedPlacedStreamCircuit,
    pub execution_plan: &'a VulkanMountedPlacedResidentStreamTickExecutionPlan,
    pub dispatch_extensions: VulkanMountedPlacedResidentStreamTickDispatchExtensions<'a>,
    pub cursor: VulkanMountedPlacedResidentStreamTickCursor,
}

impl<'a> VulkanMountedPlacedResidentInProcessStreamTickSlice<'a> {
    pub fn new(
        device: &'a VulkanComputeDevice,
        mounted: &'a VulkanMountedPlacedStreamCircuit,
        execution_plan: &'a VulkanMountedPlacedResidentStreamTickExecutionPlan,
        stream_tick: u64,
    ) -> Self {
        Self {
            device,
            mounted,
            execution_plan,
            dispatch_extensions: VulkanMountedPlacedResidentStreamTickDispatchExtensions::default(),
            cursor: execution_plan.resident_stream_tick_cursor(stream_tick),
        }
    }

    pub fn new_with_dispatch_extensions(
        device: &'a VulkanComputeDevice,
        mounted: &'a VulkanMountedPlacedStreamCircuit,
        execution_plan: &'a VulkanMountedPlacedResidentStreamTickExecutionPlan,
        dispatch_extensions: VulkanMountedPlacedResidentStreamTickDispatchExtensions<'a>,
        stream_tick: u64,
    ) -> Self {
        Self {
            device,
            mounted,
            execution_plan,
            dispatch_extensions,
            cursor: execution_plan.compact_resident_stream_tick_cursor(stream_tick),
        }
    }

    pub fn device_id(&self) -> &str {
        self.mounted.device_id()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentInProcessStreamTickRun {
    pub status: VulkanMountedPlacedResidentInProcessStreamTickRunStatus,
    pub scheduler_turn_count: usize,
    pub completed_stage_delta: usize,
    pub completed_slice_count: usize,
    pub pending_slice_count: usize,
    pub transport_stats: VulkanPlacedCableTransportStats,
    pub device_runs: Vec<VulkanMountedPlacedResidentStreamTickRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedResidentInProcessStreamTickRunStatus {
    Completed,
    Blocked { pending_device_ids: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanMountedPlacedResidentInProcessSchedule {
    device_ids: Vec<String>,
    turns: Vec<Vec<usize>>,
}

impl VulkanMountedPlacedResidentInProcessSchedule {
    fn from_tick_plans(
        tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    ) -> Result<Self, VulkanError> {
        let device_ids = tick_plans
            .iter()
            .map(|plan| plan.device_id.clone())
            .collect::<Vec<_>>();
        let mut unique_device_ids = BTreeSet::new();
        for device_id in &device_ids {
            if !unique_device_ids.insert(device_id.clone()) {
                return Err(VulkanError(format!(
                    "placed activation schedule repeats device {device_id:?}"
                )));
            }
        }

        let mut next_stage_indices = vec![0usize; tick_plans.len()];
        let mut ready_cables = BTreeSet::<VulkanPlacedCablePacketKey>::new();
        let mut turns = Vec::new();

        while tick_plans
            .iter()
            .enumerate()
            .any(|(index, plan)| next_stage_indices[index] < plan.stages.len())
        {
            let mut turn = Vec::new();
            for (device_index, plan) in tick_plans.iter().enumerate() {
                let completed_before = next_stage_indices[device_index];
                while next_stage_indices[device_index] < plan.stages.len() {
                    match &plan.stages[next_stage_indices[device_index]] {
                        VulkanMountedPlacedStreamTickStage::ReceiveCable {
                            cable_index,
                            remote_device_id,
                            ..
                        } => {
                            let key = VulkanPlacedCablePacketKey {
                                cable_index: *cable_index,
                                from_device_id: remote_device_id.clone(),
                                to_device_id: plan.device_id.clone(),
                            };
                            if !ready_cables.remove(&key) {
                                break;
                            }
                        }
                        VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {}
                        VulkanMountedPlacedStreamTickStage::PublishCable {
                            cable_index,
                            remote_device_id,
                            ..
                        } => {
                            ready_cables.insert(VulkanPlacedCablePacketKey {
                                cable_index: *cable_index,
                                from_device_id: plan.device_id.clone(),
                                to_device_id: remote_device_id.clone(),
                            });
                        }
                    }
                    next_stage_indices[device_index] += 1;
                }
                if next_stage_indices[device_index] != completed_before {
                    turn.push(device_index);
                }
            }

            if turn.is_empty() {
                let pending_device_ids = tick_plans
                    .iter()
                    .enumerate()
                    .filter(|(index, plan)| next_stage_indices[*index] < plan.stages.len())
                    .map(|(_, plan)| plan.device_id.clone())
                    .collect::<Vec<_>>();
                return Err(VulkanError(format!(
                    "placed activation topology is blocked with pending devices {pending_device_ids:?}"
                )));
            }
            turns.push(turn);
        }

        if !ready_cables.is_empty() {
            return Err(VulkanError(format!(
                "placed activation topology leaves unconsumed cables {:?}",
                ready_cables.into_iter().collect::<Vec<_>>()
            )));
        }

        Ok(Self { device_ids, turns })
    }

    fn validate_slices(
        &self,
        slices: &[VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
    ) -> Result<(), VulkanError> {
        if slices.len() != self.device_ids.len() {
            return Err(VulkanError(format!(
                "placed activation schedule expects {} device slices, found {}",
                self.device_ids.len(),
                slices.len()
            )));
        }
        for (device_index, (expected, slice)) in self.device_ids.iter().zip(slices).enumerate() {
            if expected != slice.device_id() {
                return Err(VulkanError(format!(
                    "placed activation schedule device {device_index} is {expected:?}, found {:?}",
                    slice.device_id()
                )));
            }
        }
        Ok(())
    }
}

