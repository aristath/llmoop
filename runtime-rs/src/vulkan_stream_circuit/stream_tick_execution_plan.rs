/// Resident execution structure for one placed device slice. Cable stages stay
/// visible to the scheduler, while every uninterrupted dispatch region becomes
/// one GPU submission.
pub struct VulkanMountedPlacedResidentStreamTickExecutionPlan {
    pub tick_plan: Arc<VulkanMountedPlacedStreamTickPlan>,
    pub dispatch_segment_count: usize,
    pub dispatch_count: usize,
    pub distributed_dispatch_count: usize,
    dispatch_segments: Vec<VulkanMountedPlacedResidentDispatchSegmentRunner>,
    distributed_dispatch_stages: BTreeMap<usize, VulkanMountedPlacedStreamTickDispatch>,
    distributed_dispatch_groups: BTreeMap<usize, VulkanMountedPlacedDistributedDispatchStageGroup>,
    distributed_dispatch_dependencies:
        BTreeMap<usize, VulkanMountedPlacedDistributedDispatchDependencies>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanMountedPlacedDistributedDispatchStageGroup {
    dispatches: Vec<VulkanMountedPlacedStreamTickDispatch>,
    end_stage_index: usize,
}

impl VulkanMountedPlacedDistributedDispatchStageGroup {
    fn leader(&self) -> &VulkanMountedPlacedStreamTickDispatch {
        self.dispatches
            .first()
            .expect("distributed stage groups are never empty")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanMountedPlacedDistributedDispatchDependencies {
    dispatch_index: usize,
    has_owner_producer: bool,
    has_owner_continuation: bool,
}

impl VulkanMountedPlacedResidentStreamTickExecutionPlan {
    pub fn from_tick_plan(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        Self::from_tick_plan_with_distributed_dispatches(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            tick_plan,
            &BTreeSet::new(),
        )
    }

    pub fn from_tick_plan_with_distributed_dispatches(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
        distributed_dispatch_indices: &BTreeSet<usize>,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        let distributed_dispatch_groups = distributed_dispatch_indices
            .iter()
            .map(|dispatch_index| vec![*dispatch_index])
            .collect::<Vec<_>>();
        Self::from_tick_plan_with_distributed_dispatch_groups(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            tick_plan,
            &distributed_dispatch_groups,
        )
    }

    pub fn from_tick_plan_with_distributed_dispatch_groups(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
        distributed_dispatch_groups: &[Vec<usize>],
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        if tick_plan.device_id != mounted.device_id() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::ExecutionPlanDeviceMismatch {
                    plan_device_id: tick_plan.device_id.clone(),
                    mounted_device_id: mounted.device_id().to_string(),
                },
            );
        }
        if tick_plan.device_id != mounted_bound_plan.device_id {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::ExecutionBoundPlanDeviceMismatch {
                    plan_device_id: tick_plan.device_id.clone(),
                    bound_plan_device_id: mounted_bound_plan.device_id.clone(),
                },
            );
        }

        let distributed_dispatch_indices = distributed_dispatch_groups
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let distributed_dispatch_stages =
            distributed_dispatch_stages(&tick_plan, &distributed_dispatch_indices)?;
        let distributed_dispatch_groups = distributed_dispatch_stage_groups(
            &distributed_dispatch_stages,
            distributed_dispatch_groups,
        )?;

        let dispatch_segment_stage_ranges =
            resident_dispatch_segment_stage_ranges_excluding_dispatches(
                &tick_plan.stages,
                &distributed_dispatch_indices,
            );
        let distributed_dispatch_dependencies = distributed_dispatch_dependency_topologies(
            &distributed_dispatch_groups,
            &dispatch_segment_stage_ranges,
        );
        let mut dispatch_segments = Vec::new();
        for &(start, end) in &dispatch_segment_stage_ranges {
            dispatch_segments.push(
                VulkanMountedPlacedResidentDispatchSegmentRunner::from_dispatch_stages(
                    device,
                    mounted,
                    mounted_bound_plan,
                    loaded_manifest,
                    &tick_plan.stages[start..end],
                )?,
            );
        }
        if dispatch_segments.is_empty() && distributed_dispatch_stages.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingExecutionDispatchSegments {
                    device_id: tick_plan.device_id.clone(),
                },
            );
        }
        let dispatch_count = dispatch_segments
            .iter()
            .map(|segment| segment.dispatch_count)
            .sum();
        let dispatch_segment_count = dispatch_segments.len();
        let distributed_dispatch_count = distributed_dispatch_stages.len();
        Ok(Self {
            tick_plan: Arc::new(tick_plan),
            dispatch_segment_count,
            dispatch_count,
            distributed_dispatch_count,
            dispatch_segments,
            distributed_dispatch_stages,
            distributed_dispatch_groups,
            distributed_dispatch_dependencies,
        })
    }

    fn segment_starting_at(
        &self,
        stage_index: usize,
    ) -> Option<&VulkanMountedPlacedResidentDispatchSegmentRunner> {
        self.dispatch_segments
            .iter()
            .find(|segment| segment.start_stage_index == stage_index)
    }

    fn first_dispatch_segment_stage_index(&self) -> Option<usize> {
        self.dispatch_segments
            .first()
            .map(|segment| segment.start_stage_index)
    }

    fn last_dispatch_segment_stage_index(&self) -> Option<usize> {
        self.dispatch_segments
            .last()
            .map(|segment| segment.start_stage_index)
    }

    pub fn distributed_dispatch_at_stage(
        &self,
        stage_index: usize,
    ) -> Option<&VulkanMountedPlacedStreamTickDispatch> {
        self.distributed_dispatch_groups
            .get(&stage_index)
            .map(VulkanMountedPlacedDistributedDispatchStageGroup::leader)
    }

    fn distributed_dispatch_group_at_stage(
        &self,
        stage_index: usize,
    ) -> Option<&VulkanMountedPlacedDistributedDispatchStageGroup> {
        self.distributed_dispatch_groups.get(&stage_index)
    }

    fn distributed_dispatch_dependencies_at_stage(
        &self,
        stage_index: usize,
    ) -> Option<VulkanMountedPlacedDistributedDispatchDependencies> {
        self.distributed_dispatch_dependencies
            .get(&stage_index)
            .copied()
    }

    fn resident_stream_tick_cursor(
        &self,
        stream_tick: u64,
    ) -> VulkanMountedPlacedResidentStreamTickCursor {
        VulkanMountedPlacedResidentStreamTickCursor::new_shared(
            Arc::clone(&self.tick_plan),
            stream_tick,
            true,
        )
    }

    fn compact_resident_stream_tick_cursor(
        &self,
        stream_tick: u64,
    ) -> VulkanMountedPlacedResidentStreamTickCursor {
        VulkanMountedPlacedResidentStreamTickCursor::new_shared(
            Arc::clone(&self.tick_plan),
            stream_tick,
            false,
        )
    }
}

fn distributed_dispatch_dependency_topologies(
    distributed_dispatch_groups: &BTreeMap<usize, VulkanMountedPlacedDistributedDispatchStageGroup>,
    dispatch_segment_stage_ranges: &[(usize, usize)],
) -> BTreeMap<usize, VulkanMountedPlacedDistributedDispatchDependencies> {
    distributed_dispatch_groups
        .iter()
        .map(|(stage_index, group)| {
            (
                *stage_index,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: group.leader().dispatch_index,
                    has_owner_producer: dispatch_segment_stage_ranges
                        .iter()
                        .any(|(_, end)| end == stage_index),
                    has_owner_continuation: dispatch_segment_stage_ranges
                        .iter()
                        .any(|(start, _)| *start == group.end_stage_index),
                },
            )
        })
        .collect()
}

fn distributed_dispatch_stage_groups(
    distributed_dispatch_stages: &BTreeMap<usize, VulkanMountedPlacedStreamTickDispatch>,
    dispatch_groups: &[Vec<usize>],
) -> Result<
    BTreeMap<usize, VulkanMountedPlacedDistributedDispatchStageGroup>,
    VulkanMountedPlacedResidentKernelDispatchError,
> {
    let stages_by_dispatch = distributed_dispatch_stages
        .iter()
        .map(|(stage_index, dispatch)| (dispatch.dispatch_index, (*stage_index, dispatch)))
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::new();
    let mut claimed_dispatches = BTreeSet::new();
    for dispatch_indices in dispatch_groups {
        let Some(leader_dispatch_index) = dispatch_indices.first().copied() else {
            continue;
        };
        let (leader_stage_index, _) = stages_by_dispatch
            .get(&leader_dispatch_index)
            .copied()
            .ok_or_else(|| {
                VulkanMountedPlacedResidentKernelDispatchError::MissingDistributedDispatchStage {
                    device_id: "distributed execution plan".to_string(),
                    dispatch_index: leader_dispatch_index,
                }
            })?;
        let mut dispatches = Vec::with_capacity(dispatch_indices.len());
        for (offset, dispatch_index) in dispatch_indices.iter().copied().enumerate() {
            if !claimed_dispatches.insert(dispatch_index) {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::DistributedDispatchMismatch {
                        device_id: "distributed execution plan".to_string(),
                        stage_index: leader_stage_index + offset,
                        expected_dispatch_index: dispatch_index,
                        completed_dispatch_index: dispatch_index,
                    },
                );
            }
            let expected_stage_index = leader_stage_index + offset;
            let (stage_index, dispatch) = stages_by_dispatch
                .get(&dispatch_index)
                .copied()
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingDistributedDispatchStage {
                        device_id: "distributed execution plan".to_string(),
                        dispatch_index,
                    }
                })?;
            if stage_index != expected_stage_index {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::DistributedDispatchMismatch {
                        device_id: "distributed execution plan".to_string(),
                        stage_index: expected_stage_index,
                        expected_dispatch_index: dispatch_index,
                        completed_dispatch_index: dispatch.dispatch_index,
                    },
                );
            }
            dispatches.push(dispatch.clone());
        }
        groups.insert(
            leader_stage_index,
            VulkanMountedPlacedDistributedDispatchStageGroup {
                dispatches,
                end_stage_index: leader_stage_index + dispatch_indices.len(),
            },
        );
    }
    Ok(groups)
}

#[cfg(test)]
fn resident_dispatch_segment_stage_ranges(
    stages: &[VulkanMountedPlacedStreamTickStage],
) -> Vec<(usize, usize)> {
    resident_dispatch_segment_stage_ranges_excluding_dispatches(stages, &BTreeSet::new())
}

fn resident_dispatch_segment_stage_ranges_excluding_dispatches(
    stages: &[VulkanMountedPlacedStreamTickStage],
    excluded_dispatch_indices: &BTreeSet<usize>,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut stage_index = 0usize;
    while stage_index < stages.len() {
        if !is_canonical_dispatch_stage(&stages[stage_index], excluded_dispatch_indices) {
            stage_index += 1;
            continue;
        }
        let start = stage_index;
        while stage_index < stages.len()
            && is_canonical_dispatch_stage(&stages[stage_index], excluded_dispatch_indices)
        {
            stage_index += 1;
        }
        ranges.push((start, stage_index));
    }
    ranges
}

fn is_canonical_dispatch_stage(
    stage: &VulkanMountedPlacedStreamTickStage,
    excluded_dispatch_indices: &BTreeSet<usize>,
) -> bool {
    matches!(
        stage,
        VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. }
            if !excluded_dispatch_indices.contains(&dispatch.dispatch_index)
    )
}

fn distributed_dispatch_stages(
    tick_plan: &VulkanMountedPlacedStreamTickPlan,
    distributed_dispatch_indices: &BTreeSet<usize>,
) -> Result<
    BTreeMap<usize, VulkanMountedPlacedStreamTickDispatch>,
    VulkanMountedPlacedResidentKernelDispatchError,
> {
    let mut stages = BTreeMap::new();
    let mut found = BTreeSet::new();
    for stage in &tick_plan.stages {
        let VulkanMountedPlacedStreamTickStage::Dispatch {
            stage_index,
            dispatch,
        } = stage
        else {
            continue;
        };
        if distributed_dispatch_indices.contains(&dispatch.dispatch_index) {
            found.insert(dispatch.dispatch_index);
            stages.insert(*stage_index, dispatch.clone());
        }
    }
    if let Some(dispatch_index) = distributed_dispatch_indices.difference(&found).next() {
        return Err(
            VulkanMountedPlacedResidentKernelDispatchError::MissingDistributedDispatchStage {
                device_id: tick_plan.device_id.clone(),
                dispatch_index: *dispatch_index,
            },
        );
    }
    Ok(stages)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentPedalboardRun {
    pub device_id: String,
    pub pedal_runs: Vec<VulkanMountedPlacedResidentPedalRun>,
}

impl VulkanMountedPlacedResidentPedalboardRun {
    pub fn pedal_count(&self) -> usize {
        self.pedal_runs.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.pedal_runs
            .iter()
            .map(VulkanMountedPlacedResidentPedalRun::dispatch_count)
            .sum()
    }

    pub fn run_time_ns(&self) -> u64 {
        self.pedal_runs.iter().fold(0u64, |total, pedal| {
            total.saturating_add(pedal.run_time_ns())
        })
    }

    pub fn pedal_ids(&self) -> Vec<&str> {
        self.pedal_runs
            .iter()
            .map(|pedal| pedal.pedal_id.as_str())
            .collect()
    }
}
