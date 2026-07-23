#[derive(Debug)]
pub enum VulkanMountedPlacedResidentInProcessStreamTickError {
    StreamTick(VulkanMountedPlacedResidentStreamTickError),
    Distributed(VulkanDistributedDispatchRunnerError),
    Schedule(VulkanError),
}

impl Display for VulkanMountedPlacedResidentInProcessStreamTickError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StreamTick(error) => Display::fmt(error, f),
            Self::Distributed(error) => Display::fmt(error, f),
            Self::Schedule(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanMountedPlacedResidentInProcessStreamTickError {}

impl From<VulkanMountedPlacedResidentStreamTickError>
    for VulkanMountedPlacedResidentInProcessStreamTickError
{
    fn from(error: VulkanMountedPlacedResidentStreamTickError) -> Self {
        Self::StreamTick(error)
    }
}

fn register_in_process_direct_edge_copies(
    slices: &[VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
    transport: &mut VulkanInProcessPlacedEdgeTransport,
) -> Result<(), VulkanMountedPlacedResidentInProcessStreamTickError> {
    for source_slice in slices {
        for outgoing in &source_slice.mounted.edge_io.outgoing_buffers {
            let destination_slice = slices
                .iter()
                .find(|slice| slice.device_id() == outgoing.endpoint.remote_device_id);
            let Some(destination_slice) = destination_slice else {
                continue;
            };
            let Some(incoming) = destination_slice
                .mounted
                .edge_io
                .incoming_buffer(outgoing.endpoint.edge_index)
            else {
                continue;
            };
            transport
                .register_direct_edge_copy(outgoing, incoming)
                .map_err(VulkanMountedPlacedResidentStreamTickError::Transport)?;
        }
    }
    Ok(())
}

fn wait_for_compact_slice_submitted_work(
    slice: &VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>,
    device_by_id: &BTreeMap<String, &VulkanComputeDevice>,
    distributed_runners: &VulkanDistributedDispatchRunners,
    last_submitted_segment: Option<&VulkanMountedPlacedResidentDispatchSegmentRunner>,
    completion_dependency: Option<(usize, u64)>,
    feedback_lane: Option<usize>,
    signal_completion: bool,
) -> Result<(), VulkanMountedPlacedResidentInProcessStreamTickError> {
    if !signal_completion {
        return Ok(());
    }
    if let Some((dispatch_index, _)) = completion_dependency {
        return distributed_runners
            .wait_dispatch(slice.device_id(), dispatch_index, |device_id| {
                device_by_id.get(device_id).copied().ok_or_else(|| {
                    VulkanError(format!(
                        "distributed shard device {device_id:?} is not mounted"
                    ))
                })
            })
            .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Distributed);
    }
    if let Some(segment) = last_submitted_segment {
        segment
            .wait_submitted(
                slice.device,
                slice.dispatch_extensions.sequence_variant,
                feedback_lane,
            )
            .map_err(|error| {
                VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                    VulkanMountedPlacedResidentStreamTickError::Dispatch(error),
                )
            })?;
    }
    Ok(())
}

fn wait_for_compact_slice_terminal_work(
    slice: &VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>,
    device_by_id: &BTreeMap<String, &VulkanComputeDevice>,
    distributed_runners: &VulkanDistributedDispatchRunners,
    feedback_lane: Option<usize>,
) -> Result<(), VulkanMountedPlacedResidentInProcessStreamTickError> {
    wait_for_compact_execution_plan_terminal_work(
        slice.device_id(),
        slice.device,
        slice.execution_plan,
        slice.dispatch_extensions.sequence_variant,
        device_by_id,
        distributed_runners,
        feedback_lane,
    )
}

fn wait_for_compact_execution_plan_terminal_work(
    device_id: &str,
    device: &VulkanComputeDevice,
    execution_plan: &VulkanMountedPlacedResidentStreamTickExecutionPlan,
    sequence_variant: u8,
    device_by_id: &BTreeMap<String, &VulkanComputeDevice>,
    distributed_runners: &VulkanDistributedDispatchRunners,
    feedback_lane: Option<usize>,
) -> Result<(), VulkanMountedPlacedResidentInProcessStreamTickError> {
    let terminal_segment = execution_plan.dispatch_segments.last();
    let terminal_distributed = execution_plan
        .distributed_dispatch_stages
        .iter()
        .next_back();
    if terminal_distributed.is_some_and(|(stage_index, _)| {
        terminal_segment.is_none_or(|segment| *stage_index >= segment.end_stage_index)
    }) {
        let (_, dispatch) = terminal_distributed.expect("terminal distributed stage exists");
        let dispatch_index = distributed_runners
            .leader_dispatch_index(device_id, dispatch.dispatch_index)
            .unwrap_or(dispatch.dispatch_index);
        return distributed_runners
            .wait_dispatch(device_id, dispatch_index, |shard_device_id| {
                device_by_id.get(shard_device_id).copied().ok_or_else(|| {
                    VulkanError(format!(
                        "distributed shard device {shard_device_id:?} is not mounted"
                    ))
                })
            })
            .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Distributed);
    }
    if let Some(segment) = terminal_segment {
        segment
            .wait_submitted(device, sequence_variant, feedback_lane)
            .map_err(|error| {
                VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                    VulkanMountedPlacedResidentStreamTickError::Dispatch(error),
                )
            })?;
    }
    Ok(())
}

fn prepare_compact_shared_stream_control(
    slices: &[VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
) -> Result<(), VulkanMountedPlacedResidentInProcessStreamTickError> {
    let mut compact_slices = slices
        .iter()
        .filter(|slice| !slice.cursor.capture_execution_trace);
    let Some(first) = compact_slices.next() else {
        return Ok(());
    };
    let dynamic_state_capacity_activations =
        u32::try_from(first.mounted.buffers.dynamic_state_capacity_activations).map_err(|_| {
            VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                VulkanMountedPlacedResidentStreamTickError::DynamicStateCapacityOverflow {
                    capacity: first.mounted.buffers.dynamic_state_capacity_activations,
                },
            )
        })?;
    for slice in compact_slices {
        let buffers_alias = Arc::ptr_eq(
            &first.mounted.stream_control_buffer,
            &slice.mounted.stream_control_buffer,
        ) || first
            .mounted
            .stream_control_buffer
            .shares_host_allocation_with(&slice.mounted.stream_control_buffer);
        if !buffers_alias
            || slice.cursor.stream_tick != first.cursor.stream_tick
            || slice.mounted.buffers.dynamic_state_capacity_activations
                != first.mounted.buffers.dynamic_state_capacity_activations
        {
            return Err(
                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                    "compact placed slices do not share one stream-control timeline".to_string(),
                )),
            );
        }
    }
    first
        .mounted
        .stream_control_buffer
        .write_bytes_at(
            VULKAN_STREAM_CONTROL_METADATA_OFFSET,
            &stream_control_metadata_bytes(VulkanMountedPlacedStreamControl {
                stream_tick: first.cursor.stream_tick,
                control_flags: 0,
                dynamic_state_capacity_activations,
            }),
        )
        .map_err(|error| {
            VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                VulkanMountedPlacedResidentStreamTickError::Dispatch(
                    VulkanMountedPlacedResidentKernelDispatchError::Vulkan(error),
                ),
            )
        })
}

#[derive(Clone, Copy)]
struct VulkanPlacedSubmissionPolicy {
    write_stream_control: bool,
    signal_completion: bool,
    wait_for_completion: bool,
    feedback_lane: Option<usize>,
}

impl VulkanPlacedSubmissionPolicy {
    const SYNCHRONOUS: Self = Self {
        write_stream_control: true,
        signal_completion: true,
        wait_for_completion: true,
        feedback_lane: None,
    };
}

#[derive(Clone, Copy)]
struct VulkanPlacedSubmissionContext<'a, 'batch> {
    policy: VulkanPlacedSubmissionPolicy,
    state_transactions: Option<&'a [VulkanResidentStateTransactionBank]>,
    feedback_turn: Option<VulkanPlacedFeedbackTimelineTurn<'a>>,
    output_turn: Option<VulkanPlacedOutputTimelineTurn<'a>>,
    submission_batch: Option<&'batch VulkanResidentQueueSubmissionBatch<'a>>,
}

impl VulkanPlacedSubmissionContext<'_, '_> {
    const SYNCHRONOUS: Self = Self {
        policy: VulkanPlacedSubmissionPolicy::SYNCHRONOUS,
        state_transactions: None,
        feedback_turn: None,
        output_turn: None,
        submission_batch: None,
    };
}

#[derive(Clone, Copy)]
struct VulkanPlacedSliceSubmissionContext<'a, 'batch> {
    policy: VulkanPlacedSubmissionPolicy,
    state_transaction: Option<&'a VulkanResidentStateTransactionBank>,
    feedback_turn: Option<VulkanPlacedFeedbackTimelineTurn<'a>>,
    output_turn: Option<VulkanPlacedOutputTimelineTurn<'a>>,
    submission_batch: Option<&'batch VulkanResidentQueueSubmissionBatch<'a>>,
}

