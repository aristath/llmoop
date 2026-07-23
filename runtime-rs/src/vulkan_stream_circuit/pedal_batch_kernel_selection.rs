struct VulkanPedalBatchDispatchStep {
    dispatch: VulkanResidentKernelDispatch,
    push_constants: Vec<VulkanKernelScalarBinding>,
    lane_index: Option<usize>,
    snapshot_state_buffer_indices: BTreeSet<usize>,
}

#[derive(Clone, Copy)]
enum VulkanPedalBatchStateSemantics<'a> {
    IndependentCandidates(&'a VulkanResidentStateTransactionBank),
    CausalSequence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanPedalBatchExecutionMode {
    IndependentCandidates,
    CausalSequence,
}

fn select_pedal_batch_kernel_artifact<'a>(
    artifacts: &'a [VulkanResidentPedalBatchKernelArtifact],
    pedal_id: &str,
    node_id: &str,
    execution_mode: VulkanPedalBatchExecutionMode,
    lane_capacity: usize,
) -> Option<&'a VulkanResidentPedalBatchKernelArtifact> {
    select_pedal_batch_kernel_artifact_where(
        artifacts,
        pedal_id,
        node_id,
        execution_mode,
        lane_capacity,
        |_| true,
    )
}

fn select_pedal_batch_kernel_artifact_where<'a>(
    artifacts: &'a [VulkanResidentPedalBatchKernelArtifact],
    pedal_id: &str,
    node_id: &str,
    execution_mode: VulkanPedalBatchExecutionMode,
    lane_capacity: usize,
    compatible: impl Fn(&VulkanResidentPedalBatchKernelArtifact) -> bool,
) -> Option<&'a VulkanResidentPedalBatchKernelArtifact> {
    artifacts
        .iter()
        .filter(|artifact| {
            artifact.pedal_id == pedal_id
                && artifact.node_id == node_id
                && artifact
                    .execution_domain
                    .supports_batch_mode(execution_mode)
                && (artifact.batch_mode == VulkanResidentPedalKernelBatchMode::WeightShared
                    || execution_mode == VulkanPedalBatchExecutionMode::CausalSequence)
                && artifact.is_exact_for(execution_mode)
                && compatible(artifact)
        })
        .min_by_key(|artifact| {
            if execution_mode == VulkanPedalBatchExecutionMode::CausalSequence {
                (0usize, usize::MAX - artifact.lane_tile_width)
            } else if artifact.lane_tile_width >= lane_capacity {
                (0usize, artifact.lane_tile_width)
            } else {
                (1usize, usize::MAX - artifact.lane_tile_width)
            }
        })
}

