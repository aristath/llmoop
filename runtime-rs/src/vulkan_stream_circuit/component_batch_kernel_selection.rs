struct VulkanComponentBatchDispatchStep {
    dispatch: VulkanResidentKernelDispatch,
    push_constants: Vec<VulkanKernelScalarBinding>,
    lane_index: Option<usize>,
    snapshot_state_buffer_indices: BTreeSet<usize>,
}

#[derive(Clone, Copy)]
enum VulkanComponentBatchStateSemantics<'a> {
    IndependentCandidates(&'a VulkanResidentStateTransactionBank),
    CausalSequence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionMode {
    IndependentCandidates,
    CausalSequence,
}

fn select_component_batch_kernel_artifact<'a>(
    artifacts: &'a [VulkanResidentComponentBatchKernelArtifact],
    component_id: &str,
    node_id: &str,
    execution_mode: VulkanComponentBatchExecutionMode,
    lane_capacity: usize,
) -> Option<&'a VulkanResidentComponentBatchKernelArtifact> {
    select_component_batch_kernel_artifact_where(
        artifacts,
        component_id,
        node_id,
        execution_mode,
        lane_capacity,
        |_| true,
    )
}

fn select_component_batch_kernel_artifact_where<'a>(
    artifacts: &'a [VulkanResidentComponentBatchKernelArtifact],
    component_id: &str,
    node_id: &str,
    execution_mode: VulkanComponentBatchExecutionMode,
    lane_capacity: usize,
    compatible: impl Fn(&VulkanResidentComponentBatchKernelArtifact) -> bool,
) -> Option<&'a VulkanResidentComponentBatchKernelArtifact> {
    artifacts
        .iter()
        .filter(|artifact| {
            artifact.component_id == component_id
                && artifact.node_id == node_id
                && artifact
                    .execution_domain
                    .supports_batch_mode(execution_mode)
                && (artifact.batch_mode == VulkanResidentComponentKernelBatchMode::WeightShared
                    || execution_mode == VulkanComponentBatchExecutionMode::CausalSequence)
                && artifact.is_exact_for(execution_mode)
                && compatible(artifact)
        })
        .min_by_key(|artifact| {
            if execution_mode == VulkanComponentBatchExecutionMode::CausalSequence {
                (0usize, usize::MAX - artifact.lane_tile_width)
            } else if artifact.lane_tile_width >= lane_capacity {
                (0usize, artifact.lane_tile_width)
            } else {
                (1usize, usize::MAX - artifact.lane_tile_width)
            }
        })
}

