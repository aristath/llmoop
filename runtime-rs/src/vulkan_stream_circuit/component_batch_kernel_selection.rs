struct VulkanComponentBatchDispatchStep {
    dispatch: VulkanResidentKernelDispatch,
    batch_control_byte_count: u32,
    push_constants: Vec<VulkanKernelScalarBinding>,
    lane_index: Option<usize>,
    commits_state: bool,
    snapshot_state_buffer_indices: BTreeSet<usize>,
}

fn component_batch_descriptors_commit_state<'a>(
    usages: impl IntoIterator<Item = &'a VulkanKernelDescriptorUsage>,
) -> bool {
    usages.into_iter().any(|usage| {
        matches!(
            usage,
            VulkanKernelDescriptorUsage::StateWrite | VulkanKernelDescriptorUsage::StateView
        )
    })
}

#[derive(Clone, Copy)]
enum VulkanComponentBatchStateSemantics<'a> {
    IndependentCandidates(&'a VulkanResidentStateTransactionBank),
    CausalSequence,
}

fn batch_stage_control_byte_count(stage: &VulkanResidentComponentBatchStageArtifact) -> u32 {
    if stage.shader_path.contains("sparse_moe_") {
        2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY
    } else if stage.shader_path.contains("append_gqa_attention_temporal_read")
        || stage.shader_path.contains("append_kv_temporal_commit")
        || stage
            .shader_path
            .contains("parallel_head_norm_rope_2way_temporal")
    {
        VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY
    } else {
        VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionMode {
    IndependentCandidates,
    CausalSequence,
}

impl VulkanComponentBatchExecutionMode {
    fn from_runtime_activation_batch_kind(kind: &RuntimeStreamActivationBatchKind) -> Self {
        match kind {
            RuntimeStreamActivationBatchKind::PrefillChunk { .. } => Self::CausalSequence,
            RuntimeStreamActivationBatchKind::DecodeFeedback { .. } => Self::IndependentCandidates,
        }
    }
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
