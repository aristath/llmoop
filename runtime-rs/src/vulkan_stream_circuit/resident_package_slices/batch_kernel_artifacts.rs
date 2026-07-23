struct VulkanResidentComponentBatchKernelArtifact {
    component_id: String,
    node_id: String,
    execution_domain: VulkanResidentComponentKernelExecutionDomain,
    batch_mode: VulkanResidentComponentKernelBatchMode,
    lane_tile_width: usize,
    exact_primary_equivalence: bool,
    exact_causal_sequence_equivalence: bool,
    device_requirements: VulkanResidentVulkanDeviceRequirements,
    stages: Vec<VulkanResidentComponentBatchStageArtifact>,
}

impl VulkanResidentComponentBatchKernelArtifact {
    fn is_exact_for(&self, mode: VulkanComponentBatchExecutionMode) -> bool {
        match mode {
            VulkanComponentBatchExecutionMode::IndependentCandidates => self.exact_primary_equivalence,
            VulkanComponentBatchExecutionMode::CausalSequence => self.exact_causal_sequence_equivalence,
        }
    }
}

struct VulkanResidentComponentBatchStageArtifact {
    spirv_words: Vec<u32>,
    local_size_x: u32,
    workgroup_count_x: u32,
}
