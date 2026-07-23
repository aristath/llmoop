struct VulkanResidentPedalBatchKernelArtifact {
    pedal_id: String,
    node_id: String,
    execution_domain: VulkanResidentPedalKernelExecutionDomain,
    batch_mode: VulkanResidentPedalKernelBatchMode,
    lane_tile_width: usize,
    exact_primary_equivalence: bool,
    exact_causal_sequence_equivalence: bool,
    device_requirements: VulkanResidentVulkanDeviceRequirements,
    stages: Vec<VulkanResidentPedalBatchStageArtifact>,
}

impl VulkanResidentPedalBatchKernelArtifact {
    fn is_exact_for(&self, mode: VulkanPedalBatchExecutionMode) -> bool {
        match mode {
            VulkanPedalBatchExecutionMode::IndependentCandidates => self.exact_primary_equivalence,
            VulkanPedalBatchExecutionMode::CausalSequence => self.exact_causal_sequence_equivalence,
        }
    }
}

struct VulkanResidentPedalBatchStageArtifact {
    spirv_words: Vec<u32>,
    local_size_x: u32,
    workgroup_count_x: u32,
}
