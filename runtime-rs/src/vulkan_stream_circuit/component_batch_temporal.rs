struct VulkanResidentPlacedTemporalBlockRunner {
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    input_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    output_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    pipeline: Vec<usize>,
}

struct VulkanResidentTemporalBlockRun {
    sampled_token_id: Option<u32>,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
    transport_stats: VulkanPlacedEdgeTransportStats,
}

enum VulkanComponentBatchEdgeTransferBinding {
    Resident(Box<VulkanResidentBufferCopy>),
    Mapped(VulkanResidentMappedBufferCopy),
}

struct VulkanComponentBatchEdgeTransfer {
    source_device_index: usize,
    destination_device_index: usize,
    edge_index: usize,
    binding: VulkanComponentBatchEdgeTransferBinding,
}

impl VulkanComponentBatchEdgeTransfer {
    fn run(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.binding {
            VulkanComponentBatchEdgeTransferBinding::Resident(copy) => copy
                .run(copy.byte_len())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop),
            VulkanComponentBatchEdgeTransferBinding::Mapped(copy) => copy
                .run(copy.byte_len())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop),
        }
    }
}

