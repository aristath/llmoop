struct VulkanResidentPlacedTemporalBlockRunner {
    pedalboard: VulkanResidentPlacedPedalBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    input_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    output_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    pipeline: Vec<usize>,
}

struct VulkanResidentTemporalBlockRun {
    sampled_token_id: Option<u32>,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
    transport_stats: VulkanPlacedCableTransportStats,
}

enum VulkanPedalBatchCableTransferBinding {
    Resident(Box<VulkanResidentBufferCopy>),
    Mapped(VulkanResidentMappedBufferCopy),
}

struct VulkanPedalBatchCableTransfer {
    source_device_index: usize,
    destination_device_index: usize,
    cable_index: usize,
    binding: VulkanPedalBatchCableTransferBinding,
}

impl VulkanPedalBatchCableTransfer {
    fn run(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.binding {
            VulkanPedalBatchCableTransferBinding::Resident(copy) => copy
                .run(copy.byte_len())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop),
            VulkanPedalBatchCableTransferBinding::Mapped(copy) => copy
                .run(copy.byte_len())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop),
        }
    }
}

