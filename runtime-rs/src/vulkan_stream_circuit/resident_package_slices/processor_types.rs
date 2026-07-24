pub struct VulkanResidentInProcessPlacedStreamProcessor {
    distributed_dispatch_runners: VulkanDistributedDispatchRunners,
    _distributed_activation_buffers: VulkanDistributedActivationBuffers,
    edge_synchronizations: VulkanPlacedEdgeTimelineSynchronizations,
    model: Arc<VulkanResidentInProcessPlacedModelPackage>,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
    sampler: VulkanResidentSamplerRunner,
    output_synchronization: VulkanResidentPlacedOutputTimelineSynchronization,
    resident_feedback_loop: Option<VulkanResidentInProcessPlacedFeedbackLoop>,
    activation_schedule: VulkanMountedPlacedResidentInProcessSchedule,
    device_slices: Vec<VulkanResidentInProcessPlacedStreamProcessorDevice>,
    execution_quantum_calibrators:
        BTreeMap<String, Rc<RefCell<RuntimeExecutionQuantumCalibrator>>>,
    speculative_decoders: Vec<VulkanResidentSpeculativeDecoderProcessor>,
    verification_state_transactions: RefCell<Option<Vec<VulkanResidentStateTransactionBank>>>,
    component_batch_execution: RefCell<Option<VulkanResidentPlacedComponentBatchRunner>>,
    verification_input_embedding: RefCell<Option<VulkanResidentBatchedInputEmbeddingRunner>>,
    temporal_block_execution: RefCell<Option<VulkanResidentPlacedTemporalBlockRunner>>,
    batched_output_projection: RefCell<Option<VulkanResidentBatchedOutputProjectionRunner>>,
}

fn create_placed_state_transactions<'a, F>(
    devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    transaction_width: usize,
    device_for: &F,
) -> Result<Vec<VulkanResidentStateTransactionBank>, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    devices
        .iter()
        .map(|slice| {
            VulkanResidentStateTransactionBank::new_transactional(
                device_for(&slice.device_id)?,
                &slice.mounted.buffers,
                transaction_width,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        })
        .collect()
}

struct VulkanResidentSpeculativeDecoderProcessor {
    id: String,
    device_id: String,
    mounted: VulkanMountedPlacedStreamCircuit,
    execution_plan: VulkanMountedPlacedResidentStreamTickExecutionPlan,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
    sampler: VulkanResidentSamplerRunner,
    draft_sequence: VulkanResidentKernelSequence,
    state_sequence: VulkanResidentKernelSequence,
    hidden_input_signal_id: String,
    recursive_hidden_copy: VulkanResidentBufferCopy,
    pending_hidden_input_copy: VulkanResidentBufferCopy,
    update_pending_hidden_copy: VulkanResidentBufferCopy,
    pending_target_hidden: VulkanResidentBuffer,
    state_transaction: VulkanResidentStateTransactionBank,
}

#[derive(Clone, Copy)]
enum VulkanDraftHiddenSource {
    PendingTarget,
    Recursive,
}
