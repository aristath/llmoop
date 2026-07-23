pub struct VulkanResidentInProcessPlacedModelPackage {
    pub package_id: String,
    pub input_device_id: String,
    pub output_device_id: String,
    pub dynamic_state_capacity_activations: usize,
    pub device_ids: Vec<String>,
    pub device_count: usize,
    pub hosted_component_count: usize,
    pub transducer_parameter_count: usize,
    pub transducer_parameter_bytes: usize,
    input_transducer_parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    output_transducer_parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    input_transducer_spirv_words: Vec<u32>,
    input_transducer_batch_spirv_words: Vec<u32>,
    embedding_norm_spirv_words: Vec<u32>,
    embedding_norm_batch_spirv_words: Vec<u32>,
    embedding_norm_batch_lane_tile_width: u32,
    tied_projection_spirv_words: Vec<u32>,
    tied_projection_batch_spirv_words: Vec<u32>,
    projection_batch_lane_tile_width: u32,
    sampler_kernels: Vec<VulkanResidentSamplerKernelArtifact>,
    input_transducer_spec: VulkanResidentInputEmbeddingTransducerSpec,
    output_transducer_spec: VulkanResidentOutputTransducerSpec,
    sampler_spec: VulkanResidentSamplerSpec,
    device_slices: Vec<Arc<VulkanResidentModelPackageDeviceSlice>>,
    speculative_decoders: Vec<VulkanResidentSpeculativeDecoderModelPackage>,
    distributed_execution_plan: VulkanDistributedExecutionPlan,
    distributed_activation_plan: VulkanDistributedActivationBufferPlan,
    distributed_parameter_allocation_plan: VulkanDistributedParameterAllocationPlan,
    distributed_parameter_exclusion_plan: VulkanDistributedParameterExclusionPlan,
    distributed_loaded_manifest: VulkanLoadedReusableKernelArtifactManifest,
    distributed_parameter_buffers: Arc<VulkanDistributedParameterBuffers>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentStreamStateDeclaration {
    pub key: TransientStateKey,
    pub block_shape: TransientStateBlockShape,
}

impl VulkanResidentInProcessPlacedModelPackage {
    pub fn transient_state_declarations(
        &self,
    ) -> Result<Vec<VulkanResidentStreamStateDeclaration>, VulkanResidentTokenModelPackageError>
    {
        let mut declarations = BTreeMap::new();
        for device_slice in &self.device_slices {
            for state in &device_slice
                .placed_plan
                .placed_resident_plan
                .resident_plan
                .stream_state_buffers
            {
                if let Some(declaration) =
                    transient_state_declaration_for_resident_state_buffer(state)?
                {
                    declarations.insert(declaration.key.clone(), declaration);
                }
            }
        }
        Ok(declarations.into_values().collect())
    }
}

fn transient_state_declaration_for_resident_state_buffer(
    state: &VulkanResidentStateBuffer,
) -> Result<Option<VulkanResidentStreamStateDeclaration>, VulkanResidentTokenModelPackageError> {
    let Some(bytes_per_activation) = state.bytes_per_activation else {
        return Ok(None);
    };
    let activation_capacity = state
        .max_dynamic_activations
        .map(|limit| limit.min(VULKAN_BACKEND_LOOP_MAX_WINDOW))
        .unwrap_or(VULKAN_BACKEND_LOOP_MAX_WINDOW);
    let block_shape = TransientStateBlockShape::new(bytes_per_activation, activation_capacity)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to declare transient state for {}.{}: {error}",
                state.component_id, state.state_id
            ))
        })?;
    Ok(Some(VulkanResidentStreamStateDeclaration {
        key: TransientStateKey::new(state.component_id.clone(), state.state_id.clone()),
        block_shape,
    }))
}
