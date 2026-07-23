pub struct VulkanResidentSpeculativeDecoderModelPackage {
    pub id: String,
    pub device_id: String,
    pub hosted_component_count: usize,
    package: VulkanResidentSpeculativeDecoderPackageSpec,
    device_slice: Arc<VulkanResidentModelPackageDeviceSlice>,
    additional_parameter_buffers: Option<Arc<VulkanPermanentParameterBuffers>>,
    input_embedding_spec: VulkanResidentInputEmbeddingTransducerSpec,
    input_embedding_spirv_words: Vec<u32>,
    output_norm_spirv_words: Vec<u32>,
    output_projection_spirv_words: Vec<u32>,
}

struct VulkanResidentSpeculativeDecoderLoadContext<'a> {
    manifest_dir: &'a Path,
    runtime_model: &'a VulkanResidentRuntimeModel,
    capacity: usize,
    tensor_index: &'a TensorIndex,
    target_output_parameters: &'a VulkanPermanentParameterBuffers,
    input_embedding_spec: &'a VulkanResidentInputEmbeddingTransducerSpec,
    input_embedding_spirv_words: &'a [u32],
}

impl VulkanResidentSpeculativeDecoderModelPackage {
    fn from_runtime_model(
        device: &VulkanComputeDevice,
        decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
        device_id: &str,
        context: &VulkanResidentSpeculativeDecoderLoadContext<'_>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let mut circuit_graph = decoder.circuit_graph.clone();
        for component in &mut circuit_graph.components {
            if matches!(
                component.runtime_role,
                CircuitRuntimeRole::DraftInputAdapter | CircuitRuntimeRole::DraftProcessor
            ) {
                component.runtime_role = CircuitRuntimeRole::SignalProcessor;
                component.circuit.runtime_role = CircuitRuntimeRole::SignalProcessor;
            }
        }
        let mut package = context.runtime_model.package.clone();
        package.package_id = format!("{}::{}", package.package_id, decoder.id);
        package.circuit_graph = circuit_graph.clone();
        package.component_executions = decoder.component_executions.clone();
        package.speculative_decoders.clear();
        let draft_runtime_model = VulkanResidentRuntimeModel {
            package,
            runtime_graph: context.runtime_model.runtime_graph.clone(),
            placement: StreamCircuitPlacementSpec::new(device_id),
            circuit_graph,
            component_executions: decoder.component_executions.clone(),
        };
        let device_slice = Arc::new(
            VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
                device,
                context.manifest_dir,
                draft_runtime_model,
                device_id,
                Some(context.capacity),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?,
        );

        let additional_tensors = [
            context.input_embedding_spec.parameter_tensor.as_str(),
            decoder.output_transducer.norm_parameter_tensor.as_str(),
            decoder
                .output_transducer
                .projection_parameter_tensor
                .as_str(),
        ]
        .into_iter()
        .filter(|tensor| {
            context
                .target_output_parameters
                .parameter_buffer(tensor)
                .is_none()
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        let additional_parameter_buffers = if additional_tensors.is_empty() {
            None
        } else {
            Some(Arc::new(
                load_resident_package_parameter_buffers_for_tensors(
                    device,
                    device_id,
                    context.tensor_index,
                    &additional_tensors,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?,
            ))
        };
        let output_norm_spirv_words = load_required_resident_model_package_shader(
            context.manifest_dir,
            &decoder.output_transducer.norm_shader_path,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let output_projection_spirv_words = load_required_resident_model_package_shader(
            context.manifest_dir,
            &decoder.output_transducer.projection_shader_path,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let mut input_embedding_spec = context.input_embedding_spec.clone();
        input_embedding_spec.transducer_id = decoder.input_adapter.component_id.clone();
        input_embedding_spec.output_signal_id =
            decoder.input_adapter.token_embedding_signal_id.clone();

        Ok(Self {
            id: decoder.id.clone(),
            device_id: device_id.to_string(),
            hosted_component_count: device_slice.hosted_component_count,
            package: decoder.clone(),
            device_slice,
            additional_parameter_buffers,
            input_embedding_spec,
            input_embedding_spirv_words: context.input_embedding_spirv_words.to_vec(),
            output_norm_spirv_words,
            output_projection_spirv_words,
        })
    }

    fn parameter<'a>(
        &'a self,
        target_output_parameters: &'a VulkanPermanentParameterBuffers,
        tensor: &str,
    ) -> Option<&'a VulkanPermanentParameterBufferAllocation> {
        self.additional_parameter_buffers
            .as_deref()
            .and_then(|buffers| buffers.parameter_buffer(tensor))
            .or_else(|| target_output_parameters.parameter_buffer(tensor))
    }

    fn output_transducer_spec(
        &self,
        input_signal_id: String,
    ) -> Result<VulkanResidentOutputTransducerSpec, VulkanResidentTokenModelPackageError> {
        let component = self
            .package
            .circuit_graph
            .components
            .iter()
            .find(|component| component.component_id == self.package.output_transducer.component_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative decoder {:?} has no output transducer component {:?}",
                    self.id, self.package.output_transducer.component_id
                ))
            })?;
        let node_ids = component
            .circuit
            .nodes
            .iter()
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        if node_ids.len() != 2 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} output transducer requires two nodes, found {}",
                self.id,
                node_ids.len()
            )));
        }
        let output = &self.package.output_transducer;
        Ok(VulkanResidentOutputTransducerSpec {
            transducer_id: output.component_id.clone(),
            input_signal_id,
            node_ids,
            norm_parameter_tensor: output.norm_parameter_tensor.clone(),
            norm_parameter_dtype: output.norm_parameter_dtype.clone(),
            norm_parameter_shape: output.norm_parameter_shape.clone(),
            norm_parameter_byte_capacity: output.norm_parameter_byte_capacity,
            projection_parameter_tensor: output.projection_parameter_tensor.clone(),
            projection_parameter_dtype: output.projection_parameter_dtype.clone(),
            projection_parameter_shape: output.projection_parameter_shape.clone(),
            projection_parameter_byte_capacity: output.projection_parameter_byte_capacity,
            projection_scale_parameter_tensor: output.projection_scale_parameter_tensor.clone(),
            projection_scale_parameter_dtype: output.projection_scale_parameter_dtype.clone(),
            projection_scale_parameter_shape: output.projection_scale_parameter_shape.clone(),
            projection_scale_parameter_byte_capacity: output
                .projection_scale_parameter_byte_capacity,
            input_frame_byte_capacity: output.input_frame_byte_capacity,
            normalized_frame_byte_capacity: output.output_hidden_byte_capacity,
            logits_byte_capacity: output.logits_byte_capacity,
            projection_workgroup_count_x: output.projection_workgroup_count_x,
            norm_local_size_x: output.norm_local_size_x,
            projection_local_size_x: output.projection_local_size_x,
        })
    }
}
