impl VulkanResidentModelPackage {
    pub fn from_manifest_file(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        Self::from_manifest_file_with_capacity(device, manifest_path, None)
    }

    pub fn from_manifest_file_with_capacity(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let manifest_path = manifest_path.as_ref();
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(manifest_path).map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load resident model package manifest {:?}: {error}",
                    manifest_path
                ))
            })?;
        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let runtime_model =
            manifest.mount_runtime_patch_controls(None, &BTreeMap::new(), &[], None)?;
        Self::from_runtime_model(
            device,
            &manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
        )
    }

    pub fn from_runtime_model(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let manifest_dir = manifest_dir.as_ref();
        let capacity = dynamic_state_capacity_activations
            .unwrap_or(runtime_model.package.max_context_activations);
        if capacity == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(
                "resident dynamic state capacity must be at least 1 activation",
            ));
        }
        validate_pedal_executions(
            &runtime_model.package.package_id,
            &runtime_model.pedal_executions,
        )?;

        let tensor_index_path = resolve_resident_model_package_path(
            manifest_dir,
            &runtime_model.package.tensor_index_path,
        );
        let default_device_id = runtime_model.placement.default_device_id.clone();
        let (tensor_index, resource_plan, placed_plan) =
            plan_resident_package_single_device_stream_circuit(
                &default_device_id,
                &runtime_model.placement,
                &runtime_model.circuit_graph,
                manifest_dir,
                &tensor_index_path,
                runtime_model.package.activation_element_bytes,
            )?;
        let parameter_buffer_plan = VulkanPermanentParameterBufferPlan::from_placed_resident_plan(
            &placed_plan.placed_resident_plan,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to create resident parameter buffer plan: {error}"
            ))
        })?;
        let parameter_buffers = Arc::new(parameter_buffer_plan.allocate_buffers(device).map_err(
            |error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to allocate resident parameter buffers: {error}"
                ))
            },
        )?);
        parameter_buffers
            .load_from_tensor_index(&tensor_index)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load resident model parameters: {error}"
                ))
            })?;

        let transducer_parameter_buffers =
            Arc::new(load_resident_package_transducer_parameter_buffers(
                device,
                &default_device_id,
                &resource_plan,
                &tensor_index,
            )?);
        let input_transducer_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model.package.input_transducer.shader_path,
        )?;
        let embedding_norm_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .embedding_norm_shader_path,
        )?;
        let tied_projection_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .projection_shader_path,
        )?;
        let sampler_kernels =
            load_resident_sampler_kernels(manifest_dir, &runtime_model.package.sampler)?;

        let probe_mounted =
            VulkanMountedPlacedStreamCircuit::from_placed_plan_with_parameter_buffers(
                device,
                placed_plan.clone(),
                capacity,
                parameter_buffers.clone(),
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to mount Vulkan stream circuit for shared model binding: {error}"
                ))
            })?;
        let reusable_manifest =
            resident_package_reusable_kernel_manifest(&probe_mounted.placed_plan);
        let prepared_plan = placed_plan
            .prepared_dispatch_plan(&reusable_manifest, capacity)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to prepare Vulkan dispatch plan: {error}"
                ))
            })?;
        let mounted_bound = probe_mounted
            .mounted_placed_bound_dispatch_plan(&reusable_manifest)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to bind Vulkan stream circuit dispatch plan: {error}"
                ))
            })?;
        validate_pedal_executions_against_mounted_dispatches(
            &runtime_model.package.package_id,
            &runtime_model.pedal_executions,
            &mounted_bound,
        )?;
        let pedal_kernel_shaders =
            resident_package_pedal_kernel_shader_refs(&runtime_model.pedal_executions);
        let loaded_manifest = loaded_kernel_pack_from_package_shader_refs(
            manifest_dir,
            &placed_plan,
            &prepared_plan,
            &pedal_kernel_shaders,
        )?;

        Ok(Self {
            package_id: runtime_model.package.package_id.clone(),
            device_id: default_device_id,
            dynamic_state_capacity_activations: capacity,
            permanent_parameter_count: parameter_buffers.plan.parameter_count,
            permanent_parameter_bytes: parameter_buffers.total_byte_capacity,
            transducer_parameter_count: transducer_parameter_buffers.plan.parameter_count,
            transducer_parameter_bytes: transducer_parameter_buffers.total_byte_capacity,
            reusable_kernel_word_count: loaded_manifest.total_word_count,
            placed_plan,
            mounted_bound,
            loaded_manifest,
            parameter_buffers,
            transducer_parameter_buffers,
            input_transducer_spirv_words,
            embedding_norm_spirv_words,
            tied_projection_spirv_words,
            sampler_kernels,
            input_transducer_spec: runtime_model.package.input_transducer.spec.clone(),
            output_transducer_spec: runtime_model.package.output_transducer.spec.clone(),
            sampler_spec: runtime_model.package.sampler.spec.clone(),
        })
    }

    pub fn create_stream_processor(
        &self,
        device: &VulkanComputeDevice,
        random_seed: u32,
    ) -> Result<VulkanResidentStreamProcessor, VulkanResidentTokenModelPackageError> {
        self.create_stream_processor_with_state_source(device, random_seed, None)
    }

    pub fn create_stream_processor_inheriting_state(
        &self,
        device: &VulkanComputeDevice,
        random_seed: u32,
        source: &VulkanResidentStreamProcessor,
    ) -> Result<VulkanResidentStreamProcessor, VulkanResidentTokenModelPackageError> {
        self.create_stream_processor_with_state_source(device, random_seed, Some(source))
    }

    fn create_stream_processor_with_state_source(
        &self,
        device: &VulkanComputeDevice,
        random_seed: u32,
        source: Option<&VulkanResidentStreamProcessor>,
    ) -> Result<VulkanResidentStreamProcessor, VulkanResidentTokenModelPackageError> {
        let mounted = VulkanMountedPlacedStreamCircuit::from_placed_plan_with_parameter_buffers(
            device,
            self.placed_plan.clone(),
            self.dynamic_state_capacity_activations,
            self.parameter_buffers.clone(),
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to mount Vulkan stream circuit for stream instance: {error}"
            ))
        })?;
        mounted.buffers.zero_state_buffers().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to zero stream state buffers: {error}"
            ))
        })?;
        let inherited = source
            .map(|source| {
                mounted
                    .buffers
                    .inherit_matching_state_from(&source._mounted.buffers)
            })
            .transpose()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to inherit stream state: {error}"
                ))
            })?
            .map(|(_, copied)| copied)
            .unwrap_or_default();
        mounted
            .buffers
            .apply_clone_state_policies_after(&inherited)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to initialize cloned stream state: {error}"
                ))
            })?;
        let pedal_ids = self
            .placed_plan
            .placed_resident_plan
            .hosted_pedal_ids
            .clone();
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding(
                device,
                &mounted,
                &self.transducer_parameter_buffers,
                &self.input_transducer_spirv_words,
                &self.input_transducer_spec,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to create input token embedding transducer: {error}"
                ))
            })?;
        let pedalboard = mounted
            .create_resident_pedalboard_runner(
                device,
                &self.mounted_bound,
                pedal_ids.iter().map(String::as_str),
                &self.loaded_manifest,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to create resident pedalboard runner: {error}"
                ))
            })?;
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
                device,
                &mounted,
                &self.transducer_parameter_buffers,
                &self.embedding_norm_spirv_words,
                &self.tied_projection_spirv_words,
                &self.output_transducer_spec,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to create output transducer: {error}"
                ))
            })?;
        let sampler = VulkanResidentSamplerRunner::from_output_transducer_with_spec(
            device,
            &mounted,
            &output_transducer,
            &self.sampler_kernels,
            &self.sampler_spec,
            random_seed,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to create sampler pedal: {error}"
            ))
        })?;
        let tick_runner = VulkanResidentSingleTokenTickRunner::new(
            device,
            input_transducer,
            pedalboard,
            output_transducer,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to create single-token tick runner: {error}"
            ))
        })?;
        let loop_runner =
            VulkanResidentFeedbackLoopRunner::new(tick_runner, sampler).map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to create feedback loop runner: {error}"
                ))
            })?;

        VulkanResidentStreamProcessor::new(
            device,
            mounted,
            self.transducer_parameter_buffers.clone(),
            loop_runner,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to create resident feedback snapshot bank: {error}"
            ))
        })
    }
}

impl VulkanResidentTokenModelPackage for VulkanResidentModelPackage {
    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn dynamic_state_capacity_activations(&self) -> usize {
        self.dynamic_state_capacity_activations
    }

    fn permanent_parameter_count(&self) -> usize {
        self.permanent_parameter_count
    }

    fn permanent_parameter_bytes(&self) -> usize {
        self.permanent_parameter_bytes
    }

    fn transducer_parameter_count(&self) -> usize {
        self.transducer_parameter_count
    }

    fn transducer_parameter_bytes(&self) -> usize {
        self.transducer_parameter_bytes
    }

    fn reusable_kernel_word_count(&self) -> usize {
        self.reusable_kernel_word_count
    }

    fn create_stream_processor(
        &self,
        device: &VulkanComputeDevice,
        random_seed: u32,
    ) -> Result<VulkanResidentStreamProcessor, VulkanResidentTokenModelPackageError> {
        VulkanResidentModelPackage::create_stream_processor(self, device, random_seed)
    }
}

