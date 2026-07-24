impl VulkanResidentInProcessPlacedModelPackage {
    pub fn from_manifest_file(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_manifest_file_with_capacity(device, manifest_path, None)
    }

    pub fn from_manifest_file_with_capacity(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let manifest_path = manifest_path.as_ref();
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(manifest_path).map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to load resident placed model package manifest {:?}: {error}",
                        manifest_path
                    )),
                )
            })?;
        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let runtime_model = manifest
            .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        Self::from_runtime_model_for_devices(
            device,
            &manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
        )
    }

    pub fn from_runtime_model_for_devices(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
            true,
            |_| Ok(device),
        )
    }

    pub fn from_runtime_model_for_bound_devices(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        mount_speculative_decoders: bool,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
            mount_speculative_decoders,
            |device_id| {
                devices
                    .get(device_id)
                    .map(|device| device.as_ref())
                    .ok_or_else(
                        || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: device_id.to_string(),
                        },
                    )
            },
        )
    }

    fn from_runtime_model_for_device_resolver<'a, F>(
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        mount_speculative_decoders: bool,
        device_for: F,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        let manifest_dir = manifest_dir.as_ref();
        let package_id = runtime_model.package.package_id.clone();
        let (input_processor_id, output_processor_id) = runtime_model
            .circuit_graph
            .signal_processor_endpoint_component_ids()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let input_device_id = runtime_model
            .placement
            .device_for_component(&input_processor_id)
            .to_string();
        let output_device_id = runtime_model
            .placement
            .device_for_component(&output_processor_id)
            .to_string();
        let capacity = dynamic_state_capacity_activations
            .unwrap_or(runtime_model.package.max_context_activations);
        if capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident dynamic state capacity must be at least 1 activation",
                ),
            ));
        }
        let runtime_execution_identity = canonical_runtime_execution_identity(
            &runtime_model,
            capacity,
            mount_speculative_decoders,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let tensor_index_path = resolve_resident_model_package_path(
            manifest_dir,
            &runtime_model.package.tensor_index_path,
        );
        let (tensor_index, resource_plan, _placement_plan, _boundary_placed_plan) =
            plan_resident_package_placed_stream_circuit(
                &input_device_id,
                &runtime_model.placement,
                &runtime_model.circuit_graph,
                manifest_dir,
                &tensor_index_path,
                runtime_model.package.activation_element_bytes,
            )?;
        let input_device = device_for(&input_device_id)?;
        let output_device = device_for(&output_device_id)?;
        let (input_transducer_parameter_buffers, output_transducer_parameter_buffers) =
            if input_device_id == output_device_id {
                let shared = Arc::new(load_resident_package_transducer_parameter_buffers(
                    input_device,
                    &input_device_id,
                    &resource_plan,
                    &tensor_index,
                )?);
                (shared.clone(), shared)
            } else {
                (
                    Arc::new(load_resident_package_transducer_parameter_buffers_for(
                        input_device,
                        &input_device_id,
                        &resource_plan,
                        &tensor_index,
                        "input_transducer",
                    )?),
                    Arc::new(load_resident_package_transducer_parameter_buffers_for(
                        output_device,
                        &output_device_id,
                        &resource_plan,
                        &tensor_index,
                        "output_transducer",
                    )?),
                )
            };
        let input_transducer_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model.package.input_transducer.shader_path,
        )?;
        let input_transducer_batch_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model.package.input_transducer.batch_shader_path,
        )?;
        let embedding_norm_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .embedding_norm_shader_path,
        )?;
        let embedding_norm_batch_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .embedding_norm_batch_shader_path,
        )?;
        let tied_projection_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .projection_shader_path,
        )?;
        let tied_projection_batch_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .projection_batch_shader_path,
        )?;
        let sampler_kernels =
            load_resident_sampler_kernels(manifest_dir, &runtime_model.package.sampler)?;
        let device_ids = runtime_model
            .circuit_graph
            .signal_processor_device_ids(&runtime_model.placement);
        let mut device_slice_plans = Vec::with_capacity(device_ids.len());
        let mut hosted_component_count = 0usize;

        for device_id in &device_ids {
            let slice_device = device_for(device_id)?;
            let package_slice = VulkanResidentModelPackageDeviceSlicePlan::prepare(
                slice_device,
                manifest_dir,
                &runtime_model,
                &tensor_index,
                device_id,
                capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            hosted_component_count = hosted_component_count
                .checked_add(package_slice.hosted_component_count)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(
                            "placed package hosted component count overflowed",
                        ),
                    )
                })?;
            device_slice_plans.push(package_slice);
        }

        let prepared_plans = device_slice_plans
            .iter()
            .map(|slice| (slice.device_id.as_str(), &slice.prepared_plan))
            .collect::<Vec<_>>();
        let distributed_loaded_manifest =
            resident_package_loaded_kernel_manifest_for_slice_plans(&device_slice_plans)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let storage_buffer_offset_alignment = device_ids
            .iter()
            .map(|device_id| {
                device_for(device_id).map(VulkanComputeDevice::min_storage_buffer_offset_alignment)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(1);
        let distributed_execution_plan = VulkanDistributedExecutionPlan::for_placed_components(
            &device_ids,
            storage_buffer_offset_alignment,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to lower distributed Vulkan execution: {error}"
                )),
            )
        })?;
        let distributed_activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&distributed_execution_plan)
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to plan distributed Vulkan activations: {error}"
                        )),
                    )
                })?;
        let distributed_parameter_allocation_plan =
            VulkanDistributedParameterAllocationPlan::from_execution_plan(
                &distributed_execution_plan,
                &tensor_index,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to plan distributed Vulkan parameter shards: {error}"
                    )),
                )
            })?;
        let distributed_parameter_exclusion_plan =
            VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
                &distributed_execution_plan,
                &prepared_plans,
                &tensor_index,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to prove distributed Vulkan parameter replacement: {error}"
                    )),
                )
            })?;
        let distributed_parameter_buffers = Arc::new(
            VulkanDistributedParameterBuffers::allocate_and_load(
                &distributed_parameter_allocation_plan,
                &tensor_index,
                |device_id| device_for(device_id),
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to allocate distributed Vulkan parameter shards: {error}"
                    )),
                )
            })?,
        );

        let mut device_slices = Vec::with_capacity(device_slice_plans.len());
        for package_slice in device_slice_plans {
            let slice_device = device_for(&package_slice.device_id)?;
            let excluded_tensors =
                distributed_parameter_exclusion_plan.tensors_for_device(&package_slice.device_id);
            let package_slice = package_slice
                .materialize(slice_device, &tensor_index, &excluded_tensors)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            device_slices.push(Arc::new(package_slice));
        }

        let transducer_parameter_count = if Arc::ptr_eq(
            &input_transducer_parameter_buffers,
            &output_transducer_parameter_buffers,
        ) {
            input_transducer_parameter_buffers.plan.parameter_count
        } else {
            input_transducer_parameter_buffers.plan.parameter_count
                + output_transducer_parameter_buffers.plan.parameter_count
        };
        let transducer_parameter_bytes = if Arc::ptr_eq(
            &input_transducer_parameter_buffers,
            &output_transducer_parameter_buffers,
        ) {
            input_transducer_parameter_buffers.total_byte_capacity
        } else {
            input_transducer_parameter_buffers.total_byte_capacity
                + output_transducer_parameter_buffers.total_byte_capacity
        };

        let speculative_decoder_count = if mount_speculative_decoders {
            runtime_model.package.speculative_decoders.len()
        } else {
            0
        };
        let mut speculative_decoders = Vec::with_capacity(speculative_decoder_count);
        let speculative_decoder_context = VulkanResidentSpeculativeDecoderLoadContext {
            manifest_dir,
            runtime_model: &runtime_model,
            capacity,
            tensor_index: &tensor_index,
            target_output_parameters: &output_transducer_parameter_buffers,
            input_embedding_spec: &runtime_model.package.input_transducer.spec,
            input_embedding_spirv_words: &input_transducer_spirv_words,
        };
        for decoder in runtime_model
            .package
            .speculative_decoders
            .iter()
            .take(speculative_decoder_count)
        {
            speculative_decoders.push(
                VulkanResidentSpeculativeDecoderModelPackage::from_runtime_model(
                    output_device,
                    decoder,
                    &output_device_id,
                    &speculative_decoder_context,
                )?,
            );
        }

        Ok(Self {
            package_id,
            runtime_execution_identity,
            input_device_id,
            output_device_id,
            dynamic_state_capacity_activations: capacity,
            device_count: device_ids.len(),
            device_ids,
            hosted_component_count,
            transducer_parameter_count,
            transducer_parameter_bytes,
            input_transducer_parameter_buffers,
            output_transducer_parameter_buffers,
            input_transducer_spirv_words,
            input_transducer_batch_spirv_words,
            embedding_norm_spirv_words,
            embedding_norm_batch_spirv_words,
            embedding_norm_batch_lane_tile_width: runtime_model
                .package
                .output_transducer
                .embedding_norm_batch_lane_tile_width,
            tied_projection_spirv_words,
            tied_projection_batch_spirv_words,
            projection_batch_lane_tile_width: runtime_model
                .package
                .output_transducer
                .projection_batch_lane_tile_width,
            sampler_kernels,
            input_transducer_spec: runtime_model.package.input_transducer.spec.clone(),
            output_transducer_spec: runtime_model.package.output_transducer.spec.clone(),
            sampler_spec: runtime_model.package.sampler.spec.clone(),
            device_slices,
            speculative_decoders,
            distributed_execution_plan,
            distributed_activation_plan,
            distributed_parameter_allocation_plan,
            distributed_parameter_exclusion_plan,
            distributed_loaded_manifest,
            distributed_parameter_buffers,
        })
    }

    pub fn device_slice(&self, device_id: &str) -> Option<&VulkanResidentModelPackageDeviceSlice> {
        self.device_slices
            .iter()
            .find(|slice| slice.device_id == device_id)
            .map(Arc::as_ref)
    }

    pub fn distributed_execution_plan(&self) -> &VulkanDistributedExecutionPlan {
        &self.distributed_execution_plan
    }

    pub fn distributed_activation_plan(&self) -> &VulkanDistributedActivationBufferPlan {
        &self.distributed_activation_plan
    }

    pub fn distributed_parameter_allocation_plan(
        &self,
    ) -> &VulkanDistributedParameterAllocationPlan {
        &self.distributed_parameter_allocation_plan
    }

    pub fn distributed_parameter_exclusion_plan(&self) -> &VulkanDistributedParameterExclusionPlan {
        &self.distributed_parameter_exclusion_plan
    }

    pub fn create_stream_processor_for_devices(
        self: &Arc<Self>,
        device: &VulkanComputeDevice,
        random_seed: u32,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, None, |_| Ok(device))
    }

    pub fn create_stream_processor_for_bound_devices(
        self: &Arc<Self>,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, None, |device_id| {
            devices
                .get(device_id)
                .map(|device| device.as_ref())
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.to_string(),
                    },
                )
        })
    }

    pub fn create_stream_processor_inheriting_state_for_devices(
        self: &Arc<Self>,
        device: &VulkanComputeDevice,
        random_seed: u32,
        source: &VulkanResidentInProcessPlacedStreamProcessor,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, Some(source), |_| Ok(device))
    }

    pub fn create_stream_processor_inheriting_state_for_bound_devices(
        self: &Arc<Self>,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
        source: &VulkanResidentInProcessPlacedStreamProcessor,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, Some(source), |device_id| {
            devices
                .get(device_id)
                .map(|device| device.as_ref())
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.to_string(),
                    },
                )
        })
    }

    fn create_stream_processor_for_device_resolver<'a, F>(
        self: &Arc<Self>,
        random_seed: u32,
        source: Option<&VulkanResidentInProcessPlacedStreamProcessor>,
        device_for: F,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        if let Some(source) = source
            && source.model.package_id != self.package_id
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "cannot inherit stream state from package {:?} into package {:?}",
                    source.model.package_id, self.package_id
                )),
            ));
        }
        let distributed_activation_buffers = VulkanDistributedActivationBuffers::allocate(
            &self.distributed_activation_plan,
            |device_id| device_for(device_id),
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to allocate distributed Vulkan activation edges: {error}"
                )),
            )
        })?;
        let VulkanPlacedDeviceLinks {
            endpoint_overrides: shared_edge_endpoint_overrides,
            synchronizations: edge_synchronizations,
            stream_control_buffers,
        } = create_placed_device_links(&self.device_slices, &device_for)?;
        let mut devices = Vec::with_capacity(self.device_slices.len());
        for package_slice in &self.device_slices {
            let device = device_for(&package_slice.device_id)?;
            let activation_overrides = distributed_activation_buffers
                .activation_overrides_for_owner_device(&package_slice.device_id);
            let mounted = package_slice
                .create_mounted_stream_circuit_with_buffer_overrides(
                    device,
                    &activation_overrides,
                    shared_edge_endpoint_overrides
                        .get(&package_slice.device_id)
                        .map(Vec::as_slice)
                        .unwrap_or_default(),
                    stream_control_buffers
                        .get(&package_slice.device_id)
                        .cloned(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            mounted.buffers.zero_state_buffers().map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to zero stream state buffers for placed device {:?}: {error}",
                        package_slice.device_id
                    )),
                )
            })?;
            let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
            let mounted_bound = mounted
                .mounted_placed_bound_dispatch_plan(&reusable_manifest)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BoundDispatchPlan)?;
            let tick_plan =
                VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(&mounted_bound);
            let distributed_dispatch_groups = self
                .distributed_execution_plan
                .dispatch_groups
                .iter()
                .filter(|group| group.owner_device_id == package_slice.device_id)
                .map(|group| group.dispatch_indices())
                .collect::<Vec<_>>();
            let resident_execution_plan =
                VulkanMountedPlacedResidentStreamTickExecutionPlan::from_tick_plan_with_distributed_dispatch_groups(
                    device,
                    &mounted,
                    &mounted_bound,
                    package_slice.loaded_manifest(),
                    tick_plan,
                    &distributed_dispatch_groups,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
            devices.push(VulkanResidentInProcessPlacedStreamProcessorDevice {
                device_id: package_slice.device_id.clone(),
                hosted_component_count: package_slice.hosted_component_count,
                incoming_edge_count: package_slice.incoming_edge_count,
                outgoing_edge_count: package_slice.outgoing_edge_count,
                dispatch_count: mounted_bound.dispatches.len(),
                package_slice: package_slice.clone(),
                mounted,
                mounted_bound,
                resident_execution_plan,
            });
        }
        let mut distributed_dispatch_runners = VulkanDistributedDispatchRunners::create(
            &self.distributed_execution_plan,
            &self.distributed_parameter_buffers,
            &distributed_activation_buffers,
            &self.distributed_loaded_manifest,
            |device_id| device_for(device_id),
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to mount distributed Vulkan dispatches: {error}"
                )),
            )
        })?;
        let inherited = source
            .map(|source| inherit_matching_placed_stream_state(&devices, &source.device_slices))
            .transpose()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to inherit mounted stream state: {error}"
                    )),
                )
            })?
            .map(|(_, copied)| copied)
            .unwrap_or_default();
        apply_placed_clone_state_policies(&devices, &inherited).map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to initialize cloned stream state: {error}"
                )),
            )
        })?;
        let activation_tick_plans = devices
            .iter()
            .map(|slice| slice.resident_execution_plan.tick_plan.as_ref())
            .collect::<Vec<_>>();
        let activation_schedule =
            VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&activation_tick_plans)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Schedule)?;
        let input_slice = devices
            .iter()
            .find(|slice| slice.device_id == self.input_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "placed package {:?} has no mounted input device slice {:?}",
                        self.package_id, self.input_device_id
                    )),
                )
            })?;
        let output_slice = devices
            .iter()
            .find(|slice| slice.device_id == self.output_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "placed package {:?} has no mounted output device slice {:?}",
                        self.package_id, self.output_device_id
                    )),
                )
            })?;
        let input_device = device_for(&self.input_device_id)?;
        let output_device = device_for(&self.output_device_id)?;
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding(
                input_device,
                &input_slice.mounted,
                &self.input_transducer_parameter_buffers,
                &self.input_transducer_spirv_words,
                &self.input_transducer_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
                output_device,
                &output_slice.mounted,
                &self.output_transducer_parameter_buffers,
                &self.embedding_norm_spirv_words,
                &self.tied_projection_spirv_words,
                &self.output_transducer_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let local_dispatch_count = devices.iter().try_fold(0usize, |total, slice| {
            total
                .checked_add(slice.resident_execution_plan.dispatch_count)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "resident feedback local dispatch count overflowed".to_string(),
                    ))
                })
        })?;
        let sampler_dispatch_count = VulkanResidentSamplerRunner::feedback_dispatch_count_for_spec(
            &self.sampler_kernels,
            &self.sampler_spec,
        );
        let feedback_dispatch_capacity = local_dispatch_count
            .checked_add(distributed_dispatch_runners.shard_count)
            .and_then(|count| count.checked_add(1))
            .and_then(|count| count.checked_add(2))
            .and_then(|count| count.checked_add(sampler_dispatch_count))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "resident feedback dispatch capacity overflowed".to_string(),
                ))
            })?;
        let feedback_device_ids = devices
            .iter()
            .map(|slice| slice.device_id.clone())
            .collect::<Vec<_>>();
        let vocabulary_size = self.sampler_spec.logits_byte_capacity / size_of::<f32>();
        let mut feedback_control = VulkanResidentFeedbackControlPlane::new(
            &feedback_device_ids,
            &self.output_device_id,
            vocabulary_size,
            feedback_dispatch_capacity,
            &device_for,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let sampler = VulkanResidentSamplerRunner::from_output_transducer_with_spec_and_feedback_control(
            output_device,
            &output_slice.mounted,
            &output_transducer,
            &self.sampler_kernels,
            &self.sampler_spec,
            random_seed,
            feedback_control
                .sampler_bindings()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        for slice in &mut devices {
            let mut prefix_dispatches =
                SmallVec::<[&VulkanResidentKernelDispatch; 2]>::new();
            let mut suffix_dispatches =
                SmallVec::<[&VulkanResidentKernelDispatch; 5]>::new();
            if slice.device_id == self.input_device_id {
                prefix_dispatches.push(&input_transducer.resident_dispatch);
            }
            if slice.device_id == self.output_device_id {
                prefix_dispatches.extend(sampler.input_tracking_dispatches());
                suffix_dispatches.push(&output_transducer.embedding_norm_dispatch);
                suffix_dispatches.push(&output_transducer.tied_projection_dispatch);
                suffix_dispatches.extend(sampler.resident_dispatches());
                suffix_dispatches.push(sampler.feedback_control_dispatch());
            }
            let generation_tail_dispatch_count = (slice.device_id == self.output_device_id)
                .then_some(
                    2usize
                        .checked_add(sampler.resident_dispatches().len())
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "resident feedback generation tail count overflowed".to_string(),
                            ))
                        })?,
                );
            slice
                .resident_execution_plan
                .configure_feedback_indirect_dispatches(
                    &mut feedback_control,
                    &slice.device_id,
                    &prefix_dispatches,
                    &suffix_dispatches,
                    generation_tail_dispatch_count,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        distributed_dispatch_runners
            .configure_feedback_indirect_dispatches(&mut feedback_control, |device_id| {
                device_for(device_id)
            })
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to configure distributed feedback dispatches: {error}"
                    )),
                )
            })?;
        feedback_control
            .finish_registration()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let output_synchronization =
            VulkanResidentPlacedOutputTimelineSynchronization::new(output_device)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let resident_feedback_loop = VulkanResidentInProcessPlacedFeedbackLoop::new_if_supported(
            self,
            &devices,
            &activation_schedule,
            VulkanResidentPlacedFeedbackMount {
                input_transducer: &input_transducer,
                output_transducer: &output_transducer,
                sampler: &sampler,
                control: feedback_control,
            },
            &device_for,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut speculative_decoders = Vec::with_capacity(self.speculative_decoders.len());
        for decoder in &self.speculative_decoders {
            let draft_device = device_for(&decoder.device_id)?;
            speculative_decoders.push(VulkanResidentSpeculativeDecoderProcessor::from_model(
                draft_device,
                decoder,
                output_transducer.normalized_frame_buffer(),
                &self.output_transducer_parameter_buffers,
                &self.sampler_kernels,
                &self.sampler_spec,
                random_seed,
            )?);
        }
        let execution_quantum_calibrators = devices
            .iter()
            .map(|slice| {
                (
                    slice.device_id.clone(),
                    Rc::new(RefCell::new(RuntimeExecutionQuantumCalibrator::default())),
                )
            })
            .collect();
        Ok(VulkanResidentInProcessPlacedStreamProcessor {
            model: self.clone(),
            distributed_dispatch_runners,
            _distributed_activation_buffers: distributed_activation_buffers,
            edge_synchronizations,
            input_transducer,
            output_transducer,
            sampler,
            output_synchronization,
            resident_feedback_loop,
            activation_schedule,
            device_slices: devices,
            execution_quantum_calibrators,
            speculative_decoders,
            verification_state_transactions: RefCell::new(None),
            component_batch_execution: RefCell::new(None),
            verification_input_embedding: RefCell::new(None),
            temporal_block_execution: RefCell::new(None),
            batched_output_projection: RefCell::new(None),
        })
    }
}
