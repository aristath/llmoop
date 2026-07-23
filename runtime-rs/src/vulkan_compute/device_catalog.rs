impl VulkanResidentKernelSequence {
    pub fn has_recorded_commands(&self) -> bool {
        self.recorded_input_copies.borrow().is_some()
            && self.recorded_steps.borrow().is_some()
            && self.recorded_snapshot_copies.borrow().is_some()
    }
}

impl VulkanComputeDeviceCatalog {
    pub fn discover() -> Result<Self, VulkanError> {
        unsafe {
            let entry = Entry::load()
                .map_err(|error| VulkanError(format!("failed to load Vulkan: {error}")))?;
            let instance = create_nerve_vulkan_instance(&entry)?;
            let physical_devices = instance.enumerate_physical_devices().map_err(|error| {
                instance.destroy_instance(None);
                VulkanError(format!("failed to enumerate Vulkan devices: {error:?}"))
            })?;
            let selected_index = select_compute_device_index(&instance, &physical_devices);
            let available_devices = physical_devices
                .iter()
                .enumerate()
                .filter_map(|(physical_device_index, physical_device)| {
                    inspect_compute_device(
                        &instance,
                        physical_device_index,
                        *physical_device,
                        Some(physical_device_index) == selected_index,
                    )
                })
                .collect::<Vec<_>>();
            Ok(Self {
                context: Arc::new(VulkanInstanceContext {
                    _entry: entry,
                    instance,
                }),
                physical_devices,
                available_devices,
            })
        }
    }

    pub fn available_compute_devices(&self) -> &[VulkanComputeDeviceInfo] {
        &self.available_devices
    }

    pub fn open_device_uuid(
        &self,
        device_uuid: [u8; vk::UUID_SIZE],
    ) -> Result<VulkanComputeDevice, VulkanError> {
        self.open_device(None, Some(device_uuid))
    }

    pub fn open_physical_device_index(
        &self,
        physical_device_index: usize,
    ) -> Result<VulkanComputeDevice, VulkanError> {
        self.open_device(Some(physical_device_index), None)
    }

    fn open_device(
        &self,
        requested_physical_device_index: Option<usize>,
        requested_device_uuid: Option<[u8; vk::UUID_SIZE]>,
    ) -> Result<VulkanComputeDevice, VulkanError> {
        unsafe {
            let instance = &self.context.instance;
            let (physical_device, queue_family_index, device_name) =
                if let Some(device_uuid) = requested_device_uuid {
                    select_compute_device_by_uuid(instance, &self.physical_devices, device_uuid)?
                } else if let Some(physical_device_index) = requested_physical_device_index {
                    select_compute_device_by_index(
                        instance,
                        &self.physical_devices,
                        physical_device_index,
                    )?
                } else {
                    select_compute_device(instance, &self.physical_devices).ok_or_else(|| {
                        VulkanError("no Vulkan device with a compute queue was found".to_string())
                    })?
                };

            let queue_priorities = [1.0_f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)];
            let shader_float8_extension_supported = physical_device_supports_extension(
                instance,
                physical_device,
                VK_EXT_SHADER_FLOAT8_NAME,
            )?;
            let shader_float8_support = if shader_float8_extension_supported {
                physical_device_shader_float8_support(instance, physical_device)
            } else {
                VulkanShaderFloat8Support::default()
            };
            let cooperative_matrix_extension_supported = physical_device_supports_extension(
                instance,
                physical_device,
                ash::khr::cooperative_matrix::NAME,
            )?;
            let cooperative_matrix_supported = cooperative_matrix_extension_supported
                && physical_device_supports_cooperative_matrix(instance, physical_device);
            let shader_bfloat16_extension_supported = physical_device_supports_extension(
                instance,
                physical_device,
                VK_KHR_SHADER_BFLOAT16_NAME,
            )?;
            let shader_bfloat16_support = if shader_bfloat16_extension_supported {
                physical_device_shader_bfloat16_support(instance, physical_device)
            } else {
                VulkanShaderBfloat16Support::default()
            };
            let mixed_float_dot_product_extension_supported = physical_device_supports_extension(
                instance,
                physical_device,
                VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME,
            )?;
            let mixed_float_dot_product_support = if mixed_float_dot_product_extension_supported {
                physical_device_shader_mixed_float_dot_product_support(instance, physical_device)
            } else {
                VulkanShaderMixedFloatDotProductSupport::default()
            };
            let cooperative_bfloat16_features_supported = cooperative_matrix_supported
                && shader_bfloat16_support.shader_bfloat16_type
                && shader_bfloat16_support.shader_bfloat16_cooperative_matrix;
            let cooperative_bfloat16_shapes = if cooperative_bfloat16_features_supported {
                physical_device_cooperative_bfloat16_shapes(
                    &self.context._entry,
                    instance,
                    physical_device,
                )?
            } else {
                BTreeSet::new()
            };
            let shared_host_memory_alignment =
                if physical_device_supports_extension(
                    instance,
                    physical_device,
                    ash::ext::external_memory_host::NAME,
                )? && physical_device_supports_shared_host_buffer(instance, physical_device)
                {
                    Some(physical_device_shared_host_memory_alignment(
                        instance,
                        physical_device,
                    )?)
                } else {
                    None
                };
            let opaque_fd_timeline_semaphore_supported = physical_device_supports_extension(
                instance,
                physical_device,
                ash::khr::external_semaphore_fd::NAME,
            )?
                && physical_device_supports_opaque_fd_timeline_semaphore(instance, physical_device);
            let (timeline_semaphore_supported, synchronization2_supported) =
                physical_device_supports_modern_submission(instance, physical_device);
            if !timeline_semaphore_supported || !synchronization2_supported {
                return Err(VulkanError(format!(
                    "Vulkan device {device_name:?} does not support the required timeline-semaphore and synchronization2 execution contract"
                )));
            }
            // Logical-device features cannot be added later. Enable every supported
            // feature in the runtime's SPIR-V contract so this device can safely
            // host different compiled pedal packages without being recreated.
            let mut enabled_shader_features =
                physical_device_standard_shader_features(instance, physical_device);
            if shader_float8_support.shader_float8 {
                enabled_shader_features.insert(VulkanShaderFeature::ShaderFloat8);
            }
            if shader_float8_support.shader_float8_cooperative_matrix
                && cooperative_matrix_supported
            {
                enabled_shader_features.insert(VulkanShaderFeature::ShaderFloat8CooperativeMatrix);
            }
            if cooperative_matrix_supported {
                enabled_shader_features.insert(VulkanShaderFeature::CooperativeMatrix);
            }
            if shader_bfloat16_support.shader_bfloat16_type {
                enabled_shader_features.insert(VulkanShaderFeature::ShaderBfloat16Type);
            }
            if shader_bfloat16_support.shader_bfloat16_dot_product {
                enabled_shader_features.insert(VulkanShaderFeature::ShaderBfloat16DotProduct);
            }
            if shader_bfloat16_support.shader_bfloat16_cooperative_matrix
                && cooperative_matrix_supported
            {
                enabled_shader_features
                    .insert(VulkanShaderFeature::ShaderBfloat16CooperativeMatrix);
            }
            if mixed_float_dot_product_support.shader_float8_acc_float32 {
                enabled_shader_features
                    .insert(VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32);
            }
            let enabled_core_features = vk::PhysicalDeviceFeatures {
                shader_float64: bool32(
                    enabled_shader_features.contains(&VulkanShaderFeature::ShaderFloat64),
                ),
                shader_int16: bool32(
                    enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt16),
                ),
                shader_int64: bool32(
                    enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt64),
                ),
                ..Default::default()
            };
            let mut shader_float16_int8_features =
                vk::PhysicalDeviceShaderFloat16Int8Features::default()
                    .shader_float16(
                        enabled_shader_features.contains(&VulkanShaderFeature::ShaderFloat16),
                    )
                    .shader_int8(
                        enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt8),
                    );
            let mut storage16_features = vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_buffer16_bit_access(
                    enabled_shader_features
                        .contains(&VulkanShaderFeature::StorageBuffer16BitAccess),
                )
                .uniform_and_storage_buffer16_bit_access(
                    enabled_shader_features
                        .contains(&VulkanShaderFeature::UniformAndStorageBuffer16BitAccess),
                )
                .storage_push_constant16(
                    enabled_shader_features.contains(&VulkanShaderFeature::StoragePushConstant16),
                )
                .storage_input_output16(
                    enabled_shader_features.contains(&VulkanShaderFeature::StorageInputOutput16),
                );
            let mut storage8_features = vk::PhysicalDevice8BitStorageFeatures::default()
                .storage_buffer8_bit_access(
                    enabled_shader_features.contains(&VulkanShaderFeature::StorageBuffer8BitAccess),
                )
                .uniform_and_storage_buffer8_bit_access(
                    enabled_shader_features
                        .contains(&VulkanShaderFeature::UniformAndStorageBuffer8BitAccess),
                )
                .storage_push_constant8(
                    enabled_shader_features.contains(&VulkanShaderFeature::StoragePushConstant8),
                );
            let mut integer_dot_product_features =
                vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
                    .shader_integer_dot_product(
                        enabled_shader_features
                            .contains(&VulkanShaderFeature::ShaderIntegerDotProduct),
                    );
            let mut vulkan_memory_model_features =
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
                    .vulkan_memory_model(
                        enabled_shader_features.contains(&VulkanShaderFeature::VulkanMemoryModel),
                    )
                    .vulkan_memory_model_device_scope(
                        enabled_shader_features
                            .contains(&VulkanShaderFeature::VulkanMemoryModelDeviceScope),
                    );
            let mut shader_float8_features =
                VulkanPhysicalDeviceShaderFloat8FeaturesExt::disabled();
            let mut shader_bfloat16_features =
                VulkanPhysicalDeviceShaderBfloat16FeaturesKhr::disabled();
            let mut mixed_float_dot_product_features =
                VulkanPhysicalDeviceShaderMixedFloatDotProductFeaturesValve::disabled();
            let mut cooperative_matrix_features =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut timeline_semaphore_features =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let mut synchronization2_features =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
            let mut extension_names = Vec::new();
            let mut enabled_device_extensions = BTreeSet::new();
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_features(&enabled_core_features)
                .push_next(&mut timeline_semaphore_features)
                .push_next(&mut synchronization2_features)
                .push_next(&mut shader_float16_int8_features)
                .push_next(&mut storage16_features)
                .push_next(&mut storage8_features)
                .push_next(&mut integer_dot_product_features)
                .push_next(&mut vulkan_memory_model_features);
            if shader_float8_support.shader_float8
                || shader_float8_support.shader_float8_cooperative_matrix
            {
                shader_float8_features.shader_float8 = bool32(shader_float8_support.shader_float8);
                shader_float8_features.shader_float8_cooperative_matrix = bool32(
                    shader_float8_support.shader_float8_cooperative_matrix
                        && cooperative_matrix_supported,
                );
                extension_names.push(VK_EXT_SHADER_FLOAT8_NAME.as_ptr());
                enabled_device_extensions
                    .insert(VK_EXT_SHADER_FLOAT8_NAME.to_string_lossy().into_owned());
            }
            if cooperative_matrix_supported {
                cooperative_matrix_features.cooperative_matrix = vk::TRUE;
                extension_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::khr::cooperative_matrix::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if shader_bfloat16_support.shader_bfloat16_type
                || shader_bfloat16_support.shader_bfloat16_dot_product
                || shader_bfloat16_support.shader_bfloat16_cooperative_matrix
            {
                shader_bfloat16_features.shader_bfloat16_type =
                    bool32(shader_bfloat16_support.shader_bfloat16_type);
                shader_bfloat16_features.shader_bfloat16_dot_product =
                    bool32(shader_bfloat16_support.shader_bfloat16_dot_product);
                shader_bfloat16_features.shader_bfloat16_cooperative_matrix = bool32(
                    shader_bfloat16_support.shader_bfloat16_cooperative_matrix
                        && cooperative_matrix_supported,
                );
                extension_names.push(VK_KHR_SHADER_BFLOAT16_NAME.as_ptr());
                enabled_device_extensions
                    .insert(VK_KHR_SHADER_BFLOAT16_NAME.to_string_lossy().into_owned());
            }
            if mixed_float_dot_product_support.shader_float8_acc_float32 {
                mixed_float_dot_product_features.shader_float8_acc_float32 = vk::TRUE;
                extension_names.push(VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME.as_ptr());
                enabled_device_extensions.insert(
                    VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if shared_host_memory_alignment.is_some() {
                extension_names.push(ash::ext::external_memory_host::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::ext::external_memory_host::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if opaque_fd_timeline_semaphore_supported {
                extension_names.push(ash::khr::external_semaphore_fd::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::khr::external_semaphore_fd::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if shader_float8_support.shader_float8
                || shader_float8_support.shader_float8_cooperative_matrix
            {
                shader_float8_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&shader_float8_features).cast();
            }
            if shader_bfloat16_support.shader_bfloat16_type
                || shader_bfloat16_support.shader_bfloat16_dot_product
                || shader_bfloat16_support.shader_bfloat16_cooperative_matrix
            {
                shader_bfloat16_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&shader_bfloat16_features).cast();
            }
            if mixed_float_dot_product_support.shader_float8_acc_float32 {
                mixed_float_dot_product_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&mixed_float_dot_product_features).cast();
            }
            if cooperative_matrix_supported {
                cooperative_matrix_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&cooperative_matrix_features).cast();
            }
            device_info = device_info.enabled_extension_names(&extension_names);
            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create Vulkan device: {error:?}"))
                })?;
            let queue = device.get_device_queue(queue_family_index, 0);
            let physical_device_properties =
                instance.get_physical_device_properties(physical_device);
            let limits = physical_device_properties.limits;
            let min_storage_buffer_offset_alignment =
                usize::try_from(limits.min_storage_buffer_offset_alignment).map_err(|_| {
                    VulkanError("Vulkan storage-buffer offset alignment exceeds usize".to_string())
                })?;
            let subgroup_support = physical_device_subgroup_support(instance, physical_device);
            let subgroup_size = subgroup_support.subgroup_size;

            Ok(VulkanComputeDevice {
                context: Arc::clone(&self.context),
                physical_device,
                device,
                queue_family_index,
                queue,
                device_name,
                enabled_device_extensions,
                enabled_shader_features,
                shared_host_memory_alignment,
                opaque_fd_timeline_semaphore_supported,
                cooperative_bfloat16_shapes,
                subgroup_size,
                subgroup_supported_stages: subgroup_support.supported_stages,
                subgroup_supported_operations: subgroup_support.supported_operations,
                max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
                max_compute_work_group_size_x: limits.max_compute_work_group_size[0],
                min_storage_buffer_offset_alignment,
                timestamp_period_ns: limits.timestamp_period,
                generic_storage_pipelines: RefCell::new(HashMap::new()),
                immediate_kernel_sequence: RefCell::new(None),
            })
        }
    }
}
