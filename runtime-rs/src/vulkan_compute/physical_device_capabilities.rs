impl Drop for VulkanComputeDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.immediate_kernel_sequence.get_mut().take();
            for (_, pipeline) in self.generic_storage_pipelines.get_mut().drain() {
                self.device.destroy_pipeline(pipeline.pipeline, None);
                self.device
                    .destroy_shader_module(pipeline.shader_module, None);
                self.device
                    .destroy_pipeline_layout(pipeline.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            }
            self.device.destroy_device(None);
        }
    }
}

unsafe fn select_compute_device(
    instance: &ash::Instance,
    physical_devices: &[vk::PhysicalDevice],
) -> Option<(vk::PhysicalDevice, u32, String)> {
    let selected_index = unsafe { select_compute_device_index(instance, physical_devices)? };
    let physical_device = physical_devices[selected_index];
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let queue_family_index = unsafe { compute_queue_family_indices(instance, physical_device) }
        .into_iter()
        .next()?;
    Some((physical_device, queue_family_index, device_name))
}

unsafe fn select_compute_device_by_index(
    instance: &ash::Instance,
    physical_devices: &[vk::PhysicalDevice],
    physical_device_index: usize,
) -> Result<(vk::PhysicalDevice, u32, String), VulkanError> {
    let physical_device = *physical_devices.get(physical_device_index).ok_or_else(|| {
        VulkanError(format!(
            "Vulkan physical device index {physical_device_index} was not found"
        ))
    })?;
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let queue_family_index = unsafe { compute_queue_family_indices(instance, physical_device) }
        .into_iter()
        .next()
        .ok_or_else(|| {
            VulkanError(format!(
                "Vulkan physical device index {physical_device_index} ({device_name}) has no compute queue"
            ))
        })?;
    Ok((physical_device, queue_family_index, device_name))
}

unsafe fn select_compute_device_by_uuid(
    instance: &ash::Instance,
    physical_devices: &[vk::PhysicalDevice],
    requested_device_uuid: [u8; vk::UUID_SIZE],
) -> Result<(vk::PhysicalDevice, u32, String), VulkanError> {
    for physical_device in physical_devices {
        if unsafe { physical_device_uuid(instance, *physical_device) } == requested_device_uuid {
            let properties = unsafe { instance.get_physical_device_properties(*physical_device) };
            let device_name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let queue_family_index =
                unsafe { compute_queue_family_indices(instance, *physical_device) }
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        VulkanError(format!(
                            "Vulkan device UUID {} ({device_name}) has no compute queue",
                            format_device_uuid(&requested_device_uuid)
                        ))
                    })?;
            return Ok((*physical_device, queue_family_index, device_name));
        }
    }
    Err(VulkanError(format!(
        "Vulkan device UUID {} was not found",
        format_device_uuid(&requested_device_uuid)
    )))
}

unsafe fn physical_device_uuid(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> [u8; vk::UUID_SIZE] {
    let mut id_properties = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id_properties);
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
    id_properties.device_uuid
}

fn format_device_uuid(device_uuid: &[u8; vk::UUID_SIZE]) -> String {
    device_uuid
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

unsafe fn select_compute_device_index(
    instance: &ash::Instance,
    physical_devices: &[vk::PhysicalDevice],
) -> Option<usize> {
    let mut fallback = None;
    for (physical_device_index, physical_device) in physical_devices.iter().enumerate() {
        let properties = unsafe { instance.get_physical_device_properties(*physical_device) };
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
        if !preferred_compute_queue_family_indices(&queue_families).is_empty() {
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            {
                return Some(physical_device_index);
            }
            fallback.get_or_insert(physical_device_index);
        }
    }
    fallback
}

fn physical_device_supports_extension(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension_name: &CStr,
) -> Result<bool, VulkanError> {
    let properties = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|error| {
                VulkanError(format!(
                    "failed to enumerate Vulkan device extensions: {error:?}"
                ))
            })?
    };
    Ok(properties.iter().any(|property| unsafe {
        CStr::from_ptr(property.extension_name.as_ptr()) == extension_name
    }))
}

fn resident_buffer_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
}

fn physical_device_supports_shared_host_buffer(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(resident_buffer_usage())
        .handle_type(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE);
    let mut properties = vk::ExternalBufferProperties::default();
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            physical_device,
            &info,
            &mut properties,
        );
    }
    properties
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        && properties
            .external_memory_properties
            .compatible_handle_types
            .contains(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE)
}

fn physical_device_shared_host_memory_alignment(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<usize, VulkanError> {
    let mut external_host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut external_host);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties);
    }
    let alignment = usize::try_from(external_host.min_imported_host_pointer_alignment)
        .map_err(|_| VulkanError("shared host-memory alignment exceeds usize".to_string()))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(VulkanError(format!(
            "Vulkan device reported invalid shared host-memory alignment {alignment}"
        )));
    }
    Ok(alignment)
}

fn physical_device_supports_opaque_fd_timeline_semaphore(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE)
        .push_next(&mut timeline_info);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    unsafe {
        instance.get_physical_device_external_semaphore_properties(
            physical_device,
            &info,
            &mut properties,
        );
    }
    properties.external_semaphore_features.contains(
        vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
            | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE,
    ) && properties
        .compatible_handle_types
        .contains(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE)
}

fn vulkan_shader_feature_for_spirv_capability(capability: u32) -> Option<VulkanShaderFeature> {
    Some(match capability {
        9 => VulkanShaderFeature::ShaderFloat16,
        10 => VulkanShaderFeature::ShaderFloat64,
        11 => VulkanShaderFeature::ShaderInt64,
        22 => VulkanShaderFeature::ShaderInt16,
        39 => VulkanShaderFeature::ShaderInt8,
        4212 => VulkanShaderFeature::ShaderFloat8,
        4213 => VulkanShaderFeature::ShaderFloat8CooperativeMatrix,
        4433 => VulkanShaderFeature::StorageBuffer16BitAccess,
        4434 => VulkanShaderFeature::UniformAndStorageBuffer16BitAccess,
        4435 => VulkanShaderFeature::StoragePushConstant16,
        4436 => VulkanShaderFeature::StorageInputOutput16,
        4448 => VulkanShaderFeature::StorageBuffer8BitAccess,
        4449 => VulkanShaderFeature::UniformAndStorageBuffer8BitAccess,
        4450 => VulkanShaderFeature::StoragePushConstant8,
        5116 => VulkanShaderFeature::ShaderBfloat16Type,
        5117 => VulkanShaderFeature::ShaderBfloat16DotProduct,
        5118 => VulkanShaderFeature::ShaderBfloat16CooperativeMatrix,
        6018 => VulkanShaderFeature::ShaderIntegerDotProduct,
        6019 => VulkanShaderFeature::ShaderIntegerDotProduct,
        6915 => VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32,
        5345 => VulkanShaderFeature::VulkanMemoryModel,
        5346 => VulkanShaderFeature::VulkanMemoryModelDeviceScope,
        6022 => VulkanShaderFeature::CooperativeMatrix,
        _ => return None,
    })
}

fn vulkan_subgroup_operation_for_spirv_capability(
    capability: u32,
) -> Option<VulkanSubgroupOperation> {
    Some(match capability {
        61 => VulkanSubgroupOperation::Basic,
        62 => VulkanSubgroupOperation::Vote,
        63 => VulkanSubgroupOperation::Arithmetic,
        64 => VulkanSubgroupOperation::Ballot,
        65 => VulkanSubgroupOperation::Shuffle,
        66 => VulkanSubgroupOperation::ShuffleRelative,
        67 => VulkanSubgroupOperation::Clustered,
        68 => VulkanSubgroupOperation::Quad,
        _ => return None,
    })
}

pub fn vulkan_spirv_requirements(
    spirv_words: &[u32],
) -> Result<VulkanSpirvRequirements, VulkanError> {
    if spirv_words.len() < 5 || spirv_words[0] != SPIRV_MAGIC {
        return Err(VulkanError(
            "shader artifact is not a valid little-endian SPIR-V module".to_string(),
        ));
    }

    let mut requirements = VulkanSpirvRequirements::default();
    let mut cursor = 5usize;
    while cursor < spirv_words.len() {
        let instruction = spirv_words[cursor];
        let word_count = (instruction >> 16) as usize;
        let opcode = (instruction & 0xffff) as u16;
        if word_count == 0 || cursor + word_count > spirv_words.len() {
            return Err(VulkanError(format!(
                "shader artifact has a malformed SPIR-V instruction at word {cursor}"
            )));
        }
        match opcode {
            SPIRV_OP_CAPABILITY => {
                if word_count != 2 {
                    return Err(VulkanError(format!(
                        "shader artifact has a malformed OpCapability at word {cursor}"
                    )));
                }
                let capability = spirv_words[cursor + 1];
                if matches!(capability, 0 | 1) {
                    // Matrix and Shader are baseline compute-shader capabilities.
                } else if let Some(feature) = vulkan_shader_feature_for_spirv_capability(capability)
                {
                    requirements.shader_features.insert(feature);
                } else if let Some(operation) =
                    vulkan_subgroup_operation_for_spirv_capability(capability)
                {
                    requirements.uses_subgroups = true;
                    requirements.subgroup_operations.insert(operation);
                } else {
                    return Err(VulkanError(format!(
                        "shader artifact declares SPIR-V capability {capability}, but the runtime has no device contract for it"
                    )));
                }
                requirements.vulkan_memory_model_capability |= capability == 5345;
                requirements.vulkan_memory_model_device_scope_capability |= capability == 5346;
            }
            SPIRV_OP_MEMORY_MODEL => {
                if word_count != 3 || requirements.memory_model.is_some() {
                    return Err(VulkanError(format!(
                        "shader artifact has an invalid OpMemoryModel at word {cursor}"
                    )));
                }
                requirements.memory_model = Some(spirv_words[cursor + 2]);
            }
            _ => {}
        }
        cursor += word_count;
    }

    let memory_model = requirements.memory_model.ok_or_else(|| {
        VulkanError("shader artifact does not declare an SPIR-V memory model".to_string())
    })?;
    if requirements.vulkan_memory_model_capability != (memory_model == SPIRV_MEMORY_MODEL_VULKAN) {
        return Err(VulkanError(
            "shader artifact has an inconsistent Vulkan SPIR-V memory-model contract".to_string(),
        ));
    }
    if requirements.vulkan_memory_model_device_scope_capability
        && !requirements.vulkan_memory_model_capability
    {
        return Err(VulkanError(
            "shader artifact declares VulkanMemoryModelDeviceScope without VulkanMemoryModel"
                .to_string(),
        ));
    }
    Ok(requirements)
}

fn validate_spirv_device_contract(
    spirv_words: &[u32],
    enabled_shader_features: &BTreeSet<VulkanShaderFeature>,
    subgroup_supported_stages: vk::ShaderStageFlags,
    subgroup_supported_operations: vk::SubgroupFeatureFlags,
) -> Result<(), VulkanError> {
    let requirements = vulkan_spirv_requirements(spirv_words)?;
    let missing_features = requirements
        .shader_features
        .difference(enabled_shader_features)
        .copied()
        .map(VulkanShaderFeature::label)
        .collect::<Vec<_>>();
    if !missing_features.is_empty() {
        return Err(VulkanError(format!(
            "shader artifact requires Vulkan features that were not enabled on the logical device: {}",
            missing_features.join(", ")
        )));
    }
    if requirements.uses_subgroups
        && !subgroup_supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
    {
        return Err(VulkanError(
            "shader artifact uses subgroup operations, but the device does not support them in compute shaders"
                .to_string(),
        ));
    }
    let missing_subgroup_operations = requirements
        .subgroup_operations
        .iter()
        .copied()
        .filter(|operation| !subgroup_supported_operations.contains(operation.flag()))
        .map(VulkanSubgroupOperation::label)
        .collect::<Vec<_>>();
    if !missing_subgroup_operations.is_empty() {
        return Err(VulkanError(format!(
            "shader artifact requires unsupported Vulkan subgroup operations: {}",
            missing_subgroup_operations.join(", ")
        )));
    }
    Ok(())
}

fn physical_device_supports_modern_submission(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> (bool, bool) {
    let mut timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut timeline_semaphore)
        .push_next(&mut synchronization2);
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }
    (
        timeline_semaphore.timeline_semaphore == vk::TRUE,
        synchronization2.synchronization2 == vk::TRUE,
    )
}

fn bool32(value: bool) -> vk::Bool32 {
    if value { vk::TRUE } else { vk::FALSE }
}

fn physical_device_standard_shader_features(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> BTreeSet<VulkanShaderFeature> {
    let core = unsafe { instance.get_physical_device_features(physical_device) };
    let mut float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut integer_dot_product = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
    let mut memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut float16_int8)
        .push_next(&mut storage16)
        .push_next(&mut storage8)
        .push_next(&mut integer_dot_product)
        .push_next(&mut memory_model);
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }

    let mut supported = BTreeSet::new();
    let mut insert = |available: vk::Bool32, feature| {
        if available == vk::TRUE {
            supported.insert(feature);
        }
    };
    insert(
        float16_int8.shader_float16,
        VulkanShaderFeature::ShaderFloat16,
    );
    insert(float16_int8.shader_int8, VulkanShaderFeature::ShaderInt8);
    insert(
        integer_dot_product.shader_integer_dot_product,
        VulkanShaderFeature::ShaderIntegerDotProduct,
    );
    insert(core.shader_float64, VulkanShaderFeature::ShaderFloat64);
    insert(core.shader_int16, VulkanShaderFeature::ShaderInt16);
    insert(core.shader_int64, VulkanShaderFeature::ShaderInt64);
    insert(
        storage16.storage_buffer16_bit_access,
        VulkanShaderFeature::StorageBuffer16BitAccess,
    );
    insert(
        storage16.uniform_and_storage_buffer16_bit_access,
        VulkanShaderFeature::UniformAndStorageBuffer16BitAccess,
    );
    insert(
        storage16.storage_push_constant16,
        VulkanShaderFeature::StoragePushConstant16,
    );
    insert(
        storage16.storage_input_output16,
        VulkanShaderFeature::StorageInputOutput16,
    );
    insert(
        storage8.storage_buffer8_bit_access,
        VulkanShaderFeature::StorageBuffer8BitAccess,
    );
    insert(
        storage8.uniform_and_storage_buffer8_bit_access,
        VulkanShaderFeature::UniformAndStorageBuffer8BitAccess,
    );
    insert(
        storage8.storage_push_constant8,
        VulkanShaderFeature::StoragePushConstant8,
    );
    insert(
        memory_model.vulkan_memory_model,
        VulkanShaderFeature::VulkanMemoryModel,
    );
    if memory_model.vulkan_memory_model == vk::TRUE {
        insert(
            memory_model.vulkan_memory_model_device_scope,
            VulkanShaderFeature::VulkanMemoryModelDeviceScope,
        );
    }
    supported
}

fn physical_device_supported_shader_features(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<BTreeSet<VulkanShaderFeature>, VulkanError> {
    let mut supported = physical_device_standard_shader_features(instance, physical_device);
    let cooperative_matrix_supported = physical_device_supports_extension(
        instance,
        physical_device,
        ash::khr::cooperative_matrix::NAME,
    )? && physical_device_supports_cooperative_matrix(instance, physical_device);
    if cooperative_matrix_supported {
        supported.insert(VulkanShaderFeature::CooperativeMatrix);
    }

    if physical_device_supports_extension(
        instance,
        physical_device,
        VK_EXT_SHADER_FLOAT8_NAME,
    )? {
        let float8 = physical_device_shader_float8_support(instance, physical_device);
        if float8.shader_float8 {
            supported.insert(VulkanShaderFeature::ShaderFloat8);
        }
        if float8.shader_float8_cooperative_matrix && cooperative_matrix_supported {
            supported.insert(VulkanShaderFeature::ShaderFloat8CooperativeMatrix);
        }
    }

    if physical_device_supports_extension(
        instance,
        physical_device,
        VK_KHR_SHADER_BFLOAT16_NAME,
    )? {
        let bfloat16 = physical_device_shader_bfloat16_support(instance, physical_device);
        if bfloat16.shader_bfloat16_type {
            supported.insert(VulkanShaderFeature::ShaderBfloat16Type);
        }
        if bfloat16.shader_bfloat16_dot_product {
            supported.insert(VulkanShaderFeature::ShaderBfloat16DotProduct);
        }
        if bfloat16.shader_bfloat16_cooperative_matrix && cooperative_matrix_supported {
            supported.insert(VulkanShaderFeature::ShaderBfloat16CooperativeMatrix);
        }
    }

    if physical_device_supports_extension(
        instance,
        physical_device,
        VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME,
    )? {
        let mixed =
            physical_device_shader_mixed_float_dot_product_support(instance, physical_device);
        if mixed.shader_float8_acc_float32 {
            supported.insert(
                VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32,
            );
        }
    }
    Ok(supported)
}

fn subgroup_operations(
    flags: vk::SubgroupFeatureFlags,
) -> BTreeSet<VulkanSubgroupOperation> {
    [
        VulkanSubgroupOperation::Basic,
        VulkanSubgroupOperation::Vote,
        VulkanSubgroupOperation::Arithmetic,
        VulkanSubgroupOperation::Ballot,
        VulkanSubgroupOperation::Shuffle,
        VulkanSubgroupOperation::ShuffleRelative,
        VulkanSubgroupOperation::Clustered,
        VulkanSubgroupOperation::Quad,
    ]
    .into_iter()
    .filter(|operation| flags.contains(operation.flag()))
    .collect()
}

fn physical_device_shader_float8_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> VulkanShaderFloat8Support {
    let mut shader_float8 = VulkanPhysicalDeviceShaderFloat8FeaturesExt::disabled();
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: std::ptr::from_mut(&mut shader_float8).cast(),
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }
    VulkanShaderFloat8Support {
        shader_float8: shader_float8.shader_float8 == vk::TRUE,
        shader_float8_cooperative_matrix: shader_float8.shader_float8_cooperative_matrix
            == vk::TRUE,
    }
}

fn physical_device_shader_mixed_float_dot_product_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> VulkanShaderMixedFloatDotProductSupport {
    let mut mixed_float_dot_product =
        VulkanPhysicalDeviceShaderMixedFloatDotProductFeaturesValve::disabled();
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: std::ptr::from_mut(&mut mixed_float_dot_product).cast(),
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }
    VulkanShaderMixedFloatDotProductSupport {
        shader_float8_acc_float32: mixed_float_dot_product.shader_float8_acc_float32 == vk::TRUE,
    }
}

fn physical_device_supports_cooperative_matrix(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let mut cooperative_matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: std::ptr::from_mut(&mut cooperative_matrix).cast(),
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }
    cooperative_matrix.cooperative_matrix == vk::TRUE
}

fn physical_device_shader_bfloat16_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> VulkanShaderBfloat16Support {
    let mut shader_bfloat16 = VulkanPhysicalDeviceShaderBfloat16FeaturesKhr::disabled();
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: std::ptr::from_mut(&mut shader_bfloat16).cast(),
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }
    VulkanShaderBfloat16Support {
        shader_bfloat16_type: shader_bfloat16.shader_bfloat16_type == vk::TRUE,
        shader_bfloat16_dot_product: shader_bfloat16.shader_bfloat16_dot_product == vk::TRUE,
        shader_bfloat16_cooperative_matrix: shader_bfloat16.shader_bfloat16_cooperative_matrix
            == vk::TRUE,
    }
}

fn physical_device_cooperative_bfloat16_shapes(
    entry: &Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<BTreeSet<(u32, u32, u32)>, VulkanError> {
    let cooperative_matrix = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let properties = unsafe {
        cooperative_matrix
            .get_physical_device_cooperative_matrix_properties(physical_device)
            .map_err(|error| {
                VulkanError(format!(
                    "failed to query cooperative-matrix properties: {error:?}"
                ))
            })?
    };
    let bfloat16 = vk::ComponentTypeKHR::from_raw(VK_COMPONENT_TYPE_BFLOAT16_KHR);
    Ok(properties
        .into_iter()
        .filter(|property| {
            property.a_type == bfloat16
                && property.b_type == bfloat16
                && property.c_type == vk::ComponentTypeKHR::FLOAT32
                && property.result_type == vk::ComponentTypeKHR::FLOAT32
                && property.scope == vk::ScopeKHR::SUBGROUP
        })
        .map(|property| (property.m_size, property.n_size, property.k_size))
        .collect())
}

fn physical_device_subgroup_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceSubgroupProperties<'static> {
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2 {
        p_next: std::ptr::from_mut(&mut subgroup).cast(),
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties);
    }
    subgroup
}

unsafe fn inspect_compute_device(
    instance: &ash::Instance,
    physical_device_index: usize,
    physical_device: vk::PhysicalDevice,
    selected_by_default: bool,
) -> Option<VulkanComputeDeviceInfo> {
    let compute_queue_family_indices =
        unsafe { compute_queue_family_indices(instance, physical_device) };
    if compute_queue_family_indices.is_empty() {
        return None;
    }
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_uuid = unsafe { physical_device_uuid(instance, physical_device) };
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let device_name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let memory_heaps = (0..memory_properties.memory_heap_count)
        .map(|heap_index| {
            let heap = memory_properties.memory_heaps[heap_index as usize];
            VulkanMemoryHeapInfo {
                heap_index,
                size_bytes: heap.size,
                device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            }
        })
        .collect();

    Some(VulkanComputeDeviceInfo {
        physical_device_index,
        physical_device_id: format!("vulkan-uuid:{}", format_device_uuid(&device_uuid)),
        device_uuid,
        device_name,
        device_type: vulkan_device_type_label(properties.device_type).to_string(),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        api_version: properties.api_version,
        driver_version: properties.driver_version,
        compute_queue_family_indices,
        memory_heaps,
        selected_by_default,
    })
}

unsafe fn compute_queue_family_indices(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<u32> {
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };
    preferred_compute_queue_family_indices(&queue_families)
}

fn preferred_compute_queue_family_indices(
    queue_families: &[vk::QueueFamilyProperties],
) -> Vec<u32> {
    let mut indices = queue_families
        .iter()
        .enumerate()
        .filter_map(|(index, family)| {
            (family.queue_count > 0
                && family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .then_some(index as u32)
        })
        .collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        let family = &queue_families[*index as usize];
        (
            family.queue_flags.contains(vk::QueueFlags::GRAPHICS),
            *index,
        )
    });
    indices
}

fn vulkan_device_type_label(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::OTHER => "other",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "unknown",
    }
}

unsafe fn create_nerve_vulkan_instance(entry: &Entry) -> Result<ash::Instance, VulkanError> {
    let app_name = CString::new("nerve-runtime").expect("static string has no nul");
    let engine_name = CString::new("nerve-dsp").expect("static string has no nul");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 4, 0));
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    unsafe { entry.create_instance(&instance_info, None) }
        .map_err(|error| VulkanError(format!("failed to create Vulkan instance: {error:?}")))
}

unsafe fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
    preferred_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    (0..memory_properties.memory_type_count)
        .filter(|index| {
            let supported = (memory_type_bits & (1 << index)) != 0;
            let properties = memory_properties.memory_types[*index as usize].property_flags;
            supported && properties.contains(required_flags)
        })
        .max_by_key(|index| {
            let memory_type = memory_properties.memory_types[*index as usize];
            let heap_size = memory_properties.memory_heaps[memory_type.heap_index as usize].size;
            let preferred_property_count = (memory_type.property_flags & preferred_flags)
                .as_raw()
                .count_ones();
            (preferred_property_count, heap_size)
        })
}

unsafe fn write_device_local_bytes(
    device: &ash::Device,
    destination: vk::Buffer,
    access: &VulkanResidentMemoryAccess,
    byte_len: vk::DeviceSize,
    input: &[u8],
) -> Result<(), VulkanError> {
    let memory_type_index = access
        .staging_memory_type_index
        .ok_or_else(|| VulkanError("device-local buffer has no staging memory type".to_string()))?;
    let (staging_buffer, staging_memory) = unsafe {
        create_temporary_staging_buffer(
            device,
            byte_len,
            vk::BufferUsageFlags::TRANSFER_SRC,
            memory_type_index,
        )?
    };
    let result = (|| {
        unsafe { write_byte_memory(device, staging_memory, byte_len, input) }?;
        unsafe {
            copy_buffer_immediately(
                device,
                access.queue,
                access.queue_family_index,
                staging_buffer,
                destination,
                byte_len,
            )
        }
    })();
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);
    }
    result
}

unsafe fn read_device_local_bytes(
    device: &ash::Device,
    source: vk::Buffer,
    access: &VulkanResidentMemoryAccess,
    byte_len: vk::DeviceSize,
) -> Result<Vec<u8>, VulkanError> {
    let memory_type_index = access
        .staging_memory_type_index
        .ok_or_else(|| VulkanError("device-local buffer has no staging memory type".to_string()))?;
    let (staging_buffer, staging_memory) = unsafe {
        create_temporary_staging_buffer(
            device,
            byte_len,
            vk::BufferUsageFlags::TRANSFER_DST,
            memory_type_index,
        )?
    };
    let result = (|| unsafe {
        copy_buffer_immediately(
            device,
            access.queue,
            access.queue_family_index,
            source,
            staging_buffer,
            byte_len,
        )?;
        read_byte_memory(device, staging_memory, byte_len, byte_len as usize)
    })();
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);
    }
    result
}

unsafe fn create_temporary_staging_buffer(
    device: &ash::Device,
    byte_len: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_type_index: u32,
) -> Result<(vk::Buffer, vk::DeviceMemory), VulkanError> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_len)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|error| VulkanError(format!("failed to create staging buffer: {error:?}")))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    if requirements.memory_type_bits & (1 << memory_type_index) == 0 {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(VulkanError(format!(
            "staging memory type {memory_type_index} is incompatible with transfer buffer"
        )));
    }
    let memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = match unsafe { device.allocate_memory(&memory_info, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(VulkanError(format!(
                "failed to allocate staging buffer memory: {error:?}"
            )));
        }
    };
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(VulkanError(format!(
            "failed to bind staging buffer memory: {error:?}"
        )));
    }
    Ok((buffer, memory))
}

unsafe fn copy_buffer_immediately(
    device: &ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    source: vk::Buffer,
    destination: vk::Buffer,
    byte_len: vk::DeviceSize,
) -> Result<(), VulkanError> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let command_pool =
        unsafe { device.create_command_pool(&command_pool_info, None) }.map_err(|error| {
            VulkanError(format!("failed to create staging command pool: {error:?}"))
        })?;
    let result = (|| {
        let command_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&command_alloc_info) }
            .map_err(|error| {
                VulkanError(format!(
                    "failed to allocate staging command buffer: {error:?}"
                ))
            })?
            .remove(0);
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.map_err(|error| {
            VulkanError(format!("failed to begin staging command buffer: {error:?}"))
        })?;
        let regions = [vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: byte_len,
        }];
        unsafe { device.cmd_copy_buffer(command_buffer, source, destination, &regions) };
        unsafe { device.end_command_buffer(command_buffer) }.map_err(|error| {
            VulkanError(format!("failed to end staging command buffer: {error:?}"))
        })?;
        let command_buffers = [command_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        unsafe { device.queue_submit(queue, &submit_info, vk::Fence::null()) }
            .map_err(|error| VulkanError(format!("failed to submit staging copy: {error:?}")))?;
        unsafe { device.queue_wait_idle(queue) }
            .map_err(|error| VulkanError(format!("failed waiting for staging copy: {error:?}")))
    })();
    unsafe { device.destroy_command_pool(command_pool, None) };
    result
}

unsafe fn write_byte_memory(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    byte_len: vk::DeviceSize,
    input: &[u8],
) -> Result<(), VulkanError> {
    let ptr = unsafe {
        device
            .map_memory(memory, 0, byte_len, vk::MemoryMapFlags::empty())
            .map_err(|error| VulkanError(format!("failed to map input memory: {error:?}")))?
    };
    let mapped = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), input.len()) };
    mapped.copy_from_slice(input);
    unsafe { device.unmap_memory(memory) };
    Ok(())
}

unsafe fn read_byte_memory(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    byte_len: vk::DeviceSize,
    len: usize,
) -> Result<Vec<u8>, VulkanError> {
    let ptr = unsafe {
        device
            .map_memory(memory, 0, byte_len, vk::MemoryMapFlags::empty())
            .map_err(|error| VulkanError(format!("failed to map output memory: {error:?}")))?
    };
    let output = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) }.to_vec();
    unsafe { device.unmap_memory(memory) };
    Ok(output)
}
