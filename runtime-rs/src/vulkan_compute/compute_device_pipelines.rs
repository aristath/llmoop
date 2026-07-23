impl VulkanComputeDevice {
    fn generic_storage_pipeline(
        &self,
        spirv_words: &[u32],
        descriptor_bindings: &[u32],
        push_constant_byte_count: u32,
        local_size_x: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), VulkanError> {
        validate_spirv_device_contract(
            spirv_words,
            &self.enabled_shader_features,
            self.subgroup_supported_stages,
            self.subgroup_supported_operations,
        )?;
        let key = VulkanGenericPipelineKey {
            spirv_words: spirv_words.to_vec(),
            descriptor_bindings: descriptor_bindings.to_vec(),
            push_constant_byte_count,
            local_size_x,
        };
        if let Some(pipeline) = self.generic_storage_pipelines.borrow().get(&key) {
            return Ok((
                pipeline.descriptor_set_layout,
                pipeline.pipeline_layout,
                pipeline.pipeline,
            ));
        }

        let pipeline = unsafe {
            self.create_generic_storage_pipeline(
                spirv_words,
                descriptor_bindings,
                push_constant_byte_count,
            )?
        };
        let handles = (
            pipeline.descriptor_set_layout,
            pipeline.pipeline_layout,
            pipeline.pipeline,
        );
        self.generic_storage_pipelines
            .borrow_mut()
            .insert(key, pipeline);
        Ok(handles)
    }

    unsafe fn create_generic_storage_pipeline(
        &self,
        spirv_words: &[u32],
        descriptor_bindings: &[u32],
        push_constant_byte_count: u32,
    ) -> Result<VulkanStoragePipeline, VulkanError> {
        let descriptor_binding = descriptor_bindings
            .iter()
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_binding);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create generic descriptor set layout: {error:?}"
                    ))
                })?
        };

        let set_layouts = [descriptor_set_layout];
        let push_constant_ranges = if push_constant_byte_count == 0 {
            Vec::new()
        } else {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_byte_count),
            ]
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|error| {
                    self.device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    VulkanError(format!(
                        "failed to create generic pipeline layout: {error:?}"
                    ))
                })?
        };

        let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(|error| {
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    VulkanError(format!("failed to create generic shader module: {error:?}"))
                })?
        };
        let entry_point = CString::new("main").expect("static string has no nul");
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .map_err(|(_, error)| {
                    self.device.destroy_shader_module(shader_module, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    VulkanError(format!(
                        "failed to create generic compute pipeline: {error:?}"
                    ))
                })?
                .remove(0)
        };

        Ok(VulkanStoragePipeline {
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
        })
    }
}

