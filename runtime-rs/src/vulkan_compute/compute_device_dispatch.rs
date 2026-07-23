impl VulkanComputeDevice {
    pub fn create_resident_kernel_dispatch(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
    ) -> Result<VulkanResidentKernelDispatch, VulkanError> {
        self.create_resident_kernel_dispatch_labeled(
            spirv_words,
            buffers,
            workgroup_count_x,
            local_size_x,
            push_constant_byte_count,
            None,
        )
    }

    pub fn create_resident_kernel_dispatch_labeled(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
        semantic_label: Option<String>,
    ) -> Result<VulkanResidentKernelDispatch, VulkanError> {
        self.create_resident_kernel_dispatch_2d_labeled(
            spirv_words,
            buffers,
            workgroup_count_x,
            1,
            local_size_x,
            push_constant_byte_count,
            semantic_label,
        )
    }

    pub fn create_resident_kernel_dispatch_2d(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
    ) -> Result<VulkanResidentKernelDispatch, VulkanError> {
        self.create_resident_kernel_dispatch_2d_labeled(
            spirv_words,
            buffers,
            workgroup_count_x,
            workgroup_count_y,
            local_size_x,
            push_constant_byte_count,
            None,
        )
    }

    pub fn create_resident_kernel_dispatch_2d_labeled(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
        semantic_label: Option<String>,
    ) -> Result<VulkanResidentKernelDispatch, VulkanError> {
        self.create_resident_kernel_dispatch_2d_with_base_z(
            spirv_words,
            buffers,
            workgroup_count_x,
            workgroup_count_y,
            0,
            local_size_x,
            push_constant_byte_count,
            semantic_label,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_resident_kernel_dispatch_2d_with_base_z(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        base_workgroup_z: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
        semantic_label: Option<String>,
    ) -> Result<VulkanResidentKernelDispatch, VulkanError> {
        if spirv_words.is_empty() {
            return Err(VulkanError("SPIR-V module must not be empty".to_string()));
        }
        if buffers.is_empty() {
            return Err(VulkanError(
                "resident kernel dispatch must bind at least one storage buffer".to_string(),
            ));
        }
        if workgroup_count_x == 0 {
            return Err(VulkanError(
                "workgroup_count_x must not be zero".to_string(),
            ));
        }
        if workgroup_count_y == 0 {
            return Err(VulkanError(
                "workgroup_count_y must not be zero".to_string(),
            ));
        }
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }

        let mut descriptor_bindings = Vec::with_capacity(buffers.len());
        let mut buffer_accesses =
            Vec::<VulkanResidentKernelBufferAccessRecord>::with_capacity(buffers.len());
        for buffer in buffers {
            buffer
                .buffer
                .byte_range(buffer.byte_offset, buffer.byte_len)?;
            if descriptor_bindings.contains(&buffer.binding) {
                return Err(VulkanError(format!(
                    "duplicate storage buffer binding {}",
                    buffer.binding
                )));
            }
            descriptor_bindings.push(buffer.binding);
            if let Some(existing) = buffer_accesses
                .iter_mut()
                .find(|existing| existing.buffer == buffer.buffer.buffer)
            {
                existing.access = existing.access.merge(buffer.access);
            } else {
                buffer_accesses.push(VulkanResidentKernelBufferAccessRecord {
                    buffer: buffer.buffer.buffer,
                    access: buffer.access,
                });
            }
        }
        descriptor_bindings.sort_unstable();

        let pipeline_key = VulkanGenericPipelineKey {
            spirv_words: spirv_words.to_vec(),
            descriptor_bindings: descriptor_bindings.clone(),
            push_constant_byte_count,
            local_size_x,
        };
        let (descriptor_set_layout, pipeline_layout, pipeline) = self.generic_storage_pipeline(
            spirv_words,
            &descriptor_bindings,
            push_constant_byte_count,
            local_size_x,
        )?;

        unsafe {
            let set_layouts = [descriptor_set_layout];
            let descriptor_count = u32::try_from(buffers.len()).map_err(|_| {
                VulkanError("resident kernel descriptor count overflowed u32".to_string())
            })?;
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count,
            }];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let descriptor_pool = self
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident kernel descriptor pool: {error:?}"
                    ))
                })?;
            let descriptor_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_set = self
                .device
                .allocate_descriptor_sets(&descriptor_alloc_info)
                .map_err(|error| {
                    self.device.destroy_descriptor_pool(descriptor_pool, None);
                    VulkanError(format!(
                        "failed to allocate resident kernel descriptor set: {error:?}"
                    ))
                })?
                .remove(0);
            let descriptor_buffers = buffers
                .iter()
                .map(|buffer| {
                    buffer
                        .buffer
                        .descriptor_buffer(buffer.byte_offset, buffer.byte_len)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let descriptor_writes = buffers
                .iter()
                .zip(&descriptor_buffers)
                .map(|(buffer, descriptor_buffer)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(buffer.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(descriptor_buffer))
                })
                .collect::<Vec<_>>();
            self.device.update_descriptor_sets(&descriptor_writes, &[]);

            Ok(VulkanResidentKernelDispatch {
                device: self.device.clone(),
                descriptor_pool,
                descriptor_set,
                pipeline_key,
                pipeline_layout,
                pipeline,
                descriptor_count: buffers.len(),
                workgroup_count_x,
                workgroup_count_y,
                base_workgroup_z,
                push_constant_byte_count,
                buffer_accesses,
                semantic_label,
            })
        }
    }

    pub fn run_resident_kernel_dispatch(
        &self,
        binding: &VulkanResidentKernelDispatch,
        push_constants: &[u8],
    ) -> Result<(), VulkanError> {
        let mut immediate = self.immediate_kernel_sequence.borrow_mut();
        if immediate.is_none() {
            *immediate = Some(self.create_resident_kernel_sequence()?);
        }
        self.run_resident_kernel_sequence(
            immediate
                .as_ref()
                .expect("immediate sequence was initialized"),
            &[VulkanResidentKernelSequenceStep::new(
                binding,
                push_constants,
            )],
        )
    }

    pub fn create_resident_kernel_sequence(
        &self,
    ) -> Result<VulkanResidentKernelSequence, VulkanError> {
        unsafe {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident kernel sequence command pool: {error:?}"
                    ))
                })?;
            let command_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = self
                .device
                .allocate_command_buffers(&command_alloc_info)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to allocate resident kernel sequence command buffer: {error:?}"
                    ))
                })?
                .remove(0);
            let completion_fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to create resident kernel sequence completion fence: {error:?}"
                    ))
                })?;

            Ok(VulkanResidentKernelSequence {
                device: self.device.clone(),
                command_pool,
                command_buffer,
                completion_fence,
                timestamp_period_ns: self.timestamp_period_ns,
                recorded_input_copies: RefCell::new(None),
                recorded_steps: RefCell::new(None),
                recorded_snapshot_copies: RefCell::new(None),
            })
        }
    }
}
