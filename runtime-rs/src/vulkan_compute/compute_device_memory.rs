impl VulkanComputeDevice {
    pub fn create_shared_host_allocation(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
    ) -> Result<Arc<VulkanSharedHostAllocation>, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "shared host allocation capacity must not be zero".to_string(),
            ));
        }
        let mut alignment = 1usize;
        let mut required_size = byte_capacity;
        for device in std::iter::once(self).chain(peer_devices.iter().copied()) {
            alignment = alignment.max(device.shared_host_memory_alignment.ok_or_else(|| {
                VulkanError(format!(
                    "Vulkan device {:?} cannot import shared host memory",
                    device.device_name
                ))
            })?);
            let requirements = device.shared_host_buffer_memory_requirements(byte_capacity)?;
            alignment =
                alignment.max(usize::try_from(requirements.alignment).map_err(|_| {
                    VulkanError("shared buffer alignment exceeds usize".to_string())
                })?);
            required_size =
                required_size.max(usize::try_from(requirements.size).map_err(|_| {
                    VulkanError("shared buffer allocation size exceeds usize".to_string())
                })?);
        }
        if !alignment.is_power_of_two() {
            return Err(VulkanError(format!(
                "shared buffer alignment {alignment} is not a power of two"
            )));
        }
        let allocation_size = required_size
            .checked_add(alignment - 1)
            .map(|size| size & !(alignment - 1))
            .ok_or_else(|| VulkanError("shared host allocation size overflowed".to_string()))?;
        let layout = Layout::from_size_align(allocation_size, alignment).map_err(|error| {
            VulkanError(format!("invalid shared host allocation layout: {error}"))
        })?;
        let pointer = unsafe { alloc_zeroed(layout) };
        if pointer.is_null() {
            return Err(VulkanError(format!(
                "failed to allocate {allocation_size} bytes of aligned shared host memory"
            )));
        }
        Ok(Arc::new(VulkanSharedHostAllocation {
            address: pointer as usize,
            layout,
            byte_capacity,
        }))
    }

    fn shared_host_buffer_memory_requirements(
        &self,
        byte_capacity: usize,
    ) -> Result<vk::MemoryRequirements, VulkanError> {
        unsafe {
            let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(byte_capacity as vk::DeviceSize)
                .usage(resident_buffer_usage())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external_buffer_info);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to query shared host-backed buffer requirements: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            self.device.destroy_buffer(buffer, None);
            Ok(requirements)
        }
    }

    pub fn import_shared_host_buffer(
        &self,
        allocation: Arc<VulkanSharedHostAllocation>,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if self.shared_host_memory_alignment.is_none() {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot import shared host memory",
                self.device_name
            )));
        }
        let loader =
            ash::ext::external_memory_host::Device::new(&self.context.instance, &self.device);
        let mut host_properties = vk::MemoryHostPointerPropertiesEXT::default();
        let result = unsafe {
            (loader.fp().get_memory_host_pointer_properties_ext)(
                loader.device(),
                VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE,
                allocation.address as *const c_void,
                &mut host_properties,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(VulkanError(format!(
                "failed to query shared host-pointer memory types: {result:?}"
            )));
        }

        unsafe {
            let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(allocation.byte_capacity as vk::DeviceSize)
                .usage(resident_buffer_usage())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external_buffer_info);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create shared host-backed storage buffer: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            if requirements.size > allocation.layout.size() as vk::DeviceSize {
                self.device.destroy_buffer(buffer, None);
                return Err(VulkanError(format!(
                    "shared host allocation has {} bytes but Vulkan requires {}",
                    allocation.layout.size(),
                    requirements.size
                )));
            }
            let compatible_memory_types =
                requirements.memory_type_bits & host_properties.memory_type_bits;
            let memory_type_index = match find_memory_type(
                &self.context.instance,
                self.physical_device,
                compatible_memory_types,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
            ) {
                Some(index) => index,
                None => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(VulkanError(format!(
                        "no host-visible memory type can import the shared allocation (buffer types {:#010x}, host types {:#010x})",
                        requirements.memory_type_bits, host_properties.memory_type_bits
                    )));
                }
            };
            let memory_access = match self.resident_memory_access(memory_type_index) {
                Ok(access) => access,
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(error);
                }
            };
            let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE)
                .host_pointer(allocation.address as *mut c_void);
            let memory_info = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation.layout.size() as vk::DeviceSize)
                .memory_type_index(memory_type_index)
                .push_next(&mut import_info);
            let memory = match self.device.allocate_memory(&memory_info, None) {
                Ok(memory) => memory,
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(VulkanError(format!(
                        "failed to import shared host allocation: {error:?}"
                    )));
                }
            };
            if let Err(error) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
                return Err(VulkanError(format!(
                    "failed to bind shared host allocation: {error:?}"
                )));
            }
            Ok(VulkanResidentBuffer {
                device: self.device.clone(),
                buffer,
                memory,
                memory_access,
                byte_capacity: allocation.byte_capacity as vk::DeviceSize,
                persistent_mapping: Some(allocation.address),
                persistent_mapping_requires_unmap: false,
                _shared_host_allocation: Some(allocation),
            })
        }
    }

    pub fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<VulkanTimelineSemaphore, VulkanError> {
        self.create_timeline_semaphore_with_opaque_fd_export(initial_value, false)
    }

    pub fn create_opaque_fd_exportable_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<VulkanTimelineSemaphore, VulkanError> {
        if !self.opaque_fd_timeline_semaphore_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot export persistent opaque-file timeline semaphores",
                self.device_name
            )));
        }
        self.create_timeline_semaphore_with_opaque_fd_export(initial_value, true)
    }

    pub fn wait_timeline_semaphore_value(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        value: u64,
    ) -> Result<(), VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        let semaphores = [semaphore.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) }.map_err(|error| {
            VulkanError(format!(
                "failed to wait for timeline semaphore value {value}: {error:?}"
            ))
        })
    }

    fn create_timeline_semaphore_with_opaque_fd_export(
        &self,
        initial_value: u64,
        opaque_fd_exportable: bool,
    ) -> Result<VulkanTimelineSemaphore, VulkanError> {
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let semaphore = if opaque_fd_exportable {
            let mut export_info = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE);
            let create_info = vk::SemaphoreCreateInfo::default()
                .push_next(&mut timeline_info)
                .push_next(&mut export_info);
            unsafe { self.device.create_semaphore(&create_info, None) }
        } else {
            let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
            unsafe { self.device.create_semaphore(&create_info, None) }
        }
        .map_err(|error| VulkanError(format!("failed to create timeline semaphore: {error:?}")))?;
        Ok(VulkanTimelineSemaphore {
            device: self.device.clone(),
            device_handle: self.device.handle(),
            semaphore,
            opaque_fd_exportable,
            permanent_opaque_fd_imported: Cell::new(false),
        })
    }

    pub fn export_timeline_semaphore_opaque_fd(
        &self,
        semaphore: &VulkanTimelineSemaphore,
    ) -> Result<OwnedFd, VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        if !semaphore.opaque_fd_exportable {
            return Err(VulkanError(
                "timeline semaphore was not created for persistent opaque-file export".to_string(),
            ));
        }
        let loader =
            ash::khr::external_semaphore_fd::Device::new(&self.context.instance, &self.device);
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore.semaphore)
            .handle_type(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE);
        let fd = unsafe { loader.get_semaphore_fd(&get_info) }.map_err(|error| {
            VulkanError(format!(
                "failed to export timeline semaphore as persistent opaque file: {error:?}"
            ))
        })?;
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    pub fn import_timeline_semaphore_opaque_fd(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        fd: OwnedFd,
    ) -> Result<(), VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        if !self.opaque_fd_timeline_semaphore_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot import persistent opaque-file timeline semaphores",
                self.device_name
            )));
        }
        if semaphore.permanent_opaque_fd_imported.get() {
            return Err(VulkanError(
                "timeline semaphore already has a permanently imported opaque-file payload"
                    .to_string(),
            ));
        }
        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore.semaphore)
            .flags(vk::SemaphoreImportFlags::empty())
            .handle_type(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE)
            .fd(fd.as_raw_fd());
        let loader =
            ash::khr::external_semaphore_fd::Device::new(&self.context.instance, &self.device);
        unsafe { loader.import_semaphore_fd(&import_info) }.map_err(|error| {
            VulkanError(format!(
                "failed to import timeline semaphore persistent opaque file: {error:?}"
            ))
        })?;
        let _fd_owned_by_vulkan = fd.into_raw_fd();
        semaphore.permanent_opaque_fd_imported.set(true);
        Ok(())
    }

    fn validate_local_timeline_semaphore(
        &self,
        semaphore: &VulkanTimelineSemaphore,
    ) -> Result<(), VulkanError> {
        if semaphore.device_handle != self.device.handle() {
            return Err(VulkanError(
                "timeline semaphore belongs to a different Vulkan logical device".to_string(),
            ));
        }
        Ok(())
    }

    fn resident_memory_access(
        &self,
        memory_type_index: u32,
    ) -> Result<VulkanResidentMemoryAccess, VulkanError> {
        let memory_properties = unsafe {
            self.context
                .instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let property_flags =
            memory_properties.memory_types[memory_type_index as usize].property_flags;
        let directly_mappable = property_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let staging_memory_type_index = if directly_mappable {
            None
        } else {
            Some(
                unsafe {
                    find_memory_type(
                        &self.context.instance,
                        self.physical_device,
                        u32::MAX,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                        vk::MemoryPropertyFlags::empty(),
                    )
                }
                .ok_or_else(|| {
                    VulkanError(
                        "no host-visible coherent memory type for resident staging transfers"
                            .to_string(),
                    )
                })?,
            )
        };
        Ok(VulkanResidentMemoryAccess {
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            property_flags,
            staging_memory_type_index,
        })
    }

    pub fn create_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "resident byte buffer capacity must not be zero".to_string(),
            ));
        }
        let (buffer, memory, byte_capacity, memory_access) = self.create_resident_storage_buffer(
            byte_capacity,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            memory_access,
            byte_capacity,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
        })
    }

    pub fn create_host_visible_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "resident byte buffer capacity must not be zero".to_string(),
            ));
        }
        let (buffer, memory, byte_capacity, memory_access) = self.create_resident_storage_buffer(
            byte_capacity,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            memory_access,
            byte_capacity,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
        })
    }

    fn create_resident_storage_buffer(
        &self,
        byte_capacity: usize,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<
        (
            vk::Buffer,
            vk::DeviceMemory,
            vk::DeviceSize,
            VulkanResidentMemoryAccess,
        ),
        VulkanError,
    > {
        let byte_capacity = byte_capacity as vk::DeviceSize;
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(byte_capacity)
                .usage(resident_buffer_usage())
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident storage buffer: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let memory_type_index = find_memory_type(
                &self.context.instance,
                self.physical_device,
                requirements.memory_type_bits,
                required_memory_flags,
                preferred_memory_flags,
            )
            .ok_or_else(|| {
                VulkanError(format!(
                    "no memory type with required flags {required_memory_flags:?} for resident storage buffer"
                ))
            })?;
            let memory_properties = self
                .context
                .instance
                .get_physical_device_memory_properties(self.physical_device);
            let property_flags =
                memory_properties.memory_types[memory_type_index as usize].property_flags;
            let directly_mappable = property_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let staging_memory_type_index = if directly_mappable {
                None
            } else {
                Some(
                    find_memory_type(
                        &self.context.instance,
                        self.physical_device,
                        u32::MAX,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                        vk::MemoryPropertyFlags::empty(),
                    )
                    .ok_or_else(|| {
                        VulkanError(
                            "no host-visible coherent memory type for resident staging transfers"
                                .to_string(),
                        )
                    })?,
                )
            };
            let memory_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index);
            let memory = self
                .device
                .allocate_memory(&memory_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to allocate resident storage buffer memory: {error:?}"
                    ))
                })?;
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to bind resident storage buffer memory: {error:?}"
                    ))
                })?;
            Ok((
                buffer,
                memory,
                byte_capacity,
                VulkanResidentMemoryAccess {
                    queue: self.queue,
                    queue_family_index: self.queue_family_index,
                    property_flags,
                    staging_memory_type_index,
                },
            ))
        }
    }

    pub fn copy_resident_buffer_bytes(
        &self,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<(), VulkanError> {
        let binding = self.create_resident_buffer_copy(source, destination, len)?;
        self.run_resident_buffer_copy(&binding, len)
    }

    pub fn create_resident_buffer_copy(
        &self,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<VulkanResidentBufferCopy, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte copy binding length must not be zero".to_string(),
            ));
        }
        let byte_len = source.byte_len(len)?;
        destination.byte_len(len)?;

        unsafe {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident byte copy binding command pool: {error:?}"
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
                        "failed to allocate resident byte copy binding command buffer: {error:?}"
                    ))
                })?
                .remove(0);

            let command_begin = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(command_buffer, &command_begin)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to begin resident byte copy binding command buffer: {error:?}"
                    ))
                })?;
            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: byte_len,
            }];
            self.device.cmd_copy_buffer(
                command_buffer,
                source.buffer,
                destination.buffer,
                &copy_regions,
            );
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to end resident byte copy binding command buffer: {error:?}"
                    ))
                })?;

            Ok(VulkanResidentBufferCopy {
                device: self.device.clone(),
                queue: self.queue,
                command_pool,
                command_buffer,
                source: source.buffer,
                destination: destination.buffer,
                byte_len,
            })
        }
    }

    pub fn create_resident_buffer_copy_batch(
        &self,
        copies: &[VulkanResidentBufferRangeCopy<'_>],
    ) -> Result<VulkanResidentBufferCopyBatch, VulkanError> {
        if copies.is_empty() {
            return Err(VulkanError(
                "resident buffer copy batch must contain at least one copy".to_string(),
            ));
        }
        unsafe {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident buffer copy batch command pool: {error:?}"
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
                        "failed to allocate resident buffer copy batch command buffer: {error:?}"
                    ))
                })?
                .remove(0);
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to begin resident buffer copy batch command buffer: {error:?}"
                    ))
                })?;
            for copy in copies {
                let regions = [vk::BufferCopy {
                    src_offset: copy.source_offset,
                    dst_offset: copy.destination_offset,
                    size: copy.byte_len,
                }];
                self.device.cmd_copy_buffer(
                    command_buffer,
                    copy.source.buffer,
                    copy.destination.buffer,
                    &regions,
                );
            }
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to end resident buffer copy batch command buffer: {error:?}"
                    ))
                })?;
            let completion_fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to create resident buffer copy batch fence: {error:?}"
                    ))
                })?;
            Ok(VulkanResidentBufferCopyBatch {
                device: self.device.clone(),
                queue: self.queue,
                command_pool,
                command_buffer,
                completion_fence,
                copy_count: copies.len(),
            })
        }
    }

    pub fn run_resident_buffer_copy(
        &self,
        binding: &VulkanResidentBufferCopy,
        len: usize,
    ) -> Result<(), VulkanError> {
        binding.run(len)
    }
}
