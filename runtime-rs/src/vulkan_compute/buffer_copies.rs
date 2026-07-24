pub struct VulkanResidentBufferCopy {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    source: vk::Buffer,
    destination: vk::Buffer,
    byte_len: vk::DeviceSize,
}

pub struct VulkanResidentBufferCopyBatch {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    completion_fence: vk::Fence,
    copy_count: usize,
}

#[derive(Clone, Copy)]
pub struct VulkanResidentBufferRangeCopy<'a> {
    source: &'a VulkanResidentBuffer,
    destination: &'a VulkanResidentBuffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

impl<'a> VulkanResidentBufferRangeCopy<'a> {
    pub fn new(
        source: &'a VulkanResidentBuffer,
        destination: &'a VulkanResidentBuffer,
        source_offset: usize,
        destination_offset: usize,
        byte_len: usize,
    ) -> Result<Self, VulkanError> {
        const VULKAN_BUFFER_COPY_ALIGNMENT: usize = 4;
        if byte_len == 0 {
            return Err(VulkanError(
                "resident buffer range copy length must not be zero".to_string(),
            ));
        }
        if !source_offset.is_multiple_of(VULKAN_BUFFER_COPY_ALIGNMENT)
            || !destination_offset.is_multiple_of(VULKAN_BUFFER_COPY_ALIGNMENT)
            || !byte_len.is_multiple_of(VULKAN_BUFFER_COPY_ALIGNMENT)
        {
            return Err(VulkanError(format!(
                "resident buffer range copy offsets and length must be multiples of {VULKAN_BUFFER_COPY_ALIGNMENT}, got source offset {source_offset}, destination offset {destination_offset}, and length {byte_len}"
            )));
        }
        source.byte_range(source_offset, byte_len)?;
        destination.byte_range(destination_offset, byte_len)?;
        Ok(Self {
            source,
            destination,
            source_offset: source_offset as vk::DeviceSize,
            destination_offset: destination_offset as vk::DeviceSize,
            byte_len: byte_len as vk::DeviceSize,
        })
    }
}

pub struct VulkanResidentMappedBufferCopy {
    source_address: usize,
    destination_address: usize,
    byte_len: usize,
}

impl VulkanResidentMappedBufferCopy {
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn run(&self, len: usize) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "persistently mapped resident copy length must not be zero".to_string(),
            ));
        }
        if len != self.byte_len {
            return Err(VulkanError(format!(
                "persistently mapped resident copy binding byte length {} cannot run {} bytes",
                self.byte_len, len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.source_address as *const u8,
                self.destination_address as *mut u8,
                len,
            );
        }
        Ok(())
    }
}

impl VulkanResidentBufferCopy {
    pub fn byte_len(&self) -> usize {
        self.byte_len as usize
    }

    pub fn run(&self, len: usize) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte copy length must not be zero".to_string(),
            ));
        }
        let byte_len = len as vk::DeviceSize;
        if byte_len != self.byte_len {
            return Err(VulkanError(format!(
                "resident byte copy binding byte length {} cannot run {} bytes",
                self.byte_len, byte_len
            )));
        }

        unsafe {
            let command_buffers = [self.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| {
                    VulkanError(format!("failed to submit resident byte copy: {error:?}"))
                })?;
            RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
            self.device.queue_wait_idle(self.queue).map_err(|error| {
                VulkanError(format!("failed waiting for resident byte copy: {error:?}"))
            })?;
            RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
}

impl VulkanResidentBufferCopyBatch {
    pub fn copy_count(&self) -> usize {
        self.copy_count
    }

    pub fn run(&self) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .reset_fences(&[self.completion_fence])
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident buffer copy batch fence: {error:?}"
                    ))
                })?;
            let command_buffers = [self.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, self.completion_fence)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to submit resident buffer copy batch: {error:?}"
                    ))
                })?;
            RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
            self.device
                .wait_for_fences(&[self.completion_fence], true, u64::MAX)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed waiting for resident buffer copy batch: {error:?}"
                    ))
                })?;
            RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

impl Drop for VulkanResidentBufferCopy {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl Drop for VulkanResidentBufferCopyBatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.completion_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl VulkanResidentKernelDispatch {
    pub fn semantic_label(&self) -> Option<&str> {
        self.semantic_label.as_deref()
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count
    }

    pub fn workgroup_count_x(&self) -> u32 {
        self.workgroup_count_x
    }

    pub fn workgroup_count_y(&self) -> u32 {
        self.workgroup_count_y
    }

    pub fn local_size_x(&self) -> u32 {
        self.pipeline_key.local_size_x
    }

    pub fn estimated_work_units(&self) -> u64 {
        u64::from(self.workgroup_count_x)
            .saturating_mul(u64::from(self.workgroup_count_y))
            .saturating_mul(u64::from(self.pipeline_key.local_size_x))
    }

    pub fn estimated_memory_bytes(&self) -> u64 {
        self.estimated_memory_bytes
    }

    pub fn execution_family(&self) -> String {
        let operation = self
            .semantic_label
            .as_deref()
            .and_then(|label| semantic_label_field(label, "op"))
            .unwrap_or("unlabeled");
        format!(
            "{operation}@{}x{}x{}",
            self.workgroup_count_x,
            self.workgroup_count_y,
            self.pipeline_key.local_size_x
        )
    }

    pub fn push_constant_byte_count(&self) -> u32 {
        self.push_constant_byte_count
    }
}

impl Drop for VulkanResidentKernelDispatch {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl Drop for VulkanResidentKernelSequence {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.completion_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
