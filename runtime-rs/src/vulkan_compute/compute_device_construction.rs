impl VulkanComputeDevice {
    pub fn available_compute_devices() -> Result<Vec<VulkanComputeDeviceInfo>, VulkanError> {
        Ok(VulkanComputeDeviceCatalog::discover()?
            .available_devices
            .clone())
    }

    pub fn new() -> Result<Self, VulkanError> {
        Self::new_with_physical_device_selector(None, None)
    }

    pub fn new_for_physical_device_index(
        physical_device_index: usize,
    ) -> Result<Self, VulkanError> {
        Self::new_with_physical_device_selector(Some(physical_device_index), None)
    }

    pub fn new_for_device_uuid(device_uuid: [u8; vk::UUID_SIZE]) -> Result<Self, VulkanError> {
        Self::new_with_physical_device_selector(None, Some(device_uuid))
    }

    fn new_with_physical_device_selector(
        requested_physical_device_index: Option<usize>,
        requested_device_uuid: Option<[u8; vk::UUID_SIZE]>,
    ) -> Result<Self, VulkanError> {
        VulkanComputeDeviceCatalog::discover()?
            .open_device(requested_physical_device_index, requested_device_uuid)
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn has_enabled_device_extension(&self, extension_name: &str) -> bool {
        self.enabled_device_extensions.contains(extension_name)
    }

    pub fn has_enabled_shader_feature(&self, feature: VulkanShaderFeature) -> bool {
        self.enabled_shader_features.contains(&feature)
    }

    pub fn supports_subgroup_operation(&self, operation: VulkanSubgroupOperation) -> bool {
        self.subgroup_supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            && self
                .subgroup_supported_operations
                .contains(operation.flag())
    }

    pub fn supports_cooperative_bfloat16_shape(&self, m: u32, n: u32, k: u32) -> bool {
        self.cooperative_bfloat16_shapes.contains(&(m, n, k))
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    pub fn supports_compute_local_size_x(&self, local_size_x: u32) -> bool {
        local_size_x > 0
            && local_size_x <= self.max_compute_work_group_invocations
            && local_size_x <= self.max_compute_work_group_size_x
    }

    pub fn min_storage_buffer_offset_alignment(&self) -> usize {
        self.min_storage_buffer_offset_alignment
    }

    pub fn device_local_memory_bytes(&self) -> u64 {
        self.device_local_memory_bytes
    }

    pub fn supports_shared_host_memory(&self) -> bool {
        self.shared_host_memory_alignment.is_some()
    }

    pub fn supports_opaque_fd_timeline_semaphores(&self) -> bool {
        self.opaque_fd_timeline_semaphore_supported
    }

    pub fn owns_resident_buffer(&self, buffer: &VulkanResidentBuffer) -> bool {
        self.device.handle() == buffer.device.handle()
    }

    pub fn shares_logical_device_with(&self, other: &Self) -> bool {
        self.device.handle() == other.device.handle()
    }
}
