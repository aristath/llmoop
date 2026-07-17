use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::error::Error;
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::path::Path;

use ash::{Entry, vk};

use crate::vulkan::SpirvPedalProgram;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSmokeResult {
    pub device_name: String,
    pub input: Vec<u32>,
    pub output: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanError(pub String);

impl Display for VulkanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanU32PedalRun {
    pub pedal_id: String,
    pub operator_type: String,
    pub device_name: String,
    pub input: Vec<u32>,
    pub output: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanU32ShaderPedal {
    program: SpirvPedalProgram,
    local_size_x: u32,
}

impl VulkanU32ShaderPedal {
    pub fn new(
        pedal_id: impl Into<String>,
        operator_type: impl Into<String>,
        spirv_words: Vec<u32>,
        local_size_x: u32,
    ) -> Result<Self, VulkanError> {
        Self::from_program(
            SpirvPedalProgram::new(pedal_id, operator_type, spirv_words)
                .with_local_size_x(local_size_x),
        )
    }

    pub fn from_program(program: SpirvPedalProgram) -> Result<Self, VulkanError> {
        if program.words.is_empty() {
            return Err(VulkanError(
                "SPIR-V pedal program must not be empty".to_string(),
            ));
        }
        if program.entry_point != "main" {
            return Err(VulkanError(format!(
                "only entry point \"main\" is currently supported, got {:?}",
                program.entry_point
            )));
        }
        if program.local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }
        let local_size_x = program.local_size_x;
        Ok(Self {
            program,
            local_size_x,
        })
    }

    pub fn pedal_id(&self) -> &str {
        &self.program.pedal_id
    }

    pub fn operator_type(&self) -> &str {
        &self.program.operator_type
    }

    pub fn process(
        &self,
        device: &VulkanComputeDevice,
        input: &[u32],
    ) -> Result<VulkanU32PedalRun, VulkanError> {
        let output =
            device.run_u32_storage_shader(&self.program.words, input, self.local_size_x)?;
        Ok(VulkanU32PedalRun {
            pedal_id: self.program.pedal_id.clone(),
            operator_type: self.program.operator_type.clone(),
            device_name: device.device_name().to_string(),
            input: input.to_vec(),
            output,
        })
    }

    pub fn process_resident(
        &self,
        device: &VulkanComputeDevice,
        buffer: &VulkanU32ResidentBuffer,
        len: usize,
    ) -> Result<VulkanU32PedalRun, VulkanError> {
        let input = buffer.read(len)?;
        device.run_u32_storage_shader_on_resident_buffer(
            &self.program.words,
            buffer,
            len,
            self.local_size_x,
        )?;
        let output = buffer.read(len)?;
        Ok(VulkanU32PedalRun {
            pedal_id: self.program.pedal_id.clone(),
            operator_type: self.program.operator_type.clone(),
            device_name: device.device_name().to_string(),
            input,
            output,
        })
    }

    pub fn create_resident_dispatch(
        &self,
        device: &VulkanComputeDevice,
        buffer: &VulkanU32ResidentBuffer,
        len: usize,
    ) -> Result<VulkanU32ResidentDispatch, VulkanError> {
        device.create_u32_resident_dispatch(&self.program.words, buffer, len, self.local_size_x)
    }

    pub fn process_bound_resident(
        &self,
        device: &VulkanComputeDevice,
        binding: &VulkanU32ResidentDispatch,
        len: usize,
    ) -> Result<VulkanU32PedalRun, VulkanError> {
        let input = binding.read(len)?;
        device.run_u32_storage_shader_on_resident_dispatch(
            &self.program.words,
            binding,
            len,
            self.local_size_x,
        )?;
        let output = binding.read(len)?;
        Ok(VulkanU32PedalRun {
            pedal_id: self.program.pedal_id.clone(),
            operator_type: self.program.operator_type.clone(),
            device_name: device.device_name().to_string(),
            input,
            output,
        })
    }

    pub fn dispatch_bound_resident(
        &self,
        device: &VulkanComputeDevice,
        binding: &VulkanU32ResidentDispatch,
        len: usize,
    ) -> Result<(), VulkanError> {
        device.run_u32_storage_shader_on_resident_dispatch(
            &self.program.words,
            binding,
            len,
            self.local_size_x,
        )
    }

    pub fn install_on_device(&self, device: &VulkanComputeDevice) -> Result<(), VulkanError> {
        device.install_u32_storage_shader(&self.program.words, self.local_size_x)
    }
}

pub struct VulkanComputeDevice {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    device_name: String,
    u32_storage_pipelines: RefCell<HashMap<VulkanPipelineKey, VulkanU32StoragePipeline>>,
    generic_storage_pipelines: RefCell<HashMap<VulkanGenericPipelineKey, VulkanStoragePipeline>>,
    pipeline_cache_hits: Cell<u64>,
    pipeline_cache_misses: Cell<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanComputeDeviceInfo {
    pub physical_device_index: usize,
    pub physical_device_id: String,
    pub device_name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub compute_queue_family_indices: Vec<u32>,
    pub memory_heaps: Vec<VulkanMemoryHeapInfo>,
    pub selected_by_default: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMemoryHeapInfo {
    pub heap_index: u32,
    pub size_bytes: u64,
    pub device_local: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPipelineCacheStats {
    pub u32_storage_pipelines: usize,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VulkanPipelineKey {
    spirv_words: Vec<u32>,
    local_size_x: u32,
}

struct VulkanU32StoragePipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VulkanGenericPipelineKey {
    spirv_words: Vec<u32>,
    descriptor_bindings: Vec<u32>,
    push_constant_byte_count: u32,
    local_size_x: u32,
}

struct VulkanStoragePipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

pub struct VulkanU32ResidentBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: usize,
    byte_capacity: vk::DeviceSize,
}

pub struct VulkanResidentBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    byte_capacity: vk::DeviceSize,
}

pub struct VulkanResidentKernelBufferBinding<'a> {
    pub binding: u32,
    pub buffer: &'a VulkanResidentBuffer,
    pub byte_len: usize,
}

impl<'a> VulkanResidentKernelBufferBinding<'a> {
    pub fn new(binding: u32, buffer: &'a VulkanResidentBuffer, byte_len: usize) -> Self {
        Self {
            binding,
            buffer,
            byte_len,
        }
    }
}

impl VulkanU32ResidentBuffer {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn write(&self, input: &[u32]) -> Result<(), VulkanError> {
        let byte_len = self.byte_len(input.len())?;
        unsafe { write_u32_memory(&self.device, self.memory, byte_len, input) }
    }

    pub fn read(&self, len: usize) -> Result<Vec<u32>, VulkanError> {
        let byte_len = self.byte_len(len)?;
        unsafe { read_u32_memory(&self.device, self.memory, byte_len, len) }
    }

    fn byte_len(&self, len: usize) -> Result<vk::DeviceSize, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident buffer length must not be zero".to_string(),
            ));
        }
        if len > self.capacity {
            return Err(VulkanError(format!(
                "resident buffer capacity {} cannot hold {} u32 values",
                self.capacity, len
            )));
        }
        let byte_len = len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError("resident buffer byte length overflowed".to_string()))?
            as vk::DeviceSize;
        if byte_len > self.byte_capacity {
            return Err(VulkanError(format!(
                "resident buffer byte capacity {} cannot hold {} bytes",
                self.byte_capacity, byte_len
            )));
        }
        Ok(byte_len)
    }

    fn descriptor_buffer(&self, len: usize) -> Result<vk::DescriptorBufferInfo, VulkanError> {
        Ok(vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: self.byte_len(len)?,
        })
    }
}

impl VulkanResidentBuffer {
    pub fn byte_capacity(&self) -> usize {
        self.byte_capacity as usize
    }

    pub fn write_bytes(&self, input: &[u8]) -> Result<(), VulkanError> {
        let byte_len = self.byte_len(input.len())?;
        unsafe { write_byte_memory(&self.device, self.memory, byte_len, input) }
    }

    pub fn read_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        let byte_len = self.byte_len(len)?;
        unsafe { read_byte_memory(&self.device, self.memory, byte_len, len) }
    }

    fn byte_len(&self, len: usize) -> Result<vk::DeviceSize, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte buffer length must not be zero".to_string(),
            ));
        }
        let byte_len = len as vk::DeviceSize;
        if byte_len > self.byte_capacity {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot hold {} bytes",
                self.byte_capacity, byte_len
            )));
        }
        Ok(byte_len)
    }

    fn descriptor_buffer(&self, len: usize) -> Result<vk::DescriptorBufferInfo, VulkanError> {
        Ok(vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: self.byte_len(len)?,
        })
    }
}

impl Drop for VulkanU32ResidentBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Drop for VulkanResidentBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct VulkanU32ResidentDispatch {
    device: ash::Device,
    memory: vk::DeviceMemory,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    pipeline_key: VulkanPipelineKey,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    capacity: usize,
    byte_capacity: vk::DeviceSize,
    len: usize,
}

pub struct VulkanResidentKernelDispatch {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    pipeline_key: VulkanGenericPipelineKey,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_count: usize,
    workgroup_count_x: u32,
    push_constant_byte_count: u32,
}

pub struct VulkanU32ResidentCopy {
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    len: usize,
    byte_len: vk::DeviceSize,
}

impl VulkanU32ResidentDispatch {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn read(&self, len: usize) -> Result<Vec<u32>, VulkanError> {
        let byte_len = self.byte_len(len)?;
        unsafe { read_u32_memory(&self.device, self.memory, byte_len, len) }
    }

    fn byte_len(&self, len: usize) -> Result<vk::DeviceSize, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident dispatch binding length must not be zero".to_string(),
            ));
        }
        if len > self.len {
            return Err(VulkanError(format!(
                "resident dispatch binding length {} cannot read {} u32 values",
                self.len, len
            )));
        }
        if len > self.capacity {
            return Err(VulkanError(format!(
                "resident dispatch binding capacity {} cannot hold {} u32 values",
                self.capacity, len
            )));
        }
        let byte_len = len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError("resident dispatch byte length overflowed".to_string()))?
            as vk::DeviceSize;
        if byte_len > self.byte_capacity {
            return Err(VulkanError(format!(
                "resident dispatch byte capacity {} cannot hold {} bytes",
                self.byte_capacity, byte_len
            )));
        }
        Ok(byte_len)
    }
}

impl VulkanU32ResidentCopy {
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for VulkanU32ResidentCopy {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl Drop for VulkanU32ResidentDispatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl VulkanResidentKernelDispatch {
    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count
    }

    pub fn workgroup_count_x(&self) -> u32 {
        self.workgroup_count_x
    }

    pub fn push_constant_byte_count(&self) -> u32 {
        self.push_constant_byte_count
    }
}

impl Drop for VulkanResidentKernelDispatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl VulkanComputeDevice {
    pub fn available_compute_devices() -> Result<Vec<VulkanComputeDeviceInfo>, VulkanError> {
        unsafe {
            let entry = Entry::load()
                .map_err(|error| VulkanError(format!("failed to load Vulkan: {error}")))?;
            let instance = create_llmoop_vulkan_instance(&entry)?;
            let physical_devices = match instance.enumerate_physical_devices() {
                Ok(physical_devices) => physical_devices,
                Err(error) => {
                    instance.destroy_instance(None);
                    return Err(VulkanError(format!(
                        "failed to enumerate Vulkan devices: {error:?}"
                    )));
                }
            };
            let selected_index = select_compute_device_index(&instance, &physical_devices);
            let devices = physical_devices
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
            instance.destroy_instance(None);
            Ok(devices)
        }
    }

    pub fn new() -> Result<Self, VulkanError> {
        Self::new_with_physical_device_index(None)
    }

    pub fn new_for_physical_device_index(
        physical_device_index: usize,
    ) -> Result<Self, VulkanError> {
        Self::new_with_physical_device_index(Some(physical_device_index))
    }

    fn new_with_physical_device_index(
        requested_physical_device_index: Option<usize>,
    ) -> Result<Self, VulkanError> {
        unsafe {
            let entry = Entry::load()
                .map_err(|error| VulkanError(format!("failed to load Vulkan: {error}")))?;
            let instance = create_llmoop_vulkan_instance(&entry)?;

            let physical_devices = instance.enumerate_physical_devices().map_err(|error| {
                VulkanError(format!("failed to enumerate Vulkan devices: {error:?}"))
            })?;
            let (physical_device, queue_family_index, device_name) =
                if let Some(physical_device_index) = requested_physical_device_index {
                    select_compute_device_by_index(
                        &instance,
                        &physical_devices,
                        physical_device_index,
                    )?
                } else {
                    select_compute_device(&instance, &physical_devices).ok_or_else(|| {
                        VulkanError("no Vulkan device with a compute queue was found".to_string())
                    })?
                };

            let queue_priorities = [1.0_f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)];
            let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create Vulkan device: {error:?}"))
                })?;
            let queue = device.get_device_queue(queue_family_index, 0);

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,
                device_name,
                u32_storage_pipelines: RefCell::new(HashMap::new()),
                generic_storage_pipelines: RefCell::new(HashMap::new()),
                pipeline_cache_hits: Cell::new(0),
                pipeline_cache_misses: Cell::new(0),
            })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn pipeline_cache_stats(&self) -> VulkanPipelineCacheStats {
        VulkanPipelineCacheStats {
            u32_storage_pipelines: self.u32_storage_pipelines.borrow().len(),
            hits: self.pipeline_cache_hits.get(),
            misses: self.pipeline_cache_misses.get(),
        }
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
        let (buffer, memory, byte_capacity) = self.create_resident_storage_buffer(byte_capacity)?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            byte_capacity,
        })
    }

    pub fn create_u32_resident_buffer(
        &self,
        capacity: usize,
    ) -> Result<VulkanU32ResidentBuffer, VulkanError> {
        if capacity == 0 {
            return Err(VulkanError(
                "resident buffer capacity must not be zero".to_string(),
            ));
        }
        let byte_capacity = capacity
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError("resident buffer byte capacity overflowed".to_string()))?
            as usize;

        let (buffer, memory, byte_capacity) = self.create_resident_storage_buffer(byte_capacity)?;
        Ok(VulkanU32ResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            capacity,
            byte_capacity,
        })
    }

    fn create_resident_storage_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, vk::DeviceSize), VulkanError> {
        let byte_capacity = byte_capacity as vk::DeviceSize;
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(byte_capacity)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
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
                &self.instance,
                self.physical_device,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| {
                VulkanError(
                    "no host-visible coherent memory type for resident storage buffer".to_string(),
                )
            })?;
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
            Ok((buffer, memory, byte_capacity))
        }
    }

    pub fn copy_u32_resident_buffer(
        &self,
        source: &VulkanU32ResidentBuffer,
        destination: &VulkanU32ResidentBuffer,
        len: usize,
    ) -> Result<(), VulkanError> {
        let binding = self.create_u32_resident_copy(source, destination, len)?;
        self.run_u32_resident_copy(&binding, len)
    }

    pub fn create_u32_resident_copy(
        &self,
        source: &VulkanU32ResidentBuffer,
        destination: &VulkanU32ResidentBuffer,
        len: usize,
    ) -> Result<VulkanU32ResidentCopy, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident copy binding length must not be zero".to_string(),
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
                        "failed to create resident copy binding command pool: {error:?}"
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
                        "failed to allocate resident copy binding command buffer: {error:?}"
                    ))
                })?
                .remove(0);

            let command_begin = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(command_buffer, &command_begin)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to begin resident copy binding command buffer: {error:?}"
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
                        "failed to end resident copy binding command buffer: {error:?}"
                    ))
                })?;

            Ok(VulkanU32ResidentCopy {
                device: self.device.clone(),
                command_pool,
                command_buffer,
                len,
                byte_len,
            })
        }
    }

    pub fn run_u32_resident_copy(
        &self,
        binding: &VulkanU32ResidentCopy,
        len: usize,
    ) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident copy length must not be zero".to_string(),
            ));
        }
        if len != binding.len {
            return Err(VulkanError(format!(
                "resident copy binding length {} cannot run {} u32 values",
                binding.len, len
            )));
        }
        let byte_len = len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError("resident copy byte length overflowed".to_string()))?
            as vk::DeviceSize;
        if byte_len != binding.byte_len {
            return Err(VulkanError(format!(
                "resident copy binding byte length {} cannot run {} bytes",
                binding.byte_len, byte_len
            )));
        }

        unsafe {
            let command_buffers = [binding.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| {
                    VulkanError(format!("failed to submit resident copy: {error:?}"))
                })?;
            self.device.queue_wait_idle(self.queue).map_err(|error| {
                VulkanError(format!("failed waiting for resident copy: {error:?}"))
            })?;
            Ok(())
        }
    }

    pub fn install_u32_storage_shader(
        &self,
        spirv_words: &[u32],
        local_size_x: u32,
    ) -> Result<(), VulkanError> {
        if spirv_words.is_empty() {
            return Err(VulkanError("SPIR-V module must not be empty".to_string()));
        }
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }
        self.u32_storage_pipeline(spirv_words, local_size_x)?;
        Ok(())
    }

    pub fn create_u32_resident_dispatch(
        &self,
        spirv_words: &[u32],
        resident_buffer: &VulkanU32ResidentBuffer,
        len: usize,
        local_size_x: u32,
    ) -> Result<VulkanU32ResidentDispatch, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident dispatch binding length must not be zero".to_string(),
            ));
        }
        if spirv_words.is_empty() {
            return Err(VulkanError("SPIR-V module must not be empty".to_string()));
        }
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }
        resident_buffer.byte_len(len)?;

        let pipeline_key = VulkanPipelineKey {
            spirv_words: spirv_words.to_vec(),
            local_size_x,
        };
        let (descriptor_set_layout, pipeline_layout, pipeline) =
            self.u32_storage_pipeline(spirv_words, local_size_x)?;

        unsafe {
            let set_layouts = [descriptor_set_layout];
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            }];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let descriptor_pool = self
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident descriptor pool: {error:?}"
                    ))
                })?;
            let descriptor_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_set = self
                .device
                .allocate_descriptor_sets(&descriptor_alloc_info)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to allocate resident descriptor set: {error:?}"
                    ))
                })?
                .remove(0);
            let descriptor_buffer = [resident_buffer.descriptor_buffer(len)?];
            let descriptor_writes = [vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer)];
            self.device.update_descriptor_sets(&descriptor_writes, &[]);

            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create resident command pool: {error:?}"))
                })?;
            let command_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = self
                .device
                .allocate_command_buffers(&command_alloc_info)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to allocate resident command buffer: {error:?}"
                    ))
                })?
                .remove(0);

            Ok(VulkanU32ResidentDispatch {
                device: self.device.clone(),
                memory: resident_buffer.memory,
                descriptor_pool,
                descriptor_set,
                command_pool,
                command_buffer,
                pipeline_key,
                pipeline_layout,
                pipeline,
                capacity: resident_buffer.capacity,
                byte_capacity: resident_buffer.byte_capacity,
                len,
            })
        }
    }

    pub fn create_resident_kernel_dispatch(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
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
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }

        let mut descriptor_bindings = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            buffer.buffer.byte_len(buffer.byte_len)?;
            if descriptor_bindings.contains(&buffer.binding) {
                return Err(VulkanError(format!(
                    "duplicate storage buffer binding {}",
                    buffer.binding
                )));
            }
            descriptor_bindings.push(buffer.binding);
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
                .map(|buffer| buffer.buffer.descriptor_buffer(buffer.byte_len))
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

            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    self.device.destroy_descriptor_pool(descriptor_pool, None);
                    VulkanError(format!(
                        "failed to create resident kernel command pool: {error:?}"
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
                    self.device.destroy_descriptor_pool(descriptor_pool, None);
                    VulkanError(format!(
                        "failed to allocate resident kernel command buffer: {error:?}"
                    ))
                })?
                .remove(0);

            Ok(VulkanResidentKernelDispatch {
                device: self.device.clone(),
                descriptor_pool,
                descriptor_set,
                command_pool,
                command_buffer,
                pipeline_key,
                pipeline_layout,
                pipeline,
                descriptor_count: buffers.len(),
                workgroup_count_x,
                push_constant_byte_count,
            })
        }
    }

    pub fn run_resident_kernel_dispatch(
        &self,
        binding: &VulkanResidentKernelDispatch,
        push_constants: &[u8],
    ) -> Result<(), VulkanError> {
        if binding.pipeline_key.push_constant_byte_count != push_constants.len() as u32 {
            return Err(VulkanError(format!(
                "resident kernel dispatch expects {} push-constant bytes, got {}",
                binding.pipeline_key.push_constant_byte_count,
                push_constants.len()
            )));
        }

        unsafe {
            self.device
                .reset_command_buffer(binding.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident kernel command buffer: {error:?}"
                    ))
                })?;

            let command_begin = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(binding.command_buffer, &command_begin)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to begin resident kernel command buffer: {error:?}"
                    ))
                })?;
            self.device.cmd_bind_pipeline(
                binding.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                binding.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                binding.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                binding.pipeline_layout,
                0,
                &[binding.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                self.device.cmd_push_constants(
                    binding.command_buffer,
                    binding.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            self.device
                .cmd_dispatch(binding.command_buffer, binding.workgroup_count_x, 1, 1);
            self.device
                .end_command_buffer(binding.command_buffer)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to end resident kernel command buffer: {error:?}"
                    ))
                })?;

            let command_buffers = [binding.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| {
                    VulkanError(format!("failed to submit resident kernel work: {error:?}"))
                })?;
            self.device.queue_wait_idle(self.queue).map_err(|error| {
                VulkanError(format!(
                    "failed waiting for resident kernel work: {error:?}"
                ))
            })?;

            Ok(())
        }
    }

    pub fn run_u32_storage_shader(
        &self,
        spirv_words: &[u32],
        input: &[u32],
        local_size_x: u32,
    ) -> Result<Vec<u32>, VulkanError> {
        if input.is_empty() {
            return Err(VulkanError("input must not be empty".to_string()));
        }
        if spirv_words.is_empty() {
            return Err(VulkanError("SPIR-V module must not be empty".to_string()));
        }
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }

        let resident_buffer = self.create_u32_resident_buffer(input.len())?;
        resident_buffer.write(input)?;
        self.run_u32_storage_shader_on_resident_buffer(
            spirv_words,
            &resident_buffer,
            input.len(),
            local_size_x,
        )?;
        resident_buffer.read(input.len())
    }

    pub fn run_u32_storage_shader_on_resident_buffer(
        &self,
        spirv_words: &[u32],
        resident_buffer: &VulkanU32ResidentBuffer,
        len: usize,
        local_size_x: u32,
    ) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident dispatch length must not be zero".to_string(),
            ));
        }
        if spirv_words.is_empty() {
            return Err(VulkanError("SPIR-V module must not be empty".to_string()));
        }
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }

        let binding =
            self.create_u32_resident_dispatch(spirv_words, resident_buffer, len, local_size_x)?;
        self.run_u32_storage_shader_on_resident_dispatch(spirv_words, &binding, len, local_size_x)
    }

    pub fn run_u32_storage_shader_on_resident_dispatch(
        &self,
        spirv_words: &[u32],
        binding: &VulkanU32ResidentDispatch,
        len: usize,
        local_size_x: u32,
    ) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident dispatch length must not be zero".to_string(),
            ));
        }
        if len > binding.len {
            return Err(VulkanError(format!(
                "resident dispatch binding length {} cannot run {} u32 values",
                binding.len, len
            )));
        }
        if spirv_words.is_empty() {
            return Err(VulkanError("SPIR-V module must not be empty".to_string()));
        }
        if local_size_x == 0 {
            return Err(VulkanError("local_size_x must not be zero".to_string()));
        }
        if binding.pipeline_key.local_size_x != local_size_x
            || binding.pipeline_key.spirv_words != spirv_words
        {
            return Err(VulkanError(
                "resident dispatch binding was created for a different shader program".to_string(),
            ));
        }

        unsafe {
            self.device
                .reset_command_buffer(binding.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident command buffer: {error:?}"
                    ))
                })?;

            let command_begin = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(binding.command_buffer, &command_begin)
                .map_err(|error| {
                    VulkanError(format!("failed to begin command buffer: {error:?}"))
                })?;
            self.device.cmd_bind_pipeline(
                binding.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                binding.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                binding.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                binding.pipeline_layout,
                0,
                &[binding.descriptor_set],
                &[],
            );
            let workgroups = (len as u32).div_ceil(local_size_x);
            self.device
                .cmd_dispatch(binding.command_buffer, workgroups, 1, 1);
            self.device
                .end_command_buffer(binding.command_buffer)
                .map_err(|error| VulkanError(format!("failed to end command buffer: {error:?}")))?;

            let command_buffers = [binding.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| {
                    VulkanError(format!("failed to submit compute work: {error:?}"))
                })?;
            self.device.queue_wait_idle(self.queue).map_err(|error| {
                VulkanError(format!("failed waiting for Vulkan queue: {error:?}"))
            })?;

            Ok(())
        }
    }

    fn u32_storage_pipeline(
        &self,
        spirv_words: &[u32],
        local_size_x: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), VulkanError> {
        let key = VulkanPipelineKey {
            spirv_words: spirv_words.to_vec(),
            local_size_x,
        };
        if let Some(pipeline) = self.u32_storage_pipelines.borrow().get(&key) {
            self.pipeline_cache_hits
                .set(self.pipeline_cache_hits.get() + 1);
            return Ok((
                pipeline.descriptor_set_layout,
                pipeline.pipeline_layout,
                pipeline.pipeline,
            ));
        }

        let pipeline = unsafe { self.create_u32_storage_pipeline(spirv_words)? };
        let handles = (
            pipeline.descriptor_set_layout,
            pipeline.pipeline_layout,
            pipeline.pipeline,
        );
        self.u32_storage_pipelines
            .borrow_mut()
            .insert(key, pipeline);
        self.pipeline_cache_misses
            .set(self.pipeline_cache_misses.get() + 1);
        Ok(handles)
    }

    fn generic_storage_pipeline(
        &self,
        spirv_words: &[u32],
        descriptor_bindings: &[u32],
        push_constant_byte_count: u32,
        local_size_x: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), VulkanError> {
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

    unsafe fn create_u32_storage_pipeline(
        &self,
        spirv_words: &[u32],
    ) -> Result<VulkanU32StoragePipeline, VulkanError> {
        let descriptor_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_binding);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create descriptor set layout: {error:?}"))
                })?
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create pipeline layout: {error:?}"))
                })?
        };

        let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create shader module: {error:?}"))
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
                    VulkanError(format!("failed to create compute pipeline: {error:?}"))
                })?
                .remove(0)
        };

        Ok(VulkanU32StoragePipeline {
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
        })
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

impl Drop for VulkanComputeDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for (_, pipeline) in self.u32_storage_pipelines.get_mut().drain() {
                self.device.destroy_pipeline(pipeline.pipeline, None);
                self.device
                    .destroy_shader_module(pipeline.shader_module, None);
                self.device
                    .destroy_pipeline_layout(pipeline.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            }
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
            self.instance.destroy_instance(None);
        }
    }
}

pub fn run_add_one_shader(
    spirv_words: &[u32],
    input: &[u32],
) -> Result<VulkanSmokeResult, VulkanError> {
    let device = VulkanComputeDevice::new()?;
    let output = device.run_u32_storage_shader(spirv_words, input, 64)?;
    Ok(VulkanSmokeResult {
        device_name: device.device_name().to_string(),
        input: input.to_vec(),
        output,
    })
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

unsafe fn select_compute_device_index(
    instance: &ash::Instance,
    physical_devices: &[vk::PhysicalDevice],
) -> Option<usize> {
    let mut fallback = None;
    for (physical_device_index, physical_device) in physical_devices.iter().enumerate() {
        let properties = unsafe { instance.get_physical_device_properties(*physical_device) };
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
        for family in queue_families {
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    || properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                {
                    return Some(physical_device_index);
                }
                fallback.get_or_insert(physical_device_index);
            }
        }
    }
    fallback
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
        physical_device_id: format!("vulkan:{physical_device_index}"),
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
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
        .iter()
        .enumerate()
        .filter_map(|(index, family)| {
            family
                .queue_flags
                .contains(vk::QueueFlags::COMPUTE)
                .then_some(index as u32)
        })
        .collect()
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

unsafe fn create_llmoop_vulkan_instance(entry: &Entry) -> Result<ash::Instance, VulkanError> {
    let app_name = CString::new("llmoop-runtime").expect("static string has no nul");
    let engine_name = CString::new("llmoop-dsp").expect("static string has no nul");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::API_VERSION_1_1);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    unsafe { entry.create_instance(&instance_info, None) }
        .map_err(|error| VulkanError(format!("failed to create Vulkan instance: {error:?}")))
}

unsafe fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    (0..memory_properties.memory_type_count).find(|index| {
        let supported = (memory_type_bits & (1 << index)) != 0;
        let properties = memory_properties.memory_types[*index as usize].property_flags;
        supported && properties.contains(required_flags)
    })
}

unsafe fn write_u32_memory(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    byte_len: vk::DeviceSize,
    input: &[u32],
) -> Result<(), VulkanError> {
    let ptr = unsafe {
        device
            .map_memory(memory, 0, byte_len, vk::MemoryMapFlags::empty())
            .map_err(|error| VulkanError(format!("failed to map input memory: {error:?}")))?
    };
    let mapped = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u32>(), input.len()) };
    mapped.copy_from_slice(input);
    unsafe { device.unmap_memory(memory) };
    Ok(())
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

unsafe fn read_u32_memory(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    byte_len: vk::DeviceSize,
    len: usize,
) -> Result<Vec<u32>, VulkanError> {
    let ptr = unsafe {
        device
            .map_memory(memory, 0, byte_len, vk::MemoryMapFlags::empty())
            .map_err(|error| VulkanError(format!("failed to map output memory: {error:?}")))?
    };
    let output = unsafe { std::slice::from_raw_parts(ptr.cast::<u32>(), len) }.to_vec();
    unsafe { device.unmap_memory(memory) };
    Ok(output)
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

#[cfg(test)]
pub(crate) fn compile_test_shader_words() -> Option<Vec<u32>> {
    compile_shader_words_from_source("add_one.comp")
}

#[cfg(test)]
pub(crate) fn compile_shader_words_from_source(shader_file: &str) -> Option<Vec<u32>> {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    compile_shader_words_from_source_path(&manifest_dir.join("shaders").join(shader_file))
}

pub(crate) fn compile_shader_words_from_source_path(shader: &Path) -> Option<Vec<u32>> {
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COMPILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    let compile_id = COMPILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let shader_file = shader
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("shader");
    let output = std::env::temp_dir().join(format!(
        "llmoop-{}-{}-{}.spv",
        shader_file.replace(['/', '.'], "-"),
        std::process::id(),
        compile_id
    ));
    let compiled = if test_command_exists("glslangValidator") {
        Command::new("glslangValidator")
            .arg("-V")
            .arg(&shader)
            .arg("-o")
            .arg(&output)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()?
            .success()
    } else if test_command_exists("glslc") {
        Command::new("glslc")
            .arg(&shader)
            .arg("-o")
            .arg(&output)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()?
            .success()
    } else {
        return None;
    };
    if !compiled {
        return None;
    }
    let bytes = std::fs::read(&output).ok()?;
    let _ = std::fs::remove_file(&output);
    if bytes.len() % 4 != 0 {
        return None;
    }
    let words = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(words)
}

#[cfg(test)]
pub(crate) fn compile_test_shader_words_from_source(shader_file: &str) -> Option<Vec<u32>> {
    compile_shader_words_from_source(shader_file)
}

fn test_command_exists(command: &str) -> bool {
    use std::process::{Command, Stdio};

    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn smoke_dispatches_add_one_shader_when_vulkan_is_available() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let input = [1, 2, 41, 255, 1024];
        let result = run_add_one_shader(&spirv_words, &input).unwrap_or_else(|error| {
            panic!("Vulkan smoke failed: {error}");
        });

        assert!(!result.device_name.is_empty());
        assert_eq!(result.input, input);
        assert_eq!(result.output, vec![2, 3, 42, 256, 1025]);
    }

    #[test]
    fn compute_device_reuses_context_for_multiple_dispatches() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };

        assert_eq!(
            device.pipeline_cache_stats(),
            VulkanPipelineCacheStats {
                u32_storage_pipelines: 0,
                hits: 0,
                misses: 0
            }
        );
        let first = device
            .run_u32_storage_shader(&spirv_words, &[7, 8, 9], 64)
            .unwrap();
        assert_eq!(
            device.pipeline_cache_stats(),
            VulkanPipelineCacheStats {
                u32_storage_pipelines: 1,
                hits: 0,
                misses: 1
            }
        );
        let second = device
            .run_u32_storage_shader(&spirv_words, &[40, 41], 64)
            .unwrap();
        assert_eq!(
            device.pipeline_cache_stats(),
            VulkanPipelineCacheStats {
                u32_storage_pipelines: 1,
                hits: 1,
                misses: 1
            }
        );

        assert!(!device.device_name().is_empty());
        assert_eq!(first, vec![8, 9, 10]);
        assert_eq!(second, vec![41, 42]);
    }

    #[test]
    fn installing_shader_warms_pipeline_cache_before_dispatch() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };

        device.install_u32_storage_shader(&spirv_words, 64).unwrap();
        assert_eq!(
            device.pipeline_cache_stats(),
            VulkanPipelineCacheStats {
                u32_storage_pipelines: 1,
                hits: 0,
                misses: 1
            }
        );

        let output = device
            .run_u32_storage_shader(&spirv_words, &[10, 20], 64)
            .unwrap();

        assert_eq!(output, vec![11, 21]);
        assert_eq!(
            device.pipeline_cache_stats(),
            VulkanPipelineCacheStats {
                u32_storage_pipelines: 1,
                hits: 1,
                misses: 1
            }
        );
    }

    #[test]
    fn resident_buffer_can_be_reused_across_dispatches() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_u32_resident_buffer(4).unwrap();

        buffer.write(&[1, 2, 3]).unwrap();
        device
            .run_u32_storage_shader_on_resident_buffer(&spirv_words, &buffer, 3, 64)
            .unwrap();
        assert_eq!(buffer.read(3).unwrap(), vec![2, 3, 4]);

        buffer.write(&[40, 41]).unwrap();
        device
            .run_u32_storage_shader_on_resident_buffer(&spirv_words, &buffer, 2, 64)
            .unwrap();

        assert_eq!(buffer.capacity(), 4);
        assert_eq!(buffer.read(2).unwrap(), vec![41, 42]);
    }

    #[test]
    fn resident_byte_buffer_can_be_reused_for_raw_model_memory() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(16).unwrap();

        buffer.write_bytes(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buffer.byte_capacity(), 16);
        assert_eq!(buffer.read_bytes(5).unwrap(), vec![1, 2, 3, 4, 5]);

        buffer.write_bytes(&[10, 20, 30]).unwrap();
        assert_eq!(buffer.read_bytes(3).unwrap(), vec![10, 20, 30]);
        assert!(buffer.read_bytes(17).is_err());
        assert!(buffer.write_bytes(&[0; 17]).is_err());
    }

    #[test]
    fn generic_resident_kernel_dispatch_runs_on_raw_byte_buffer() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 12);

        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();

        assert_eq!(dispatch.descriptor_count(), 1);
        assert_eq!(dispatch.workgroup_count_x(), 1);
        assert_eq!(dispatch.push_constant_byte_count(), 0);
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
    }

    #[test]
    fn generic_resident_kernel_dispatch_validates_push_constant_size() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(4).unwrap();
        buffer.write_bytes(&u32_bytes(&[10])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 4);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 4)
            .unwrap();

        let error = device
            .run_resident_kernel_dispatch(&dispatch, &[])
            .unwrap_err();

        assert_eq!(
            error,
            VulkanError(
                "resident kernel dispatch expects 4 push-constant bytes, got 0".to_string()
            )
        );
    }

    #[test]
    fn resident_buffers_can_copy_on_device() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_u32_resident_buffer(4).unwrap();
        let destination = device.create_u32_resident_buffer(4).unwrap();
        source.write(&[7, 8, 9]).unwrap();
        destination.write(&[0, 0, 0]).unwrap();

        device
            .copy_u32_resident_buffer(&source, &destination, 3)
            .unwrap();

        assert_eq!(destination.read(3).unwrap(), vec![7, 8, 9]);
    }

    #[test]
    fn resident_copy_binding_can_be_reused() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_u32_resident_buffer(4).unwrap();
        let destination = device.create_u32_resident_buffer(4).unwrap();
        let binding = device
            .create_u32_resident_copy(&source, &destination, 3)
            .unwrap();

        source.write(&[1, 2, 3]).unwrap();
        device.run_u32_resident_copy(&binding, 3).unwrap();
        assert_eq!(destination.read(3).unwrap(), vec![1, 2, 3]);

        source.write(&[10, 20, 30]).unwrap();
        device.run_u32_resident_copy(&binding, 3).unwrap();
        assert_eq!(destination.read(3).unwrap(), vec![10, 20, 30]);
        assert_eq!(binding.len(), 3);
    }

    #[test]
    fn u32_shader_pedal_runs_on_compute_device() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let pedal =
            VulkanU32ShaderPedal::new("pedal_add_one", "u32_token_transform", spirv_words, 64)
                .unwrap();

        let run = pedal.process(&device, &[4, 5, 6]).unwrap();

        assert_eq!(pedal.pedal_id(), "pedal_add_one");
        assert_eq!(pedal.operator_type(), "u32_token_transform");
        assert_eq!(run.pedal_id, "pedal_add_one");
        assert_eq!(run.operator_type, "u32_token_transform");
        assert!(!run.device_name.is_empty());
        assert_eq!(run.input, vec![4, 5, 6]);
        assert_eq!(run.output, vec![5, 6, 7]);
    }

    #[test]
    fn u32_shader_pedal_can_process_a_resident_buffer() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let pedal =
            VulkanU32ShaderPedal::new("pedal_add_one", "u32_token_transform", spirv_words, 64)
                .unwrap();
        let buffer = device.create_u32_resident_buffer(3).unwrap();
        buffer.write(&[8, 9, 10]).unwrap();

        let run = pedal.process_resident(&device, &buffer, 3).unwrap();

        assert_eq!(run.input, vec![8, 9, 10]);
        assert_eq!(run.output, vec![9, 10, 11]);
        assert_eq!(buffer.read(3).unwrap(), vec![9, 10, 11]);
    }
}
