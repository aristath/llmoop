use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::ffi::CString;
use std::fmt::{Display, Formatter};
#[cfg(test)]
use std::path::Path;
use std::time::Instant;

use ash::{Entry, vk};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanError(pub String);

impl Display for VulkanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanError {}

pub struct VulkanComputeDevice {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    device_name: String,
    timestamp_period_ns: f32,
    generic_storage_pipelines: RefCell<HashMap<VulkanGenericPipelineKey, VulkanStoragePipeline>>,
    immediate_kernel_sequence: RefCell<Option<VulkanResidentKernelSequence>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanComputeDeviceInfo {
    pub physical_device_index: usize,
    pub physical_device_id: String,
    pub device_uuid: [u8; vk::UUID_SIZE],
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

pub struct VulkanResidentBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_access: VulkanResidentMemoryAccess,
    byte_capacity: vk::DeviceSize,
    persistent_mapping: Option<usize>,
}

#[derive(Clone)]
struct VulkanResidentMemoryAccess {
    queue: vk::Queue,
    queue_family_index: u32,
    property_flags: vk::MemoryPropertyFlags,
    staging_memory_type_index: Option<u32>,
}

impl VulkanResidentMemoryAccess {
    fn is_directly_mappable(&self) -> bool {
        self.property_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
}

pub struct VulkanResidentKernelBufferBinding<'a> {
    pub binding: u32,
    pub buffer: &'a VulkanResidentBuffer,
    pub byte_offset: usize,
    pub byte_len: usize,
    pub access: VulkanResidentKernelBufferAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentKernelBufferAccess {
    Read,
    Write,
    ReadWrite,
}

impl VulkanResidentKernelBufferAccess {
    fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }

    fn conflicts_with(self, next: Self) -> bool {
        self.writes() || next.writes()
    }

    fn merge(self, other: Self) -> Self {
        match (
            self.reads() || other.reads(),
            self.writes() || other.writes(),
        ) {
            (true, true) => Self::ReadWrite,
            (true, false) => Self::Read,
            (false, true) => Self::Write,
            (false, false) => unreachable!("a resident buffer access must read or write"),
        }
    }
}

impl<'a> VulkanResidentKernelBufferBinding<'a> {
    pub fn new(binding: u32, buffer: &'a VulkanResidentBuffer, byte_len: usize) -> Self {
        Self {
            binding,
            buffer,
            byte_offset: 0,
            byte_len,
            access: VulkanResidentKernelBufferAccess::ReadWrite,
        }
    }

    pub fn with_byte_offset(mut self, byte_offset: usize) -> Self {
        self.byte_offset = byte_offset;
        self
    }

    pub fn with_access(mut self, access: VulkanResidentKernelBufferAccess) -> Self {
        self.access = access;
        self
    }
}

impl VulkanResidentBuffer {
    pub fn persistently_map(&mut self) -> Result<(), VulkanError> {
        if self.persistent_mapping.is_some() {
            return Ok(());
        }
        if !self.memory_access.is_directly_mappable() {
            return Err(VulkanError(
                "resident buffer memory is not host-visible and coherent".to_string(),
            ));
        }
        let pointer = unsafe {
            self.device
                .map_memory(
                    self.memory,
                    0,
                    self.byte_capacity,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to persistently map resident buffer memory: {error:?}"
                    ))
                })?
        };
        self.persistent_mapping = Some(pointer as usize);
        Ok(())
    }

    pub fn byte_capacity(&self) -> usize {
        self.byte_capacity as usize
    }

    pub fn is_persistently_mapped(&self) -> bool {
        self.persistent_mapping.is_some()
    }

    pub fn create_persistently_mapped_copy_to(
        &self,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<VulkanResidentMappedBufferCopy, VulkanError> {
        self.byte_len(len)?;
        destination.byte_len(len)?;
        let source_address = self.persistent_mapping.ok_or_else(|| {
            VulkanError("resident copy source is not persistently mapped".to_string())
        })?;
        let destination_address = destination.persistent_mapping.ok_or_else(|| {
            VulkanError("resident copy destination is not persistently mapped".to_string())
        })?;
        Ok(VulkanResidentMappedBufferCopy {
            source_address,
            destination_address,
            byte_len: len,
        })
    }

    pub fn write_bytes(&self, input: &[u8]) -> Result<(), VulkanError> {
        self.write_bytes_at(0, input)
    }

    pub fn write_bytes_at(&self, offset: usize, input: &[u8]) -> Result<(), VulkanError> {
        if input.is_empty() {
            return Err(VulkanError(
                "resident byte buffer write must not be empty".to_string(),
            ));
        }
        let end = offset
            .checked_add(input.len())
            .ok_or_else(|| VulkanError("resident byte buffer write overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot write {} bytes at offset {}",
                self.byte_capacity,
                input.len(),
                offset
            )));
        }
        let byte_len = input.len() as vk::DeviceSize;
        if let Some(address) = self.persistent_mapping {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    (address as *mut u8).add(offset),
                    input.len(),
                );
            }
            Ok(())
        } else if offset != 0 {
            Err(VulkanError(
                "offset resident buffer writes require persistent mapping".to_string(),
            ))
        } else if self.memory_access.is_directly_mappable() {
            unsafe { write_byte_memory(&self.device, self.memory, byte_len, input) }
        } else {
            unsafe {
                write_device_local_bytes(
                    &self.device,
                    self.buffer,
                    &self.memory_access,
                    byte_len,
                    input,
                )
            }
        }
    }

    pub fn read_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.read_bytes_at(0, len)
    }

    pub fn read_bytes_at(&self, offset: usize, len: usize) -> Result<Vec<u8>, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte buffer length must not be zero".to_string(),
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| VulkanError("resident byte buffer read overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot read {} bytes at offset {}",
                self.byte_capacity, len, offset
            )));
        }
        let byte_len = len as vk::DeviceSize;
        if let Some(address) = self.persistent_mapping {
            Ok(
                unsafe { std::slice::from_raw_parts((address as *const u8).add(offset), len) }
                    .to_vec(),
            )
        } else if offset == 0 && self.memory_access.is_directly_mappable() {
            unsafe { read_byte_memory(&self.device, self.memory, byte_len, len) }
        } else if offset != 0 {
            Err(VulkanError(
                "offset resident buffer reads require persistent mapping".to_string(),
            ))
        } else {
            unsafe {
                read_device_local_bytes(&self.device, self.buffer, &self.memory_access, byte_len)
            }
        }
    }

    pub fn read_persistently_mapped_u32_le_at(&self, offset: usize) -> Result<u32, VulkanError> {
        let byte_count = std::mem::size_of::<u32>();
        let end = offset
            .checked_add(byte_count)
            .ok_or_else(|| VulkanError("resident u32 read overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot read a u32 at offset {}",
                self.byte_capacity, offset
            )));
        }
        let address = self.persistent_mapping.ok_or_else(|| {
            VulkanError("resident u32 read requires persistent mapping".to_string())
        })?;
        let bytes =
            unsafe { std::slice::from_raw_parts((address as *const u8).add(offset), byte_count) };
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
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

    fn byte_range(&self, offset: usize, len: usize) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte buffer range length must not be zero".to_string(),
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| VulkanError("resident byte buffer range overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot address {} bytes at offset {}",
                self.byte_capacity, len, offset
            )));
        }
        Ok(())
    }

    fn descriptor_buffer(
        &self,
        offset: usize,
        len: usize,
    ) -> Result<vk::DescriptorBufferInfo, VulkanError> {
        self.byte_range(offset, len)?;
        Ok(vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: offset as vk::DeviceSize,
            range: len as vk::DeviceSize,
        })
    }
}

impl Drop for VulkanResidentBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.persistent_mapping.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct VulkanResidentKernelDispatch {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_key: VulkanGenericPipelineKey,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_count: usize,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    push_constant_byte_count: u32,
    buffer_accesses: Vec<(vk::Buffer, VulkanResidentKernelBufferAccess)>,
}

/// Owns the Vulkan recording/submission resources for a composed sequence of
/// resident kernel dispatches. Kernel bindings remain independently reusable;
/// this object defines an execution boundary, not a model or pedal boundary.
pub struct VulkanResidentKernelSequence {
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    completion_fence: vk::Fence,
    timestamp_period_ns: f32,
    recorded_steps: RefCell<Option<Vec<VulkanResidentKernelRecordedStep>>>,
    recorded_snapshot_copies: RefCell<Option<Vec<VulkanResidentKernelRecordedSnapshotCopy>>>,
}

#[derive(Clone, PartialEq, Eq)]
struct VulkanResidentKernelRecordedStep {
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    push_constants: Vec<u8>,
}

#[derive(Clone, PartialEq, Eq)]
struct VulkanResidentKernelRecordedSnapshotCopy {
    after_step_index: usize,
    source: vk::Buffer,
    destination: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

#[derive(Clone, Copy)]
pub struct VulkanResidentKernelSequenceStep<'a> {
    dispatch: &'a VulkanResidentKernelDispatch,
    push_constants: &'a [u8],
}

impl<'a> VulkanResidentKernelSequenceStep<'a> {
    pub fn new(dispatch: &'a VulkanResidentKernelDispatch, push_constants: &'a [u8]) -> Self {
        Self {
            dispatch,
            push_constants,
        }
    }
}

#[derive(Clone, Copy)]
pub struct VulkanResidentKernelSequenceSnapshotCopy<'a> {
    pub after_step_index: usize,
    source: &'a VulkanResidentBuffer,
    destination: &'a VulkanResidentBuffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

impl<'a> VulkanResidentKernelSequenceSnapshotCopy<'a> {
    pub fn new(
        after_step_index: usize,
        source: &'a VulkanResidentBuffer,
        destination: &'a VulkanResidentBuffer,
        source_offset: usize,
        destination_offset: usize,
        byte_len: usize,
    ) -> Result<Self, VulkanError> {
        if byte_len == 0 {
            return Err(VulkanError(
                "resident kernel sequence snapshot length must not be zero".to_string(),
            ));
        }
        let source_end = source_offset
            .checked_add(byte_len)
            .ok_or_else(|| VulkanError("resident snapshot source range overflowed".to_string()))?;
        let destination_end = destination_offset.checked_add(byte_len).ok_or_else(|| {
            VulkanError("resident snapshot destination range overflowed".to_string())
        })?;
        if source_end > source.byte_capacity() {
            return Err(VulkanError(format!(
                "resident snapshot source capacity {} cannot copy {} bytes at offset {}",
                source.byte_capacity(),
                byte_len,
                source_offset
            )));
        }
        if destination_end > destination.byte_capacity() {
            return Err(VulkanError(format!(
                "resident snapshot destination capacity {} cannot copy {} bytes at offset {}",
                destination.byte_capacity(),
                byte_len,
                destination_offset
            )));
        }
        Ok(Self {
            after_step_index,
            source,
            destination,
            source_offset: source_offset as vk::DeviceSize,
            destination_offset: destination_offset as vk::DeviceSize,
            byte_len: byte_len as vk::DeviceSize,
        })
    }

    fn recorded(self) -> VulkanResidentKernelRecordedSnapshotCopy {
        VulkanResidentKernelRecordedSnapshotCopy {
            after_step_index: self.after_step_index,
            source: self.source.buffer,
            destination: self.destination.buffer,
            source_offset: self.source_offset,
            destination_offset: self.destination_offset,
            byte_len: self.byte_len,
        }
    }
}

fn print_resident_kernel_timestamp_summary(
    steps: &[VulkanResidentKernelSequenceStep<'_>],
    timestamps: &[u64],
    timestamp_period_ns: f32,
    host_elapsed_ns: u128,
) {
    if timestamps.len() != steps.len() + 1 {
        eprintln!(
            "llmoop Vulkan timings unavailable: expected {} timestamps, received {}",
            steps.len() + 1,
            timestamps.len()
        );
        return;
    }

    let mut groups = HashMap::<(u32, u32, usize, u32), (usize, f64)>::new();
    let mut intervals = Vec::with_capacity(steps.len());
    for (step_index, step) in steps.iter().enumerate() {
        let elapsed_ticks = timestamps[step_index + 1].saturating_sub(timestamps[step_index]);
        let elapsed_ns = elapsed_ticks as f64 * f64::from(timestamp_period_ns);
        let key = (
            step.dispatch.workgroup_count_x,
            step.dispatch.workgroup_count_y,
            step.dispatch.descriptor_count,
            step.dispatch.push_constant_byte_count,
        );
        let group = groups.entry(key).or_insert((0, 0.0));
        group.0 += 1;
        group.1 += elapsed_ns;
        intervals.push((step_index, key, elapsed_ns));
    }

    let total_ns = timestamps
        .last()
        .copied()
        .unwrap_or_default()
        .saturating_sub(timestamps[0]) as f64
        * f64::from(timestamp_period_ns);
    eprintln!(
        "llmoop Vulkan timings: steps={}, gpu_total_ms={:.3}, host_submit_wait_ms={:.3}, host_minus_gpu_ms={:.3}",
        steps.len(),
        total_ns / 1_000_000.0,
        host_elapsed_ns as f64 / 1_000_000.0,
        (host_elapsed_ns as f64 - total_ns).max(0.0) / 1_000_000.0,
    );

    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by(|left, right| {
        right
            .1
            .1
            .partial_cmp(&left.1.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("  grouped step intervals (dispatch plus preceding dependency):");
    for ((workgroups_x, workgroups_y, descriptors, push_bytes), (count, elapsed_ns)) in groups {
        eprintln!(
            "    workgroups={workgroups_x}x{workgroups_y:<5} descriptors={descriptors:<2} push_bytes={push_bytes:<3} count={count:<3} total_us={:.3} avg_us={:.3}",
            elapsed_ns / 1_000.0,
            elapsed_ns / count as f64 / 1_000.0,
        );
    }

    intervals.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("  slowest step intervals:");
    for (step_index, (workgroups_x, workgroups_y, descriptors, push_bytes), elapsed_ns) in
        intervals.into_iter().take(12)
    {
        eprintln!(
            "    step={step_index:<3} workgroups={workgroups_x}x{workgroups_y:<5} descriptors={descriptors:<2} push_bytes={push_bytes:<3} elapsed_us={:.3}",
            elapsed_ns / 1_000.0,
        );
    }
}

pub struct VulkanResidentBufferCopy {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
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
        if byte_len == 0 {
            return Err(VulkanError(
                "resident buffer range copy length must not be zero".to_string(),
            ));
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
            self.device.queue_wait_idle(self.queue).map_err(|error| {
                VulkanError(format!("failed waiting for resident byte copy: {error:?}"))
            })?;
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
            self.device
                .wait_for_fences(&[self.completion_fence], true, u64::MAX)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed waiting for resident buffer copy batch: {error:?}"
                    ))
                })?;
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
    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count
    }

    pub fn workgroup_count_x(&self) -> u32 {
        self.workgroup_count_x
    }

    pub fn workgroup_count_y(&self) -> u32 {
        self.workgroup_count_y
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

impl VulkanResidentKernelSequence {
    pub fn has_recorded_commands(&self) -> bool {
        self.recorded_steps.borrow().is_some() && self.recorded_snapshot_copies.borrow().is_some()
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
        unsafe {
            let entry = Entry::load()
                .map_err(|error| VulkanError(format!("failed to load Vulkan: {error}")))?;
            let instance = create_llmoop_vulkan_instance(&entry)?;

            let physical_devices = instance.enumerate_physical_devices().map_err(|error| {
                VulkanError(format!("failed to enumerate Vulkan devices: {error:?}"))
            })?;
            let (physical_device, queue_family_index, device_name) = if let Some(device_uuid) =
                requested_device_uuid
            {
                select_compute_device_by_uuid(&instance, &physical_devices, device_uuid)?
            } else if let Some(physical_device_index) = requested_physical_device_index {
                select_compute_device_by_index(&instance, &physical_devices, physical_device_index)?
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
            let timestamp_period_ns = instance
                .get_physical_device_properties(physical_device)
                .limits
                .timestamp_period;

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,
                device_name,
                timestamp_period_ns,
                generic_storage_pipelines: RefCell::new(HashMap::new()),
                immediate_kernel_sequence: RefCell::new(None),
            })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
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
                required_memory_flags,
                preferred_memory_flags,
            )
            .ok_or_else(|| {
                VulkanError(format!(
                    "no memory type with required flags {required_memory_flags:?} for resident storage buffer"
                ))
            })?;
            let memory_properties = self
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
                        &self.instance,
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

    pub fn create_resident_kernel_dispatch(
        &self,
        spirv_words: &[u32],
        buffers: &[VulkanResidentKernelBufferBinding<'_>],
        workgroup_count_x: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
    ) -> Result<VulkanResidentKernelDispatch, VulkanError> {
        self.create_resident_kernel_dispatch_2d(
            spirv_words,
            buffers,
            workgroup_count_x,
            1,
            local_size_x,
            push_constant_byte_count,
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
            Vec::<(vk::Buffer, VulkanResidentKernelBufferAccess)>::with_capacity(buffers.len());
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
            if let Some((_, access)) = buffer_accesses
                .iter_mut()
                .find(|(resident_buffer, _)| *resident_buffer == buffer.buffer.buffer)
            {
                *access = access.merge(buffer.access);
            } else {
                buffer_accesses.push((buffer.buffer.buffer, buffer.access));
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
                push_constant_byte_count,
                buffer_accesses,
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
                recorded_steps: RefCell::new(None),
                recorded_snapshot_copies: RefCell::new(None),
            })
        }
    }

    pub fn run_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.run_resident_kernel_sequence_with_snapshot_copies(sequence, steps, &[])
    }

    pub fn run_recorded_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_recorded_resident_kernel_sequence(sequence)?;
        self.wait_resident_kernel_sequence(sequence)
    }

    pub fn submit_recorded_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        self.submit_resident_kernel_sequence(sequence)
    }

    fn submit_resident_kernel_sequence_and_wait(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_resident_kernel_sequence(sequence)?;
        self.wait_resident_kernel_sequence(sequence)
    }

    fn submit_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        unsafe {
            let command_buffers = [sequence.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .reset_fences(&[sequence.completion_fence])
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident kernel sequence completion fence: {error:?}"
                    ))
                })?;
            self.device
                .queue_submit(self.queue, &submit_info, sequence.completion_fence)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to submit resident kernel sequence: {error:?}"
                    ))
                })?;
        }
        Ok(())
    }

    pub fn wait_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .wait_for_fences(&[sequence.completion_fence], true, u64::MAX)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed waiting for resident kernel sequence: {error:?}"
                    ))
                })?;
        }
        Ok(())
    }

    pub fn run_resident_kernel_sequence_with_snapshot_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
    ) -> Result<(), VulkanError> {
        if steps.is_empty() {
            return Err(VulkanError(
                "resident kernel sequence must contain at least one dispatch".to_string(),
            ));
        }
        for (step_index, step) in steps.iter().enumerate() {
            if step.dispatch.pipeline_key.push_constant_byte_count
                != step.push_constants.len() as u32
            {
                return Err(VulkanError(format!(
                    "resident kernel sequence step {step_index} expects {} push-constant bytes, got {}",
                    step.dispatch.pipeline_key.push_constant_byte_count,
                    step.push_constants.len()
                )));
            }
        }
        if let Some(copy) = snapshot_copies
            .iter()
            .find(|copy| copy.after_step_index >= steps.len())
        {
            return Err(VulkanError(format!(
                "resident snapshot follows step {}, but sequence contains {} steps",
                copy.after_step_index,
                steps.len()
            )));
        }

        unsafe {
            let profiling_enabled = std::env::var_os("LLMOOP_VK_PERF_LOGGER").is_some();
            let command_buffer_matches = !profiling_enabled
                && sequence
                    .recorded_steps
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == steps.len()
                            && recorded.iter().zip(steps).all(|(recorded, step)| {
                                recorded.pipeline == step.dispatch.pipeline
                                    && recorded.descriptor_set == step.dispatch.descriptor_set
                                    && recorded.workgroup_count_x == step.dispatch.workgroup_count_x
                                    && recorded.workgroup_count_y == step.dispatch.workgroup_count_y
                                    && recorded.push_constants == step.push_constants
                            })
                    })
                && sequence
                    .recorded_snapshot_copies
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == snapshot_copies.len()
                            && recorded
                                .iter()
                                .zip(snapshot_copies)
                                .all(|(recorded, copy)| *recorded == copy.recorded())
                    });
            let host_start = profiling_enabled.then(Instant::now);
            let query_count = u32::try_from(steps.len() + 1).map_err(|_| {
                VulkanError("resident kernel timestamp count overflowed".to_string())
            })?;
            let query_pool = if profiling_enabled {
                let query_pool_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(query_count);
                Some(
                    self.device
                        .create_query_pool(&query_pool_info, None)
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to create resident kernel timestamp pool: {error:?}"
                            ))
                        })?,
                )
            } else {
                None
            };

            if !command_buffer_matches {
                self.device
                    .reset_command_buffer(
                        sequence.command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to reset resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;

                let command_begin = vk::CommandBufferBeginInfo::default();
                self.device
                    .begin_command_buffer(sequence.command_buffer, &command_begin)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to begin resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;
            }

            if !command_buffer_matches && let Some(query_pool) = query_pool {
                self.device.cmd_reset_query_pool(
                    sequence.command_buffer,
                    query_pool,
                    0,
                    query_count,
                );
                self.device.cmd_write_timestamp(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }

            if !command_buffer_matches {
                let host_write_barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)];
                self.device.cmd_pipeline_barrier(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &host_write_barrier,
                    &[],
                    &[],
                );
            }

            let mut pending_buffer_accesses =
                Vec::<(vk::Buffer, VulkanResidentKernelBufferAccess)>::new();
            if !command_buffer_matches {
                for (step_index, step) in steps.iter().enumerate() {
                    let has_buffer_hazard = step.dispatch.buffer_accesses.iter().any(
                        |(current_buffer, current_access)| {
                            pending_buffer_accesses.iter().any(
                                |(pending_buffer, pending_access)| {
                                    pending_buffer == current_buffer
                                        && pending_access.conflicts_with(*current_access)
                                },
                            )
                        },
                    );
                    if has_buffer_hazard {
                        let memory_barrier = [vk::MemoryBarrier::default()
                            .src_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &memory_barrier,
                            &[],
                            &[],
                        );
                        pending_buffer_accesses.clear();
                    }

                    self.device.cmd_bind_pipeline(
                        sequence.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        step.dispatch.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        sequence.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        step.dispatch.pipeline_layout,
                        0,
                        &[step.dispatch.descriptor_set],
                        &[],
                    );
                    if !step.push_constants.is_empty() {
                        self.device.cmd_push_constants(
                            sequence.command_buffer,
                            step.dispatch.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            step.push_constants,
                        );
                    }
                    self.device.cmd_dispatch(
                        sequence.command_buffer,
                        step.dispatch.workgroup_count_x,
                        step.dispatch.workgroup_count_y,
                        1,
                    );
                    for (current_buffer, current_access) in &step.dispatch.buffer_accesses {
                        if let Some((_, pending_access)) = pending_buffer_accesses
                            .iter_mut()
                            .find(|(pending_buffer, _)| pending_buffer == current_buffer)
                        {
                            *pending_access = pending_access.merge(*current_access);
                        } else {
                            pending_buffer_accesses.push((*current_buffer, *current_access));
                        }
                    }
                    if let Some(query_pool) = query_pool {
                        self.device.cmd_write_timestamp(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query_pool,
                            u32::try_from(step_index + 1).map_err(|_| {
                                VulkanError(
                                    "resident kernel timestamp index overflowed".to_string(),
                                )
                            })?,
                        );
                    }

                    let step_snapshot_copies = snapshot_copies
                        .iter()
                        .filter(|copy| copy.after_step_index == step_index)
                        .collect::<Vec<_>>();
                    if !step_snapshot_copies.is_empty() {
                        let compute_to_transfer = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &compute_to_transfer,
                            &[],
                            &[],
                        );
                        for copy in step_snapshot_copies {
                            let regions = [vk::BufferCopy {
                                src_offset: copy.source_offset,
                                dst_offset: copy.destination_offset,
                                size: copy.byte_len,
                            }];
                            self.device.cmd_copy_buffer(
                                sequence.command_buffer,
                                copy.source.buffer,
                                copy.destination.buffer,
                                &regions,
                            );
                        }
                        let transfer_to_compute = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &transfer_to_compute,
                            &[],
                            &[],
                        );
                        pending_buffer_accesses.clear();
                    }
                }

                let host_visibility_barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                self.device.cmd_pipeline_barrier(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &host_visibility_barrier,
                    &[],
                    &[],
                );

                self.device
                    .end_command_buffer(sequence.command_buffer)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to end resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;

                if profiling_enabled {
                    *sequence.recorded_steps.borrow_mut() = None;
                    *sequence.recorded_snapshot_copies.borrow_mut() = None;
                } else {
                    *sequence.recorded_steps.borrow_mut() = Some(
                        steps
                            .iter()
                            .map(|step| VulkanResidentKernelRecordedStep {
                                pipeline: step.dispatch.pipeline,
                                descriptor_set: step.dispatch.descriptor_set,
                                workgroup_count_x: step.dispatch.workgroup_count_x,
                                workgroup_count_y: step.dispatch.workgroup_count_y,
                                push_constants: step.push_constants.to_vec(),
                            })
                            .collect(),
                    );
                    *sequence.recorded_snapshot_copies.borrow_mut() = Some(
                        snapshot_copies
                            .iter()
                            .copied()
                            .map(VulkanResidentKernelSequenceSnapshotCopy::recorded)
                            .collect(),
                    );
                }
            }

            self.submit_resident_kernel_sequence_and_wait(sequence)?;
            let host_submit_wait_ns = host_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();

            if let Some(query_pool) = query_pool {
                let mut timestamps = vec![0u64; query_count as usize];
                let result = self.device.get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                );
                self.device.destroy_query_pool(query_pool, None);
                result.map_err(|error| {
                    VulkanError(format!(
                        "failed to read resident kernel timestamps: {error:?}"
                    ))
                })?;
                print_resident_kernel_timestamp_summary(
                    steps,
                    &timestamps,
                    sequence.timestamp_period_ns,
                    host_submit_wait_ns,
                );
            }

            Ok(())
        }
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
            self.instance.destroy_instance(None);
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
            (heap_size, preferred_property_count)
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

#[cfg(test)]
pub(crate) fn compile_test_shader_words() -> Option<Vec<u32>> {
    use std::sync::atomic::{AtomicU64, Ordering};

    const SOURCE: &str = r#"#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint values[];
} data;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < data.values.length()) {
        data.values[index] = data.values[index] + 1;
    }
}
"#;

    static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let source_id = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let source_path = std::env::temp_dir().join(format!(
        "llmoop-test-increment-{}-{source_id}.comp",
        std::process::id()
    ));
    std::fs::write(&source_path, SOURCE).ok()?;
    let words = compile_shader_words_from_source_path(&source_path);
    let _ = std::fs::remove_file(source_path);
    words
}

#[cfg(test)]
pub(crate) fn compile_shader_words_from_source(shader_file: &str) -> Option<Vec<u32>> {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_path = manifest_dir.join("shaders").join(shader_file);
    if shader_path.exists() {
        return compile_shader_words_from_source_path(&shader_path);
    }

    let shape = shader_file
        .strip_prefix("linear_bf16_")?
        .strip_suffix(".comp")?;
    let (input_size, output_size) = shape.split_once('x')?;
    if !input_size.bytes().all(|byte| byte.is_ascii_digit())
        || !output_size.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }

    let template = std::fs::read_to_string(
        manifest_dir
            .join("shaders")
            .join("linear_bf16.comp.template"),
    )
    .ok()?;
    let rendered = template
        .replace("{{INPUT_SIZE}}", input_size)
        .replace("{{OUTPUT_SIZE}}", output_size);
    static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let source_id = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let rendered_path = std::env::temp_dir().join(format!(
        "llmoop-linear-{input_size}x{output_size}-{}-{source_id}.comp",
        std::process::id()
    ));
    std::fs::write(&rendered_path, rendered).ok()?;
    let words = compile_shader_words_from_source_path(&rendered_path);
    let _ = std::fs::remove_file(rendered_path);
    words
}

#[cfg(test)]
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
            .arg("--target-env")
            .arg("vulkan1.4")
            .arg(shader)
            .arg("-o")
            .arg(&output)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()?
            .success()
    } else if test_command_exists("glslc") {
        Command::new("glslc")
            .arg("--target-env=vulkan1.4")
            .arg(shader)
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

#[cfg(test)]
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
    fn persistently_mapped_copy_moves_exact_bound_bytes() {
        let source = [1u8, 2, 3, 4, 5, 6];
        let mut destination = [0u8; 6];
        let copy = VulkanResidentMappedBufferCopy {
            source_address: source.as_ptr() as usize,
            destination_address: destination.as_mut_ptr() as usize,
            byte_len: source.len(),
        };

        copy.run(source.len()).unwrap();

        assert_eq!(destination, source);
        assert!(copy.run(source.len() - 1).is_err());
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
    fn resident_kernel_sequence_records_and_replays_composed_dispatches() {
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
        let sequence = device.create_resident_kernel_sequence().unwrap();
        assert!(!sequence.has_recorded_commands());
        assert!(
            device
                .run_recorded_resident_kernel_sequence(&sequence)
                .is_err()
        );

        device
            .run_resident_kernel_sequence(
                &sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                ],
            )
            .unwrap();
        assert!(sequence.has_recorded_commands());

        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );

        device
            .run_recorded_resident_kernel_sequence(&sequence)
            .unwrap();
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![5, 6, 45]
        );
    }

    #[test]
    fn resident_kernel_sequence_snapshots_state_between_dispatch_groups() {
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
        let state = device.create_resident_buffer(12).unwrap();
        state.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let snapshots = device.create_host_visible_resident_buffer(24).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &state, 12);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        let steps = [
            VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
            VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
        ];
        let copies = [
            VulkanResidentKernelSequenceSnapshotCopy::new(0, &state, &snapshots, 0, 0, 12).unwrap(),
            VulkanResidentKernelSequenceSnapshotCopy::new(1, &state, &snapshots, 0, 12, 12)
                .unwrap(),
        ];

        device
            .run_resident_kernel_sequence_with_snapshot_copies(&sequence, &steps, &copies)
            .unwrap();

        assert_eq!(
            bytes_to_u32(&snapshots.read_bytes(24).unwrap()),
            vec![2, 3, 42, 3, 4, 43]
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
                "resident kernel sequence step 0 expects 4 push-constant bytes, got 0".to_string()
            )
        );
    }

    #[test]
    fn resident_byte_buffers_can_copy_on_device() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_resident_buffer(8).unwrap();
        let destination = device.create_resident_buffer(8).unwrap();
        source.write_bytes(&[1, 2, 3, 4, 5, 6]).unwrap();
        destination.write_bytes(&[0, 0, 0, 0, 0, 0]).unwrap();

        device
            .copy_resident_buffer_bytes(&source, &destination, 6)
            .unwrap();

        assert_eq!(destination.read_bytes(6).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn resident_byte_copy_binding_can_be_reused() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_resident_buffer(8).unwrap();
        let destination = device.create_resident_buffer(8).unwrap();
        let binding = device
            .create_resident_buffer_copy(&source, &destination, 6)
            .unwrap();

        source.write_bytes(&[1, 2, 3, 4, 5, 6]).unwrap();
        device.run_resident_buffer_copy(&binding, 6).unwrap();
        assert_eq!(destination.read_bytes(6).unwrap(), vec![1, 2, 3, 4, 5, 6]);

        source.write_bytes(&[10, 20, 30, 40, 50, 60]).unwrap();
        device.run_resident_buffer_copy(&binding, 6).unwrap();
        assert_eq!(
            destination.read_bytes(6).unwrap(),
            vec![10, 20, 30, 40, 50, 60]
        );
        assert_eq!(binding.byte_len(), 6);
    }
}
