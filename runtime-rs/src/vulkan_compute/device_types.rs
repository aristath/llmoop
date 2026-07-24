pub struct VulkanComputeDevice {
    context: Arc<VulkanInstanceContext>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    device_name: String,
    enabled_device_extensions: BTreeSet<String>,
    enabled_shader_features: BTreeSet<VulkanShaderFeature>,
    shared_host_memory_alignment: Option<usize>,
    opaque_fd_timeline_semaphore_supported: bool,
    cooperative_bfloat16_shapes: BTreeSet<(u32, u32, u32)>,
    subgroup_size: u32,
    subgroup_supported_stages: vk::ShaderStageFlags,
    subgroup_supported_operations: vk::SubgroupFeatureFlags,
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size_x: u32,
    min_storage_buffer_offset_alignment: usize,
    device_local_memory_bytes: u64,
    timestamp_period_ns: f32,
    generic_storage_pipelines: RefCell<HashMap<VulkanGenericPipelineKey, VulkanStoragePipeline>>,
    immediate_kernel_sequence: RefCell<Option<VulkanResidentKernelSequence>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanResidentExecutionCounters {
    pub resident_sequence_prepare_calls: u64,
    pub resident_sequence_recorded_command_buffers: u64,
    pub resident_sequence_reused_command_buffers: u64,
    pub resident_sequence_queue_submits: u64,
    pub resident_sequence_fence_waits: u64,
    pub resident_queue_batch_submits: u64,
    pub resident_queue_batch_commands: u64,
    pub resident_copy_queue_submits: u64,
    pub resident_copy_waits: u64,
}

static RESIDENT_SEQUENCE_PREPARE_CALLS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_QUEUE_SUBMITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_FENCE_WAITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_QUEUE_BATCH_SUBMITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_QUEUE_BATCH_COMMANDS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COPY_QUEUE_SUBMITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COPY_WAITS: AtomicU64 = AtomicU64::new(0);

pub fn reset_vulkan_resident_execution_counters() {
    RESIDENT_SEQUENCE_PREPARE_CALLS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_QUEUE_SUBMITS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_FENCE_WAITS.store(0, Ordering::Relaxed);
    RESIDENT_QUEUE_BATCH_SUBMITS.store(0, Ordering::Relaxed);
    RESIDENT_QUEUE_BATCH_COMMANDS.store(0, Ordering::Relaxed);
    RESIDENT_COPY_QUEUE_SUBMITS.store(0, Ordering::Relaxed);
    RESIDENT_COPY_WAITS.store(0, Ordering::Relaxed);
}

pub fn vulkan_resident_execution_counters() -> VulkanResidentExecutionCounters {
    VulkanResidentExecutionCounters {
        resident_sequence_prepare_calls: RESIDENT_SEQUENCE_PREPARE_CALLS.load(Ordering::Relaxed),
        resident_sequence_recorded_command_buffers: RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS
            .load(Ordering::Relaxed),
        resident_sequence_reused_command_buffers: RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS
            .load(Ordering::Relaxed),
        resident_sequence_queue_submits: RESIDENT_SEQUENCE_QUEUE_SUBMITS.load(Ordering::Relaxed),
        resident_sequence_fence_waits: RESIDENT_SEQUENCE_FENCE_WAITS.load(Ordering::Relaxed),
        resident_queue_batch_submits: RESIDENT_QUEUE_BATCH_SUBMITS.load(Ordering::Relaxed),
        resident_queue_batch_commands: RESIDENT_QUEUE_BATCH_COMMANDS.load(Ordering::Relaxed),
        resident_copy_queue_submits: RESIDENT_COPY_QUEUE_SUBMITS.load(Ordering::Relaxed),
        resident_copy_waits: RESIDENT_COPY_WAITS.load(Ordering::Relaxed),
    }
}

struct VulkanInstanceContext {
    _entry: Entry,
    instance: ash::Instance,
}

impl Drop for VulkanInstanceContext {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

pub struct VulkanComputeDeviceCatalog {
    context: Arc<VulkanInstanceContext>,
    physical_devices: Vec<vk::PhysicalDevice>,
    available_devices: Vec<VulkanComputeDeviceInfo>,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanComputeTargetCapabilities {
    pub physical_device_index: usize,
    pub physical_device_id: String,
    pub device_name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub shader_features: BTreeSet<VulkanShaderFeature>,
    pub subgroup_operations: BTreeSet<VulkanSubgroupOperation>,
    pub subgroup_compute_supported: bool,
    pub subgroup_size: u32,
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size_x: u32,
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
    persistent_mapping_requires_unmap: bool,
    _shared_host_allocation: Option<Arc<VulkanSharedHostAllocation>>,
}

/// Page-aligned host memory imported into multiple Vulkan devices. GPUs access
/// the same bytes directly; the host does not relay activation data.
pub struct VulkanSharedHostAllocation {
    address: usize,
    layout: Layout,
    byte_capacity: usize,
}

pub struct VulkanTimelineSemaphore {
    device: ash::Device,
    device_handle: vk::Device,
    semaphore: vk::Semaphore,
    opaque_fd_exportable: bool,
    permanent_opaque_fd_imported: Cell<bool>,
}

#[derive(Clone, Copy)]
pub struct VulkanTimelineSemaphorePoint<'a> {
    semaphore: &'a VulkanTimelineSemaphore,
    value: u64,
}

impl<'a> VulkanTimelineSemaphorePoint<'a> {
    pub fn new(semaphore: &'a VulkanTimelineSemaphore, value: u64) -> Self {
        Self { semaphore, value }
    }
}

/// Collects already-recorded resident command buffers by logical device.
/// Timeline waits and signals remain attached to their original command, so a
/// caller can enqueue a complete cross-device DAG before issuing one
/// `vkQueueSubmit2` call per participating queue.
pub struct VulkanResidentQueueSubmissionBatch<'a> {
    groups: RefCell<Vec<VulkanResidentQueueSubmissionGroup<'a>>>,
}

/// A mounted queue-submission topology. Command buffers, queue ordering, and
/// semaphore edges stay fixed; replay only advances timeline values.
pub struct VulkanResidentQueueSubmissionTemplate<'a> {
    groups: Vec<VulkanResidentQueueSubmissionGroup<'a>>,
    submission_count: usize,
}

struct VulkanResidentQueueSubmissionGroup<'a> {
    device: &'a VulkanComputeDevice,
    submissions: Vec<VulkanPreparedResidentQueueSubmission>,
}

struct VulkanPreparedResidentQueueSubmission {
    command_buffer: vk::CommandBuffer,
    wait_points: Vec<(vk::Semaphore, u64)>,
    signal_points: Vec<(vk::Semaphore, u64)>,
    completion_fence: Option<vk::Fence>,
}

impl Default for VulkanResidentQueueSubmissionBatch<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> VulkanResidentQueueSubmissionBatch<'a> {
    pub fn new() -> Self {
        Self {
            groups: RefCell::new(Vec::new()),
        }
    }

    pub fn enqueue_recorded_sequence(
        &self,
        device: &'a VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        if sequence.device.handle() != device.device.handle() {
            return Err(VulkanError(
                "resident queue submission sequence belongs to another logical device".to_string(),
            ));
        }
        for point in wait_points.iter().chain(signal_points) {
            device.validate_local_timeline_semaphore(point.semaphore)?;
        }
        let submission = VulkanPreparedResidentQueueSubmission {
            command_buffer: sequence.command_buffer,
            wait_points: wait_points
                .iter()
                .map(|point| (point.semaphore.semaphore, point.value))
                .collect(),
            signal_points: signal_points
                .iter()
                .map(|point| (point.semaphore.semaphore, point.value))
                .collect(),
            completion_fence: signal_completion.then_some(sequence.completion_fence),
        };
        let mut groups = self.groups.borrow_mut();
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.device.shares_logical_device_with(device))
        {
            group.submissions.push(submission);
        } else {
            groups.push(VulkanResidentQueueSubmissionGroup {
                device,
                submissions: vec![submission],
            });
        }
        Ok(())
    }

    pub fn pending_submission_count(&self) -> usize {
        self.groups
            .borrow()
            .iter()
            .map(|group| group.submissions.len())
            .sum()
    }

    pub fn mount(self) -> Result<VulkanResidentQueueSubmissionTemplate<'a>, VulkanError> {
        let groups = self.groups.into_inner();
        let submission_count = groups.iter().try_fold(0usize, |total, group| {
            total.checked_add(group.submissions.len()).ok_or_else(|| {
                VulkanError("resident queue submission count overflowed".to_string())
            })
        })?;
        Ok(VulkanResidentQueueSubmissionTemplate {
            groups,
            submission_count,
        })
    }
}

impl VulkanResidentQueueSubmissionTemplate<'_> {
    pub fn submission_count(&self) -> usize {
        self.submission_count
    }

    pub fn submit_with_timeline_value_offset(
        &self,
        timeline_value_offset: u64,
    ) -> Result<usize, VulkanError> {
        for group in &self.groups {
            for submission in &group.submissions {
                for (_, value) in submission
                    .wait_points
                    .iter()
                    .chain(&submission.signal_points)
                {
                    offset_timeline_value(*value, timeline_value_offset)?;
                }
            }
        }
        for group in &self.groups {
            group
                .device
                .submit_prepared_resident_queue_batch(&group.submissions, timeline_value_offset)?;
        }
        Ok(self.submission_count)
    }
}

fn offset_timeline_value(value: u64, offset: u64) -> Result<u64, VulkanError> {
    value.checked_add(offset).ok_or_else(|| {
        VulkanError(format!(
            "timeline semaphore value {value} overflows with replay offset {offset}"
        ))
    })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentKernelBufferAccessRecord {
    // A descriptor may expose a byte range, but the compiled shader contract does not yet
    // prove that every physical access stays inside that logical range. Keep synchronization
    // at the Vulkan-buffer boundary until the compiler can certify exact access footprints.
    buffer: vk::Buffer,
    access: VulkanResidentKernelBufferAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentKernelBufferDependency {
    buffer: vk::Buffer,
}

fn take_resident_kernel_buffer_dependencies(
    pending: &mut Vec<VulkanResidentKernelBufferAccessRecord>,
    current: &[VulkanResidentKernelBufferAccessRecord],
) -> Vec<VulkanResidentKernelBufferDependency> {
    let dependencies = current
        .iter()
        .filter(|current_access| {
            pending.iter().any(|pending_access| {
                pending_access.buffer == current_access.buffer
                    && pending_access.access.conflicts_with(current_access.access)
            })
        })
        .map(|current_access| VulkanResidentKernelBufferDependency {
            buffer: current_access.buffer,
        })
        .collect::<Vec<_>>();
    pending.retain(|pending_access| {
        !current.iter().any(|current_access| {
            pending_access.buffer == current_access.buffer
                && pending_access.access.conflicts_with(current_access.access)
        })
    });
    dependencies
}

fn merge_resident_kernel_buffer_accesses(
    pending: &mut Vec<VulkanResidentKernelBufferAccessRecord>,
    current: &[VulkanResidentKernelBufferAccessRecord],
) {
    for current_access in current {
        if let Some(pending_access) = pending
            .iter_mut()
            .find(|pending_access| pending_access.buffer == current_access.buffer)
        {
            pending_access.access = pending_access.access.merge(current_access.access);
        } else {
            pending.push(*current_access);
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
