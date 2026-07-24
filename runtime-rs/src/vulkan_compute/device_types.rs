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
    cooperative_float8_e4m3_shapes: BTreeSet<(u32, u32, u32)>,
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
    pub execution_quantum_count: u64,
    pub execution_quantum_region_count: u64,
    pub execution_quantum_forced_yield_count: u64,
    pub execution_quantum_estimated_work_units: u64,
    pub execution_quantum_estimated_memory_bytes: u64,
    pub execution_quantum_dispatch_count: u64,
    pub execution_quantum_predicted_duration_ns: u64,
    pub execution_quantum_actual_duration_ns: u64,
    pub execution_quantum_max_region_count: u64,
    pub execution_quantum_max_actual_duration_ns: u64,
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
static EXECUTION_QUANTUM_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_REGION_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_FORCED_YIELD_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_PREDICTED_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_ACTUAL_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_MAX_REGION_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_MAX_ACTUAL_DURATION_NS: AtomicU64 = AtomicU64::new(0);

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
    EXECUTION_QUANTUM_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_REGION_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_FORCED_YIELD_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_DISPATCH_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_PREDICTED_DURATION_NS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_ACTUAL_DURATION_NS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_MAX_REGION_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_MAX_ACTUAL_DURATION_NS.store(0, Ordering::Relaxed);
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
        execution_quantum_count: EXECUTION_QUANTUM_COUNT.load(Ordering::Relaxed),
        execution_quantum_region_count: EXECUTION_QUANTUM_REGION_COUNT.load(Ordering::Relaxed),
        execution_quantum_forced_yield_count: EXECUTION_QUANTUM_FORCED_YIELD_COUNT
            .load(Ordering::Relaxed),
        execution_quantum_estimated_work_units: EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS
            .load(Ordering::Relaxed),
        execution_quantum_estimated_memory_bytes: EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES
            .load(Ordering::Relaxed),
        execution_quantum_dispatch_count: EXECUTION_QUANTUM_DISPATCH_COUNT.load(Ordering::Relaxed),
        execution_quantum_predicted_duration_ns: EXECUTION_QUANTUM_PREDICTED_DURATION_NS
            .load(Ordering::Relaxed),
        execution_quantum_actual_duration_ns: EXECUTION_QUANTUM_ACTUAL_DURATION_NS
            .load(Ordering::Relaxed),
        execution_quantum_max_region_count: EXECUTION_QUANTUM_MAX_REGION_COUNT
            .load(Ordering::Relaxed),
        execution_quantum_max_actual_duration_ns: EXECUTION_QUANTUM_MAX_ACTUAL_DURATION_NS
            .load(Ordering::Relaxed),
    }
}

pub(crate) fn record_vulkan_execution_quantum_measurement(
    measurement: &VulkanResidentExecutionQuantumMeasurement,
) {
    EXECUTION_QUANTUM_COUNT.fetch_add(1, Ordering::Relaxed);
    EXECUTION_QUANTUM_REGION_COUNT.fetch_add(
        u64::try_from(measurement.region_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_FORCED_YIELD_COUNT.fetch_add(
        u64::from(measurement.forced_yield_after),
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS
        .fetch_add(measurement.cost.work_units, Ordering::Relaxed);
    EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES
        .fetch_add(measurement.cost.memory_bytes, Ordering::Relaxed);
    EXECUTION_QUANTUM_DISPATCH_COUNT
        .fetch_add(measurement.cost.dispatches, Ordering::Relaxed);
    EXECUTION_QUANTUM_PREDICTED_DURATION_NS.fetch_add(
        measurement.cost.predicted_duration_ns,
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_ACTUAL_DURATION_NS
        .fetch_add(measurement.duration_ns, Ordering::Relaxed);
    EXECUTION_QUANTUM_MAX_REGION_COUNT.fetch_max(
        u64::try_from(measurement.region_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_MAX_ACTUAL_DURATION_NS
        .fetch_max(measurement.duration_ns, Ordering::Relaxed);
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
    pub cooperative_bfloat16_shapes: BTreeSet<(u32, u32, u32)>,
    pub cooperative_float8_e4m3_shapes: BTreeSet<(u32, u32, u32)>,
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
/// Timeline waits and signals remain attached to their original command. A
/// bounded batch is partitioned into execution quanta before submission so a
/// complete graph cannot accidentally become one watchdog-visible GPU job.
pub struct VulkanResidentQueueSubmissionBatch<'a> {
    groups: RefCell<Vec<VulkanResidentQueueSubmissionGroup<'a>>>,
    quantum_budget: Option<RuntimeExecutionQuantumBudget>,
}

/// A mounted queue-submission topology. Command buffers, queue ordering, and
/// semaphore edges stay fixed; replay only advances timeline values. The
/// template owns the lightweight queue handles it needs, so its lifetime is
/// independent from the temporary references used while recording it.
pub struct VulkanResidentQueueSubmissionTemplate {
    groups: Vec<VulkanResidentQueueSubmissionTemplateGroup>,
    submission_count: usize,
}

struct VulkanResidentQueueSubmissionGroup<'a> {
    device: &'a VulkanComputeDevice,
    submissions: Vec<VulkanPreparedResidentQueueSubmission>,
    quantum_ranges: Vec<std::ops::Range<usize>>,
    quanta: Vec<Option<RuntimeExecutionQuantum>>,
}

struct VulkanResidentQueueSubmissionTemplateGroup {
    submitter: VulkanResidentQueueSubmitter,
    submissions: Vec<VulkanPreparedResidentQueueSubmission>,
    quantum_ranges: Vec<std::ops::Range<usize>>,
    quanta: Vec<Option<RuntimeExecutionQuantum>>,
}

#[derive(Clone)]
struct VulkanResidentQueueSubmitter {
    device: ash::Device,
    queue: vk::Queue,
}

struct VulkanPreparedResidentQueueSubmission {
    command_buffer: vk::CommandBuffer,
    wait_points: Vec<(vk::Semaphore, u64)>,
    signal_points: Vec<(vk::Semaphore, u64)>,
    completion_fence: vk::Fence,
    signal_completion: bool,
    execution_region: Option<RuntimeExecutionRegion>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentExecutionQuantumMeasurement {
    pub cost: crate::execution_schedule::RuntimeExecutionCost,
    pub region_count: usize,
    pub component_ids: Vec<String>,
    pub kernel_families: Vec<String>,
    pub duration_ns: u64,
    pub forced_yield_after: bool,
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
            quantum_budget: None,
        }
    }

    pub fn new_bounded(quantum_budget: RuntimeExecutionQuantumBudget) -> Self {
        Self {
            groups: RefCell::new(Vec::new()),
            quantum_budget: Some(quantum_budget),
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
        self.enqueue_recorded_sequence_with_execution_region(
            device,
            sequence,
            wait_points,
            signal_points,
            signal_completion,
            None,
        )
    }

    pub fn enqueue_recorded_sequence_with_execution_region(
        &self,
        device: &'a VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        execution_region: Option<RuntimeExecutionRegion>,
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
            completion_fence: sequence.completion_fence,
            signal_completion,
            execution_region,
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
                quantum_ranges: Vec::new(),
                quanta: Vec::new(),
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

    pub fn mount(self) -> Result<VulkanResidentQueueSubmissionTemplate, VulkanError> {
        self.mount_with_calibrator(None)
    }

    pub fn mount_calibrated(
        self,
        calibrator: &RuntimeExecutionQuantumCalibrator,
    ) -> Result<VulkanResidentQueueSubmissionTemplate, VulkanError> {
        self.mount_with_calibrator(Some(calibrator))
    }

    fn mount_with_calibrator(
        self,
        calibrator: Option<&RuntimeExecutionQuantumCalibrator>,
    ) -> Result<VulkanResidentQueueSubmissionTemplate, VulkanError> {
        let quantum_budget = self.quantum_budget;
        let mut groups = self.groups.into_inner();
        if quantum_budget.is_some() || calibrator.is_some() {
            for group in &mut groups {
                let mut regions = group
                    .submissions
                    .iter()
                    .enumerate()
                    .map(|(submission_index, submission)| {
                        submission.execution_region.clone().ok_or_else(|| {
                            VulkanError(format!(
                                "bounded resident queue submission {submission_index} has no execution-region contract"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let quantum_budget = if let Some(calibrator) = calibrator {
                    calibrator.prepare_regions(&mut regions)
                } else {
                    quantum_budget
                        .expect("bounded submission has a quantum budget")
                };
                let schedule = RuntimeExecutionSchedule::linear(&regions, quantum_budget)
                    .map_err(|error| VulkanError(error.to_string()))?;
                group.quantum_ranges = schedule
                    .quanta
                    .iter()
                    .map(|quantum| quantum.region_range.clone())
                    .collect();
                group.quanta = schedule.quanta.into_iter().map(Some).collect();
            }
        } else {
            for group in &mut groups {
                if !group.submissions.is_empty() {
                    group.quantum_ranges.push(0..group.submissions.len());
                    group.quanta.push(None);
                }
            }
        }
        let submission_count = groups.iter().try_fold(0usize, |total, group| {
            total.checked_add(group.submissions.len()).ok_or_else(|| {
                VulkanError("resident queue submission count overflowed".to_string())
            })
        })?;
        let groups = groups
            .into_iter()
            .map(|group| VulkanResidentQueueSubmissionTemplateGroup {
                submitter: VulkanResidentQueueSubmitter {
                    device: group.device.device.clone(),
                    queue: group.device.queue,
                },
                submissions: group.submissions,
                quantum_ranges: group.quantum_ranges,
                quanta: group.quanta,
            })
            .collect();
        Ok(VulkanResidentQueueSubmissionTemplate {
            groups,
            submission_count,
        })
    }
}

impl VulkanResidentQueueSubmissionTemplate {
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
            for quantum_range in &group.quantum_ranges {
                group.submitter.submit_prepared_resident_queue_batch(
                    &group.submissions[quantum_range.clone()],
                    timeline_value_offset,
                    None,
                )?;
            }
        }
        Ok(self.submission_count)
    }

    pub fn submit_calibrated_quanta_and_wait(
        &self,
        timeline_value_offset: u64,
    ) -> Result<Vec<VulkanResidentExecutionQuantumMeasurement>, VulkanError> {
        if self.groups.len() != 1 {
            return Err(VulkanError(
                "calibrated execution quanta require one logical device per mounted template"
                    .to_string(),
            ));
        }
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
        let total_quantum_count = self
            .groups
            .iter()
            .map(|group| group.quantum_ranges.len())
            .sum::<usize>();
        let mut measurements = Vec::with_capacity(total_quantum_count);
        for group in &self.groups {
            for (quantum_index, (quantum_range, quantum)) in group
                .quantum_ranges
                .iter()
                .zip(&group.quanta)
                .enumerate()
            {
                let quantum = quantum.as_ref().ok_or_else(|| {
                    VulkanError(
                        "execution quantum measurement requires calibrated schedule metadata"
                            .to_string(),
                    )
                })?;
                let submissions = &group.submissions[quantum_range.clone()];
                let completion_fence = submissions
                    .last()
                    .map(|submission| submission.completion_fence)
                    .ok_or_else(|| {
                        VulkanError("execution quantum contains no submissions".to_string())
                    })?;
                let started = Instant::now();
                group.submitter.submit_prepared_resident_queue_batch(
                    submissions,
                    timeline_value_offset,
                    Some(completion_fence),
                )?;
                group.submitter.wait_for_completion_fence(completion_fence)?;
                measurements.push(VulkanResidentExecutionQuantumMeasurement {
                    cost: quantum.cost,
                    region_count: quantum.region_count(),
                    component_ids: quantum.component_ids.clone(),
                    kernel_families: quantum.kernel_families.clone(),
                    duration_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    forced_yield_after: quantum_index + 1 < total_quantum_count,
                });
            }
        }
        Ok(measurements)
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
