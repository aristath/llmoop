pub struct VulkanMountedPlacedResidentPedalRunner {
    pub pedal_id: String,
    pub dispatches: Vec<VulkanMountedPlacedResidentPedalDispatch>,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    stream_control_buffer: Arc<VulkanResidentBuffer>,
}

impl VulkanMountedPlacedResidentPedalRunner {
    fn from_mounted_bound_plan(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        pedal_id: &str,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        let mut dispatches = Vec::new();
        let mut total_descriptor_count = 0usize;
        let mut total_push_constant_byte_count = 0u32;

        for dispatch in mounted_bound_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.pedal_id == pedal_id)
        {
            let resident_dispatch = mounted.create_resident_kernel_dispatch_for_bound_dispatch(
                device,
                dispatch,
                loaded_manifest,
            )?;
            total_descriptor_count = total_descriptor_count
                .checked_add(resident_dispatch.descriptor_count())
                .ok_or(VulkanMountedPlacedResidentKernelDispatchError::PedalRunnerDescriptorCountOverflow {
                    pedal_id: pedal_id.to_string(),
                })?;
            total_push_constant_byte_count = total_push_constant_byte_count
                .checked_add(resident_dispatch.push_constant_byte_count())
                .ok_or(VulkanMountedPlacedResidentKernelDispatchError::PedalRunnerPushConstantByteCountOverflow {
                    pedal_id: pedal_id.to_string(),
                })?;
            dispatches.push(VulkanMountedPlacedResidentPedalDispatch {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                pedal_id: dispatch.pedal_id.clone(),
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                push_constants: dispatch.push_constants.clone(),
                resident_dispatch,
            });
        }

        if dispatches.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingPedalDispatches {
                    pedal_id: pedal_id.to_string(),
                },
            );
        }

        Ok(Self {
            pedal_id: pedal_id.to_string(),
            dispatches,
            total_descriptor_count,
            total_push_constant_byte_count,
            stream_control_buffer: mounted.stream_control_buffer.clone(),
        })
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    pub fn run_zeroed_push_constants(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanMountedPlacedResidentPedalRun, VulkanMountedPlacedResidentKernelDispatchError>
    {
        self.run_with_push_constant_bytes(device, |dispatch| {
            Ok(vec![
                0u8;
                dispatch.resident_dispatch.push_constant_byte_count()
                    as usize
            ])
        })
    }

    pub fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<VulkanMountedPlacedResidentPedalRun, VulkanMountedPlacedResidentKernelDispatchError>
    {
        self.stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        self.run_with_push_constant_bytes(device, |dispatch| {
            stream_control_push_constant_bytes(&dispatch.push_constants, control)
        })
    }

    fn run_with_push_constant_bytes<F>(
        &self,
        device: &VulkanComputeDevice,
        mut push_constant_bytes_for: F,
    ) -> Result<VulkanMountedPlacedResidentPedalRun, VulkanMountedPlacedResidentKernelDispatchError>
    where
        F: FnMut(
            &VulkanMountedPlacedResidentPedalDispatch,
        ) -> Result<Vec<u8>, VulkanMountedPlacedResidentKernelDispatchError>,
    {
        let mut dispatch_runs = Vec::with_capacity(self.dispatches.len());
        for dispatch in &self.dispatches {
            let push_constants = push_constant_bytes_for(dispatch)?;
            let run_start = Instant::now();
            device
                .run_resident_kernel_dispatch(&dispatch.resident_dispatch, &push_constants)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            let run_time_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            dispatch_runs.push(VulkanMountedPlacedResidentPedalDispatchRun {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                descriptor_count: dispatch.resident_dispatch.descriptor_count(),
                workgroup_count_x: dispatch.resident_dispatch.workgroup_count_x(),
                push_constant_byte_count: dispatch.resident_dispatch.push_constant_byte_count(),
                run_time_ns,
            });
        }

        Ok(VulkanMountedPlacedResidentPedalRun {
            pedal_id: self.pedal_id.clone(),
            dispatch_runs,
        })
    }

    fn completed_sequence_run(&self) -> VulkanMountedPlacedResidentPedalRun {
        VulkanMountedPlacedResidentPedalRun {
            pedal_id: self.pedal_id.clone(),
            dispatch_runs: self
                .dispatches
                .iter()
                .map(|dispatch| VulkanMountedPlacedResidentPedalDispatchRun {
                    dispatch_index: dispatch.dispatch_index,
                    kernel_id: dispatch.kernel_id.clone(),
                    node_id: dispatch.node_id.clone(),
                    op: dispatch.op.clone(),
                    reusable_family_id: dispatch.reusable_family_id.clone(),
                    descriptor_count: dispatch.resident_dispatch.descriptor_count(),
                    workgroup_count_x: dispatch.resident_dispatch.workgroup_count_x(),
                    push_constant_byte_count: dispatch.resident_dispatch.push_constant_byte_count(),
                    // The composed sequence has one measurable execution boundary.
                    // Per-dispatch timings would be fabricated without timestamp queries.
                    run_time_ns: 0,
                })
                .collect(),
        }
    }
}

pub struct VulkanMountedPlacedResidentPedalDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub resident_dispatch: VulkanResidentKernelDispatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentPedalRun {
    pub pedal_id: String,
    pub dispatch_runs: Vec<VulkanMountedPlacedResidentPedalDispatchRun>,
}

impl VulkanMountedPlacedResidentPedalRun {
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_runs.len()
    }

    pub fn run_time_ns(&self) -> u64 {
        self.dispatch_runs.iter().fold(0u64, |total, dispatch| {
            total.saturating_add(dispatch.run_time_ns)
        })
    }

    pub fn node_ids(&self) -> Vec<&str> {
        self.dispatch_runs
            .iter()
            .map(|dispatch| dispatch.node_id.as_str())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentPedalDispatchRun {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
    pub run_time_ns: u64,
}

