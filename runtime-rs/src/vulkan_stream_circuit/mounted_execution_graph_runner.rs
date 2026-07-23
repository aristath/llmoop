pub struct VulkanMountedPlacedResidentExecutionGraphRunner {
    pub device_id: String,
    pub components: Vec<VulkanMountedPlacedResidentComponentRunner>,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
}

impl VulkanMountedPlacedResidentExecutionGraphRunner {
    fn from_mounted_bound_plan<I, S>(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        component_ids: I,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut components = Vec::new();
        let mut total_descriptor_count = 0usize;
        let mut total_push_constant_byte_count = 0u32;

        for component_id in component_ids {
            let component_id = component_id.as_ref();
            let runner = VulkanMountedPlacedResidentComponentRunner::from_mounted_bound_plan(
                device,
                mounted,
                mounted_bound_plan,
                component_id,
                loaded_manifest,
            )?;
            total_descriptor_count = total_descriptor_count
                .checked_add(runner.total_descriptor_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::ExecutionGraphRunnerDescriptorCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            total_push_constant_byte_count = total_push_constant_byte_count
                .checked_add(runner.total_push_constant_byte_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::ExecutionGraphRunnerPushConstantByteCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            components.push(runner);
        }

        if components.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingExecutionGraphComponents {
                    device_id: mounted_bound_plan.device_id.clone(),
                },
            );
        }

        Ok(Self {
            device_id: mounted_bound_plan.device_id.clone(),
            components,
            total_descriptor_count,
            total_push_constant_byte_count,
        })
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.components
            .iter()
            .map(VulkanMountedPlacedResidentComponentRunner::dispatch_count)
            .sum()
    }

    pub fn component_ids(&self) -> Vec<&str> {
        self.components
            .iter()
            .map(|component| component.component_id.as_str())
            .collect()
    }

    pub fn run_zeroed_push_constants(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<
        VulkanMountedPlacedResidentExecutionGraphRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut component_runs = Vec::with_capacity(self.components.len());
        for component in &self.components {
            component_runs.push(component.run_zeroed_push_constants(device)?);
        }

        Ok(VulkanMountedPlacedResidentExecutionGraphRun {
            device_id: self.device_id.clone(),
            component_runs,
        })
    }

    pub fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<
        VulkanMountedPlacedResidentExecutionGraphRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut component_runs = Vec::with_capacity(self.components.len());
        for component in &self.components {
            component_runs.push(component.run_with_stream_control(device, control)?);
        }

        Ok(VulkanMountedPlacedResidentExecutionGraphRun {
            device_id: self.device_id.clone(),
            component_runs,
        })
    }

    fn completed_sequence_run(&self) -> VulkanMountedPlacedResidentExecutionGraphRun {
        VulkanMountedPlacedResidentExecutionGraphRun {
            device_id: self.device_id.clone(),
            component_runs: self
                .components
                .iter()
                .map(VulkanMountedPlacedResidentComponentRunner::completed_sequence_run)
                .collect(),
        }
    }
}
