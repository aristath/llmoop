pub struct VulkanMountedPlacedResidentPedalboardRunner {
    pub device_id: String,
    pub pedals: Vec<VulkanMountedPlacedResidentPedalRunner>,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
}

impl VulkanMountedPlacedResidentPedalboardRunner {
    fn from_mounted_bound_plan<I, S>(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        pedal_ids: I,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut pedals = Vec::new();
        let mut total_descriptor_count = 0usize;
        let mut total_push_constant_byte_count = 0u32;

        for pedal_id in pedal_ids {
            let pedal_id = pedal_id.as_ref();
            let runner = VulkanMountedPlacedResidentPedalRunner::from_mounted_bound_plan(
                device,
                mounted,
                mounted_bound_plan,
                pedal_id,
                loaded_manifest,
            )?;
            total_descriptor_count = total_descriptor_count
                .checked_add(runner.total_descriptor_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::PedalboardRunnerDescriptorCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            total_push_constant_byte_count = total_push_constant_byte_count
                .checked_add(runner.total_push_constant_byte_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::PedalboardRunnerPushConstantByteCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            pedals.push(runner);
        }

        if pedals.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingPedalboardPedals {
                    device_id: mounted_bound_plan.device_id.clone(),
                },
            );
        }

        Ok(Self {
            device_id: mounted_bound_plan.device_id.clone(),
            pedals,
            total_descriptor_count,
            total_push_constant_byte_count,
        })
    }

    pub fn pedal_count(&self) -> usize {
        self.pedals.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.pedals
            .iter()
            .map(VulkanMountedPlacedResidentPedalRunner::dispatch_count)
            .sum()
    }

    pub fn pedal_ids(&self) -> Vec<&str> {
        self.pedals
            .iter()
            .map(|pedal| pedal.pedal_id.as_str())
            .collect()
    }

    pub fn run_zeroed_push_constants(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<
        VulkanMountedPlacedResidentPedalboardRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut pedal_runs = Vec::with_capacity(self.pedals.len());
        for pedal in &self.pedals {
            pedal_runs.push(pedal.run_zeroed_push_constants(device)?);
        }

        Ok(VulkanMountedPlacedResidentPedalboardRun {
            device_id: self.device_id.clone(),
            pedal_runs,
        })
    }

    pub fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<
        VulkanMountedPlacedResidentPedalboardRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut pedal_runs = Vec::with_capacity(self.pedals.len());
        for pedal in &self.pedals {
            pedal_runs.push(pedal.run_with_stream_control(device, control)?);
        }

        Ok(VulkanMountedPlacedResidentPedalboardRun {
            device_id: self.device_id.clone(),
            pedal_runs,
        })
    }

    fn completed_sequence_run(&self) -> VulkanMountedPlacedResidentPedalboardRun {
        VulkanMountedPlacedResidentPedalboardRun {
            device_id: self.device_id.clone(),
            pedal_runs: self
                .pedals
                .iter()
                .map(VulkanMountedPlacedResidentPedalRunner::completed_sequence_run)
                .collect(),
        }
    }
}
