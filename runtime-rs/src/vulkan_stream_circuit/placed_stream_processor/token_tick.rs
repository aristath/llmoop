impl VulkanResidentInProcessPlacedStreamProcessor {
    pub fn mounted_device(&self, device_id: &str) -> Option<&VulkanMountedPlacedStreamCircuit> {
        self.device(device_id).map(|slice| &slice.mounted)
    }

    pub fn run_stream_tick_in_process(
        &self,
        device: &VulkanComputeDevice,
        stream_tick: u64,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedCableTransport::new();
        self.run_stream_tick_in_process_with_transport(device, &mut transport, stream_tick)
    }

    pub fn run_stream_tick_in_process_with_transport(
        &self,
        device: &VulkanComputeDevice,
        transport: &mut VulkanInProcessPlacedCableTransport,
        stream_tick: u64,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut tick_slices = SmallVec::<
            [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>; 4],
        >::with_capacity(self.device_slices.len());

        for slice in &self.device_slices {
            tick_slices.push(VulkanMountedPlacedResidentInProcessStreamTickSlice::new(
                device,
                &slice.mounted,
                &slice.resident_execution_plan,
                stream_tick,
            ));
        }

        run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
            &mut tick_slices,
            transport,
            &self.activation_schedule,
            Some(&self.distributed_dispatch_runners),
            Some(&self.cable_synchronizations),
            VulkanPlacedSubmissionContext::SYNCHRONOUS,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Tick)
    }

    pub fn run_stream_tick_on_bound_devices_in_process(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        stream_tick: u64,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedCableTransport::new();
        self.run_stream_tick_on_bound_devices_in_process_with_transport(
            devices,
            &mut transport,
            stream_tick,
        )
    }

    pub fn run_stream_tick_on_bound_devices_in_process_with_transport(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        transport: &mut VulkanInProcessPlacedCableTransport,
        stream_tick: u64,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut tick_slices = SmallVec::<
            [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>; 4],
        >::with_capacity(self.device_slices.len());

        for slice in &self.device_slices {
            let slice_device = devices
                .get(&slice.device_id)
                .map(|device| device.as_ref())
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    },
                )?;
            tick_slices.push(VulkanMountedPlacedResidentInProcessStreamTickSlice::new(
                slice_device,
                &slice.mounted,
                &slice.resident_execution_plan,
                stream_tick,
            ));
        }

        run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
            &mut tick_slices,
            transport,
            &self.activation_schedule,
            Some(&self.distributed_dispatch_runners),
            Some(&self.cable_synchronizations),
            VulkanPlacedSubmissionContext::SYNCHRONOUS,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Tick)
    }

    fn prepared_token_tick_slices_for_device<'a>(
        &'a self,
        device: &'a VulkanComputeDevice,
        stream_tick: u64,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> SmallVec<[VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>; 4]> {
        let mut tick_slices = SmallVec::<
            [VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>; 4],
        >::with_capacity(self.device_slices.len());
        for slice in &self.device_slices {
            tick_slices.push(
                VulkanMountedPlacedResidentInProcessStreamTickSlice::new_with_dispatch_extensions(
                    device,
                    &slice.mounted,
                    &slice.resident_execution_plan,
                    self.prepared_token_tick_dispatch_extensions(&slice.device_id, tail),
                    stream_tick,
                ),
            );
        }
        tick_slices
    }

    fn prepared_token_tick_slices_for_bound_devices<'a>(
        &'a self,
        devices: &'a BTreeMap<String, Rc<VulkanComputeDevice>>,
        stream_tick: u64,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> Result<
        SmallVec<[VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>; 4]>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut tick_slices = SmallVec::<
            [VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>; 4],
        >::with_capacity(self.device_slices.len());
        for slice in &self.device_slices {
            let slice_device = devices
                .get(&slice.device_id)
                .map(|device| device.as_ref())
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    },
                )?;
            tick_slices.push(
                VulkanMountedPlacedResidentInProcessStreamTickSlice::new_with_dispatch_extensions(
                    slice_device,
                    &slice.mounted,
                    &slice.resident_execution_plan,
                    self.prepared_token_tick_dispatch_extensions(&slice.device_id, tail),
                    stream_tick,
                ),
            );
        }
        Ok(tick_slices)
    }

    fn prepared_token_tick_dispatch_extensions(
        &self,
        device_id: &str,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> VulkanMountedPlacedResidentStreamTickDispatchExtensions<'_> {
        let mut dispatch_extensions = VulkanMountedPlacedResidentStreamTickDispatchExtensions {
            sequence_variant: tail.sequence_variant(),
            ..Default::default()
        };
        if device_id == self.model.input_device_id {
            dispatch_extensions
                .prefix_dispatches
                .push(&self.input_transducer.resident_dispatch);
        }
        if device_id == self.model.output_device_id {
            dispatch_extensions
                .prefix_dispatches
                .extend(self.sampler.input_tracking_dispatches());
        }
        if device_id == self.model.output_device_id
            && tail != VulkanResidentPlacedTokenTickTail::None
        {
            dispatch_extensions
                .suffix_dispatches
                .push(&self.output_transducer.embedding_norm_dispatch);
            if tail.produces_logits() {
                dispatch_extensions
                    .suffix_dispatches
                    .push(&self.output_transducer.tied_projection_dispatch);
            }
            if tail == VulkanResidentPlacedTokenTickTail::Sample {
                dispatch_extensions
                    .suffix_dispatches
                    .extend(self.sampler.resident_dispatches());
            }
        }
        dispatch_extensions
    }

    fn run_prepared_token_tick_slices_deferred<'a>(
        &'a self,
        tick_slices: &mut SmallVec<[VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>; 4]>,
        transport: &mut VulkanInProcessPlacedCableTransport,
        output_device: &VulkanComputeDevice,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let submission_batch = VulkanResidentQueueSubmissionBatch::new();
        let output_turn = self
            .output_synchronization
            .prepare_turn(&self.model.output_device_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let output_timeline_value = output_turn.value;
        let placed_run =
            run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
                tick_slices,
                transport,
                &self.activation_schedule,
                Some(&self.distributed_dispatch_runners),
                Some(&self.cable_synchronizations),
                VulkanPlacedSubmissionContext {
                    policy: VulkanPlacedSubmissionPolicy {
                        write_stream_control: true,
                        signal_completion: false,
                        wait_for_completion: false,
                        feedback_lane: None,
                    },
                    state_transactions: None,
                    feedback_turn: None,
                    output_turn: Some(output_turn),
                    submission_batch: Some(&submission_batch),
                },
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Tick)?;
        if placed_run.status != VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed {
            return Err(VulkanResidentInProcessPlacedRuntimeError::IncompleteTick(
                placed_run.status,
            ));
        }
        let submission_template = submission_batch
            .mount()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        submission_template
            .submit_with_timeline_value_offset(0)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.output_synchronization
            .wait_for_turn(output_device, output_timeline_value)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(placed_run)
    }

    fn execute_prepared_token_id_stream_tick_in_process_with_transport(
        &self,
        device: &VulkanComputeDevice,
        transport: &mut VulkanInProcessPlacedCableTransport,
        stream_tick: u64,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut tick_slices = self.prepared_token_tick_slices_for_device(device, stream_tick, tail);
        self.run_prepared_token_tick_slices_deferred(&mut tick_slices, transport, device)
    }

    fn run_prepared_token_id_stream_tick_in_process_with_transport(
        &self,
        device: &VulkanComputeDevice,
        transport: &mut VulkanInProcessPlacedCableTransport,
        input: VulkanResidentPlacedTokenInput,
        stream_tick: u64,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> Result<
        (
            VulkanResidentInProcessPlacedSingleTokenTickRun,
            Option<VulkanResidentSamplerRun>,
        ),
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let token_id = input.token_id();
        let input_run = self.prepare_token_input(input)?;
        let placed_run = self.execute_prepared_token_id_stream_tick_in_process_with_transport(
            device,
            transport,
            stream_tick,
            tail,
        )?;
        let output_run = tail
            .produces_logits()
            .then(|| self.output_transducer.completed_run());
        let sampler_run = if tail == VulkanResidentPlacedTokenTickTail::Sample {
            Some(
                self.sampler
                    .completed_run()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?,
            )
        } else {
            None
        };
        Ok((
            VulkanResidentInProcessPlacedSingleTokenTickRun {
                input_device_id: self.model.input_device_id.clone(),
                output_device_id: self.model.output_device_id.clone(),
                token_id,
                stream_tick,
                input_run,
                placed_run,
                output_run,
            },
            sampler_run,
        ))
    }

    fn execute_prepared_token_id_stream_tick_on_bound_devices_in_process_with_transport(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        transport: &mut VulkanInProcessPlacedCableTransport,
        stream_tick: u64,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> Result<
        VulkanMountedPlacedResidentInProcessStreamTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut tick_slices =
            self.prepared_token_tick_slices_for_bound_devices(devices, stream_tick, tail)?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        self.run_prepared_token_tick_slices_deferred(
            &mut tick_slices,
            transport,
            output_device.as_ref(),
        )
    }

    fn run_prepared_token_id_stream_tick_on_bound_devices_in_process_with_transport(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        transport: &mut VulkanInProcessPlacedCableTransport,
        input: VulkanResidentPlacedTokenInput,
        stream_tick: u64,
        tail: VulkanResidentPlacedTokenTickTail,
    ) -> Result<
        (
            VulkanResidentInProcessPlacedSingleTokenTickRun,
            Option<VulkanResidentSamplerRun>,
        ),
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let token_id = input.token_id();
        let input_run = self.prepare_token_input(input)?;
        let placed_run = self
            .execute_prepared_token_id_stream_tick_on_bound_devices_in_process_with_transport(
                devices,
                transport,
                stream_tick,
                tail,
            )?;
        let output_run = tail
            .produces_logits()
            .then(|| self.output_transducer.completed_run());
        let sampler_run = if tail == VulkanResidentPlacedTokenTickTail::Sample {
            Some(
                self.sampler
                    .completed_run()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?,
            )
        } else {
            None
        };
        Ok((
            VulkanResidentInProcessPlacedSingleTokenTickRun {
                input_device_id: self.model.input_device_id.clone(),
                output_device_id: self.model.output_device_id.clone(),
                token_id,
                stream_tick,
                input_run,
                placed_run,
                output_run,
            },
            sampler_run,
        ))
    }

    pub fn run_token_id_stream_tick_in_process(
        &self,
        device: &VulkanComputeDevice,
        token_id: u32,
        stream_tick: u64,
    ) -> Result<
        VulkanResidentInProcessPlacedSingleTokenTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedCableTransport::new();
        self.run_token_id_stream_tick_in_process_with_transport(
            device,
            &mut transport,
            token_id,
            stream_tick,
        )
    }

    pub fn run_token_id_stream_tick_in_process_with_transport(
        &self,
        device: &VulkanComputeDevice,
        transport: &mut VulkanInProcessPlacedCableTransport,
        token_id: u32,
        stream_tick: u64,
    ) -> Result<
        VulkanResidentInProcessPlacedSingleTokenTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.run_prepared_token_id_stream_tick_in_process_with_transport(
            device,
            transport,
            VulkanResidentPlacedTokenInput::HostSupplied(token_id),
            stream_tick,
            VulkanResidentPlacedTokenTickTail::Logits,
        )
        .map(|(tick_run, _)| tick_run)
    }

    pub fn run_token_id_stream_tick_on_bound_devices_in_process(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        token_id: u32,
        stream_tick: u64,
    ) -> Result<
        VulkanResidentInProcessPlacedSingleTokenTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedCableTransport::new();
        self.run_token_id_stream_tick_on_bound_devices_in_process_with_transport(
            devices,
            &mut transport,
            token_id,
            stream_tick,
        )
    }

    pub fn run_token_id_stream_tick_on_bound_devices_in_process_with_transport(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        transport: &mut VulkanInProcessPlacedCableTransport,
        token_id: u32,
        stream_tick: u64,
    ) -> Result<
        VulkanResidentInProcessPlacedSingleTokenTickRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.run_prepared_token_id_stream_tick_on_bound_devices_in_process_with_transport(
            devices,
            transport,
            VulkanResidentPlacedTokenInput::HostSupplied(token_id),
            stream_tick,
            VulkanResidentPlacedTokenTickTail::Logits,
        )
        .map(|(tick_run, _)| tick_run)
    }

    pub fn sample_token_id_stream_tick_in_process(
        &self,
        device: &VulkanComputeDevice,
        token_id: u32,
        stream_tick: u64,
    ) -> Result<
        VulkanResidentInProcessPlacedSingleTokenSampleRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedCableTransport::new();
        self.sample_token_id_stream_tick_in_process_with_transport(
            device,
            &mut transport,
            token_id,
            stream_tick,
        )
    }

    pub fn sample_token_id_stream_tick_in_process_with_transport(
        &self,
        device: &VulkanComputeDevice,
        transport: &mut VulkanInProcessPlacedCableTransport,
        token_id: u32,
        stream_tick: u64,
    ) -> Result<
        VulkanResidentInProcessPlacedSingleTokenSampleRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let (tick_run, sampler_run) = self
            .run_prepared_token_id_stream_tick_in_process_with_transport(
                device,
                transport,
                VulkanResidentPlacedTokenInput::HostSupplied(token_id),
                stream_tick,
                VulkanResidentPlacedTokenTickTail::Sample,
            )?;
        Ok(VulkanResidentInProcessPlacedSingleTokenSampleRun {
            tick_run,
            sampler_run: sampler_run
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingFusedSamplerRun)?,
        })
    }
}
