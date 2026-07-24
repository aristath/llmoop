pub struct VulkanResidentSamplerRunner {
    pub sampler_id: String,
    pub logits_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub dispatch_count: usize,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
    pub history_capacity_activations: usize,
    output_buffer: VulkanResidentBuffer,
    _scratch_buffer: Option<VulkanResidentBuffer>,
    _sampler_seed_buffer: Option<VulkanResidentBuffer>,
    _seen_token_buffer: Option<VulkanResidentBuffer>,
    _seen_token_snapshot_buffer: Option<VulkanResidentBuffer>,
    capture_seen_token_copy: Option<VulkanResidentBufferCopy>,
    restore_seen_token_copy: Option<VulkanResidentBufferCopy>,
    _sampler_parameter_buffer: Option<VulkanResidentBuffer>,
    _feedback_control_buffer: Arc<VulkanResidentBuffer>,
    seen_token_batch_buffer: Option<VulkanResidentBuffer>,
    _stream_control_buffer: Arc<VulkanResidentBuffer>,
    input_tracking_dispatches: Vec<VulkanResidentKernelDispatch>,
    seen_token_batch_dispatch: Option<VulkanResidentKernelDispatch>,
    seen_token_batch_sequence: Option<VulkanResidentKernelSequence>,
    resident_dispatches: Vec<VulkanResidentKernelDispatch>,
    feedback_control_dispatch: VulkanResidentKernelDispatch,
    sequence: VulkanResidentKernelSequence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentSamplerSpec {
    pub sampler_id: String,
    pub method: String,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub top_k_capacity: u32,
    pub runtime_parameterized: bool,
    pub logits_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub scratch_byte_capacity: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VulkanResidentSamplerRuntimeConfig {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl VulkanResidentSamplerRuntimeConfig {
    pub fn is_empty(self) -> bool {
        self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.presence_penalty.is_none()
            && self.repetition_penalty.is_none()
    }

    fn apply_to(
        self,
        default: &VulkanResidentSamplerSpec,
    ) -> Result<VulkanResidentSamplerSpec, VulkanResidentSamplerRunnerError> {
        if self.is_empty() {
            return Ok(default.clone());
        }
        let mut effective = default.clone();
        let requests_sampled_decoding = self.temperature.is_some()
            || self.top_k.is_some()
            || self.top_p.is_some()
            || self.min_p.is_some();
        if default.method == "greedy" && requests_sampled_decoding {
            effective.sampler_id = "runtime_temperature_top_k_top_p_sampler".to_string();
            effective.method = "temperature_top_k_top_p".to_string();
            effective.temperature = 1.0;
            effective.top_k = effective.top_k_capacity.min(50);
            effective.top_p = 1.0;
            effective.min_p = 0.0;
        }
        if let Some(value) = self.temperature {
            effective.temperature = value;
        }
        if let Some(value) = self.top_k {
            effective.top_k = value;
        }
        if let Some(value) = self.top_p {
            effective.top_p = value;
        }
        if let Some(value) = self.min_p {
            effective.min_p = value;
        }
        if let Some(value) = self.presence_penalty {
            effective.presence_penalty = value;
        }
        if let Some(value) = self.repetition_penalty {
            effective.repetition_penalty = value;
        }
        let valid_common = effective.presence_penalty.is_finite()
            && effective.repetition_penalty.is_finite()
            && effective.repetition_penalty > 0.0;
        let valid_method = match effective.method.as_str() {
            "greedy" => effective.min_p == 0.0 && valid_common,
            "temperature_top_k_top_p" => {
                effective.temperature.is_finite()
                    && effective.temperature > 0.0
                    && effective.top_k > 0
                    && effective.top_k <= effective.top_k_capacity
                    && effective.top_p.is_finite()
                    && effective.top_p > 0.0
                    && effective.top_p <= 1.0
                    && effective.min_p.is_finite()
                    && effective.min_p >= 0.0
                    && effective.min_p <= 1.0
                    && valid_common
            }
            _ => false,
        };
        if !valid_method {
            return Err(
                VulkanResidentSamplerRunnerError::UnsupportedRuntimeSamplingOverride(format!(
                    "runtime sampler override is invalid for method {:?}: temperature={}, top_k={} (capacity {}), top_p={}, min_p={}, presence_penalty={}, repetition_penalty={}",
                    effective.method,
                    effective.temperature,
                    effective.top_k,
                    effective.top_k_capacity,
                    effective.top_p,
                    effective.min_p,
                    effective.presence_penalty,
                    effective.repetition_penalty,
                )),
            );
        }
        effective.runtime_parameterized = true;
        Ok(effective)
    }
}

fn sampler_kernel_role_matches(role: &str, runtime_parameterized: bool, method: &str) -> bool {
    matches!(
        (runtime_parameterized, method, role),
        (false, "greedy", "sample_logits")
            | (true, "greedy", "runtime_sample_logits")
            | (
                false,
                "temperature_top_k_top_p",
                "partition_top_k" | "sample_candidates"
            )
            | (
                true,
                "temperature_top_k_top_p",
                "runtime_partition_top_k" | "runtime_sample_candidates",
            )
    )
}

pub struct VulkanResidentSamplerKernelArtifact {
    pub role: String,
    pub spirv_words: Vec<u32>,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanResidentSamplerStreamConfig {
    pub history_capacity_activations: usize,
    pub random_seed: u32,
}

impl VulkanResidentSamplerRunner {
    pub fn from_output_transducer_with_spec(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        output_transducer: &VulkanResidentOutputTransducerRunner,
        kernels: &[VulkanResidentSamplerKernelArtifact],
        spec: &VulkanResidentSamplerSpec,
        random_seed: u32,
    ) -> Result<Self, VulkanResidentSamplerRunnerError> {
        Self::from_logits_buffer_with_feedback_control(
            device,
            mounted.stream_control_buffer.clone(),
            output_transducer.logits_buffer(),
            output_transducer.logits_byte_capacity,
            kernels,
            spec,
            VulkanResidentSamplerStreamConfig {
                history_capacity_activations: mounted.buffers.dynamic_state_capacity_activations,
                random_seed,
            },
            None,
        )
    }

    fn from_output_transducer_with_spec_and_feedback_control(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        output_transducer: &VulkanResidentOutputTransducerRunner,
        kernels: &[VulkanResidentSamplerKernelArtifact],
        spec: &VulkanResidentSamplerSpec,
        random_seed: u32,
        feedback_control: VulkanResidentSamplerFeedbackControlBindings,
    ) -> Result<Self, VulkanResidentSamplerRunnerError> {
        Self::from_logits_buffer_with_feedback_control(
            device,
            mounted.stream_control_buffer.clone(),
            output_transducer.logits_buffer(),
            output_transducer.logits_byte_capacity,
            kernels,
            spec,
            VulkanResidentSamplerStreamConfig {
                history_capacity_activations: mounted.buffers.dynamic_state_capacity_activations,
                random_seed,
            },
            Some(feedback_control),
        )
    }

    pub fn from_logits_buffer(
        device: &VulkanComputeDevice,
        stream_control_buffer: Arc<VulkanResidentBuffer>,
        logits_buffer: &VulkanResidentBuffer,
        logits_byte_capacity: usize,
        kernels: &[VulkanResidentSamplerKernelArtifact],
        spec: &VulkanResidentSamplerSpec,
        stream: VulkanResidentSamplerStreamConfig,
    ) -> Result<Self, VulkanResidentSamplerRunnerError> {
        Self::from_logits_buffer_with_feedback_control(
            device,
            stream_control_buffer,
            logits_buffer,
            logits_byte_capacity,
            kernels,
            spec,
            stream,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_logits_buffer_with_feedback_control(
        device: &VulkanComputeDevice,
        stream_control_buffer: Arc<VulkanResidentBuffer>,
        logits_buffer: &VulkanResidentBuffer,
        logits_byte_capacity: usize,
        kernels: &[VulkanResidentSamplerKernelArtifact],
        spec: &VulkanResidentSamplerSpec,
        stream: VulkanResidentSamplerStreamConfig,
        feedback_control: Option<VulkanResidentSamplerFeedbackControlBindings>,
    ) -> Result<Self, VulkanResidentSamplerRunnerError> {
        let VulkanResidentSamplerStreamConfig {
            history_capacity_activations,
            random_seed,
        } = stream;
        if logits_byte_capacity != spec.logits_byte_capacity {
            return Err(
                VulkanResidentSamplerRunnerError::InvalidLogitsByteCapacity {
                    byte_capacity: logits_byte_capacity,
                    expected_byte_capacity: spec.logits_byte_capacity,
                },
            );
        }
        if history_capacity_activations == 0 {
            return Err(VulkanResidentSamplerRunnerError::ZeroHistoryCapacity);
        }
        let vocabulary_size = logits_byte_capacity / std::mem::size_of::<f32>();
        let runtime_parameterized = spec.runtime_parameterized;
        let sampling_kernels = kernels
            .iter()
            .filter(|kernel| {
                sampler_kernel_role_matches(
                    kernel.role.as_str(),
                    runtime_parameterized,
                    spec.method.as_str(),
                )
            })
            .collect::<Vec<_>>();
        let token_state_is_active = spec.repetition_penalty != 1.0 || spec.presence_penalty != 0.0;
        let current_token_role = if runtime_parameterized {
            "runtime_record_current_token"
        } else {
            "record_current_token"
        };
        let token_batch_role = if runtime_parameterized {
            "runtime_record_token_batch"
        } else {
            "record_token_batch"
        };
        let current_token_kernel = kernels
            .iter()
            .find(|kernel| kernel.role == current_token_role);
        let token_batch_kernel = kernels
            .iter()
            .find(|kernel| kernel.role == token_batch_role);
        let feedback_control_kernel = kernels
            .iter()
            .find(|kernel| kernel.role == "feedback_control");
        let tracking_kernel_plan_is_valid = if token_state_is_active {
            current_token_kernel
                .is_some_and(|kernel| kernel.local_size_x == 1 && kernel.workgroup_count_x == 1)
                && token_batch_kernel.is_some_and(|kernel| {
                    kernel.local_size_x == VULKAN_BACKEND_LOOP_MAX_WINDOW as u32
                        && kernel.workgroup_count_x == 1
                })
        } else {
            true
        };
        let sampled_kernel_plan_is_valid = sampling_kernels.len() == 2
            && matches!(
                sampling_kernels[0].role.as_str(),
                "partition_top_k" | "runtime_partition_top_k"
            )
            && sampling_kernels[0].local_size_x > 0
            && sampling_kernels[0].workgroup_count_x > 0
            && matches!(
                sampling_kernels[1].role.as_str(),
                "sample_candidates" | "runtime_sample_candidates"
            )
            && sampling_kernels[1].local_size_x >= sampling_kernels[0].workgroup_count_x
            && sampling_kernels[1].workgroup_count_x == 1
            && usize::try_from(sampling_kernels[0].workgroup_count_x)
                .ok()
                .and_then(|partitions| {
                    partitions.checked_mul(if runtime_parameterized {
                        spec.top_k_capacity as usize
                    } else {
                        spec.top_k as usize
                    })
                })
                .and_then(|candidates| candidates.checked_mul(2 * std::mem::size_of::<u32>()))
                .is_some_and(|required| required <= spec.scratch_byte_capacity);
        let feedback_control_kernel_is_valid = feedback_control_kernel
            .is_some_and(|kernel| kernel.local_size_x == 1 && kernel.workgroup_count_x == 1);
        match spec.method.as_str() {
            "greedy"
                if spec.min_p == 0.0
                    && spec.presence_penalty.is_finite()
                    && spec.repetition_penalty.is_finite()
                    && spec.repetition_penalty > 0.0
                    && tracking_kernel_plan_is_valid
                    && sampling_kernels.len() == 1
                    && matches!(
                        sampling_kernels[0].role.as_str(),
                        "sample_logits" | "runtime_sample_logits"
                    )
                    && sampling_kernels[0].local_size_x > 0
                    && sampling_kernels[0].workgroup_count_x == 1
                    && feedback_control_kernel_is_valid => {}
            "temperature_top_k_top_p"
                if spec.temperature.is_finite()
                    && spec.temperature > 0.0
                    && spec.top_k > 0
                    && spec.top_k as usize <= vocabulary_size
                    && spec.top_p.is_finite()
                    && spec.top_p > 0.0
                    && spec.top_p <= 1.0
                    && spec.min_p.is_finite()
                    && spec.min_p >= 0.0
                    && spec.min_p <= 1.0
                    && spec.presence_penalty.is_finite()
                    && spec.repetition_penalty.is_finite()
                    && spec.repetition_penalty > 0.0
                    && spec.top_k <= spec.top_k_capacity
                    && tracking_kernel_plan_is_valid
                    && sampled_kernel_plan_is_valid
                    && feedback_control_kernel_is_valid => {}
            _ => {
                return Err(VulkanResidentSamplerRunnerError::InvalidSamplingSpec {
                    method: spec.method.clone(),
                    temperature: spec.temperature,
                    top_k: spec.top_k,
                    top_p: spec.top_p,
                    min_p: spec.min_p,
                    presence_penalty: spec.presence_penalty,
                    repetition_penalty: spec.repetition_penalty,
                });
            }
        }
        let history_byte_capacity = history_capacity_activations
            .checked_mul(VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY)
            .and_then(|bytes| bytes.checked_add(spec.output_byte_capacity))
            .ok_or(VulkanResidentSamplerRunnerError::HistoryCapacityOverflow)?;
        let mut output_buffer =
            device.create_host_visible_resident_buffer(history_byte_capacity)?;
        output_buffer.persistently_map()?;
        let sampler_seed_buffer = if spec.method == "temperature_top_k_top_p" {
            let buffer = device.create_host_visible_resident_buffer(4)?;
            buffer.write_bytes(&random_seed.to_le_bytes())?;
            Some(buffer)
        } else {
            None
        };
        let scratch_buffer = (spec.method == "temperature_top_k_top_p")
            .then(|| device.create_resident_buffer(spec.scratch_byte_capacity))
            .transpose()?;
        let seen_token_byte_capacity = vocabulary_size
            .div_ceil(32)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(VulkanResidentSamplerRunnerError::HistoryCapacityOverflow)?;
        let seen_token_buffer = if token_state_is_active || runtime_parameterized {
            let buffer = device.create_resident_buffer(seen_token_byte_capacity)?;
            buffer.write_bytes(&vec![0; seen_token_byte_capacity])?;
            Some(buffer)
        } else {
            None
        };
        let seen_token_snapshot_buffer = if token_state_is_active {
            Some(device.create_resident_buffer(seen_token_byte_capacity)?)
        } else {
            None
        };
        let capture_seen_token_copy = seen_token_buffer
            .as_ref()
            .zip(seen_token_snapshot_buffer.as_ref())
            .map(|(source, destination)| {
                device.create_resident_buffer_copy(source, destination, seen_token_byte_capacity)
            })
            .transpose()?;
        let restore_seen_token_copy = seen_token_snapshot_buffer
            .as_ref()
            .zip(seen_token_buffer.as_ref())
            .map(|(source, destination)| {
                device.create_resident_buffer_copy(source, destination, seen_token_byte_capacity)
            })
            .transpose()?;
        let seen_token_batch_buffer = if token_state_is_active {
            let mut buffer = device.create_host_visible_resident_buffer(
                VULKAN_BACKEND_LOOP_MAX_WINDOW * std::mem::size_of::<u32>(),
            )?;
            buffer.persistently_map()?;
            Some(buffer)
        } else {
            None
        };
        let sampler_parameter_buffer = if runtime_parameterized {
            let words = [
                spec.temperature.to_bits(),
                spec.top_k,
                spec.top_p.to_bits(),
                spec.min_p.to_bits(),
                spec.presence_penalty.to_bits(),
                spec.repetition_penalty.to_bits(),
            ];
            let bytes = words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            let buffer = device.create_resident_buffer(bytes.len())?;
            buffer.write_bytes(&bytes)?;
            Some(buffer)
        } else {
            None
        };
        let feedback_control = match feedback_control {
            Some(feedback_control) => feedback_control,
            None => {
                let byte_capacity =
                    (VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + 1) * size_of::<u32>();
                let buffer = device.create_resident_buffer(byte_capacity)?;
                buffer.write_bytes(&vec![0; byte_capacity])?;
                VulkanResidentSamplerFeedbackControlBindings {
                    control_buffer: Arc::new(buffer),
                    stop_mask_byte_offset: VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT
                        * size_of::<u32>(),
                    stop_mask_byte_capacity: size_of::<u32>(),
                }
            }
        };
        let mut input_tracking_dispatches = Vec::with_capacity(usize::from(token_state_is_active));
        if token_state_is_active && let Some(kernel) = current_token_kernel {
            input_tracking_dispatches.push(
                device.create_resident_kernel_dispatch(
                    &kernel.spirv_words,
                    &[
                        VulkanResidentKernelBufferBinding::new(
                            0,
                            &stream_control_buffer,
                            VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                        VulkanResidentKernelBufferBinding::new(
                            1,
                            seen_token_buffer
                                .as_ref()
                                .expect("validated repetition sampler has seen-token state"),
                            seen_token_byte_capacity,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                    ],
                    kernel.workgroup_count_x,
                    kernel.local_size_x,
                    0,
                )?,
            );
        }
        let seen_token_batch_dispatch = token_state_is_active
            .then_some(token_batch_kernel)
            .flatten()
            .map(|kernel| {
                device.create_resident_kernel_dispatch(
                    &kernel.spirv_words,
                    &[
                        VulkanResidentKernelBufferBinding::new(
                            0,
                            seen_token_batch_buffer
                                .as_ref()
                                .expect("validated repetition sampler has batch input state"),
                            VULKAN_BACKEND_LOOP_MAX_WINDOW * std::mem::size_of::<u32>(),
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                        VulkanResidentKernelBufferBinding::new(
                            1,
                            seen_token_buffer
                                .as_ref()
                                .expect("validated repetition sampler has seen-token state"),
                            seen_token_byte_capacity,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                    ],
                    kernel.workgroup_count_x,
                    kernel.local_size_x,
                    std::mem::size_of::<u32>() as u32,
                )
            })
            .transpose()?;
        let seen_token_batch_sequence = seen_token_batch_dispatch
            .as_ref()
            .map(|_| device.create_resident_kernel_sequence())
            .transpose()?;
        let mut resident_dispatches = Vec::with_capacity(sampling_kernels.len());
        for kernel in sampling_kernels {
            let mut bindings = match kernel.role.as_str() {
                "sample_logits" | "runtime_sample_logits" => vec![
                    VulkanResidentKernelBufferBinding::new(0, logits_buffer, logits_byte_capacity)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &output_buffer,
                        history_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        2,
                        &stream_control_buffer,
                        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                ],
                "partition_top_k" | "runtime_partition_top_k" => vec![
                    VulkanResidentKernelBufferBinding::new(0, logits_buffer, logits_byte_capacity)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        scratch_buffer.as_ref().expect("sampling plan has scratch"),
                        spec.scratch_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                ],
                "sample_candidates" | "runtime_sample_candidates" => vec![
                    VulkanResidentKernelBufferBinding::new(
                        0,
                        scratch_buffer.as_ref().expect("sampling plan has scratch"),
                        spec.scratch_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &output_buffer,
                        history_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        2,
                        &stream_control_buffer,
                        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                    VulkanResidentKernelBufferBinding::new(
                        3,
                        sampler_seed_buffer
                            .as_ref()
                            .expect("sampled plan has a seed buffer"),
                        4,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                ],
                _ => unreachable!("sampling plan roles were validated"),
            };
            if let Some(seen_token_buffer) = &seen_token_buffer {
                let binding = match kernel.role.as_str() {
                    "sample_logits" | "runtime_sample_logits" => 3,
                    "partition_top_k" | "runtime_partition_top_k" => 2,
                    _ => u32::MAX,
                };
                if binding != u32::MAX {
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            binding,
                            seen_token_buffer,
                            seen_token_byte_capacity,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
            }
            if let Some(parameter_buffer) = &sampler_parameter_buffer {
                let binding = match kernel.role.as_str() {
                    "runtime_sample_logits" => 4,
                    "runtime_partition_top_k" => 3,
                    "runtime_sample_candidates" => 4,
                    _ => u32::MAX,
                };
                if binding != u32::MAX {
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(binding, parameter_buffer, 24)
                            .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
            }
            if matches!(
                kernel.role.as_str(),
                "sample_logits"
                    | "runtime_sample_logits"
                    | "sample_candidates"
                    | "runtime_sample_candidates"
            ) {
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        7,
                        &feedback_control.control_buffer,
                        feedback_control.control_buffer.byte_capacity(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                );
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        8,
                        &feedback_control.control_buffer,
                        feedback_control.stop_mask_byte_capacity,
                    )
                    .with_byte_offset(feedback_control.stop_mask_byte_offset)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                );
            }
            resident_dispatches.push(device.create_resident_kernel_dispatch(
                &kernel.spirv_words,
                &bindings,
                kernel.workgroup_count_x,
                kernel.local_size_x,
                0,
            )?);
        }
        let feedback_control_kernel =
            feedback_control_kernel.expect("feedback control kernel plan was validated");
        let feedback_control_dispatch = device.create_resident_kernel_dispatch(
            &feedback_control_kernel.spirv_words,
            &[
                VulkanResidentKernelBufferBinding::new(
                    0,
                    &feedback_control.control_buffer,
                    feedback_control.control_buffer.byte_capacity(),
                )
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                VulkanResidentKernelBufferBinding::new(
                    1,
                    &stream_control_buffer,
                    VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                )
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
            ],
            feedback_control_kernel.workgroup_count_x,
            feedback_control_kernel.local_size_x,
            0,
        )?;
        let descriptor_count = resident_dispatches
            .iter()
            .chain(input_tracking_dispatches.iter())
            .map(VulkanResidentKernelDispatch::descriptor_count)
            .sum();
        let workgroup_count_x = resident_dispatches
            .iter()
            .chain(input_tracking_dispatches.iter())
            .try_fold(0u32, |total, dispatch| {
                total.checked_add(dispatch.workgroup_count_x())
            })
            .ok_or(VulkanResidentSamplerRunnerError::WorkgroupCountOverflow)?;
        let push_constant_byte_count = resident_dispatches
            .iter()
            .chain(input_tracking_dispatches.iter())
            .try_fold(0u32, |total, dispatch| {
                total.checked_add(dispatch.push_constant_byte_count())
            })
            .ok_or(VulkanResidentSamplerRunnerError::PushConstantByteCountOverflow)?;
        let sequence = device.create_resident_kernel_sequence()?;

        Ok(Self {
            sampler_id: spec.sampler_id.clone(),
            logits_byte_capacity,
            output_byte_capacity: spec.output_byte_capacity,
            dispatch_count: resident_dispatches.len() + input_tracking_dispatches.len(),
            descriptor_count,
            workgroup_count_x,
            push_constant_byte_count,
            history_capacity_activations,
            output_buffer,
            _scratch_buffer: scratch_buffer,
            _sampler_seed_buffer: sampler_seed_buffer,
            _seen_token_buffer: seen_token_buffer,
            _seen_token_snapshot_buffer: seen_token_snapshot_buffer,
            capture_seen_token_copy,
            restore_seen_token_copy,
            _sampler_parameter_buffer: sampler_parameter_buffer,
            _feedback_control_buffer: feedback_control.control_buffer,
            seen_token_batch_buffer,
            _stream_control_buffer: stream_control_buffer,
            input_tracking_dispatches,
            seen_token_batch_dispatch,
            seen_token_batch_sequence,
            resident_dispatches,
            feedback_control_dispatch,
            sequence,
        })
    }

    fn feedback_dispatch_count_for_spec(
        kernels: &[VulkanResidentSamplerKernelArtifact],
        spec: &VulkanResidentSamplerSpec,
    ) -> usize {
        let sampling = kernels
            .iter()
            .filter(|kernel| {
                sampler_kernel_role_matches(
                    kernel.role.as_str(),
                    spec.runtime_parameterized,
                    spec.method.as_str(),
                )
            })
            .count();
        sampling
            + usize::from(spec.repetition_penalty != 1.0 || spec.presence_penalty != 0.0)
            + 1
    }

    pub fn run(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentSamplerRun, VulkanResidentSamplerRunnerError> {
        let steps = self
            .resident_dispatches
            .iter()
            .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[]))
            .collect::<Vec<_>>();
        device.run_resident_kernel_sequence(&self.sequence, &steps)?;
        self.completed_run()
    }

    fn resident_dispatches(&self) -> &[VulkanResidentKernelDispatch] {
        &self.resident_dispatches
    }

    fn feedback_control_dispatch(&self) -> &VulkanResidentKernelDispatch {
        &self.feedback_control_dispatch
    }

    fn input_tracking_dispatches(&self) -> &[VulkanResidentKernelDispatch] {
        &self.input_tracking_dispatches
    }

    fn record_input_tokens(
        &self,
        device: &VulkanComputeDevice,
        token_ids: &[u32],
    ) -> Result<(), VulkanResidentSamplerRunnerError> {
        let Some(dispatch) = &self.seen_token_batch_dispatch else {
            return Ok(());
        };
        if token_ids.is_empty() || token_ids.len() > VULKAN_BACKEND_LOOP_MAX_WINDOW {
            return Err(
                VulkanResidentSamplerRunnerError::TokenBatchCapacityExceeded {
                    requested: token_ids.len(),
                    capacity: VULKAN_BACKEND_LOOP_MAX_WINDOW,
                },
            );
        }
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(token_ids));
        for token_id in token_ids {
            bytes.extend_from_slice(&token_id.to_le_bytes());
        }
        self.seen_token_batch_buffer
            .as_ref()
            .expect("repetition token batch dispatch has an input buffer")
            .write_bytes(&bytes)?;
        device.run_resident_kernel_sequence(
            self.seen_token_batch_sequence
                .as_ref()
                .expect("repetition token batch dispatch has a sequence"),
            &[VulkanResidentKernelSequenceStep::new(
                dispatch,
                &u32::try_from(token_ids.len())
                    .map_err(
                        |_| VulkanResidentSamplerRunnerError::TokenBatchCapacityExceeded {
                            requested: token_ids.len(),
                            capacity: VULKAN_BACKEND_LOOP_MAX_WINDOW,
                        },
                    )?
                    .to_le_bytes(),
            )],
        )?;
        Ok(())
    }

    fn capture_token_state(&self) -> Result<(), VulkanResidentSamplerRunnerError> {
        if let Some(copy) = &self.capture_seen_token_copy {
            copy.run(copy.byte_len())?;
        }
        Ok(())
    }

    fn restore_token_state(&self) -> Result<(), VulkanResidentSamplerRunnerError> {
        if let Some(copy) = &self.restore_seen_token_copy {
            copy.run(copy.byte_len())?;
        }
        Ok(())
    }

    fn completed_run(&self) -> Result<VulkanResidentSamplerRun, VulkanResidentSamplerRunnerError> {
        let output = self.output_buffer.read_bytes(self.output_byte_capacity)?;
        let token_id = u32::from_le_bytes([output[0], output[1], output[2], output[3]]);
        let selected_logit_bits = u32::from_le_bytes([output[4], output[5], output[6], output[7]]);
        let control_flags = u32::from_le_bytes([output[8], output[9], output[10], output[11]]);
        Ok(VulkanResidentSamplerRun {
            sampler_id: self.sampler_id.clone(),
            token_id,
            selected_logit_bits,
            control_flags,
            descriptor_count: self.descriptor_count,
            workgroup_count_x: self.workgroup_count_x,
            push_constant_byte_count: self.push_constant_byte_count,
        })
    }

    fn completed_token_id(&self) -> Result<u32, VulkanResidentSamplerRunnerError> {
        self.output_buffer
            .read_persistently_mapped_u32_le_at(0)
            .map_err(VulkanResidentSamplerRunnerError::Vulkan)
    }

    fn completed_run_at(
        &self,
        stream_tick: u64,
    ) -> Result<VulkanResidentSamplerRun, VulkanResidentSamplerRunnerError> {
        let slot = (stream_tick as u32 as usize) % self.history_capacity_activations;
        let offset = slot
            .checked_mul(VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY)
            .and_then(|bytes| bytes.checked_add(self.output_byte_capacity))
            .ok_or(VulkanResidentSamplerRunnerError::HistoryCapacityOverflow)?;
        let output = self
            .output_buffer
            .read_bytes_at(offset, VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY)?;
        Ok(VulkanResidentSamplerRun {
            sampler_id: self.sampler_id.clone(),
            token_id: u32::from_le_bytes(output[0..4].try_into().unwrap()),
            selected_logit_bits: u32::from_le_bytes(output[4..8].try_into().unwrap()),
            control_flags: u32::from_le_bytes(output[8..12].try_into().unwrap()),
            descriptor_count: self.descriptor_count,
            workgroup_count_x: self.workgroup_count_x,
            push_constant_byte_count: self.push_constant_byte_count,
        })
    }

    pub fn read_output_bytes(&self) -> Result<Vec<u8>, VulkanError> {
        self.output_buffer.read_bytes(self.output_byte_capacity)
    }

    fn create_logits_view(
        &self,
        device: &VulkanComputeDevice,
        logits_buffer: &VulkanResidentBuffer,
        logits_byte_offset: usize,
        kernels: &[VulkanResidentSamplerKernelArtifact],
        spec: &VulkanResidentSamplerSpec,
    ) -> Result<VulkanResidentSamplerLogitsView, VulkanResidentSamplerRunnerError> {
        let scratch_buffer = (spec.method == "temperature_top_k_top_p")
            .then(|| device.create_resident_buffer(spec.scratch_byte_capacity))
            .transpose()?;
        let mut stream_control_buffer =
            device.create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)?;
        stream_control_buffer.persistently_map()?;
        stream_control_buffer.write_bytes(&[0; VULKAN_STREAM_CONTROL_BYTE_CAPACITY])?;
        let runtime_parameterized = spec.runtime_parameterized;
        let seen_token_buffer = self
            ._seen_token_buffer
            .as_ref()
            .map(|source| {
                let destination = device.create_resident_buffer(source.byte_capacity())?;
                destination.write_bytes(&vec![0; source.byte_capacity()])?;
                Ok::<_, VulkanError>(destination)
            })
            .transpose()?;
        let seen_token_copy = self
            ._seen_token_buffer
            .as_ref()
            .zip(seen_token_buffer.as_ref())
            .map(|(source, destination)| {
                device.create_resident_buffer_copy(source, destination, source.byte_capacity())
            })
            .transpose()?;
        let token_state_is_active = spec.repetition_penalty != 1.0 || spec.presence_penalty != 0.0;
        let token_batch_role = if runtime_parameterized {
            "runtime_record_token_batch"
        } else {
            "record_token_batch"
        };
        let token_batch_kernel = token_state_is_active
            .then(|| {
                kernels
                    .iter()
                    .find(|kernel| kernel.role == token_batch_role)
            })
            .flatten();
        let seen_token_batch_buffer = if token_batch_kernel.is_some() {
            let mut buffer = device.create_host_visible_resident_buffer(
                VULKAN_BACKEND_LOOP_MAX_WINDOW * std::mem::size_of::<u32>(),
            )?;
            buffer.persistently_map()?;
            Some(buffer)
        } else {
            None
        };
        let seen_token_batch_dispatch = token_batch_kernel
            .map(|kernel| {
                device.create_resident_kernel_dispatch(
                    &kernel.spirv_words,
                    &[
                        VulkanResidentKernelBufferBinding::new(
                            0,
                            seen_token_batch_buffer
                                .as_ref()
                                .expect("token-state view has prefix input buffer"),
                            VULKAN_BACKEND_LOOP_MAX_WINDOW * std::mem::size_of::<u32>(),
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                        VulkanResidentKernelBufferBinding::new(
                            1,
                            seen_token_buffer
                                .as_ref()
                                .expect("token-state view has private seen-token state"),
                            seen_token_buffer
                                .as_ref()
                                .expect("token-state view has private seen-token state")
                                .byte_capacity(),
                        )
                        .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                    ],
                    kernel.workgroup_count_x,
                    kernel.local_size_x,
                    std::mem::size_of::<u32>() as u32,
                )
            })
            .transpose()?;
        let seen_token_batch_sequence = seen_token_batch_dispatch
            .as_ref()
            .map(|_| device.create_resident_kernel_sequence())
            .transpose()?;
        let sampling_kernels = kernels.iter().filter(|kernel| {
            sampler_kernel_role_matches(
                kernel.role.as_str(),
                runtime_parameterized,
                spec.method.as_str(),
            )
        });
        let feedback_control_byte_capacity =
            (VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + 1) * size_of::<u32>();
        let feedback_control_buffer =
            Arc::new(device.create_resident_buffer(feedback_control_byte_capacity)?);
        feedback_control_buffer.write_bytes(&vec![0; feedback_control_byte_capacity])?;
        let mut resident_dispatches = Vec::new();
        for kernel in sampling_kernels {
            let logits_binding = || {
                VulkanResidentKernelBufferBinding::new(0, logits_buffer, self.logits_byte_capacity)
                    .with_byte_offset(logits_byte_offset)
                    .with_access(VulkanResidentKernelBufferAccess::Read)
            };
            let mut bindings = match kernel.role.as_str() {
                "sample_logits" | "runtime_sample_logits" => vec![
                    logits_binding(),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &self.output_buffer,
                        self.output_buffer.byte_capacity(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        2,
                        &stream_control_buffer,
                        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                ],
                "partition_top_k" | "runtime_partition_top_k" => vec![
                    logits_binding(),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        scratch_buffer
                            .as_ref()
                            .expect("validated sampling plan has scratch"),
                        spec.scratch_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                ],
                "sample_candidates" | "runtime_sample_candidates" => vec![
                    VulkanResidentKernelBufferBinding::new(
                        0,
                        scratch_buffer
                            .as_ref()
                            .expect("validated sampling plan has scratch"),
                        spec.scratch_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &self.output_buffer,
                        self.output_buffer.byte_capacity(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        2,
                        &stream_control_buffer,
                        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                    VulkanResidentKernelBufferBinding::new(
                        3,
                        self._sampler_seed_buffer
                            .as_ref()
                            .expect("validated sampled plan has a seed buffer"),
                        4,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                ],
                role => {
                    return Err(VulkanResidentSamplerRunnerError::InvalidKernelRole(
                        role.to_string(),
                    ));
                }
            };
            if let Some(seen_token_buffer) = &seen_token_buffer {
                let binding = match kernel.role.as_str() {
                    "sample_logits" | "runtime_sample_logits" => Some(3),
                    "partition_top_k" | "runtime_partition_top_k" => Some(2),
                    _ => None,
                };
                if let Some(binding) = binding {
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            binding,
                            seen_token_buffer,
                            seen_token_buffer.byte_capacity(),
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
            }
            if let Some(parameter_buffer) = &self._sampler_parameter_buffer {
                let binding = match kernel.role.as_str() {
                    "runtime_sample_logits" => Some(4),
                    "runtime_partition_top_k" => Some(3),
                    "runtime_sample_candidates" => Some(4),
                    _ => None,
                };
                if let Some(binding) = binding {
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(binding, parameter_buffer, 24)
                            .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
            }
            if matches!(
                kernel.role.as_str(),
                "sample_logits"
                    | "runtime_sample_logits"
                    | "sample_candidates"
                    | "runtime_sample_candidates"
            ) {
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        7,
                        &feedback_control_buffer,
                        feedback_control_byte_capacity,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                );
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        8,
                        &feedback_control_buffer,
                        size_of::<u32>(),
                    )
                    .with_byte_offset(
                        VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT * size_of::<u32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                );
            }
            resident_dispatches.push(device.create_resident_kernel_dispatch(
                &kernel.spirv_words,
                &bindings,
                kernel.workgroup_count_x,
                kernel.local_size_x,
                0,
            )?);
        }
        Ok(VulkanResidentSamplerLogitsView {
            resident_dispatches,
            sequence: device.create_resident_kernel_sequence()?,
            _scratch_buffer: scratch_buffer,
            stream_control_buffer,
            _feedback_control_buffer: feedback_control_buffer,
            _seen_token_buffer: seen_token_buffer,
            seen_token_copy,
            seen_token_batch_buffer,
            seen_token_batch_dispatch,
            seen_token_batch_sequence,
        })
    }
}

struct VulkanResidentSamplerLogitsView {
    resident_dispatches: Vec<VulkanResidentKernelDispatch>,
    sequence: VulkanResidentKernelSequence,
    _scratch_buffer: Option<VulkanResidentBuffer>,
    stream_control_buffer: VulkanResidentBuffer,
    _feedback_control_buffer: Arc<VulkanResidentBuffer>,
    _seen_token_buffer: Option<VulkanResidentBuffer>,
    seen_token_copy: Option<VulkanResidentBufferCopy>,
    seen_token_batch_buffer: Option<VulkanResidentBuffer>,
    seen_token_batch_dispatch: Option<VulkanResidentKernelDispatch>,
    seen_token_batch_sequence: Option<VulkanResidentKernelSequence>,
}

impl VulkanResidentSamplerLogitsView {
    fn prepare_token_state(
        &self,
        device: &VulkanComputeDevice,
        prefix_token_ids: &[u32],
    ) -> Result<(), VulkanResidentSamplerRunnerError> {
        if let Some(copy) = &self.seen_token_copy {
            copy.run(copy.byte_len())?;
        }
        if prefix_token_ids.is_empty() {
            return Ok(());
        }
        let Some(dispatch) = &self.seen_token_batch_dispatch else {
            return Ok(());
        };
        if prefix_token_ids.len() > VULKAN_BACKEND_LOOP_MAX_WINDOW {
            return Err(
                VulkanResidentSamplerRunnerError::TokenBatchCapacityExceeded {
                    requested: prefix_token_ids.len(),
                    capacity: VULKAN_BACKEND_LOOP_MAX_WINDOW,
                },
            );
        }
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(prefix_token_ids));
        for token_id in prefix_token_ids {
            bytes.extend_from_slice(&token_id.to_le_bytes());
        }
        self.seen_token_batch_buffer
            .as_ref()
            .expect("token-state view tracker has input buffer")
            .write_bytes(&bytes)?;
        device.run_resident_kernel_sequence(
            self.seen_token_batch_sequence
                .as_ref()
                .expect("token-state view tracker has sequence"),
            &[VulkanResidentKernelSequenceStep::new(
                dispatch,
                &u32::try_from(prefix_token_ids.len())
                    .map_err(
                        |_| VulkanResidentSamplerRunnerError::TokenBatchCapacityExceeded {
                            requested: prefix_token_ids.len(),
                            capacity: VULKAN_BACKEND_LOOP_MAX_WINDOW,
                        },
                    )?
                    .to_le_bytes(),
            )],
        )?;
        Ok(())
    }

    fn prepare_stream_tick(
        &self,
        stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanError> {
        self.stream_control_buffer
            .write_bytes(&stream_control_bytes(
                0,
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            ))
    }

    fn record(&self, device: &VulkanComputeDevice) -> Result<(), VulkanError> {
        let steps = self
            .resident_dispatches
            .iter()
            .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[]))
            .collect::<Vec<_>>();
        device.record_resident_kernel_sequence(&self.sequence, &steps)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentSamplerRun {
    pub sampler_id: String,
    pub token_id: u32,
    pub selected_logit_bits: u32,
    pub control_flags: u32,
    pub descriptor_count: usize,
    pub workgroup_count_x: u32,
    pub push_constant_byte_count: u32,
}

#[derive(Debug)]
pub enum VulkanResidentSamplerRunnerError {
    InvalidLogitsByteCapacity {
        byte_capacity: usize,
        expected_byte_capacity: usize,
    },
    ZeroHistoryCapacity,
    HistoryCapacityOverflow,
    WorkgroupCountOverflow,
    PushConstantByteCountOverflow,
    TokenBatchCapacityExceeded {
        requested: usize,
        capacity: usize,
    },
    InvalidKernelRole(String),
    UnsupportedRuntimeSamplingOverride(String),
    InvalidSamplingSpec {
        method: String,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        presence_penalty: f32,
        repetition_penalty: f32,
    },
    Vulkan(VulkanError),
}

impl Display for VulkanResidentSamplerRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLogitsByteCapacity {
                byte_capacity,
                expected_byte_capacity,
            } => write!(
                f,
                "sampler logits buffer has {byte_capacity} bytes, expected {expected_byte_capacity}"
            ),
            Self::ZeroHistoryCapacity => f.write_str("sampler history capacity must be nonzero"),
            Self::HistoryCapacityOverflow => {
                f.write_str("sampler history byte capacity overflowed")
            }
            Self::WorkgroupCountOverflow => f.write_str("sampler workgroup count overflowed"),
            Self::PushConstantByteCountOverflow => {
                f.write_str("sampler push constant byte count overflowed")
            }
            Self::TokenBatchCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "sampler repetition-state batch requests {requested} tokens, capacity is {capacity}"
            ),
            Self::InvalidKernelRole(role) => {
                write!(f, "invalid resident sampler kernel role {role:?}")
            }
            Self::UnsupportedRuntimeSamplingOverride(message) => f.write_str(message),
            Self::InvalidSamplingSpec {
                method,
                temperature,
                top_k,
                top_p,
                min_p,
                presence_penalty,
                repetition_penalty,
            } => write!(
                f,
                "invalid resident sampling spec method={method:?} temperature={temperature} top_k={top_k} top_p={top_p} min_p={min_p} presence_penalty={presence_penalty} repetition_penalty={repetition_penalty}"
            ),
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentSamplerRunnerError {}

impl From<VulkanError> for VulkanResidentSamplerRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}
