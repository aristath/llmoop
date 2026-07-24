const VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT: usize = 12;
const VULKAN_FEEDBACK_CONTROL_ENABLED: u32 = 1 << 0;
const VULKAN_FEEDBACK_CONTROL_CANCEL_REQUESTED: u32 = 1 << 1;
const VULKAN_FEEDBACK_STOP_REASON_NONE: u32 = 0;
const VULKAN_FEEDBACK_STOP_REASON_EOS: u32 = 1;
const VULKAN_FEEDBACK_STOP_REASON_CANCELLED: u32 = 2;

pub(crate) struct VulkanResidentFeedbackControlPlane {
    vocabulary_size: usize,
    stop_mask_word_count: usize,
    dispatch_word_offset: usize,
    dispatch_capacity: usize,
    dispatch_dimensions: Vec<[u32; 3]>,
    generation_tail: Option<(usize, usize)>,
    buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    host_buffer_device_id: String,
}

#[derive(Clone)]
pub(crate) struct VulkanResidentFeedbackIndirectSequence {
    pub(crate) buffer: Arc<VulkanResidentBuffer>,
    pub(crate) byte_offsets: Vec<usize>,
    pub(crate) first_dispatch_index: usize,
}

#[derive(Clone)]
struct VulkanResidentSamplerFeedbackControlBindings {
    control_buffer: Arc<VulkanResidentBuffer>,
    stop_mask_byte_offset: usize,
    stop_mask_byte_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentFeedbackControlCompletion {
    executed_tick_count: usize,
    sampled_tick_count: usize,
    stop_reason: u32,
    template_replayed: bool,
}

impl VulkanResidentFeedbackControlPlane {
    fn new<'a, F, E>(
        device_ids: &[String],
        output_device_id: &str,
        vocabulary_size: usize,
        dispatch_capacity: usize,
        device_for: &F,
    ) -> Result<Self, VulkanError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        if vocabulary_size == 0 {
            return Err(VulkanError(
                "resident feedback control requires a nonzero vocabulary".to_string(),
            ));
        }
        if dispatch_capacity == 0 {
            return Err(VulkanError(
                "resident feedback control requires at least one dispatch".to_string(),
            ));
        }
        let stop_mask_word_count = vocabulary_size.div_ceil(u32::BITS as usize);
        let dispatch_word_offset = VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT
            .checked_add(stop_mask_word_count)
            .ok_or_else(|| {
                VulkanError("resident feedback control mask size overflowed".to_string())
            })?;
        let dispatch_word_count = dispatch_capacity
            .checked_mul(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT / size_of::<u32>())
            .ok_or_else(|| {
                VulkanError("resident feedback indirect table size overflowed".to_string())
            })?;
        let byte_capacity = dispatch_word_offset
            .checked_add(dispatch_word_count)
            .and_then(|words| words.checked_mul(size_of::<u32>()))
            .ok_or_else(|| {
                VulkanError("resident feedback control buffer size overflowed".to_string())
            })?;
        let resolved_devices = device_ids
            .iter()
            .map(|device_id| {
                device_for(device_id)
                    .map(|device| (device_id.as_str(), device))
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to resolve resident feedback device {device_id:?}: {error}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_device = resolved_devices
            .iter()
            .find(|(device_id, _)| *device_id == output_device_id)
            .map(|(_, device)| *device)
            .ok_or_else(|| {
                VulkanError(format!(
                    "resident feedback output device {output_device_id:?} is not bound"
                ))
            })?;

        let mut unique_devices = Vec::<(&str, &VulkanComputeDevice)>::new();
        for (device_id, device) in &resolved_devices {
            if unique_devices
                .iter()
                .all(|(_, existing)| !existing.shares_logical_device_with(device))
            {
                unique_devices.push((*device_id, *device));
            }
        }
        let mut buffers = BTreeMap::new();
        if unique_devices.len() == 1 {
            let mut buffer = output_device.create_host_visible_resident_buffer(byte_capacity)?;
            buffer.persistently_map()?;
            let buffer = Arc::new(buffer);
            for (device_id, _) in &resolved_devices {
                buffers.insert((*device_id).to_string(), Arc::clone(&buffer));
            }
        } else {
            let peers = unique_devices
                .iter()
                .map(|(_, device)| *device)
                .filter(|device| !device.shares_logical_device_with(output_device))
                .collect::<Vec<_>>();
            let allocation =
                output_device.create_shared_host_allocation(&peers, byte_capacity)?;
            let mut imported = Vec::<(&VulkanComputeDevice, Arc<VulkanResidentBuffer>)>::new();
            for (_, device) in &unique_devices {
                imported.push((
                    *device,
                    Arc::new(device.import_shared_host_buffer(Arc::clone(&allocation))?),
                ));
            }
            for (device_id, device) in &resolved_devices {
                let buffer = imported
                    .iter()
                    .find(|(imported_device, _)| {
                        imported_device.shares_logical_device_with(device)
                    })
                    .map(|(_, buffer)| Arc::clone(buffer))
                    .ok_or_else(|| {
                        VulkanError(format!(
                            "resident feedback control has no buffer import for {device_id:?}"
                        ))
                    })?;
                buffers.insert((*device_id).to_string(), buffer);
            }
        }

        Ok(Self {
            vocabulary_size,
            stop_mask_word_count,
            dispatch_word_offset,
            dispatch_capacity,
            dispatch_dimensions: Vec::with_capacity(dispatch_capacity),
            generation_tail: None,
            buffers,
            host_buffer_device_id: output_device_id.to_string(),
        })
    }

    fn sampler_bindings(
        &self,
    ) -> Result<VulkanResidentSamplerFeedbackControlBindings, VulkanError> {
        let control_buffer = self
            .buffers
            .get(&self.host_buffer_device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanError("resident feedback control host buffer disappeared".to_string())
            })?;
        Ok(VulkanResidentSamplerFeedbackControlBindings {
            control_buffer,
            stop_mask_byte_offset: VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT * size_of::<u32>(),
            stop_mask_byte_capacity: self.stop_mask_word_count * size_of::<u32>(),
        })
    }

    pub(crate) fn register_sequence<'a>(
        &mut self,
        device_id: &str,
        dispatches: impl IntoIterator<Item = &'a VulkanResidentKernelDispatch>,
    ) -> Result<VulkanResidentFeedbackIndirectSequence, VulkanError> {
        let buffer = self.buffers.get(device_id).cloned().ok_or_else(|| {
            VulkanError(format!(
                "resident feedback control has no buffer for {device_id:?}"
            ))
        })?;
        let mut byte_offsets = Vec::new();
        let first_dispatch_index = self.dispatch_dimensions.len();
        for dispatch in dispatches {
            if self.dispatch_dimensions.len() == self.dispatch_capacity {
                return Err(VulkanError(format!(
                    "resident feedback control registered more than {} dispatches",
                    self.dispatch_capacity
                )));
            }
            let dispatch_index = self.dispatch_dimensions.len();
            let byte_offset = self
                .dispatch_word_offset
                .checked_mul(size_of::<u32>())
                .and_then(|base| {
                    dispatch_index
                        .checked_mul(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
                        .and_then(|offset| base.checked_add(offset))
                })
                .ok_or_else(|| {
                    VulkanError("resident feedback indirect offset overflowed".to_string())
                })?;
            self.dispatch_dimensions.push([
                dispatch.workgroup_count_x(),
                dispatch.workgroup_count_y(),
                1,
            ]);
            byte_offsets.push(byte_offset);
        }
        if byte_offsets.is_empty() {
            return Err(VulkanError(
                "resident feedback indirect sequence must not be empty".to_string(),
            ));
        }
        Ok(VulkanResidentFeedbackIndirectSequence {
            buffer,
            byte_offsets,
            first_dispatch_index,
        })
    }

    fn set_generation_tail(
        &mut self,
        first_dispatch_index: usize,
        dispatch_count: usize,
    ) -> Result<(), VulkanError> {
        if dispatch_count == 0 {
            return Err(VulkanError(
                "resident feedback generation tail must not be empty".to_string(),
            ));
        }
        let end = first_dispatch_index
            .checked_add(dispatch_count)
            .ok_or_else(|| {
                VulkanError("resident feedback generation tail overflowed".to_string())
            })?;
        if end > self.dispatch_capacity {
            return Err(VulkanError(format!(
                "resident feedback generation tail {first_dispatch_index}..{end} exceeds dispatch capacity {}",
                self.dispatch_capacity
            )));
        }
        if self
            .generation_tail
            .replace((first_dispatch_index, dispatch_count))
            .is_some()
        {
            return Err(VulkanError(
                "resident feedback generation tail was registered twice".to_string(),
            ));
        }
        Ok(())
    }

    fn finish_registration(&self) -> Result<(), VulkanError> {
        if self.dispatch_dimensions.len() != self.dispatch_capacity {
            return Err(VulkanError(format!(
                "resident feedback control registered {} of {} dispatches",
                self.dispatch_dimensions.len(),
                self.dispatch_capacity
            )));
        }
        if self.generation_tail.is_none() {
            return Err(VulkanError(
                "resident feedback control has no generation tail".to_string(),
            ));
        }
        Ok(())
    }

    fn arm(
        &self,
        planned_tick_count: usize,
        stop_token_ids: &[u32],
    ) -> Result<(), VulkanError> {
        if planned_tick_count == 0 {
            return Err(VulkanError(
                "resident feedback control cannot arm an empty window".to_string(),
            ));
        }
        self.finish_registration()?;
        let (generation_tail_start, generation_tail_count) = self
            .generation_tail
            .expect("resident feedback registration was validated");
        let words = resident_feedback_control_words(
            self.vocabulary_size,
            self.stop_mask_word_count,
            self.dispatch_word_offset,
            &self.dispatch_dimensions,
            generation_tail_start,
            generation_tail_count,
            planned_tick_count,
            stop_token_ids,
        )?;
        let bytes = words.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        self.host_buffer()?.write_bytes(&bytes)
    }

    fn completion(&self) -> Result<VulkanResidentFeedbackControlCompletion, VulkanError> {
        let buffer = self.host_buffer()?;
        Ok(VulkanResidentFeedbackControlCompletion {
            executed_tick_count: buffer.read_persistently_mapped_u32_le_at(size_of::<u32>())?
                as usize,
            sampled_tick_count: buffer
                .read_persistently_mapped_u32_le_at(9 * size_of::<u32>())?
                as usize,
            stop_reason: buffer.read_persistently_mapped_u32_le_at(2 * size_of::<u32>())?,
            template_replayed: false,
        })
    }

    fn request_cancel(&self) -> Result<(), VulkanError> {
        let buffer = self.host_buffer()?;
        let flags = buffer.read_persistently_mapped_u32_le_at(0)?;
        buffer.write_bytes_at(
            0,
            &(flags | VULKAN_FEEDBACK_CONTROL_CANCEL_REQUESTED).to_le_bytes(),
        )
    }

    fn host_buffer(&self) -> Result<&VulkanResidentBuffer, VulkanError> {
        self.buffers
            .get(&self.host_buffer_device_id)
            .map(Arc::as_ref)
            .ok_or_else(|| {
                VulkanError("resident feedback control host buffer disappeared".to_string())
            })
    }
}

#[allow(clippy::too_many_arguments)]
fn resident_feedback_control_words(
    vocabulary_size: usize,
    stop_mask_word_count: usize,
    dispatch_word_offset: usize,
    dispatch_dimensions: &[[u32; 3]],
    generation_tail_start: usize,
    generation_tail_count: usize,
    planned_tick_count: usize,
    stop_token_ids: &[u32],
) -> Result<Vec<u32>, VulkanError> {
    let planned_tick_count = u32::try_from(planned_tick_count)
        .map_err(|_| VulkanError("resident feedback window width exceeds u32".to_string()))?;
    if dispatch_word_offset
        != VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + stop_mask_word_count
    {
        return Err(VulkanError(
            "resident feedback dispatch table does not follow its stop mask".to_string(),
        ));
    }
    let mut words = vec![
        0u32;
        dispatch_word_offset
            + dispatch_dimensions.len()
                * (VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT / size_of::<u32>())
    ];
    words[0] = VULKAN_FEEDBACK_CONTROL_ENABLED;
    words[2] = VULKAN_FEEDBACK_STOP_REASON_NONE;
    words[3] = planned_tick_count;
    words[4] = u32::try_from(dispatch_word_offset).map_err(|_| {
        VulkanError("resident feedback dispatch word offset exceeds u32".to_string())
    })?;
    words[5] = u32::try_from(dispatch_dimensions.len())
        .map_err(|_| VulkanError("resident feedback dispatch count exceeds u32".to_string()))?;
    words[6] = u32::try_from(generation_tail_start).map_err(|_| {
        VulkanError("resident feedback generation tail offset exceeds u32".to_string())
    })?;
    words[7] = u32::try_from(generation_tail_count).map_err(|_| {
        VulkanError("resident feedback generation tail count exceeds u32".to_string())
    })?;
    for &token_id in stop_token_ids {
        let token_index = usize::try_from(token_id)
            .map_err(|_| VulkanError("stop token id exceeds usize".to_string()))?;
        if token_index >= vocabulary_size {
            return Err(VulkanError(format!(
                "stop token id {token_id} exceeds vocabulary size {vocabulary_size}"
            )));
        }
        let word = VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + token_index / u32::BITS as usize;
        words[word] |= 1u32 << (token_id % u32::BITS);
    }
    for (dispatch_index, dimensions) in dispatch_dimensions.iter().enumerate() {
        let word = dispatch_word_offset + dispatch_index * 3;
        words[word..word + 3].copy_from_slice(dimensions);
    }
    Ok(words)
}
