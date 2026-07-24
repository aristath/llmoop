#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanResidentPlacedPrefixStateRangeKind {
    Static,
    Dynamic {
        block_id: TransientStateBlockId,
        logical_page_index: usize,
        block_activation_capacity: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentPlacedPrefixStateRange {
    key: TransientStateKey,
    device_id: String,
    cache_byte_offset: usize,
    byte_len: usize,
    kind: VulkanResidentPlacedPrefixStateRangeKind,
}

struct VulkanResidentPlacedPrefixStateEntry {
    key: RuntimePrefixStateCacheKey,
    device_buffers: BTreeMap<String, VulkanResidentPlacedPrefixDeviceBuffer>,
    ranges: Vec<VulkanResidentPlacedPrefixStateRange>,
    byte_count: usize,
}

struct VulkanResidentPlacedPrefixDeviceBuffer {
    // Field order is intentional: Vulkan memory must be destroyed before the
    // final device owner is released.
    buffer: VulkanResidentBuffer,
    _device: Rc<VulkanComputeDevice>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentPlacedPrefixStateCacheStats {
    pub hit_count: usize,
    pub miss_count: usize,
    pub insertion_count: usize,
    pub eviction_count: usize,
    pub reused_token_count: usize,
    pub saved_prefill_token_count: usize,
    pub resident_entry_count: usize,
    pub resident_byte_count: usize,
}

#[derive(Default)]
struct VulkanResidentPlacedPrefixStateCache {
    entries: BTreeMap<RuntimePrefixStateCacheKey, VulkanResidentPlacedPrefixStateEntry>,
    stats: VulkanResidentPlacedPrefixStateCacheStats,
}

impl VulkanResidentPlacedPrefixStateCache {
    fn record_miss(&mut self) {
        self.stats.miss_count = self.stats.miss_count.saturating_add(1);
    }

    fn stats(&self) -> VulkanResidentPlacedPrefixStateCacheStats {
        let mut stats = self.stats.clone();
        stats.resident_entry_count = self.entries.len();
        stats.resident_byte_count = self.entries.values().map(|entry| entry.byte_count).sum();
        stats
    }

    fn evict(&mut self, key: &RuntimePrefixStateCacheKey) -> bool {
        let removed = self.entries.remove(key).is_some();
        if removed {
            self.stats.eviction_count = self.stats.eviction_count.saturating_add(1);
        }
        removed
    }

    fn evict_keys(&mut self, keys: &[RuntimePrefixStateCacheKey]) {
        for key in keys {
            self.evict(key);
        }
    }

    fn prepare_capture(
        &self,
        key: RuntimePrefixStateCacheKey,
        source: &VulkanResidentInProcessPlacedPromptStream,
        state: &TransientStateTableSnapshot,
    ) -> Result<
        VulkanResidentPlacedPrefixStateEntry,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut ranges = Vec::new();
        let mut device_byte_counts = BTreeMap::<String, usize>::new();
        for state_entry in &state.entries {
            let (device_id, resident_state) = source
                .processor
                .mounted_state_buffer_with_device_id(&state_entry.key)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "cannot capture non-resident prefix state {}.{}",
                            state_entry.key.node_instance_id, state_entry.key.state_id
                        )),
                    )
                })?;
            if resident_state.layout.static_byte_capacity > 0 {
                let cache_byte_offset = append_cache_range(
                    &mut device_byte_counts,
                    device_id,
                    resident_state.layout.static_byte_capacity,
                )?;
                ranges.push(VulkanResidentPlacedPrefixStateRange {
                    key: state_entry.key.clone(),
                    device_id: device_id.to_string(),
                    cache_byte_offset,
                    byte_len: resident_state.layout.static_byte_capacity,
                    kind: VulkanResidentPlacedPrefixStateRangeKind::Static,
                });
            }
            if state_entry.shape.retention == TransientStateRetention::Append {
                for (logical_page_index, block_id) in
                    state_entry.block_ids.iter().copied().enumerate()
                {
                    let resident_page_index = source
                        .transient_state_pages
                        .resident_page_for_block(&state_entry.key, block_id)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "prefix state block {:?} for {}.{} has no resident page",
                                    block_id,
                                    state_entry.key.node_instance_id,
                                    state_entry.key.state_id
                                )),
                            )
                        })?;
                    resident_state
                        .layout
                        .dynamic_physical_page_offset(resident_page_index)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    let cache_byte_offset = append_cache_range(
                        &mut device_byte_counts,
                        device_id,
                        resident_state.layout.dynamic_page_byte_capacity,
                    )?;
                    ranges.push(VulkanResidentPlacedPrefixStateRange {
                        key: state_entry.key.clone(),
                        device_id: device_id.to_string(),
                        cache_byte_offset,
                        byte_len: resident_state.layout.dynamic_page_byte_capacity,
                        kind: VulkanResidentPlacedPrefixStateRangeKind::Dynamic {
                            block_id,
                            logical_page_index,
                            block_activation_capacity: state_entry.shape.activation_capacity,
                        },
                    });
                }
            }
        }

        let mut device_buffers = BTreeMap::new();
        for (device_id, byte_count) in &device_byte_counts {
            let device = source.devices.get(device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: device_id.clone(),
                }
            })?;
            device_buffers.insert(
                device_id.clone(),
                VulkanResidentPlacedPrefixDeviceBuffer {
                    buffer: device
                        .create_resident_buffer(*byte_count)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    _device: Rc::clone(device),
                },
            );
        }
        for (device_id, cache_device_buffer) in &device_buffers {
            let device = source.devices.get(device_id).expect("cache device was validated");
            let copies = ranges
                .iter()
                .filter(|range| range.device_id == *device_id)
                .map(|range| {
                    let (_, resident_state) = source
                        .processor
                        .mounted_state_buffer_with_device_id(&range.key)
                        .expect("captured resident state remains mounted");
                    let source_offset = match range.kind {
                        VulkanResidentPlacedPrefixStateRangeKind::Static => {
                            resident_state.layout.static_data_offset
                        }
                        VulkanResidentPlacedPrefixStateRangeKind::Dynamic { block_id, .. } => {
                            let page = source
                                .transient_state_pages
                                .resident_page_for_block(&range.key, block_id)
                                .expect("captured state page remains bound");
                            resident_state.layout.dynamic_physical_page_offset(page)?
                        }
                    };
                    VulkanResidentBufferRangeCopy::new(
                        &resident_state.buffer,
                        &cache_device_buffer.buffer,
                        source_offset,
                        range.cache_byte_offset,
                        range.byte_len,
                    )
                })
                .collect::<Result<Vec<_>, VulkanError>>()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if !copies.is_empty() {
                device
                    .create_resident_buffer_copy_batch(&copies)
                    .and_then(|copy| copy.run())
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
        }

        let byte_count = device_byte_counts.values().copied().sum();
        Ok(VulkanResidentPlacedPrefixStateEntry {
            key,
            device_buffers,
            ranges,
            byte_count,
        })
    }

    fn install(
        &mut self,
        entry: VulkanResidentPlacedPrefixStateEntry,
        cache_insert: &RuntimePrefixStateCacheInsert,
    ) {
        if cache_insert.replaced_existing {
            self.entries.remove(&entry.key);
        }
        self.evict_keys(&cache_insert.evicted_keys);
        self.entries.insert(entry.key.clone(), entry);
        self.stats.insertion_count = self.stats.insertion_count.saturating_add(1);
    }

    fn restore(
        &mut self,
        key: &RuntimePrefixStateCacheKey,
        target: &mut VulkanResidentInProcessPlacedPromptStream,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let entry = self.entries.get(key).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "logical prefix cache restored {:?}, but its resident pages were absent",
                    key.runtime_graph_id
                )),
            )
        })?;
        if entry.key != *key {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident prefix cache key identity changed",
                ),
            ));
        }
        if !target.is_idle()
            || target.pending_scheduler_activation.is_some()
            || target.next_stream_tick() != 0
        {
            return Err(placed_scheduler_divergence(
                "prefix state can only be restored into a fresh idle stream",
            ));
        }

        for range in &entry.ranges {
            if let VulkanResidentPlacedPrefixStateRangeKind::Dynamic {
                block_id,
                logical_page_index,
                block_activation_capacity,
            } = range.kind
            {
                let state = target.processor.mounted_state_buffer(&range.key).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "cannot restore non-resident prefix state {}.{}",
                            range.key.node_instance_id, range.key.state_id
                        )),
                    )
                })?;
                let logical_activation_index = logical_page_index
                    .checked_mul(block_activation_capacity)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                target.transient_state_pages.bind_slot(
                    state,
                    &TransientStateSlot {
                        key: range.key.clone(),
                        logical_activation_index,
                        block_id,
                        block_activation_offset: 0,
                        block_activation_capacity,
                        allocated_block: false,
                        copy_from_block_id: None,
                    },
                )?;
            }
        }

        for (device_id, cache_device_buffer) in &entry.device_buffers {
            let device = target.devices.get(device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: device_id.clone(),
                }
            })?;
            let copies = entry
                .ranges
                .iter()
                .filter(|range| range.device_id == *device_id)
                .map(|range| {
                    let (_, resident_state) = target
                        .processor
                        .mounted_state_buffer_with_device_id(&range.key)
                        .expect("restored resident state remains mounted");
                    let destination_offset = match range.kind {
                        VulkanResidentPlacedPrefixStateRangeKind::Static => {
                            resident_state.layout.static_data_offset
                        }
                        VulkanResidentPlacedPrefixStateRangeKind::Dynamic { block_id, .. } => {
                            let page = target
                                .transient_state_pages
                                .resident_page_for_block(&range.key, block_id)
                                .expect("restored state page was just bound");
                            resident_state.layout.dynamic_physical_page_offset(page)?
                        }
                    };
                    VulkanResidentBufferRangeCopy::new(
                        &cache_device_buffer.buffer,
                        &resident_state.buffer,
                        range.cache_byte_offset,
                        destination_offset,
                        range.byte_len,
                    )
                })
                .collect::<Result<Vec<_>, VulkanError>>()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if !copies.is_empty() {
                device
                    .create_resident_buffer_copy_batch(&copies)
                    .and_then(|copy| copy.run())
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
        }
        let output_device = target
            .devices
            .get(&target.package.output_device_id)
            .ok_or_else(|| VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: target.package.output_device_id.clone(),
            })?;
        for token_batch in key.token_ids.chunks(VULKAN_BACKEND_LOOP_MAX_WINDOW) {
            target
                .processor
                .sampler
                .record_input_tokens(output_device, token_batch)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        }
        target.session.next_stream_tick = u64::try_from(key.token_count)
            .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        self.stats.hit_count = self.stats.hit_count.saturating_add(1);
        self.stats.reused_token_count = self.stats.reused_token_count.saturating_add(key.token_count);
        self.stats.saved_prefill_token_count = self
            .stats
            .saved_prefill_token_count
            .saturating_add(key.token_count);
        Ok(())
    }
}

fn append_cache_range(
    device_byte_counts: &mut BTreeMap<String, usize>,
    device_id: &str,
    byte_len: usize,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    const VULKAN_BUFFER_COPY_ALIGNMENT: usize = 4;
    if byte_len == 0 || !byte_len.is_multiple_of(VULKAN_BUFFER_COPY_ALIGNMENT) {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "resident prefix cache range length {byte_len} is not a non-zero multiple of the Vulkan buffer-copy alignment {VULKAN_BUFFER_COPY_ALIGNMENT}"
            )),
        ));
    }
    let byte_count = device_byte_counts.entry(device_id.to_string()).or_default();
    let offset = *byte_count;
    *byte_count = byte_count.checked_add(byte_len).ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
            "resident prefix cache byte capacity overflowed".to_string(),
        ))
    })?;
    Ok(offset)
}
