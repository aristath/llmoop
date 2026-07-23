struct VulkanResidentStateTransactionEntry {
    state_buffer_index: usize,
    byte_capacity: usize,
    snapshots: VulkanResidentBuffer,
}

struct VulkanResidentStateTransactionBank {
    cycle_width: usize,
    has_baseline: bool,
    capture_batches: Vec<Option<VulkanResidentBufferCopyBatch>>,
    restore_batches: Vec<Option<VulkanResidentBufferCopyBatch>>,
    entries: Vec<VulkanResidentStateTransactionEntry>,
}

impl VulkanResidentStateTransactionBank {
    fn new(
        device: &VulkanComputeDevice,
        buffers: &VulkanStreamCircuitStreamBuffers,
        cycle_width: usize,
    ) -> Result<Self, VulkanError> {
        Self::new_with_baseline(device, buffers, cycle_width, false)
    }

    fn new_transactional(
        device: &VulkanComputeDevice,
        buffers: &VulkanStreamCircuitStreamBuffers,
        cycle_width: usize,
    ) -> Result<Self, VulkanError> {
        Self::new_with_baseline(device, buffers, cycle_width, true)
    }

    fn new_with_baseline(
        device: &VulkanComputeDevice,
        buffers: &VulkanStreamCircuitStreamBuffers,
        cycle_width: usize,
        has_baseline: bool,
    ) -> Result<Self, VulkanError> {
        if cycle_width == 0 {
            return Err(VulkanError(
                "resident feedback snapshot cycle width must not be zero".to_string(),
            ));
        }
        let mut entries = Vec::new();
        for (state_buffer_index, state) in buffers.state_buffers.iter().enumerate() {
            if state.static_byte_capacity.is_none() {
                continue;
            }
            let snapshot_count = cycle_width
                .checked_add(usize::from(has_baseline))
                .ok_or_else(|| {
                    VulkanError("resident state transaction snapshot count overflowed".to_string())
                })?;
            let snapshot_byte_capacity = state
                .byte_capacity
                .checked_mul(snapshot_count)
                .ok_or_else(|| {
                    VulkanError("resident feedback snapshot capacity overflowed".to_string())
                })?;
            let snapshots = device.create_resident_buffer(snapshot_byte_capacity)?;
            entries.push(VulkanResidentStateTransactionEntry {
                state_buffer_index,
                byte_capacity: state.byte_capacity,
                snapshots,
            });
        }
        let snapshot_count = cycle_width
            .checked_add(usize::from(has_baseline))
            .ok_or_else(|| {
                VulkanError("resident state transaction snapshot count overflowed".to_string())
            })?;
        let mut capture_batches = Vec::with_capacity(snapshot_count);
        let mut restore_batches = Vec::with_capacity(snapshot_count);
        for snapshot_index in 0..snapshot_count {
            let mut captures = Vec::with_capacity(entries.len());
            let mut restores = Vec::with_capacity(entries.len());
            for entry in &entries {
                let state = &buffers.state_buffers[entry.state_buffer_index];
                let snapshot_offset =
                    snapshot_index
                        .checked_mul(entry.byte_capacity)
                        .ok_or_else(|| {
                            VulkanError(
                                "resident state transaction snapshot offset overflowed".to_string(),
                            )
                        })?;
                captures.push(VulkanResidentBufferRangeCopy::new(
                    &state.buffer,
                    &entry.snapshots,
                    0,
                    snapshot_offset,
                    entry.byte_capacity,
                )?);
                restores.push(VulkanResidentBufferRangeCopy::new(
                    &entry.snapshots,
                    &state.buffer,
                    snapshot_offset,
                    0,
                    entry.byte_capacity,
                )?);
            }
            capture_batches.push(
                (!captures.is_empty())
                    .then(|| device.create_resident_buffer_copy_batch(&captures))
                    .transpose()?,
            );
            restore_batches.push(
                (!restores.is_empty())
                    .then(|| device.create_resident_buffer_copy_batch(&restores))
                    .transpose()?,
            );
        }
        Ok(Self {
            cycle_width,
            has_baseline,
            capture_batches,
            restore_batches,
            entries,
        })
    }

    fn copies_for_cycle<'a>(
        &'a self,
        buffers: &'a VulkanStreamCircuitStreamBuffers,
        steps_per_tick: usize,
        tick_count: usize,
    ) -> Result<Vec<VulkanResidentKernelSequenceSnapshotCopy<'a>>, VulkanError> {
        if tick_count == 0 || tick_count > self.cycle_width {
            return Err(VulkanError(format!(
                "resident feedback snapshot cycle contains {tick_count} ticks, capacity is {}",
                self.cycle_width
            )));
        }
        let mut copies = Vec::with_capacity(self.entries.len() * tick_count);
        for tick_index in 0..tick_count {
            let after_step_index = (tick_index + 1)
                .checked_mul(steps_per_tick)
                .and_then(|step| step.checked_sub(1))
                .ok_or_else(|| {
                    VulkanError("resident feedback snapshot step index overflowed".to_string())
                })?;
            for entry in &self.entries {
                let state = &buffers.state_buffers[entry.state_buffer_index];
                copies.push(VulkanResidentKernelSequenceSnapshotCopy::new(
                    after_step_index,
                    &state.buffer,
                    &entry.snapshots,
                    0,
                    (tick_index + usize::from(self.has_baseline)) * entry.byte_capacity,
                    entry.byte_capacity,
                )?);
            }
        }
        Ok(copies)
    }

    fn copies_for_tick<'a>(
        &'a self,
        buffers: &'a VulkanStreamCircuitStreamBuffers,
        after_step_index: usize,
        tick_index: usize,
    ) -> Result<Vec<VulkanResidentKernelSequenceSnapshotCopy<'a>>, VulkanError> {
        if tick_index >= self.cycle_width {
            return Err(VulkanError(format!(
                "resident state transaction tick {tick_index} exceeds capacity {}",
                self.cycle_width
            )));
        }
        let snapshot_index = tick_index + usize::from(self.has_baseline);
        self.entries
            .iter()
            .map(|entry| {
                let state = &buffers.state_buffers[entry.state_buffer_index];
                VulkanResidentKernelSequenceSnapshotCopy::new(
                    after_step_index,
                    &state.buffer,
                    &entry.snapshots,
                    0,
                    snapshot_index * entry.byte_capacity,
                    entry.byte_capacity,
                )
            })
            .collect()
    }

    fn copies_for_state_buffers<'a>(
        &'a self,
        buffers: &'a VulkanStreamCircuitStreamBuffers,
        after_step_index: usize,
        tick_index: usize,
        state_buffer_indices: &BTreeSet<usize>,
    ) -> Result<Vec<VulkanResidentKernelSequenceSnapshotCopy<'a>>, VulkanError> {
        if tick_index >= self.cycle_width {
            return Err(VulkanError(format!(
                "resident state transaction tick {tick_index} exceeds capacity {}",
                self.cycle_width
            )));
        }
        let snapshot_index = tick_index + usize::from(self.has_baseline);
        self.entries
            .iter()
            .filter(|entry| state_buffer_indices.contains(&entry.state_buffer_index))
            .map(|entry| {
                let state = &buffers.state_buffers[entry.state_buffer_index];
                VulkanResidentKernelSequenceSnapshotCopy::new(
                    after_step_index,
                    &state.buffer,
                    &entry.snapshots,
                    0,
                    snapshot_index * entry.byte_capacity,
                    entry.byte_capacity,
                )
            })
            .collect()
    }

    fn commit_prefix(
        &self,
        buffers: &VulkanStreamCircuitStreamBuffers,
        processed_tick_count: usize,
    ) -> Result<(), VulkanError> {
        if processed_tick_count == 0 {
            return Err(VulkanError(
                "resident state transaction cannot commit an empty processed prefix".to_string(),
            ));
        }
        if processed_tick_count > self.cycle_width {
            return Err(VulkanError(format!(
                "resident state transaction prefix {processed_tick_count} exceeds capacity {}",
                self.cycle_width
            )));
        }
        let _ = buffers;
        let snapshot_index = processed_tick_count - 1 + usize::from(self.has_baseline);
        if let Some(batch) = &self.restore_batches[snapshot_index] {
            batch.run()?;
        }
        Ok(())
    }

    fn capture_baseline(
        &self,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<(), VulkanError> {
        if !self.has_baseline {
            return Err(VulkanError(
                "resident state snapshot bank has no transactional baseline".to_string(),
            ));
        }
        let _ = buffers;
        if let Some(batch) = &self.capture_batches[0] {
            batch.run()?;
        }
        Ok(())
    }

    fn restore_baseline(
        &self,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<(), VulkanError> {
        if !self.has_baseline {
            return Err(VulkanError(
                "resident state snapshot bank has no transactional baseline".to_string(),
            ));
        }
        let _ = buffers;
        if let Some(batch) = &self.restore_batches[0] {
            batch.run()?;
        }
        Ok(())
    }
}
