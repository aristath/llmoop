#[derive(Default)]
pub struct VulkanResidentTransientStatePageTable {
    states: BTreeMap<TransientStateKey, VulkanResidentTransientStatePages>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentTransientStatePages {
    block_activation_capacity: usize,
    resident_page_count: usize,
    block_to_page: BTreeMap<TransientStateBlockId, usize>,
    next_page_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTransientStatePageBinding {
    pub key: TransientStateKey,
    pub logical_activation_index: usize,
    pub transient_block_id: TransientStateBlockId,
    pub transient_block_activation_offset: usize,
    pub transient_block_activation_capacity: usize,
    pub resident_page_index: usize,
    pub resident_activation_offset: usize,
    pub resident_byte_offset: usize,
    pub bytes_per_activation: usize,
}

impl VulkanResidentTransientStatePageTable {
    pub fn clear(&mut self) {
        self.states.clear();
    }

    pub fn bind_slot(
        &mut self,
        resident_state: &VulkanResidentStateBuffer,
        package_dynamic_state_capacity_activations: usize,
        slot: &TransientStateSlot,
    ) -> Result<VulkanResidentTransientStatePageBinding, VulkanResidentTokenModelPackageError> {
        if resident_state.component_id != slot.key.node_instance_id
            || resident_state.state_id != slot.key.state_id
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} cannot bind resident state {}.{}",
                slot.key.node_instance_id,
                slot.key.state_id,
                resident_state.component_id,
                resident_state.state_id
            )));
        }

        let bytes_per_activation = resident_state.bytes_per_activation.ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} has no dynamic bytes per activation",
                resident_state.component_id, resident_state.state_id
            ))
        })?;
        let state_capacity = resident_state
            .max_dynamic_activations
            .map(|limit| limit.min(package_dynamic_state_capacity_activations))
            .unwrap_or(package_dynamic_state_capacity_activations);
        if state_capacity == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} has zero resident activation capacity",
                resident_state.component_id, resident_state.state_id
            )));
        }
        if slot.block_activation_capacity == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} has zero transient block activation capacity",
                resident_state.component_id, resident_state.state_id
            )));
        }
        if slot.block_activation_offset >= slot.block_activation_capacity {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} block offset {} exceeds transient block capacity {}",
                resident_state.component_id,
                resident_state.state_id,
                slot.block_activation_offset,
                slot.block_activation_capacity
            )));
        }
        if state_capacity % slot.block_activation_capacity != 0 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} resident activation capacity {} is not divisible by transient block activation capacity {}",
                resident_state.component_id,
                resident_state.state_id,
                state_capacity,
                slot.block_activation_capacity
            )));
        }

        let resident_page_count = state_capacity / slot.block_activation_capacity;
        if resident_page_count == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} has no resident transient pages",
                resident_state.component_id, resident_state.state_id
            )));
        }

        let pages =
            self.states
                .entry(slot.key.clone())
                .or_insert_with(|| VulkanResidentTransientStatePages {
                    block_activation_capacity: slot.block_activation_capacity,
                    resident_page_count,
                    block_to_page: BTreeMap::new(),
                    next_page_index: 0,
                });

        if pages.block_activation_capacity != slot.block_activation_capacity {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} changed transient block capacity from {} to {}",
                resident_state.component_id,
                resident_state.state_id,
                pages.block_activation_capacity,
                slot.block_activation_capacity
            )));
        }
        if pages.resident_page_count != resident_page_count {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} changed resident page count from {} to {}",
                resident_state.component_id,
                resident_state.state_id,
                pages.resident_page_count,
                resident_page_count
            )));
        }

        let resident_page_index = if let Some(page_index) = pages.block_to_page.get(&slot.block_id)
        {
            *page_index
        } else {
            if pages.next_page_index >= pages.resident_page_count {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "scheduled transient state {}.{} resident page capacity exhausted: {} pages of {} activations are already mapped",
                    resident_state.component_id,
                    resident_state.state_id,
                    pages.resident_page_count,
                    pages.block_activation_capacity
                )));
            }
            let page_index = pages.next_page_index;
            pages.next_page_index = pages.next_page_index.saturating_add(1);
            pages.block_to_page.insert(slot.block_id, page_index);
            page_index
        };

        let resident_activation_offset = resident_page_index
            .checked_mul(slot.block_activation_capacity)
            .and_then(|page_offset| page_offset.checked_add(slot.block_activation_offset))
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "scheduled transient state {}.{} resident activation offset overflowed",
                    resident_state.component_id, resident_state.state_id
                ))
            })?;
        let resident_byte_offset = resident_state
            .static_bytes
            .unwrap_or(0)
            .checked_add(
                bytes_per_activation
                    .checked_mul(resident_activation_offset)
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "scheduled transient state {}.{} byte offset overflowed",
                            resident_state.component_id, resident_state.state_id
                        ))
                    })?,
            )
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "scheduled transient state {}.{} byte offset overflowed",
                    resident_state.component_id, resident_state.state_id
                ))
            })?;

        Ok(VulkanResidentTransientStatePageBinding {
            key: slot.key.clone(),
            logical_activation_index: slot.logical_activation_index,
            transient_block_id: slot.block_id,
            transient_block_activation_offset: slot.block_activation_offset,
            transient_block_activation_capacity: slot.block_activation_capacity,
            resident_page_index,
            resident_activation_offset,
            resident_byte_offset,
            bytes_per_activation,
        })
    }
}

#[cfg(test)]
mod transient_state_page_tests {
    use super::*;

    fn key() -> TransientStateKey {
        TransientStateKey::new("layer_00", "kv")
    }

    fn resident_state(max_dynamic_activations: Option<usize>) -> VulkanResidentStateBuffer {
        VulkanResidentStateBuffer {
            component_id: "layer_00".to_string(),
            state_id: "kv".to_string(),
            state_type: "kv".to_string(),
            layout: Some("paged".to_string()),
            static_elements: None,
            elements_per_activation: Some(4),
            max_dynamic_activations,
            static_bytes: Some(32),
            bytes_per_activation: Some(16),
            clone_from: None,
        }
    }

    fn slot(
        block: u64,
        logical_activation_index: usize,
        block_activation_offset: usize,
        block_activation_capacity: usize,
    ) -> TransientStateSlot {
        TransientStateSlot {
            key: key(),
            logical_activation_index,
            block_id: TransientStateBlockId(block),
            block_activation_offset,
            block_activation_capacity,
        }
    }

    #[test]
    fn transient_state_page_table_reuses_resident_page_for_same_block() {
        let mut table = VulkanResidentTransientStatePageTable::default();
        let state = resident_state(None);

        let first = table.bind_slot(&state, 128, &slot(7, 0, 0, 64)).unwrap();
        let last = table.bind_slot(&state, 128, &slot(7, 63, 63, 64)).unwrap();

        assert_eq!(first.resident_page_index, 0);
        assert_eq!(last.resident_page_index, 0);
        assert_eq!(first.resident_activation_offset, 0);
        assert_eq!(last.resident_activation_offset, 63);
        assert_eq!(first.resident_byte_offset, 32);
        assert_eq!(last.resident_byte_offset, 32 + 16 * 63);
    }

    #[test]
    fn transient_state_page_table_allocates_distinct_pages_for_distinct_blocks() {
        let mut table = VulkanResidentTransientStatePageTable::default();
        let state = resident_state(None);

        let first = table.bind_slot(&state, 128, &slot(7, 0, 0, 64)).unwrap();
        let second = table.bind_slot(&state, 128, &slot(8, 64, 0, 64)).unwrap();

        assert_eq!(first.resident_page_index, 0);
        assert_eq!(second.resident_page_index, 1);
        assert_eq!(second.resident_activation_offset, 64);
        assert_eq!(second.resident_byte_offset, 32 + 16 * 64);
    }

    #[test]
    fn transient_state_page_table_rejects_page_capacity_exhaustion_without_aliasing() {
        let mut table = VulkanResidentTransientStatePageTable::default();
        let state = resident_state(Some(64));

        table.bind_slot(&state, 128, &slot(7, 0, 0, 64)).unwrap();
        let err = table
            .bind_slot(&state, 128, &slot(8, 64, 0, 64))
            .unwrap_err();

        assert!(err.to_string().contains("resident page capacity exhausted"));
    }

    #[test]
    fn transient_state_page_table_rejects_non_divisible_resident_capacity() {
        let mut table = VulkanResidentTransientStatePageTable::default();
        let state = resident_state(Some(96));
        let err = table
            .bind_slot(&state, 128, &slot(7, 0, 0, 64))
            .unwrap_err();

        assert!(err.to_string().contains("is not divisible"));
    }
}
