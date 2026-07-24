const VULKAN_TRANSIENT_STATE_PAGE_TABLE_MAGIC: u32 = 0x4e_52_56_53;
const VULKAN_TRANSIENT_STATE_PAGE_TABLE_VERSION: u32 = 1;
const VULKAN_TRANSIENT_STATE_PAGE_TABLE_HEADER_WORDS: usize = 9;
const VULKAN_TRANSIENT_STATE_DATA_ALIGNMENT: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanTransientStateBufferLayout {
    pub static_byte_capacity: usize,
    pub bytes_per_activation: Option<usize>,
    pub dynamic_activation_capacity: usize,
    pub block_activation_capacity: usize,
    pub dynamic_page_byte_capacity: usize,
    pub dynamic_page_count: usize,
    pub page_table_byte_capacity: usize,
    pub static_data_offset: usize,
    pub dynamic_data_offset: usize,
    pub byte_capacity: usize,
}

impl VulkanTransientStateBufferLayout {
    fn for_state(
        state: &VulkanResidentStateBuffer,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanError> {
        let static_byte_capacity = state.static_bytes.unwrap_or(0);
        let (dynamic_activation_capacity, block_activation_capacity, dynamic_page_byte_capacity) =
            match state.bytes_per_activation {
                Some(bytes_per_activation) => {
                    if dynamic_state_capacity_activations == 0 {
                        return Err(VulkanError(format!(
                            "{}.{} requires non-zero dynamic state capacity",
                            state.component_id, state.state_id
                        )));
                    }
                    let dynamic_activation_capacity = state
                        .max_dynamic_activations
                        .map(|limit| limit.min(dynamic_state_capacity_activations))
                        .unwrap_or(dynamic_state_capacity_activations);
                    let block_activation_capacity =
                        dynamic_activation_capacity.min(VULKAN_BACKEND_LOOP_MAX_WINDOW);
                    let dynamic_page_byte_capacity = bytes_per_activation
                        .checked_mul(block_activation_capacity)
                        .ok_or_else(|| {
                            VulkanError(format!(
                                "{}.{} transient page byte capacity overflowed",
                                state.component_id, state.state_id
                            ))
                        })?;
                    (
                        dynamic_activation_capacity,
                        block_activation_capacity,
                        dynamic_page_byte_capacity,
                    )
                }
                None => (0, 0, 0),
            };
        let dynamic_page_count = if dynamic_activation_capacity == 0 {
            0
        } else {
            dynamic_activation_capacity.div_ceil(block_activation_capacity)
        };
        let page_table_byte_capacity = align_up(
            VULKAN_TRANSIENT_STATE_PAGE_TABLE_HEADER_WORDS
                .checked_add(dynamic_page_count)
                .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{}.{} transient page table capacity overflowed",
                        state.component_id, state.state_id
                    ))
                })?,
            VULKAN_TRANSIENT_STATE_DATA_ALIGNMENT,
        )?;
        let static_data_offset = page_table_byte_capacity;
        let dynamic_data_offset = align_up(
            static_data_offset
                .checked_add(static_byte_capacity)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{}.{} static state capacity overflowed",
                        state.component_id, state.state_id
                    ))
                })?,
            VULKAN_TRANSIENT_STATE_DATA_ALIGNMENT,
        )?;
        let byte_capacity = dynamic_page_byte_capacity
            .checked_mul(dynamic_page_count)
            .and_then(|dynamic_bytes| dynamic_data_offset.checked_add(dynamic_bytes))
            .ok_or_else(|| {
                VulkanError(format!(
                    "{}.{} physical transient state capacity overflowed",
                    state.component_id, state.state_id
                ))
            })?;
        if static_byte_capacity == 0 && dynamic_page_count == 0 {
            return Err(VulkanError(format!(
                "{}.{} has unknown or zero byte capacity",
                state.component_id, state.state_id
            )));
        }
        Ok(Self {
            static_byte_capacity,
            bytes_per_activation: state.bytes_per_activation,
            dynamic_activation_capacity,
            block_activation_capacity,
            dynamic_page_byte_capacity,
            dynamic_page_count,
            page_table_byte_capacity,
            static_data_offset,
            dynamic_data_offset,
            byte_capacity,
        })
    }

    fn initial_page_table_bytes(&self) -> Result<Vec<u8>, VulkanError> {
        let values = [
            VULKAN_TRANSIENT_STATE_PAGE_TABLE_MAGIC,
            VULKAN_TRANSIENT_STATE_PAGE_TABLE_VERSION,
            u32::try_from(self.static_byte_capacity)
                .map_err(|_| VulkanError("static state byte capacity exceeds u32".to_string()))?,
            u32::try_from(self.static_data_offset)
                .map_err(|_| VulkanError("static state data offset exceeds u32".to_string()))?,
            u32::try_from(self.dynamic_page_byte_capacity).map_err(|_| {
                VulkanError("dynamic state page byte capacity exceeds u32".to_string())
            })?,
            u32::try_from(self.dynamic_page_count)
                .map_err(|_| VulkanError("dynamic state page count exceeds u32".to_string()))?,
            u32::try_from(self.dynamic_activation_capacity).map_err(|_| {
                VulkanError("dynamic state activation capacity exceeds u32".to_string())
            })?,
            u32::try_from(self.block_activation_capacity).map_err(|_| {
                VulkanError("state block activation capacity exceeds u32".to_string())
            })?,
            u32::try_from(self.dynamic_data_offset)
                .map_err(|_| VulkanError("dynamic state data offset exceeds u32".to_string()))?,
        ];
        let mut bytes = vec![0u8; self.page_table_byte_capacity];
        for (index, value) in values.into_iter().enumerate() {
            let offset = index * std::mem::size_of::<u32>();
            bytes[offset..offset + std::mem::size_of::<u32>()]
                .copy_from_slice(&value.to_le_bytes());
        }
        for page_index in 0..self.dynamic_page_count {
            let offset =
                (VULKAN_TRANSIENT_STATE_PAGE_TABLE_HEADER_WORDS + page_index)
                    * std::mem::size_of::<u32>();
            bytes[offset..offset + std::mem::size_of::<u32>()].copy_from_slice(
                &u32::try_from(page_index)
                    .map_err(|_| VulkanError("dynamic state page index exceeds u32".to_string()))?
                    .to_le_bytes(),
            );
        }
        Ok(bytes)
    }

    fn page_table_bytes_with_mapping(
        &self,
        logical_to_physical: &[usize],
    ) -> Result<Vec<u8>, VulkanError> {
        if logical_to_physical.len() != self.dynamic_page_count {
            return Err(VulkanError(format!(
                "transient page mapping has {} entries but layout requires {}",
                logical_to_physical.len(),
                self.dynamic_page_count
            )));
        }
        let mut bytes = self.initial_page_table_bytes()?;
        for (logical_page, physical_page) in logical_to_physical.iter().copied().enumerate() {
            if physical_page >= self.dynamic_page_count {
                return Err(VulkanError(format!(
                    "transient physical page {physical_page} exceeds capacity {}",
                    self.dynamic_page_count
                )));
            }
            let offset =
                (VULKAN_TRANSIENT_STATE_PAGE_TABLE_HEADER_WORDS + logical_page)
                    * std::mem::size_of::<u32>();
            bytes[offset..offset + std::mem::size_of::<u32>()].copy_from_slice(
                &u32::try_from(physical_page)
                    .map_err(|_| VulkanError("transient physical page exceeds u32".to_string()))?
                    .to_le_bytes(),
            );
        }
        Ok(bytes)
    }

    fn dynamic_physical_page_offset(&self, page_index: usize) -> Result<usize, VulkanError> {
        if page_index >= self.dynamic_page_count {
            return Err(VulkanError(format!(
                "transient physical page {page_index} exceeds capacity {}",
                self.dynamic_page_count
            )));
        }
        self.dynamic_page_byte_capacity
            .checked_mul(page_index)
            .and_then(|offset| self.dynamic_data_offset.checked_add(offset))
            .ok_or_else(|| VulkanError("transient physical page offset overflowed".to_string()))
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, VulkanError> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| VulkanError("transient state alignment overflowed".to_string()))
}

#[cfg(test)]
mod transient_state_buffer_layout_tests {
    use super::*;

    fn state(
        static_bytes: Option<usize>,
        bytes_per_activation: Option<usize>,
    ) -> VulkanResidentStateBuffer {
        VulkanResidentStateBuffer {
            component_id: "component".to_string(),
            state_id: "state".to_string(),
            state_type: "test".to_string(),
            layout: None,
            static_elements: None,
            elements_per_activation: None,
            max_dynamic_activations: None,
            static_bytes,
            bytes_per_activation,
            clone_from: None,
        }
    }

    #[test]
    fn dynamic_state_layout_separates_device_page_table_from_physical_pages() {
        let layout =
            VulkanTransientStateBufferLayout::for_state(&state(None, Some(16)), 130).unwrap();

        assert_eq!(layout.block_activation_capacity, 64);
        assert_eq!(layout.dynamic_page_count, 3);
        assert_eq!(layout.dynamic_page_byte_capacity, 1024);
        assert_eq!(layout.dynamic_data_offset, layout.page_table_byte_capacity);
        assert_eq!(
            layout.byte_capacity,
            layout.dynamic_data_offset + 3 * 1024
        );
        let table = layout.initial_page_table_bytes().unwrap();
        assert_eq!(
            u32::from_le_bytes(table[0..4].try_into().unwrap()),
            VULKAN_TRANSIENT_STATE_PAGE_TABLE_MAGIC
        );
        assert_eq!(
            u32::from_le_bytes(table[36..40].try_into().unwrap()),
            0
        );
        assert_eq!(
            u32::from_le_bytes(table[44..48].try_into().unwrap()),
            2
        );
    }

    #[test]
    fn fixed_state_layout_uses_the_same_abi_without_dynamic_pages() {
        let layout =
            VulkanTransientStateBufferLayout::for_state(&state(Some(3072), None), 65536).unwrap();

        assert_eq!(layout.dynamic_page_count, 0);
        assert_eq!(layout.static_data_offset, layout.page_table_byte_capacity);
        assert_eq!(layout.dynamic_data_offset, layout.byte_capacity);
        assert_eq!(layout.byte_capacity, layout.static_data_offset + 3072);
    }
}
