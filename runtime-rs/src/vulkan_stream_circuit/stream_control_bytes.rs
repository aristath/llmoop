#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamControl {
    pub stream_tick: u64,
    pub control_flags: u32,
    pub dynamic_state_capacity_activations: u32,
}

fn stream_control_bytes(
    token_id: u32,
    control: VulkanMountedPlacedStreamControl,
) -> [u8; VULKAN_STREAM_CONTROL_BYTE_CAPACITY] {
    let mut bytes = [0; VULKAN_STREAM_CONTROL_BYTE_CAPACITY];
    bytes[0..4].copy_from_slice(&token_id.to_le_bytes());
    bytes[4..12].copy_from_slice(&control.stream_tick.to_le_bytes());
    bytes[12..16].copy_from_slice(&control.control_flags.to_le_bytes());
    bytes[16..20].copy_from_slice(&control.dynamic_state_capacity_activations.to_le_bytes());
    bytes
}

fn stream_control_metadata_bytes(
    control: VulkanMountedPlacedStreamControl,
) -> [u8; VULKAN_STREAM_CONTROL_BYTE_CAPACITY - VULKAN_STREAM_CONTROL_METADATA_OFFSET] {
    let mut bytes =
        [0; VULKAN_STREAM_CONTROL_BYTE_CAPACITY - VULKAN_STREAM_CONTROL_METADATA_OFFSET];
    bytes[0..8].copy_from_slice(&control.stream_tick.to_le_bytes());
    bytes[8..12].copy_from_slice(&control.control_flags.to_le_bytes());
    bytes[12..16].copy_from_slice(&control.dynamic_state_capacity_activations.to_le_bytes());
    bytes
}

fn component_batch_lane_stream_control_bytes(
    input_token_ids: &[u32],
    start_stream_tick: u64,
    dynamic_state_capacity_activations: u32,
) -> Result<Vec<[u8; VULKAN_STREAM_CONTROL_BYTE_CAPACITY]>, VulkanResidentInProcessPlacedRuntimeError>
{
    input_token_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(lane_index, token_id)| {
            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(lane_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            Ok(stream_control_bytes(
                token_id,
                VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                },
            ))
        })
        .collect()
}

fn component_batch_control_bytes(
    batch_width: u32,
    start_stream_tick: u64,
    dynamic_state_capacity_activations: u32,
) -> [u8; VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY as usize] {
    let mut bytes = [0; VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY as usize];
    bytes[0..4].copy_from_slice(&batch_width.to_le_bytes());
    bytes[4..12].copy_from_slice(&start_stream_tick.to_le_bytes());
    bytes[12..16].copy_from_slice(&dynamic_state_capacity_activations.to_le_bytes());
    bytes
}

fn component_batch_push_constant_bytes(
    byte_count: u32,
    control: &[u8; VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY as usize],
) -> Result<Vec<u8>, VulkanResidentInProcessPlacedRuntimeError> {
    match byte_count {
        byte_count if byte_count == 2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY => {
            let mut bytes =
                Vec::with_capacity(2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize);
            bytes.extend_from_slice(
                &control[..VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize],
            );
            bytes.extend_from_slice(&0u32.to_le_bytes());
            Ok(bytes)
        }
        VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY => {
            Ok(control[..VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize].to_vec())
        }
        VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY => Ok(control.to_vec()),
        0 => Ok(Vec::new()),
        _ => Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "unsupported component batch control byte count {byte_count}"
            )),
        )),
    }
}
