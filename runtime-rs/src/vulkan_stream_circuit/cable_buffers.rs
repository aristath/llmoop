#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPlacedCableDirection {
    Incoming,
    Outgoing,
}

pub struct VulkanPlacedCableIoBuffers {
    pub plan: VulkanPlacedCableIoPlan,
    pub local_buffers: Vec<VulkanPlacedLocalCableBufferAllocation>,
    pub incoming_buffers: Vec<VulkanPlacedCableBufferAllocation>,
    pub outgoing_buffers: Vec<VulkanPlacedCableBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPlacedCableIoBuffers {
    pub fn local_buffer(
        &self,
        cable_index: usize,
    ) -> Option<(usize, &VulkanPlacedLocalCableBufferAllocation)> {
        self.local_buffers
            .iter()
            .enumerate()
            .find(|(_, buffer)| buffer.cable.cable_index == cable_index)
    }

    pub fn local_cable_buffer(
        &self,
        cable_index: usize,
    ) -> Option<&VulkanPlacedLocalCableBufferAllocation> {
        self.local_buffers
            .iter()
            .find(|buffer| buffer.cable.cable_index == cable_index)
    }

    pub fn buffer(
        &self,
        direction: VulkanPlacedCableDirection,
        cable_index: usize,
    ) -> Option<(usize, &VulkanPlacedCableBufferAllocation)> {
        match direction {
            VulkanPlacedCableDirection::Incoming => self
                .incoming_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.cable_index == cable_index),
            VulkanPlacedCableDirection::Outgoing => self
                .outgoing_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.cable_index == cable_index),
        }
    }

    pub fn incoming_buffer(
        &self,
        cable_index: usize,
    ) -> Option<&VulkanPlacedCableBufferAllocation> {
        self.incoming_buffers
            .iter()
            .find(|buffer| buffer.endpoint.cable_index == cable_index)
    }

    pub fn outgoing_buffer(
        &self,
        cable_index: usize,
    ) -> Option<&VulkanPlacedCableBufferAllocation> {
        self.outgoing_buffers
            .iter()
            .find(|buffer| buffer.endpoint.cable_index == cable_index)
    }
}

pub struct VulkanPlacedLocalCableBufferAllocation {
    pub cable: VulkanPlacedLocalCable,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

pub struct VulkanPlacedCableBufferAllocation {
    pub endpoint: VulkanPlacedCableEndpoint,
    pub byte_capacity: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanPlacedCableEndpointBufferOverride {
    pub direction: VulkanPlacedCableDirection,
    pub cable_index: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanPlacedCablePacketKey {
    pub cable_index: usize,
    pub from_device_id: String,
    pub to_device_id: String,
}

impl VulkanPlacedCablePacketKey {
    pub fn from_outgoing_endpoint(endpoint: &VulkanPlacedCableEndpoint) -> Self {
        Self {
            cable_index: endpoint.cable_index,
            from_device_id: endpoint.local_device_id.clone(),
            to_device_id: endpoint.remote_device_id.clone(),
        }
    }

    pub fn from_incoming_endpoint(endpoint: &VulkanPlacedCableEndpoint) -> Self {
        Self {
            cable_index: endpoint.cable_index,
            from_device_id: endpoint.remote_device_id.clone(),
            to_device_id: endpoint.local_device_id.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCablePacket {
    pub key: VulkanPlacedCablePacketKey,
    pub signal: String,
    pub source_pedal_id: String,
    pub destination_pedal_id: String,
    pub byte_count: usize,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableTransportReceipt {
    pub key: VulkanPlacedCablePacketKey,
    pub signal: String,
    pub byte_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanPlacedCableTransportReceiveBatch {
    pub received: Vec<VulkanPlacedCableTransportReceipt>,
    pub missing_packets: Vec<VulkanPlacedCablePacketKey>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacedCableTransportStats {
    pub pending_packet_count: usize,
    pub pending_byte_count: usize,
    pub pending_direct_cable_count: usize,
    pub pending_direct_byte_count: usize,
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
}

impl VulkanPlacedCableTransportStats {
    fn accumulate(&mut self, tick: &Self) {
        self.pending_packet_count = tick.pending_packet_count;
        self.pending_byte_count = tick.pending_byte_count;
        self.pending_direct_cable_count = tick.pending_direct_cable_count;
        self.pending_direct_byte_count = tick.pending_direct_byte_count;
        self.published_packet_count = self
            .published_packet_count
            .saturating_add(tick.published_packet_count);
        self.published_byte_count = self
            .published_byte_count
            .saturating_add(tick.published_byte_count);
        self.received_packet_count = self
            .received_packet_count
            .saturating_add(tick.received_packet_count);
        self.received_byte_count = self
            .received_byte_count
            .saturating_add(tick.received_byte_count);
        self.direct_copy_count = self
            .direct_copy_count
            .saturating_add(tick.direct_copy_count);
        self.direct_copy_byte_count = self
            .direct_copy_byte_count
            .saturating_add(tick.direct_copy_byte_count);
        self.direct_receive_count = self
            .direct_receive_count
            .saturating_add(tick.direct_receive_count);
        self.direct_receive_byte_count = self
            .direct_receive_byte_count
            .saturating_add(tick.direct_receive_byte_count);
    }
}

