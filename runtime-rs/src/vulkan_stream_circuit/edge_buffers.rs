#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPlacedEdgeDirection {
    Incoming,
    Outgoing,
}

pub struct VulkanPlacedEdgeIoBuffers {
    pub plan: VulkanPlacedEdgeIoPlan,
    pub local_buffers: Vec<VulkanPlacedLocalEdgeBufferAllocation>,
    pub incoming_buffers: Vec<VulkanPlacedEdgeBufferAllocation>,
    pub outgoing_buffers: Vec<VulkanPlacedEdgeBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPlacedEdgeIoBuffers {
    pub fn local_buffer(
        &self,
        edge_index: usize,
    ) -> Option<(usize, &VulkanPlacedLocalEdgeBufferAllocation)> {
        self.local_buffers
            .iter()
            .enumerate()
            .find(|(_, buffer)| buffer.edge.edge_index == edge_index)
    }

    pub fn local_edge_buffer(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanPlacedLocalEdgeBufferAllocation> {
        self.local_buffers
            .iter()
            .find(|buffer| buffer.edge.edge_index == edge_index)
    }

    pub fn buffer(
        &self,
        direction: VulkanPlacedEdgeDirection,
        edge_index: usize,
    ) -> Option<(usize, &VulkanPlacedEdgeBufferAllocation)> {
        match direction {
            VulkanPlacedEdgeDirection::Incoming => self
                .incoming_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.edge_index == edge_index),
            VulkanPlacedEdgeDirection::Outgoing => self
                .outgoing_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.edge_index == edge_index),
        }
    }

    pub fn incoming_buffer(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanPlacedEdgeBufferAllocation> {
        self.incoming_buffers
            .iter()
            .find(|buffer| buffer.endpoint.edge_index == edge_index)
    }

    pub fn outgoing_buffer(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanPlacedEdgeBufferAllocation> {
        self.outgoing_buffers
            .iter()
            .find(|buffer| buffer.endpoint.edge_index == edge_index)
    }
}

pub struct VulkanPlacedLocalEdgeBufferAllocation {
    pub edge: VulkanPlacedLocalEdge,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

pub struct VulkanPlacedEdgeBufferAllocation {
    pub endpoint: VulkanPlacedEdgeEndpoint,
    pub byte_capacity: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanPlacedEdgeEndpointBufferOverride {
    pub direction: VulkanPlacedEdgeDirection,
    pub edge_index: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanPlacedEdgePacketKey {
    pub edge_index: usize,
    pub from_device_id: String,
    pub to_device_id: String,
}

impl VulkanPlacedEdgePacketKey {
    pub fn from_outgoing_endpoint(endpoint: &VulkanPlacedEdgeEndpoint) -> Self {
        Self {
            edge_index: endpoint.edge_index,
            from_device_id: endpoint.local_device_id.clone(),
            to_device_id: endpoint.remote_device_id.clone(),
        }
    }

    pub fn from_incoming_endpoint(endpoint: &VulkanPlacedEdgeEndpoint) -> Self {
        Self {
            edge_index: endpoint.edge_index,
            from_device_id: endpoint.remote_device_id.clone(),
            to_device_id: endpoint.local_device_id.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgePacket {
    pub key: VulkanPlacedEdgePacketKey,
    pub signal: String,
    pub source_component_id: String,
    pub destination_component_id: String,
    pub byte_count: usize,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgeTransportReceipt {
    pub key: VulkanPlacedEdgePacketKey,
    pub signal: String,
    pub byte_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanPlacedEdgeTransportReceiveBatch {
    pub received: Vec<VulkanPlacedEdgeTransportReceipt>,
    pub missing_packets: Vec<VulkanPlacedEdgePacketKey>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacedEdgeTransportStats {
    pub pending_packet_count: usize,
    pub pending_byte_count: usize,
    pub pending_direct_edge_count: usize,
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

impl VulkanPlacedEdgeTransportStats {
    fn accumulate(&mut self, tick: &Self) {
        self.pending_packet_count = tick.pending_packet_count;
        self.pending_byte_count = tick.pending_byte_count;
        self.pending_direct_edge_count = tick.pending_direct_edge_count;
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

