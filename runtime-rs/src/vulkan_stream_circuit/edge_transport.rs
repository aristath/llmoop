#[derive(Default)]
pub struct VulkanInProcessPlacedEdgeTransport {
    packets: BTreeMap<VulkanPlacedEdgePacketKey, VulkanPlacedEdgePacket>,
    direct_copies: BTreeMap<VulkanPlacedEdgePacketKey, VulkanPlacedEdgeDirectCopy>,
    ready_direct_edges: BTreeSet<VulkanPlacedEdgePacketKey>,
    published_packet_count: usize,
    published_byte_count: usize,
    received_packet_count: usize,
    received_byte_count: usize,
    direct_copy_count: usize,
    direct_copy_byte_count: usize,
    direct_receive_count: usize,
    direct_receive_byte_count: usize,
}

pub struct VulkanPlacedEdgeDirectCopy {
    pub key: VulkanPlacedEdgePacketKey,
    pub signal: String,
    pub source_component_id: String,
    pub destination_component_id: String,
    pub byte_count: usize,
    copy: Option<VulkanResidentMappedBufferCopy>,
}

impl VulkanInProcessPlacedEdgeTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn packet_count(&self) -> usize {
        self.packets.len()
    }

    pub fn stats(&self) -> VulkanPlacedEdgeTransportStats {
        VulkanPlacedEdgeTransportStats {
            pending_packet_count: self.packets.len(),
            pending_byte_count: self.packets.values().map(|packet| packet.byte_count).sum(),
            pending_direct_edge_count: self.ready_direct_edges.len(),
            pending_direct_byte_count: self
                .ready_direct_edges
                .iter()
                .filter_map(|key| self.direct_copies.get(key))
                .map(|copy| copy.byte_count)
                .sum(),
            published_packet_count: self.published_packet_count,
            published_byte_count: self.published_byte_count,
            received_packet_count: self.received_packet_count,
            received_byte_count: self.received_byte_count,
            direct_copy_count: self.direct_copy_count,
            direct_copy_byte_count: self.direct_copy_byte_count,
            direct_receive_count: self.direct_receive_count,
            direct_receive_byte_count: self.direct_receive_byte_count,
        }
    }

    pub fn contains_packet(&self, key: &VulkanPlacedEdgePacketKey) -> bool {
        self.packets.contains_key(key)
    }

    pub fn contains_ready_direct_edge(&self, key: &VulkanPlacedEdgePacketKey) -> bool {
        self.ready_direct_edges.contains(key)
    }

    pub fn direct_edge_binding_count(&self) -> usize {
        self.direct_copies.len()
    }

    fn edge_uses_shared_allocation(&self, key: &VulkanPlacedEdgePacketKey) -> bool {
        self.direct_copies
            .get(key)
            .is_some_and(|direct_copy| direct_copy.copy.is_none())
    }

    pub fn reset_tick_state(&mut self) {
        self.packets.clear();
        self.ready_direct_edges.clear();
        self.published_packet_count = 0;
        self.published_byte_count = 0;
        self.received_packet_count = 0;
        self.received_byte_count = 0;
        self.direct_copy_count = 0;
        self.direct_copy_byte_count = 0;
        self.direct_receive_count = 0;
        self.direct_receive_byte_count = 0;
    }

    pub fn register_direct_edge_copy(
        &mut self,
        outgoing: &VulkanPlacedEdgeBufferAllocation,
        incoming: &VulkanPlacedEdgeBufferAllocation,
    ) -> Result<(), VulkanPlacedEdgeTransportError> {
        let outgoing_key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
        let incoming_key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(&incoming.endpoint);
        if outgoing_key != incoming_key {
            return Err(VulkanPlacedEdgeTransportError::EndpointMismatch {
                outgoing_key,
                incoming_key,
            });
        }
        if self.direct_copies.contains_key(&outgoing_key) {
            return Ok(());
        }
        if outgoing.byte_capacity != incoming.byte_capacity {
            return Err(VulkanPlacedEdgeTransportError::ByteCapacityMismatch {
                key: outgoing_key,
                packet_byte_count: outgoing.byte_capacity,
                incoming_byte_capacity: incoming.byte_capacity,
            });
        }
        let copy = if Arc::ptr_eq(&outgoing.buffer, &incoming.buffer)
            || outgoing
                .buffer
                .shares_host_allocation_with(&incoming.buffer)
        {
            None
        } else {
            Some(
                outgoing
                    .buffer
                    .create_persistently_mapped_copy_to(&incoming.buffer, outgoing.byte_capacity)
                    .map_err(|error| VulkanPlacedEdgeTransportError::Vulkan {
                        operation: "create persistently mapped edge buffer copy",
                        error,
                    })?,
            )
        };
        self.direct_copies.insert(
            outgoing_key.clone(),
            VulkanPlacedEdgeDirectCopy {
                key: outgoing_key,
                signal: outgoing.endpoint.signal.clone(),
                source_component_id: outgoing.endpoint.local_component_id.clone(),
                destination_component_id: outgoing.endpoint.remote_component_id.clone(),
                byte_count: outgoing.byte_capacity,
                copy,
            },
        );
        Ok(())
    }

    pub fn publish_outgoing_edge(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        edge_index: usize,
    ) -> Result<VulkanPlacedEdgeTransportReceipt, VulkanPlacedEdgeTransportError> {
        let outgoing = mounted
            .edge_io
            .outgoing_buffer(edge_index)
            .ok_or_else(|| VulkanPlacedEdgeTransportError::MissingOutgoingEdge {
                device_id: mounted.device_id().to_string(),
                edge_index,
            })?;
        let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
        if let Some(direct_copy) = self.direct_copies.get(&key) {
            if let Some(copy) = &direct_copy.copy {
                copy.run(direct_copy.byte_count).map_err(|error| {
                    VulkanPlacedEdgeTransportError::Vulkan {
                        operation: "run direct edge buffer copy",
                        error,
                    }
                })?;
            }
            self.ready_direct_edges.insert(key);
            self.direct_copy_count += 1;
            self.direct_copy_byte_count += direct_copy.byte_count;
            return Ok(VulkanPlacedEdgeTransportReceipt {
                key: direct_copy.key.clone(),
                signal: direct_copy.signal.clone(),
                byte_count: direct_copy.byte_count,
            });
        }
        let bytes = outgoing
            .buffer
            .read_bytes(outgoing.byte_capacity)
            .map_err(|error| VulkanPlacedEdgeTransportError::Vulkan {
                operation: "read outgoing edge buffer",
                error,
            })?;
        let packet = VulkanPlacedEdgePacket {
            key: key.clone(),
            signal: outgoing.endpoint.signal.clone(),
            source_component_id: outgoing.endpoint.local_component_id.clone(),
            destination_component_id: outgoing.endpoint.remote_component_id.clone(),
            byte_count: bytes.len(),
            bytes,
        };
        let receipt = VulkanPlacedEdgeTransportReceipt {
            key: key.clone(),
            signal: packet.signal.clone(),
            byte_count: packet.byte_count,
        };
        let byte_count = packet.byte_count;
        self.packets.insert(key, packet);
        self.published_packet_count += 1;
        self.published_byte_count += byte_count;
        Ok(receipt)
    }

    pub fn publish_all_outgoing_edges(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
    ) -> Result<Vec<VulkanPlacedEdgeTransportReceipt>, VulkanPlacedEdgeTransportError> {
        let edge_indices = mounted
            .edge_io
            .outgoing_buffers
            .iter()
            .map(|outgoing| outgoing.endpoint.edge_index)
            .collect::<Vec<_>>();
        edge_indices
            .into_iter()
            .map(|edge_index| self.publish_outgoing_edge(mounted, edge_index))
            .collect()
    }

    pub fn receive_incoming_edge(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        edge_index: usize,
    ) -> Result<VulkanPlacedEdgeTransportReceipt, VulkanPlacedEdgeTransportError> {
        let incoming = mounted
            .edge_io
            .incoming_buffer(edge_index)
            .ok_or_else(|| VulkanPlacedEdgeTransportError::MissingIncomingEdge {
                device_id: mounted.device_id().to_string(),
                edge_index,
            })?;
        let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(&incoming.endpoint);
        if let Some(direct_copy) = self.direct_copies.get(&key) {
            if !self.ready_direct_edges.remove(&key) {
                return Err(VulkanPlacedEdgeTransportError::MissingPacket { key });
            }
            self.direct_receive_count += 1;
            self.direct_receive_byte_count += direct_copy.byte_count;
            return Ok(VulkanPlacedEdgeTransportReceipt {
                key: direct_copy.key.clone(),
                signal: direct_copy.signal.clone(),
                byte_count: direct_copy.byte_count,
            });
        }
        let packet = self
            .packets
            .remove(&key)
            .ok_or_else(|| VulkanPlacedEdgeTransportError::MissingPacket { key: key.clone() })?;
        if packet.byte_count != incoming.byte_capacity {
            return Err(VulkanPlacedEdgeTransportError::ByteCapacityMismatch {
                key,
                packet_byte_count: packet.byte_count,
                incoming_byte_capacity: incoming.byte_capacity,
            });
        }
        incoming
            .buffer
            .write_bytes(&packet.bytes)
            .map_err(|error| VulkanPlacedEdgeTransportError::Vulkan {
                operation: "write incoming edge buffer",
                error,
            })?;
        self.received_packet_count += 1;
        self.received_byte_count += packet.byte_count;
        Ok(VulkanPlacedEdgeTransportReceipt {
            key: packet.key,
            signal: packet.signal,
            byte_count: packet.byte_count,
        })
    }

    pub fn receive_available_incoming_edges(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
    ) -> Result<VulkanPlacedEdgeTransportReceiveBatch, VulkanPlacedEdgeTransportError> {
        let edge_indices = mounted
            .edge_io
            .incoming_buffers
            .iter()
            .map(|incoming| incoming.endpoint.edge_index)
            .collect::<Vec<_>>();
        let mut batch = VulkanPlacedEdgeTransportReceiveBatch::default();
        for edge_index in edge_indices {
            match self.receive_incoming_edge(mounted, edge_index) {
                Ok(receipt) => batch.received.push(receipt),
                Err(VulkanPlacedEdgeTransportError::MissingPacket { key }) => {
                    batch.missing_packets.push(key);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(batch)
    }
}

#[derive(Debug)]
pub enum VulkanPlacedEdgeTransportError {
    MissingOutgoingEdge {
        device_id: String,
        edge_index: usize,
    },
    MissingIncomingEdge {
        device_id: String,
        edge_index: usize,
    },
    MissingPacket {
        key: VulkanPlacedEdgePacketKey,
    },
    ByteCapacityMismatch {
        key: VulkanPlacedEdgePacketKey,
        packet_byte_count: usize,
        incoming_byte_capacity: usize,
    },
    EndpointMismatch {
        outgoing_key: VulkanPlacedEdgePacketKey,
        incoming_key: VulkanPlacedEdgePacketKey,
    },
    Vulkan {
        operation: &'static str,
        error: VulkanError,
    },
}

impl Display for VulkanPlacedEdgeTransportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingOutgoingEdge {
                device_id,
                edge_index,
            } => write!(
                f,
                "device {device_id:?} has no outgoing edge buffer for edge {edge_index}"
            ),
            Self::MissingIncomingEdge {
                device_id,
                edge_index,
            } => write!(
                f,
                "device {device_id:?} has no incoming edge buffer for edge {edge_index}"
            ),
            Self::MissingPacket { key } => write!(
                f,
                "no in-process edge packet for edge {} from {:?} to {:?}",
                key.edge_index, key.from_device_id, key.to_device_id
            ),
            Self::ByteCapacityMismatch {
                key,
                packet_byte_count,
                incoming_byte_capacity,
            } => write!(
                f,
                "in-process edge packet for edge {} from {:?} to {:?} has {} bytes, but incoming buffer has {} bytes",
                key.edge_index,
                key.from_device_id,
                key.to_device_id,
                packet_byte_count,
                incoming_byte_capacity
            ),
            Self::EndpointMismatch {
                outgoing_key,
                incoming_key,
            } => write!(
                f,
                "direct in-process edge endpoint mismatch: outgoing edge {} from {:?} to {:?}, incoming edge {} from {:?} to {:?}",
                outgoing_key.edge_index,
                outgoing_key.from_device_id,
                outgoing_key.to_device_id,
                incoming_key.edge_index,
                incoming_key.from_device_id,
                incoming_key.to_device_id
            ),
            Self::Vulkan { operation, error } => write!(f, "{operation} failed: {error}"),
        }
    }
}

impl Error for VulkanPlacedEdgeTransportError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgeIoPlanError(pub String);

impl Display for VulkanPlacedEdgeIoPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPlacedEdgeIoPlanError {}

