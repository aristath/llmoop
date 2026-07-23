#[derive(Default)]
pub struct VulkanInProcessPlacedCableTransport {
    packets: BTreeMap<VulkanPlacedCablePacketKey, VulkanPlacedCablePacket>,
    direct_copies: BTreeMap<VulkanPlacedCablePacketKey, VulkanPlacedCableDirectCopy>,
    ready_direct_cables: BTreeSet<VulkanPlacedCablePacketKey>,
    published_packet_count: usize,
    published_byte_count: usize,
    received_packet_count: usize,
    received_byte_count: usize,
    direct_copy_count: usize,
    direct_copy_byte_count: usize,
    direct_receive_count: usize,
    direct_receive_byte_count: usize,
}

pub struct VulkanPlacedCableDirectCopy {
    pub key: VulkanPlacedCablePacketKey,
    pub signal: String,
    pub source_pedal_id: String,
    pub destination_pedal_id: String,
    pub byte_count: usize,
    copy: Option<VulkanResidentMappedBufferCopy>,
}

impl VulkanInProcessPlacedCableTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn packet_count(&self) -> usize {
        self.packets.len()
    }

    pub fn stats(&self) -> VulkanPlacedCableTransportStats {
        VulkanPlacedCableTransportStats {
            pending_packet_count: self.packets.len(),
            pending_byte_count: self.packets.values().map(|packet| packet.byte_count).sum(),
            pending_direct_cable_count: self.ready_direct_cables.len(),
            pending_direct_byte_count: self
                .ready_direct_cables
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

    pub fn contains_packet(&self, key: &VulkanPlacedCablePacketKey) -> bool {
        self.packets.contains_key(key)
    }

    pub fn contains_ready_direct_cable(&self, key: &VulkanPlacedCablePacketKey) -> bool {
        self.ready_direct_cables.contains(key)
    }

    pub fn direct_cable_binding_count(&self) -> usize {
        self.direct_copies.len()
    }

    fn cable_uses_shared_allocation(&self, key: &VulkanPlacedCablePacketKey) -> bool {
        self.direct_copies
            .get(key)
            .is_some_and(|direct_copy| direct_copy.copy.is_none())
    }

    pub fn reset_tick_state(&mut self) {
        self.packets.clear();
        self.ready_direct_cables.clear();
        self.published_packet_count = 0;
        self.published_byte_count = 0;
        self.received_packet_count = 0;
        self.received_byte_count = 0;
        self.direct_copy_count = 0;
        self.direct_copy_byte_count = 0;
        self.direct_receive_count = 0;
        self.direct_receive_byte_count = 0;
    }

    pub fn register_direct_cable_copy(
        &mut self,
        outgoing: &VulkanPlacedCableBufferAllocation,
        incoming: &VulkanPlacedCableBufferAllocation,
    ) -> Result<(), VulkanPlacedCableTransportError> {
        let outgoing_key = VulkanPlacedCablePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
        let incoming_key = VulkanPlacedCablePacketKey::from_incoming_endpoint(&incoming.endpoint);
        if outgoing_key != incoming_key {
            return Err(VulkanPlacedCableTransportError::EndpointMismatch {
                outgoing_key,
                incoming_key,
            });
        }
        if self.direct_copies.contains_key(&outgoing_key) {
            return Ok(());
        }
        if outgoing.byte_capacity != incoming.byte_capacity {
            return Err(VulkanPlacedCableTransportError::ByteCapacityMismatch {
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
                    .map_err(|error| VulkanPlacedCableTransportError::Vulkan {
                        operation: "create persistently mapped cable buffer copy",
                        error,
                    })?,
            )
        };
        self.direct_copies.insert(
            outgoing_key.clone(),
            VulkanPlacedCableDirectCopy {
                key: outgoing_key,
                signal: outgoing.endpoint.signal.clone(),
                source_pedal_id: outgoing.endpoint.local_pedal_id.clone(),
                destination_pedal_id: outgoing.endpoint.remote_pedal_id.clone(),
                byte_count: outgoing.byte_capacity,
                copy,
            },
        );
        Ok(())
    }

    pub fn publish_outgoing_cable(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        cable_index: usize,
    ) -> Result<VulkanPlacedCableTransportReceipt, VulkanPlacedCableTransportError> {
        let outgoing = mounted
            .cable_io
            .outgoing_buffer(cable_index)
            .ok_or_else(|| VulkanPlacedCableTransportError::MissingOutgoingCable {
                device_id: mounted.device_id().to_string(),
                cable_index,
            })?;
        let key = VulkanPlacedCablePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
        if let Some(direct_copy) = self.direct_copies.get(&key) {
            if let Some(copy) = &direct_copy.copy {
                copy.run(direct_copy.byte_count).map_err(|error| {
                    VulkanPlacedCableTransportError::Vulkan {
                        operation: "run direct cable buffer copy",
                        error,
                    }
                })?;
            }
            self.ready_direct_cables.insert(key);
            self.direct_copy_count += 1;
            self.direct_copy_byte_count += direct_copy.byte_count;
            return Ok(VulkanPlacedCableTransportReceipt {
                key: direct_copy.key.clone(),
                signal: direct_copy.signal.clone(),
                byte_count: direct_copy.byte_count,
            });
        }
        let bytes = outgoing
            .buffer
            .read_bytes(outgoing.byte_capacity)
            .map_err(|error| VulkanPlacedCableTransportError::Vulkan {
                operation: "read outgoing cable buffer",
                error,
            })?;
        let packet = VulkanPlacedCablePacket {
            key: key.clone(),
            signal: outgoing.endpoint.signal.clone(),
            source_pedal_id: outgoing.endpoint.local_pedal_id.clone(),
            destination_pedal_id: outgoing.endpoint.remote_pedal_id.clone(),
            byte_count: bytes.len(),
            bytes,
        };
        let receipt = VulkanPlacedCableTransportReceipt {
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

    pub fn publish_all_outgoing_cables(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
    ) -> Result<Vec<VulkanPlacedCableTransportReceipt>, VulkanPlacedCableTransportError> {
        let cable_indices = mounted
            .cable_io
            .outgoing_buffers
            .iter()
            .map(|outgoing| outgoing.endpoint.cable_index)
            .collect::<Vec<_>>();
        cable_indices
            .into_iter()
            .map(|cable_index| self.publish_outgoing_cable(mounted, cable_index))
            .collect()
    }

    pub fn receive_incoming_cable(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        cable_index: usize,
    ) -> Result<VulkanPlacedCableTransportReceipt, VulkanPlacedCableTransportError> {
        let incoming = mounted
            .cable_io
            .incoming_buffer(cable_index)
            .ok_or_else(|| VulkanPlacedCableTransportError::MissingIncomingCable {
                device_id: mounted.device_id().to_string(),
                cable_index,
            })?;
        let key = VulkanPlacedCablePacketKey::from_incoming_endpoint(&incoming.endpoint);
        if let Some(direct_copy) = self.direct_copies.get(&key) {
            if !self.ready_direct_cables.remove(&key) {
                return Err(VulkanPlacedCableTransportError::MissingPacket { key });
            }
            self.direct_receive_count += 1;
            self.direct_receive_byte_count += direct_copy.byte_count;
            return Ok(VulkanPlacedCableTransportReceipt {
                key: direct_copy.key.clone(),
                signal: direct_copy.signal.clone(),
                byte_count: direct_copy.byte_count,
            });
        }
        let packet = self
            .packets
            .remove(&key)
            .ok_or_else(|| VulkanPlacedCableTransportError::MissingPacket { key: key.clone() })?;
        if packet.byte_count != incoming.byte_capacity {
            return Err(VulkanPlacedCableTransportError::ByteCapacityMismatch {
                key,
                packet_byte_count: packet.byte_count,
                incoming_byte_capacity: incoming.byte_capacity,
            });
        }
        incoming
            .buffer
            .write_bytes(&packet.bytes)
            .map_err(|error| VulkanPlacedCableTransportError::Vulkan {
                operation: "write incoming cable buffer",
                error,
            })?;
        self.received_packet_count += 1;
        self.received_byte_count += packet.byte_count;
        Ok(VulkanPlacedCableTransportReceipt {
            key: packet.key,
            signal: packet.signal,
            byte_count: packet.byte_count,
        })
    }

    pub fn receive_available_incoming_cables(
        &mut self,
        mounted: &VulkanMountedPlacedStreamCircuit,
    ) -> Result<VulkanPlacedCableTransportReceiveBatch, VulkanPlacedCableTransportError> {
        let cable_indices = mounted
            .cable_io
            .incoming_buffers
            .iter()
            .map(|incoming| incoming.endpoint.cable_index)
            .collect::<Vec<_>>();
        let mut batch = VulkanPlacedCableTransportReceiveBatch::default();
        for cable_index in cable_indices {
            match self.receive_incoming_cable(mounted, cable_index) {
                Ok(receipt) => batch.received.push(receipt),
                Err(VulkanPlacedCableTransportError::MissingPacket { key }) => {
                    batch.missing_packets.push(key);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(batch)
    }
}

#[derive(Debug)]
pub enum VulkanPlacedCableTransportError {
    MissingOutgoingCable {
        device_id: String,
        cable_index: usize,
    },
    MissingIncomingCable {
        device_id: String,
        cable_index: usize,
    },
    MissingPacket {
        key: VulkanPlacedCablePacketKey,
    },
    ByteCapacityMismatch {
        key: VulkanPlacedCablePacketKey,
        packet_byte_count: usize,
        incoming_byte_capacity: usize,
    },
    EndpointMismatch {
        outgoing_key: VulkanPlacedCablePacketKey,
        incoming_key: VulkanPlacedCablePacketKey,
    },
    Vulkan {
        operation: &'static str,
        error: VulkanError,
    },
}

impl Display for VulkanPlacedCableTransportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingOutgoingCable {
                device_id,
                cable_index,
            } => write!(
                f,
                "device {device_id:?} has no outgoing cable buffer for cable {cable_index}"
            ),
            Self::MissingIncomingCable {
                device_id,
                cable_index,
            } => write!(
                f,
                "device {device_id:?} has no incoming cable buffer for cable {cable_index}"
            ),
            Self::MissingPacket { key } => write!(
                f,
                "no in-process cable packet for cable {} from {:?} to {:?}",
                key.cable_index, key.from_device_id, key.to_device_id
            ),
            Self::ByteCapacityMismatch {
                key,
                packet_byte_count,
                incoming_byte_capacity,
            } => write!(
                f,
                "in-process cable packet for cable {} from {:?} to {:?} has {} bytes, but incoming buffer has {} bytes",
                key.cable_index,
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
                "direct in-process cable endpoint mismatch: outgoing cable {} from {:?} to {:?}, incoming cable {} from {:?} to {:?}",
                outgoing_key.cable_index,
                outgoing_key.from_device_id,
                outgoing_key.to_device_id,
                incoming_key.cable_index,
                incoming_key.from_device_id,
                incoming_key.to_device_id
            ),
            Self::Vulkan { operation, error } => write!(f, "{operation} failed: {error}"),
        }
    }
}

impl Error for VulkanPlacedCableTransportError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableIoPlanError(pub String);

impl Display for VulkanPlacedCableIoPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPlacedCableIoPlanError {}

