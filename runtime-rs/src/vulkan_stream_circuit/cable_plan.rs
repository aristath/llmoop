#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableIoPlan {
    pub backend_id: String,
    pub device_id: String,
    pub signal_element_bytes: Option<usize>,
    pub local_cables: Vec<VulkanPlacedLocalCable>,
    pub endpoints: Vec<VulkanPlacedCableEndpoint>,
    pub local_cable_count: usize,
    pub incoming_endpoint_count: usize,
    pub outgoing_endpoint_count: usize,
    pub total_buffer_count: usize,
    pub total_endpoint_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_byte_cables: Vec<usize>,
}

impl VulkanPlacedCableIoPlan {
    pub fn from_placed_resident_plan(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
        let mut local_cables = Vec::with_capacity(placed_resident_plan.local_cables.len());
        for cable in &placed_resident_plan.local_cables {
            local_cables.push(VulkanPlacedLocalCable::from_cable(
                local_cables.len(),
                &placed_resident_plan.device_id,
                cable,
                placed_resident_plan.signal_element_bytes,
            )?);
        }

        let mut endpoints = Vec::with_capacity(
            placed_resident_plan.incoming_cables.len() + placed_resident_plan.outgoing_cables.len(),
        );

        for cable in &placed_resident_plan.incoming_cables {
            endpoints.push(VulkanPlacedCableEndpoint::from_cable(
                endpoints.len(),
                VulkanPlacedCableDirection::Incoming,
                &placed_resident_plan.device_id,
                cable,
                placed_resident_plan.signal_element_bytes,
            )?);
        }
        for cable in &placed_resident_plan.outgoing_cables {
            endpoints.push(VulkanPlacedCableEndpoint::from_cable(
                endpoints.len(),
                VulkanPlacedCableDirection::Outgoing,
                &placed_resident_plan.device_id,
                cable,
                placed_resident_plan.signal_element_bytes,
            )?);
        }

        let local_cable_count = local_cables.len();
        let incoming_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedCableDirection::Incoming)
            .count();
        let outgoing_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedCableDirection::Outgoing)
            .count();
        let unresolved_byte_cables = local_cables
            .iter()
            .filter(|cable| cable.byte_capacity.is_none())
            .map(|cable| cable.cable_index)
            .chain(
                endpoints
                    .iter()
                    .filter(|endpoint| endpoint.byte_capacity.is_none())
                    .map(|endpoint| endpoint.cable_index),
            )
            .collect::<Vec<_>>();
        let total_byte_capacity = local_cables
            .iter()
            .map(|cable| cable.byte_capacity)
            .chain(endpoints.iter().map(|endpoint| endpoint.byte_capacity))
            .try_fold(Some(0usize), |total, byte_capacity| {
                match (total, byte_capacity) {
                    (Some(total), Some(bytes)) => Some(total.checked_add(bytes).ok_or_else(|| {
                        VulkanPlacedCableIoPlanError(
                            "placed cable buffer byte capacity overflowed".to_string(),
                        )
                    }))
                    .transpose(),
                    _ => Ok(None),
                }
            })?;

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_resident_plan.device_id.clone(),
            signal_element_bytes: placed_resident_plan.signal_element_bytes,
            local_cables,
            local_cable_count,
            total_buffer_count: local_cable_count + endpoints.len(),
            total_endpoint_count: endpoints.len(),
            endpoints,
            incoming_endpoint_count,
            outgoing_endpoint_count,
            total_byte_capacity,
            unresolved_byte_cables,
        })
    }

    pub fn endpoint(
        &self,
        direction: VulkanPlacedCableDirection,
        cable_index: usize,
    ) -> Option<&VulkanPlacedCableEndpoint> {
        self.endpoints
            .iter()
            .find(|endpoint| endpoint.direction == direction && endpoint.cable_index == cable_index)
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanPlacedCableIoBuffers, VulkanError> {
        self.allocate_buffers_with_endpoint_overrides(device, &[])
    }

    pub fn allocate_buffers_with_endpoint_overrides(
        &self,
        device: &VulkanComputeDevice,
        endpoint_overrides: &[VulkanPlacedCableEndpointBufferOverride],
    ) -> Result<VulkanPlacedCableIoBuffers, VulkanError> {
        let mut local_buffers = Vec::with_capacity(self.local_cable_count);
        let mut incoming_buffers = Vec::with_capacity(self.incoming_endpoint_count);
        let mut outgoing_buffers = Vec::with_capacity(self.outgoing_endpoint_count);
        let mut total_byte_capacity = 0usize;
        let mut overrides = BTreeMap::new();

        for endpoint_override in endpoint_overrides {
            let key = (endpoint_override.direction, endpoint_override.cable_index);
            if overrides.insert(key, endpoint_override).is_some() {
                return Err(VulkanError(format!(
                    "cable endpoint buffer override repeats {:?} cable {} on {:?}",
                    endpoint_override.direction, endpoint_override.cable_index, self.device_id
                )));
            }
            if !device.owns_resident_buffer(&endpoint_override.buffer) {
                return Err(VulkanError(format!(
                    "cable endpoint buffer override for cable {} on {:?} belongs to a different Vulkan logical device",
                    endpoint_override.cable_index, self.device_id
                )));
            }
            let endpoint = self
                .endpoint(endpoint_override.direction, endpoint_override.cable_index)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "cable endpoint buffer override does not address {:?} cable {} on {:?}",
                        endpoint_override.direction, endpoint_override.cable_index, self.device_id
                    ))
                })?;
            let required_byte_capacity = endpoint.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} endpoint {} for cable {} has unknown byte capacity",
                    self.device_id, endpoint.endpoint_id, endpoint.cable_index
                ))
            })?;
            if endpoint_override.buffer.byte_capacity() < required_byte_capacity {
                return Err(VulkanError(format!(
                    "cable endpoint buffer override for cable {} on {:?} has {} bytes, needs {required_byte_capacity}",
                    endpoint_override.cable_index,
                    self.device_id,
                    endpoint_override.buffer.byte_capacity()
                )));
            }
        }

        for cable in &self.local_cables {
            let byte_capacity = cable.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} local cable {} has unknown byte capacity",
                    self.device_id, cable.cable_index
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "placed local cable buffer allocation",
            )?;
            local_buffers.push(VulkanPlacedLocalCableBufferAllocation {
                cable: cable.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        for endpoint in &self.endpoints {
            let byte_capacity = endpoint.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} endpoint {} for cable {} has unknown byte capacity",
                    self.device_id, endpoint.endpoint_id, endpoint.cable_index
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "placed cable endpoint buffer allocation",
            )?;
            let buffer = if let Some(endpoint_override) =
                overrides.get(&(endpoint.direction, endpoint.cable_index))
            {
                endpoint_override.buffer.clone()
            } else {
                let mut buffer = device.create_host_visible_resident_buffer(byte_capacity)?;
                buffer.persistently_map()?;
                Arc::new(buffer)
            };
            let allocation = VulkanPlacedCableBufferAllocation {
                endpoint: endpoint.clone(),
                byte_capacity,
                buffer,
            };
            match endpoint.direction {
                VulkanPlacedCableDirection::Incoming => incoming_buffers.push(allocation),
                VulkanPlacedCableDirection::Outgoing => outgoing_buffers.push(allocation),
            }
        }

        Ok(VulkanPlacedCableIoBuffers {
            plan: self.clone(),
            local_buffers,
            incoming_buffers,
            outgoing_buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedLocalCable {
    pub buffer_index: usize,
    pub cable_id: String,
    pub cable_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub device_id: String,
    pub source_pedal_id: String,
    pub source_port_id: String,
    pub source_pedal_port: Option<String>,
    pub destination_pedal_id: String,
    pub destination_port_id: String,
    pub destination_pedal_port: Option<String>,
    pub transport: CableTransport,
}

impl VulkanPlacedLocalCable {
    fn from_cable(
        buffer_index: usize,
        device_id: &str,
        cable: &PedalCablePlacement,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
        let CableTransport::LocalBuffer {
            device_id: transport_device_id,
        } = &cable.transport
        else {
            return Err(VulkanPlacedCableIoPlanError(format!(
                "cable {} is not a local cable",
                cable.cable_index
            )));
        };

        if transport_device_id != device_id
            || cable.source_device_id != device_id
            || cable.destination_device_id != device_id
        {
            return Err(VulkanPlacedCableIoPlanError(format!(
                "local cable {} is not fully resident on device {:?}",
                cable.cable_index, device_id
            )));
        }

        let element_count = cable_element_count(cable)?;
        let byte_capacity = cable_byte_capacity(cable, element_count, signal_element_bytes)?;

        Ok(Self {
            buffer_index,
            cable_id: format!("cable_{}_local", cable.cable_index),
            cable_index: cable.cable_index,
            signal: cable.signal.clone(),
            shape: cable.shape.clone(),
            element_count,
            byte_capacity,
            device_id: device_id.to_string(),
            source_pedal_id: cable.source_pedal_id.clone(),
            source_port_id: cable.source_port_id.clone(),
            source_pedal_port: cable.source_pedal_port.clone(),
            destination_pedal_id: cable.destination_pedal_id.clone(),
            destination_port_id: cable.destination_port_id.clone(),
            destination_pedal_port: cable.destination_pedal_port.clone(),
            transport: cable.transport.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableEndpoint {
    pub endpoint_index: usize,
    pub endpoint_id: String,
    pub direction: VulkanPlacedCableDirection,
    pub cable_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub local_device_id: String,
    pub remote_device_id: String,
    pub local_pedal_id: String,
    pub remote_pedal_id: String,
    pub local_port_id: String,
    pub remote_port_id: String,
    pub local_pedal_port: Option<String>,
    pub remote_pedal_port: Option<String>,
    pub transport: CableTransport,
}

impl VulkanPlacedCableEndpoint {
    fn from_cable(
        endpoint_index: usize,
        direction: VulkanPlacedCableDirection,
        device_id: &str,
        cable: &PedalCablePlacement,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanPlacedCableIoPlanError> {
        let CableTransport::CrossDevice {
            from_device_id,
            to_device_id,
        } = &cable.transport
        else {
            return Err(VulkanPlacedCableIoPlanError(format!(
                "cable {} is not a cross-device cable",
                cable.cable_index
            )));
        };

        match direction {
            VulkanPlacedCableDirection::Incoming => {
                if to_device_id != device_id || cable.destination_device_id != device_id {
                    return Err(VulkanPlacedCableIoPlanError(format!(
                        "incoming cable {} does not terminate on device {:?}",
                        cable.cable_index, device_id
                    )));
                }
            }
            VulkanPlacedCableDirection::Outgoing => {
                if from_device_id != device_id || cable.source_device_id != device_id {
                    return Err(VulkanPlacedCableIoPlanError(format!(
                        "outgoing cable {} does not originate on device {:?}",
                        cable.cable_index, device_id
                    )));
                }
            }
        }

        let element_count = cable_element_count(cable)?;
        let byte_capacity = cable_byte_capacity(cable, element_count, signal_element_bytes)?;

        let (
            local_device_id,
            remote_device_id,
            local_pedal_id,
            remote_pedal_id,
            local_port_id,
            remote_port_id,
            local_pedal_port,
            remote_pedal_port,
        ) = match direction {
            VulkanPlacedCableDirection::Incoming => (
                cable.destination_device_id.clone(),
                cable.source_device_id.clone(),
                cable.destination_pedal_id.clone(),
                cable.source_pedal_id.clone(),
                cable.destination_port_id.clone(),
                cable.source_port_id.clone(),
                cable.destination_pedal_port.clone(),
                cable.source_pedal_port.clone(),
            ),
            VulkanPlacedCableDirection::Outgoing => (
                cable.source_device_id.clone(),
                cable.destination_device_id.clone(),
                cable.source_pedal_id.clone(),
                cable.destination_pedal_id.clone(),
                cable.source_port_id.clone(),
                cable.destination_port_id.clone(),
                cable.source_pedal_port.clone(),
                cable.destination_pedal_port.clone(),
            ),
        };
        let direction_suffix = match direction {
            VulkanPlacedCableDirection::Incoming => "in",
            VulkanPlacedCableDirection::Outgoing => "out",
        };

        Ok(Self {
            endpoint_index,
            endpoint_id: format!("cable_{}_{}", cable.cable_index, direction_suffix),
            direction,
            cable_index: cable.cable_index,
            signal: cable.signal.clone(),
            shape: cable.shape.clone(),
            element_count,
            byte_capacity,
            local_device_id,
            remote_device_id,
            local_pedal_id,
            remote_pedal_id,
            local_port_id,
            remote_port_id,
            local_pedal_port,
            remote_pedal_port,
            transport: cable.transport.clone(),
        })
    }
}

fn cable_element_count(cable: &PedalCablePlacement) -> Result<usize, VulkanPlacedCableIoPlanError> {
    let element_count = product(&cable.shape).ok_or_else(|| {
        VulkanPlacedCableIoPlanError(format!(
            "cable {} signal shape {:?} overflows",
            cable.cable_index, cable.shape
        ))
    })?;
    if element_count == 0 {
        return Err(VulkanPlacedCableIoPlanError(format!(
            "cable {} signal shape {:?} has zero elements",
            cable.cable_index, cable.shape
        )));
    }
    Ok(element_count)
}

fn cable_byte_capacity(
    cable: &PedalCablePlacement,
    element_count: usize,
    signal_element_bytes: Option<usize>,
) -> Result<Option<usize>, VulkanPlacedCableIoPlanError> {
    match signal_element_bytes {
        Some(bytes_per_element) => element_count
            .checked_mul(bytes_per_element)
            .map(Some)
            .ok_or_else(|| {
                VulkanPlacedCableIoPlanError(format!(
                    "cable {} byte capacity overflowed",
                    cable.cable_index
                ))
            }),
        None => Ok(None),
    }
}

