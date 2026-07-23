#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgeIoPlan {
    pub backend_id: String,
    pub device_id: String,
    pub signal_element_bytes: Option<usize>,
    pub local_edges: Vec<VulkanPlacedLocalEdge>,
    pub endpoints: Vec<VulkanPlacedEdgeEndpoint>,
    pub local_edge_count: usize,
    pub incoming_endpoint_count: usize,
    pub outgoing_endpoint_count: usize,
    pub total_buffer_count: usize,
    pub total_endpoint_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_byte_edges: Vec<usize>,
}

impl VulkanPlacedEdgeIoPlan {
    pub fn from_placed_resident_plan(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanPlacedEdgeIoPlanError> {
        let mut local_edges = Vec::with_capacity(placed_resident_plan.local_edges.len());
        for edge in &placed_resident_plan.local_edges {
            local_edges.push(VulkanPlacedLocalEdge::from_edge(
                local_edges.len(),
                &placed_resident_plan.device_id,
                edge,
                placed_resident_plan.signal_element_bytes,
            )?);
        }

        let mut endpoints = Vec::with_capacity(
            placed_resident_plan.incoming_edges.len() + placed_resident_plan.outgoing_edges.len(),
        );

        for edge in &placed_resident_plan.incoming_edges {
            endpoints.push(VulkanPlacedEdgeEndpoint::from_edge(
                endpoints.len(),
                VulkanPlacedEdgeDirection::Incoming,
                &placed_resident_plan.device_id,
                edge,
                placed_resident_plan.signal_element_bytes,
            )?);
        }
        for edge in &placed_resident_plan.outgoing_edges {
            endpoints.push(VulkanPlacedEdgeEndpoint::from_edge(
                endpoints.len(),
                VulkanPlacedEdgeDirection::Outgoing,
                &placed_resident_plan.device_id,
                edge,
                placed_resident_plan.signal_element_bytes,
            )?);
        }

        let local_edge_count = local_edges.len();
        let incoming_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
            .count();
        let outgoing_endpoint_count = endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
            .count();
        let unresolved_byte_edges = local_edges
            .iter()
            .filter(|edge| edge.byte_capacity.is_none())
            .map(|edge| edge.edge_index)
            .chain(
                endpoints
                    .iter()
                    .filter(|endpoint| endpoint.byte_capacity.is_none())
                    .map(|endpoint| endpoint.edge_index),
            )
            .collect::<Vec<_>>();
        let total_byte_capacity = local_edges
            .iter()
            .map(|edge| edge.byte_capacity)
            .chain(endpoints.iter().map(|endpoint| endpoint.byte_capacity))
            .try_fold(Some(0usize), |total, byte_capacity| {
                match (total, byte_capacity) {
                    (Some(total), Some(bytes)) => Some(total.checked_add(bytes).ok_or_else(|| {
                        VulkanPlacedEdgeIoPlanError(
                            "placed edge buffer byte capacity overflowed".to_string(),
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
            local_edges,
            local_edge_count,
            total_buffer_count: local_edge_count + endpoints.len(),
            total_endpoint_count: endpoints.len(),
            endpoints,
            incoming_endpoint_count,
            outgoing_endpoint_count,
            total_byte_capacity,
            unresolved_byte_edges,
        })
    }

    pub fn endpoint(
        &self,
        direction: VulkanPlacedEdgeDirection,
        edge_index: usize,
    ) -> Option<&VulkanPlacedEdgeEndpoint> {
        self.endpoints
            .iter()
            .find(|endpoint| endpoint.direction == direction && endpoint.edge_index == edge_index)
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanPlacedEdgeIoBuffers, VulkanError> {
        self.allocate_buffers_with_endpoint_overrides(device, &[])
    }

    pub fn allocate_buffers_with_endpoint_overrides(
        &self,
        device: &VulkanComputeDevice,
        endpoint_overrides: &[VulkanPlacedEdgeEndpointBufferOverride],
    ) -> Result<VulkanPlacedEdgeIoBuffers, VulkanError> {
        let mut local_buffers = Vec::with_capacity(self.local_edge_count);
        let mut incoming_buffers = Vec::with_capacity(self.incoming_endpoint_count);
        let mut outgoing_buffers = Vec::with_capacity(self.outgoing_endpoint_count);
        let mut total_byte_capacity = 0usize;
        let mut overrides = BTreeMap::new();

        for endpoint_override in endpoint_overrides {
            let key = (endpoint_override.direction, endpoint_override.edge_index);
            if overrides.insert(key, endpoint_override).is_some() {
                return Err(VulkanError(format!(
                    "edge endpoint buffer override repeats {:?} edge {} on {:?}",
                    endpoint_override.direction, endpoint_override.edge_index, self.device_id
                )));
            }
            if !device.owns_resident_buffer(&endpoint_override.buffer) {
                return Err(VulkanError(format!(
                    "edge endpoint buffer override for edge {} on {:?} belongs to a different Vulkan logical device",
                    endpoint_override.edge_index, self.device_id
                )));
            }
            let endpoint = self
                .endpoint(endpoint_override.direction, endpoint_override.edge_index)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "edge endpoint buffer override does not address {:?} edge {} on {:?}",
                        endpoint_override.direction, endpoint_override.edge_index, self.device_id
                    ))
                })?;
            let required_byte_capacity = endpoint.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} endpoint {} for edge {} has unknown byte capacity",
                    self.device_id, endpoint.endpoint_id, endpoint.edge_index
                ))
            })?;
            if endpoint_override.buffer.byte_capacity() < required_byte_capacity {
                return Err(VulkanError(format!(
                    "edge endpoint buffer override for edge {} on {:?} has {} bytes, needs {required_byte_capacity}",
                    endpoint_override.edge_index,
                    self.device_id,
                    endpoint_override.buffer.byte_capacity()
                )));
            }
        }

        for edge in &self.local_edges {
            let byte_capacity = edge.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} local edge {} has unknown byte capacity",
                    self.device_id, edge.edge_index
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "placed local edge buffer allocation",
            )?;
            local_buffers.push(VulkanPlacedLocalEdgeBufferAllocation {
                edge: edge.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        for endpoint in &self.endpoints {
            let byte_capacity = endpoint.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} endpoint {} for edge {} has unknown byte capacity",
                    self.device_id, endpoint.endpoint_id, endpoint.edge_index
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "placed edge endpoint buffer allocation",
            )?;
            let buffer = if let Some(endpoint_override) =
                overrides.get(&(endpoint.direction, endpoint.edge_index))
            {
                endpoint_override.buffer.clone()
            } else {
                let mut buffer = device.create_host_visible_resident_buffer(byte_capacity)?;
                buffer.persistently_map()?;
                Arc::new(buffer)
            };
            let allocation = VulkanPlacedEdgeBufferAllocation {
                endpoint: endpoint.clone(),
                byte_capacity,
                buffer,
            };
            match endpoint.direction {
                VulkanPlacedEdgeDirection::Incoming => incoming_buffers.push(allocation),
                VulkanPlacedEdgeDirection::Outgoing => outgoing_buffers.push(allocation),
            }
        }

        Ok(VulkanPlacedEdgeIoBuffers {
            plan: self.clone(),
            local_buffers,
            incoming_buffers,
            outgoing_buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedLocalEdge {
    pub buffer_index: usize,
    pub edge_id: String,
    pub edge_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub device_id: String,
    pub source_component_id: String,
    pub source_port_id: String,
    pub source_component_port: Option<String>,
    pub destination_component_id: String,
    pub destination_port_id: String,
    pub destination_component_port: Option<String>,
    pub transport: EdgeTransport,
}

impl VulkanPlacedLocalEdge {
    fn from_edge(
        buffer_index: usize,
        device_id: &str,
        edge: &ComponentEdgePlacement,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanPlacedEdgeIoPlanError> {
        let EdgeTransport::LocalBuffer {
            device_id: transport_device_id,
        } = &edge.transport
        else {
            return Err(VulkanPlacedEdgeIoPlanError(format!(
                "edge {} is not a local edge",
                edge.edge_index
            )));
        };

        if transport_device_id != device_id
            || edge.source_device_id != device_id
            || edge.destination_device_id != device_id
        {
            return Err(VulkanPlacedEdgeIoPlanError(format!(
                "local edge {} is not fully resident on device {:?}",
                edge.edge_index, device_id
            )));
        }

        let element_count = edge_element_count(edge)?;
        let byte_capacity = edge_byte_capacity(edge, element_count, signal_element_bytes)?;

        Ok(Self {
            buffer_index,
            edge_id: format!("edge_{}_local", edge.edge_index),
            edge_index: edge.edge_index,
            signal: edge.signal.clone(),
            shape: edge.shape.clone(),
            element_count,
            byte_capacity,
            device_id: device_id.to_string(),
            source_component_id: edge.source_component_id.clone(),
            source_port_id: edge.source_port_id.clone(),
            source_component_port: edge.source_component_port.clone(),
            destination_component_id: edge.destination_component_id.clone(),
            destination_port_id: edge.destination_port_id.clone(),
            destination_component_port: edge.destination_component_port.clone(),
            transport: edge.transport.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgeEndpoint {
    pub endpoint_index: usize,
    pub endpoint_id: String,
    pub direction: VulkanPlacedEdgeDirection,
    pub edge_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub local_device_id: String,
    pub remote_device_id: String,
    pub local_component_id: String,
    pub remote_component_id: String,
    pub local_port_id: String,
    pub remote_port_id: String,
    pub local_component_port: Option<String>,
    pub remote_component_port: Option<String>,
    pub transport: EdgeTransport,
}

impl VulkanPlacedEdgeEndpoint {
    fn from_edge(
        endpoint_index: usize,
        direction: VulkanPlacedEdgeDirection,
        device_id: &str,
        edge: &ComponentEdgePlacement,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanPlacedEdgeIoPlanError> {
        let EdgeTransport::CrossDevice {
            from_device_id,
            to_device_id,
        } = &edge.transport
        else {
            return Err(VulkanPlacedEdgeIoPlanError(format!(
                "edge {} is not a cross-device edge",
                edge.edge_index
            )));
        };

        match direction {
            VulkanPlacedEdgeDirection::Incoming => {
                if to_device_id != device_id || edge.destination_device_id != device_id {
                    return Err(VulkanPlacedEdgeIoPlanError(format!(
                        "incoming edge {} does not terminate on device {:?}",
                        edge.edge_index, device_id
                    )));
                }
            }
            VulkanPlacedEdgeDirection::Outgoing => {
                if from_device_id != device_id || edge.source_device_id != device_id {
                    return Err(VulkanPlacedEdgeIoPlanError(format!(
                        "outgoing edge {} does not originate on device {:?}",
                        edge.edge_index, device_id
                    )));
                }
            }
        }

        let element_count = edge_element_count(edge)?;
        let byte_capacity = edge_byte_capacity(edge, element_count, signal_element_bytes)?;

        let (
            local_device_id,
            remote_device_id,
            local_component_id,
            remote_component_id,
            local_port_id,
            remote_port_id,
            local_component_port,
            remote_component_port,
        ) = match direction {
            VulkanPlacedEdgeDirection::Incoming => (
                edge.destination_device_id.clone(),
                edge.source_device_id.clone(),
                edge.destination_component_id.clone(),
                edge.source_component_id.clone(),
                edge.destination_port_id.clone(),
                edge.source_port_id.clone(),
                edge.destination_component_port.clone(),
                edge.source_component_port.clone(),
            ),
            VulkanPlacedEdgeDirection::Outgoing => (
                edge.source_device_id.clone(),
                edge.destination_device_id.clone(),
                edge.source_component_id.clone(),
                edge.destination_component_id.clone(),
                edge.source_port_id.clone(),
                edge.destination_port_id.clone(),
                edge.source_component_port.clone(),
                edge.destination_component_port.clone(),
            ),
        };
        let direction_suffix = match direction {
            VulkanPlacedEdgeDirection::Incoming => "in",
            VulkanPlacedEdgeDirection::Outgoing => "out",
        };

        Ok(Self {
            endpoint_index,
            endpoint_id: format!("edge_{}_{}", edge.edge_index, direction_suffix),
            direction,
            edge_index: edge.edge_index,
            signal: edge.signal.clone(),
            shape: edge.shape.clone(),
            element_count,
            byte_capacity,
            local_device_id,
            remote_device_id,
            local_component_id,
            remote_component_id,
            local_port_id,
            remote_port_id,
            local_component_port,
            remote_component_port,
            transport: edge.transport.clone(),
        })
    }
}

fn edge_element_count(edge: &ComponentEdgePlacement) -> Result<usize, VulkanPlacedEdgeIoPlanError> {
    let element_count = product(&edge.shape).ok_or_else(|| {
        VulkanPlacedEdgeIoPlanError(format!(
            "edge {} signal shape {:?} overflows",
            edge.edge_index, edge.shape
        ))
    })?;
    if element_count == 0 {
        return Err(VulkanPlacedEdgeIoPlanError(format!(
            "edge {} signal shape {:?} has zero elements",
            edge.edge_index, edge.shape
        )));
    }
    Ok(element_count)
}

fn edge_byte_capacity(
    edge: &ComponentEdgePlacement,
    element_count: usize,
    signal_element_bytes: Option<usize>,
) -> Result<Option<usize>, VulkanPlacedEdgeIoPlanError> {
    match signal_element_bytes {
        Some(bytes_per_element) => element_count
            .checked_mul(bytes_per_element)
            .map(Some)
            .ok_or_else(|| {
                VulkanPlacedEdgeIoPlanError(format!(
                    "edge {} byte capacity overflowed",
                    edge.edge_index
                ))
            }),
        None => Ok(None),
    }
}

