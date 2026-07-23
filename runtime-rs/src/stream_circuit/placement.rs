#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPlacementPlan {
    pub schema: String,
    pub topology: String,
    pub components: Vec<ComponentPlacement>,
    pub edges: Vec<ComponentEdgePlacement>,
    pub local_edge_count: usize,
    pub cross_device_edge_count: usize,
}

impl StreamCircuitPlacementPlan {
    pub fn from_graph(
        graph: &ResolvedLoweredExecutionGraph,
        spec: &StreamCircuitPlacementSpec,
    ) -> Result<Self, CircuitPlacementError> {
        if spec.schema != STREAM_CIRCUIT_PLACEMENT_SCHEMA {
            return Err(CircuitPlacementError(format!(
                "unsupported stream-circuit placement schema {:?}",
                spec.schema
            )));
        }
        if spec.default_device_id.is_empty() {
            return Err(CircuitPlacementError(
                "placement default_device_id must not be empty".to_string(),
            ));
        }
        let component_ids: BTreeSet<_> = graph
            .circuits
            .iter()
            .map(|artifact| artifact.component.id.as_str())
            .collect();
        for component_id in spec.node_devices.keys() {
            if !component_ids.contains(component_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "placement references unknown component {component_id:?}"
                )));
            }
        }

        let components = graph
            .circuits
            .iter()
            .enumerate()
            .map(|(component_index, artifact)| ComponentPlacement {
                component_index,
                component_id: artifact.component.id.clone(),
                circuit_id: artifact.circuit.id.clone(),
                operator_type: artifact.component.operator_type.clone(),
                device_id: spec.device_for_component(&artifact.component.id).to_string(),
            })
            .collect::<Vec<_>>();

        let circuit_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.component.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut edges = Vec::with_capacity(graph.index.graph.edges.len());
        let mut local_edge_count = 0usize;
        let mut cross_device_edge_count = 0usize;
        for (edge_index, edge) in graph.index.graph.edges.iter().enumerate() {
            let source = circuit_by_id
                .get(edge.source.component_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "placement edge {} references unknown source component {}",
                        edge.id, edge.source.component_id
                    ))
                })?;
            let destination = circuit_by_id
                .get(edge.destination.component_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "placement edge {} references unknown destination component {}",
                        edge.id, edge.destination.component_id
                    ))
                })?;
            let output = source
                .circuit
                .boundary
                .outputs
                .iter()
                .find(|port| port.id == edge.source.port_id)
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "{} has no output port {} for placement edge {}",
                        source.component.id, edge.source.port_id, edge.id
                    ))
                })?;
            let input = destination
                .circuit
                .boundary
                .inputs
                .iter()
                .find(|port| port.id == edge.destination.port_id)
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "{} has no input port {} for placement edge {}",
                        destination.component.id, edge.destination.port_id, edge.id
                    ))
                })?;
            if output.signal != input.signal || output.shape != input.shape {
                return Err(CircuitPlacementError(format!(
                    "cannot place edge {} -> {} without an adapter: output {:?}/{:?}, input {:?}/{:?}",
                    source.component.id,
                    destination.component.id,
                    output.signal,
                    output.shape,
                    input.signal,
                    input.shape
                )));
            }

            let source_device_id = spec.device_for_component(&source.component.id).to_string();
            let destination_device_id = spec.device_for_component(&destination.component.id).to_string();
            let transport = if source_device_id == destination_device_id {
                local_edge_count += 1;
                EdgeTransport::LocalBuffer {
                    device_id: source_device_id.clone(),
                }
            } else {
                cross_device_edge_count += 1;
                EdgeTransport::CrossDevice {
                    from_device_id: source_device_id.clone(),
                    to_device_id: destination_device_id.clone(),
                }
            };

            edges.push(ComponentEdgePlacement {
                edge_index,
                connection: edge.connection.clone(),
                signal: output.signal.clone(),
                shape: output.shape.clone(),
                source_component_id: source.component.id.clone(),
                source_device_id,
                source_port_id: output.id.clone(),
                source_component_port: output.component_port.clone(),
                destination_component_id: destination.component.id.clone(),
                destination_device_id,
                destination_port_id: input.id.clone(),
                destination_component_port: input.component_port.clone(),
                transport,
            });
        }

        Ok(Self {
            schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
            topology: graph.index.graph.topology.clone(),
            components,
            edges,
            local_edge_count,
            cross_device_edge_count,
        })
    }

    pub fn component(&self, component_id: &str) -> Option<&ComponentPlacement> {
        self.components.iter().find(|component| component.component_id == component_id)
    }

    pub fn cross_device_edges(&self) -> Vec<&ComponentEdgePlacement> {
        self.edges
            .iter()
            .filter(|edge| edge.transport.is_cross_device())
            .collect()
    }

    pub fn runtime_edge_routes<F>(&self, target_for: F) -> RuntimeEdgeRoutes
    where
        F: FnMut(&str) -> RuntimeEdgeRouteTarget,
    {
        RuntimeEdgeRoutes::from_edges(&self.edges, target_for)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentPlacement {
    pub component_index: usize,
    pub component_id: String,
    pub circuit_id: String,
    pub operator_type: String,
    pub device_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentEdgePlacement {
    pub edge_index: usize,
    pub connection: StreamCircuitConnection,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source_component_id: String,
    pub source_device_id: String,
    pub source_port_id: String,
    pub source_component_port: Option<String>,
    pub destination_component_id: String,
    pub destination_device_id: String,
    pub destination_port_id: String,
    pub destination_component_port: Option<String>,
    pub transport: EdgeTransport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EdgeTransport {
    LocalBuffer {
        device_id: String,
    },
    CrossDevice {
        from_device_id: String,
        to_device_id: String,
    },
}

impl EdgeTransport {
    pub fn is_cross_device(&self) -> bool {
        matches!(self, Self::CrossDevice { .. })
    }
}

