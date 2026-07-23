#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPlacementPlan {
    pub schema: String,
    pub wiring: String,
    pub pedals: Vec<PedalPlacement>,
    pub cables: Vec<PedalCablePlacement>,
    pub local_cable_count: usize,
    pub cross_device_cable_count: usize,
}

impl StreamCircuitPlacementPlan {
    pub fn from_graph(
        graph: &ResolvedLoweredPedalboard,
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
        let pedal_ids: BTreeSet<_> = graph
            .circuits
            .iter()
            .map(|artifact| artifact.pedal.id.as_str())
            .collect();
        for pedal_id in spec.pedal_devices.keys() {
            if !pedal_ids.contains(pedal_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "placement references unknown pedal {pedal_id:?}"
                )));
            }
        }

        let pedals = graph
            .circuits
            .iter()
            .enumerate()
            .map(|(pedal_index, artifact)| PedalPlacement {
                pedal_index,
                pedal_id: artifact.pedal.id.clone(),
                circuit_id: artifact.circuit.id.clone(),
                operator_type: artifact.pedal.operator_type.clone(),
                device_id: spec.device_for_pedal(&artifact.pedal.id).to_string(),
            })
            .collect::<Vec<_>>();

        let circuit_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.pedal.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut cables = Vec::with_capacity(graph.index.graph.cables.len());
        let mut local_cable_count = 0usize;
        let mut cross_device_cable_count = 0usize;
        for (cable_index, cable) in graph.index.graph.cables.iter().enumerate() {
            let source = circuit_by_id
                .get(cable.source.pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "placement cable {} references unknown source pedal {}",
                        cable.id, cable.source.pedal_id
                    ))
                })?;
            let destination = circuit_by_id
                .get(cable.destination.pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "placement cable {} references unknown destination pedal {}",
                        cable.id, cable.destination.pedal_id
                    ))
                })?;
            let output = source
                .circuit
                .boundary
                .outputs
                .iter()
                .find(|port| port.id == cable.source.port_id)
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "{} has no output port {} for placement cable {}",
                        source.pedal.id, cable.source.port_id, cable.id
                    ))
                })?;
            let input = destination
                .circuit
                .boundary
                .inputs
                .iter()
                .find(|port| port.id == cable.destination.port_id)
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "{} has no input port {} for placement cable {}",
                        destination.pedal.id, cable.destination.port_id, cable.id
                    ))
                })?;
            if output.signal != input.signal || output.shape != input.shape {
                return Err(CircuitPlacementError(format!(
                    "cannot place cable {} -> {} without an adapter: output {:?}/{:?}, input {:?}/{:?}",
                    source.pedal.id,
                    destination.pedal.id,
                    output.signal,
                    output.shape,
                    input.signal,
                    input.shape
                )));
            }

            let source_device_id = spec.device_for_pedal(&source.pedal.id).to_string();
            let destination_device_id = spec.device_for_pedal(&destination.pedal.id).to_string();
            let transport = if source_device_id == destination_device_id {
                local_cable_count += 1;
                CableTransport::LocalBuffer {
                    device_id: source_device_id.clone(),
                }
            } else {
                cross_device_cable_count += 1;
                CableTransport::CrossDevice {
                    from_device_id: source_device_id.clone(),
                    to_device_id: destination_device_id.clone(),
                }
            };

            cables.push(PedalCablePlacement {
                cable_index,
                connection: cable.connection.clone(),
                signal: output.signal.clone(),
                shape: output.shape.clone(),
                source_pedal_id: source.pedal.id.clone(),
                source_device_id,
                source_port_id: output.id.clone(),
                source_pedal_port: output.pedal_port.clone(),
                destination_pedal_id: destination.pedal.id.clone(),
                destination_device_id,
                destination_port_id: input.id.clone(),
                destination_pedal_port: input.pedal_port.clone(),
                transport,
            });
        }

        Ok(Self {
            schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
            wiring: graph.index.graph.wiring.clone(),
            pedals,
            cables,
            local_cable_count,
            cross_device_cable_count,
        })
    }

    pub fn pedal(&self, pedal_id: &str) -> Option<&PedalPlacement> {
        self.pedals.iter().find(|pedal| pedal.pedal_id == pedal_id)
    }

    pub fn cross_device_cables(&self) -> Vec<&PedalCablePlacement> {
        self.cables
            .iter()
            .filter(|cable| cable.transport.is_cross_device())
            .collect()
    }

    pub fn runtime_cable_routes<F>(&self, target_for: F) -> RuntimeCableRoutes
    where
        F: FnMut(&str) -> RuntimeCableRouteTarget,
    {
        RuntimeCableRoutes::from_cables(&self.cables, target_for)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PedalPlacement {
    pub pedal_index: usize,
    pub pedal_id: String,
    pub circuit_id: String,
    pub operator_type: String,
    pub device_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PedalCablePlacement {
    pub cable_index: usize,
    pub connection: StreamCircuitConnection,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source_pedal_id: String,
    pub source_device_id: String,
    pub source_port_id: String,
    pub source_pedal_port: Option<String>,
    pub destination_pedal_id: String,
    pub destination_device_id: String,
    pub destination_port_id: String,
    pub destination_pedal_port: Option<String>,
    pub transport: CableTransport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CableTransport {
    LocalBuffer {
        device_id: String,
    },
    CrossDevice {
        from_device_id: String,
        to_device_id: String,
    },
}

impl CableTransport {
    pub fn is_cross_device(&self) -> bool {
        matches!(self, Self::CrossDevice { .. })
    }
}

