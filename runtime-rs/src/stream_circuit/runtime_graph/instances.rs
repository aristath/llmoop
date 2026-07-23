#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitNodeInstance {
    pub instance_id: String,
    pub source_component_id: String,
    pub device_id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub control_values: BTreeMap<String, serde_json::Value>,
    pub state_policy: StreamCircuitNodeInstanceStatePolicy,
}

fn default_enabled() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCircuitNodeInstanceStatePolicy {
    Fresh,
    CloneFrom { instance_id: String },
    ShareWith { instance_id: String },
}

fn series_edges_for_instances(
    graph: &ResolvedLoweredExecutionGraph,
    instances: &[StreamCircuitNodeInstance],
) -> Result<Vec<StreamCircuitGraphEdge>, CircuitPlacementError> {
    let source_by_id = graph
        .circuits
        .iter()
        .map(|artifact| (artifact.component.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    instances
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            let source = source_by_id
                .get(pair[0].source_component_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "runtime graph instance {} references unknown source component {}",
                        pair[0].instance_id, pair[0].source_component_id
                    ))
                })?;
            let destination = source_by_id
                .get(pair[1].source_component_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "runtime graph instance {} references unknown source component {}",
                        pair[1].instance_id, pair[1].source_component_id
                    ))
                })?;
            let output = single_series_port(
                &source.circuit.boundary.outputs,
                &pair[0].instance_id,
                "output",
            )?;
            let input = single_series_port(
                &destination.circuit.boundary.inputs,
                &pair[1].instance_id,
                "input",
            )?;
            Ok(StreamCircuitGraphEdge {
                id: format!("edge_{index:04}"),
                source: StreamCircuitEdgeEndpoint {
                    component_id: pair[0].instance_id.clone(),
                    port_id: output.id.clone(),
                },
                destination: StreamCircuitEdgeEndpoint {
                    component_id: pair[1].instance_id.clone(),
                    port_id: input.id.clone(),
                },
                connection: StreamCircuitConnection::Forward,
            })
        })
        .collect()
}

fn series_boundary_for_instances(
    graph: &ResolvedLoweredExecutionGraph,
    instances: &[StreamCircuitNodeInstance],
) -> Result<StreamCircuitGraphBoundary, CircuitPlacementError> {
    let source_by_id = graph
        .circuits
        .iter()
        .map(|artifact| (artifact.component.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    let first = instances.first().ok_or_else(|| {
        CircuitPlacementError("runtime source sequence must contain at least one component".to_string())
    })?;
    let last = instances
        .last()
        .expect("non-empty source sequence must have a last component");
    let first_source = source_by_id
        .get(first.source_component_id.as_str())
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime graph instance {} references unknown source component {}",
                first.instance_id, first.source_component_id
            ))
        })?;
    let last_source = source_by_id
        .get(last.source_component_id.as_str())
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime graph instance {} references unknown source component {}",
                last.instance_id, last.source_component_id
            ))
        })?;
    let input = single_series_port(
        &first_source.circuit.boundary.inputs,
        &first.instance_id,
        "input",
    )?;
    let output = single_series_port(
        &last_source.circuit.boundary.outputs,
        &last.instance_id,
        "output",
    )?;
    Ok(StreamCircuitGraphBoundary {
        external_inputs: vec![StreamCircuitGraphBoundaryPort {
            id: "model_input".to_string(),
            endpoint: StreamCircuitEdgeEndpoint {
                component_id: first.instance_id.clone(),
                port_id: input.id.clone(),
            },
        }],
        public_outputs: vec![StreamCircuitGraphBoundaryPort {
            id: "model_output".to_string(),
            endpoint: StreamCircuitEdgeEndpoint {
                component_id: last.instance_id.clone(),
                port_id: output.id.clone(),
            },
        }],
    })
}

fn single_series_port<'a>(
    ports: &'a [CircuitPort],
    instance_id: &str,
    direction: &str,
) -> Result<&'a CircuitPort, CircuitPlacementError> {
    if ports.len() != 1 {
        return Err(CircuitPlacementError(format!(
            "runtime series topology requires instance {instance_id} to expose exactly one {direction} port, found {}",
            ports.len()
        )));
    }
    Ok(&ports[0])
}

fn allocate_edge_id(
    edges: &[StreamCircuitGraphEdge],
    source_id: &str,
    destination_id: &str,
) -> String {
    let base = format!("{source_id}_to_{destination_id}");
    if !edges.iter().any(|edge| edge.id == base) {
        return base;
    }
    (2..)
        .map(|suffix| format!("{base}_{suffix}"))
        .find(|candidate| !edges.iter().any(|edge| edge.id == *candidate))
        .expect("unbounded edge id suffix space")
}

fn validate_instance_state_policy(
    instance: &StreamCircuitNodeInstance,
    instances: &[StreamCircuitNodeInstance],
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
) -> Result<(), CircuitPlacementError> {
    let target_id = match &instance.state_policy {
        StreamCircuitNodeInstanceStatePolicy::Fresh => return Ok(()),
        StreamCircuitNodeInstanceStatePolicy::CloneFrom { instance_id }
        | StreamCircuitNodeInstanceStatePolicy::ShareWith { instance_id } => instance_id,
    };
    if target_id == &instance.instance_id {
        return Err(CircuitPlacementError(format!(
            "runtime graph instance {} cannot source state from itself",
            instance.instance_id
        )));
    }
    let target = instances
        .iter()
        .find(|candidate| candidate.instance_id == *target_id)
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime graph instance {} sources state from unknown instance {}",
                instance.instance_id, target_id
            ))
        })?;
    if !target.enabled || !instance.enabled {
        return Err(CircuitPlacementError(format!(
            "state-linked instances {} and {} must both be enabled",
            instance.instance_id, target.instance_id
        )));
    }
    let source_state = &source_by_id[instance.source_component_id.as_str()]
        .state
        .state_ports;
    let target_state = &source_by_id[target.source_component_id.as_str()]
        .state
        .state_ports;
    if source_state != target_state {
        return Err(CircuitPlacementError(format!(
            "runtime graph instances {} and {} have incompatible state contracts",
            instance.instance_id, target.instance_id
        )));
    }
    if matches!(
        instance.state_policy,
        StreamCircuitNodeInstanceStatePolicy::ShareWith { .. }
    ) && instance.device_id != target.device_id
    {
        return Err(CircuitPlacementError(format!(
            "runtime graph instances {} and {} cannot share state across devices {} and {}",
            instance.instance_id, target.instance_id, instance.device_id, target.device_id
        )));
    }
    Ok(())
}
