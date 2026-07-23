fn effective_runtime_graph_edges(
    instances: &[StreamCircuitNodeInstance],
    edges: &[StreamCircuitGraphEdge],
) -> Result<Vec<StreamCircuitGraphEdge>, CircuitPlacementError> {
    let enabled = instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.enabled))
        .collect::<BTreeMap<_, _>>();
    let outgoing = edges
        .iter()
        .filter(|edge| edge.connection.is_forward())
        .fold(
            BTreeMap::<&str, Vec<&StreamCircuitGraphEdge>>::new(),
            |mut map, edge| {
                map.entry(edge.source.component_id.as_str())
                    .or_default()
                    .push(edge);
                map
            },
        );
    let mut effective = edges
        .iter()
        .filter(|edge| {
            !edge.connection.is_forward()
                && enabled
                    .get(edge.source.component_id.as_str())
                    .copied()
                    .unwrap_or(false)
                && enabled
                    .get(edge.destination.component_id.as_str())
                    .copied()
                    .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    for edge in edges.iter().filter(|edge| edge.connection.is_forward()) {
        if !enabled
            .get(edge.source.component_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let mut destination = edge.destination.clone();
        let mut visited = BTreeSet::new();
        while !enabled
            .get(destination.component_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            if !visited.insert(destination.component_id.clone()) {
                return Err(CircuitPlacementError(format!(
                    "runtime graph bypass path contains a cycle at {}",
                    destination.component_id
                )));
            }
            let next = outgoing
                .get(destination.component_id.as_str())
                .and_then(|candidates| candidates.first())
                .copied();
            let Some(next) = next else {
                break;
            };
            destination = next.destination.clone();
        }
        if enabled
            .get(destination.component_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            effective.push(StreamCircuitGraphEdge {
                id: edge.id.clone(),
                source: edge.source.clone(),
                destination,
                connection: StreamCircuitConnection::Forward,
            });
        }
    }
    Ok(effective)
}

fn topological_runtime_graph_order(
    instances: &[StreamCircuitNodeInstance],
    edges: &[StreamCircuitGraphEdge],
) -> Result<Vec<String>, CircuitPlacementError> {
    let enabled_ids = instances
        .iter()
        .filter(|instance| instance.enabled)
        .map(|instance| instance.instance_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut indegree = enabled_ids
        .iter()
        .map(|id| (*id, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<&str, Vec<&str>>::new();
    for edge in edges.iter().filter(|edge| edge.connection.is_forward()) {
        *indegree
            .get_mut(edge.destination.component_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "effective edge {} has a disabled destination",
                    edge.id
                ))
            })? += 1;
        outgoing
            .entry(edge.source.component_id.as_str())
            .or_default()
            .push(edge.destination.component_id.as_str());
    }
    let mut remaining = enabled_ids;
    let mut ordered = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let ready = instances
            .iter()
            .filter(|instance| instance.enabled)
            .map(|instance| instance.instance_id.as_str())
            .find(|id| remaining.contains(id) && indegree[id] == 0)
            .ok_or_else(|| {
                CircuitPlacementError(
                    "runtime graph contains an instantaneous cycle; use a temporal feedback connection with a positive delay"
                        .to_string(),
                )
            })?;
        remaining.remove(ready);
        ordered.push(ready.to_string());
        for destination in outgoing.get(ready).into_iter().flatten() {
            let value = indegree
                .get_mut(destination)
                .expect("validated edge destination must have indegree");
            *value -= 1;
        }
    }
    Ok(ordered)
}

fn validate_runtime_graph_source_graph(
    graph: &ResolvedLoweredExecutionGraph,
) -> Result<(), CircuitPlacementError> {
    if graph.circuits.is_empty() {
        return Err(CircuitPlacementError(
            "cannot create runtime graph for an empty execution_graph".to_string(),
        ));
    }
    Ok(())
}

fn validate_placement_spec_against_graph(
    graph: &ResolvedLoweredExecutionGraph,
    spec: &StreamCircuitPlacementSpec,
) -> Result<(), CircuitPlacementError> {
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
    let component_ids = graph
        .circuits
        .iter()
        .map(|artifact| artifact.component.id.as_str())
        .collect::<BTreeSet<_>>();
    for component_id in spec.node_devices.keys() {
        if !component_ids.contains(component_id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "placement references unknown component {component_id:?}"
            )));
        }
    }
    Ok(())
}
