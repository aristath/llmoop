fn effective_runtime_patch_cables(
    instances: &[StreamCircuitPedalInstance],
    cables: &[StreamCircuitGraphCable],
) -> Result<Vec<StreamCircuitGraphCable>, CircuitPlacementError> {
    let enabled = instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.enabled))
        .collect::<BTreeMap<_, _>>();
    let outgoing = cables
        .iter()
        .filter(|cable| cable.connection.is_forward())
        .fold(
            BTreeMap::<&str, Vec<&StreamCircuitGraphCable>>::new(),
            |mut map, cable| {
                map.entry(cable.source.pedal_id.as_str())
                    .or_default()
                    .push(cable);
                map
            },
        );
    let mut effective = cables
        .iter()
        .filter(|cable| {
            !cable.connection.is_forward()
                && enabled
                    .get(cable.source.pedal_id.as_str())
                    .copied()
                    .unwrap_or(false)
                && enabled
                    .get(cable.destination.pedal_id.as_str())
                    .copied()
                    .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    for cable in cables.iter().filter(|cable| cable.connection.is_forward()) {
        if !enabled
            .get(cable.source.pedal_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let mut destination = cable.destination.clone();
        let mut visited = BTreeSet::new();
        while !enabled
            .get(destination.pedal_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            if !visited.insert(destination.pedal_id.clone()) {
                return Err(CircuitPlacementError(format!(
                    "runtime patch bypass path contains a cycle at {}",
                    destination.pedal_id
                )));
            }
            let next = outgoing
                .get(destination.pedal_id.as_str())
                .and_then(|candidates| candidates.first())
                .copied();
            let Some(next) = next else {
                break;
            };
            destination = next.destination.clone();
        }
        if enabled
            .get(destination.pedal_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            effective.push(StreamCircuitGraphCable {
                id: cable.id.clone(),
                source: cable.source.clone(),
                destination,
                connection: StreamCircuitConnection::Forward,
            });
        }
    }
    Ok(effective)
}

fn topological_runtime_patch_order(
    instances: &[StreamCircuitPedalInstance],
    cables: &[StreamCircuitGraphCable],
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
    for cable in cables.iter().filter(|cable| cable.connection.is_forward()) {
        *indegree
            .get_mut(cable.destination.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "effective cable {} has a disabled destination",
                    cable.id
                ))
            })? += 1;
        outgoing
            .entry(cable.source.pedal_id.as_str())
            .or_default()
            .push(cable.destination.pedal_id.as_str());
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
                    "runtime patch graph contains an instantaneous cycle; use a temporal feedback connection with a positive delay"
                        .to_string(),
                )
            })?;
        remaining.remove(ready);
        ordered.push(ready.to_string());
        for destination in outgoing.get(ready).into_iter().flatten() {
            let value = indegree
                .get_mut(destination)
                .expect("validated cable destination must have indegree");
            *value -= 1;
        }
    }
    Ok(ordered)
}

fn validate_runtime_patch_source_graph(
    graph: &ResolvedLoweredPedalboard,
) -> Result<(), CircuitPlacementError> {
    if graph.circuits.is_empty() {
        return Err(CircuitPlacementError(
            "cannot create runtime patch for an empty pedalboard".to_string(),
        ));
    }
    Ok(())
}

fn validate_placement_spec_against_graph(
    graph: &ResolvedLoweredPedalboard,
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
    let pedal_ids = graph
        .circuits
        .iter()
        .map(|artifact| artifact.pedal.id.as_str())
        .collect::<BTreeSet<_>>();
    for pedal_id in spec.pedal_devices.keys() {
        if !pedal_ids.contains(pedal_id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "placement references unknown pedal {pedal_id:?}"
            )));
        }
    }
    Ok(())
}
