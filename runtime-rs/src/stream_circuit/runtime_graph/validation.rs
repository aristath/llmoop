fn validate_state_policy_dependencies(
    instances: &[StreamCircuitNodeInstance],
) -> Result<(), CircuitPlacementError> {
    let dependencies = instances
        .iter()
        .filter_map(|instance| {
            let source = match &instance.state_policy {
                StreamCircuitNodeInstanceStatePolicy::Fresh => return None,
                StreamCircuitNodeInstanceStatePolicy::CloneFrom { instance_id }
                | StreamCircuitNodeInstanceStatePolicy::ShareWith { instance_id } => instance_id,
            };
            Some((instance.instance_id.as_str(), source.as_str()))
        })
        .collect::<BTreeMap<_, _>>();
    for start in dependencies.keys() {
        let mut visited = BTreeSet::new();
        let mut current = *start;
        while let Some(next) = dependencies.get(current) {
            if !visited.insert(current) {
                return Err(CircuitPlacementError(format!(
                    "runtime graph state policies contain a dependency cycle at {current}"
                )));
            }
            current = next;
        }
    }
    Ok(())
}

fn validate_explicit_edges(
    runtime_graph: &StreamCircuitRuntimeGraph,
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
) -> Result<(), CircuitPlacementError> {
    let instances = runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut forward_destination_ports = BTreeSet::new();
    let mut feedback_destination_ports = BTreeSet::new();
    let mut incoming_count = BTreeMap::<&str, usize>::new();
    let mut outgoing_count = BTreeMap::<&str, usize>::new();
    for edge in &runtime_graph.edges {
        if edge.id.is_empty() || !ids.insert(edge.id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "runtime graph contains an empty or duplicate edge id {:?}",
                edge.id
            )));
        }
        let source_instance = instances
            .get(edge.source.component_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime graph edge {} references unknown source instance {}",
                    edge.id, edge.source.component_id
                ))
            })?;
        let destination_instance = instances
            .get(edge.destination.component_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime graph edge {} references unknown destination instance {}",
                    edge.id, edge.destination.component_id
                ))
            })?;
        edge.connection.validate(&edge.id)?;
        if source_instance.instance_id == destination_instance.instance_id
            && edge.connection.is_forward()
        {
            return Err(CircuitPlacementError(format!(
                "runtime graph edge {} creates an un-delayed self-loop on {}",
                edge.id, source_instance.instance_id
            )));
        }
        let destination_ports = if edge.connection.is_forward() {
            &mut forward_destination_ports
        } else {
            &mut feedback_destination_ports
        };
        if !destination_ports.insert((
            destination_instance.instance_id.as_str(),
            edge.destination.port_id.as_str(),
        )) {
            return Err(CircuitPlacementError(format!(
                "runtime graph input {}.{} has more than one {} edge",
                destination_instance.instance_id,
                edge.destination.port_id,
                if edge.connection.is_forward() {
                    "forward"
                } else {
                    "temporal feedback"
                }
            )));
        }
        *incoming_count
            .entry(destination_instance.instance_id.as_str())
            .or_default() += 1;
        *outgoing_count
            .entry(source_instance.instance_id.as_str())
            .or_default() += 1;
        validate_graph_edge_contract(
            edge,
            source_instance,
            source_by_id[source_instance.source_component_id.as_str()],
            destination_instance,
            source_by_id[destination_instance.source_component_id.as_str()],
        )?;
    }
    for instance in runtime_graph.instances.iter().filter(|instance| !instance.enabled) {
        if incoming_count
            .get(instance.instance_id.as_str())
            .copied()
            .unwrap_or(0)
            > 1
            || outgoing_count
                .get(instance.instance_id.as_str())
                .copied()
                .unwrap_or(0)
                > 1
        {
            return Err(CircuitPlacementError(format!(
                "disabled branching or joining component {} cannot be bypassed automatically",
                instance.instance_id
            )));
        }
    }
    let effective = runtime_graph.effective_edges()?;
    let enabled = runtime_graph
        .instances
        .iter()
        .filter(|instance| instance.enabled)
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    for edge in &effective {
        let source_instance = enabled[edge.source.component_id.as_str()];
        let destination_instance = enabled[edge.destination.component_id.as_str()];
        validate_graph_edge_contract(
            edge,
            source_instance,
            source_by_id[source_instance.source_component_id.as_str()],
            destination_instance,
            source_by_id[destination_instance.source_component_id.as_str()],
        )?;
    }
    if runtime_graph.boundary.external_inputs.is_empty() {
        return Err(CircuitPlacementError(
            "runtime graph must declare at least one external input".to_string(),
        ));
    }
    if runtime_graph.boundary.public_outputs.is_empty() {
        return Err(CircuitPlacementError(
            "runtime graph must declare at least one public output".to_string(),
        ));
    }
    let external_inputs = validate_runtime_boundary_ports(
        "external input",
        &runtime_graph.boundary.external_inputs,
        &enabled,
        source_by_id,
        true,
    )?;
    let public_outputs = validate_runtime_boundary_ports(
        "public output",
        &runtime_graph.boundary.public_outputs,
        &enabled,
        source_by_id,
        false,
    )?;
    let connected_outputs = effective
        .iter()
        .map(|edge| {
            (
                edge.source.component_id.as_str(),
                edge.source.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let connected_inputs = effective
        .iter()
        .map(|edge| {
            (
                edge.destination.component_id.as_str(),
                edge.destination.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut unrouted_inputs = Vec::new();
    let mut unrouted_outputs = Vec::new();
    for instance in enabled.values() {
        let artifact = source_by_id[instance.source_component_id.as_str()];
        for port in &artifact.circuit.boundary.inputs {
            let endpoint = (instance.instance_id.as_str(), port.id.as_str());
            if !connected_inputs.contains(&endpoint) && !external_inputs.contains(&endpoint) {
                unrouted_inputs.push(format!("{}.{}", instance.instance_id, port.id));
            }
        }
        for port in &artifact.circuit.boundary.outputs {
            let endpoint = (instance.instance_id.as_str(), port.id.as_str());
            if !connected_outputs.contains(&endpoint) && !public_outputs.contains(&endpoint) {
                unrouted_outputs.push(format!("{}.{}", instance.instance_id, port.id));
            }
        }
    }
    if !unrouted_inputs.is_empty() || !unrouted_outputs.is_empty() {
        return Err(CircuitPlacementError(format!(
            "runtime graph has unrouted ports; inputs={unrouted_inputs:?}, outputs={unrouted_outputs:?}"
        )));
    }
    Ok(())
}

fn validate_runtime_boundary_ports<'a>(
    kind: &str,
    ports: &'a [StreamCircuitGraphBoundaryPort],
    enabled: &BTreeMap<&'a str, &'a StreamCircuitNodeInstance>,
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
    input: bool,
) -> Result<BTreeSet<(&'a str, &'a str)>, CircuitPlacementError> {
    let mut ids = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for port in ports {
        if port.id.is_empty() || !ids.insert(port.id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "runtime graph contains an empty or duplicate {kind} id {:?}",
                port.id
            )));
        }
        let instance = enabled
            .get(port.endpoint.component_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime graph {kind} {} references missing or disabled instance {}",
                    port.id, port.endpoint.component_id
                ))
            })?;
        let artifact = source_by_id[instance.source_component_id.as_str()];
        let declared = if input {
            &artifact.circuit.boundary.inputs
        } else {
            &artifact.circuit.boundary.outputs
        };
        if !declared
            .iter()
            .any(|candidate| candidate.id == port.endpoint.port_id)
        {
            return Err(CircuitPlacementError(format!(
                "runtime graph {kind} {} references unknown {} port {}.{}",
                port.id,
                if input { "input" } else { "output" },
                port.endpoint.component_id,
                port.endpoint.port_id
            )));
        }
        if !endpoints.insert((
            port.endpoint.component_id.as_str(),
            port.endpoint.port_id.as_str(),
        )) {
            return Err(CircuitPlacementError(format!(
                "runtime graph declares {}.{} as more than one {kind}",
                port.endpoint.component_id, port.endpoint.port_id
            )));
        }
    }
    Ok(endpoints)
}

fn validate_graph_edge_contract(
    edge: &StreamCircuitGraphEdge,
    source_instance: &StreamCircuitNodeInstance,
    source: &ResolvedCircuitArtifact,
    destination_instance: &StreamCircuitNodeInstance,
    destination: &ResolvedCircuitArtifact,
) -> Result<(), CircuitPlacementError> {
    let output = source
        .circuit
        .boundary
        .outputs
        .iter()
        .find(|port| port.id == edge.source.port_id)
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime graph edge {} references unknown output {}.{}",
                edge.id, source_instance.instance_id, edge.source.port_id
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
                "runtime graph edge {} references unknown input {}.{}",
                edge.id, destination_instance.instance_id, edge.destination.port_id
            ))
        })?;
    if output.signal != input.signal || output.shape != input.shape {
        return Err(CircuitPlacementError(format!(
            "cannot connect edge {} ({}.{} -> {}.{}) without an adapter: output {:?}/{:?}, input {:?}/{:?}",
            edge.id,
            source_instance.instance_id,
            output.id,
            destination_instance.instance_id,
            input.id,
            output.signal,
            output.shape,
            input.signal,
            input.shape
        )));
    }
    Ok(())
}
