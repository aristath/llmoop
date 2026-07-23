fn validate_state_policy_dependencies(
    instances: &[StreamCircuitPedalInstance],
) -> Result<(), CircuitPlacementError> {
    let dependencies = instances
        .iter()
        .filter_map(|instance| {
            let source = match &instance.state_policy {
                StreamCircuitPedalInstanceStatePolicy::Fresh => return None,
                StreamCircuitPedalInstanceStatePolicy::CloneFrom { instance_id }
                | StreamCircuitPedalInstanceStatePolicy::ShareWith { instance_id } => instance_id,
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
                    "runtime patch state policies contain a dependency cycle at {current}"
                )));
            }
            current = next;
        }
    }
    Ok(())
}

fn validate_explicit_cables(
    patch: &StreamCircuitRuntimePatch,
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
) -> Result<(), CircuitPlacementError> {
    let instances = patch
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut forward_destination_ports = BTreeSet::new();
    let mut feedback_destination_ports = BTreeSet::new();
    let mut incoming_count = BTreeMap::<&str, usize>::new();
    let mut outgoing_count = BTreeMap::<&str, usize>::new();
    for cable in &patch.cables {
        if cable.id.is_empty() || !ids.insert(cable.id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "runtime patch contains an empty or duplicate cable id {:?}",
                cable.id
            )));
        }
        let source_instance = instances
            .get(cable.source.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch cable {} references unknown source instance {}",
                    cable.id, cable.source.pedal_id
                ))
            })?;
        let destination_instance = instances
            .get(cable.destination.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch cable {} references unknown destination instance {}",
                    cable.id, cable.destination.pedal_id
                ))
            })?;
        cable.connection.validate(&cable.id)?;
        if source_instance.instance_id == destination_instance.instance_id
            && cable.connection.is_forward()
        {
            return Err(CircuitPlacementError(format!(
                "runtime patch cable {} creates an un-delayed self-loop on {}",
                cable.id, source_instance.instance_id
            )));
        }
        let destination_ports = if cable.connection.is_forward() {
            &mut forward_destination_ports
        } else {
            &mut feedback_destination_ports
        };
        if !destination_ports.insert((
            destination_instance.instance_id.as_str(),
            cable.destination.port_id.as_str(),
        )) {
            return Err(CircuitPlacementError(format!(
                "runtime patch input {}.{} has more than one {} cable",
                destination_instance.instance_id,
                cable.destination.port_id,
                if cable.connection.is_forward() {
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
        validate_graph_cable_contract(
            cable,
            source_instance,
            source_by_id[source_instance.source_pedal_id.as_str()],
            destination_instance,
            source_by_id[destination_instance.source_pedal_id.as_str()],
        )?;
    }
    for instance in patch.instances.iter().filter(|instance| !instance.enabled) {
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
                "disabled branching or joining pedal {} cannot be bypassed automatically",
                instance.instance_id
            )));
        }
    }
    let effective = patch.effective_cables()?;
    let enabled = patch
        .instances
        .iter()
        .filter(|instance| instance.enabled)
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    for cable in &effective {
        let source_instance = enabled[cable.source.pedal_id.as_str()];
        let destination_instance = enabled[cable.destination.pedal_id.as_str()];
        validate_graph_cable_contract(
            cable,
            source_instance,
            source_by_id[source_instance.source_pedal_id.as_str()],
            destination_instance,
            source_by_id[destination_instance.source_pedal_id.as_str()],
        )?;
    }
    if patch.boundary.external_inputs.is_empty() {
        return Err(CircuitPlacementError(
            "runtime patch must declare at least one external input".to_string(),
        ));
    }
    if patch.boundary.public_outputs.is_empty() {
        return Err(CircuitPlacementError(
            "runtime patch must declare at least one public output".to_string(),
        ));
    }
    let external_inputs = validate_runtime_boundary_ports(
        "external input",
        &patch.boundary.external_inputs,
        &enabled,
        source_by_id,
        true,
    )?;
    let public_outputs = validate_runtime_boundary_ports(
        "public output",
        &patch.boundary.public_outputs,
        &enabled,
        source_by_id,
        false,
    )?;
    let connected_outputs = effective
        .iter()
        .map(|cable| {
            (
                cable.source.pedal_id.as_str(),
                cable.source.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let connected_inputs = effective
        .iter()
        .map(|cable| {
            (
                cable.destination.pedal_id.as_str(),
                cable.destination.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut unrouted_inputs = Vec::new();
    let mut unrouted_outputs = Vec::new();
    for instance in enabled.values() {
        let artifact = source_by_id[instance.source_pedal_id.as_str()];
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
            "runtime patch has unrouted ports; inputs={unrouted_inputs:?}, outputs={unrouted_outputs:?}"
        )));
    }
    Ok(())
}

fn validate_runtime_boundary_ports<'a>(
    kind: &str,
    ports: &'a [StreamCircuitGraphBoundaryPort],
    enabled: &BTreeMap<&'a str, &'a StreamCircuitPedalInstance>,
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
    input: bool,
) -> Result<BTreeSet<(&'a str, &'a str)>, CircuitPlacementError> {
    let mut ids = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for port in ports {
        if port.id.is_empty() || !ids.insert(port.id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "runtime patch contains an empty or duplicate {kind} id {:?}",
                port.id
            )));
        }
        let instance = enabled
            .get(port.endpoint.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch {kind} {} references missing or disabled instance {}",
                    port.id, port.endpoint.pedal_id
                ))
            })?;
        let artifact = source_by_id[instance.source_pedal_id.as_str()];
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
                "runtime patch {kind} {} references unknown {} port {}.{}",
                port.id,
                if input { "input" } else { "output" },
                port.endpoint.pedal_id,
                port.endpoint.port_id
            )));
        }
        if !endpoints.insert((
            port.endpoint.pedal_id.as_str(),
            port.endpoint.port_id.as_str(),
        )) {
            return Err(CircuitPlacementError(format!(
                "runtime patch declares {}.{} as more than one {kind}",
                port.endpoint.pedal_id, port.endpoint.port_id
            )));
        }
    }
    Ok(endpoints)
}

fn validate_graph_cable_contract(
    cable: &StreamCircuitGraphCable,
    source_instance: &StreamCircuitPedalInstance,
    source: &ResolvedCircuitArtifact,
    destination_instance: &StreamCircuitPedalInstance,
    destination: &ResolvedCircuitArtifact,
) -> Result<(), CircuitPlacementError> {
    let output = source
        .circuit
        .boundary
        .outputs
        .iter()
        .find(|port| port.id == cable.source.port_id)
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime patch cable {} references unknown output {}.{}",
                cable.id, source_instance.instance_id, cable.source.port_id
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
                "runtime patch cable {} references unknown input {}.{}",
                cable.id, destination_instance.instance_id, cable.destination.port_id
            ))
        })?;
    if output.signal != input.signal || output.shape != input.shape {
        return Err(CircuitPlacementError(format!(
            "cannot patch cable {} ({}.{} -> {}.{}) without an adapter: output {:?}/{:?}, input {:?}/{:?}",
            cable.id,
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
