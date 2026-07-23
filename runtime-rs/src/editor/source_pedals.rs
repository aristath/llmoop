fn source_pedals(manifest: &VulkanResidentModelPackageManifest) -> Vec<RuntimeEditorSourcePedal> {
    let execution_by_pedal = manifest
        .pedal_executions
        .iter()
        .map(|execution| (execution.pedal_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();
    manifest
        .circuit_graph
        .pedals
        .iter()
        .map(|pedal| RuntimeEditorSourcePedal {
            source_id: pedal.pedal_id.clone(),
            layer_index: pedal.circuit.source.source_layer_index,
            operator_type: pedal.operator_type.clone(),
            runtime_role: pedal.circuit.runtime_role,
            implementation: pedal.implementation.clone(),
            behavioral_role: pedal.behavioral_role.clone(),
            input_shape: pedal
                .circuit
                .boundary
                .inputs
                .first()
                .map(|port| port.shape.clone())
                .unwrap_or_default(),
            output_shape: pedal
                .circuit
                .boundary
                .outputs
                .first()
                .map(|port| port.shape.clone())
                .unwrap_or_default(),
            state_ports: pedal
                .circuit
                .state_ports
                .iter()
                .filter_map(|state| serde_json::to_value(state).ok())
                .collect(),
            controls: pedal.circuit.boundary.controls.clone(),
            control_schemas: pedal
                .circuit
                .boundary
                .controls
                .iter()
                .enumerate()
                .map(|(index, control)| runtime_editor_control_schema(index, control))
                .collect(),
            parameter_ref_count: pedal.params.refs.len(),
            node_count: pedal.circuit.nodes.len(),
            kernel_count: match pedal.runtime_role {
                CircuitRuntimeRole::SignalProcessor => execution_by_pedal
                    .get(pedal.pedal_id.as_str())
                    .map(|execution| execution.kernels.len())
                    .unwrap_or(0),
                CircuitRuntimeRole::InputTransducer => 1,
                CircuitRuntimeRole::OutputTransducer => 2,
                CircuitRuntimeRole::Sampler => manifest.sampler.kernels.len(),
                CircuitRuntimeRole::DraftProcessor
                | CircuitRuntimeRole::DraftInputAdapter
                | CircuitRuntimeRole::DraftOutputTransducer => 0,
            },
        })
        .collect()
}
