fn source_components(manifest: &VulkanResidentModelPackageManifest) -> Vec<RuntimeEditorSourceComponent> {
    let execution_by_component = manifest
        .component_executions
        .iter()
        .map(|execution| (execution.component_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();
    manifest
        .circuit_graph
        .components
        .iter()
        .map(|component| RuntimeEditorSourceComponent {
            source_id: component.component_id.clone(),
            layer_index: component.circuit.source.source_layer_index,
            operator_type: component.operator_type.clone(),
            runtime_role: component.circuit.runtime_role,
            implementation: component.implementation.clone(),
            behavioral_role: component.behavioral_role.clone(),
            input_shape: component
                .circuit
                .boundary
                .inputs
                .first()
                .map(|port| port.shape.clone())
                .unwrap_or_default(),
            output_shape: component
                .circuit
                .boundary
                .outputs
                .first()
                .map(|port| port.shape.clone())
                .unwrap_or_default(),
            state_ports: component
                .circuit
                .state_ports
                .iter()
                .filter_map(|state| serde_json::to_value(state).ok())
                .collect(),
            controls: component.circuit.boundary.controls.clone(),
            control_schemas: component
                .circuit
                .boundary
                .controls
                .iter()
                .enumerate()
                .map(|(index, control)| runtime_editor_control_schema(index, control))
                .collect(),
            parameter_ref_count: component.params.refs.len(),
            node_count: component.circuit.nodes.len(),
            kernel_count: match component.runtime_role {
                CircuitRuntimeRole::SignalProcessor => execution_by_component
                    .get(component.component_id.as_str())
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
