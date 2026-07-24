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
        .map(|component| {
            let execution = execution_by_component
                .get(component.component_id.as_str())
                .copied();
            let (semantic_modules, semantic_module_root_id) =
                runtime_editor_semantic_modules(component, execution);
            RuntimeEditorSourceComponent {
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
            semantic_modules,
            semantic_module_root_id,
        }})
        .collect()
}

fn runtime_editor_semantic_modules(
    component: &crate::VulkanResidentPackageComponentCircuit,
    execution: Option<&crate::VulkanResidentComponentExecutionSpec>,
) -> (Vec<RuntimeEditorSemanticModule>, Option<String>) {
    let Some(tree) = &component.circuit.semantic_module_tree else {
        return (Vec::new(), None);
    };
    let module_by_id = tree
        .modules
        .iter()
        .map(|module| (module.id.as_str(), module))
        .collect::<BTreeMap<_, _>>();
    let mut optimized_by_module = BTreeMap::<String, BTreeSet<String>>::new();
    let mut kernels_by_module = BTreeMap::<String, BTreeSet<String>>::new();
    if let Some(execution) = execution {
        for kernel in &execution.kernels {
            for direct_module_id in &kernel.semantic_module_ids {
                let mut current = Some(direct_module_id.as_str());
                while let Some(module_id) = current {
                    optimized_by_module
                        .entry(module_id.to_string())
                        .or_default()
                        .insert(kernel.node_id.clone());
                    kernels_by_module
                        .entry(module_id.to_string())
                        .or_default()
                        .insert(kernel.node_id.clone());
                    current = module_by_id
                        .get(module_id)
                        .and_then(|module| module.parent_id.as_deref());
                }
            }
        }
    }
    (
        tree.modules
            .iter()
            .map(|module| RuntimeEditorSemanticModule {
                id: module.id.clone(),
                role: module.role.clone(),
                responsibility: module.responsibility.clone(),
                parent_id: module.parent_id.clone(),
                child_ids: module.child_ids.clone(),
                source_node_ids: module.source_node_ids.clone(),
                parameter_ref_ids: module.parameter_ref_ids.clone(),
                owned_state_port_ids: module.owned_state_port_ids.clone(),
                input_signals: module.input_signals.clone(),
                output_signals: module.output_signals.clone(),
                optimized_node_ids: optimized_by_module
                    .remove(&module.id)
                    .map(BTreeSet::into_iter)
                    .unwrap_or_default()
                    .collect(),
                kernel_node_ids: kernels_by_module
                    .remove(&module.id)
                    .map(BTreeSet::into_iter)
                    .unwrap_or_default()
                    .collect(),
                measured_cost: None,
            })
            .collect(),
        Some(tree.root_module_id.clone()),
    )
}

#[cfg(test)]
mod semantic_module_tests {
    use super::*;

    #[test]
    fn editor_modules_attribute_fused_kernels_to_leaf_and_parent_modules() {
        let component: crate::VulkanResidentPackageComponentCircuit =
            serde_json::from_value(serde_json::json!({
                "component_id": "layer_00",
                "operator_type": "conv",
                "runtime_role": "signal_processor",
                "implementation": "fixture",
                "behavioral_role": "fixture",
                "circuit": {
                    "schema": crate::STREAM_CIRCUIT_SCHEMA,
                    "id": "layer_00_circuit",
                    "source": {
                        "component_id": "layer_00",
                        "source_layer_index": 0,
                        "source_operator_type": "conv"
                    },
                    "runtime_role": "signal_processor",
                    "behavioral_role": "fixture",
                    "implementation": "fixture",
                    "boundary": {
                        "inputs": [{
                            "id": "input_frame",
                            "signal": "frame",
                            "shape": [8],
                            "component_port": "input"
                        }],
                        "outputs": [{
                            "id": "output_frame",
                            "signal": "frame",
                            "shape": [8],
                            "source": "output_frame",
                            "component_port": "output"
                        }]
                    },
                    "parameters": {
                        "layout": "row_major",
                        "storage": "safetensors",
                        "refs": {"weight": {"tensor": "layer.weight"}}
                    },
                    "semantic_module_tree": {
                        "schema": crate::SEMANTIC_MODULE_TREE_SCHEMA,
                        "root_module_id": "layer",
                        "modules": [{
                            "id": "layer",
                            "role": "layer",
                            "responsibility": "Editable layer",
                            "child_ids": ["layer.token_mixer"],
                            "source_node_ids": [],
                            "parameter_ref_ids": [],
                            "owned_state_port_ids": [],
                            "input_signals": ["input_frame"],
                            "output_signals": ["output_frame"]
                        }, {
                            "id": "layer.token_mixer",
                            "role": "token_mixer",
                            "responsibility": "Mix tokens",
                            "parent_id": "layer",
                            "child_ids": [],
                            "source_node_ids": ["project"],
                            "parameter_ref_ids": ["weight"],
                            "owned_state_port_ids": [],
                            "input_signals": ["input_frame"],
                            "output_signals": ["output_frame"]
                        }]
                    },
                    "semantic_execution_nodes": [{
                        "id": "project",
                        "op": "linear",
                        "inputs": ["input_frame"],
                        "outputs": ["output_frame"],
                        "params": ["weight"]
                    }],
                    "nodes": [{
                        "id": "fused_project",
                        "op": "linear",
                        "inputs": ["input_frame"],
                        "outputs": ["output_frame"],
                        "params": ["weight"],
                        "attrs": {"compiled_from": ["project"]}
                    }]
                },
                "params": {
                    "schema": crate::CIRCUIT_PARAMS_SCHEMA,
                    "circuit": "layer_00_circuit",
                    "layout": "row_major",
                    "storage": "safetensors",
                    "refs": {"weight": {"tensor": "layer.weight"}}
                },
                "state": {
                    "schema": crate::CIRCUIT_STATE_SCHEMA,
                    "circuit": "layer_00_circuit",
                    "state_ports": []
                }
            }))
            .unwrap();
        let execution: crate::VulkanResidentComponentExecutionSpec =
            serde_json::from_value(serde_json::json!({
                "component_id": "layer_00",
                "operator_type": "conv",
                "implementation": "fixture",
                "kernels": [{
                    "execution_index": 0,
                    "node_id": "fused_project",
                    "op": "linear",
                    "source_node_ids": ["project"],
                    "semantic_module_ids": ["layer.token_mixer"],
                    "execution_domain": "decode",
                    "shader_path": "shaders/fixture.spv",
                    "local_size_x": 64,
                    "workgroup_count_x": 1,
                    "batch_mode": "serial_lanes",
                    "batch_implementations": []
                }]
            }))
            .unwrap();

        let (modules, root) = runtime_editor_semantic_modules(&component, Some(&execution));

        assert_eq!(root.as_deref(), Some("layer"));
        let root = modules.iter().find(|module| module.id == "layer").unwrap();
        let mixer = modules
            .iter()
            .find(|module| module.id == "layer.token_mixer")
            .unwrap();
        assert_eq!(root.optimized_node_ids, vec!["fused_project"]);
        assert_eq!(mixer.optimized_node_ids, vec!["fused_project"]);
        assert_eq!(mixer.kernel_node_ids, vec!["fused_project"]);
    }
}
