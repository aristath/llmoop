    #[test]
    fn circuit_contract_rejects_ambiguous_boundary_ports() {
        let circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "fixture_circuit",
            "source": {
                "component_id": "fixture_component",
                "source_layer_index": null,
                "source_operator_type": "fixture"
            },
            "runtime_role": "input_transducer",
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
                "refs": {}
            },
            "nodes": [{
                "id": "identity",
                "op": "identity",
                "inputs": ["input_frame"],
                "outputs": ["output_frame"]
            }]
        }))
        .unwrap();

        assert_eq!(circuit.source.source_layer_index, None);
        assert_eq!(circuit.runtime_role, CircuitRuntimeRole::InputTransducer);

        let mut duplicate_input = circuit.clone();
        duplicate_input
            .boundary
            .inputs
            .push(duplicate_input.boundary.inputs[0].clone());
        let input_error = duplicate_input.validate_contract().unwrap_err();
        assert!(
            input_error
                .to_string()
                .contains("duplicate boundary input port id")
        );

        let mut duplicate_output = circuit.clone();
        duplicate_output
            .boundary
            .outputs
            .push(duplicate_output.boundary.outputs[0].clone());
        let output_error = duplicate_output.validate_contract().unwrap_err();
        assert!(
            output_error
                .to_string()
                .contains("duplicate boundary output port id")
        );

        let mut malformed = circuit.clone();
        malformed.boundary.inputs[0].shape.clear();
        malformed.boundary.outputs[0].component_port = Some(String::new());
        let malformed_error = malformed.validate_contract().unwrap_err().to_string();
        assert!(malformed_error.contains("shape must contain positive dimensions"));
        assert!(malformed_error.contains("must map to a non-empty component_port"));
    }

    #[test]
    fn circuit_contract_requires_exact_semantic_module_ownership() {
        let circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "semantic_fixture",
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
            "state_ports": [{
                "id": "memory",
                "type": "rolling_memory",
                "shape": [2, 8],
                "update": "replace"
            }],
            "parameters": {
                "layout": "row_major",
                "storage": "safetensors",
                "refs": {"weight": {"tensor": "layer.weight"}}
            },
            "semantic_module_tree": {
                "schema": SEMANTIC_MODULE_TREE_SCHEMA,
                "root_module_id": "layer",
                "modules": [{
                    "id": "layer",
                    "role": "layer",
                    "responsibility": "Editable layer",
                    "parent_id": null,
                    "child_ids": ["layer.token_mixer"],
                    "source_node_ids": [],
                    "parameter_ref_ids": [],
                    "owned_state_port_ids": [],
                    "input_signals": ["input_frame"],
                    "output_signals": ["output_frame"]
                }, {
                    "id": "layer.token_mixer",
                    "role": "token_mixer",
                    "responsibility": "Stateful projection",
                    "parent_id": "layer",
                    "child_ids": [],
                    "source_node_ids": ["project"],
                    "parameter_ref_ids": ["weight"],
                    "owned_state_port_ids": ["memory"],
                    "input_signals": ["input_frame"],
                    "output_signals": ["output_frame"]
                }]
            },
            "nodes": [{
                "id": "project",
                "op": "linear",
                "inputs": ["input_frame", "memory"],
                "outputs": ["output_frame"],
                "params": ["weight"],
                "state_reads": ["memory"],
                "state_writes": ["memory"]
            }]
        }))
        .unwrap();

        circuit.validate_contract().unwrap();

        let mut duplicate_node = circuit.clone();
        duplicate_node
            .semantic_module_tree
            .as_mut()
            .unwrap()
            .modules[0]
            .source_node_ids
            .push("project".to_string());
        assert!(
            duplicate_node
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("belongs to semantic modules")
        );

        let mut missing_state = circuit;
        missing_state
            .semantic_module_tree
            .as_mut()
            .unwrap()
            .modules[1]
            .owned_state_port_ids
            .clear();
        assert!(
            missing_state
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("does not own every state port exactly once")
        );
    }
