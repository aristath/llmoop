    #[test]
    fn circuit_contract_rejects_ambiguous_boundary_ports() {
        let circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "fixture_circuit",
            "source": {
                "pedal_id": "fixture_pedal",
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
                    "pedal_port": "input"
                }],
                "outputs": [{
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                    "pedal_port": "output"
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
        malformed.boundary.outputs[0].pedal_port = Some(String::new());
        let malformed_error = malformed.validate_contract().unwrap_err().to_string();
        assert!(malformed_error.contains("shape must contain positive dimensions"));
        assert!(malformed_error.contains("must map to a non-empty pedal_port"));
    }

