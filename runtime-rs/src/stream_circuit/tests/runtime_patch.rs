    #[test]
    fn placement_plan_rejects_unknown_pedal_overrides() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0").with_pedal_device("layer_99", "gpu1");

        let error = resolved.placement_plan(&spec).unwrap_err();

        assert!(error.0.contains("unknown pedal"));
        assert!(error.0.contains("layer_99"));
    }

    #[test]
    fn runtime_patch_defaults_to_source_series_with_device_overrides() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_07", "lan:worker-a");

        let patch = resolved.runtime_patch_from_placement(&spec).unwrap();

        assert_eq!(patch.schema, STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA);
        assert_eq!(patch.wiring, "explicit_graph");
        assert_eq!(patch.instances.len(), resolved.circuits.len());
        assert_eq!(patch.instances[0].instance_id, "input_transducer");
        assert_eq!(patch.instances[0].source_pedal_id, "input_transducer");
        assert_eq!(patch.instances[0].device_id, "gpu0");
        assert_eq!(
            patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_00")
                .unwrap()
                .device_id,
            "gpu0"
        );
        assert_eq!(
            patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_02")
                .unwrap()
                .device_id,
            "gpu1"
        );
        assert_eq!(
            patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_07")
                .unwrap()
                .device_id,
            "lan:worker-a"
        );
        assert_eq!(patch.placement_spec(), spec);
    }

    #[test]
    fn runtime_patch_can_duplicate_a_layer_as_a_new_pedal_instance() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = resolved
            .default_runtime_patch("gpu0")
            .unwrap()
            .duplicate_after_instance(&resolved, "layer_05", "layer_05_repeat")
            .unwrap()
            .with_instance_device("layer_05_repeat", "gpu1")
            .unwrap();

        let instantiated = resolved.instantiate_runtime_patch(&patch).unwrap();
        let placement = instantiated
            .placement_plan(&patch.placement_spec())
            .unwrap();

        assert_eq!(instantiated.circuits.len(), resolved.circuits.len() + 1);
        assert_eq!(
            instantiated.index.summary.circuit_count,
            resolved.circuits.len() + 1
        );
        let original_index = instantiated
            .circuits
            .iter()
            .position(|artifact| artifact.pedal.id == "layer_05")
            .unwrap();
        let duplicate_index = original_index + 1;
        let duplicate = &instantiated.circuits[duplicate_index];
        let source = resolved
            .circuits
            .iter()
            .find(|artifact| artifact.pedal.id == "layer_05")
            .unwrap();

        assert_eq!(duplicate.pedal.id, "layer_05_repeat");
        assert_eq!(duplicate.circuit.source.pedal_id, "layer_05");
        assert_eq!(
            duplicate.params.refs.keys().collect::<Vec<_>>(),
            source.params.refs.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            duplicate.circuit.state_ports,
            instantiated.circuits[original_index].circuit.state_ports
        );
        assert_eq!(
            placement.pedal("layer_05_repeat").unwrap().device_id,
            "gpu1"
        );
        let incoming = placement
            .cables
            .iter()
            .find(|cable| cable.destination_pedal_id == "layer_05_repeat")
            .unwrap();
        let outgoing = placement
            .cables
            .iter()
            .find(|cable| cable.source_pedal_id == "layer_05_repeat")
            .unwrap();
        assert_eq!(incoming.source_pedal_id, "layer_05");
        assert_eq!(outgoing.destination_pedal_id, "layer_06");
        assert_eq!(placement.cross_device_cable_count, 2);
    }

    #[test]
    fn runtime_patch_can_use_an_explicit_source_chain() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let chain = vec![
            ("layer_00".to_string(), "layer_00".to_string()),
            ("layer_01".to_string(), "layer_01".to_string()),
            ("layer_05".to_string(), "layer_05".to_string()),
            ("layer_05_repeat".to_string(), "layer_05".to_string()),
            ("layer_06".to_string(), "layer_06".to_string()),
            ("layer_13".to_string(), "layer_13".to_string()),
        ];
        let patch = StreamCircuitRuntimePatch::from_source_chain(&resolved, "gpu0", &chain)
            .unwrap()
            .with_instance_device("layer_05_repeat", "gpu1")
            .unwrap();

        assert_eq!(
            patch
                .instances
                .iter()
                .map(|instance| (
                    instance.instance_id.as_str(),
                    instance.source_pedal_id.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("layer_00", "layer_00"),
                ("layer_01", "layer_01"),
                ("layer_05", "layer_05"),
                ("layer_05_repeat", "layer_05"),
                ("layer_06", "layer_06"),
                ("layer_13", "layer_13"),
            ]
        );

        let instantiated = resolved.instantiate_runtime_patch(&patch).unwrap();
        let placement = instantiated
            .placement_plan(&patch.placement_spec())
            .unwrap();

        assert_eq!(instantiated.circuits.len(), chain.len());
        assert!(
            instantiated
                .circuits
                .iter()
                .all(|artifact| artifact.pedal.id != "layer_02")
        );
        assert_eq!(
            placement
                .pedals
                .iter()
                .map(|pedal| pedal.pedal_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "layer_00",
                "layer_01",
                "layer_05",
                "layer_05_repeat",
                "layer_06",
                "layer_13",
            ]
        );
        assert_eq!(
            placement.pedal("layer_05_repeat").unwrap().device_id,
            "gpu1"
        );
        assert_eq!(placement.cross_device_cable_count, 2);
    }

    #[test]
    fn processor_chain_edit_preserves_generation_pedals_and_feedback() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let original = resolved.default_runtime_patch("gpu0").unwrap();
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("repeat".to_string(), "layer_00".to_string()),
            ("last".to_string(), "layer_13".to_string()),
        ];

        let patch = original
            .with_signal_processor_chain(&resolved, &chain)
            .unwrap();

        assert_eq!(patch.instances.len(), 6);
        for system_pedal in ["input_transducer", "output_transducer", "sampler"] {
            assert!(
                patch
                    .instances
                    .iter()
                    .any(|instance| instance.instance_id == system_pedal)
            );
        }
        assert_eq!(patch.boundary, resolved.index.graph.boundary);
        assert!(patch.cables.iter().any(|cable| {
            cable.source.pedal_id == "sampler"
                && cable.destination.pedal_id == "input_transducer"
                && cable.connection
                    == StreamCircuitConnection::TemporalFeedback {
                        delay_activations: 1,
                    }
        }));
        assert!(patch.cables.iter().any(|cable| {
            cable.source.pedal_id == "input_transducer" && cable.destination.pedal_id == "first"
        }));
        assert!(patch.cables.iter().any(|cable| {
            cable.source.pedal_id == "last" && cable.destination.pedal_id == "output_transducer"
        }));
        patch.validate_against_graph(&resolved).unwrap();
    }

    #[test]
    fn runtime_series_wiring_rejects_ambiguous_multiport_pedals() {
        let mut resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let source = resolved
            .circuits
            .iter_mut()
            .find(|artifact| artifact.pedal.id == "layer_00")
            .unwrap();
        let mut auxiliary = source.circuit.boundary.outputs[0].clone();
        auxiliary.id = "auxiliary_output".to_string();
        source.circuit.boundary.outputs.push(auxiliary);
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("second".to_string(), "layer_01".to_string()),
        ];

        let error =
            StreamCircuitRuntimePatch::from_source_chain(&resolved, "gpu0", &chain).unwrap_err();

        assert!(error.0.contains("exactly one output port"));
    }

    #[test]
    fn runtime_patch_rejects_control_values_that_execution_would_ignore() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let mut patch = resolved.default_runtime_patch("gpu0").unwrap();
        patch.instances[0]
            .control_values
            .insert("unused".to_string(), serde_json::json!(true));

        let error = patch.validate_against_graph(&resolved).unwrap_err();

        assert!(
            error
                .0
                .contains("executable pedal controls are not implemented")
        );
    }

    #[test]
    fn runtime_patch_execution_order_comes_from_cables_not_instance_storage_order() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("second".to_string(), "layer_01".to_string()),
            ("third".to_string(), "layer_02".to_string()),
        ];
        let mut patch =
            StreamCircuitRuntimePatch::from_source_chain(&resolved, "gpu0", &chain).unwrap();
        patch.instances.reverse();

        let instantiated = patch.instantiate_graph(&resolved).unwrap();

        assert_eq!(
            instantiated
                .circuits
                .iter()
                .map(|artifact| artifact.pedal.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn runtime_patch_validates_state_policy_targets_compatibility_and_cycles() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let mut patch = resolved
            .default_runtime_patch("gpu0")
            .unwrap()
            .duplicate_after_instance(&resolved, "layer_05", "layer_05_repeat")
            .unwrap();
        patch
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05_repeat")
            .unwrap()
            .state_policy = StreamCircuitPedalInstanceStatePolicy::ShareWith {
            instance_id: "layer_05".to_string(),
        };
        patch.validate_against_graph(&resolved).unwrap();

        let mut cross_device_share = patch.clone();
        cross_device_share
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05_repeat")
            .unwrap()
            .device_id = "gpu1".to_string();
        assert!(
            cross_device_share
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("cannot share state across devices")
        );

        let mut cycle = patch;
        cycle
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05")
            .unwrap()
            .state_policy = StreamCircuitPedalInstanceStatePolicy::CloneFrom {
            instance_id: "layer_05_repeat".to_string(),
        };
        assert!(
            cycle
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("dependency cycle")
        );
    }

    #[test]
    fn runtime_patch_rejects_unrouted_ports_and_multiple_writers_to_one_input() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = resolved.default_runtime_patch("gpu0").unwrap();

        let mut disconnected = patch.clone();
        disconnected.cables.remove(4);
        assert!(
            disconnected
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("unrouted ports")
        );

        let mut multiple_writers = patch;
        let mut duplicate = multiple_writers.cables[0].clone();
        duplicate.id = "second_writer".to_string();
        duplicate.destination = multiple_writers.cables[1].destination.clone();
        multiple_writers.cables.push(duplicate);
        assert!(
            multiple_writers
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("more than one forward cable")
        );
    }

    #[test]
    fn runtime_patch_supports_fanout_to_distinct_inputs() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let mut patch = resolved.default_runtime_patch("gpu0").unwrap();
        patch.instances.push(StreamCircuitPedalInstance {
            instance_id: "branch".to_string(),
            source_pedal_id: "layer_01".to_string(),
            device_id: "gpu0".to_string(),
            enabled: true,
            control_values: BTreeMap::new(),
            state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
        });
        patch.cables.push(StreamCircuitGraphCable {
            id: "layer_00_to_branch".to_string(),
            source: StreamCircuitCableEndpoint {
                pedal_id: "layer_00".to_string(),
                port_id: "output_frame".to_string(),
            },
            destination: StreamCircuitCableEndpoint {
                pedal_id: "branch".to_string(),
                port_id: "input_frame".to_string(),
            },
            connection: StreamCircuitConnection::Forward,
        });
        patch
            .boundary
            .public_outputs
            .push(StreamCircuitGraphBoundaryPort {
                id: "branch_output".to_string(),
                endpoint: StreamCircuitCableEndpoint {
                    pedal_id: "branch".to_string(),
                    port_id: "output_frame".to_string(),
                },
            });

        patch.validate_against_graph(&resolved).unwrap();
        assert_eq!(
            patch
                .effective_cables()
                .unwrap()
                .iter()
                .filter(|cable| cable.source.pedal_id == "layer_00")
                .count(),
            2
        );
    }

    #[test]
    fn runtime_patch_accepts_delayed_feedback_and_rejects_instantaneous_cycles() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = resolved.default_runtime_patch("gpu0").unwrap();
        let feedback_index = patch
            .cables
            .iter()
            .position(|cable| !cable.connection.is_forward())
            .unwrap();
        patch.validate_against_graph(&resolved).unwrap();
        assert_eq!(
            patch.topological_instance_ids(&resolved).unwrap().len(),
            resolved.circuits.len()
        );

        let mut instantaneous = patch.clone();
        instantaneous.cables[feedback_index].connection = StreamCircuitConnection::Forward;
        assert!(
            instantaneous
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("instantaneous cycle")
        );

        let mut zero_delay = patch;
        zero_delay.cables[feedback_index].connection = StreamCircuitConnection::TemporalFeedback {
            delay_activations: 0,
        };
        assert!(
            zero_delay
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("must delay at least one activation")
        );
    }
