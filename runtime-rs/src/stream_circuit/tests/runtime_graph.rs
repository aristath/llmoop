    #[test]
    fn placement_plan_rejects_unknown_component_overrides() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_99", "gpu1");

        let error = resolved.placement_plan(&spec).unwrap_err();

        assert!(error.0.contains("unknown component"));
        assert!(error.0.contains("layer_99"));
    }

    #[test]
    fn runtime_graph_defaults_to_source_series_with_device_overrides() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_component_device("layer_02", "gpu1")
            .with_component_device("layer_07", "lan:worker-a");

        let runtime_graph = resolved.runtime_graph_from_placement(&spec).unwrap();

        assert_eq!(runtime_graph.schema, STREAM_CIRCUIT_RUNTIME_GRAPH_SCHEMA);
        assert_eq!(runtime_graph.topology, "explicit_graph");
        assert_eq!(runtime_graph.instances.len(), resolved.circuits.len());
        assert_eq!(runtime_graph.instances[0].instance_id, "input_transducer");
        assert_eq!(runtime_graph.instances[0].source_component_id, "input_transducer");
        assert_eq!(runtime_graph.instances[0].device_id, "gpu0");
        assert_eq!(
            runtime_graph
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_00")
                .unwrap()
                .device_id,
            "gpu0"
        );
        assert_eq!(
            runtime_graph
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_02")
                .unwrap()
                .device_id,
            "gpu1"
        );
        assert_eq!(
            runtime_graph
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_07")
                .unwrap()
                .device_id,
            "lan:worker-a"
        );
        assert_eq!(runtime_graph.placement_spec(), spec);
    }

    #[test]
    fn runtime_graph_can_duplicate_a_layer_as_a_new_node_instance() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let runtime_graph = resolved
            .default_runtime_graph("gpu0")
            .unwrap()
            .duplicate_after_instance(&resolved, "layer_05", "layer_05_repeat")
            .unwrap()
            .with_instance_device("layer_05_repeat", "gpu1")
            .unwrap();

        let instantiated = resolved.instantiate_runtime_graph(&runtime_graph).unwrap();
        let placement = instantiated
            .placement_plan(&runtime_graph.placement_spec())
            .unwrap();

        assert_eq!(instantiated.circuits.len(), resolved.circuits.len() + 1);
        assert_eq!(
            instantiated.index.summary.circuit_count,
            resolved.circuits.len() + 1
        );
        let original_index = instantiated
            .circuits
            .iter()
            .position(|artifact| artifact.component.id == "layer_05")
            .unwrap();
        let duplicate_index = original_index + 1;
        let duplicate = &instantiated.circuits[duplicate_index];
        let source = resolved
            .circuits
            .iter()
            .find(|artifact| artifact.component.id == "layer_05")
            .unwrap();

        assert_eq!(duplicate.component.id, "layer_05_repeat");
        assert_eq!(duplicate.circuit.source.component_id, "layer_05");
        assert_eq!(
            duplicate.params.refs.keys().collect::<Vec<_>>(),
            source.params.refs.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            duplicate.circuit.state_ports,
            instantiated.circuits[original_index].circuit.state_ports
        );
        assert_eq!(
            placement.component("layer_05_repeat").unwrap().device_id,
            "gpu1"
        );
        let incoming = placement
            .edges
            .iter()
            .find(|edge| edge.destination_component_id == "layer_05_repeat")
            .unwrap();
        let outgoing = placement
            .edges
            .iter()
            .find(|edge| edge.source_component_id == "layer_05_repeat")
            .unwrap();
        assert_eq!(incoming.source_component_id, "layer_05");
        assert_eq!(outgoing.destination_component_id, "layer_06");
        assert_eq!(placement.cross_device_edge_count, 2);
    }

    #[test]
    fn runtime_graph_can_use_an_explicit_source_chain() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let chain = vec![
            ("layer_00".to_string(), "layer_00".to_string()),
            ("layer_01".to_string(), "layer_01".to_string()),
            ("layer_05".to_string(), "layer_05".to_string()),
            ("layer_05_repeat".to_string(), "layer_05".to_string()),
            ("layer_06".to_string(), "layer_06".to_string()),
            ("layer_13".to_string(), "layer_13".to_string()),
        ];
        let runtime_graph = StreamCircuitRuntimeGraph::from_source_chain(&resolved, "gpu0", &chain)
            .unwrap()
            .with_instance_device("layer_05_repeat", "gpu1")
            .unwrap();

        assert_eq!(
            runtime_graph
                .instances
                .iter()
                .map(|instance| (
                    instance.instance_id.as_str(),
                    instance.source_component_id.as_str()
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

        let instantiated = resolved.instantiate_runtime_graph(&runtime_graph).unwrap();
        let placement = instantiated
            .placement_plan(&runtime_graph.placement_spec())
            .unwrap();

        assert_eq!(instantiated.circuits.len(), chain.len());
        assert!(
            instantiated
                .circuits
                .iter()
                .all(|artifact| artifact.component.id != "layer_02")
        );
        assert_eq!(
            placement
                .components
                .iter()
                .map(|component| component.component_id.as_str())
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
            placement.component("layer_05_repeat").unwrap().device_id,
            "gpu1"
        );
        assert_eq!(placement.cross_device_edge_count, 2);
    }

    #[test]
    fn processor_chain_edit_preserves_generation_components_and_feedback() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let original = resolved.default_runtime_graph("gpu0").unwrap();
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("repeat".to_string(), "layer_00".to_string()),
            ("last".to_string(), "layer_13".to_string()),
        ];

        let runtime_graph = original
            .with_signal_processor_chain(&resolved, &chain)
            .unwrap();

        assert_eq!(runtime_graph.instances.len(), 6);
        for system_component in ["input_transducer", "output_transducer", "sampler"] {
            assert!(
                runtime_graph
                    .instances
                    .iter()
                    .any(|instance| instance.instance_id == system_component)
            );
        }
        assert_eq!(runtime_graph.boundary, resolved.index.graph.boundary);
        assert!(runtime_graph.edges.iter().any(|edge| {
            edge.source.component_id == "sampler"
                && edge.destination.component_id == "input_transducer"
                && edge.connection
                    == StreamCircuitConnection::TemporalFeedback {
                        delay_activations: 1,
                    }
        }));
        assert!(runtime_graph.edges.iter().any(|edge| {
            edge.source.component_id == "input_transducer" && edge.destination.component_id == "first"
        }));
        assert!(runtime_graph.edges.iter().any(|edge| {
            edge.source.component_id == "last" && edge.destination.component_id == "output_transducer"
        }));
        runtime_graph.validate_against_graph(&resolved).unwrap();
    }

    #[test]
    fn runtime_series_topology_rejects_ambiguous_multiport_components() {
        let mut resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let source = resolved
            .circuits
            .iter_mut()
            .find(|artifact| artifact.component.id == "layer_00")
            .unwrap();
        let mut auxiliary = source.circuit.boundary.outputs[0].clone();
        auxiliary.id = "auxiliary_output".to_string();
        source.circuit.boundary.outputs.push(auxiliary);
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("second".to_string(), "layer_01".to_string()),
        ];

        let error =
            StreamCircuitRuntimeGraph::from_source_chain(&resolved, "gpu0", &chain).unwrap_err();

        assert!(error.0.contains("exactly one output port"));
    }

    #[test]
    fn runtime_graph_rejects_control_values_that_execution_would_ignore() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let mut runtime_graph = resolved.default_runtime_graph("gpu0").unwrap();
        runtime_graph.instances[0]
            .control_values
            .insert("unused".to_string(), serde_json::json!(true));

        let error = runtime_graph.validate_against_graph(&resolved).unwrap_err();

        assert!(
            error
                .0
                .contains("executable component controls are not implemented")
        );
    }

    #[test]
    fn runtime_graph_execution_order_comes_from_edges_not_instance_storage_order() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("second".to_string(), "layer_01".to_string()),
            ("third".to_string(), "layer_02".to_string()),
        ];
        let mut runtime_graph =
            StreamCircuitRuntimeGraph::from_source_chain(&resolved, "gpu0", &chain).unwrap();
        runtime_graph.instances.reverse();

        let instantiated = runtime_graph.instantiate_graph(&resolved).unwrap();

        assert_eq!(
            instantiated
                .circuits
                .iter()
                .map(|artifact| artifact.component.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn runtime_graph_validates_state_policy_targets_compatibility_and_cycles() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let mut runtime_graph = resolved
            .default_runtime_graph("gpu0")
            .unwrap()
            .duplicate_after_instance(&resolved, "layer_05", "layer_05_repeat")
            .unwrap();
        runtime_graph
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05_repeat")
            .unwrap()
            .state_policy = StreamCircuitNodeInstanceStatePolicy::ShareWith {
            instance_id: "layer_05".to_string(),
        };
        runtime_graph.validate_against_graph(&resolved).unwrap();

        let mut cross_device_share = runtime_graph.clone();
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

        let mut cycle = runtime_graph;
        cycle
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05")
            .unwrap()
            .state_policy = StreamCircuitNodeInstanceStatePolicy::CloneFrom {
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
    fn runtime_graph_rejects_unrouted_ports_and_multiple_writers_to_one_input() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let runtime_graph = resolved.default_runtime_graph("gpu0").unwrap();

        let mut disconnected = runtime_graph.clone();
        disconnected.edges.remove(4);
        assert!(
            disconnected
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("unrouted ports")
        );

        let mut multiple_writers = runtime_graph;
        let mut duplicate = multiple_writers.edges[0].clone();
        duplicate.id = "second_writer".to_string();
        duplicate.destination = multiple_writers.edges[1].destination.clone();
        multiple_writers.edges.push(duplicate);
        assert!(
            multiple_writers
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("more than one forward edge")
        );
    }

    #[test]
    fn runtime_graph_supports_fanout_to_distinct_inputs() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let mut runtime_graph = resolved.default_runtime_graph("gpu0").unwrap();
        runtime_graph.instances.push(StreamCircuitNodeInstance {
            instance_id: "branch".to_string(),
            source_component_id: "layer_01".to_string(),
            device_id: "gpu0".to_string(),
            enabled: true,
            control_values: BTreeMap::new(),
            state_policy: StreamCircuitNodeInstanceStatePolicy::Fresh,
        });
        runtime_graph.edges.push(StreamCircuitGraphEdge {
            id: "layer_00_to_branch".to_string(),
            source: StreamCircuitEdgeEndpoint {
                component_id: "layer_00".to_string(),
                port_id: "output_frame".to_string(),
            },
            destination: StreamCircuitEdgeEndpoint {
                component_id: "branch".to_string(),
                port_id: "input_frame".to_string(),
            },
            connection: StreamCircuitConnection::Forward,
        });
        runtime_graph
            .boundary
            .public_outputs
            .push(StreamCircuitGraphBoundaryPort {
                id: "branch_output".to_string(),
                endpoint: StreamCircuitEdgeEndpoint {
                    component_id: "branch".to_string(),
                    port_id: "output_frame".to_string(),
                },
            });

        runtime_graph.validate_against_graph(&resolved).unwrap();
        assert_eq!(
            runtime_graph
                .effective_edges()
                .unwrap()
                .iter()
                .filter(|edge| edge.source.component_id == "layer_00")
                .count(),
            2
        );
    }

    #[test]
    fn runtime_graph_accepts_delayed_feedback_and_rejects_instantaneous_cycles() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let runtime_graph = resolved.default_runtime_graph("gpu0").unwrap();
        let feedback_index = runtime_graph
            .edges
            .iter()
            .position(|edge| !edge.connection.is_forward())
            .unwrap();
        runtime_graph.validate_against_graph(&resolved).unwrap();
        assert_eq!(
            runtime_graph.topological_instance_ids(&resolved).unwrap().len(),
            resolved.circuits.len()
        );

        let mut instantaneous = runtime_graph.clone();
        instantaneous.edges[feedback_index].connection = StreamCircuitConnection::Forward;
        assert!(
            instantaneous
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("instantaneous cycle")
        );

        let mut zero_delay = runtime_graph;
        zero_delay.edges[feedback_index].connection = StreamCircuitConnection::TemporalFeedback {
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
