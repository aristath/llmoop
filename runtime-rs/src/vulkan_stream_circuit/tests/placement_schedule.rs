#[test]
fn placed_token_input_route_follows_the_execution_graph_edge() {
    assert_eq!(
        placed_token_input(7, "gpu0", "gpu0", false),
        VulkanResidentPlacedTokenInput::HostSupplied(7)
    );
    assert_eq!(
        placed_token_input(11, "gpu0", "gpu0", true),
        VulkanResidentPlacedTokenInput::ResidentFeedback(11)
    );
    assert_eq!(
        placed_token_input(13, "gpu0", "gpu1", true),
        VulkanResidentPlacedTokenInput::EdgeFeedback(13)
    );
}

#[test]
fn placed_activation_schedule_is_compiled_from_edge_dependencies() {
    fn dispatch(stage_index: usize, device_id: &str) -> VulkanMountedPlacedStreamTickStage {
        VulkanMountedPlacedStreamTickStage::Dispatch {
            stage_index,
            dispatch: VulkanMountedPlacedStreamTickDispatch {
                dispatch_index: stage_index,
                kernel_id: format!("{device_id}.kernel_{stage_index}"),
                component_id: format!("{device_id}.component"),
                node_id: format!("node_{stage_index}"),
                op: "test".to_string(),
                descriptor_count: 0,
                resident_descriptor_count: 0,
                reads: Vec::new(),
                writes: Vec::new(),
            },
        }
    }

    fn plan(
        device_id: &str,
        stages: Vec<VulkanMountedPlacedStreamTickStage>,
    ) -> VulkanMountedPlacedStreamTickPlan {
        let stage_count = stages.len();
        let receive_stage_count = stages
            .iter()
            .filter(|stage| {
                matches!(
                    stage,
                    VulkanMountedPlacedStreamTickStage::ReceiveEdge { .. }
                )
            })
            .count();
        let dispatch_stage_count = stages
            .iter()
            .filter(|stage| matches!(stage, VulkanMountedPlacedStreamTickStage::Dispatch { .. }))
            .count();
        let publish_stage_count = stages
            .iter()
            .filter(|stage| {
                matches!(
                    stage,
                    VulkanMountedPlacedStreamTickStage::PublishEdge { .. }
                )
            })
            .count();
        VulkanMountedPlacedStreamTickPlan {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: device_id.to_string(),
            stages,
            stage_count,
            receive_stage_count,
            dispatch_stage_count,
            publish_stage_count,
            local_edge_read_count: 0,
            local_edge_write_count: 0,
            incoming_edge_read_count: receive_stage_count,
            outgoing_edge_write_count: publish_stage_count,
            model_input_read_count: 0,
            model_output_write_count: 0,
            can_execute: true,
        }
    }

    let gpu0 = plan(
        "gpu0",
        vec![
            dispatch(0, "gpu0"),
            VulkanMountedPlacedStreamTickStage::PublishEdge {
                stage_index: 1,
                edge_index: 0,
                endpoint_id: "edge_0_out".to_string(),
                buffer_index: 0,
                byte_capacity: 16,
                remote_device_id: "gpu1".to_string(),
                remote_component_id: "gpu1.component".to_string(),
            },
            VulkanMountedPlacedStreamTickStage::ReceiveEdge {
                stage_index: 2,
                edge_index: 1,
                endpoint_id: "edge_1_in".to_string(),
                buffer_index: 0,
                byte_capacity: 16,
                remote_device_id: "gpu1".to_string(),
                remote_component_id: "gpu1.component".to_string(),
            },
            dispatch(3, "gpu0"),
        ],
    );
    let gpu1 = plan(
        "gpu1",
        vec![
            VulkanMountedPlacedStreamTickStage::ReceiveEdge {
                stage_index: 0,
                edge_index: 0,
                endpoint_id: "edge_0_in".to_string(),
                buffer_index: 0,
                byte_capacity: 16,
                remote_device_id: "gpu0".to_string(),
                remote_component_id: "gpu0.component".to_string(),
            },
            dispatch(1, "gpu1"),
            VulkanMountedPlacedStreamTickStage::PublishEdge {
                stage_index: 2,
                edge_index: 1,
                endpoint_id: "edge_1_out".to_string(),
                buffer_index: 0,
                byte_capacity: 16,
                remote_device_id: "gpu0".to_string(),
                remote_component_id: "gpu0.component".to_string(),
            },
        ],
    );

    let schedule =
        VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[&gpu0, &gpu1]).unwrap();
    assert_eq!(schedule.device_ids, ["gpu0", "gpu1"]);
    assert_eq!(schedule.turns, [vec![0, 1], vec![0]]);
    assert_eq!(schedule.turns.len(), 2);
    assert_eq!(schedule.turns.iter().map(Vec::len).sum::<usize>(), 3);

    let reversed =
        VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[&gpu1, &gpu0]).unwrap();
    assert_eq!(reversed.device_ids, ["gpu1", "gpu0"]);
    assert_eq!(reversed.turns, [vec![1], vec![0, 1]]);
    assert_eq!(reversed.turns.iter().map(Vec::len).sum::<usize>(), 3);
}

#[test]
fn placed_activation_schedule_rejects_unroutable_edges() {
    let blocked = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stages: vec![VulkanMountedPlacedStreamTickStage::ReceiveEdge {
            stage_index: 0,
            edge_index: 7,
            endpoint_id: "missing_in".to_string(),
            buffer_index: 0,
            byte_capacity: 16,
            remote_device_id: "gpu1".to_string(),
            remote_component_id: "remote".to_string(),
        }],
        stage_count: 1,
        receive_stage_count: 1,
        dispatch_stage_count: 0,
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 1,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: true,
    };
    assert_eq!(
        VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[&blocked])
            .unwrap_err()
            .0,
        "placed activation topology is blocked with pending devices [\"gpu0\"]"
    );

    let mut unconsumed = blocked.clone();
    unconsumed.stages[0] = VulkanMountedPlacedStreamTickStage::PublishEdge {
        stage_index: 0,
        edge_index: 7,
        endpoint_id: "orphan_out".to_string(),
        buffer_index: 0,
        byte_capacity: 16,
        remote_device_id: "gpu1".to_string(),
        remote_component_id: "remote".to_string(),
    };
    assert!(
        VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[&unconsumed])
            .unwrap_err()
            .0
            .contains("leaves unconsumed edges")
    );

    assert_eq!(
        VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[&blocked, &blocked])
            .unwrap_err()
            .0,
        "placed activation schedule repeats device \"gpu0\""
    );
}

#[test]
fn runtime_graph_state_policies_change_resident_state_allocation_and_binding() {
    let manifest = fixture_model_package_manifest();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(PathBuf::from("."))
        .unwrap();
    let state_id = source_graph
        .circuits
        .iter()
        .find(|artifact| artifact.component.id == "layer_05")
        .and_then(|artifact| artifact.state.state_ports.first())
        .map(|state| state.id.clone())
        .unwrap();
    let mut shared_patch = StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0")
        .unwrap()
        .duplicate_after_instance(&source_graph, "layer_05", "layer_05_repeat")
        .unwrap();
    shared_patch
        .instances
        .iter_mut()
        .find(|instance| instance.instance_id == "layer_05_repeat")
        .unwrap()
        .state_policy = StreamCircuitNodeInstanceStatePolicy::ShareWith {
        instance_id: "layer_05".to_string(),
    };
    let shared_runtime_model = manifest.clone().mount_runtime_graph(&shared_patch).unwrap();
    let shared_graph = shared_runtime_model
        .resolved_graph(PathBuf::from("."))
        .unwrap();
    let shared_execution = StreamCircuitExecutionPlan::from_graph(&shared_graph).unwrap();
    let shared_resources =
        StreamCircuitResourcePlan::from_graph_and_plan(&shared_graph, &shared_execution).unwrap();
    let shared_resident =
        VulkanStreamCircuitResidentPlan::from_resource_plan(&shared_resources, None, Some(2))
            .unwrap();

    assert_eq!(shared_resident.stream_state_buffers.len(), 14);
    let shared_bindings = state_binding_index(&shared_resources, &shared_resident).unwrap();
    assert_eq!(
        shared_bindings
            .get(&("layer_05_repeat".to_string(), state_id.clone()))
            .unwrap()
            .component_id,
        "layer_05"
    );

    let mut cloned_patch = shared_patch;
    cloned_patch
        .instances
        .iter_mut()
        .find(|instance| instance.instance_id == "layer_05_repeat")
        .unwrap()
        .state_policy = StreamCircuitNodeInstanceStatePolicy::CloneFrom {
        instance_id: "layer_05".to_string(),
    };
    let cloned_runtime_model = manifest.mount_runtime_graph(&cloned_patch).unwrap();
    let cloned_graph = cloned_runtime_model
        .resolved_graph(PathBuf::from("."))
        .unwrap();
    let cloned_execution = StreamCircuitExecutionPlan::from_graph(&cloned_graph).unwrap();
    let cloned_resources =
        StreamCircuitResourcePlan::from_graph_and_plan(&cloned_graph, &cloned_execution).unwrap();
    let cloned_resident =
        VulkanStreamCircuitResidentPlan::from_resource_plan(&cloned_resources, None, Some(2))
            .unwrap();
    let cloned = cloned_resident
        .stream_state_buffers
        .iter()
        .find(|state| state.component_id == "layer_05_repeat" && state.state_id == state_id)
        .unwrap();
    assert_eq!(cloned.clone_from, Some(("layer_05".to_string(), state_id)));
    assert_eq!(cloned_resident.stream_state_buffers.len(), 15);
}

#[test]
fn clone_state_copy_order_preserves_inherited_instances_and_initializes_new_clones() {
    let state_id = |component: &str| (component.to_string(), "memory".to_string());
    let inherited = BTreeSet::from([state_id("existing_clone")]);

    let copies = ordered_clone_state_copies(
        [
            (state_id("source"), None),
            (state_id("existing_clone"), Some(state_id("source"))),
            (state_id("new_clone"), Some(state_id("source"))),
            (state_id("chained_clone"), Some(state_id("new_clone"))),
        ],
        &inherited,
    )
    .unwrap();

    assert_eq!(
        copies,
        vec![
            (state_id("new_clone"), state_id("source")),
            (state_id("chained_clone"), state_id("new_clone")),
        ]
    );
}

#[test]
fn clone_state_copy_order_rejects_missing_sources_and_cycles() {
    let state_id = |component: &str| (component.to_string(), "memory".to_string());
    let missing = ordered_clone_state_copies(
        [(state_id("clone"), Some(state_id("missing")))],
        &BTreeSet::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing.contains("references unavailable source missing.memory"));

    let cycle = ordered_clone_state_copies(
        [
            (state_id("a"), Some(state_id("b"))),
            (state_id("b"), Some(state_id("a"))),
        ],
        &BTreeSet::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(cycle.contains("dependency cycle"));
}

fn fixture_model_input_embedding_transducer_spec() -> VulkanResidentInputEmbeddingTransducerSpec {
    fixture_model_package_manifest().input_transducer.spec
}

fn fixture_model_output_transducer_spec() -> VulkanResidentOutputTransducerSpec {
    fixture_model_package_manifest().output_transducer.spec
}

fn fixture_model_greedy_sampler_spec() -> VulkanResidentSamplerSpec {
    VulkanResidentSamplerSpec {
        sampler_id: FIXTURE_MODEL_GREEDY_SAMPLER_COMPONENT_ID.to_string(),
        method: "greedy".to_string(),
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: 1,
        runtime_parameterized: false,
        logits_byte_capacity: FIXTURE_MODEL_LOGITS_BYTES,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: 0,
    }
}

