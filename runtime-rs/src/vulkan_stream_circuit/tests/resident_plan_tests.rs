#[test]
fn resident_plan_uses_typed_activation_slot_byte_capacity() {
    let resource_plan = StreamCircuitResourcePlan {
        circuit_count: 1,
        node_count: 2,
        parameter_ref_count: 0,
        parameters: Vec::new(),
        transducer_parameter_ref_count: 0,
        transducer_parameters: Vec::new(),
        state_allocations: Vec::new(),
        activation_banks: vec![crate::stream_plan::PlannedActivationSlotBank {
            component_id: "layer_00".to_string(),
            circuit_id: "layer_00".to_string(),
            temporary_signal_count: 2,
            state_view_signal_count: 0,
            slot_count: 1,
            slots: vec![crate::stream_plan::PlannedActivationSlot {
                slot: 0,
                signal_ids: vec!["bf16_signal".to_string(), "f32_signal".to_string()],
                max_elements: Some(8),
                max_bytes: Some(32),
            }],
            assignments: Vec::new(),
        }],
        temporary_signal_count: 2,
        state_view_signal_count: 0,
        layer_local_activation_slot_count: 1,
        unknown_temporary_shape_count: 0,
        unknown_state_view_shape_count: 0,
    };

    let resident_plan =
        VulkanStreamCircuitResidentPlan::from_resource_plan(&resource_plan, None, Some(2))
            .unwrap();

    assert_eq!(resident_plan.per_stream_activation_slot_elements, Some(8));
    assert_eq!(resident_plan.per_stream_activation_slot_bytes, Some(32));
    assert_eq!(
        resident_plan.activation_banks[0].slots[0].bytes,
        Some(32)
    );
    assert!(resident_plan.unresolved_activation_slots.is_empty());
}

#[test]
fn plans_fixture_model_vulkan_resident_allocations_from_stream_circuit_resources() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();

    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    assert_eq!(resident_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(resident_plan.circuit_count, 14);
    assert_eq!(resident_plan.permanent_parameters.len(), 130);
    assert_eq!(resident_plan.permanent_parameter_bytes, Some(325_166_592));
    assert!(resident_plan.unresolved_parameter_tensors.is_empty());
    assert_eq!(resident_plan.stream_state_buffers.len(), 14);
    assert_eq!(resident_plan.state_view_signal_count, 20);
    assert_eq!(resident_plan.activation_banks.len(), 14);
    assert_eq!(resident_plan.per_stream_static_state_elements, 8 * 3 * 1024);
    assert_eq!(
        resident_plan.per_stream_dynamic_state_elements_per_activation,
        6 * 1024
    );
    assert_eq!(
        resident_plan.per_stream_activation_slot_elements,
        Some(138_240)
    );
    assert_eq!(resident_plan.per_stream_static_state_bytes, Some(49_152));
    assert_eq!(
        resident_plan.per_stream_dynamic_state_bytes_per_activation,
        Some(12_288)
    );
    assert_eq!(
        resident_plan.per_stream_activation_slot_bytes,
        Some(276_480)
    );
    assert!(resident_plan.unresolved_activation_slots.is_empty());

    let conv_in = resident_plan
        .permanent_parameters
        .iter()
        .find(|parameter| parameter.tensor == "model.layers.0.conv.in_proj.weight")
        .unwrap();
    assert_eq!(conv_in.dtype.as_deref(), Some("BF16"));
    assert_eq!(conv_in.shape, Some(vec![3072, 1024]));
    assert_eq!(conv_in.byte_count, Some(6_291_456));
    assert_eq!(conv_in.use_count, 1);

    let layer_00_bank = resident_plan.activation_bank("layer_00").unwrap();
    assert_eq!(
        layer_00_bank
            .slots
            .iter()
            .map(|slot| slot.bytes)
            .collect::<Vec<_>>(),
        vec![Some(5120), Some(6144), Some(5120), Some(5120)]
    );

    let layer_02_bank = resident_plan.activation_bank("layer_02").unwrap();
    assert_eq!(
        layer_02_bank
            .slots
            .iter()
            .map(|slot| slot.bytes)
            .collect::<Vec<_>>(),
        vec![Some(2048), Some(5120), Some(5120), Some(5120)]
    );
}

#[test]
fn bounds_each_dynamic_state_buffer_by_its_own_activation_limit() {
    let mut graph = fixture_model_execution_graph();
    let artifact = graph
        .circuits
        .iter_mut()
        .find(|artifact| artifact.component.id == "layer_02")
        .unwrap();
    artifact
        .state
        .state_ports
        .iter_mut()
        .find(|state| state.id == "kv_memory")
        .unwrap()
        .max_dynamic_activations = Some(2);
    artifact
        .circuit
        .state_ports
        .iter_mut()
        .find(|state| state.id == "kv_memory")
        .unwrap()
        .max_dynamic_activations = Some(2);

    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let state = resident_plan
        .stream_state_buffers
        .iter()
        .find(|state| state.component_id == "layer_02" && state.state_id == "kv_memory")
        .unwrap();

    assert_eq!(state.max_dynamic_activations, Some(2));
    let layout = VulkanTransientStateBufferLayout::for_state(state, 4).unwrap();
    assert_eq!(layout.dynamic_page_byte_capacity, 4_096);
    assert_eq!(layout.byte_capacity, 4_352);
    assert_eq!(descriptor_state_byte_capacity(state, 4).unwrap(), 4_352);
}

#[test]
fn placed_resident_plan_hosts_only_the_components_assigned_to_a_device() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_spec = StreamCircuitPlacementSpec::new("gpu0")
        .with_component_device("layer_01", "cpu0")
        .with_component_device("layer_02", "gpu1")
        .with_component_device("layer_03", "lan:worker-a");
    let placement_plan = graph.placement_plan(&placement_spec).unwrap();

    let gpu0 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu1 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu1",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let cpu0 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "cpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let lan = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "lan:worker-a",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    assert_eq!(gpu0.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(gpu0.device_id, "gpu0");
    assert_eq!(gpu0.hosted_component_ids.len(), 11);
    assert!(gpu0.hosts_component("layer_00"));
    assert!(!gpu0.hosts_component("layer_02"));
    assert_eq!(gpu0.resident_plan.circuit_count, 11);
    assert_eq!(gpu0.resident_plan.permanent_parameters.len(), 103);
    assert_eq!(gpu0.resident_plan.stream_state_buffers.len(), 11);
    assert_eq!(gpu0.resident_plan.activation_banks.len(), 11);
    assert_eq!(gpu0.resident_plan.state_view_signal_count, 16);
    assert_eq!(gpu0.signal_element_bytes, Some(2));
    assert_eq!(gpu0.local_edges.len(), 9);
    assert_eq!(gpu0.incoming_edges.len(), 1);
    assert_eq!(gpu0.outgoing_edges.len(), 1);
    assert_eq!(gpu0.incoming_edges[0].source_component_id, "layer_03");
    assert_eq!(gpu0.incoming_edges[0].destination_component_id, "layer_04");
    assert_eq!(gpu0.outgoing_edges[0].source_component_id, "layer_00");
    assert_eq!(gpu0.outgoing_edges[0].destination_component_id, "layer_01");

    let gpu0_edge_io = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&gpu0).unwrap();
    assert_eq!(gpu0_edge_io.device_id, "gpu0");
    assert_eq!(gpu0_edge_io.local_edge_count, 9);
    assert_eq!(gpu0_edge_io.total_endpoint_count, 2);
    assert_eq!(gpu0_edge_io.total_buffer_count, 11);
    assert_eq!(gpu0_edge_io.incoming_endpoint_count, 1);
    assert_eq!(gpu0_edge_io.outgoing_endpoint_count, 1);
    assert_eq!(gpu0_edge_io.total_byte_capacity, Some(22_528));
    let gpu0_local = &gpu0_edge_io.local_edges[0];
    assert_eq!(gpu0_local.edge_id, "edge_4_local");
    assert_eq!(gpu0_local.source_component_id, "layer_04");
    assert_eq!(gpu0_local.destination_component_id, "layer_05");
    assert_eq!(gpu0_local.byte_capacity, Some(2_048));

    assert_eq!(gpu1.hosted_component_ids, vec!["layer_02".to_string()]);
    assert_eq!(gpu1.resident_plan.circuit_count, 1);
    assert_eq!(gpu1.resident_plan.permanent_parameters.len(), 11);
    assert_eq!(gpu1.resident_plan.stream_state_buffers.len(), 1);
    assert_eq!(gpu1.resident_plan.state_view_signal_count, 2);
    assert_eq!(gpu1.incoming_edges[0].source_component_id, "layer_01");
    assert_eq!(gpu1.outgoing_edges[0].destination_component_id, "layer_03");
    let gpu1_edge_io = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&gpu1).unwrap();
    assert_eq!(gpu1_edge_io.device_id, "gpu1");
    assert_eq!(gpu1_edge_io.local_edge_count, 0);
    assert_eq!(gpu1_edge_io.total_endpoint_count, 2);
    assert_eq!(gpu1_edge_io.total_buffer_count, 2);
    assert_eq!(gpu1_edge_io.total_byte_capacity, Some(4_096));
    assert_eq!(gpu1_edge_io.unresolved_byte_edges, Vec::<usize>::new());
    let gpu1_incoming = gpu1_edge_io
        .endpoint(VulkanPlacedEdgeDirection::Incoming, 1)
        .unwrap();
    assert_eq!(gpu1_incoming.endpoint_id, "edge_1_in");
    assert_eq!(gpu1_incoming.signal, "frame");
    assert_eq!(gpu1_incoming.shape, vec![1024]);
    assert_eq!(gpu1_incoming.element_count, 1024);
    assert_eq!(gpu1_incoming.byte_capacity, Some(2_048));
    assert_eq!(gpu1_incoming.local_device_id, "gpu1");
    assert_eq!(gpu1_incoming.remote_device_id, "cpu0");
    assert_eq!(gpu1_incoming.local_component_id, "layer_02");
    assert_eq!(gpu1_incoming.remote_component_id, "layer_01");
    assert_eq!(gpu1_incoming.local_port_id, "input_frame");
    assert_eq!(gpu1_incoming.remote_port_id, "output_frame");
    let gpu1_outgoing = gpu1_edge_io
        .endpoint(VulkanPlacedEdgeDirection::Outgoing, 2)
        .unwrap();
    assert_eq!(gpu1_outgoing.endpoint_id, "edge_2_out");
    assert_eq!(gpu1_outgoing.byte_capacity, Some(2_048));
    assert_eq!(gpu1_outgoing.local_device_id, "gpu1");
    assert_eq!(gpu1_outgoing.remote_device_id, "lan:worker-a");
    assert_eq!(gpu1_outgoing.local_component_id, "layer_02");
    assert_eq!(gpu1_outgoing.remote_component_id, "layer_03");

    assert_eq!(cpu0.hosted_component_ids, vec!["layer_01".to_string()]);
    assert_eq!(cpu0.resident_plan.permanent_parameters.len(), 8);
    assert_eq!(cpu0.resident_plan.state_view_signal_count, 1);
    assert_eq!(lan.hosted_component_ids, vec!["layer_03".to_string()]);
    assert_eq!(lan.resident_plan.permanent_parameters.len(), 8);
    assert_eq!(lan.resident_plan.state_view_signal_count, 1);

    let edge_plans = vec![
        gpu0_edge_io,
        gpu1_edge_io,
        VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&cpu0).unwrap(),
        VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&lan).unwrap(),
    ];
    let edge_pairs = pair_placed_edge_endpoints(&edge_plans).unwrap();
    assert_eq!(edge_pairs.len(), 4);
    assert!(edge_pairs.iter().all(|(outgoing, incoming)| {
        VulkanPlacedEdgePacketKey::from_outgoing_endpoint(outgoing)
            == VulkanPlacedEdgePacketKey::from_incoming_endpoint(incoming)
            && outgoing.byte_capacity == incoming.byte_capacity
    }));

    let mut incomplete_plans = edge_plans;
    let plan_with_incoming = incomplete_plans
        .iter_mut()
        .find(|plan| plan.incoming_endpoint_count > 0)
        .unwrap();
    let incoming_index = plan_with_incoming
        .endpoints
        .iter()
        .position(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
        .unwrap();
    plan_with_incoming.endpoints.remove(incoming_index);
    let error = pair_placed_edge_endpoints(&incomplete_plans).unwrap_err();
    assert!(error.to_string().contains("has no incoming endpoint"));
}

#[test]
fn placed_stream_circuit_plan_dispatches_only_hosted_components() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_spec = StreamCircuitPlacementSpec::new("gpu0")
        .with_component_device("layer_01", "cpu0")
        .with_component_device("layer_02", "gpu1")
        .with_component_device("layer_03", "lan:worker-a");
    let placement_plan = graph.placement_plan(&placement_spec).unwrap();
    let gpu0_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu1_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu1",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    let gpu0_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu0_resident)
            .unwrap();
    let gpu1_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu1_resident)
            .unwrap();

    assert_eq!(gpu0_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(gpu0_plan.device_id, "gpu0");
    assert_eq!(gpu0_plan.binding_plan.circuits.len(), 11);
    assert_eq!(gpu0_plan.binding_plan.total_node_count(), 191);
    assert_eq!(gpu0_plan.kernel_interface_plan.total_kernel_count(), 191);
    assert_eq!(gpu0_plan.dispatch_plan.total_dispatch_count(), 191);
    assert!(gpu0_plan.binding_plan.circuit("layer_00").is_some());
    assert!(gpu0_plan.binding_plan.circuit("layer_04").is_some());
    assert!(gpu0_plan.binding_plan.circuit("layer_01").is_none());
    assert!(gpu0_plan.binding_plan.circuit("layer_02").is_none());
    assert!(
        gpu0_plan
            .dispatch_plan
            .command("layer_02", "kv_memory_append")
            .is_none()
    );
    assert_eq!(
        gpu0_plan
            .dispatch_plan
            .command("layer_04", "operator_norm")
            .map(|command| command.dispatch_index),
        Some(16)
    );

    assert_eq!(gpu1_plan.device_id, "gpu1");
    assert_eq!(gpu1_plan.binding_plan.circuits.len(), 1);
    assert_eq!(gpu1_plan.binding_plan.total_node_count(), 19);
    assert_eq!(gpu1_plan.dispatch_plan.total_dispatch_count(), 19);
    assert_eq!(
        gpu1_plan
            .dispatch_plan
            .command("layer_02", "operator_norm")
            .map(|command| command.dispatch_index),
        Some(0)
    );
    assert_eq!(
        gpu1_plan
            .dispatch_plan
            .command("layer_02", "kv_memory_append")
            .map(|command| command.dispatch_index),
        Some(8)
    );
    assert!(
        gpu1_plan
            .dispatch_plan
            .command("layer_00", "operator_norm")
            .is_none()
    );

    let gpu1_manifest = resident_package_reusable_kernel_manifest(&gpu1_plan);
    let gpu1_prepared = gpu1_plan.prepared_dispatch_plan(&gpu1_manifest, 4).unwrap();
    assert_eq!(gpu1_prepared.dispatches.len(), 19);
    assert!(
        gpu1_prepared
            .dispatch("layer_02", "operator_norm")
            .is_some()
    );
    assert!(
        gpu1_prepared
            .dispatch("layer_00", "operator_norm")
            .is_none()
    );

    let gpu1_descriptors = VulkanDescriptorResourcePlan::from_plans(
        &gpu1_plan.dispatch_plan,
        &gpu1_plan.placed_resident_plan.resident_plan,
        4,
    )
    .unwrap();
    assert_eq!(gpu1_descriptors.dispatches.len(), 19);
    let kv_append = gpu1_descriptors
        .dispatch("layer_02", "kv_memory_append")
        .unwrap();
    assert_eq!(kv_append.descriptors.len(), 9);
    assert!(matches!(
        kv_append.descriptors[2].resource,
        VulkanDescriptorResourceAddress::StateBuffer {
            ref component_id,
            ref state_id,
            byte_capacity: 8192,
            ..
        } if component_id == "layer_02" && state_id == "kv_memory"
    ));
}
