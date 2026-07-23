#[test]
fn resident_plan_infers_state_sizes_without_a_tensor_index() {
    let graph = fixture_model_execution_graph();
    let resource_plan = StreamCircuitResourcePlan::from_graph(&graph).unwrap();

    let resident_plan =
        VulkanStreamCircuitResidentPlan::from_resource_plan(&resource_plan, None, None).unwrap();

    assert_eq!(resident_plan.permanent_parameters.len(), 130);
    assert_eq!(resident_plan.permanent_parameter_bytes, None);
    assert_eq!(resident_plan.unresolved_parameter_tensors.len(), 130);
    assert_eq!(resident_plan.per_stream_static_state_elements, 8 * 3 * 1024);
    assert_eq!(
        resident_plan.per_stream_dynamic_state_elements_per_activation,
        6 * 1024
    );
    assert_eq!(resident_plan.per_stream_static_state_bytes, Some(49_152));
    assert_eq!(
        resident_plan.per_stream_dynamic_state_bytes_per_activation,
        Some(12_288)
    );
    assert_eq!(resident_plan.per_stream_activation_slot_elements, None);
    assert_eq!(resident_plan.per_stream_activation_slot_bytes, None);
    assert!(!resident_plan.unresolved_activation_slots.is_empty());
}

#[test]
fn permanent_parameter_plan_excludes_physically_lowered_tensors() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_plan = graph
        .placement_plan(&StreamCircuitPlacementSpec::new("gpu0"))
        .unwrap();
    let placed_resident_plan =
        VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
    let full = VulkanPermanentParameterBufferPlan::from_placed_resident_plan(&placed_resident_plan)
        .unwrap();
    let removed = full.parameters.iter().take(2).cloned().collect::<Vec<_>>();
    let excluded = removed
        .iter()
        .map(|parameter| parameter.tensor.clone())
        .collect::<BTreeSet<_>>();

    let pruned = VulkanPermanentParameterBufferPlan::from_placed_resident_plan_excluding_tensors(
        &placed_resident_plan,
        &excluded,
    )
    .unwrap();

    assert_eq!(pruned.parameter_count, full.parameter_count - 2);
    assert_eq!(
        pruned.total_byte_capacity,
        Some(
            full.total_byte_capacity.unwrap()
                - removed
                    .iter()
                    .map(|parameter| parameter.byte_capacity.unwrap())
                    .sum::<usize>()
        )
    );
    assert!(
        pruned
            .parameters
            .iter()
            .all(|parameter| !excluded.contains(&parameter.tensor))
    );
    assert!(
        pruned
            .parameters
            .iter()
            .enumerate()
            .all(|(index, parameter)| parameter.buffer_index == index)
    );

    let error = VulkanPermanentParameterBufferPlan::from_placed_resident_plan_excluding_tensors(
        &placed_resident_plan,
        &BTreeSet::from(["not-a-resident-tensor".to_string()]),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot exclude unavailable permanent parameter tensor")
    );
}

#[test]
fn allocates_fixture_model_per_stream_vulkan_buffers_from_resident_plan() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping Vulkan stream-circuit allocation: {error}");
            return;
        }
    };
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

    let buffers = resident_plan.allocate_stream_buffers(&device, 4).unwrap();

    assert_eq!(buffers.dynamic_state_capacity_activations, 4);
    assert_eq!(buffers.state_buffers.len(), 14);
    assert_eq!(buffers.activation_slot_buffers.len(), 56);
    assert_eq!(buffers.total_byte_capacity, 49_152 + 12_288 * 4 + 276_480);

    let layer_00_state = buffers
        .state_buffers
        .iter()
        .find(|buffer| buffer.component_id == "layer_00")
        .unwrap();
    assert_eq!(layer_00_state.state_id, "temporal_memory");
    assert_eq!(layer_00_state.byte_capacity, 6_144);
    assert_eq!(layer_00_state.buffer.byte_capacity(), 6_144);

    let layer_02_state = buffers
        .state_buffers
        .iter()
        .find(|buffer| buffer.component_id == "layer_02")
        .unwrap();
    assert_eq!(layer_02_state.state_id, "kv_memory");
    assert_eq!(layer_02_state.byte_capacity, 8_192);
    assert_eq!(layer_02_state.buffer.byte_capacity(), 8_192);

    let layer_00_slot_1 = buffers
        .activation_slot_buffers
        .iter()
        .find(|buffer| buffer.component_id == "layer_00" && buffer.slot == 1)
        .unwrap();
    assert_eq!(layer_00_slot_1.byte_capacity, 6_144);
    assert!(
        layer_00_slot_1
            .signal_ids
            .contains(&"conv_projected".to_string())
    );
    assert_eq!(
        buffers
            .state_buffer("layer_02", "kv_memory")
            .map(|buffer| buffer.byte_capacity),
        Some(8_192)
    );
    assert_eq!(
        buffers
            .activation_slot_buffer("layer_02", 0)
            .map(|buffer| buffer.byte_capacity),
        Some(2_048)
    );
}

#[test]
fn binds_fixture_model_nodes_to_vulkan_resident_resources() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();

    assert_eq!(binding_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(binding_plan.circuits.len(), 14);
    assert_eq!(binding_plan.total_node_count(), 242);

    let layer_00 = binding_plan.circuit("layer_00").unwrap();
    let operator_norm = layer_00.node("operator_norm").unwrap();
    assert_eq!(
        operator_norm.input("input_frame").unwrap().resource,
        VulkanSignalResource::BoundaryInput
    );
    assert_eq!(
        operator_norm.parameter("operator_norm").unwrap().tensor,
        "model.layers.0.operator_norm.weight"
    );

    let conv_in = layer_00.node("conv_in_projection").unwrap();
    assert_eq!(
        conv_in.parameter("conv_in_projection").unwrap().tensor,
        "model.layers.0.conv.in_proj.weight"
    );
    assert_eq!(
        conv_in.output("conv_projected").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            component_id: "layer_00".to_string(),
            slot: 1,
            bytes: Some(6144),
            signal_bytes: Some(6144),
        }
    );

    let temporal_update = layer_00.node("temporal_memory_update").unwrap();
    assert_eq!(
        temporal_update.input("temporal_memory").unwrap().resource,
        VulkanSignalResource::StateBuffer {
            component_id: "layer_00".to_string(),
            state_id: "temporal_memory".to_string(),
            static_bytes: Some(6144),
            bytes_per_activation: None,
        }
    );
    assert_eq!(
        temporal_update.output("temporal_window").unwrap().resource,
        VulkanSignalResource::StateView {
            component_id: "layer_00".to_string(),
            state_id: "temporal_memory".to_string(),
            static_bytes: Some(6144),
            bytes_per_activation: None,
        }
    );

    let layer_02 = binding_plan.circuit("layer_02").unwrap();
    let kv_append = layer_02.node("kv_memory_append").unwrap();
    assert_eq!(
        kv_append.input("kv_memory").unwrap().resource,
        VulkanSignalResource::StateBuffer {
            component_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );
    assert_eq!(
        kv_append.output("k_memory").unwrap().resource,
        VulkanSignalResource::StateView {
            component_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );
    assert_eq!(
        kv_append.output("v_memory").unwrap().resource,
        VulkanSignalResource::StateView {
            component_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );

    let attention = layer_02.node("attention_read").unwrap();
    assert_eq!(
        attention.input("q_positioned").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            component_id: "layer_02".to_string(),
            slot: 2,
            bytes: Some(5120),
            signal_bytes: Some(2048),
        }
    );
    assert!(matches!(
        attention.input("k_memory").unwrap().resource,
        VulkanSignalResource::StateView { .. }
    ));
    assert_eq!(
        attention.output("attention_out").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            component_id: "layer_02".to_string(),
            slot: 0,
            bytes: Some(2048),
            signal_bytes: Some(2048),
        }
    );
}

