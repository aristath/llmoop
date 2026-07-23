#[test]
fn descriptor_resource_plan_resolves_fixture_model_dispatch_patch_bay() {
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
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();

    assert_eq!(descriptor_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(descriptor_plan.dynamic_state_capacity_activations, 4);
    assert_eq!(descriptor_plan.dispatches.len(), 242);
    assert_eq!(descriptor_plan.total_descriptor_count, 794);

    let first = descriptor_plan
        .dispatch("layer_00", "operator_norm")
        .unwrap();
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(first.descriptors.len(), 3);
    assert_eq!(
        first.descriptors[0].resource,
        VulkanDescriptorResourceAddress::BoundaryInput {
            signal_id: "input_frame".to_string(),
        }
    );
    assert_eq!(
        first.descriptors[1].resource,
        VulkanDescriptorResourceAddress::ActivationSlot {
            pedal_id: "layer_00".to_string(),
            signal_id: "operator_norm_out".to_string(),
            slot: 0,
            byte_capacity: 5120,
            signal_byte_capacity: 2048,
        }
    );
    assert_eq!(
        first.descriptors[2].resource,
        VulkanDescriptorResourceAddress::PermanentParameter {
            param_id: "operator_norm".to_string(),
            tensor: "model.layers.0.operator_norm.weight".to_string(),
            byte_count: Some(2048),
        }
    );

    let kv_append = descriptor_plan
        .dispatch("layer_02", "kv_memory_append")
        .unwrap();
    assert_eq!(kv_append.descriptors.len(), 9);
    assert_eq!(
        kv_append.descriptors[2].resource,
        VulkanDescriptorResourceAddress::StateBuffer {
            pedal_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            state_type: "append_only_attention_memory".to_string(),
            byte_capacity: 8192,
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );
    assert_eq!(
        kv_append.descriptors[6].resource,
        VulkanDescriptorResourceAddress::StateBuffer {
            pedal_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            state_type: "append_only_attention_memory".to_string(),
            byte_capacity: 8192,
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );
    assert_eq!(
        kv_append.descriptors[7].resource,
        VulkanDescriptorResourceAddress::StateView {
            pedal_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            state_type: "append_only_attention_memory".to_string(),
            byte_capacity: 8192,
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );

    let last = descriptor_plan
        .dispatch("layer_13", "ffn_residual")
        .unwrap();
    assert_eq!(
        last.descriptors.last().unwrap().resource,
        VulkanDescriptorResourceAddress::BoundaryOutput {
            signal_id: "output_frame".to_string(),
        }
    );
}

#[test]
fn descriptor_resource_plan_requires_dynamic_capacity_for_kv_state() {
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
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

    let error =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 0).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("layer_02.kv_memory requires non-zero dynamic state capacity")
    );
}

