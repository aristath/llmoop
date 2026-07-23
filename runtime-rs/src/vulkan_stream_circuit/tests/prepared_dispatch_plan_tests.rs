#[test]
fn prepared_dispatch_plan_links_artifacts_to_descriptor_resources() {
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
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let conv_in_family = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
    let conv_in_family_id = conv_in_family.family_id.as_str();
    let conv_in_artifact_path = artifact_path_for_family(conv_in_family);
    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
    let manifest = VulkanReusableKernelArtifactManifest::new(
        reusable_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );

    let prepared = VulkanPreparedDispatchPlan::from_plans(
        &dispatch_plan,
        &reusable_plan,
        &descriptor_plan,
        &manifest,
    )
    .unwrap();

    assert_eq!(prepared.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(prepared.reusable_family_count, 26);
    assert_eq!(prepared.dispatches.len(), 242);
    assert_eq!(prepared.total_descriptor_count, 794);

    let first = prepared.dispatch("layer_00", "operator_norm").unwrap();
    let first_family = reusable_family_with_kernel(&reusable_plan, "layer_00.operator_norm");
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(first.kernel_id, "layer_00.operator_norm");
    assert_eq!(first.reusable_family_id, first_family.family_id);
    assert_eq!(first.artifact_path, artifact_path_for_family(first_family));
    assert_eq!(first.entry_point, DEFAULT_SPIRV_ENTRY_POINT);
    assert_eq!(first.local_size_x, DEFAULT_COMPUTE_LOCAL_SIZE_X);
    assert_eq!(first.descriptors.len(), 3);

    let linear = prepared.dispatch("layer_00", "conv_in_projection").unwrap();
    assert_eq!(linear.dispatch_index, 1);
    assert_eq!(linear.reusable_family_id, conv_in_family_id);
    assert_eq!(linear.artifact_path, conv_in_artifact_path);
    assert_eq!(linear.descriptors.len(), 3);

    let kv_append = prepared.dispatch("layer_02", "kv_memory_append").unwrap();
    let kv_append_family = reusable_family_with_kernel(&reusable_plan, "layer_02.kv_memory_append");
    assert_eq!(kv_append.dispatch_index, 40);
    assert_eq!(kv_append.reusable_family_id, kv_append_family.family_id);
    assert_eq!(
        kv_append.artifact_path,
        artifact_path_for_family(kv_append_family)
    );
    assert!(kv_append.uses_stream_tick);
    assert_eq!(kv_append.descriptors.len(), 9);
    assert!(matches!(
        kv_append.descriptors[2].resource,
        VulkanDescriptorResourceAddress::StateBuffer {
            ref pedal_id,
            ref state_id,
            byte_capacity: 8192,
            ..
        } if pedal_id == "layer_02" && state_id == "kv_memory"
    ));
    assert!(matches!(
        kv_append.descriptors[6].resource,
        VulkanDescriptorResourceAddress::StateBuffer {
            ref pedal_id,
            ref state_id,
            byte_capacity: 8192,
            ..
        } if pedal_id == "layer_02" && state_id == "kv_memory"
    ));
}

#[test]
fn prepared_dispatch_plan_rejects_unlinked_reusable_kernels() {
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
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
    let linear = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
    let append = reusable_family_with_kernel(&reusable_plan, "layer_02.kv_memory_append");
    let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
        VulkanReusableKernelArtifact::from_family(linear, artifact_path_for_family(linear)),
    );

    let error = VulkanPreparedDispatchPlan::from_plans(
        &dispatch_plan,
        &reusable_plan,
        &descriptor_plan,
        &partial_manifest,
    )
    .unwrap_err();

    let VulkanPreparedDispatchPlanError::Link(link_plan) = error else {
        panic!("expected reusable kernel link failure");
    };
    assert_eq!(link_plan.linked_family_count, 1);
    assert_eq!(link_plan.missing_family_count, 25);
    assert_eq!(link_plan.linked_command_count, 8);
    assert_eq!(link_plan.missing_command_count, 242 - 8);
    assert!(
        link_plan
            .family(&append.family_id)
            .is_some_and(|family| family.status == VulkanReusableKernelLinkStatus::Missing)
    );
}

#[test]
fn bound_dispatch_plan_maps_prepared_descriptors_to_mounted_stream_buffers() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping Vulkan stream-circuit binding: {error}");
            return;
        }
    };
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
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
    let manifest = VulkanReusableKernelArtifactManifest::new(
        reusable_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );
    let prepared = VulkanPreparedDispatchPlan::from_plans(
        &dispatch_plan,
        &reusable_plan,
        &descriptor_plan,
        &manifest,
    )
    .unwrap();
    let buffers = resident_plan.allocate_stream_buffers(&device, 4).unwrap();

    let bound = VulkanBoundDispatchPlan::from_prepared_plan(&prepared, &buffers).unwrap();

    assert_eq!(bound.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(bound.dispatches.len(), 242);
    assert_eq!(bound.total_descriptor_count, 794);
    assert_eq!(bound.boundary_descriptor_count, 42);
    assert_eq!(bound.permanent_parameter_descriptor_count, 130);
    assert_eq!(bound.stream_state_descriptor_count, 122);
    assert_eq!(bound.activation_slot_descriptor_count, 500);
    assert_eq!(
        bound.boundary_descriptor_count
            + bound.permanent_parameter_descriptor_count
            + bound.stream_state_descriptor_count
            + bound.activation_slot_descriptor_count,
        bound.total_descriptor_count
    );

    let first = bound.dispatch("layer_00", "operator_norm").unwrap();
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(
        first.descriptors[0].target,
        VulkanBoundDescriptorTarget::BoundaryInput {
            signal_id: "input_frame".to_string(),
        }
    );
    assert_eq!(
        first.descriptors[1].target,
        VulkanBoundDescriptorTarget::ActivationSlot {
            buffer_index: buffers.activation_slot_buffer_index("layer_00", 0).unwrap(),
            pedal_id: "layer_00".to_string(),
            signal_id: "operator_norm_out".to_string(),
            circuit_id: "layer_00_shortconv_circuit_v1".to_string(),
            slot: 0,
            byte_capacity: 5120,
            signal_byte_capacity: 2048,
        }
    );
    assert_eq!(
        first.descriptors[2].target,
        VulkanBoundDescriptorTarget::PermanentParameter {
            param_id: "operator_norm".to_string(),
            tensor: "model.layers.0.operator_norm.weight".to_string(),
            byte_count: Some(2048),
        }
    );

    let kv_append = bound.dispatch("layer_02", "kv_memory_append").unwrap();
    assert!(matches!(
        kv_append.descriptors[2].target,
        VulkanBoundDescriptorTarget::StreamStateBuffer {
            ref pedal_id,
            ref state_id,
            byte_capacity: 8192,
            ..
        } if pedal_id == "layer_02" && state_id == "kv_memory"
    ));
    assert!(matches!(
        kv_append.descriptors[6].target,
        VulkanBoundDescriptorTarget::StreamStateBuffer {
            ref pedal_id,
            ref state_id,
            byte_capacity: 8192,
            ..
        } if pedal_id == "layer_02" && state_id == "kv_memory"
    ));
    assert!(matches!(
        kv_append.descriptors[7].target,
        VulkanBoundDescriptorTarget::StreamStateView {
            ref pedal_id,
            ref state_id,
            byte_capacity: 8192,
            ..
        } if pedal_id == "layer_02" && state_id == "kv_memory"
    ));
}

#[test]
fn mounts_fixture_model_stream_circuit_resources_without_claiming_execution() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping Vulkan stream-circuit mount: {error}");
            return;
        }
    };
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

    let mounted = VulkanMountedStreamCircuit::from_plans(
        &device,
        &execution_plan,
        &resource_plan,
        resident_plan,
        4,
    )
    .unwrap();

    assert!(!mounted.can_execute());
    assert_eq!(mounted.resident_plan.permanent_parameters.len(), 130);
    assert_eq!(mounted.binding_plan.total_node_count(), 242);
    assert_eq!(mounted.kernel_interface_plan.total_kernel_count(), 242);
    assert_eq!(mounted.dispatch_plan.total_dispatch_count(), 242);
    assert_eq!(mounted.reusable_kernel_plan.total_family_count(), 26);
    assert_eq!(mounted.reusable_kernel_plan.total_command_count, 242);
    let empty_coverage = mounted.reusable_kernel_coverage_report(std::iter::empty::<&str>());
    assert!(!empty_coverage.all_available());
    assert_eq!(empty_coverage.missing_family_count, 26);
    assert_eq!(empty_coverage.missing_command_count, 242);
    let empty_link = mounted.link_reusable_kernels(&VulkanReusableKernelArtifactManifest::empty());
    assert!(!empty_link.is_fully_linked());
    assert_eq!(empty_link.missing_family_count, 26);
    assert_eq!(empty_link.missing_command_count, 242);
    let descriptor_plan = mounted.descriptor_resource_plan().unwrap();
    assert_eq!(descriptor_plan.total_descriptor_count, 794);
    assert_eq!(descriptor_plan.dynamic_state_capacity_activations, 4);
    let manifest = VulkanReusableKernelArtifactManifest::new(
        mounted
            .reusable_kernel_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );
    let prepared = mounted.prepared_dispatch_plan(&manifest).unwrap();
    assert_eq!(prepared.dispatches.len(), 242);
    assert_eq!(prepared.total_descriptor_count, 794);
    let bound = mounted.bound_dispatch_plan(&manifest).unwrap();
    assert_eq!(bound.dispatches.len(), 242);
    assert_eq!(bound.total_descriptor_count, 794);
    assert_eq!(mounted.buffers.state_buffers.len(), 14);
    assert_eq!(mounted.buffers.activation_slot_buffers.len(), 56);
    assert_eq!(mounted.buffers.total_byte_capacity, 374_784);

    let attention = mounted
        .binding_plan
        .circuit("layer_02")
        .unwrap()
        .node("attention_read")
        .unwrap();
    assert!(matches!(
        attention.input("k_memory").unwrap().resource,
        VulkanSignalResource::StateView { .. }
    ));
    assert_eq!(
        mounted
            .buffers
            .activation_slot_buffer("layer_02", 0)
            .map(|buffer| buffer.byte_capacity),
        Some(2_048)
    );
    assert_eq!(
        mounted
            .dispatch_plan
            .command("layer_02", "kv_memory_append")
            .map(|command| command.dispatch_index),
        Some(40)
    );
}
