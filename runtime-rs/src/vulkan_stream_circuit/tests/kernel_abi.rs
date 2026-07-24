#[test]
fn kernel_interfaces_describe_fixture_model_compiled_component_abi() {
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

    let kernel_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);

    assert_eq!(kernel_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(kernel_plan.circuits.len(), 14);
    assert_eq!(kernel_plan.total_kernel_count(), 242);

    let conv_in = kernel_plan
        .kernel("layer_00", "conv_in_projection")
        .unwrap();
    assert_eq!(conv_in.kernel_id, "layer_00.conv_in_projection");
    assert_eq!(conv_in.op, "linear");
    assert_eq!(conv_in.inputs.len(), 1);
    assert_eq!(conv_in.outputs.len(), 1);
    assert_eq!(conv_in.parameters.len(), 1);
    assert!(conv_in.state_reads.is_empty());
    assert!(conv_in.state_writes.is_empty());
    assert!(conv_in.state_views.is_empty());
    assert!(!conv_in.stream_metadata.uses_stream_tick);
    assert_eq!(
        conv_in.parameters[0],
        VulkanParameterBinding {
            param_id: "conv_in_projection".to_string(),
            tensor: "model.layers.0.conv.in_proj.weight".to_string(),
            byte_count: Some(6_291_456),
            shape: Some(vec![3072, 1024]),
        }
    );
    assert_eq!(
        conv_in.outputs[0].resource,
        VulkanSignalResource::ActivationSlot {
            component_id: "layer_00".to_string(),
            slot: 1,
            bytes: Some(6144),
            signal_bytes: Some(6144),
        }
    );

    let q_rope = kernel_plan.kernel("layer_02", "q_rope").unwrap();
    assert_eq!(q_rope.op, "rotary_position_embedding");
    assert!(q_rope.stream_metadata.uses_stream_tick);
    assert_eq!(
        q_rope.stream_metadata.stream_tick,
        VulkanKernelScalarBinding {
            name: "stream_tick".to_string(),
            scalar_type: "u64".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }
    );
    assert_eq!(q_rope.stream_metadata.control_flags.name, "control_flags");
    assert_eq!(
        q_rope.outputs[0].resource,
        VulkanSignalResource::ActivationSlot {
            component_id: "layer_02".to_string(),
            slot: 2,
            bytes: Some(5120),
            signal_bytes: Some(2048),
        }
    );

    let kv_append = kernel_plan.kernel("layer_02", "kv_memory_append").unwrap();
    assert_eq!(kv_append.op, "append_state_update");
    assert!(kv_append.stream_metadata.uses_stream_tick);
    assert_eq!(kv_append.inputs.len(), 3);
    assert_eq!(kv_append.outputs.len(), 2);
    assert_eq!(kv_append.state_reads.len(), 1);
    assert_eq!(kv_append.state_writes.len(), 1);
    assert_eq!(kv_append.state_views.len(), 2);
    assert_eq!(
        kv_append.inputs[2].resource,
        VulkanSignalResource::StateBuffer {
            component_id: "layer_02".to_string(),
            state_id: "kv_memory".to_string(),
            static_bytes: None,
            bytes_per_activation: Some(2048),
        }
    );
    assert!(
        kv_append
            .state_views
            .iter()
            .all(|view| matches!(view.resource, VulkanSignalResource::StateView { .. }))
    );
    assert_eq!(
        kv_append
            .stream_metadata
            .dynamic_state_capacity_activations
            .name,
        "dynamic_state_capacity_activations"
    );
}

#[test]
fn stream_control_buffer_bytes_follow_kernel_abi_order() {
    let push_constants =
        VulkanKernelStreamMetadata::for_op("rotary_position_embedding").push_constants();
    assert!(push_constants.is_empty());
    let control = VulkanMountedPlacedStreamControl {
        stream_tick: 42,
        control_flags: 7,
        dynamic_state_capacity_activations: 4,
    };
    let push_bytes = stream_control_push_constant_bytes(&push_constants, control).unwrap();
    assert!(push_bytes.is_empty());

    let bytes = stream_control_bytes(11, control);
    assert_eq!(&bytes[0..4], &11u32.to_le_bytes());
    assert_eq!(&bytes[4..12], &42u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &7u32.to_le_bytes());
    assert_eq!(&bytes[16..20], &4u32.to_le_bytes());
}

#[test]
fn component_batch_lane_controls_preserve_each_token_identity() {
    let controls = component_batch_lane_stream_control_bytes(&[9259, 1902], 41, 65_536).unwrap();

    assert_eq!(controls.len(), 2);
    assert_eq!(&controls[0][0..4], &9259u32.to_le_bytes());
    assert_eq!(&controls[0][4..12], &41u64.to_le_bytes());
    assert_eq!(&controls[1][0..4], &1902u32.to_le_bytes());
    assert_eq!(&controls[1][4..12], &42u64.to_le_bytes());
    assert_eq!(&controls[0][16..20], &65_536u32.to_le_bytes());
    assert_eq!(&controls[1][16..20], &65_536u32.to_le_bytes());
}

#[test]
fn recurrent_gate_kernel_receives_stream_control_metadata() {
    let metadata = VulkanKernelStreamMetadata::for_op("rg_lru_step");

    assert!(metadata.uses_stream_tick);
    assert!(metadata.push_constants().is_empty());
}

#[test]
fn sparse_moe_kernels_receive_an_explicit_expert_start() {
    for op in ["sparse_moe_gate_up", "sparse_moe_down"] {
        let metadata = VulkanKernelStreamMetadata::for_op(op);
        let push_constants = metadata.push_constants();

        assert_eq!(
            push_constants,
            vec![VulkanKernelScalarBinding {
                name: "expert_start".to_string(),
                scalar_type: "u32".to_string(),
                source: VulkanKernelScalarSource::PushConstant,
            }]
        );
        assert_eq!(
            stream_control_push_constant_bytes(
                &push_constants,
                VulkanMountedPlacedStreamControl {
                    stream_tick: 42,
                    control_flags: 7,
                    dynamic_state_capacity_activations: 65_536,
                },
            )
            .unwrap(),
            0u32.to_le_bytes()
        );
    }
}

#[test]
fn fused_head_norm_rope_kernel_receives_stream_control_metadata() {
    let metadata = VulkanKernelStreamMetadata::for_op("parallel_head_norm_rope_2way");

    assert!(metadata.uses_stream_tick);
    assert!(metadata.push_constants().is_empty());
}

#[test]
fn fused_append_attention_kernel_receives_stream_control_metadata() {
    let metadata = VulkanKernelStreamMetadata::for_op("append_scaled_dot_product_attention");

    assert!(metadata.uses_stream_tick);
    assert!(metadata.push_constants().is_empty());
}

#[test]
fn dispatch_plan_orders_fixture_model_kernel_commands_for_stream_ticks() {
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

    assert_eq!(dispatch_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(dispatch_plan.total_dispatch_count(), 242);
    assert_eq!(dispatch_plan.op_counts().get("linear"), Some(&82));

    let first = &dispatch_plan.commands[0];
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(first.circuit_index, 0);
    assert_eq!(first.kernel_id, "layer_00.operator_norm");
    assert_eq!(first.component_id, "layer_00");
    assert_eq!(first.node_index, 0);
    assert_eq!(first.op, "rms_norm");
    assert_eq!(first.descriptor_bindings.len(), 3);
    assert_eq!(
        first
            .descriptor_bindings
            .iter()
            .map(|binding| binding.usage.clone())
            .collect::<Vec<_>>(),
        vec![
            VulkanKernelDescriptorUsage::InputSignal,
            VulkanKernelDescriptorUsage::OutputSignal,
            VulkanKernelDescriptorUsage::Parameter,
        ]
    );
    assert!(first.push_constants.is_empty());
    assert!(!first.uses_stream_tick);

    let kv_append = dispatch_plan
        .command("layer_02", "kv_memory_append")
        .unwrap();
    assert_eq!(kv_append.dispatch_index, 40);
    assert_eq!(kv_append.circuit_index, 2);
    assert_eq!(kv_append.node_index, 8);
    assert_eq!(kv_append.op, "append_state_update");
    assert!(kv_append.uses_stream_tick);
    assert_eq!(
        kv_append
            .descriptor_bindings
            .iter()
            .map(|binding| (
                binding.binding,
                binding.usage.clone(),
                binding.name.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, VulkanKernelDescriptorUsage::InputSignal, "k_positioned"),
            (1, VulkanKernelDescriptorUsage::InputSignal, "v_projected"),
            (2, VulkanKernelDescriptorUsage::InputSignal, "kv_memory"),
            (3, VulkanKernelDescriptorUsage::OutputSignal, "k_memory"),
            (4, VulkanKernelDescriptorUsage::OutputSignal, "v_memory"),
            (5, VulkanKernelDescriptorUsage::StateRead, "kv_memory"),
            (6, VulkanKernelDescriptorUsage::StateWrite, "kv_memory"),
            (7, VulkanKernelDescriptorUsage::StateView, "k_memory"),
            (8, VulkanKernelDescriptorUsage::StateView, "v_memory"),
        ]
    );
    assert_eq!(
        kv_append.descriptor_bindings[2].resource,
        VulkanKernelDescriptorResource::Signal(VulkanSignalBinding {
            signal_id: "kv_memory".to_string(),
            resource: VulkanSignalResource::StateBuffer {
                component_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                static_bytes: None,
                bytes_per_activation: Some(2048),
            },
        })
    );
    assert_eq!(
        kv_append.descriptor_bindings[6].resource,
        VulkanKernelDescriptorResource::State {
            component_id: "layer_02".to_string(),
            binding: VulkanStateBinding {
                component_id: "layer_02".to_string(),
                state_id: "kv_memory".to_string(),
                state_type: "append_only_attention_memory".to_string(),
                static_bytes: None,
                bytes_per_activation: Some(2048),
            },
        }
    );

    let last = dispatch_plan.commands.last().unwrap();
    assert_eq!(last.dispatch_index, 241);
    assert_eq!(last.circuit_index, 13);
    assert_eq!(last.kernel_id, "layer_13.ffn_residual");
    assert_eq!(last.node_index, 15);
    assert_eq!(
        last.descriptor_bindings.last().unwrap().resource,
        VulkanKernelDescriptorResource::Signal(VulkanSignalBinding {
            signal_id: "output_frame".to_string(),
            resource: VulkanSignalResource::BoundaryOutput,
        })
    );
}
