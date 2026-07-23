#[test]
fn mounted_placed_stream_circuit_binds_only_local_device_slice() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed Vulkan stream-circuit mount: {error}");
            return;
        }
    };
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
    let gpu1_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu1",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu1_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu1_resident)
            .unwrap();

    let mounted =
        VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, gpu1_plan, 4).unwrap();

    assert_eq!(mounted.device_id(), "gpu1");
    assert!(!mounted.can_execute());
    assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 1);
    assert_eq!(mounted.placed_plan.dispatch_plan.total_dispatch_count(), 19);
    assert_eq!(mounted.parameter_buffers.plan.device_id, "gpu1");
    assert_eq!(mounted.parameter_buffers.plan.parameter_count, 11);
    assert_eq!(
        mounted.parameter_buffers.plan.total_byte_capacity,
        mounted
            .placed_plan
            .placed_resident_plan
            .resident_plan
            .permanent_parameter_bytes
    );
    assert_eq!(
        Some(mounted.parameter_buffers.total_byte_capacity),
        mounted.parameter_buffers.plan.total_byte_capacity
    );
    assert!(mounted.parameter_buffers.plan.unresolved_tensors.is_empty());
    assert_eq!(mounted.boundary_io.plan.device_id, "gpu1");
    assert_eq!(mounted.boundary_io.plan.input_count, 0);
    assert_eq!(mounted.boundary_io.plan.output_count, 0);
    assert_eq!(mounted.boundary_io.plan.total_buffer_count, 0);
    assert_eq!(mounted.boundary_io.plan.total_byte_capacity, Some(0));
    assert_eq!(mounted.boundary_io.total_byte_capacity, 0);
    assert_eq!(mounted.buffers.state_buffers.len(), 1);
    assert_eq!(mounted.buffers.activation_slot_buffers.len(), 4);
    assert_eq!(mounted.buffers.total_byte_capacity, 25_600);
    assert_eq!(mounted.edge_io.plan.device_id, "gpu1");
    assert_eq!(mounted.edge_io.plan.total_endpoint_count, 2);
    assert_eq!(mounted.edge_io.plan.total_byte_capacity, Some(4_096));
    assert_eq!(mounted.edge_io.incoming_buffers.len(), 1);
    assert_eq!(mounted.edge_io.outgoing_buffers.len(), 1);
    assert_eq!(mounted.edge_io.total_byte_capacity, 4_096);
    let incoming_edge = mounted.edge_io.incoming_buffer(1).unwrap();
    assert_eq!(
        incoming_edge.endpoint.direction,
        VulkanPlacedEdgeDirection::Incoming
    );
    assert_eq!(incoming_edge.endpoint.local_component_id, "layer_02");
    assert_eq!(incoming_edge.endpoint.remote_component_id, "layer_01");
    assert_eq!(incoming_edge.byte_capacity, 2_048);
    assert_eq!(incoming_edge.buffer.byte_capacity(), 2_048);
    assert!(incoming_edge.buffer.is_persistently_mapped());
    incoming_edge.buffer.write_bytes(&[7, 8, 9, 10]).unwrap();
    assert_eq!(
        incoming_edge.buffer.read_bytes(4).unwrap(),
        vec![7, 8, 9, 10]
    );
    let outgoing_edge = mounted.edge_io.outgoing_buffer(2).unwrap();
    assert_eq!(
        outgoing_edge.endpoint.direction,
        VulkanPlacedEdgeDirection::Outgoing
    );
    assert_eq!(outgoing_edge.endpoint.local_component_id, "layer_02");
    assert_eq!(outgoing_edge.endpoint.remote_component_id, "layer_03");
    assert_eq!(outgoing_edge.byte_capacity, 2_048);
    assert_eq!(outgoing_edge.buffer.byte_capacity(), 2_048);
    assert!(outgoing_edge.buffer.is_persistently_mapped());
    assert_eq!(
        mounted
            .buffers
            .state_buffer("layer_02", "kv_memory")
            .map(|buffer| buffer.byte_capacity),
        Some(8_192)
    );

    let descriptor_plan = mounted.descriptor_resource_plan().unwrap();
    assert_eq!(descriptor_plan.dispatches.len(), 19);
    assert!(
        descriptor_plan
            .dispatch("layer_00", "operator_norm")
            .is_none()
    );
    assert!(
        descriptor_plan
            .dispatch("layer_02", "kv_memory_append")
            .is_some()
    );

    let manifest = VulkanReusableKernelArtifactManifest::new(
        mounted
            .placed_plan
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
    assert_eq!(prepared.dispatches.len(), 19);
    assert_eq!(
        prepared
            .dispatch("layer_02", "kv_memory_append")
            .map(|dispatch| dispatch.artifact_path.as_str()),
        Some("kernels/append_state_update.spv")
    );
    let bound = mounted.bound_dispatch_plan(&manifest).unwrap();
    assert_eq!(bound.dispatches.len(), 19);
    assert_eq!(
        bound.total_descriptor_count,
        prepared.total_descriptor_count
    );
    assert!(bound.boundary_descriptor_count > 0);
    assert!(bound.permanent_parameter_descriptor_count > 0);
    assert!(bound.stream_state_descriptor_count > 0);
    assert!(bound.activation_slot_descriptor_count > 0);

    let placed_bound = mounted.placed_bound_dispatch_plan(&manifest).unwrap();
    assert_eq!(placed_bound.device_id, "gpu1");
    assert_eq!(placed_bound.dispatches.len(), 19);
    assert_eq!(
        placed_bound.total_descriptor_count,
        bound.total_descriptor_count
    );
    assert_eq!(placed_bound.model_boundary_descriptor_count, 0);
    assert_eq!(placed_bound.local_edge_descriptor_count, 0);
    assert_eq!(placed_bound.incoming_edge_descriptor_count, 2);
    assert_eq!(placed_bound.outgoing_edge_descriptor_count, 1);
    assert_eq!(
        placed_bound
            .dispatch("layer_02", "operator_norm")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanPlacedBoundDescriptorTarget::IncomingEdge {
            edge: mounted.placed_plan.placed_resident_plan.incoming_edges[0].clone(),
        }
    );
    assert_eq!(
        placed_bound
            .dispatch("layer_02", "operator_residual")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanPlacedBoundDescriptorTarget::IncomingEdge {
            edge: mounted.placed_plan.placed_resident_plan.incoming_edges[0].clone(),
        }
    );
    assert_eq!(
        placed_bound
            .dispatch("layer_02", "ffn_residual")
            .unwrap()
            .descriptors
            .last()
            .unwrap()
            .target,
        VulkanPlacedBoundDescriptorTarget::OutgoingEdge {
            edge: mounted.placed_plan.placed_resident_plan.outgoing_edges[0].clone(),
        }
    );

    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&manifest)
        .unwrap();
    assert_eq!(mounted_bound.device_id, "gpu1");
    assert_eq!(mounted_bound.dispatches.len(), 19);
    assert_eq!(
        mounted_bound.total_descriptor_count,
        placed_bound.total_descriptor_count
    );
    assert_eq!(
        mounted_bound.resident_descriptor_count,
        placed_bound.resident_descriptor_count
    );
    assert_eq!(mounted_bound.model_boundary_descriptor_count, 0);
    assert_eq!(mounted_bound.local_edge_descriptor_count, 0);
    assert_eq!(mounted_bound.edge_endpoint_descriptor_count, 3);
    assert_eq!(mounted_bound.incoming_edge_descriptor_count, 2);
    assert_eq!(mounted_bound.outgoing_edge_descriptor_count, 1);
    assert_eq!(
        mounted_bound
            .dispatch("layer_02", "operator_norm")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::IncomingEdgeBuffer {
            endpoint: VulkanPlacedEdgeEndpointBufferBinding {
                buffer_index: 0,
                endpoint: mounted
                    .edge_io
                    .incoming_buffer(1)
                    .unwrap()
                    .endpoint
                    .clone(),
                byte_capacity: 2_048,
            },
        }
    );
    assert_eq!(
        mounted_bound
            .dispatch("layer_02", "operator_residual")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::IncomingEdgeBuffer {
            endpoint: VulkanPlacedEdgeEndpointBufferBinding {
                buffer_index: 0,
                endpoint: mounted
                    .edge_io
                    .incoming_buffer(1)
                    .unwrap()
                    .endpoint
                    .clone(),
                byte_capacity: 2_048,
            },
        }
    );
    assert_eq!(
        mounted_bound
            .dispatch("layer_02", "ffn_residual")
            .unwrap()
            .descriptors
            .last()
            .unwrap()
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::OutgoingEdgeBuffer {
            endpoint: VulkanPlacedEdgeEndpointBufferBinding {
                buffer_index: 0,
                endpoint: mounted
                    .edge_io
                    .outgoing_buffer(2)
                    .unwrap()
                    .endpoint
                    .clone(),
                byte_capacity: 2_048,
            },
        }
    );

    let tick_plan = mounted.stream_tick_plan(&manifest).unwrap();
    assert_eq!(tick_plan.device_id, "gpu1");
    assert!(!tick_plan.can_execute);
    assert_eq!(tick_plan.stage_count, 21);
    assert_eq!(tick_plan.receive_stage_count, 1);
    assert_eq!(tick_plan.dispatch_stage_count, 19);
    assert_eq!(tick_plan.publish_stage_count, 1);
    assert_eq!(tick_plan.local_edge_read_count, 0);
    assert_eq!(tick_plan.local_edge_write_count, 0);
    assert_eq!(tick_plan.incoming_edge_read_count, 2);
    assert_eq!(tick_plan.outgoing_edge_write_count, 1);
    assert_eq!(tick_plan.model_input_read_count, 0);
    assert_eq!(tick_plan.model_output_write_count, 0);
    assert_eq!(
        tick_plan.stages[0],
        VulkanMountedPlacedStreamTickStage::ReceiveEdge {
            stage_index: 0,
            edge_index: 1,
            endpoint_id: "edge_1_in".to_string(),
            buffer_index: 0,
            byte_capacity: 2_048,
            remote_device_id: "cpu0".to_string(),
            remote_component_id: "layer_01".to_string(),
        }
    );
    assert_eq!(
        tick_plan.stages[1],
        VulkanMountedPlacedStreamTickStage::Dispatch {
            stage_index: 1,
            dispatch: VulkanMountedPlacedStreamTickDispatch {
                dispatch_index: 0,
                kernel_id: "layer_02.operator_norm".to_string(),
                component_id: "layer_02".to_string(),
                node_id: "operator_norm".to_string(),
                op: "rms_norm".to_string(),
                descriptor_count: mounted_bound
                    .dispatch("layer_02", "operator_norm")
                    .unwrap()
                    .descriptors
                    .len(),
                resident_descriptor_count: 2,
                reads: vec![VulkanMountedPlacedStreamTickIo::IncomingEdgeBuffer {
                    edge_index: 1,
                    buffer_index: 0,
                    byte_capacity: 2_048,
                }],
                writes: vec![],
            },
        }
    );
    assert_eq!(
        tick_plan.stages[20],
        VulkanMountedPlacedStreamTickStage::PublishEdge {
            stage_index: 20,
            edge_index: 2,
            endpoint_id: "edge_2_out".to_string(),
            buffer_index: 0,
            byte_capacity: 2_048,
            remote_device_id: "lan:worker-a".to_string(),
            remote_component_id: "layer_03".to_string(),
        }
    );
    let tick_run = mounted.advance_stream_tick(&manifest, 7).unwrap();
    assert_eq!(tick_run.device_id, "gpu1");
    assert_eq!(tick_run.stream_tick, 7);
    assert!(!tick_run.can_execute);
    assert_eq!(tick_run.planned_stage_count, 21);
    assert_eq!(tick_run.attempted_stage_count, 1);
    assert_eq!(tick_run.completed_stage_count, 0);
    assert_eq!(tick_run.pending_stage_count, 20);
    assert_eq!(
        tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 0,
            reason: VulkanMountedPlacedStreamTickBlockReason::EdgeReceiveTransportUnavailable,
        }
    );
    assert_eq!(tick_run.stages[0].stage, tick_plan.stages[0]);
    assert_eq!(
        tick_run.stages[0].status,
        VulkanMountedPlacedStreamTickStageStatus::Blocked {
            reason: VulkanMountedPlacedStreamTickBlockReason::EdgeReceiveTransportUnavailable,
        }
    );
    assert_eq!(
        tick_run.stages[1].status,
        VulkanMountedPlacedStreamTickStageStatus::Pending
    );
    assert_eq!(
        tick_run.stages[20].status,
        VulkanMountedPlacedStreamTickStageStatus::Pending
    );

    let kv_append = bound.dispatch("layer_02", "kv_memory_append").unwrap();
    assert!(matches!(
        kv_append.descriptors[2].target,
        VulkanBoundDescriptorTarget::StreamStateBuffer {
            ref component_id,
            ref state_id,
            buffer_index: 0,
            byte_capacity: 8192,
            ..
        } if component_id == "layer_02" && state_id == "kv_memory"
    ));
    let mounted_kv_append = mounted_bound
        .dispatch("layer_02", "kv_memory_append")
        .unwrap();
    let kv_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(mounted_kv_append)
        .unwrap();
    assert_eq!(kv_bindings.len(), mounted_kv_append.descriptors.len() + 1);
    assert_eq!(kv_bindings[0].binding, 0);
    assert_eq!(kv_bindings[0].byte_len, 2_048);
    assert_eq!(kv_bindings[2].binding, 2);
    assert_eq!(kv_bindings[2].byte_len, 8_192);
    assert_eq!(kv_bindings.last().unwrap().binding, 9);
    assert_eq!(
        kv_bindings.last().unwrap().byte_len,
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );

    if let Some(spirv_words) = crate::vulkan_compute::compile_test_shader_words() {
        let family = mounted
            .placed_plan
            .reusable_kernel_plan
            .family(&mounted_kv_append.reusable_family_id)
            .unwrap();
        let loaded_manifest = VulkanLoadedReusableKernelArtifactManifest {
            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            total_word_count: spirv_words.len(),
            artifacts: vec![VulkanLoadedReusableKernelArtifact {
                artifact: VulkanReusableKernelArtifact::from_family(
                    family,
                    "kernels/append_state_update.spv",
                ),
                resolved_path: PathBuf::from("kernels/append_state_update.spv"),
                words: spirv_words,
            }],
        };
        let resident_dispatch = mounted
            .create_resident_kernel_dispatch_for_bound_dispatch(
                &device,
                mounted_kv_append,
                &loaded_manifest,
            )
            .unwrap();

        assert_eq!(resident_dispatch.descriptor_count(), kv_bindings.len());
        assert_eq!(resident_dispatch.workgroup_count_x(), 1);
        assert_eq!(resident_dispatch.push_constant_byte_count(), 0);
    } else {
        eprintln!(
            "skipping resident kernel dispatch handle smoke: no GLSL to SPIR-V compiler found"
        );
    }
}

#[test]
fn in_process_edge_transport_moves_bytes_between_mounted_device_slices() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping in-process placed edge transport: {error}");
            return;
        }
    };
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_spec =
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1");
    let placement_plan = graph.placement_plan(&placement_spec).unwrap();

    let gpu0_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu0_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu0_resident)
            .unwrap();
    let gpu0 = VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, gpu0_plan, 4).unwrap();

    let gpu1_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu1",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu1_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu1_resident)
            .unwrap();
    let gpu1 = VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, gpu1_plan, 4).unwrap();

    assert_eq!(
        gpu0.edge_io.outgoing_buffer(1).unwrap().byte_capacity,
        2_048
    );
    assert_eq!(
        gpu1.edge_io.incoming_buffer(1).unwrap().byte_capacity,
        2_048
    );
    assert_eq!(
        gpu1.edge_io.outgoing_buffer(2).unwrap().byte_capacity,
        2_048
    );
    assert_eq!(
        gpu0.edge_io.incoming_buffer(2).unwrap().byte_capacity,
        2_048
    );

    let mut layer_01_to_02 = vec![0u8; 2_048];
    layer_01_to_02[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    gpu0.edge_io
        .outgoing_buffer(1)
        .unwrap()
        .buffer
        .write_bytes(&layer_01_to_02)
        .unwrap();

    let mut transport = VulkanInProcessPlacedEdgeTransport::new();
    let publish = transport.publish_outgoing_edge(&gpu0, 1).unwrap();
    assert_eq!(
        publish.key,
        VulkanPlacedEdgePacketKey {
            edge_index: 1,
            from_device_id: "gpu0".to_string(),
            to_device_id: "gpu1".to_string(),
        }
    );
    assert_eq!(publish.byte_count, 2_048);
    assert_eq!(transport.packet_count(), 1);
    assert!(transport.contains_packet(&publish.key));

    let gpu1_manifest = VulkanReusableKernelArtifactManifest::new(
        gpu1.placed_plan
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
    let gpu1_tick_plan = gpu1.stream_tick_plan(&gpu1_manifest).unwrap();
    let transport_tick = gpu1_tick_plan
        .advance_with_in_process_transport(&gpu1, &mut transport, 11)
        .unwrap();
    assert_eq!(transport_tick.stream_tick, 11);
    assert_eq!(transport_tick.attempted_stage_count, 2);
    assert_eq!(transport_tick.completed_stage_count, 1);
    assert_eq!(transport_tick.pending_stage_count, 19);
    assert_eq!(
        transport_tick.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 1,
            reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
        }
    );
    assert_eq!(
        transport_tick.stages[0].status,
        VulkanMountedPlacedStreamTickStageStatus::Completed
    );
    assert_eq!(
        transport_tick.stages[1].status,
        VulkanMountedPlacedStreamTickStageStatus::Blocked {
            reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
        }
    );
    assert_eq!(
        gpu1.edge_io
            .incoming_buffer(1)
            .unwrap()
            .buffer
            .read_bytes(8)
            .unwrap(),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(transport.packet_count(), 0);
    let missing_transport_tick = gpu1_tick_plan
        .advance_with_in_process_transport(&gpu1, &mut transport, 12)
        .unwrap();
    assert_eq!(
        missing_transport_tick.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 0,
            reason: VulkanMountedPlacedStreamTickBlockReason::EdgeReceiveTransportUnavailable,
        }
    );

    let mut layer_02_to_03 = vec![0u8; 2_048];
    layer_02_to_03[..8].copy_from_slice(&[21, 22, 23, 24, 25, 26, 27, 28]);
    gpu1.edge_io
        .outgoing_buffer(2)
        .unwrap()
        .buffer
        .write_bytes(&layer_02_to_03)
        .unwrap();
    let published_back = transport.publish_all_outgoing_edges(&gpu1).unwrap();
    assert_eq!(published_back.len(), 1);
    let publish_back = &published_back[0];
    assert_eq!(
        publish_back.key,
        VulkanPlacedEdgePacketKey {
            edge_index: 2,
            from_device_id: "gpu1".to_string(),
            to_device_id: "gpu0".to_string(),
        }
    );
    let received_back = transport.receive_available_incoming_edges(&gpu0).unwrap();
    assert_eq!(received_back.received.len(), 1);
    assert_eq!(received_back.missing_packets.len(), 0);
    assert_eq!(received_back.received[0].key, publish_back.key);
    assert_eq!(
        gpu0.edge_io
            .incoming_buffer(2)
            .unwrap()
            .buffer
            .read_bytes(8)
            .unwrap(),
        vec![21, 22, 23, 24, 25, 26, 27, 28]
    );
}

