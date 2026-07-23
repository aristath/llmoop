#[test]
fn placed_tick_plan_interleaves_cross_device_edges_with_hosted_components() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping interleaved placed tick plan: {error}");
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
    let manifest = VulkanReusableKernelArtifactManifest::new(
        gpu0.placed_plan
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
    let tick_plan = gpu0.stream_tick_plan(&manifest).unwrap();

    let publish_to_gpu1_index = tick_plan
        .stages
        .iter()
        .position(|stage| {
            matches!(
                stage,
                VulkanMountedPlacedStreamTickStage::PublishEdge {
                    edge_index: 1,
                    remote_device_id,
                    remote_component_id,
                    ..
                } if remote_device_id == "gpu1" && remote_component_id == "layer_02"
            )
        })
        .unwrap();
    let receive_from_gpu1_index = tick_plan
        .stages
        .iter()
        .position(|stage| {
            matches!(
                stage,
                VulkanMountedPlacedStreamTickStage::ReceiveEdge {
                    edge_index: 2,
                    remote_device_id,
                    remote_component_id,
                    ..
                } if remote_device_id == "gpu1" && remote_component_id == "layer_02"
            )
        })
        .unwrap();
    let first_layer_03_dispatch_index = tick_plan
        .stages
        .iter()
        .position(|stage| {
            matches!(
                stage,
                VulkanMountedPlacedStreamTickStage::Dispatch {
                    dispatch,
                    ..
                } if dispatch.component_id == "layer_03" && dispatch.node_id == "operator_norm"
            )
        })
        .unwrap();

    assert_eq!(tick_plan.receive_stage_count, 1);
    assert_eq!(tick_plan.publish_stage_count, 1);
    assert!(
        publish_to_gpu1_index < receive_from_gpu1_index,
        "gpu0 must publish its prefix output before waiting for the remote layer output"
    );
    assert!(
        receive_from_gpu1_index < first_layer_03_dispatch_index,
        "gpu0 must receive the remote layer output before running the downstream suffix"
    );
}

#[test]
fn resident_stream_tick_executes_split_device_slice_and_publishes_output_edge() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident split-device stream tick: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let gpu0_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model.clone(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let gpu1_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model,
        "gpu1",
        Some(4),
    )
    .unwrap();
    let gpu0 = gpu0_slice.create_mounted_stream_circuit(&device).unwrap();
    let gpu1 = gpu1_slice.create_mounted_stream_circuit(&device).unwrap();
    gpu0.buffers.zero_state_buffers().unwrap();
    gpu1.buffers.zero_state_buffers().unwrap();

    let mut layer_01_to_02 = Vec::with_capacity(2_048);
    for _ in 0..1_024 {
        layer_01_to_02.extend_from_slice(&[0x80, 0x3f]);
    }
    gpu0.edge_io
        .outgoing_buffer(1)
        .unwrap()
        .buffer
        .write_bytes(&layer_01_to_02)
        .unwrap();

    let mut transport = VulkanInProcessPlacedEdgeTransport::new();
    transport.publish_outgoing_edge(&gpu0, 1).unwrap();

    let reusable_manifest = resident_package_reusable_kernel_manifest(&gpu1.placed_plan);
    let gpu1_bound = gpu1
        .mounted_placed_bound_dispatch_plan(&reusable_manifest)
        .unwrap();
    let gpu1_tick_plan = gpu1.stream_tick_plan(&reusable_manifest).unwrap();
    let run = gpu1_tick_plan
        .advance_with_resident_execution_graph_and_in_process_transport(
            &device,
            &gpu1,
            &gpu1_bound,
            gpu1_slice.loaded_manifest(),
            &mut transport,
            0,
        )
        .unwrap();

    assert_eq!(
        run.tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Completed
    );
    assert_eq!(run.tick_run.attempted_stage_count, 18);
    assert_eq!(run.tick_run.completed_stage_count, 18);
    assert_eq!(run.tick_run.pending_stage_count, 0);
    assert!(run.tick_run.can_execute);
    assert_eq!(run.execution_graph_dispatch_count(), 16);
    assert_eq!(
        run.execution_graph_run.as_ref().unwrap().component_ids(),
        vec!["layer_02"]
    );

    let kv_memory = gpu1.buffers.state_buffer("layer_02", "kv_memory").unwrap();
    assert_ne!(kv_memory.buffer.read_bytes(16).unwrap(), vec![0; 16]);

    let output_packet_key = VulkanPlacedEdgePacketKey {
        edge_index: 2,
        from_device_id: "gpu1".to_string(),
        to_device_id: "gpu0".to_string(),
    };
    assert_eq!(transport.packet_count(), 1);
    assert!(transport.contains_packet(&output_packet_key));

    let received_back = transport.receive_available_incoming_edges(&gpu0).unwrap();
    assert_eq!(received_back.received.len(), 1);
    assert_eq!(received_back.received[0].key, output_packet_key);
    assert_eq!(received_back.received[0].byte_count, 2_048);
    assert_eq!(received_back.missing_packets.len(), 0);
}

#[test]
fn resident_stream_tick_cursor_resumes_split_prefix_and_suffix_without_rerunning_prefix() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident split cursor stream tick: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let gpu0_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model.clone(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let gpu1_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model,
        "gpu1",
        Some(4),
    )
    .unwrap();
    let gpu0 = gpu0_slice.create_mounted_stream_circuit(&device).unwrap();
    let gpu1 = gpu1_slice.create_mounted_stream_circuit(&device).unwrap();
    gpu0.buffers.zero_state_buffers().unwrap();
    gpu1.buffers.zero_state_buffers().unwrap();

    let mut input_frame = Vec::with_capacity(2_048);
    for _ in 0..1_024 {
        input_frame.extend_from_slice(&[0x80, 0x3f]);
    }
    gpu0.boundary_io
        .input_buffer("input_frame")
        .unwrap()
        .buffer
        .write_bytes(&input_frame)
        .unwrap();

    let gpu0_reusable_manifest = resident_package_reusable_kernel_manifest(&gpu0.placed_plan);
    let gpu0_bound = gpu0
        .mounted_placed_bound_dispatch_plan(&gpu0_reusable_manifest)
        .unwrap();
    let gpu0_tick_plan = gpu0.stream_tick_plan(&gpu0_reusable_manifest).unwrap();
    let mut gpu0_cursor = gpu0_tick_plan.resident_stream_tick_cursor(0);

    let gpu1_reusable_manifest = resident_package_reusable_kernel_manifest(&gpu1.placed_plan);
    let gpu1_bound = gpu1
        .mounted_placed_bound_dispatch_plan(&gpu1_reusable_manifest)
        .unwrap();
    let gpu1_tick_plan = gpu1.stream_tick_plan(&gpu1_reusable_manifest).unwrap();
    let mut gpu1_cursor = gpu1_tick_plan.resident_stream_tick_cursor(0);

    let mut transport = VulkanInProcessPlacedEdgeTransport::new();
    let gpu0_prefix = gpu0_cursor
        .advance_with_resident_components_and_in_process_transport(
            &device,
            &gpu0,
            &gpu0_bound,
            gpu0_slice.loaded_manifest(),
            &mut transport,
        )
        .unwrap();
    let gpu0_prefix_run = gpu0_cursor.snapshot();
    let blocked_stage_index = match &gpu0_prefix_run.tick_run.status {
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index,
            reason,
        } => {
            assert_eq!(
                *reason,
                VulkanMountedPlacedStreamTickBlockReason::EdgeReceiveTransportUnavailable
            );
            *stage_index
        }
        ref status => panic!("expected gpu0 prefix to block on remote return, got {status:?}"),
    };
    assert_eq!(gpu0_prefix.completed_stage_delta, 27);
    assert_eq!(gpu0_prefix_run.execution_graph_dispatch_count(), 26);
    assert!(matches!(
        gpu0_prefix_run.tick_run.stages[blocked_stage_index].stage,
        VulkanMountedPlacedStreamTickStage::ReceiveEdge {
            edge_index: 2,
            ref remote_device_id,
            ref remote_component_id,
            ..
        } if remote_device_id == "gpu1" && remote_component_id == "layer_02"
    ));
    assert_eq!(transport.packet_count(), 1);
    assert!(transport.contains_packet(&VulkanPlacedEdgePacketKey {
        edge_index: 1,
        from_device_id: "gpu0".to_string(),
        to_device_id: "gpu1".to_string(),
    }));

    let gpu1_remote = gpu1_cursor
        .advance_with_resident_components_and_in_process_transport(
            &device,
            &gpu1,
            &gpu1_bound,
            gpu1_slice.loaded_manifest(),
            &mut transport,
        )
        .unwrap();
    let gpu1_remote_run = gpu1_cursor.snapshot();
    assert!(gpu1_remote.completed);
    assert_eq!(gpu1_remote_run.execution_graph_dispatch_count(), 16);
    assert_eq!(transport.packet_count(), 1);
    assert!(transport.contains_packet(&VulkanPlacedEdgePacketKey {
        edge_index: 2,
        from_device_id: "gpu1".to_string(),
        to_device_id: "gpu0".to_string(),
    }));

    let gpu0_suffix = gpu0_cursor
        .advance_with_resident_components_and_in_process_transport(
            &device,
            &gpu0,
            &gpu0_bound,
            gpu0_slice.loaded_manifest(),
            &mut transport,
        )
        .unwrap();
    let gpu0_suffix_run = gpu0_cursor.snapshot();
    assert!(gpu0_suffix.completed);
    assert_eq!(
        gpu0_suffix_run.tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Completed
    );
    assert_eq!(gpu0_suffix_run.tick_run.completed_stage_count, 186);
    assert_eq!(gpu0_suffix_run.execution_graph_dispatch_count(), 184);
    assert_eq!(transport.packet_count(), 0);
    assert_eq!(
        gpu0_cursor
            .component_runs
            .iter()
            .map(|run| run.component_id.as_str())
            .collect::<Vec<_>>()[..3],
        ["layer_00", "layer_01", "layer_03"]
    );

    let output = gpu0
        .boundary_io
        .output_buffer("output_frame")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(output, vec![0; 16]);
}

#[test]
fn in_process_resident_stream_tick_runner_completes_split_package_slices() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping in-process resident split runner: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let gpu0_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model.clone(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let gpu1_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model,
        "gpu1",
        Some(4),
    )
    .unwrap();
    let gpu0 = gpu0_slice.create_mounted_stream_circuit(&device).unwrap();
    let gpu1 = gpu1_slice.create_mounted_stream_circuit(&device).unwrap();
    gpu0.buffers.zero_state_buffers().unwrap();
    gpu1.buffers.zero_state_buffers().unwrap();

    let mut input_frame = Vec::with_capacity(2_048);
    for _ in 0..1_024 {
        input_frame.extend_from_slice(&[0x80, 0x3f]);
    }
    gpu0.boundary_io
        .input_buffer("input_frame")
        .unwrap()
        .buffer
        .write_bytes(&input_frame)
        .unwrap();

    let gpu0_reusable_manifest = resident_package_reusable_kernel_manifest(&gpu0.placed_plan);
    let gpu0_bound = gpu0
        .mounted_placed_bound_dispatch_plan(&gpu0_reusable_manifest)
        .unwrap();
    let gpu0_tick_plan = gpu0.stream_tick_plan(&gpu0_reusable_manifest).unwrap();

    let gpu1_reusable_manifest = resident_package_reusable_kernel_manifest(&gpu1.placed_plan);
    let gpu1_bound = gpu1
        .mounted_placed_bound_dispatch_plan(&gpu1_reusable_manifest)
        .unwrap();
    let gpu1_tick_plan = gpu1.stream_tick_plan(&gpu1_reusable_manifest).unwrap();
    let gpu0_execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan::from_tick_plan(
        &device,
        &gpu0,
        &gpu0_bound,
        gpu0_slice.loaded_manifest(),
        gpu0_tick_plan,
    )
    .unwrap();
    let gpu1_execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan::from_tick_plan(
        &device,
        &gpu1,
        &gpu1_bound,
        gpu1_slice.loaded_manifest(),
        gpu1_tick_plan,
    )
    .unwrap();

    let mut transport = VulkanInProcessPlacedEdgeTransport::new();
    let mut slices = vec![
        VulkanMountedPlacedResidentInProcessStreamTickSlice::new(
            &device,
            &gpu0,
            &gpu0_execution_plan,
            0,
        ),
        VulkanMountedPlacedResidentInProcessStreamTickSlice::new(
            &device,
            &gpu1,
            &gpu1_execution_plan,
            0,
        ),
    ];
    register_in_process_direct_edge_copies(&slices, &mut transport).unwrap();
    assert_eq!(transport.direct_edge_binding_count(), 2);
    register_in_process_direct_edge_copies(&slices, &mut transport).unwrap();
    assert_eq!(transport.direct_edge_binding_count(), 2);
    let run =
        run_mounted_placed_resident_stream_tick_slices_in_process(&mut slices, &mut transport)
            .unwrap();

    assert_eq!(
        run.status,
        VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed
    );
    assert_eq!(run.scheduler_turn_count, 2);
    assert_eq!(run.completed_slice_count, 2);
    assert_eq!(run.pending_slice_count, 0);
    assert_eq!(run.completed_stage_delta, 204);
    assert_eq!(run.transport_stats.pending_packet_count, 0);
    assert_eq!(run.transport_stats.pending_byte_count, 0);
    assert_eq!(run.transport_stats.pending_direct_edge_count, 0);
    assert_eq!(run.transport_stats.pending_direct_byte_count, 0);
    assert_eq!(run.transport_stats.published_packet_count, 0);
    assert_eq!(run.transport_stats.published_byte_count, 0);
    assert_eq!(run.transport_stats.received_packet_count, 0);
    assert_eq!(run.transport_stats.received_byte_count, 0);
    assert_eq!(run.transport_stats.direct_copy_count, 2);
    assert_eq!(run.transport_stats.direct_copy_byte_count, 4096);
    assert_eq!(run.transport_stats.direct_receive_count, 2);
    assert_eq!(run.transport_stats.direct_receive_byte_count, 4096);
    assert_eq!(transport.direct_edge_binding_count(), 2);
    assert_eq!(transport.packet_count(), 0);

    let gpu0_run = run
        .device_runs
        .iter()
        .find(|run| run.tick_run.device_id == "gpu0")
        .unwrap();
    let gpu1_run = run
        .device_runs
        .iter()
        .find(|run| run.tick_run.device_id == "gpu1")
        .unwrap();
    assert_eq!(gpu0_run.tick_run.completed_stage_count, 186);
    assert_eq!(gpu0_run.execution_graph_dispatch_count(), 184);
    assert_eq!(gpu1_run.tick_run.completed_stage_count, 18);
    assert_eq!(gpu1_run.execution_graph_dispatch_count(), 16);

    let output = gpu0
        .boundary_io
        .output_buffer("output_frame")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(output, vec![0; 16]);
}

#[test]
fn placed_model_package_runs_split_stream_tick_in_process() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed model package split stream tick: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let placed_model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_devices(
            &device,
            manifest_dir,
            runtime_model,
            Some(4),
        )
        .unwrap(),
    );
    let placed_package = placed_model
        .create_stream_processor_for_devices(&device, 0)
        .unwrap();
    assert_eq!(placed_model.device_ids, vec!["gpu0", "gpu1"]);
    assert_eq!(placed_model.device_count, 2);
    assert_eq!(placed_model.hosted_component_count, 14);
    assert_eq!(
        placed_package.device("gpu0").unwrap().hosted_component_count,
        13
    );
    assert_eq!(placed_package.device("gpu1").unwrap().hosted_component_count, 1);

    let mut input_frame = Vec::with_capacity(2_048);
    for _ in 0..1_024 {
        input_frame.extend_from_slice(&[0x80, 0x3f]);
    }
    placed_package
        .mounted_device("gpu0")
        .unwrap()
        .boundary_io
        .input_buffer("input_frame")
        .unwrap()
        .buffer
        .write_bytes(&input_frame)
        .unwrap();

    let run = placed_package
        .run_stream_tick_in_process(&device, 0)
        .unwrap();
    assert_eq!(
        run.status,
        VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed
    );
    assert_eq!(run.scheduler_turn_count, 2);
    assert_eq!(run.completed_stage_delta, 204);
    assert_eq!(run.transport_stats.pending_packet_count, 0);
    assert_eq!(run.transport_stats.pending_direct_edge_count, 0);
    assert_eq!(run.transport_stats.published_packet_count, 0);
    assert_eq!(run.transport_stats.received_packet_count, 0);
    assert_eq!(run.transport_stats.direct_copy_count, 2);
    assert_eq!(run.transport_stats.direct_copy_byte_count, 4096);
    assert_eq!(run.transport_stats.direct_receive_count, 2);
    assert_eq!(run.transport_stats.direct_receive_byte_count, 4096);

    let output = placed_package
        .mounted_device("gpu0")
        .unwrap()
        .boundary_io
        .output_buffer("output_frame")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(output, vec![0; 16]);
}

#[test]
fn placed_model_package_runs_split_single_token_tick_and_sampler() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed model package split token tick: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();

    let placed_model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_devices(
            &device,
            manifest_dir,
            runtime_model,
            Some(4),
        )
        .unwrap(),
    );
    let placed_package = placed_model
        .create_stream_processor_for_devices(&device, 0)
        .unwrap();
    assert_eq!(placed_model.input_device_id, "gpu0");
    assert_eq!(placed_model.output_device_id, "gpu1");
    assert_eq!(placed_model.transducer_parameter_count, 3);
    assert_eq!(
        placed_model.transducer_parameter_bytes,
        2 * FIXTURE_MODEL_EMBED_TOKENS_BYTES + 2_048
    );

    let run = placed_package
        .sample_token_id_stream_tick_in_process(&device, 1, 0)
        .unwrap();
    assert_eq!(run.tick_run.input_device_id, "gpu0");
    assert_eq!(run.tick_run.output_device_id, "gpu1");
    assert_eq!(run.tick_run.token_id, 1);
    assert_eq!(run.tick_run.stream_tick, 0);
    assert_eq!(run.tick_run.input_run.dispatch_count, 1);
    assert_eq!(
        run.tick_run.placed_run.status,
        VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed
    );
    assert_eq!(run.tick_run.placed_run.completed_stage_delta, 204);
    assert_eq!(run.tick_run.output_run.as_ref().unwrap().dispatch_count, 2);
    assert_eq!(run.sampler_run.descriptor_count, 3);
    assert_eq!(run.sampler_run.token_id, 1);
    assert_eq!(run.sampler_run.selected_logit_bits, 1_100_541_195);

    let input_frame = placed_package
        .mounted_device("gpu0")
        .unwrap()
        .boundary_io
        .input_buffer(FIXTURE_MODEL_INPUT_FRAME_SIGNAL)
        .unwrap();
    assert_eq!(
        input_frame
            .buffer
            .read_bytes(FIXTURE_MODEL_FRAME_BYTES)
            .unwrap(),
        fixture_model_embedding_row_bytes(&tensor_index, 1)
    );
}

#[test]
fn placed_model_package_runs_split_greedy_feedback_loop() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed model package split feedback loop: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let placed_model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_devices(
            &device,
            manifest_dir,
            runtime_model,
            Some(4),
        )
        .unwrap(),
    );
    let placed_package = placed_model
        .create_stream_processor_for_devices(&device, 0)
        .unwrap();
    let run = placed_package
        .run_feedback_bounded_in_process(&device, 1, 0, 2)
        .unwrap();

    assert_eq!(run.input_device_id, "gpu0");
    assert_eq!(run.output_device_id, "gpu1");
    assert_eq!(run.initial_token_id, 1);
    assert_eq!(run.sampled_token_ids, vec![1, 1]);
    assert_eq!(run.tick_runs.len(), 2);
    assert_eq!(run.tick_runs[0].stream_tick, 0);
    assert_eq!(run.tick_runs[0].input_token_id, 1);
    assert_eq!(run.tick_runs[0].sampled_token_id, 1);
    assert_eq!(run.tick_runs[1].stream_tick, 1);
    assert_eq!(run.tick_runs[1].input_token_id, 1);
    assert_eq!(run.tick_runs[1].sampled_token_id, 1);
    assert_eq!(
        run.tick_runs
            .iter()
            .map(|tick| tick.tick_run.sampler_run.selected_logit_bits)
            .collect::<Vec<_>>(),
        vec![1_100_541_195, 1_101_457_177]
    );
    assert_eq!(
        run.tick_runs
            .iter()
            .map(|tick| tick.tick_run.tick_run.placed_run.completed_stage_delta)
            .collect::<Vec<_>>(),
        vec![204, 204]
    );
}

#[test]
fn placed_model_package_drains_split_prompt_event_before_feedback() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed model package split prompt event: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let device = Rc::new(device);
    let devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device),
    ]);
    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(4),
            0,
            0,
        )
        .unwrap();
    let submitted = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "split_prompt",
            vec![1, 36_309],
            1,
        ))
        .unwrap();
    let run = &submitted.session_run.run;

    assert_eq!(run.input_device_id, "gpu0");
    assert_eq!(run.output_device_id, "gpu1");
    assert_eq!(run.prompt_token_ids, vec![1, 36_309]);
    assert_eq!(run.generated_token_ids.len(), 1);
    assert_eq!(
        run.output_token_ids,
        vec![1, 36_309, run.generated_token_ids[0]]
    );
    assert_eq!(run.stop_reason, "max_new_tokens");
    assert_eq!(run.tick_count, 3);
    assert_eq!(run.scheduler_turn_count, 6);
    assert_eq!(run.completed_stage_count, 612);
    assert_eq!(run.output_source_stream_ticks, vec![1]);
    assert_eq!(run.transport_stats.pending_packet_count, 0);
    assert_eq!(run.transport_stats.pending_direct_edge_count, 0);
    assert_eq!(run.transport_stats.direct_copy_count, 6);
    assert_eq!(run.transport_stats.direct_copy_byte_count, 12_288);
    assert_eq!(run.transport_stats.direct_receive_count, 6);
    assert_eq!(run.transport_stats.direct_receive_byte_count, 12_288);
}

