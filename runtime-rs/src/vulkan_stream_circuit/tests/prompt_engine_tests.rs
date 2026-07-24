#[test]
fn placed_prompt_engine_owns_streams_and_submits_input_events() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        0,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let added = engine.add_stream("main", stream).unwrap();
    assert_eq!(added.stream_id, "main");
    assert_eq!(added.pending_input_event_count, 0);
    assert!(added.idle);
    assert_eq!(engine.snapshot().stream_count, 1);
    assert!(engine.snapshot().idle);

    let mut streamed_output_events = Vec::new();
    let submitted = engine
        .submit_input_event_until_idle_with_output(
            "main",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
            |event| streamed_output_events.push(event),
        )
        .unwrap();

    assert_eq!(submitted.stream_id, "main");
    assert_eq!(submitted.input_event_id, "event_a");
    assert_eq!(submitted.queued_input_event.stream_id, "main");
    assert_eq!(
        submitted
            .queued_input_event
            .queued_input_event
            .pending_input_event_count,
        1
    );
    assert_eq!(submitted.output_events.len(), 1);
    assert_eq!(submitted.output_events[0].stream_id, "main");
    assert_eq!(
        submitted.output_events[0].output_event.input_event_id,
        "event_a"
    );
    assert_eq!(
        submitted.output_events[0].output_event.source_stream_tick,
        0
    );
    assert_eq!(submitted.generated_token_ids.len(), 1);
    assert_eq!(streamed_output_events, submitted.output_events);
    assert_eq!(submitted.engine_run.processed_input_event_count, 1);
    assert_eq!(
        submitted.engine_run.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(submitted.engine_run.input_runs.len(), 1);
    assert_eq!(
        submitted.engine_run.input_runs[0]
            .submitted_run
            .input_event
            .id,
        "event_a"
    );
    assert_eq!(
        submitted.engine_run.end_snapshot.streams[0].next_stream_tick,
        2
    );

    let snapshot = engine.snapshot();
    assert!(snapshot.idle);
    assert_eq!(snapshot.streams[0].next_stream_tick, 2);
    assert_eq!(snapshot.streams[0].completed_prompt_event_count, 1);
}

#[test]
fn placed_prompt_engine_returns_completion_from_a_boundary_closing_drain() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine closing-drain test: {error}");
            return;
        }
    };
    let manifest_path = fixture_model_package_manifest_path();
    let devices = BTreeMap::from([(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
        Rc::new(device),
    )]);
    let stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_path.parent().unwrap(),
            fixture_model_runtime_model(),
            Some(64),
            0,
            0,
        )
        .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();

    let submitted = engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("boundary", vec![1], 3),
        )
        .unwrap();

    assert_eq!(submitted.generated_token_ids.len(), 3);
    assert_eq!(submitted.engine_run.input_runs.len(), 1);
    assert_eq!(
        submitted.engine_run.input_runs[0]
            .submitted_run
            .session_run
            .run
            .stop_reason,
        "max_new_tokens"
    );
    assert!(submitted.engine_run.end_snapshot.idle);
    assert!(engine.snapshot().idle);
}

#[test]
fn placed_prompt_engine_single_submit_runs_the_engine_queue() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine single-submit queue test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);
    let model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            runtime_model,
            Some(64),
            false,
        )
        .unwrap(),
    );
    let stream_a =
        VulkanResidentInProcessPlacedPromptStream::new(model.clone(), devices.clone(), 0).unwrap();
    let stream_b =
        VulkanResidentInProcessPlacedPromptStream::new(model.clone(), devices, 1).unwrap();
    assert!(Arc::ptr_eq(&stream_a.package, &stream_b.package));
    assert!(Arc::ptr_eq(
        &stream_a.processor.device_slices[0]
            .package_slice
            .parameter_buffers,
        &stream_b.processor.device_slices[0]
            .package_slice
            .parameter_buffers,
    ));
    assert!(!std::ptr::eq(
        &stream_a.processor.device_slices[0]
            .mounted
            .buffers
            .state_buffers[0]
            .buffer,
        &stream_b.processor.device_slices[0]
            .mounted
            .buffers
            .state_buffers[0]
            .buffer,
    ));

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![4], 1),
        )
        .unwrap();

    let submitted = engine
        .submit_input_event_until_idle(
            "stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![5], 1),
        )
        .unwrap();

    assert_eq!(submitted.output_events.len(), 1);
    assert_eq!(submitted.output_events[0].stream_id, "stream_a");
    assert_eq!(submitted.engine_run.processed_input_event_count, 2);
    assert_eq!(submitted.engine_run.input_runs.len(), 2);
    assert!(
        submitted.engine_run.physical_multi_stream_batch_count > 0,
        "shared-package streams must execute as physical Vulkan batches"
    );
    assert_eq!(
        submitted.engine_run.max_physical_multi_stream_batch_width,
        2
    );
    assert_eq!(submitted.engine_run.input_runs[0].stream_id, "stream_b");
    assert_eq!(
        submitted.engine_run.input_runs[0]
            .submitted_run
            .input_event
            .id,
        "event_b"
    );
    assert_eq!(submitted.engine_run.input_runs[1].stream_id, "stream_a");
    assert_eq!(
        submitted.engine_run.input_runs[1]
            .submitted_run
            .input_event
            .id,
        "event_a"
    );
    assert!(submitted.engine_run.end_snapshot.idle);
}

#[test]
fn placed_prompt_engine_batches_fairly_and_cancels_between_physical_batches() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine batch cancellation test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device)]);
    let model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            runtime_model,
            Some(64),
            false,
        )
        .unwrap(),
    );
    let short =
        VulkanResidentInProcessPlacedPromptStream::new(model.clone(), devices.clone(), 0).unwrap();
    let long = VulkanResidentInProcessPlacedPromptStream::new(model, devices, 1).unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("short", short).unwrap();
    engine.add_stream("long", long).unwrap();
    engine
        .enqueue_input_event(
            "short",
            VulkanResidentTokenInputEvent::new("short_event", vec![4], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "long",
            VulkanResidentTokenInputEvent::new("long_event", vec![5], 5),
        )
        .unwrap();

    let first_completion = engine.run_until_idle_bounded(1).unwrap();

    assert_eq!(first_completion.processed_input_event_count, 1);
    assert_eq!(first_completion.input_runs[0].stream_id, "short");
    assert!(first_completion.physical_multi_stream_batch_count > 0);
    assert_eq!(first_completion.max_physical_multi_stream_batch_width, 2);
    assert_eq!(
        first_completion
            .output_events
            .iter()
            .filter(|event| event.stream_id == "short")
            .count(),
        1
    );
    assert_eq!(
        first_completion
            .output_events
            .iter()
            .filter(|event| event.stream_id == "long")
            .count(),
        1
    );
    assert_eq!(first_completion.end_snapshot.active_stream_ids, ["long"]);

    let cancellation = engine.interrupt_stream("long", "test cancellation").unwrap();
    assert!(cancellation.stream_control_run.completed_input_run.is_some());
    assert!(engine.snapshot().idle);
}

#[test]
fn placed_prompt_engine_runs_queued_streams_until_idle() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine run-until-idle test: {error}");
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
        ("gpu1".to_string(), device.clone()),
    ]);

    let stream_a = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model.clone(),
        Some(8),
        0,
        0,
    )
    .unwrap();
    let stream_b = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(8),
        1,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![36_309], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b_repeat", vec![36_309], 1),
        )
        .unwrap();
    let queued_snapshot = engine.snapshot();
    assert!(!queued_snapshot.idle);
    assert_eq!(queued_snapshot.active_stream_count, 2);
    assert_eq!(
        queued_snapshot.active_stream_ids,
        vec!["stream_a".to_string(), "stream_b".to_string()]
    );

    let run = engine.run_until_idle_bounded(3).unwrap();

    assert_eq!(
        run.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(run.processed_input_event_count, 3);
    assert_eq!(run.input_runs.len(), 3);
    assert_eq!(run.input_runs[0].stream_id, "stream_b");
    assert_eq!(run.input_runs[0].submitted_run.input_event.id, "event_b");
    assert_eq!(run.input_runs[1].stream_id, "stream_a");
    assert_eq!(run.input_runs[1].submitted_run.input_event.id, "event_a");
    assert_eq!(run.input_runs[2].stream_id, "stream_b");
    assert_eq!(
        run.input_runs[2].submitted_run.input_event.id,
        "event_b_repeat"
    );
    assert_eq!(run.output_events.len(), 3);
    assert_eq!(run.output_events[0].stream_id, "stream_b");
    assert_eq!(run.output_events[0].output_event.source_stream_tick, 0);
    assert_eq!(run.output_events[1].stream_id, "stream_a");
    assert_eq!(run.output_events[1].output_event.source_stream_tick, 0);
    assert_eq!(run.output_events[2].stream_id, "stream_b");
    assert_eq!(run.output_events[2].output_event.source_stream_tick, 2);
    assert_eq!(run.generated_token_ids.len(), 3);
    assert!(!run.start_snapshot.idle);
    assert!(run.end_snapshot.idle);
    assert_eq!(run.end_snapshot.active_stream_count, 0);
    assert_eq!(run.end_snapshot.streams[0].stream_id, "stream_a");
    assert_eq!(run.end_snapshot.streams[0].next_stream_tick, 2);
    assert_eq!(run.end_snapshot.streams[1].stream_id, "stream_b");
    assert_eq!(run.end_snapshot.streams[1].next_stream_tick, 4);
}

#[test]
fn placed_prompt_engine_batches_input_events_across_streams() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine batch test: {error}");
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
        ("gpu1".to_string(), device.clone()),
    ]);

    let stream_a = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model.clone(),
        Some(8),
        0,
        0,
    )
    .unwrap();
    let stream_b = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(8),
        1,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();

    let batch = engine
        .submit_input_events_until_idle_bounded(
            vec![
                VulkanResidentInProcessPlacedPromptEngineInputRequest::new(
                    "stream_b",
                    VulkanResidentTokenInputEvent::new("event_b", vec![36_309], 1),
                ),
                VulkanResidentInProcessPlacedPromptEngineInputRequest::new(
                    "stream_a",
                    VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
                ),
            ],
            2,
        )
        .unwrap();

    assert_eq!(batch.queued_input_events.len(), 2);
    assert_eq!(batch.queued_input_events[0].stream_id, "stream_b");
    assert_eq!(batch.queued_input_events[1].stream_id, "stream_a");
    assert_eq!(
        batch.engine_run.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(batch.engine_run.input_runs.len(), 2);
    assert_eq!(batch.engine_run.processed_input_event_count, 2);
    assert_eq!(batch.engine_run.input_runs[0].stream_id, "stream_b");
    assert_eq!(
        batch.engine_run.input_runs[0].submitted_run.input_event.id,
        "event_b"
    );
    assert_eq!(batch.engine_run.input_runs[1].stream_id, "stream_a");
    assert_eq!(
        batch.engine_run.input_runs[1].submitted_run.input_event.id,
        "event_a"
    );
    assert_eq!(batch.output_events.len(), 2);
    assert_eq!(batch.output_events[0].stream_id, "stream_b");
    assert_eq!(batch.output_events[1].stream_id, "stream_a");
    assert_eq!(batch.generated_token_ids.len(), 2);
    assert!(engine.snapshot().idle);
}

#[test]
fn placed_prompt_engine_overlaps_resident_feedback_windows_across_streams() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine asynchronous feedback test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);
    let stream_a = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model.clone(),
        Some(64),
        0,
        0,
    )
    .unwrap();
    let stream_b = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        1,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();
    engine
        .enqueue_input_event(
            "stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 5),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![1], 5),
        )
        .unwrap();

    let run = engine.run_until_idle_bounded(2).unwrap();

    assert_eq!(run.processed_input_event_count, 2);
    assert_eq!(run.max_pending_activation_count, 2);
    for input_run in &run.input_runs {
        let feedback = input_run
            .submitted_run
            .session_run
            .run
            .resident_feedback;
        assert!(feedback.asynchronous_submission_count > 0);
        assert!(feedback.completion_poll_count > 0);
        assert!(feedback.bounded_wait_count > 0);
        assert_eq!(
            feedback.asynchronous_submission_count,
            feedback.window_count
        );
    }
    assert!(run.end_snapshot.idle);
}

#[test]
fn placed_prompt_engine_preserves_queued_work_at_input_event_budget() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine budget test: {error}");
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
        ("gpu1".to_string(), device.clone()),
    ]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(8),
        0,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    engine
        .enqueue_input_event(
            "main",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "main",
            VulkanResidentTokenInputEvent::new("event_b", vec![36_309], 1),
        )
        .unwrap();

    let budgeted = engine.run_until_idle_bounded(1).unwrap();

    assert_eq!(
        budgeted.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::InputEventBudget
    );
    assert_eq!(budgeted.processed_input_event_count, 1);
    assert_eq!(budgeted.input_runs.len(), 1);
    assert_eq!(
        budgeted.input_runs[0].submitted_run.input_event.id,
        "event_a"
    );
    assert_eq!(budgeted.output_events.len(), 1);
    assert_eq!(
        budgeted.output_events[0].output_event.input_event_id,
        "event_a"
    );
    assert!(!budgeted.end_snapshot.idle);
    assert_eq!(
        budgeted.end_snapshot.streams[0].pending_input_event_count,
        1
    );
    assert_eq!(budgeted.end_snapshot.streams[0].next_stream_tick, 2);

    let completed_b = engine.run_until_idle_bounded(1).unwrap();
    assert_eq!(
        completed_b.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(completed_b.processed_input_event_count, 1);
    assert_eq!(
        completed_b.input_runs[0].submitted_run.input_event.id,
        "event_b"
    );
    assert_eq!(completed_b.output_events.len(), 1);
    assert_eq!(
        completed_b.output_events[0].output_event.input_event_id,
        "event_b"
    );
    assert!(completed_b.end_snapshot.idle);
    assert_eq!(completed_b.end_snapshot.streams[0].next_stream_tick, 4);
}

#[test]
fn placed_model_package_runs_runtime_graphed_duplicate_layer() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed model package duplicate layer runtime graph: {error}");
            return;
        }
    };
    let manifest = fixture_model_package_manifest();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(manifest_dir)
        .unwrap();
    let runtime_graph = StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0")
        .unwrap()
        .duplicate_after_instance(&source_graph, "layer_05", "layer_05_repeat")
        .unwrap()
        .with_instance_device("layer_05_repeat", "gpu1")
        .unwrap();
    let runtime_model = manifest.mount_runtime_graph(&runtime_graph).unwrap();

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
    assert_eq!(placed_model.hosted_component_count, 15);
    assert_eq!(placed_package.device("gpu1").unwrap().hosted_component_count, 1);

    let run = placed_package
        .sample_token_id_stream_tick_in_process(&device, 1, 0)
        .unwrap();

    assert_eq!(
        run.tick_run.placed_run.status,
        VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed
    );
    assert!(run.tick_run.placed_run.completed_stage_delta > 204);
    assert_eq!(run.sampler_run.descriptor_count, 5);
}
