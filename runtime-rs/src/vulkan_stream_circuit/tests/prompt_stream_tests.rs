#[test]
fn placed_prompt_stream_owns_package_devices_and_session() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt stream test: {error}");
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

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();

    assert_eq!(stream.package().input_device_id, "gpu0");
    assert_eq!(stream.package().output_device_id, "gpu1");
    assert_eq!(stream.devices().len(), 2);
    assert_eq!(stream.next_stream_tick(), 0);

    let first = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("event_a", vec![1], 1))
        .unwrap();
    assert_eq!(first.session_run.prompt_event_index, 0);
    assert_eq!(first.session_run.start_stream_tick, 0);
    assert_eq!(first.session_run.next_stream_tick, 2);
    assert_eq!(stream.next_stream_tick(), 2);
    assert_eq!(stream.completed_prompt_event_count(), 1);

    let second = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "event_b",
            vec![36_309],
            1,
        ))
        .unwrap();
    assert_eq!(second.session_run.prompt_event_index, 1);
    assert_eq!(second.session_run.start_stream_tick, 2);
    assert_eq!(second.session_run.next_stream_tick, 4);
    assert_eq!(stream.next_stream_tick(), 4);
    assert_eq!(stream.completed_prompt_event_count(), 2);
    assert_eq!(second.session_run.run.output_source_stream_ticks, vec![2]);
}

#[test]
fn placed_prompt_stream_runs_resident_feedback_across_bridged_slices() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed resident feedback-window test: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
        device.clone(),
    )]);

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();
    assert!(stream.processor.resident_feedback_window_width() >= 3);

    let mut first_streamed_output_events = Vec::new();
    stream.enqueue_input_event(
        VulkanResidentTokenInputEvent::new("event", vec![1], 8).with_stop_tokens(vec![558]),
    );
    let resident_first = stream
        .run_next_queued_input_event_with_output(|event| first_streamed_output_events.push(event))
        .unwrap()
        .unwrap();

    assert_eq!(resident_first.generated_token_ids, vec![1, 1, 558]);
    assert_eq!(resident_first.output_events, first_streamed_output_events);
    assert_eq!(
        resident_first
            .output_events
            .iter()
            .map(|event| event.source_stream_tick)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(resident_first.session_run.run.stop_reason, "eos");
    assert_eq!(resident_first.session_run.tick_count, 4);
    assert_eq!(resident_first.session_run.next_stream_tick, 4);
    assert_eq!(
        resident_first.session_run.run.resident_feedback,
        VulkanResidentFeedbackExecutionStats {
            window_count: 1,
            planned_tick_count: 7,
            submitted_tick_count: 7,
            executed_tick_count: 3,
            retained_tick_count: 3,
            sampled_tick_count: 2,
            discarded_tick_count: 4,
            template_record_count: 1,
            template_replay_count: 0,
        }
    );

    let resident_second = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "event_after_eos",
            vec![36_309],
            3,
        ))
        .unwrap();
    assert_eq!(
        resident_second
            .output_events
            .iter()
            .map(|event| event.source_stream_tick)
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );
    assert_eq!(
        resident_second.session_run.run.stop_reason,
        "max_new_tokens"
    );
    assert_eq!(resident_second.session_run.tick_count, 4);
    assert_eq!(resident_second.session_run.next_stream_tick, 8);
    assert_eq!(stream.next_stream_tick(), 8);
    assert!(stream.is_idle());
    drop(stream);

    let bridged_runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0").with_component_device("layer_02", "gpu1"),
    );
    let bridged_devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device),
    ]);
    let mut bridged_stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            bridged_devices,
            manifest_dir,
            bridged_runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();
    assert!(bridged_stream.processor.resident_feedback_window_width() >= 3);
    let bridged_first = bridged_stream
        .submit_input_event(
            VulkanResidentTokenInputEvent::new("event", vec![1], 8).with_stop_tokens(vec![558]),
        )
        .unwrap();
    let bridged_second = bridged_stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "event_after_eos",
            vec![36_309],
            3,
        ))
        .unwrap();

    assert_eq!(
        resident_first.generated_token_ids,
        bridged_first.generated_token_ids
    );
    assert_eq!(resident_first.output_events, bridged_first.output_events);
    assert_eq!(
        resident_first.session_run.run.output_token_ids,
        bridged_first.session_run.run.output_token_ids
    );
    assert_eq!(
        resident_first.session_run.run.stop_reason,
        bridged_first.session_run.run.stop_reason
    );
    assert_eq!(
        resident_first.session_run.run.output_source_stream_ticks,
        bridged_first.session_run.run.output_source_stream_ticks
    );
    assert_eq!(
        resident_second.generated_token_ids,
        bridged_second.generated_token_ids
    );
    assert_eq!(resident_second.output_events, bridged_second.output_events);
    assert_eq!(
        resident_second.session_run.run.output_token_ids,
        bridged_second.session_run.run.output_token_ids
    );
    assert_eq!(
        resident_second.session_run.run.stop_reason,
        bridged_second.session_run.run.stop_reason
    );
    assert_eq!(
        resident_second.session_run.run.output_source_stream_ticks,
        bridged_second.session_run.run.output_source_stream_ticks
    );
    assert_eq!(resident_first.session_run.run.scheduler_turn_count, 4);
    assert_eq!(bridged_first.session_run.run.scheduler_turn_count, 8);
    assert_eq!(resident_second.session_run.run.scheduler_turn_count, 4);
    assert_eq!(bridged_second.session_run.run.scheduler_turn_count, 8);
    assert_eq!(
        resident_first.session_run.run.transport_stats,
        VulkanPlacedEdgeTransportStats::default()
    );
    assert_eq!(
        resident_second.session_run.run.transport_stats,
        VulkanPlacedEdgeTransportStats::default()
    );
    assert_eq!(
        bridged_first
            .session_run
            .run
            .transport_stats
            .direct_copy_count,
        2
    );
    assert_eq!(
        bridged_second
            .session_run
            .run
            .transport_stats
            .direct_copy_count,
        4
    );
    assert!(bridged_stream.is_idle());
}

#[test]
fn placed_prompt_stream_reuses_full_width_feedback_submission_template() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed feedback-template replay test: {error}");
            return;
        }
    };
    let manifest_path = fixture_model_package_manifest_path();
    let devices = BTreeMap::from([(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
        Rc::new(device),
    )]);
    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_path.parent().unwrap(),
            fixture_model_runtime_model(),
            Some(4),
            0,
            0,
        )
        .unwrap();
    assert_eq!(stream.processor.resident_feedback_window_width(), 4);

    let first = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("first", vec![1], 5))
        .unwrap();
    let second = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("second", vec![1], 5))
        .unwrap();

    assert_eq!(
        first.session_run.run.resident_feedback,
        VulkanResidentFeedbackExecutionStats {
            window_count: 1,
            planned_tick_count: 4,
            submitted_tick_count: 4,
            executed_tick_count: 4,
            retained_tick_count: 4,
            sampled_tick_count: 4,
            discarded_tick_count: 0,
            template_record_count: 1,
            template_replay_count: 0,
        }
    );
    assert_eq!(
        second.session_run.run.resident_feedback,
        VulkanResidentFeedbackExecutionStats {
            template_record_count: 0,
            template_replay_count: 1,
            ..first.session_run.run.resident_feedback
        }
    );
}

#[test]
fn placed_prompt_stream_queues_input_events_and_emits_output_events() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt stream queue test: {error}");
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

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();

    let queued_a =
        stream.enqueue_input_event(VulkanResidentTokenInputEvent::new("event_a", vec![1], 1));
    assert_eq!(queued_a.pending_input_event_count, 1);
    assert_eq!(queued_a.next_stream_tick, 0);
    let queued_b = stream.enqueue_input_event(VulkanResidentTokenInputEvent::new(
        "event_b",
        vec![36_309],
        1,
    ));
    assert_eq!(queued_b.pending_input_event_count, 2);
    assert_eq!(stream.pending_input_event_count(), 2);
    assert!(!stream.is_idle());

    let mut streamed_output_events = Vec::new();
    let first = stream
        .run_next_queued_input_event_with_output(|event| streamed_output_events.push(event))
        .unwrap()
        .unwrap();
    assert_eq!(first.input_event.id, "event_a");
    assert_eq!(first.pending_input_event_count, 1);
    assert_eq!(first.generated_token_ids.len(), 1);
    assert_eq!(first.output_events.len(), 1);
    assert_eq!(first.output_events[0].input_event_id, "event_a");
    assert_eq!(first.output_events[0].output_index, 0);
    assert_eq!(first.output_events[0].source_stream_tick, 0);
    assert_eq!(streamed_output_events, first.output_events);
    assert_eq!(stream.next_stream_tick(), 2);

    let second = stream.run_next_queued_input_event().unwrap().unwrap();
    assert_eq!(second.input_event.id, "event_b");
    assert_eq!(second.pending_input_event_count, 0);
    assert_eq!(second.generated_token_ids.len(), 1);
    assert_eq!(second.output_events.len(), 1);
    assert_eq!(second.output_events[0].input_event_id, "event_b");
    assert_eq!(second.output_events[0].source_stream_tick, 2);
    assert_eq!(stream.next_stream_tick(), 4);
    assert!(stream.is_idle());

    let idle = stream.run_next_queued_input_event().unwrap();
    assert!(idle.is_none());
}

#[test]
fn placed_prompt_stream_drains_queued_input_events_until_idle() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt stream drain test: {error}");
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

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();

    let run = stream
        .submit_input_events_until_idle(vec![
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
            VulkanResidentTokenInputEvent::new("event_b", vec![36_309], 1),
        ])
        .unwrap();

    assert_eq!(run.start_stream_tick, 0);
    assert_eq!(run.next_stream_tick, 4);
    assert_eq!(run.tick_count, 4);
    assert_eq!(run.submitted_runs.len(), 2);
    assert_eq!(run.output_events.len(), 2);
    assert_eq!(run.generated_token_ids.len(), 2);
    assert_eq!(run.pending_input_event_count, 0);
    assert_eq!(run.output_events[0].input_event_id, "event_a");
    assert_eq!(run.output_events[0].source_stream_tick, 0);
    assert_eq!(run.output_events[1].input_event_id, "event_b");
    assert_eq!(run.output_events[1].source_stream_tick, 2);
    assert!(stream.is_idle());
}
