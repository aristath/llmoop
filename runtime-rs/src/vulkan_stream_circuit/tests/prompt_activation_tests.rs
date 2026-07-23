#[test]
fn temporal_prompt_block_matches_scalar_ticks_and_component_state() {
    let Some(manifest_path) = std::env::var_os("NERVE_TEMPORAL_TEST_PACKAGE").map(PathBuf::from)
    else {
        eprintln!("skipping temporal prompt equivalence: NERVE_TEMPORAL_TEST_PACKAGE is unset");
        return;
    };
    let device_index = std::env::var("NERVE_TEMPORAL_TEST_VULKAN_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let device = match VulkanComputeDevice::new_for_physical_device_index(device_index) {
        Ok(device) => Rc::new(device),
        Err(error) => {
            eprintln!("skipping temporal prompt equivalence: {error}");
            return;
        }
    };
    let manifest = VulkanResidentModelPackageManifest::from_json_file(&manifest_path).unwrap();
    let state_dtypes = manifest
        .circuit_graph
        .components
        .iter()
        .flat_map(|component| {
            component.state.state_ports.iter().map(|state| {
                (
                    (component.component_id.clone(), state.id.clone()),
                    state
                        .extra
                        .get("dtype")
                        .and_then(serde_json::Value::as_str)
                        .unwrap()
                        .to_string(),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let runtime_model = manifest
        .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
        .unwrap();
    let logical_devices = runtime_model
        .circuit_graph
        .signal_processor_device_ids(&runtime_model.placement);
    let devices = logical_devices
        .into_iter()
        .map(|device_id| (device_id, device.clone()))
        .collect::<BTreeMap<_, _>>();
    let package = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_path.parent().unwrap(),
            runtime_model,
            Some(256),
            false,
        )
        .unwrap(),
    );
    let mut scalar = VulkanResidentInProcessPlacedPromptStream::from_package_devices_and_session(
        package.clone(),
        devices.clone(),
        0,
        0,
    )
    .unwrap();
    let mut temporal = VulkanResidentInProcessPlacedPromptStream::from_package_devices_and_session(
        package.clone(),
        devices.clone(),
        0,
        0,
    )
    .unwrap();
    scalar.speculative_draft_tokens = 0;
    temporal.speculative_draft_tokens = 0;
    let event = VulkanResidentTokenInputEvent::new(
        "temporal-equivalence",
        vec![760, 6_511, 314, 9_338, 369],
        1,
    );

    scalar.enqueue_input_event(event.clone());
    let scalar_run = loop {
        let activation = scalar.run_next_activation().unwrap().unwrap();
        if let Some(run) = activation.completed_input_run {
            break run;
        }
    };
    let temporal_run = temporal.submit_input_event(event).unwrap();

    assert_eq!(
        temporal_run.generated_token_ids,
        scalar_run.generated_token_ids
    );
    assert_eq!(
        temporal_run.session_run.run.output_source_stream_ticks,
        scalar_run.session_run.run.output_source_stream_ticks
    );
    for (temporal_slice, scalar_slice) in temporal
        .processor
        .device_slices
        .iter()
        .zip(&scalar.processor.device_slices)
    {
        assert_eq!(
            temporal_slice.mounted.buffers.state_buffers.len(),
            scalar_slice.mounted.buffers.state_buffers.len()
        );
        for (temporal_state, scalar_state) in temporal_slice
            .mounted
            .buffers
            .state_buffers
            .iter()
            .zip(&scalar_slice.mounted.buffers.state_buffers)
        {
            assert_eq!(temporal_state.component_id, scalar_state.component_id);
            assert_eq!(temporal_state.state_id, scalar_state.state_id);
            let temporal_bytes = temporal_state
                .buffer
                .read_bytes(temporal_state.byte_capacity)
                .unwrap();
            let scalar_bytes = scalar_state
                .buffer
                .read_bytes(scalar_state.byte_capacity)
                .unwrap();
            if temporal_bytes != scalar_bytes {
                let key = (
                    temporal_state.component_id.clone(),
                    temporal_state.state_id.clone(),
                );
                let dtype = state_dtypes.get(&key).unwrap();
                let (relative_rmse, max_absolute_error) =
                    numeric_state_error(&temporal_bytes, &scalar_bytes, dtype);
                eprintln!(
                    "state error {}.{} ({dtype}): relative_rmse={relative_rmse:.6}, max_absolute_error={max_absolute_error:.6}",
                    temporal_state.component_id, temporal_state.state_id
                );
                assert!(
                    relative_rmse.is_finite() && relative_rmse < 0.02,
                    "state mismatch in {}.{} has relative RMSE {relative_rmse}",
                    temporal_state.component_id,
                    temporal_state.state_id,
                );
            }
        }
    }

    let mut scalar = VulkanResidentInProcessPlacedPromptStream::from_package_devices_and_session(
        package.clone(),
        devices.clone(),
        0,
        0,
    )
    .unwrap();
    let mut temporal = VulkanResidentInProcessPlacedPromptStream::from_package_devices_and_session(
        package, devices, 0, 0,
    )
    .unwrap();
    scalar.speculative_draft_tokens = 0;
    temporal.speculative_draft_tokens = 0;
    let token_pattern = [760, 6_511, 314, 9_338, 369];
    let prompt_tokens = (0..64)
        .map(|index| token_pattern[index % token_pattern.len()])
        .collect();
    let event = VulkanResidentTokenInputEvent::new("full-temporal-equivalence", prompt_tokens, 8);

    scalar.enqueue_input_event(event.clone());
    let scalar_run = loop {
        let activation = scalar.run_next_activation().unwrap().unwrap();
        if let Some(run) = activation.completed_input_run {
            break run;
        }
    };
    let temporal_run = temporal.submit_input_event(event).unwrap();

    assert_eq!(
        temporal_run.generated_token_ids,
        scalar_run.generated_token_ids
    );
    assert_eq!(
        temporal_run.session_run.run.output_source_stream_ticks,
        scalar_run.session_run.run.output_source_stream_ticks
    );
}

#[test]
fn placed_active_prompt_event_advances_one_activation_at_a_time() {
    let input_event =
        VulkanResidentTokenInputEvent::new("event", vec![10, 20], 2).with_stop_tokens(vec![99]);
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(input_event, 5).unwrap();

    let first = active.next_activation().unwrap();
    assert_eq!(first.input_token_id, 10);
    assert!(!first.input_is_feedback);
    assert!(!first.should_emit_public_output);
    assert_eq!(
        active
            .complete_activation(
                &first,
                5,
                3,
                7,
                &VulkanPlacedEdgeTransportStats::default(),
                None,
            )
            .unwrap(),
        None
    );
    assert!(!active.is_complete());

    let second = active.next_activation().unwrap();
    assert_eq!(second.input_token_id, 20);
    assert!(!second.input_is_feedback);
    assert!(second.should_emit_public_output);
    let first_output = active
        .complete_activation(
            &second,
            6,
            3,
            7,
            &VulkanPlacedEdgeTransportStats::default(),
            Some(30),
        )
        .unwrap()
        .unwrap();
    assert_eq!(first_output.id, "event.0");
    assert_eq!(first_output.token_id, 30);
    assert_eq!(first_output.source_stream_tick, 6);

    let third = active.next_activation().unwrap();
    assert_eq!(third.input_token_id, 30);
    assert!(third.input_is_feedback);
    assert_eq!(third.input_feedback_depth, 1);
    assert!(!third.input_closes_loop_after_processing);
    assert!(third.should_emit_public_output);
    let second_output = active
        .complete_activation(
            &third,
            7,
            3,
            7,
            &VulkanPlacedEdgeTransportStats::default(),
            Some(99),
        )
        .unwrap()
        .unwrap();
    assert_eq!(second_output.id, "event.1");
    assert_eq!(second_output.token_id, 99);

    let closing = active.next_activation().unwrap();
    assert_eq!(closing.input_token_id, 99);
    assert!(closing.input_is_feedback);
    assert_eq!(closing.input_feedback_depth, 2);
    assert!(closing.input_closes_loop_after_processing);
    assert!(!closing.should_emit_public_output);
    assert_eq!(
        active
            .complete_activation(
                &closing,
                8,
                3,
                7,
                &VulkanPlacedEdgeTransportStats::default(),
                None,
            )
            .unwrap(),
        None
    );
    assert!(active.is_complete());
    assert!(active.next_activation().is_none());

    let run = active.into_event_run("gpu0".to_string(), "gpu1".to_string());
    assert_eq!(run.prompt_token_ids, vec![10, 20]);
    assert_eq!(run.generated_token_ids, vec![30, 99]);
    assert_eq!(run.output_token_ids, vec![10, 20, 30, 99]);
    assert_eq!(run.output_source_stream_ticks, vec![6, 7]);
    assert_eq!(run.stop_reason, "eos");
    assert_eq!(run.tick_count, 4);
    assert_eq!(run.scheduler_turn_count, 12);
    assert_eq!(run.completed_stage_count, 28);
}

#[test]
fn placed_active_prompt_event_windows_only_open_private_feedback() {
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(
        VulkanResidentTokenInputEvent::new("event", vec![10, 11], 4),
        0,
    )
    .unwrap();
    assert_eq!(active.resident_feedback_window_tick_count(64), 0);

    let first_prompt = active.next_activation().unwrap();
    active
        .complete_activation(
            &first_prompt,
            0,
            1,
            1,
            &VulkanPlacedEdgeTransportStats::default(),
            None,
        )
        .unwrap();
    assert_eq!(active.resident_feedback_window_tick_count(64), 0);

    let final_prompt = active.next_activation().unwrap();
    active
        .complete_activation(
            &final_prompt,
            1,
            1,
            1,
            &VulkanPlacedEdgeTransportStats::default(),
            Some(20),
        )
        .unwrap();
    assert_eq!(active.resident_feedback_window_tick_count(64), 3);
    assert_eq!(active.resident_feedback_window_tick_count(2), 2);

    active.stop_after_current("user_stop");
    assert_eq!(active.resident_feedback_window_tick_count(64), 0);

    let mut eos = VulkanResidentInProcessPlacedActivePromptEvent::new(
        VulkanResidentTokenInputEvent::new("eos", vec![10], 8).with_stop_tokens(vec![20]),
        0,
    )
    .unwrap();
    let prompt = eos.next_activation().unwrap();
    eos.complete_activation(
        &prompt,
        0,
        1,
        1,
        &VulkanPlacedEdgeTransportStats::default(),
        Some(20),
    )
    .unwrap();
    assert_eq!(eos.resident_feedback_window_tick_count(64), 0);
}

#[test]
fn placed_active_prompt_event_exposes_repeated_full_feedback_windows() {
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(
        VulkanResidentTokenInputEvent::new("long-generation", vec![10], 130),
        0,
    )
    .unwrap();
    let prompt = active.next_activation().unwrap();
    active
        .complete_activation(
            &prompt,
            0,
            1,
            1,
            &VulkanPlacedEdgeTransportStats::default(),
            Some(20),
        )
        .unwrap();
    assert_eq!(active.resident_feedback_window_tick_count(64), 64);

    for stream_tick in 1..=64 {
        let activation = active.next_activation().unwrap();
        active
            .complete_activation(
                &activation,
                stream_tick,
                1,
                1,
                &VulkanPlacedEdgeTransportStats::default(),
                Some(20),
            )
            .unwrap();
    }
    assert_eq!(active.resident_feedback_window_tick_count(64), 64);

    for stream_tick in 65..=128 {
        let activation = active.next_activation().unwrap();
        active
            .complete_activation(
                &activation,
                stream_tick,
                1,
                1,
                &VulkanPlacedEdgeTransportStats::default(),
                Some(20),
            )
            .unwrap();
    }
    assert_eq!(active.resident_feedback_window_tick_count(64), 0);
    assert_eq!(active.remaining_public_outputs, 1);
}

#[test]
fn placed_active_prompt_event_without_output_drains_only_external_input() {
    let input_event = VulkanResidentTokenInputEvent::new("prefill", vec![10, 20], 0);
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(input_event, 12).unwrap();

    for stream_tick in [12, 13] {
        let activation = active.next_activation().unwrap();
        assert!(!activation.input_is_feedback);
        assert!(!activation.should_emit_public_output);
        assert!(
            active
                .complete_activation(
                    &activation,
                    stream_tick,
                    1,
                    2,
                    &VulkanPlacedEdgeTransportStats::default(),
                    None,
                )
                .unwrap()
                .is_none()
        );
    }

    assert!(active.is_complete());
    assert!(active.next_activation().is_none());
    let run = active.into_event_run("gpu0".to_string(), "gpu0".to_string());
    assert_eq!(run.prompt_token_ids, vec![10, 20]);
    assert!(run.generated_token_ids.is_empty());
    assert_eq!(run.output_token_ids, vec![10, 20]);
    assert_eq!(run.stop_reason, "max_new_tokens");
    assert_eq!(run.tick_count, 2);
}

#[test]
fn placed_active_prompt_event_interrupt_closes_feedback_without_losing_state() {
    let input_event = VulkanResidentTokenInputEvent::new("event", vec![10], 3);
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(input_event, 4).unwrap();
    let activation = active.next_activation().unwrap();
    let output = active
        .complete_activation(
            &activation,
            4,
            2,
            5,
            &VulkanPlacedEdgeTransportStats::default(),
            Some(20),
        )
        .unwrap()
        .unwrap();
    assert_eq!(output.token_id, 20);
    assert!(!active.is_complete());

    let control = active.interrupt("user_interrupt");
    assert_eq!(
        control.event_type,
        VulkanResidentStreamControlEventType::Interrupt
    );
    assert_eq!(control.reason, "user_interrupt");
    assert_eq!(
        control.cleared_private_feedback_ids,
        vec!["event.feedback.0"]
    );
    assert!(control.closing_private_feedback_id.is_none());
    assert!(control.state_preserved);
    assert!(active.is_complete());
    assert!(active.next_activation().is_none());

    let run = active.into_event_run("gpu0".to_string(), "gpu0".to_string());
    assert_eq!(run.prompt_token_ids, vec![10]);
    assert_eq!(run.generated_token_ids, vec![20]);
    assert_eq!(run.output_token_ids, vec![10, 20]);
    assert_eq!(run.stop_reason, "user_interrupt");
    assert_eq!(run.tick_count, 1);
}

#[test]
fn placed_active_prompt_event_interrupt_excludes_unprocessed_prompt_input() {
    let input_event = VulkanResidentTokenInputEvent::new("event", vec![10, 11], 1);
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(input_event, 4).unwrap();
    let activation = active.next_activation().unwrap();
    assert!(!activation.should_emit_public_output);
    active
        .complete_activation(
            &activation,
            4,
            2,
            5,
            &VulkanPlacedEdgeTransportStats::default(),
            None,
        )
        .unwrap();

    active.interrupt("user_interrupt");
    let run = active.into_event_run("gpu0".to_string(), "gpu0".to_string());
    assert_eq!(run.prompt_token_ids, vec![10, 11]);
    assert!(run.generated_token_ids.is_empty());
    assert_eq!(run.output_token_ids, vec![10]);
    assert_eq!(run.stop_reason, "user_interrupt");
    assert_eq!(run.tick_count, 1);
}

#[test]
fn placed_active_prompt_event_stop_after_current_processes_closing_feedback() {
    let input_event = VulkanResidentTokenInputEvent::new("event", vec![10], 3);
    let mut active = VulkanResidentInProcessPlacedActivePromptEvent::new(input_event, 4).unwrap();
    let activation = active.next_activation().unwrap();
    active
        .complete_activation(
            &activation,
            4,
            2,
            5,
            &VulkanPlacedEdgeTransportStats::default(),
            Some(20),
        )
        .unwrap();

    let control = active.stop_after_current("user_stop");
    assert_eq!(
        control.event_type,
        VulkanResidentStreamControlEventType::StopAfterCurrent
    );
    assert_eq!(control.reason, "user_stop");
    assert!(control.cleared_private_feedback_ids.is_empty());
    assert_eq!(
        control.closing_private_feedback_id.as_deref(),
        Some("event.feedback.0")
    );
    assert!(!active.is_complete());

    let closing = active.next_activation().unwrap();
    assert_eq!(closing.input_token_id, 20);
    assert!(closing.input_is_feedback);
    assert!(closing.input_closes_loop_after_processing);
    assert!(!closing.should_emit_public_output);
    assert!(
        active
            .complete_activation(
                &closing,
                5,
                2,
                5,
                &VulkanPlacedEdgeTransportStats::default(),
                None,
            )
            .unwrap()
            .is_none()
    );
    assert!(active.is_complete());

    let run = active.into_event_run("gpu0".to_string(), "gpu0".to_string());
    assert_eq!(run.generated_token_ids, vec![20]);
    assert_eq!(run.stop_reason, "user_stop");
    assert_eq!(run.tick_count, 2);
}

