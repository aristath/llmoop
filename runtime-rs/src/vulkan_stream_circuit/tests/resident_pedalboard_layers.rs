#[test]
fn resident_greedy_running_stream_stop_after_current_processes_one_feedback_then_idles() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident running stream stop-after-current: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor(
        &device,
        "resident running stream stop-after-current",
    ) else {
        return;
    };
    let mut stream = processor.into_running_stream("stream_0");

    stream.inject_prompt(&[1], 3, None).unwrap();
    let first_tick = stream.tick(&device).unwrap();
    assert_eq!(first_tick.stream_tick, Some(0));
    assert!(first_tick.public_output.is_some());
    assert!(
        !first_tick
            .private_feedback
            .as_ref()
            .unwrap()
            .closes_loop_after_processing
    );
    assert_eq!(stream.pending_private_feedback_count(), 1);
    assert_eq!(stream.remaining_public_outputs, 2);

    let event = stream.stop_after_current("user_stop");
    assert_eq!(
        event.event_type,
        VulkanResidentStreamControlEventType::StopAfterCurrent
    );
    assert_eq!(event.reason, "user_stop");
    assert_eq!(
        event.closing_private_feedback_id.as_deref(),
        Some("feedback_0")
    );
    assert!(event.cleared_private_feedback_ids.is_empty());
    assert!(event.state_preserved);
    assert_eq!(stream.pending_private_feedback_count(), 1);
    assert_eq!(stream.remaining_public_outputs, 0);
    assert!(stream.loop_open);
    assert!(stream.private_feedback_history()[0].closes_loop_after_processing);
    assert_eq!(
        stream.private_feedback_history()[0].stop_reason.as_deref(),
        Some("user_stop")
    );

    let closing_tick = stream.tick(&device).unwrap();
    assert_eq!(closing_tick.stream_tick, Some(1));
    assert_eq!(
        closing_tick.input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert!(
        closing_tick
            .input_signal
            .as_ref()
            .unwrap()
            .closes_loop_after_processing()
    );
    assert!(closing_tick.public_output.is_none());
    assert!(closing_tick.private_feedback.is_none());
    assert_eq!(closing_tick.stop_reason.as_deref(), Some("user_stop"));
    assert!(!stream.loop_open);
    assert_eq!(stream.last_stop_reason.as_deref(), Some("user_stop"));

    let idle = stream.tick(&device).unwrap();
    assert_eq!(idle.status, VulkanResidentRunningStreamTickStatus::Idle);
    assert_eq!(idle.stream_tick, None);
    assert_eq!(stream.next_stream_tick, 2);
    assert_eq!(stream.pending_private_feedback_count(), 0);
    assert_eq!(stream.public_outputs().len(), 1);
    assert_eq!(stream.private_feedback_history().len(), 1);
}

#[test]
fn resident_pedalboard_runner_executes_layer_00_to_layer_01_over_local_cable() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident pedalboard runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) = layer_00_level_1_loaded_kernel_pack(&mounted, &mounted_bound)
    else {
        eprintln!("skipping resident pedalboard runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 1);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 32, 104, 0);

    let run = runner.run_zeroed_push_constants(&device).unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 32);

    let layer_00_output_dispatch = mounted_bound.dispatch("layer_00", "ffn_residual").unwrap();
    let layer_00_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_00_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_00_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x86, 0x3f, 0x82, 0x3f, 0x81, 0x3f, 0x7e, 0x3f, 0x83, 0x3f, 0x83, 0x3f, 0x83, 0x3f,
            0x83, 0x3f,
        ]
    );

    let layer_01_output_dispatch = mounted_bound.dispatch("layer_01", "ffn_residual").unwrap();
    let layer_01_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_01_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_01_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x86, 0x3f, 0x84, 0x3f, 0x80, 0x3f, 0x7e, 0x3f, 0x83, 0x3f, 0x84, 0x3f, 0x88, 0x3f,
            0x83, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_attention_layer_02_with_per_pedal_kv_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident attention pedalboard runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!(
            "skipping resident attention pedalboard runner: no GLSL to SPIR-V compiler found"
        );
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 2);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 51, 171, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 51);

    let kv_memory = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap();
    assert_ne!(kv_memory.buffer.read_bytes(16).unwrap(), vec![0; 16]);

    let layer_02_output_dispatch = mounted_bound.dispatch("layer_02", "ffn_residual").unwrap();
    let layer_02_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_02_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_02_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x8b, 0x3f, 0x7e, 0x3f, 0x87, 0x3f, 0x6a, 0x3f, 0x71, 0x3f, 0x87, 0x3f, 0x8a, 0x3f,
            0x7e, 0x3f,
        ]
    );
}

#[test]
fn resident_attention_pedal_reuses_kv_state_across_stream_ticks() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident multi-tick attention runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!(
            "skipping resident multi-tick attention runner: no GLSL to SPIR-V compiler found"
        );
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 2);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    let layer_02_runner = mounted
        .create_resident_pedal_runner(&device, &mounted_bound, "layer_02", &loaded_manifest)
        .unwrap();
    let dynamic_state_capacity_activations =
        mounted.buffers.dynamic_state_capacity_activations as u32;

    runner
        .run_with_stream_control(
            &device,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations,
            },
        )
        .unwrap();
    let kv_memory = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap();
    let kv_after_tick_0 = kv_memory.buffer.read_bytes(2_064).unwrap();
    let tick_0_slot_0 = kv_after_tick_0[0..16].to_vec();
    assert_ne!(tick_0_slot_0, vec![0; 16]);
    assert_eq!(&kv_after_tick_0[2_048..2_064], &[0u8; 16]);

    write_layer_00_constant_input(&mounted, [0x00, 0x3f]);
    runner
        .run_with_stream_control(
            &device,
            VulkanMountedPlacedStreamControl {
                stream_tick: 1,
                control_flags: 0,
                dynamic_state_capacity_activations,
            },
        )
        .unwrap();

    let layer_02_output_dispatch = mounted_bound.dispatch("layer_02", "ffn_residual").unwrap();
    let layer_02_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_02_output_dispatch)
        .unwrap();
    let historical_output = layer_02_output_bindings[2]
        .buffer
        .read_bytes(2_048)
        .unwrap();
    let kv_after_tick_1 = kv_memory.buffer.read_bytes(4_112).unwrap();
    assert_eq!(&kv_after_tick_1[0..16], tick_0_slot_0.as_slice());
    assert_ne!(&kv_after_tick_1[2_048..2_064], &[0u8; 16]);
    assert_ne!(&kv_after_tick_1[2_048..2_064], tick_0_slot_0.as_slice());

    zero_fixture_model_kv_memory(&mounted, "layer_02");
    layer_02_runner
        .run_with_stream_control(
            &device,
            VulkanMountedPlacedStreamControl {
                stream_tick: 1,
                control_flags: 0,
                dynamic_state_capacity_activations,
            },
        )
        .unwrap();
    let no_history_output = layer_02_output_bindings[2]
        .buffer
        .read_bytes(2_048)
        .unwrap();
    let kv_after_no_history = kv_memory.buffer.read_bytes(4_112).unwrap();
    assert_eq!(&kv_after_no_history[0..16], &[0u8; 16]);
    assert_ne!(&kv_after_no_history[2_048..2_064], &[0u8; 16]);
    assert_ne!(historical_output, no_history_output);
}

#[test]
fn resident_pedalboard_runner_executes_attention_output_into_next_conv_layer() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_03 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_03 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 3);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 67, 223, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 67);

    let layer_03_output_dispatch = mounted_bound.dispatch("layer_03", "ffn_residual").unwrap();
    let layer_03_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_03_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_03_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x89, 0x3f, 0x73, 0x3f, 0x86, 0x3f, 0x6b, 0x3f, 0x6f, 0x3f, 0x88, 0x3f, 0x88, 0x3f,
            0x83, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_second_attention_layer_with_independent_kv_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_04 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_04 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 4);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 86, 290, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 86);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);

    let layer_04_output_dispatch = mounted_bound.dispatch("layer_04", "ffn_residual").unwrap();
    let layer_04_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_04_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_04_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x8c, 0x3f, 0x62, 0x3f, 0x88, 0x3f, 0x61, 0x3f, 0x73, 0x3f, 0x85, 0x3f, 0x89, 0x3f,
            0x84, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_second_attention_output_into_next_conv_layer() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_05 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_05 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 5);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 102, 342, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 102);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);

    let layer_05_output_dispatch = mounted_bound.dispatch("layer_05", "ffn_residual").unwrap();
    let layer_05_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_05_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_05_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x8a, 0x3f, 0x61, 0x3f, 0x86, 0x3f, 0x60, 0x3f, 0x74, 0x3f, 0x85, 0x3f, 0x8a, 0x3f,
            0x83, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_third_attention_layer_with_independent_kv_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_06 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_06 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 6);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 121, 409, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 121);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_06_kv);

    let layer_06_output_dispatch = mounted_bound.dispatch("layer_06", "ffn_residual").unwrap();
    let layer_06_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_06_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_06_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x8b, 0x3f, 0x5f, 0x3f, 0x80, 0x3f, 0x68, 0x3f, 0x7a, 0x3f, 0x8d, 0x3f, 0x88, 0x3f,
            0x80, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_third_attention_output_into_next_conv_layer() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_07 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_07 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 7);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 137, 461, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 137);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_06_kv);

    let layer_07_output_dispatch = mounted_bound.dispatch("layer_07", "ffn_residual").unwrap();
    let layer_07_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_07_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_07_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x8a, 0x3f, 0x62, 0x3f, 0x7f, 0x3f, 0x6d, 0x3f, 0x7c, 0x3f, 0x8a, 0x3f, 0x8a, 0x3f,
            0x7f, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_fourth_attention_layer_with_independent_kv_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_08 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_08 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 8);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 156, 528, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 156);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_08_kv = mounted
        .buffers
        .state_buffer("layer_08", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_08_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_02_kv, layer_08_kv);
    assert_ne!(layer_04_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_08_kv);
    assert_ne!(layer_06_kv, layer_08_kv);

    let layer_08_output_dispatch = mounted_bound.dispatch("layer_08", "ffn_residual").unwrap();
    let layer_08_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_08_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_08_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x97, 0x3f, 0x63, 0x3f, 0x86, 0x3f, 0x5e, 0x3f, 0x69, 0x3f, 0x8d, 0x3f, 0x83, 0x3f,
            0x71, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_fourth_attention_output_into_next_conv_layer() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_09 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_09 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 9);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 172, 580, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 172);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_08_kv = mounted
        .buffers
        .state_buffer("layer_08", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_08_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_02_kv, layer_08_kv);
    assert_ne!(layer_04_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_08_kv);
    assert_ne!(layer_06_kv, layer_08_kv);

    let layer_09_output_dispatch = mounted_bound.dispatch("layer_09", "ffn_residual").unwrap();
    let layer_09_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_09_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_09_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x95, 0x3f, 0x5c, 0x3f, 0x83, 0x3f, 0x60, 0x3f, 0x78, 0x3f, 0x8e, 0x3f, 0x82, 0x3f,
            0x7b, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_fifth_attention_layer_with_independent_kv_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_10 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_10 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 10);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 191, 647, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 191);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_08_kv = mounted
        .buffers
        .state_buffer("layer_08", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_10_kv = mounted
        .buffers
        .state_buffer("layer_10", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_08_kv, vec![0; 16]);
    assert_ne!(layer_10_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_02_kv, layer_08_kv);
    assert_ne!(layer_02_kv, layer_10_kv);
    assert_ne!(layer_04_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_08_kv);
    assert_ne!(layer_04_kv, layer_10_kv);
    assert_ne!(layer_06_kv, layer_08_kv);
    assert_ne!(layer_06_kv, layer_10_kv);
    assert_ne!(layer_08_kv, layer_10_kv);

    let layer_10_output_dispatch = mounted_bound.dispatch("layer_10", "ffn_residual").unwrap();
    let layer_10_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_10_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_10_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x94, 0x3f, 0x53, 0x3f, 0x85, 0x3f, 0x43, 0x3f, 0x90, 0x3f, 0x94, 0x3f, 0x87, 0x3f,
            0x63, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_fifth_attention_output_into_next_conv_layer() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_11 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_11 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 11);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 207, 699, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 207);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_08_kv = mounted
        .buffers
        .state_buffer("layer_08", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_10_kv = mounted
        .buffers
        .state_buffer("layer_10", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_08_kv, vec![0; 16]);
    assert_ne!(layer_10_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_02_kv, layer_08_kv);
    assert_ne!(layer_02_kv, layer_10_kv);
    assert_ne!(layer_04_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_08_kv);
    assert_ne!(layer_04_kv, layer_10_kv);
    assert_ne!(layer_06_kv, layer_08_kv);
    assert_ne!(layer_06_kv, layer_10_kv);
    assert_ne!(layer_08_kv, layer_10_kv);

    let layer_11_output_dispatch = mounted_bound.dispatch("layer_11", "ffn_residual").unwrap();
    let layer_11_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_11_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_11_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x95, 0x3f, 0x4b, 0x3f, 0x86, 0x3f, 0x2d, 0x3f, 0x93, 0x3f, 0x9c, 0x3f, 0x82, 0x3f,
            0x64, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_final_attention_layer_with_independent_kv_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident layer_12 prefix runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!("skipping resident layer_12 prefix runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 12);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 226, 766, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 226);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_08_kv = mounted
        .buffers
        .state_buffer("layer_08", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_10_kv = mounted
        .buffers
        .state_buffer("layer_10", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_12_kv = mounted
        .buffers
        .state_buffer("layer_12", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_08_kv, vec![0; 16]);
    assert_ne!(layer_10_kv, vec![0; 16]);
    assert_ne!(layer_12_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_02_kv, layer_08_kv);
    assert_ne!(layer_02_kv, layer_10_kv);
    assert_ne!(layer_02_kv, layer_12_kv);
    assert_ne!(layer_04_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_08_kv);
    assert_ne!(layer_04_kv, layer_10_kv);
    assert_ne!(layer_04_kv, layer_12_kv);
    assert_ne!(layer_06_kv, layer_08_kv);
    assert_ne!(layer_06_kv, layer_10_kv);
    assert_ne!(layer_06_kv, layer_12_kv);
    assert_ne!(layer_08_kv, layer_10_kv);
    assert_ne!(layer_08_kv, layer_12_kv);
    assert_ne!(layer_10_kv, layer_12_kv);

    let layer_12_output_dispatch = mounted_bound.dispatch("layer_12", "ffn_residual").unwrap();
    let layer_12_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_12_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_12_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x98, 0x3f, 0x4a, 0x3f, 0x6f, 0x3f, 0x19, 0x3f, 0x8f, 0x3f, 0x9c, 0x3f, 0x36, 0x3f,
            0x6c, 0x3f,
        ]
    );
}

#[test]
fn resident_pedalboard_runner_executes_full_fixture_model_layer_stack() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident full FIXTURE_MODEL layer-stack runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) =
        fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
            &mounted,
            &mounted_bound,
        )
    else {
        eprintln!(
            "skipping resident full FIXTURE_MODEL layer-stack runner: no GLSL to SPIR-V compiler found"
        );
        return;
    };
    let pedal_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 13);

    let runner = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &pedal_ids,
    );
    assert_fixture_model_resident_prefix_runner(&runner, &pedal_ids, 242, 818, 0);

    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_fixture_model_resident_prefix_run(&run, &pedal_ids, 242);

    let layer_02_kv = mounted
        .buffers
        .state_buffer("layer_02", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_04_kv = mounted
        .buffers
        .state_buffer("layer_04", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_06_kv = mounted
        .buffers
        .state_buffer("layer_06", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_08_kv = mounted
        .buffers
        .state_buffer("layer_08", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_10_kv = mounted
        .buffers
        .state_buffer("layer_10", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    let layer_12_kv = mounted
        .buffers
        .state_buffer("layer_12", "kv_memory")
        .unwrap()
        .buffer
        .read_bytes(16)
        .unwrap();
    assert_ne!(layer_02_kv, vec![0; 16]);
    assert_ne!(layer_04_kv, vec![0; 16]);
    assert_ne!(layer_06_kv, vec![0; 16]);
    assert_ne!(layer_08_kv, vec![0; 16]);
    assert_ne!(layer_10_kv, vec![0; 16]);
    assert_ne!(layer_12_kv, vec![0; 16]);
    assert_ne!(layer_02_kv, layer_04_kv);
    assert_ne!(layer_02_kv, layer_06_kv);
    assert_ne!(layer_02_kv, layer_08_kv);
    assert_ne!(layer_02_kv, layer_10_kv);
    assert_ne!(layer_02_kv, layer_12_kv);
    assert_ne!(layer_04_kv, layer_06_kv);
    assert_ne!(layer_04_kv, layer_08_kv);
    assert_ne!(layer_04_kv, layer_10_kv);
    assert_ne!(layer_04_kv, layer_12_kv);
    assert_ne!(layer_06_kv, layer_08_kv);
    assert_ne!(layer_06_kv, layer_10_kv);
    assert_ne!(layer_06_kv, layer_12_kv);
    assert_ne!(layer_08_kv, layer_10_kv);
    assert_ne!(layer_08_kv, layer_12_kv);
    assert_ne!(layer_10_kv, layer_12_kv);

    let layer_13_output_dispatch = mounted_bound.dispatch("layer_13", "ffn_residual").unwrap();
    let layer_13_output_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(layer_13_output_dispatch)
        .unwrap();
    assert_eq!(
        layer_13_output_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x84, 0x3f, 0x48, 0x3f, 0x82, 0x3f, 0x0c, 0x3f, 0x92, 0x3f, 0x9f, 0x3f, 0x15, 0x3f,
            0x62, 0x3f,
        ]
    );
}

#[test]
