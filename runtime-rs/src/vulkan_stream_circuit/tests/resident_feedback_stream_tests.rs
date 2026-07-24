fn resident_single_token_tick_runs_input_graph_and_output_to_logits() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident single-token tick runner: {error}");
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
        eprintln!("skipping resident single-token tick runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let Some(input_transducer_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source(
            "embedding_lookup_bf16_65536x1024.comp",
        )
    else {
        eprintln!("skipping resident single-token tick runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let Some(embedding_norm_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source("rms_norm_bf16_serial.comp")
    else {
        eprintln!("skipping resident single-token tick runner: no GLSL to SPIR-V compiler found");
        return;
    };
    let Some(tied_projection_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source(
            "tied_output_projection_bf16_65536x1024_to_f32.comp",
        )
    else {
        eprintln!("skipping resident single-token tick runner: no GLSL to SPIR-V compiler found");
        return;
    };

    let transducer_parameter_buffers =
        load_fixture_model_transducer_parameter_buffers(&device, &tensor_index);
    let component_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 13);
    let input_transducer =
        VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding(
            &device,
            &mounted,
            &transducer_parameter_buffers,
            &input_transducer_spirv_words,
            &fixture_model_input_embedding_transducer_spec(),
        )
        .unwrap();
    let execution_graph = create_fixture_model_resident_prefix_runner(
        &device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &component_ids,
    );
    let output_transducer = VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
        &device,
        &mounted,
        &transducer_parameter_buffers,
        &embedding_norm_spirv_words,
        &tied_projection_spirv_words,
        &fixture_model_output_transducer_spec(),
    )
    .unwrap();
    let runner = VulkanResidentSingleTokenTickRunner::new(
        &device,
        input_transducer,
        execution_graph,
        output_transducer,
    )
    .unwrap();
    assert_eq!(runner.device_id, "gpu0");
    assert_eq!(runner.component_count, 14);
    assert_eq!(runner.dispatch_count, 245);
    assert_eq!(runner.total_descriptor_count, 827);
    assert_eq!(runner.total_push_constant_byte_count, 0);

    let token_id = 1u32;
    let run = runner
        .run_token_id_with_stream_control(
            &device,
            token_id,
            fixture_model_stream_control(&mounted, 0),
        )
        .unwrap();
    assert_eq!(run.device_id, "gpu0");
    assert_eq!(run.token_id, token_id);
    assert_eq!(run.dispatch_count, 245);
    assert_eq!(run.total_descriptor_count, 827);
    assert_eq!(run.total_push_constant_byte_count, 0);
    assert_eq!(run.input_run.dispatch_count, 1);
    assert_eq!(
        run.input_run.output_signal_id,
        FIXTURE_MODEL_INPUT_FRAME_SIGNAL
    );
    assert_fixture_model_resident_prefix_run(&run.execution_graph_run, &component_ids, 242);
    assert_eq!(run.output_run.as_ref().unwrap().dispatch_count, 2);
    assert_eq!(
        run.output_run.as_ref().unwrap().logits_byte_capacity,
        FIXTURE_MODEL_LOGITS_BYTES
    );

    let input_frame = mounted
        .boundary_io
        .input_buffer(FIXTURE_MODEL_INPUT_FRAME_SIGNAL)
        .unwrap();
    assert_eq!(
        input_frame
            .buffer
            .read_bytes(FIXTURE_MODEL_FRAME_BYTES)
            .unwrap(),
        fixture_model_embedding_row_bytes(&tensor_index, token_id)
    );
    assert_bf16_bytes_close(
        &runner.read_normalized_frame_bytes(16).unwrap(),
        &[
            0xb5, 0xbf, 0xee, 0xbf, 0x51, 0x3f, 0x99, 0xbf, 0x35, 0xbf, 0xc4, 0xbe, 0x7a, 0x3f,
            0x94, 0xbf,
        ],
        0.02,
    );
    assert_f32_bytes_close(
        &runner.read_logits_bytes(16).unwrap(),
        &[
            0xa3, 0xc8, 0x12, 0xc0, 0xf7, 0xc1, 0x9d, 0x41, 0x84, 0x92, 0x6a, 0x41, 0x16, 0x9c,
            0x17, 0xc0,
        ],
        0.05,
    );
}

fn create_fixture_model_resident_greedy_stream_processor(
    device: &VulkanComputeDevice,
    skip_label: &str,
) -> Option<VulkanResidentStreamProcessor> {
    create_fixture_model_resident_greedy_stream_processor_with_capacity(
        device,
        skip_label,
        4,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    )
}

fn create_fixture_model_resident_greedy_stream_processor_with_capacity(
    device: &VulkanComputeDevice,
    skip_label: &str,
    dynamic_state_capacity_activations: usize,
    attention_shader: &str,
) -> Option<VulkanResidentStreamProcessor> {
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit_with_capacity(
            device,
            dynamic_state_capacity_activations,
        );
    let Some(loaded_manifest) =
            fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
                &mounted,
                &mounted_bound,
                attention_shader,
            )
        else {
            eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
            return None;
        };
    let Some(input_transducer_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source(
            "embedding_lookup_bf16_65536x1024.comp",
        )
    else {
        eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
        return None;
    };
    let Some(embedding_norm_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source("rms_norm_bf16_serial.comp")
    else {
        eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
        return None;
    };
    let Some(tied_projection_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source(
            "tied_output_projection_bf16_65536x1024_to_f32.comp",
        )
    else {
        eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
        return None;
    };
    let Some(sampler_spirv_words) = crate::vulkan_compute::compile_test_shader_words_from_source(
        "greedy_sampler_f32_65536.comp",
    ) else {
        eprintln!("skipping {skip_label}: no GLSL to SPIR-V compiler found");
        return None;
    };
    let Some(sampler_kernels) = greedy_sampler_test_kernels(sampler_spirv_words) else {
        eprintln!("skipping resident feedback smoke: feedback control shader did not compile");
        return None;
    };

    let transducer_parameter_buffers = Arc::new(load_fixture_model_transducer_parameter_buffers(
        device,
        &tensor_index,
    ));
    let component_ids = prepare_fixture_model_resident_prefix(&mounted, &tensor_index, 13);
    let input_transducer =
        VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding(
            device,
            &mounted,
            &transducer_parameter_buffers,
            &input_transducer_spirv_words,
            &fixture_model_input_embedding_transducer_spec(),
        )
        .unwrap();
    let execution_graph = create_fixture_model_resident_prefix_runner(
        device,
        &mounted,
        &mounted_bound,
        &loaded_manifest,
        &component_ids,
    );
    let output_transducer = VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
        device,
        &mounted,
        &transducer_parameter_buffers,
        &embedding_norm_spirv_words,
        &tied_projection_spirv_words,
        &fixture_model_output_transducer_spec(),
    )
    .unwrap();
    let sampler = VulkanResidentSamplerRunner::from_output_transducer_with_spec(
        device,
        &mounted,
        &output_transducer,
        &sampler_kernels,
        &fixture_model_greedy_sampler_spec(),
        0,
    )
    .unwrap();
    let tick_runner = VulkanResidentSingleTokenTickRunner::new(
        device,
        input_transducer,
        execution_graph,
        output_transducer,
    )
    .unwrap();
    let loop_runner = VulkanResidentFeedbackLoopRunner::new(tick_runner, sampler).unwrap();
    Some(
        VulkanResidentStreamProcessor::new(
            device,
            mounted,
            transducer_parameter_buffers,
            loop_runner,
        )
        .unwrap(),
    )
}

#[test]
fn resident_greedy_feedback_loop_runs_two_ticks() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident feedback loop: {error}");
            return;
        }
    };
    let Some(processor) =
        create_fixture_model_resident_greedy_stream_processor(&device, "resident feedback loop")
    else {
        return;
    };
    assert_eq!(processor.device_id, "gpu0");
    assert_eq!(processor.component_count, 14);
    assert_eq!(processor.per_tick_dispatch_count, 247);
    assert_eq!(processor.per_tick_descriptor_count, 834);
    assert_eq!(processor.per_tick_push_constant_byte_count, 0);
    assert_eq!(processor.dynamic_state_capacity_activations, 4);

    let run = processor.run_bounded(&device, 1, 0, 2).unwrap();
    assert_eq!(run.device_id, "gpu0");
    assert_eq!(run.initial_token_id, 1);
    assert_eq!(run.tick_runs.len(), 2);
    assert_eq!(run.per_tick_dispatch_count, 247);
    assert_eq!(run.per_tick_descriptor_count, 834);
    assert_eq!(run.per_tick_push_constant_byte_count, 0);
    assert_eq!(run.tick_runs[0].stream_tick, 0);
    assert_eq!(run.tick_runs[0].input_token_id, 1);
    assert_eq!(run.tick_runs[1].stream_tick, 1);
    assert_eq!(
        run.tick_runs[1].input_token_id,
        run.tick_runs[0].sampled_token_id
    );
    assert_eq!(run.tick_runs[0].tick_run.dispatch_count, 245);
    assert_eq!(run.tick_runs[0].sampler_run.descriptor_count, 5);
    assert_eq!(run.tick_runs[1].tick_run.dispatch_count, 245);
    assert_eq!(run.tick_runs[1].sampler_run.descriptor_count, 5);
    assert_eq!(run.sampled_token_ids, vec![1, 1]);
    assert_eq!(run.tick_runs[0].sampler_run.token_id, 1);
    assert_eq!(run.tick_runs[1].sampler_run.token_id, 1);
    for (actual, expected) in run
        .tick_runs
        .iter()
        .map(|tick| tick.sampler_run.selected_logit_bits)
        .zip([1_100_857_847, 1_101_582_210])
    {
        assert_f32_bits_close(actual, expected, 0.01, 0.01);
    }
}

#[test]
fn resident_greedy_prompt_event_drains_external_input_before_feedback() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident prompt event: {error}");
            return;
        }
    };
    let Some(processor) =
        create_fixture_model_resident_greedy_stream_processor(&device, "resident prompt event")
    else {
        return;
    };

    let run = processor
        .run_prompt_event_bounded(&device, &[1, 36_309], 0, 1, None)
        .unwrap();

    assert_eq!(run.device_id, "gpu0");
    assert_eq!(run.prompt_token_ids, vec![1, 36_309]);
    assert_eq!(run.generated_token_ids.len(), 1);
    assert_eq!(
        run.output_token_ids,
        vec![1, 36_309, run.generated_token_ids[0]]
    );
    assert_eq!(run.stop_reason, "max_new_tokens");
    assert_eq!(run.tick_runs.len(), 3);
    assert_eq!(run.per_tick_dispatch_count, 247);
    assert_eq!(run.per_tick_descriptor_count, 834);
    assert_eq!(run.per_tick_push_constant_byte_count, 0);

    assert_eq!(run.tick_runs[0].stream_tick, 0);
    assert_eq!(run.tick_runs[0].input_token_id, 1);
    assert_eq!(
        run.tick_runs[0].input_route,
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(run.tick_runs[0].public_output_token_id, None);
    assert_eq!(run.tick_runs[0].private_feedback_token_id, None);
    assert!(run.tick_runs[0].sampler_run.is_none());
    assert_eq!(run.tick_runs[0].tick_run.dispatch_count, 243);
    assert!(run.tick_runs[0].tick_run.output_run.is_none());

    assert_eq!(run.tick_runs[1].stream_tick, 1);
    assert_eq!(run.tick_runs[1].input_token_id, 36_309);
    assert_eq!(
        run.tick_runs[1].input_route,
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(
        run.tick_runs[1].public_output_token_id,
        Some(run.generated_token_ids[0])
    );
    assert_eq!(
        run.tick_runs[1].private_feedback_token_id,
        Some(run.generated_token_ids[0])
    );
    assert_eq!(
        run.tick_runs[1].private_feedback_closes_loop_after_processing,
        Some(true)
    );
    assert_eq!(
        run.tick_runs[1].sampler_run.as_ref().unwrap().token_id,
        run.generated_token_ids[0]
    );
    assert_eq!(run.tick_runs[1].tick_run.dispatch_count, 245);
    assert!(run.tick_runs[1].tick_run.output_run.is_some());

    assert_eq!(run.tick_runs[2].stream_tick, 2);
    assert_eq!(run.tick_runs[2].input_token_id, run.generated_token_ids[0]);
    assert_eq!(
        run.tick_runs[2].input_route,
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(run.tick_runs[2].input_feedback_depth, 1);
    assert!(run.tick_runs[2].input_closes_loop_after_processing);
    assert_eq!(run.tick_runs[2].public_output_token_id, None);
    assert_eq!(run.tick_runs[2].private_feedback_token_id, None);
    assert!(run.tick_runs[2].sampler_run.is_none());
    assert_eq!(run.tick_runs[2].tick_run.dispatch_count, 243);
    assert!(run.tick_runs[2].tick_run.output_run.is_none());
}

#[test]
fn resident_greedy_running_stream_accepts_later_input_without_resetting_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident running stream: {error}");
            return;
        }
    };
    let Some(processor) =
        create_fixture_model_resident_greedy_stream_processor(&device, "resident running stream")
    else {
        return;
    };
    let mut stream = processor.into_running_stream("stream_0");
    assert_eq!(stream.stream_id, "stream_0");
    assert_eq!(stream.next_stream_tick, 0);
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.pending_private_feedback_count(), 0);

    let first = stream.run_prompt(&device, &[1], 1, None).unwrap();
    assert_eq!(first.stream_id, "stream_0");
    assert_eq!(first.prompt_token_ids, vec![1]);
    assert_eq!(first.generated_token_ids.len(), 1);
    assert_eq!(
        first.output_token_ids,
        vec![1, first.generated_token_ids[0]]
    );
    assert_eq!(first.stop_reason, "max_new_tokens");
    assert_eq!(first.start_stream_tick, 0);
    assert_eq!(first.next_stream_tick, 2);
    assert_eq!(first.ticks.len(), 3);
    assert_eq!(
        first.ticks[0].status,
        VulkanResidentRunningStreamTickStatus::Processed
    );
    assert_eq!(first.ticks[0].stream_tick, Some(0));
    assert_eq!(
        first.ticks[0].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(
        first.ticks[0].public_output.as_ref().unwrap().token_id,
        first.generated_token_ids[0]
    );
    assert_eq!(first.ticks[1].stream_tick, Some(1));
    assert_eq!(
        first.ticks[1].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(
        first.ticks[1].input_signal.as_ref().unwrap().token_id(),
        first.generated_token_ids[0]
    );
    assert_eq!(
        first.ticks[2].status,
        VulkanResidentRunningStreamTickStatus::Idle
    );
    assert_eq!(first.ticks[2].stream_tick, None);
    assert_eq!(stream.next_stream_tick, 2);
    assert_eq!(stream.public_outputs().len(), 1);
    assert_eq!(stream.private_feedback_history().len(), 1);
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.pending_private_feedback_count(), 0);

    let second = stream.run_prompt(&device, &[36_309], 1, None).unwrap();
    assert_eq!(second.prompt_token_ids, vec![36_309]);
    assert_eq!(second.generated_token_ids.len(), 1);
    assert_eq!(
        second.output_token_ids,
        vec![36_309, second.generated_token_ids[0]]
    );
    assert_eq!(second.stop_reason, "max_new_tokens");
    assert_eq!(second.start_stream_tick, 2);
    assert_eq!(second.next_stream_tick, 4);
    assert_eq!(second.ticks.len(), 3);
    assert_eq!(second.ticks[0].stream_tick, Some(2));
    assert_eq!(
        second.ticks[0].input_signal.as_ref().unwrap().token_id(),
        36_309
    );
    assert_eq!(
        second.ticks[0].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(second.ticks[1].stream_tick, Some(3));
    assert_eq!(
        second.ticks[1].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(
        second.ticks[2].status,
        VulkanResidentRunningStreamTickStatus::Idle
    );
    assert_eq!(second.ticks[2].stream_tick, None);
    assert_eq!(stream.next_stream_tick, 4);
    assert_eq!(stream.public_outputs().len(), 2);
    assert_eq!(stream.private_feedback_history().len(), 2);
    assert_eq!(stream.ticks().len(), 6);
    assert!(!stream.loop_open);
    assert_eq!(stream.last_stop_reason.as_deref(), Some("max_new_tokens"));

    stream.inject_prompt(&[1], 0, None).unwrap();
    let rolled = stream.tick(&device).unwrap();
    assert_eq!(rolled.stream_tick, Some(4));
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.next_stream_tick, 5);
}

#[test]
fn resident_greedy_running_stream_uses_configured_capacity() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident running stream capacity: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident running stream capacity",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    assert_eq!(processor.dynamic_state_capacity_activations, 8);

    let mut stream = processor.into_running_stream("stream_0");
    let run = stream.run_prompt(&device, &[1], 7, None).unwrap();
    assert_eq!(run.prompt_token_ids, vec![1]);
    assert_eq!(run.generated_token_ids.len(), 7);
    assert_eq!(run.output_token_ids.len(), 8);
    assert_eq!(run.stop_reason, "max_new_tokens");
    assert_eq!(run.start_stream_tick, 0);
    assert_eq!(run.next_stream_tick, 8);
    assert_eq!(stream.next_stream_tick, 8);
    assert_eq!(stream.public_outputs().len(), 7);
    assert_eq!(stream.private_feedback_history().len(), 7);
    assert_eq!(run.ticks.len(), 9);
    assert_eq!(run.ticks[0].stream_tick, Some(0));
    assert_eq!(run.ticks[7].stream_tick, Some(7));
    assert_eq!(
        run.ticks[7].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(
        run.ticks[8].status,
        VulkanResidentRunningStreamTickStatus::Idle
    );
    assert_eq!(run.ticks[8].stream_tick, None);

    stream.inject_prompt(&[36_309], 0, None).unwrap();
    let rolled = stream.tick(&device).unwrap();
    assert_eq!(rolled.stream_tick, Some(8));
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.next_stream_tick, 9);
}

#[test]
fn resident_token_stream_api_accepts_external_events_and_emits_public_events() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token stream API: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token stream API",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut stream = processor.into_token_stream("host_stream_0");
    assert_eq!(stream.stream_id(), "host_stream_0");
    assert_eq!(stream.next_stream_tick(), 0);

    let first_event =
        VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host");
    let first = stream
        .submit_external_event(&device, first_event.clone())
        .unwrap();
    assert_eq!(first.stream_id, "host_stream_0");
    assert_eq!(first.input_event, first_event);
    assert_eq!(first.generated_token_ids.len(), 3);
    assert_eq!(first.output_events.len(), 3);
    assert_eq!(first.stop_reason, "max_new_tokens");
    assert_eq!(first.start_stream_tick, 0);
    assert_eq!(first.next_stream_tick, 4);
    assert_eq!(first.processed_tick_count, 4);
    assert_eq!(first.idle_tick_count, 1);
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| event.input_event_id.as_str())
            .collect::<Vec<_>>(),
        vec!["event_0", "event_0", "event_0"]
    );
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| event.output_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| event.source_stream_tick)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    let second_event =
        VulkanResidentTokenInputEvent::new("event_1", vec![36_309], 1).with_origin("test_host");
    let second = stream
        .submit_external_event(&device, second_event.clone())
        .unwrap();
    assert_eq!(second.input_event, second_event);
    assert_eq!(second.generated_token_ids.len(), 1);
    assert_eq!(second.output_events.len(), 1);
    assert_eq!(second.output_events[0].input_event_id, "event_1");
    assert_eq!(second.output_events[0].output_index, 0);
    assert_eq!(second.output_events[0].source_stream_tick, 4);
    assert_eq!(second.start_stream_tick, 4);
    assert_eq!(second.next_stream_tick, 6);
    assert_eq!(second.processed_tick_count, 2);
    assert_eq!(second.idle_tick_count, 1);

    let snapshot = stream.snapshot();
    assert_eq!(snapshot.stream_id, "host_stream_0");
    assert_eq!(snapshot.next_stream_tick, 6);
    assert!(!snapshot.loop_open);
    assert!(snapshot.idle);
    assert_eq!(snapshot.total_public_outputs, 4);
    assert_eq!(snapshot.total_ticks, 8);
    assert_eq!(snapshot.last_stop_reason.as_deref(), Some("max_new_tokens"));
}

#[test]
fn resident_token_stream_can_be_pumped_one_tick_at_a_time() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token stream pump: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token stream pump",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut stream = processor.into_token_stream("host_stream_0");
    let event = VulkanResidentTokenInputEvent::new("event_0", vec![1], 2).with_origin("test_host");
    let queued = stream.enqueue_external_event(event.clone()).unwrap();
    assert_eq!(queued.input_event, event);
    assert_eq!(queued.start_stream_tick, 0);
    assert_eq!(queued.enqueued_token_count, 1);
    assert!(!stream.snapshot().idle);

    let first = stream.pump_once(&device).unwrap();
    assert_eq!(first.stream_id, "host_stream_0");
    assert_eq!(
        first.status,
        VulkanResidentRunningStreamTickStatus::Processed
    );
    assert_eq!(first.stream_tick, Some(0));
    assert_eq!(first.input_token_id, Some(1));
    assert_eq!(
        first.input_route,
        Some(VulkanResidentPromptEventInputRoute::ExternalInput)
    );
    assert_eq!(
        first.output_event.as_ref().unwrap().input_event_id,
        "event_0"
    );
    assert_eq!(first.output_event.as_ref().unwrap().output_index, 0);
    assert_eq!(first.output_event.as_ref().unwrap().source_stream_tick, 0);

    let second = stream.pump_once(&device).unwrap();
    assert_eq!(second.stream_tick, Some(1));
    assert_eq!(
        second.input_route,
        Some(VulkanResidentPromptEventInputRoute::PrivateFeedback)
    );
    assert_eq!(
        second.output_event.as_ref().unwrap().input_event_id,
        "event_0"
    );
    assert_eq!(second.output_event.as_ref().unwrap().output_index, 1);
    assert_eq!(second.output_event.as_ref().unwrap().source_stream_tick, 1);

    let closing = stream.pump_once(&device).unwrap();
    assert_eq!(closing.stream_tick, Some(2));
    assert_eq!(
        closing.input_route,
        Some(VulkanResidentPromptEventInputRoute::PrivateFeedback)
    );
    assert!(closing.output_event.is_none());
    assert_eq!(closing.stop_reason.as_deref(), Some("max_new_tokens"));

    let idle = stream.pump_once(&device).unwrap();
    assert_eq!(idle.status, VulkanResidentRunningStreamTickStatus::Idle);
    assert_eq!(idle.stream_tick, None);
    assert!(idle.output_event.is_none());
    assert_eq!(idle.stop_reason.as_deref(), Some("max_new_tokens"));

    let snapshot = stream.snapshot();
    assert_eq!(snapshot.next_stream_tick, 3);
    assert!(snapshot.idle);
    assert_eq!(snapshot.total_public_outputs, 2);
    assert_eq!(snapshot.total_ticks, 4);
}

#[test]
fn resident_token_stream_can_pump_bounded_runtime_cycles() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token stream bounded pump: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token stream bounded pump",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut stream = processor.into_token_stream("host_stream_0");
    stream
        .enqueue_external_event(
            VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
        )
        .unwrap();

    let first_cycle = stream.pump_bounded(&device, 2).unwrap();
    assert_eq!(first_cycle.stream_id, "host_stream_0");
    assert_eq!(first_cycle.start_stream_tick, 0);
    assert_eq!(first_cycle.next_stream_tick, 2);
    assert_eq!(
        first_cycle.stop_condition,
        VulkanResidentTokenStreamPumpStopCondition::TickBudget
    );
    assert_eq!(first_cycle.processed_tick_count, 2);
    assert_eq!(first_cycle.idle_tick_count, 0);
    assert_eq!(first_cycle.output_events.len(), 2);
    assert_eq!(first_cycle.ticks.len(), 2);
    assert_eq!(first_cycle.ticks[0].stream_tick, Some(0));
    assert_eq!(first_cycle.ticks[1].stream_tick, Some(1));
    assert_eq!(
        first_cycle
            .output_events
            .iter()
            .map(|event| event.output_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(stream.snapshot().next_stream_tick, 2);
    assert!(!stream.snapshot().idle);

    let second_cycle = stream.pump_bounded(&device, 3).unwrap();
    assert_eq!(second_cycle.start_stream_tick, 2);
    assert_eq!(second_cycle.next_stream_tick, 4);
    assert_eq!(
        second_cycle.stop_condition,
        VulkanResidentTokenStreamPumpStopCondition::Idle
    );
    assert_eq!(second_cycle.processed_tick_count, 2);
    assert_eq!(second_cycle.idle_tick_count, 1);
    assert_eq!(second_cycle.output_events.len(), 1);
    assert_eq!(second_cycle.output_events[0].output_index, 2);
    assert_eq!(second_cycle.output_events[0].source_stream_tick, 2);
    assert_eq!(second_cycle.ticks.len(), 3);
    assert_eq!(second_cycle.ticks[0].stream_tick, Some(2));
    assert_eq!(second_cycle.ticks[1].stream_tick, Some(3));
    assert_eq!(second_cycle.ticks[2].stream_tick, None);
    assert_eq!(
        second_cycle.last_stop_reason.as_deref(),
        Some("max_new_tokens")
    );

    let snapshot = stream.snapshot();
    assert_eq!(snapshot.next_stream_tick, 4);
    assert!(snapshot.idle);
    assert_eq!(snapshot.total_public_outputs, 3);
    assert_eq!(snapshot.total_ticks, 5);

    let no_budget = stream.pump_bounded(&device, 0).unwrap();
    assert_eq!(
        no_budget.stop_condition,
        VulkanResidentTokenStreamPumpStopCondition::TickBudget
    );
    assert_eq!(no_budget.processed_tick_count, 0);
    assert_eq!(no_budget.idle_tick_count, 0);
    assert!(no_budget.output_events.is_empty());
    assert!(no_budget.ticks.is_empty());
    assert_eq!(no_budget.start_stream_tick, 4);
    assert_eq!(no_budget.next_stream_tick, 4);
}

#[test]
fn resident_feedback_cycle_restores_recurrent_state_when_eos_arrives_mid_cycle() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("requested Vulkan test device is unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping resident EOS feedback cycle: {error}");
            return;
        }
    };
    let create_stream = |stream_id: &str| {
        fixture_model_resident_greedy_model(&device, 16)
            .unwrap()
            .create_stream_processor(&device, 0)
            .unwrap()
            .into_token_stream(stream_id)
    };
    let event = VulkanResidentTokenInputEvent::new("eos_event", vec![1, 50_471, 1_413], 8)
        .with_stop_tokens(vec![510]);

    let mut scalar = create_stream("scalar_stream");
    scalar.enqueue_external_event(event.clone()).unwrap();
    let mut scalar_output = Vec::new();
    loop {
        let tick = scalar.pump_once(&device).unwrap();
        if let Some(output) = tick.output_event {
            scalar_output.push(output.token_id);
        }
        if tick.status == VulkanResidentRunningStreamTickStatus::Idle {
            break;
        }
    }
    let scalar_static_state = scalar
        .inner
        .processor
        ._mounted
        .buffers
        .state_buffers
        .iter()
        .filter(|state| state.static_byte_capacity.is_some())
        .map(|state| {
            (
                state.component_id.clone(),
                state.state_id.clone(),
                state.buffer.read_bytes(state.byte_capacity).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let scalar_snapshot = scalar.snapshot();
    drop(scalar);

    let mut batched = create_stream("batched_stream");
    batched.enqueue_external_event(event).unwrap();
    let mut batched_output = Vec::new();
    loop {
        let cycle = batched.pump_bounded(&device, 4).unwrap();
        batched_output.extend(cycle.output_events.iter().map(|output| output.token_id));
        if cycle.stop_condition == VulkanResidentTokenStreamPumpStopCondition::Idle {
            break;
        }
    }
    let batched_static_state = batched
        .inner
        .processor
        ._mounted
        .buffers
        .state_buffers
        .iter()
        .filter(|state| state.static_byte_capacity.is_some())
        .map(|state| {
            (
                state.component_id.clone(),
                state.state_id.clone(),
                state.buffer.read_bytes(state.byte_capacity).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let batched_snapshot = batched.snapshot();

    assert_eq!(scalar_output, vec![510]);
    assert_eq!(batched_output, scalar_output);
    assert_eq!(
        batched_snapshot.next_stream_tick,
        scalar_snapshot.next_stream_tick
    );
    assert_eq!(
        batched_snapshot.last_stop_reason,
        scalar_snapshot.last_stop_reason
    );
    assert_eq!(batched_static_state, scalar_static_state);
}
