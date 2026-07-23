#[test]
fn resident_pedal_runner_executes_layer_00_end_to_end() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping layer_00 resident pedal runner: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) = layer_00_level_1_loaded_kernel_pack(&mounted, &mounted_bound)
    else {
        eprintln!("skipping layer_00 resident pedal runner: no GLSL to SPIR-V compiler found");
        return;
    };
    load_layer_00_parameters(&mounted, &tensor_index);
    write_layer_00_unit_input_and_zero_state(&mounted);

    let runner = mounted
        .create_resident_pedal_runner(&device, &mounted_bound, "layer_00", &loaded_manifest)
        .unwrap();
    assert_eq!(runner.pedal_id, "layer_00");
    assert_eq!(runner.dispatch_count(), 16);
    assert_eq!(runner.total_descriptor_count, 52);
    assert_eq!(runner.total_push_constant_byte_count, 0);

    let run = runner
        .run_with_stream_control(
            &device,
            VulkanMountedPlacedStreamControl {
                stream_tick: 7,
                control_flags: 0,
                dynamic_state_capacity_activations: mounted
                    .buffers
                    .dynamic_state_capacity_activations
                    as u32,
            },
        )
        .unwrap();
    assert_eq!(run.pedal_id, "layer_00");
    assert_eq!(run.dispatch_count(), 16);
    assert_eq!(
        run.node_ids(),
        vec![
            "operator_norm",
            "conv_in_projection",
            "split_b_c_x",
            "input_gate",
            "temporal_memory_update",
            "depthwise_temporal_conv",
            "output_gate",
            "conv_out_projection",
            "operator_residual",
            "ffn_norm",
            "ffn_gate_projection",
            "ffn_up_projection",
            "ffn_gate_activation",
            "ffn_gate_multiply",
            "ffn_down_projection",
            "ffn_residual",
        ]
    );

    let final_residual_dispatch = mounted_bound.dispatch("layer_00", "ffn_residual").unwrap();
    let final_residual_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(final_residual_dispatch)
        .unwrap();
    assert_eq!(
        final_residual_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x86, 0x3f, 0x82, 0x3f, 0x81, 0x3f, 0x7e, 0x3f, 0x83, 0x3f, 0x83, 0x3f, 0x83, 0x3f,
            0x83, 0x3f,
        ]
    );
}

#[test]
fn resident_input_transducer_feeds_layer_00_from_token_embedding() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident input transducer: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(loaded_manifest) = layer_00_level_1_loaded_kernel_pack(&mounted, &mounted_bound)
    else {
        eprintln!("skipping resident input transducer: no GLSL to SPIR-V compiler found");
        return;
    };
    let Some(input_transducer_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source(
            "embedding_lookup_bf16_65536x1024.comp",
        )
    else {
        eprintln!("skipping resident input transducer: no GLSL to SPIR-V compiler found");
        return;
    };
    let graph = fixture_model_execution_graph();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let transducer_parameter_plan = VulkanPermanentParameterBufferPlan::from_transducer_parameters(
        "gpu0",
        &resource_plan,
        Some(&tensor_index),
    )
    .unwrap();
    assert_eq!(transducer_parameter_plan.parameter_count, 2);
    assert_eq!(
        transducer_parameter_plan.total_byte_capacity,
        Some(134_219_776)
    );
    assert!(transducer_parameter_plan.unresolved_tensors.is_empty());
    let embed_tokens = transducer_parameter_plan
        .parameters
        .iter()
        .find(|parameter| parameter.tensor == FIXTURE_MODEL_EMBED_TOKENS_TENSOR)
        .unwrap();
    assert_eq!(embed_tokens.use_count, 2);
    assert_eq!(
        embed_tokens.byte_capacity,
        Some(FIXTURE_MODEL_EMBED_TOKENS_BYTES)
    );
    let transducer_parameter_buffers = transducer_parameter_plan.allocate_buffers(&device).unwrap();
    assert_eq!(
        transducer_parameter_buffers.total_byte_capacity,
        134_219_776
    );
    assert!(
        transducer_parameter_buffers
            .parameter_buffer("model.embedding_norm.weight")
            .is_some()
    );
    let loaded_embedding = transducer_parameter_buffers
        .load_parameter_from_tensor_index(&tensor_index, FIXTURE_MODEL_EMBED_TOKENS_TENSOR)
        .unwrap();
    assert_eq!(
        loaded_embedding.byte_count,
        FIXTURE_MODEL_EMBED_TOKENS_BYTES
    );
    let token_id = 1u32;

    let input_transducer_runner =
        VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding(
            &device,
            &mounted,
            &transducer_parameter_buffers,
            &input_transducer_spirv_words,
            &fixture_model_input_embedding_transducer_spec(),
        )
        .unwrap();
    assert_eq!(
        input_transducer_runner.transducer_id,
        FIXTURE_MODEL_TOKEN_EMBEDDING_TRANSDUCER_ID
    );
    assert_eq!(
        input_transducer_runner.parameter_tensor,
        FIXTURE_MODEL_EMBED_TOKENS_TENSOR
    );
    assert_eq!(
        input_transducer_runner.output_signal_id,
        FIXTURE_MODEL_INPUT_FRAME_SIGNAL
    );
    assert_eq!(input_transducer_runner.descriptor_count, 3);
    assert_eq!(input_transducer_runner.workgroup_count_x, 2);
    assert_eq!(input_transducer_runner.push_constant_byte_count, 0);

    let transducer_run = input_transducer_runner
        .run_token_id(&device, token_id)
        .unwrap();
    assert_eq!(
        transducer_run.transducer_id,
        FIXTURE_MODEL_TOKEN_EMBEDDING_TRANSDUCER_ID
    );
    assert_eq!(transducer_run.token_id, token_id);
    assert_eq!(
        transducer_run.output_signal_id,
        FIXTURE_MODEL_INPUT_FRAME_SIGNAL
    );
    assert_eq!(transducer_run.dispatch_count, 1);
    assert_eq!(transducer_run.descriptor_count, 3);
    assert_eq!(transducer_run.workgroup_count_x, 2);
    assert_eq!(transducer_run.push_constant_byte_count, 0);
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

    load_layer_00_parameters(&mounted, &tensor_index);
    zero_fixture_model_temporal_memory(&mounted, "layer_00");
    let runner = mounted
        .create_resident_pedal_runner(&device, &mounted_bound, "layer_00", &loaded_manifest)
        .unwrap();
    let run = runner
        .run_with_stream_control(&device, fixture_model_stream_control(&mounted, 0))
        .unwrap();
    assert_eq!(run.pedal_id, "layer_00");
    assert_eq!(run.dispatch_count(), 16);

    let final_residual_dispatch = mounted_bound.dispatch("layer_00", "ffn_residual").unwrap();
    let final_residual_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(final_residual_dispatch)
        .unwrap();
    assert_eq!(
        final_residual_bindings[2].buffer.read_bytes(16).unwrap(),
        vec![
            0x84, 0x3c, 0x09, 0x3c, 0xbf, 0x3b, 0x90, 0xbc, 0x30, 0x3b, 0xc6, 0xba, 0x8e, 0x3b,
            0x34, 0xbb,
        ]
    );
}

#[test]
fn resident_output_transducer_projects_output_frame_to_logits() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident output transducer: {error}");
            return;
        }
    };
    let (tensor_index, mounted, _manifest, _mounted_bound) =
        mount_fixture_model_single_device_stream_circuit(&device);
    let Some(embedding_norm_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source("rms_norm_bf16_serial.comp")
    else {
        eprintln!("skipping resident output transducer: no GLSL to SPIR-V compiler found");
        return;
    };
    let Some(tied_projection_spirv_words) =
        crate::vulkan_compute::compile_test_shader_words_from_source(
            "tied_output_projection_bf16_65536x1024_to_f32.comp",
        )
    else {
        eprintln!("skipping resident output transducer: no GLSL to SPIR-V compiler found");
        return;
    };
    let graph = fixture_model_execution_graph();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let transducer_parameter_plan = VulkanPermanentParameterBufferPlan::from_transducer_parameters(
        "gpu0",
        &resource_plan,
        Some(&tensor_index),
    )
    .unwrap();
    let transducer_parameter_buffers = transducer_parameter_plan.allocate_buffers(&device).unwrap();
    let loaded_transducers = transducer_parameter_buffers
        .load_from_tensor_index(&tensor_index)
        .unwrap();
    assert_eq!(loaded_transducers.parameter_count, 2);
    assert_eq!(loaded_transducers.loaded_count, 2);
    assert_eq!(loaded_transducers.total_bytes_loaded, 134_219_776);

    write_fixture_model_constant_output_frame(&mounted, [0x80, 0x3f]);
    let runner = VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
        &device,
        &mounted,
        &transducer_parameter_buffers,
        &embedding_norm_spirv_words,
        &tied_projection_spirv_words,
        &fixture_model_output_transducer_spec(),
    )
    .unwrap();
    assert_eq!(runner.transducer_id, "output_transducer");
    assert_eq!(runner.input_signal_id, FIXTURE_MODEL_OUTPUT_FRAME_SIGNAL);
    assert_eq!(runner.logits_byte_capacity, FIXTURE_MODEL_LOGITS_BYTES);
    assert_eq!(runner.dispatch_count, 2);
    assert_eq!(runner.total_descriptor_count, 6);
    assert_eq!(runner.total_push_constant_byte_count, 0);

    let run = runner.run(&device).unwrap();
    assert_eq!(run.transducer_id, "output_transducer");
    assert_eq!(run.input_signal_id, FIXTURE_MODEL_OUTPUT_FRAME_SIGNAL);
    assert_eq!(run.dispatch_count, 2);
    assert_eq!(
        run.node_ids,
        vec![
            FIXTURE_MODEL_OUTPUT_EMBEDDING_NORM_TRANSDUCER_ID.to_string(),
            FIXTURE_MODEL_TIED_OUTPUT_PROJECTION_TRANSDUCER_ID.to_string(),
        ]
    );
    assert_eq!(run.descriptor_counts, vec![3, 3]);
    assert_eq!(run.workgroup_counts_x, vec![1, 32_768]);
    assert_eq!(run.push_constant_byte_counts, vec![0, 0]);
    assert_eq!(run.logits_byte_capacity, FIXTURE_MODEL_LOGITS_BYTES);

    assert_eq!(
        runner.read_normalized_frame_bytes(16).unwrap(),
        vec![
            0x1a, 0x40, 0x5b, 0x40, 0x56, 0x40, 0x58, 0x40, 0x59, 0x40, 0x55, 0x40, 0x4e, 0x40,
            0x3c, 0x40,
        ]
    );
    assert_eq!(
        runner.read_logits_bytes(16).unwrap(),
        vec![
            0x86, 0x09, 0xa6, 0x3f, 0x18, 0x7a, 0x4f, 0xbd, 0x21, 0xee, 0x7a, 0xc0, 0xdd, 0x90,
            0xa1, 0x3f,
        ]
    );
}

#[test]
fn resident_greedy_sampler_selects_largest_logit() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident sampler: {error}");
            return;
        }
    };
    let Some(sampler_spirv_words) = crate::vulkan_compute::compile_test_shader_words_from_source(
        "greedy_sampler_f32_65536.comp",
    ) else {
        eprintln!("skipping resident sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let sampler_kernels = greedy_sampler_test_kernels(sampler_spirv_words);

    let logits_buffer = device
        .create_resident_buffer(FIXTURE_MODEL_LOGITS_BYTES)
        .unwrap();
    let mut logits = vec![0u8; FIXTURE_MODEL_LOGITS_BYTES];
    let token_7 = 7usize;
    let token_1024 = 1_024usize;
    logits[(token_7 * 4)..((token_7 + 1) * 4)].copy_from_slice(&3.5f32.to_le_bytes());
    logits[(token_1024 * 4)..((token_1024 + 1) * 4)].copy_from_slice(&9.25f32.to_le_bytes());
    logits_buffer.write_bytes(&logits).unwrap();

    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0x0000_0007_ffff_ffff,
                control_flags: 0,
                dynamic_state_capacity_activations: 8,
            },
        ))
        .unwrap();
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer.clone(),
        &logits_buffer,
        FIXTURE_MODEL_LOGITS_BYTES,
        &sampler_kernels,
        &fixture_model_greedy_sampler_spec(),
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 8,
            random_seed: 0,
        },
    )
    .unwrap();
    assert_eq!(runner.sampler_id, FIXTURE_MODEL_GREEDY_SAMPLER_PEDAL_ID);
    assert_eq!(runner.logits_byte_capacity, FIXTURE_MODEL_LOGITS_BYTES);
    assert_eq!(
        runner.output_byte_capacity,
        FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES
    );
    assert_eq!(runner.descriptor_count, 3);
    assert_eq!(runner.workgroup_count_x, 1);
    assert_eq!(runner.push_constant_byte_count, 0);

    let run = runner.run(&device).unwrap();
    assert_eq!(run.sampler_id, FIXTURE_MODEL_GREEDY_SAMPLER_PEDAL_ID);
    assert_eq!(run.token_id, token_1024 as u32);
    assert_eq!(run.selected_logit_bits, 9.25f32.to_bits());
    assert_eq!(run.control_flags, 0);
    assert_eq!(run.descriptor_count, 3);
    assert_eq!(run.workgroup_count_x, 1);
    assert_eq!(run.push_constant_byte_count, 0);
    assert_eq!(
        runner.read_output_bytes().unwrap(),
        vec![
            0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x14, 0x41, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    );
    let control = stream_control_buffer
        .read_bytes(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
        .unwrap();
    assert_eq!(u32::from_le_bytes(control[0..4].try_into().unwrap()), 1_024);
    assert_eq!(u32::from_le_bytes(control[4..8].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(control[8..12].try_into().unwrap()), 8);
    assert_eq!(runner.completed_run_at(0x0000_0007_ffff_ffff).unwrap(), run);
}

#[test]
fn resident_temperature_top_k_top_p_sampler_matches_explicit_random_signal() {
    const VOCAB_SIZE: usize = 64;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
    const SEED: u32 = 0x5eed_1234;
    const TOP_TOKENS: [u32; 4] = [7, 8, 19, 51];

    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident sampled sampler: {error}");
            return;
        }
    };
    let partition_count = 4;
    let Some(sampler_kernels) = compile_temperature_top_k_top_p_sampler_test_kernels(
        VOCAB_SIZE,
        1.0,
        4,
        1.0,
        partition_count,
        16,
    ) else {
        eprintln!("skipping resident sampled sampler: no GLSL to SPIR-V compiler found");
        return;
    };

    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let mut logits = vec![-100.0f32; VOCAB_SIZE];
    for token_id in TOP_TOKENS {
        logits[token_id as usize] = 2.0;
    }
    logits_buffer
        .write_bytes(
            &logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "temperature_top_k_top_p_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 4,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: 4,
        runtime_parameterized: false,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: partition_count as usize * 4 * 8,
    };
    let mut invalid_spec = spec.clone();
    invalid_spec.scratch_byte_capacity -= 8;
    let invalid = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer.clone(),
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &invalid_spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: SEED,
        },
    )
    .err()
    .expect("undersized sampler scratch must be rejected");
    assert!(
        invalid
            .to_string()
            .contains("invalid resident sampling spec")
    );
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: SEED,
        },
    )
    .unwrap();

    let mut selected_tokens = Vec::new();
    for stream_tick in 0..16u32 {
        let run = runner.run(&device).unwrap();
        let random_bits = sampler_test_hash_u32(SEED ^ stream_tick);
        let selected_index = (((random_bits >> 8) as u64 * 4) >> 24) as usize;
        let expected = TOP_TOKENS[selected_index];
        assert_eq!(run.token_id, expected);
        assert_eq!(run.selected_logit_bits, 2.0f32.to_bits());
        assert_eq!(run.control_flags, 1);
        assert_eq!(runner.dispatch_count, 2);
        assert_eq!(run.descriptor_count, 6);
        assert_eq!(run.workgroup_count_x, partition_count + 1);
        selected_tokens.push(run.token_id);
    }
    assert!(
        TOP_TOKENS
            .iter()
            .all(|token_id| selected_tokens.contains(token_id))
    );
}

#[test]
fn resident_repetition_sampler_tracks_prompt_and_feedback_tokens_on_gpu() {
    const VOCAB_SIZE: usize = 64;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
    const PARTITION_COUNT: u32 = 4;
    const REPETITION_PENALTY: f32 = 1.1;

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident repetition sampler: {error}");
            return;
        }
    };
    let Some(kernels) = compile_repetition_temperature_sampler_test_kernels(
        VOCAB_SIZE,
        REPETITION_PENALTY,
        1,
        PARTITION_COUNT,
        16,
    ) else {
        eprintln!("skipping resident repetition sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let write_logits = |values: &[f32]| {
        logits_buffer
            .write_bytes(
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "repetition_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: REPETITION_PENALTY,
        top_k_capacity: 1,
        runtime_parameterized: false,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize * 8,
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer.clone(),
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 7,
        },
    )
    .unwrap();
    assert_eq!(runner.dispatch_count, 3);
    assert_eq!(runner.descriptor_count, 9);
    assert_eq!(runner.workgroup_count_x, 6);

    let mut logits = vec![-100.0; VOCAB_SIZE];
    logits[7] = 10.0;
    logits[8] = 9.6;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 7);

    runner.record_input_tokens(&device, &[7]).unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 8);

    logits.fill(-100.0);
    logits[7] = -1.0;
    logits[8] = -1.05;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 8);

    logits.fill(-100.0);
    logits[9] = 5.0;
    logits[10] = 4.8;
    write_logits(&logits);
    stream_control_buffer
        .write_bytes(&9u32.to_le_bytes())
        .unwrap();
    device
        .run_resident_kernel_dispatch(&runner.input_tracking_dispatches()[0], &[])
        .unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 10);
}

#[test]
fn resident_runtime_sampler_applies_presence_penalty_from_gpu_state() {
    const VOCAB_SIZE: usize = 64;
    const PARTITION_COUNT: u32 = 4;
    const TOP_K_CAPACITY: u32 = 8;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping runtime presence sampler: {error}");
            return;
        }
    };
    let Some(kernels) = compile_runtime_temperature_sampler_test_kernels(
        VOCAB_SIZE,
        TOP_K_CAPACITY,
        PARTITION_COUNT,
        16,
    ) else {
        eprintln!("skipping runtime presence sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let write_logits = |values: &[f32]| {
        logits_buffer
            .write_bytes(
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "runtime_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 1.5,
        repetition_penalty: 1.0,
        top_k_capacity: TOP_K_CAPACITY,
        runtime_parameterized: true,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize
            * TOP_K_CAPACITY as usize
            * 2
            * std::mem::size_of::<u32>(),
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 7,
        },
    )
    .unwrap();

    let mut logits = vec![-100.0; VOCAB_SIZE];
    logits[7] = 10.0;
    logits[8] = 9.0;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 7);

    runner.record_input_tokens(&device, &[7]).unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 8);

    logits.fill(-100.0);
    logits[7] = -1.0;
    logits[9] = -1.25;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 9);

    runner.capture_token_state().unwrap();
    runner.record_input_tokens(&device, &[11]).unwrap();
    logits.fill(-100.0);
    logits[11] = 10.0;
    logits[12] = 9.0;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 12);
    runner.restore_token_state().unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 11);
}

#[test]
fn speculative_sampler_views_isolate_hypothetical_presence_state() {
    const VOCAB_SIZE: usize = 64;
    const PARTITION_COUNT: u32 = 4;
    const TOP_K_CAPACITY: u32 = 8;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping speculative presence sampler: {error}");
            return;
        }
    };
    let Some(kernels) = compile_runtime_temperature_sampler_test_kernels(
        VOCAB_SIZE,
        TOP_K_CAPACITY,
        PARTITION_COUNT,
        16,
    ) else {
        eprintln!("skipping speculative presence sampler: no GLSL compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "runtime_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 1.5,
        repetition_penalty: 1.0,
        top_k_capacity: TOP_K_CAPACITY,
        runtime_parameterized: true,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize
            * TOP_K_CAPACITY as usize
            * 2
            * std::mem::size_of::<u32>(),
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 7,
        },
    )
    .unwrap();
    runner.record_input_tokens(&device, &[7]).unwrap();

    let batched_logits = device
        .create_resident_buffer(2 * LOGITS_BYTE_CAPACITY)
        .unwrap();
    let mut lane_zero = vec![-100.0f32; VOCAB_SIZE];
    lane_zero[8] = 10.0;
    lane_zero[11] = 9.0;
    let mut lane_one = vec![-100.0f32; VOCAB_SIZE];
    lane_one[10] = 10.0;
    lane_one[12] = 9.0;
    let batched_bytes = lane_zero
        .iter()
        .chain(&lane_one)
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    batched_logits.write_bytes(&batched_bytes).unwrap();

    let view_zero = runner
        .create_logits_view(&device, &batched_logits, 0, &kernels, &spec)
        .unwrap();
    view_zero.prepare_token_state(&device, &[8]).unwrap();
    view_zero.prepare_stream_tick(0, 32).unwrap();
    view_zero.record(&device).unwrap();
    device
        .run_recorded_resident_kernel_sequence(&view_zero.sequence)
        .unwrap();
    assert_eq!(runner.completed_run_at(0).unwrap().token_id, 11);

    let view_one = runner
        .create_logits_view(
            &device,
            &batched_logits,
            LOGITS_BYTE_CAPACITY,
            &kernels,
            &spec,
        )
        .unwrap();
    view_one.prepare_token_state(&device, &[9, 10]).unwrap();
    view_one.prepare_stream_tick(1, 32).unwrap();
    view_one.record(&device).unwrap();
    device
        .run_recorded_resident_kernel_sequence(&view_one.sequence)
        .unwrap();
    assert_eq!(runner.completed_run_at(1).unwrap().token_id, 12);

    logits_buffer
        .write_bytes(
            &lane_zero
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 8);
}

#[test]
fn resident_temperature_top_64_sampler_matches_explicit_random_signal() {
    const VOCAB_SIZE: usize = 262_144;
    const TOP_K: u32 = 64;
    const PARTITION_COUNT: u32 = 128;
    const LOCAL_SIZE_X: u32 = 256;
    const SEED: u32 = 46;
    const HISTORY_CAPACITY: usize = 32;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();

    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping resident top-64 sampler: NERVE_TEST_VULKAN_DEVICE_INDEX is unset");
        return;
    };
    let device = match VulkanComputeDevice::new_for_physical_device_index(device_index) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident top-64 sampler: {error}");
            return;
        }
    };
    let Some(sampler_kernels) = compile_temperature_top_k_top_p_sampler_test_kernels(
        VOCAB_SIZE,
        1.0,
        TOP_K,
        0.95,
        PARTITION_COUNT,
        LOCAL_SIZE_X,
    ) else {
        eprintln!("skipping resident top-64 sampler: no GLSL to SPIR-V compiler found");
        return;
    };

    let mut top_tokens = (0..TOP_K)
        .map(|index| (index * 4_093 + 7) % VOCAB_SIZE as u32)
        .collect::<Vec<_>>();
    top_tokens.sort_unstable();
    let mut logits = vec![-100.0f32; VOCAB_SIZE];
    for token_id in &top_tokens {
        logits[*token_id as usize] = 2.0;
    }
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    logits_buffer
        .write_bytes(
            &logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: HISTORY_CAPACITY as u32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "temperature_top_k_top_p_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: TOP_K,
        top_p: 0.95,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: TOP_K,
        runtime_parameterized: false,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize * TOP_K as usize * 8,
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: HISTORY_CAPACITY,
            random_seed: SEED,
        },
    )
    .unwrap();

    // With 64 equal top-k weights, top-p=0.95 retains the first 61.
    for stream_tick in 0..16u32 {
        let run = runner.run(&device).unwrap();
        let random_bits = sampler_test_hash_u32(SEED ^ stream_tick);
        let selected_index = (((random_bits >> 8) as u64 * 61) >> 24) as usize;
        assert_eq!(run.token_id, top_tokens[selected_index]);
        assert_eq!(run.selected_logit_bits, 2.0f32.to_bits());
    }
}

#[test]
