fn load_fixture_model_conv_layer_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
) {
    for suffix in [
        "operator_norm.weight",
        "conv.in_proj.weight",
        "conv.conv.weight",
        "conv.out_proj.weight",
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ] {
        let tensor = format!("model.layers.{layer_index}.{suffix}");
        mounted
            .parameter_buffers
            .load_parameter_from_tensor_index(tensor_index, &tensor)
            .unwrap();
    }
}

fn load_fixture_model_attention_layer_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
) {
    for suffix in [
        "operator_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_layernorm.weight",
        "self_attn.k_layernorm.weight",
        "self_attn.out_proj.weight",
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ] {
        let tensor = format!("model.layers.{layer_index}.{suffix}");
        mounted
            .parameter_buffers
            .load_parameter_from_tensor_index(tensor_index, &tensor)
            .unwrap();
    }
}

fn write_layer_00_unit_input_and_zero_state(mounted: &VulkanMountedPlacedStreamCircuit) {
    write_layer_00_constant_input(mounted, [0x80, 0x3f]);
    zero_fixture_model_temporal_memory(mounted, "layer_00");
}

fn write_layer_00_constant_input(
    mounted: &VulkanMountedPlacedStreamCircuit,
    bf16_little_endian: [u8; 2],
) {
    let mut input_frame = Vec::with_capacity(2_048);
    for _ in 0..1024 {
        input_frame.extend_from_slice(&bf16_little_endian);
    }
    mounted
        .boundary_io
        .input_buffer("input_frame")
        .unwrap()
        .buffer
        .write_bytes(&input_frame)
        .unwrap();
}

fn write_fixture_model_constant_output_frame(
    mounted: &VulkanMountedPlacedStreamCircuit,
    bf16_little_endian: [u8; 2],
) {
    let mut output_frame = Vec::with_capacity(FIXTURE_MODEL_FRAME_BYTES);
    for _ in 0..FIXTURE_MODEL_HIDDEN_SIZE {
        output_frame.extend_from_slice(&bf16_little_endian);
    }
    mounted
        .boundary_io
        .output_buffer(FIXTURE_MODEL_OUTPUT_FRAME_SIGNAL)
        .unwrap()
        .buffer
        .write_bytes(&output_frame)
        .unwrap();
}

fn zero_fixture_model_temporal_memory(mounted: &VulkanMountedPlacedStreamCircuit, pedal_id: &str) {
    let temporal_memory = mounted
        .buffers
        .state_buffer(pedal_id, "temporal_memory")
        .unwrap();
    temporal_memory
        .buffer
        .write_bytes(&vec![0u8; temporal_memory.byte_capacity])
        .unwrap();
}

fn zero_fixture_model_kv_memory(mounted: &VulkanMountedPlacedStreamCircuit, pedal_id: &str) {
    let kv_memory = mounted.buffers.state_buffer(pedal_id, "kv_memory").unwrap();
    kv_memory
        .buffer
        .write_bytes(&vec![0u8; kv_memory.byte_capacity])
        .unwrap();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureModelLayerKind {
    ShortConv,
    Attention,
}

fn fixture_model_layer_kind(layer_index: usize) -> FixtureModelLayerKind {
    match layer_index {
        0 | 1 | 3 | 5 | 7 | 9 | 11 | 13 => FixtureModelLayerKind::ShortConv,
        2 | 4 | 6 | 8 | 10 | 12 => FixtureModelLayerKind::Attention,
        _ => panic!("unknown FIXTURE_MODEL layer index {layer_index}"),
    }
}

fn fixture_model_layer_id(layer_index: usize) -> String {
    format!("layer_{layer_index:02}")
}

fn fixture_model_prefix_pedal_ids(last_layer_index: usize) -> Vec<String> {
    (0..=last_layer_index).map(fixture_model_layer_id).collect()
}

fn load_fixture_model_layer_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    layer_index: usize,
) {
    match fixture_model_layer_kind(layer_index) {
        FixtureModelLayerKind::ShortConv => {
            load_fixture_model_conv_layer_parameters(mounted, tensor_index, layer_index);
        }
        FixtureModelLayerKind::Attention => {
            load_fixture_model_attention_layer_parameters(mounted, tensor_index, layer_index);
        }
    }
}

fn zero_fixture_model_layer_state(mounted: &VulkanMountedPlacedStreamCircuit, layer_index: usize) {
    let pedal_id = fixture_model_layer_id(layer_index);
    match fixture_model_layer_kind(layer_index) {
        FixtureModelLayerKind::ShortConv => {
            zero_fixture_model_temporal_memory(mounted, &pedal_id);
        }
        FixtureModelLayerKind::Attention => {
            zero_fixture_model_kv_memory(mounted, &pedal_id);
        }
    }
}

fn prepare_fixture_model_resident_prefix(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
    last_layer_index: usize,
) -> Vec<String> {
    for layer_index in 0..=last_layer_index {
        load_fixture_model_layer_parameters(mounted, tensor_index, layer_index);
    }

    write_layer_00_unit_input_and_zero_state(mounted);
    for layer_index in 1..=last_layer_index {
        zero_fixture_model_layer_state(mounted, layer_index);
    }

    fixture_model_prefix_pedal_ids(last_layer_index)
}

fn fixture_model_stream_control(
    mounted: &VulkanMountedPlacedStreamCircuit,
    stream_tick: u64,
) -> VulkanMountedPlacedStreamControl {
    VulkanMountedPlacedStreamControl {
        stream_tick,
        control_flags: 0,
        dynamic_state_capacity_activations: mounted.buffers.dynamic_state_capacity_activations
            as u32,
    }
}

fn create_fixture_model_resident_prefix_runner(
    device: &VulkanComputeDevice,
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    pedal_ids: &[String],
) -> VulkanMountedPlacedResidentPedalboardRunner {
    mounted
        .create_resident_pedalboard_runner(
            device,
            mounted_bound,
            pedal_ids.iter().map(String::as_str),
            loaded_manifest,
        )
        .unwrap()
}

fn assert_fixture_model_resident_prefix_runner(
    runner: &VulkanMountedPlacedResidentPedalboardRunner,
    pedal_ids: &[String],
    dispatch_count: usize,
    descriptor_count: usize,
    push_constant_byte_count: u32,
) {
    let expected_pedal_ids = pedal_ids.iter().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(runner.device_id, "gpu0");
    assert_eq!(runner.pedal_count(), pedal_ids.len());
    assert_eq!(runner.pedal_ids(), expected_pedal_ids);
    assert_eq!(runner.dispatch_count(), dispatch_count);
    assert_eq!(runner.total_descriptor_count, descriptor_count);
    assert_eq!(
        runner.total_push_constant_byte_count,
        push_constant_byte_count
    );
}

fn assert_fixture_model_resident_prefix_run(
    run: &VulkanMountedPlacedResidentPedalboardRun,
    pedal_ids: &[String],
    dispatch_count: usize,
) {
    let expected_pedal_ids = pedal_ids.iter().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(run.device_id, "gpu0");
    assert_eq!(run.pedal_count(), pedal_ids.len());
    assert_eq!(run.pedal_ids(), expected_pedal_ids);
    assert_eq!(run.dispatch_count(), dispatch_count);
}

fn layer_00_level_1_loaded_kernel_pack(
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
    loaded_kernel_pack_for_dispatch_shaders(
        mounted,
        mounted_bound,
        &[
            ("layer_00", "operator_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_00",
                "conv_in_projection",
                "linear_bf16_1024x3072.comp",
            ),
            ("layer_00", "split_b_c_x", "split_bf16_3072_to_3x1024.comp"),
            ("layer_00", "input_gate", "multiply_bf16_1024.comp"),
            (
                "layer_00",
                "temporal_memory_update",
                "rolling_state_update_bf16_3x1024.comp",
            ),
            (
                "layer_00",
                "depthwise_temporal_conv",
                "depthwise_conv1d_bf16_3x1024.comp",
            ),
            ("layer_00", "output_gate", "multiply_bf16_1024.comp"),
            (
                "layer_00",
                "conv_out_projection",
                "linear_bf16_1024x1024.comp",
            ),
            ("layer_00", "operator_residual", "add_bf16_1024.comp"),
            ("layer_00", "ffn_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_00",
                "ffn_gate_projection",
                "linear_bf16_1024x2560.comp",
            ),
            (
                "layer_00",
                "ffn_up_projection",
                "linear_bf16_1024x2560.comp",
            ),
            ("layer_00", "ffn_gate_activation", "silu_bf16_2560.comp"),
            ("layer_00", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
            (
                "layer_00",
                "ffn_down_projection",
                "linear_bf16_2560x1024.comp",
            ),
            ("layer_00", "ffn_residual", "add_bf16_1024.comp"),
        ],
    )
}

fn fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families(
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
    fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
        mounted,
        mounted_bound,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    )
}

fn fixture_model_level_1_loaded_kernel_pack_for_conv_and_attention_families_with_attention_shader(
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    attention_shader: &str,
) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
    loaded_kernel_pack_for_dispatch_shaders(
        mounted,
        mounted_bound,
        &[
            ("layer_00", "operator_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_00",
                "conv_in_projection",
                "linear_bf16_1024x3072.comp",
            ),
            ("layer_00", "split_b_c_x", "split_bf16_3072_to_3x1024.comp"),
            ("layer_00", "input_gate", "multiply_bf16_1024.comp"),
            (
                "layer_00",
                "temporal_memory_update",
                "rolling_state_update_bf16_3x1024.comp",
            ),
            (
                "layer_00",
                "depthwise_temporal_conv",
                "depthwise_conv1d_bf16_3x1024.comp",
            ),
            ("layer_00", "output_gate", "multiply_bf16_1024.comp"),
            (
                "layer_00",
                "conv_out_projection",
                "linear_bf16_1024x1024.comp",
            ),
            ("layer_00", "operator_residual", "add_bf16_1024.comp"),
            ("layer_00", "ffn_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_00",
                "ffn_gate_projection",
                "linear_bf16_1024x2560.comp",
            ),
            (
                "layer_00",
                "ffn_up_projection",
                "linear_bf16_1024x2560.comp",
            ),
            ("layer_00", "ffn_gate_activation", "silu_bf16_2560.comp"),
            ("layer_00", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
            (
                "layer_00",
                "ffn_down_projection",
                "linear_bf16_2560x1024.comp",
            ),
            ("layer_00", "ffn_residual", "add_bf16_1024.comp"),
            ("layer_02", "operator_norm", "rms_norm_bf16_serial.comp"),
            ("layer_02", "q_projection", "linear_bf16_1024x1024.comp"),
            ("layer_02", "k_projection", "linear_bf16_1024x512.comp"),
            ("layer_02", "v_projection", "linear_bf16_1024x512.comp"),
            (
                "layer_02",
                "q_head_norm",
                "rms_norm_per_head_bf16_16x64.comp",
            ),
            (
                "layer_02",
                "k_head_norm",
                "rms_norm_per_head_bf16_8x64.comp",
            ),
            ("layer_02", "q_rope", "rotary_bf16_16x64.comp"),
            ("layer_02", "k_rope", "rotary_bf16_8x64.comp"),
            (
                "layer_02",
                "kv_memory_append",
                "append_kv_state_bf16_8x64.comp",
            ),
            ("layer_02", "attention_read", attention_shader),
            (
                "layer_02",
                "attention_out_projection",
                "linear_bf16_1024x1024.comp",
            ),
            ("layer_02", "operator_residual", "add_bf16_1024.comp"),
            ("layer_02", "ffn_norm", "rms_norm_bf16_serial.comp"),
            (
                "layer_02",
                "ffn_gate_projection",
                "linear_bf16_1024x2560.comp",
            ),
            (
                "layer_02",
                "ffn_up_projection",
                "linear_bf16_1024x2560.comp",
            ),
            ("layer_02", "ffn_gate_activation", "silu_bf16_2560.comp"),
            ("layer_02", "ffn_gate_multiply", "multiply_bf16_2560.comp"),
            (
                "layer_02",
                "ffn_down_projection",
                "linear_bf16_2560x1024.comp",
            ),
            ("layer_02", "ffn_residual", "add_bf16_1024.comp"),
        ],
    )
}

fn loaded_kernel_pack_for_dispatch_shaders(
    mounted: &VulkanMountedPlacedStreamCircuit,
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
    dispatch_shaders: &[(&str, &str, &str)],
) -> Option<VulkanLoadedReusableKernelArtifactManifest> {
    let mut loaded_artifacts = Vec::new();
    let mut loaded_families = BTreeSet::new();
    let mut total_word_count = 0usize;

    for (pedal_id, node_id, shader_file) in dispatch_shaders {
        let dispatch = mounted_bound.dispatch(pedal_id, node_id).unwrap();
        if !loaded_families.insert(dispatch.reusable_family_id.clone()) {
            continue;
        }
        let spirv_words =
            crate::vulkan_compute::compile_test_shader_words_from_source(shader_file)?;
        total_word_count = total_word_count.checked_add(spirv_words.len())?;
        let family = mounted
            .placed_plan
            .reusable_kernel_plan
            .family(&dispatch.reusable_family_id)
            .unwrap();
        let artifact_path = format!("kernels/{}.spv", dispatch.reusable_family_id);
        let workgroup_count_x = if shader_file.starts_with("gqa_attention_") {
            16
        } else if shader_file.starts_with("linear_") {
            let output_descriptor = dispatch
                .descriptors
                .iter()
                .find(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
                .unwrap();
            let output_binding = u32::try_from(output_descriptor.binding).unwrap();
            let bindings = mounted
                .resident_kernel_buffer_bindings_for_bound_dispatch(dispatch)
                .unwrap();
            let output = bindings
                .iter()
                .find(|binding| binding.binding == output_binding)
                .unwrap();
            u32::try_from((output.byte_len / 2).div_ceil(2)).unwrap()
        } else {
            1
        };
        loaded_artifacts.push(VulkanLoadedReusableKernelArtifact {
            artifact: VulkanReusableKernelArtifact::from_family(family, artifact_path.clone())
                .with_workgroup_count_x(workgroup_count_x),
            resolved_path: PathBuf::from(artifact_path),
            words: spirv_words,
        });
    }

    Some(VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts: loaded_artifacts,
        total_word_count,
    })
}

fn reusable_family_with_kernel<'a>(
    reusable_plan: &'a VulkanReusableKernelPlan,
    kernel_id: &str,
) -> &'a VulkanReusableKernelFamily {
    reusable_plan
        .families
        .iter()
        .find(|family| {
            family
                .command_refs
                .iter()
                .any(|command| command.kernel_id == kernel_id)
        })
        .unwrap()
}

fn artifact_path_for_family(family: &VulkanReusableKernelFamily) -> String {
    format!("kernels/{}.spv", family.family_id)
}

