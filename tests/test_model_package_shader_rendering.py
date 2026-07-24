from model_package_layout_common import *

def test_compiler_renders_parallel_linear_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    pair = "parallel_linear_2way_bf16_1024x2560_2560.comp"
    triple = "parallel_linear_3way_bf16_1024x1024_512_512.comp"
    fp8_pair = (
        "parallel_linear_batch16_2way_fp8_e4m3_b128x128_5120x5120_1024.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {pair, triple, fp8_pair})

    pair_source = (tmp_path / pair).read_text()
    triple_source = (tmp_path / triple).read_text()
    fp8_pair_source = (tmp_path / fp8_pair).read_text()
    assert "binding = 3) readonly buffer WeightA" in pair_source
    assert "binding = 4) readonly buffer WeightB" in pair_source
    assert "const uint OUTPUT_A_WORDS = 2560u / 2u;" in pair_source
    assert "binding = 6) readonly buffer WeightC" in triple_source
    assert "const uint OUTPUT_C_WORDS = 512u / 2u;" in triple_source
    assert "binding = 3) readonly buffer WeightA" in fp8_pair_source
    assert "binding = 4) readonly buffer WeightScaleInvA" in fp8_pair_source
    assert "binding = 5) readonly buffer WeightB" in fp8_pair_source
    assert "binding = 6) readonly buffer WeightScaleInvB" in fp8_pair_source
    assert "shared fe4m3vec4 quantized_input[INPUT_FP8_WORDS];" in fp8_pair_source
    assert "const uint OUTPUT_A_SIZE = 5120u;" in fp8_pair_source
    assert "const uint OUTPUT_B_SIZE = 1024u;" in fp8_pair_source
    assert "for (uint branch = 0u; branch < BRANCH_COUNT; branch++)" in fp8_pair_source
    assert "PAIRED_WEIGHT_LAYOUT" not in pair_source
    assert "PAIRED_WEIGHT_LAYOUT" not in triple_source
    assert "PAIRED_WEIGHT_LAYOUT" not in fp8_pair_source
    assert "{{" not in pair_source
    assert "{{" not in triple_source
    assert "{{" not in fp8_pair_source


def test_compiler_renders_fp8_output_projection_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "tied_output_projection_fp8_e4m3_b16x128_248320x5120_scale1_to_f32.comp",
        "tied_output_projection_batch1_fp8_e4m3_b16x128_248320x5120_scale1_to_f32.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    decode = (
        tmp_path
        / "tied_output_projection_fp8_e4m3_b16x128_248320x5120_scale1_to_f32.comp"
    ).read_text()
    batch = (
        tmp_path
        / "tied_output_projection_batch1_fp8_e4m3_b16x128_248320x5120_scale1_to_f32.comp"
    ).read_text()
    for source in (decode, batch):
        assert "binding = 1) readonly buffer ProjectionWeight" in source
        assert "binding = 3) readonly buffer ProjectionWeightScaleInv" in source
        assert "const uint BLOCK_ROWS = 16u;" in source
        assert "const uint BLOCK_COLUMNS = 128u;" in source
        assert "const uint OUTPUT_TILE_ROWS = 32u;" in source
        assert "const uint ROW_CLUSTER_LANES = 32u;" in source
        assert "uint local_row = gl_SubgroupID * rows_per_subgroup + row_cluster;" in source
        assert "sum = subgroupClusteredAdd(sum, ROW_CLUSTER_LANES);" in source
        assert "fp8_dot4_acc32" in source
        assert "{{" not in source
    assert "layout(push_constant) uniform BatchControl" not in decode
    assert "layout(push_constant) uniform BatchControl" in batch
    assert "batch_index * VOCAB_SIZE + row" in batch


def test_compiler_renders_reusable_fp8_activation_kernel_family(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "quantize_fp8_e4m3_b128_h5120.comp",
        "quantize_batch16_fp8_e4m3_b128_h5120.comp",
        "linear_residual_prequant_fp8_e4m3_b128x128_5120x5120.comp",
        "linear_residual_prequant_batch16_fp8_e4m3_b128x128_5120x5120.comp",
        "parallel_linear_batch16_3way_prequant_fp8_e4m3_"
        "b128x128_5120x4096_1024_1024.comp",
        "parallel_linear_silu_multiply_prequant_batch16_fp8_e4m3_"
        "b128x128_5120x17408.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    quantize = (tmp_path / "quantize_fp8_e4m3_b128_h5120.comp").read_text()
    residual = (
        tmp_path
        / "linear_residual_prequant_fp8_e4m3_b128x128_5120x5120.comp"
    ).read_text()
    parallel = (
        tmp_path
        / "parallel_linear_batch16_3way_prequant_fp8_e4m3_"
        "b128x128_5120x4096_1024_1024.comp"
    ).read_text()
    fused = (
        tmp_path
        / "parallel_linear_silu_multiply_prequant_batch16_fp8_e4m3_"
        "b128x128_5120x17408.comp"
    ).read_text()
    assert "const uint ELEMENT_COUNT = 5120u;" in quantize
    assert "binding = 2) readonly buffer ResidualFrame" in residual
    assert "const uint OUTPUT_TILE_ROWS = 16u;" in residual
    assert "binding = 2) buffer OutputA" in parallel
    assert "binding = 5) readonly buffer WeightA" in parallel
    assert "const uint OUTPUT_C_SIZE = 1024u;" in parallel
    assert "layout(local_size_x = 1024" in residual
    assert "layout(local_size_x = 1024" in parallel
    assert "shared fe4m3vec4 quantized_input" not in residual
    assert "shared fe4m3vec4 quantized_input" not in parallel
    assert "read_gate_scale" in fused
    assert "layout(push_constant) uniform BatchControl" in fused
    for shader_file in shader_files:
        assert "{{" not in (tmp_path / shader_file).read_text()


def test_compiler_renders_fused_parallel_ffn_projection_shader(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "parallel_linear_silu_multiply_bf16_1024x2560.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "binding = 2) readonly buffer GateWeight" in source
    assert "binding = 3) readonly buffer UpWeight" in source
    assert "const uint INPUT_SIZE = 1024u;" in source
    assert "const uint OUTPUT_SIZE = 2560u;" in source
    assert "PAIRED_WEIGHT_LAYOUT" not in source
    assert "rounded_silu" in source
    assert "{{" not in source


def test_compiler_selects_and_renders_fused_block_scaled_fp8_ffn_shader(
    tmp_path: Path,
) -> None:
    params = [
        "gate_weight",
        "gate_weight_scale_inv",
        "up_weight",
        "up_weight_scale_inv",
    ]
    node = {
        "id": "fused_ffn",
        "op": "parallel_linear_silu_multiply",
        "inputs": ["hidden"],
        "outputs": ["ffn_hidden"],
        "params": params,
        "attrs": {
            "branch_count": 2,
            "intermediate_rounding": "BF16",
            "element_count": 17408,
        },
    }
    circuit = {
        "parameters": {
            "refs": {parameter_id: {"tensor": parameter_id} for parameter_id in params}
        }
    }
    tensor_index = {
        "tensors": {
            "gate_weight": {
                "dtype": "F8_E4M3",
                "shape": [17408, 5120],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "gate_weight_scale_inv": {
                "dtype": "BF16",
                "shape": [136, 40],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "up_weight": {
                "dtype": "F8_E4M3",
                "shape": [17408, 5120],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "up_weight_scale_inv": {
                "dtype": "BF16",
                "shape": [136, 40],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }
    dimensions = {"hidden_size": 5120, "intermediate_size": 17408}

    shader_file = shader_file_for_node(circuit, node, tensor_index, dimensions)
    assert (
        shader_file == "parallel_linear_silu_multiply_fp8_e4m3_b128x128_5120x17408.comp"
    )

    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})
    source = (tmp_path / shader_file).read_text()
    assert "binding = 3) readonly buffer GateScaleInv" in source
    assert "binding = 5) readonly buffer UpScaleInv" in source
    assert "const uint BLOCK_ROWS = 128u;" in source
    assert "const uint OUTPUT_TILE_ROWS = 16u;" in source
    assert "shared fe4m3vec4 quantized_input" in source
    assert "fp8_dot4_acc32" in source
    assert "uint word = gl_SubgroupInvocationID;" in source
    assert "rounded_silu" in source
    assert "{{" not in source


def test_compiler_renders_parallel_head_norm_rope_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "parallel_head_norm_rope_2way_bf16_h16_8_d64_r64_"
        "eps1e-05_offset0_theta1000000_half__sc6.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "const uint BRANCH_A_HEADS = 16u;" in source
    assert "const uint BRANCH_B_HEADS = 8u;" in source
    assert "const uint HEAD_WIDTH = 64u;" in source
    assert "const uint ROTARY_WIDTH = 64u;" in source
    assert "const bool ROPE_INTERLEAVED = false;" in source
    assert "layout(set = 0, binding = 6) readonly buffer StreamControl" in source
    assert "shared uint normalized_words" in source
    assert "float cosine = round_bf16(cos(angle) * ROPE_ATTENTION_FACTOR);" in source
    assert "return round_bf16(direct + rotated);" in source
    assert "{{" not in source


def test_compiler_renders_yarn_rope_profile_into_fused_shader(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "parallel_head_norm_rope_2way_bf16_h48_8_d128_r64_"
        "eps1e-06_offset0_theta500000_yarn_f32_lo9_hi18_a1.34657359_half__sc6.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "const bool ROPE_YARN = true;" in source
    assert "const float ROPE_FACTOR = 32;" in source
    assert "const float ROPE_CORRECTION_LOW = 9;" in source
    assert "const float ROPE_CORRECTION_HIGH = 18;" in source
    assert "const float ROPE_ATTENTION_FACTOR = 1.34657359;" in source
    assert "return mix(inverse_frequency, inverse_frequency / ROPE_FACTOR, ramp);" in source
    assert "{{" not in source


def test_compiler_renders_fused_recurrent_depthwise_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "multiply_rolling_depthwise_bf16_3x1024.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "binding = 0) readonly buffer GateInput" in source
    assert "binding = 4) readonly buffer ConvKernel" in source
    assert "binding = 5) readonly buffer StateRead" in source
    assert "binding = 6) buffer StateWrite" in source
    assert "const uint FRAME_COUNT = 3u;" in source
    assert "const uint HIDDEN_SIZE = 1024u;" in source
    assert "uint temporal_words[FRAME_COUNT];" in source
    assert "multiply_pair(" in source
    assert "{{" not in source


def test_compiler_renders_unfused_recurrent_and_activation_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rolling_state_update_bf16_5x768.comp",
        "depthwise_conv1d_bf16_5x768.comp",
        "silu_bf16_3072.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    rolling = (tmp_path / "rolling_state_update_bf16_5x768.comp").read_text()
    depthwise = (tmp_path / "depthwise_conv1d_bf16_5x768.comp").read_text()
    silu = (tmp_path / "silu_bf16_3072.comp").read_text()
    assert "const uint FRAME_COUNT = 5u;" in rolling
    assert "const uint FRAME_WORD_COUNT = (768u + 1u) / 2u;" in rolling
    assert "const uint HIDDEN_SIZE = 768u;" in depthwise
    assert "const uint KERNEL_TAPS = 5u;" in depthwise
    assert "const uint WORD_COUNT = (3072u + 1u) / 2u;" in silu
    assert all("{{" not in source for source in (rolling, depthwise, silu))


def test_compiler_renders_output_gated_recurrent_depthwise_shader(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "multiply_rolling_depthwise_gate_bf16_3x1024.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "binding = 3) readonly buffer OutputGate" in source
    assert "binding = 4) buffer ConvOutput" in source
    assert "binding = 5) readonly buffer ConvKernel" in source
    assert "binding = 6) readonly buffer StateRead" in source
    assert "binding = 7) buffer StateWrite" in source
    assert "bf16_to_f32(conv_pair) * bf16_to_f32(gate_pair)" in source
    assert "finalize_output(word_index, conv_pair)" in source
    assert "{{" not in source


def test_compiler_renders_projected_recurrent_depthwise_shader(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "linear_split_recurrent_depthwise_gate_bf16_1024x1024_k3_ig0_2_og1.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "binding = 3) readonly buffer ProjectionWeight" in source
    assert "binding = 4) readonly buffer ConvKernel" in source
    assert "const uint INPUT_SIZE = 1024u;" in source
    assert "const uint HIDDEN_SIZE = 1024u;" in source
    assert "const uint FRAME_COUNT = 3u;" in source
    assert "const uint INPUT_GATE_A_INDEX = 0u;" in source
    assert "const uint INPUT_GATE_B_INDEX = 2u;" in source
    assert "const uint OUTPUT_GATE_INDEX = 1u;" in source
    assert "PAIRED_WEIGHT_LAYOUT" not in source
    assert "{{" not in source
