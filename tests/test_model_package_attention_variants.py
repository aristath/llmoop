from model_package_layout_common import *

def test_compiler_renders_biased_recurrent_and_windowed_attention_components(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_bias_bf16_16x24.comp",
        "gelu_tanh_bf16_24.comp",
        "rg_lru_step_bf16_h16_b2x8_k4__sc13.comp",
        "gqa_attention_bf16_q2_kv1_d8_scale0.353553391_w8__sc6.comp",
        "add_bf16_16.comp",
        "multiply_bf16_24.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_bias_bf16_16x24.comp").read_text()
    recurrent = (tmp_path / "rg_lru_step_bf16_h16_b2x8_k4__sc13.comp").read_text()
    attention = (
        tmp_path / "gqa_attention_bf16_q2_kv1_d8_scale0.353553391_w8__sc6.comp"
    ).read_text()
    assert "binding = 3) readonly buffer Bias" in linear
    assert "const uint HEADS = 2u;" in recurrent
    assert "binding = 13) readonly buffer StreamControl" in recurrent
    assert "const uint ATTENTION_WINDOW = 8u;" in attention
    assert "min(runtime_capacity, ATTENTION_WINDOW)" in attention
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )


def test_compiler_renders_windowed_attention_with_learned_sink_logits(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "gqa_attention_bf16_q20_kv4_d64_scale0.015625_w128_sinks__sc7.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    attention = (tmp_path / shader_file).read_text()
    assert "binding = 4) readonly buffer AttentionSinks" in attention
    assert "binding = 7) readonly buffer StreamControl" in attention
    assert "const uint ATTENTION_WINDOW = 128u;" in attention
    assert "float logsumexp = maximum + log(denominator);" in attention
    assert "{{" not in attention


def test_compiler_renders_fused_kv_append_attention_variants(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    plain_file = "append_gqa_attention_bf16_q16_kv8_d64_scale0.125__sc7.comp"
    sinks_file = (
        "append_gqa_attention_bf16_q20_kv4_d64_scale0.015625_w128_sinks__sc8.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {plain_file, sinks_file})

    plain = (tmp_path / plain_file).read_text()
    sinks = (tmp_path / sinks_file).read_text()
    assert "const uint QUERY_HEADS = 16u;" in plain
    assert "const uint QUERY_GROUPS_PER_KV_HEAD = 2u;" in plain
    assert "binding = 5) readonly buffer KvStateRead" in plain
    assert "binding = 6) buffer KvStateWrite" in plain
    assert "binding = 7) readonly buffer StreamControl" in plain
    assert "token_index + 1u == tokens" in plain
    assert "query_head % QUERY_GROUPS_PER_KV_HEAD == 0u" in plain
    assert "const uint TOKEN_BATCHES = TILE_TOKENS / PHYSICAL_TILE_TOKENS;" in plain
    assert "for (uint batch = 0u; batch < TOKEN_BATCHES; batch++)" in plain
    assert "tile_token * subgroups_per_token + subgroup_part" in plain
    assert "binding = 5) readonly buffer AttentionSinks" in sinks
    assert "binding = 6) readonly buffer KvStateRead" in sinks
    assert "binding = 7) buffer KvStateWrite" in sinks
    assert "binding = 8) readonly buffer StreamControl" in sinks
    assert "const uint ATTENTION_WINDOW = 128u;" in sinks
    assert "float logsumexp = maximum + log(denominator);" in sinks
    assert "for (uint batch = 0u; batch < TOKEN_BATCHES; batch++)" in sinks
    assert "{{" not in plain
    assert "{{" not in sinks


def test_compiler_renders_subgroup_padded_attention_and_unequal_qkv_split(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    attention_file = "gqa_attention_bf16_q32_kv32_d96_scale0.102062073_w2047__sc6.comp"
    split_file = "split_bf16_3x16_8_8.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {attention_file, split_file})

    attention = (tmp_path / attention_file).read_text()
    split = (tmp_path / split_file).read_text()
    assert "layout(local_size_x = 1024" in attention
    assert "const uint HEAD_WIDTH = 96u;" in attention
    assert "const uint TILE_TOKENS = 56u;" in attention
    assert "const uint TOKEN_BATCHES = TILE_TOKENS / PHYSICAL_TILE_TOKENS;" in attention
    assert "value_contributions[56 * 96]" in attention
    assert "const uint PART_A_WORDS = 16u / 2u;" in split
    assert "const uint PART_B_WORDS = 8u / 2u;" in split
    assert "{{" not in attention
    assert "{{" not in split


def test_compiler_renders_hybrid_recurrent_and_gated_attention_components(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "causal_conv1d_silu_bf16_c6144_k4.comp",
        "gated_delta_step_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06.comp",
        "gated_delta_step_k16x128_v16x128_abf16_dtbf16_nbf16_eps1e-06.comp",
        "gated_delta_scan_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06.comp",
        "gated_delta_step_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06_qfp8b128.comp",
        "gated_delta_scan_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06_qfp8b128.comp",
        "split_bf16_2x8x256_head_interleaved.comp",
        "sigmoid_multiply_bf16.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    convolution = (tmp_path / "causal_conv1d_silu_bf16_c6144_k4.comp").read_text()
    recurrence = (
        tmp_path / "gated_delta_step_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06.comp"
    ).read_text()
    bf16_recurrence = (
        tmp_path / "gated_delta_step_k16x128_v16x128_abf16_dtbf16_nbf16_eps1e-06.comp"
    ).read_text()
    temporal_recurrence = (
        tmp_path / "gated_delta_scan_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06.comp"
    ).read_text()
    quantized_recurrence = (
        tmp_path
        / "gated_delta_step_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06_qfp8b128.comp"
    ).read_text()
    quantized_temporal_recurrence = (
        tmp_path
        / "gated_delta_scan_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06_qfp8b128.comp"
    ).read_text()
    split = (tmp_path / "split_bf16_2x8x256_head_interleaved.comp").read_text()
    assert "const uint CHANNELS = 6144u;" in convolution
    assert "const uint KEY_HEADS = 16u;" in recurrence
    assert "const uint VALUE_HEAD_WIDTH = 128u;" in recurrence
    assert "uintBitsToFloat(a_log.words[index])" in recurrence
    assert "unpack_bf16(dt_bias.words[index >> 1u], index)" in recurrence
    assert "uintBitsToFloat(norm_weight.words[index])" in recurrence
    assert "unpack_bf16(a_log.words[index >> 1u], index)" in bf16_recurrence
    assert "unpack_bf16(norm_weight.words[index >> 1u], index)" in bf16_recurrence
    assert "shared float raw_query[KEY_HEAD_WIDTH];" in temporal_recurrence
    assert "shared float raw_key[KEY_HEAD_WIDTH];" in temporal_recurrence
    assert "q_sum = subgroupAdd(q_sum);" in temporal_recurrence
    assert "k_sum = subgroupAdd(k_sum);" in temporal_recurrence
    assert "reduction[gl_SubgroupID] = q_sum;" in temporal_recurrence
    assert "head_output[gl_SubgroupID] = k_sum;" in temporal_recurrence
    assert "head_beta = 1.0 /" in temporal_recurrence
    assert (
        "float previous = recurrent_state[key_dim] * head_decay;" in temporal_recurrence
    )
    assert "recurrent_state[key_dim] = previous;" in temporal_recurrence
    assert "float next = recurrent_state[key_dim] + key * delta;" in temporal_recurrence
    for source in (quantized_recurrence, quantized_temporal_recurrence):
        assert "binding = 5) buffer QuantizedOutput" in source
        assert "binding = 6) buffer OutputScale" in source
        assert "binding = 10) readonly buffer StateRead" in source
        assert "binding = 11) buffer StateWrite" in source
        assert "subgroupMax(abs(head_output[value_dim]))" in source
        assert "pack_fp8(" in source
        assert "{{" not in source
    assert (
        "recurrent_state[key_dim] * head_decay + key * delta" not in temporal_recurrence
    )
    assert "const uint BLOCKS = 8u;" in split
    assert "const uint BLOCK_PART_WIDTH = 256u;" in split
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )
