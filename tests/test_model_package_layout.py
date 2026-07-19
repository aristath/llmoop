from __future__ import annotations

import json
import struct
from pathlib import Path

from llmoop.model_package import (
    ROW_MAJOR_LAYOUT,
    VULKAN_BF16_ROW_PAIR_LAYOUT,
    compiled_tensor_layout,
    copy_shader_templates,
    write_compiled_tensor,
)


def test_bf16_matrix_layout_is_compiled_as_interleaved_row_pairs() -> None:
    assert (
        compiled_tensor_layout({"dtype": "BF16", "shape": [4, 4]})
        == VULKAN_BF16_ROW_PAIR_LAYOUT
    )


def test_fp8_block_scales_remain_row_major() -> None:
    assert (
        compiled_tensor_layout(
            {"dtype": "BF16", "shape": [4, 4]},
            tensor_name="projection.weight_scale_inv",
        )
        == ROW_MAJOR_LAYOUT
    )


def test_write_compiled_tensor_interleaves_bf16_row_pairs(tmp_path: Path) -> None:
    tensor_name = "matrix.weight"
    values = tuple(range(16))
    source_header = {
        tensor_name: {
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, len(values) * 2],
        }
    }
    source_header_payload = json.dumps(source_header).encode("utf-8")
    source = tmp_path / "source.safetensors"
    source.write_bytes(
        struct.pack("<Q", len(source_header_payload))
        + source_header_payload
        + struct.pack("<16H", *values)
    )
    destination = tmp_path / "compiled.safetensors"

    write_compiled_tensor(
        tensor_name=tensor_name,
        info={
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, len(values) * 2],
            "byte_count": len(values) * 2,
        },
        source=source,
        destination=destination,
        layout=VULKAN_BF16_ROW_PAIR_LAYOUT,
    )

    compiled = destination.read_bytes()
    header_bytes = struct.unpack("<Q", compiled[:8])[0]
    payload = compiled[8 + header_bytes :]
    assert struct.unpack("<16H", payload) == (
        0,
        1,
        4,
        5,
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        13,
        10,
        11,
        14,
        15,
    )


def test_compiler_renders_paired_matrix_and_transducer_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_paired_bf16_768x2048.comp",
        "linear_residual_paired_bf16_2048x768.comp",
        "embedding_lookup_paired_bf16_32000x768_scale12.comp",
        "tied_output_projection_paired_bf16_32000x768_scale0.166666667_to_f32.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "{{" not in shader
        assert "uvec2 words[]" in shader
    assert (
        "const uint INPUT_SIZE = 768u;"
        in (tmp_path / "linear_paired_bf16_768x2048.comp").read_text()
    )
    assert (
        "const uint VOCAB_SIZE = 32000u;"
        in (
            tmp_path / "embedding_lookup_paired_bf16_32000x768_scale12.comp"
        ).read_text()
    )
    assert (
        "const float EMBEDDING_SCALE = 12;"
        in (
            tmp_path / "embedding_lookup_paired_bf16_32000x768_scale12.comp"
        ).read_text()
    )
    assert (
        "const float OUTPUT_SCALE = 0.166666667;"
        in (
            tmp_path
            / "tied_output_projection_paired_bf16_32000x768_scale0.166666667_to_f32.comp"
        ).read_text()
    )


def test_compiler_renders_native_block_scaled_fp8_linear_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_bias_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_residual_fp8_e4m3_b128x128_17408x5120.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "const uint BLOCK_ROWS = 128u;" in shader
        assert "const uint BLOCK_COLUMNS = 128u;" in shader
        assert "fp8_e4m3_to_f32" in shader
        assert "WeightScaleInv" in shader
        assert "{{" not in shader
    assert (
        "binding = 4) readonly buffer Bias"
        in (tmp_path / "linear_bias_fp8_e4m3_b128x128_5120x17408.comp").read_text()
    )


def test_compiler_renders_native_block_scaled_fp8_sparse_experts(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "moe_topk_bf16_e256_k8.comp",
        "sparse_moe_experts_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "moe_reduce_bf16_h2048_e256.comp",
        "sigmoid_scalar_multiply_bf16_2048.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    expert_shader = (
        tmp_path
        / "sparse_moe_experts_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    ).read_text()
    assert "const uint NUM_EXPERTS = 256u;" in expert_shader
    assert "const uint EXPERTS_PER_TOKEN = 8u;" in expert_shader
    assert "ExpertInputScaleInv" in expert_shader
    assert "ExpertOutputScaleInv" in expert_shader
    assert "{{" not in expert_shader
    assert (
        "const uint HIDDEN_SIZE = 2048u;"
        in (tmp_path / "sigmoid_scalar_multiply_bf16_2048.comp").read_text()
    )


def test_compiler_renders_attention_pedals_from_discovered_dimensions(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rms_norm_bf16_h960_eps1e-05_offset0.comp",
        "rotary_bf16_15x64_r64_theta100000_half__sc2.comp",
        "append_kv_state_bf16_5x64__sc9.comp",
        "gqa_attention_bf16_q15_kv5_d64_scale0.125__sc6.comp",
        "greedy_sampler_f32_49152.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    norm = (tmp_path / "rms_norm_bf16_h960_eps1e-05_offset0.comp").read_text()
    rope = (tmp_path / "rotary_bf16_15x64_r64_theta100000_half__sc2.comp").read_text()
    append = (tmp_path / "append_kv_state_bf16_5x64__sc9.comp").read_text()
    attention = (
        tmp_path / "gqa_attention_bf16_q15_kv5_d64_scale0.125__sc6.comp"
    ).read_text()
    sampler = (tmp_path / "greedy_sampler_f32_49152.comp").read_text()

    assert "const uint HIDDEN_SIZE = 960u;" in norm
    assert "const float NORM_EPS = 1e-05;" in norm
    assert "const float WEIGHT_OFFSET = 0;" in norm
    assert "const uint HEAD_COUNT = 15u;" in rope
    assert "const uint ROTARY_WIDTH = 64u;" in rope
    assert "const float ROPE_THETA = 100000;" in rope
    assert "const bool ROPE_INTERLEAVED = false;" in rope
    assert "binding = 2) readonly buffer StreamControl" in rope
    assert "const uint KV_HEADS = 5u;" in append
    assert "binding = 9) readonly buffer StreamControl" in append
    assert "const uint QUERY_HEADS = 15u;" in attention
    assert "const uint QUERY_GROUPS_PER_KV_HEAD = 3u;" in attention
    assert "const float ATTENTION_SCALE = 0.125;" in attention
    assert "binding = 6) readonly buffer StreamControl" in attention
    assert "const uint VOCAB_SIZE = 49152u;" in sampler
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )


def test_compiler_renders_biased_recurrent_and_windowed_attention_pedals(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_bias_paired_bf16_16x24.comp",
        "gelu_tanh_bf16_24.comp",
        "rg_lru_step_bf16_h16_b2x8_k4__sc13.comp",
        "gqa_attention_bf16_q2_kv1_d8_scale0.353553391_w8__sc6.comp",
        "add_bf16_16.comp",
        "multiply_bf16_24.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_bias_paired_bf16_16x24.comp").read_text()
    recurrent = (tmp_path / "rg_lru_step_bf16_h16_b2x8_k4__sc13.comp").read_text()
    attention = (
        tmp_path / "gqa_attention_bf16_q2_kv1_d8_scale0.353553391_w8__sc6.comp"
    ).read_text()
    assert "binding = 3) readonly buffer Bias" in linear
    assert "const uint HEADS = 2u;" in recurrent
    assert "binding = 13) readonly buffer StreamControl" in recurrent
    assert "const uint ATTENTION_WINDOW = 8u;" in attention
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
    assert "const uint TILE_TOKENS = 8u;" in attention
    assert "local_index < TILE_TOKENS * HEAD_WIDTH" in attention
    assert "const uint PART_A_WORDS = 16u / 2u;" in split
    assert "const uint PART_B_WORDS = 8u / 2u;" in split
    assert "{{" not in attention
    assert "{{" not in split


def test_compiler_renders_hybrid_recurrent_and_gated_attention_pedals(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "causal_conv1d_silu_bf16_c6144_k4.comp",
        "gated_delta_step_k16x128_v16x128_eps1e-06.comp",
        "split_bf16_2x8x256_head_interleaved.comp",
        "sigmoid_multiply_bf16.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    convolution = (tmp_path / "causal_conv1d_silu_bf16_c6144_k4.comp").read_text()
    recurrence = (
        tmp_path / "gated_delta_step_k16x128_v16x128_eps1e-06.comp"
    ).read_text()
    split = (tmp_path / "split_bf16_2x8x256_head_interleaved.comp").read_text()
    assert "const uint CHANNELS = 6144u;" in convolution
    assert "const uint KEY_HEADS = 16u;" in recurrence
    assert "const uint VALUE_HEAD_WIDTH = 128u;" in recurrence
    assert "const uint BLOCKS = 8u;" in split
    assert "const uint BLOCK_PART_WIDTH = 256u;" in split
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )


def test_compiler_renders_sparse_moe_and_scaled_residual_pedals(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "scaled_add_bf16_1024_scale0.22.comp",
        "moe_topk_bf16_e32_k8.comp",
        "sparse_moe_experts_bf16_h1024_i512_e32_k8.comp",
        "moe_reduce_bf16_h1024_e32.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    scaled_add = (tmp_path / "scaled_add_bf16_1024_scale0.22.comp").read_text()
    router = (tmp_path / "moe_topk_bf16_e32_k8.comp").read_text()
    experts = (tmp_path / "sparse_moe_experts_bf16_h1024_i512_e32_k8.comp").read_text()
    reduce = (tmp_path / "moe_reduce_bf16_h1024_e32.comp").read_text()
    assert "const float RESIDUAL_SCALE = 0.22;" in scaled_add
    assert "const uint NUM_EXPERTS = 32u;" in router
    assert "const uint EXPERTS_PER_TOKEN = 8u;" in router
    assert "const uint INTERMEDIATE_SIZE = 512u;" in experts
    assert "const uint HIDDEN_SIZE = 1024u;" in reduce
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )
