from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from llmoop.compilation import ModelCompileError
from llmoop.model_package import (
    CAUSAL_SCAN_LANE_TILE_WIDTH,
    SCALAR_BATCH_LANE_TILE_WIDTH,
    ROW_MAJOR_LAYOUT,
    VULKAN_BF16_ROW_PAIR_LAYOUT,
    attention_workgroup_shape,
    causal_scan_batch_shader_file,
    causal_scan_workgroup_count_x,
    compiled_tensor_layout,
    cooperative_bfloat16_batch_shader_file,
    cooperative_bfloat16_workgroup_count_x,
    copy_shader_templates,
    pedal_kernel_spec,
    required_vulkan_device_extensions,
    shader_file_for_node,
    weight_shared_batch_shader_file,
    write_compiled_tensor,
)


def test_compiler_selects_only_compatible_weight_shared_batch_kernels() -> None:
    assert SCALAR_BATCH_LANE_TILE_WIDTH == 16
    assert weight_shared_batch_shader_file(
        "rms_norm_bf16_h5120_eps1e-06_offset1.comp"
    ) == "rms_norm_batch16_bf16_h5120_eps1e-06_offset1.comp"
    assert weight_shared_batch_shader_file(
        "linear_fp8_e4m3_b128x128_5120x17408.comp"
    ) == "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp"
    assert weight_shared_batch_shader_file(
        "linear_residual_fp8_e4m3_b128x128_17408x5120.comp"
    ) == "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp"
    assert weight_shared_batch_shader_file(
        "parallel_linear_2way_paired_bf16_1024x2560_2560.comp"
    ) == "parallel_linear_batch16_2way_paired_bf16_1024x2560_2560.comp"
    assert weight_shared_batch_shader_file(
        "parallel_linear_silu_multiply_fp8_e4m3_b128x128_5120x17408.comp"
    ) == (
        "parallel_linear_silu_multiply_batch16_fp8_e4m3_"
        "b128x128_5120x17408.comp"
    )
    assert weight_shared_batch_shader_file(
        "linear_paired_bf16_1024x1024.comp"
    ) == "linear_batch16_paired_bf16_1024x1024.comp"
    assert weight_shared_batch_shader_file(
        "linear_residual_bf16_1024x1024.comp"
    ) == "linear_residual_batch16_row_major_bf16_1024x1024.comp"
    assert weight_shared_batch_shader_file(
        "parallel_linear_silu_multiply_paired_bf16_1024x4096.comp"
    ) == "parallel_linear_silu_multiply_batch16_paired_bf16_1024x4096.comp"
    assert weight_shared_batch_shader_file(
        "split_bf16_2x16x256_head_interleaved.comp"
    ) == "split_batch16_bf16_2x16x256_head_interleaved.comp"
    assert (
        weight_shared_batch_shader_file("sigmoid_multiply_bf16.comp")
        == "sigmoid_multiply_batch16_bf16.comp"
    )
    assert weight_shared_batch_shader_file(
        "linear_fp8_e4m3_b127x128_5120x17408.comp"
    ) is None
    assert weight_shared_batch_shader_file("linear_paired_bf16_1023x1024.comp") is None


def test_compiler_selects_stateful_causal_scan_kernels() -> None:
    assert CAUSAL_SCAN_LANE_TILE_WIDTH == 64
    assert causal_scan_batch_shader_file(
        "causal_conv1d_silu_bf16_c8192_k4.comp"
    ) == "causal_conv1d_silu_temporal_bf16_c8192_k4.comp"
    assert causal_scan_batch_shader_file(
        "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
    ) == "gated_delta_scan_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
    assert causal_scan_batch_shader_file(
        "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
        "offset1_theta10000000_half__sc6.comp"
    ) == (
        "parallel_head_norm_rope_2way_temporal_bf16_h16_4_d256_r64_"
        "eps1e-06_offset1_theta10000000_half.comp"
    )
    assert causal_scan_batch_shader_file("linear_paired_bf16_4096x4096.comp") is None
    assert causal_scan_workgroup_count_x(
        "causal_conv1d_silu_bf16_c8192_k4.comp"
    ) == 64
    assert causal_scan_workgroup_count_x(
        "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
    ) == 32
    assert causal_scan_workgroup_count_x(
        "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
        "offset1_theta10000000_half__sc6.comp"
    ) == 20


def test_compiler_selects_cooperative_bfloat16_projection_kernels() -> None:
    assert cooperative_bfloat16_batch_shader_file(
        "linear_paired_bf16_1024x4096.comp"
    ) == "linear_batch64_cooperative_paired_bf16_1024x4096.comp"
    assert cooperative_bfloat16_batch_shader_file(
        "linear_residual_bf16_4096x1024.comp"
    ) == "linear_residual_batch64_cooperative_row_major_bf16_4096x1024.comp"
    assert cooperative_bfloat16_batch_shader_file(
        "parallel_linear_3way_paired_bf16_1024x1024_256_256.comp"
    ) == (
        "parallel_linear_batch64_cooperative_3way_paired_bf16_"
        "1024x1024_256_256.comp"
    )
    assert cooperative_bfloat16_batch_shader_file(
        "parallel_linear_silu_multiply_paired_bf16_1024x4096.comp"
    ) == (
        "parallel_linear_silu_multiply_batch64_cooperative_paired_bf16_"
        "1024x4096.comp"
    )
    assert cooperative_bfloat16_workgroup_count_x(
        "parallel_linear_3way_paired_bf16_1024x1024_256_256.comp"
    ) == 24
    assert cooperative_bfloat16_batch_shader_file(
        "linear_fp8_e4m3_b128x128_1024x4096.comp"
    ) is None


def test_projection_pedal_compiles_ordered_target_native_and_scalar_implementations() -> None:
    spec = pedal_kernel_spec(
        execution_index=0,
        node={"id": "project", "op": "linear"},
        shader_file="linear_paired_bf16_1024x4096.comp",
        local_size_x=64,
        workgroup_count_x=2048,
    )

    assert spec["batch_mode"] == "weight_shared"
    assert "batch_shader_path" not in spec
    assert "batch_lane_tile_width" not in spec
    cooperative, scalar = spec["batch_implementations"]
    assert cooperative == {
        "lane_tile_width": 64,
        "device_requirements": {
            "vulkan_device_extensions": [
                "VK_KHR_cooperative_matrix",
                "VK_KHR_shader_bfloat16",
            ],
            "cooperative_bfloat16_shape": [16, 16, 16],
            "subgroup_size": 64,
        },
        "stages": [
            {
                "shader_path": (
                    "shaders/linear_batch64_cooperative_paired_bf16_1024x4096.comp"
                ),
                "local_size_x": 256,
                "workgroup_count_x": 64,
            }
        ],
    }
    assert scalar == {
        "lane_tile_width": 16,
        "device_requirements": {"vulkan_device_extensions": []},
        "stages": [
            {
                "shader_path": "shaders/linear_batch16_paired_bf16_1024x4096.comp",
                "local_size_x": 64,
                "workgroup_count_x": 2048,
            }
        ],
    }


def test_compiler_renders_weight_shared_pedal_batch_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rms_norm_batch16_bf16_h5120_eps1e-06_offset1.comp",
        "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp",
        "parallel_linear_batch16_2way_paired_bf16_1024x2560_2560.comp",
        "parallel_linear_silu_multiply_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_batch16_paired_bf16_1024x4096.comp",
        "linear_residual_batch16_row_major_bf16_4096x1024.comp",
        "parallel_linear_silu_multiply_batch16_paired_bf16_1024x4096.comp",
        "split_batch16_bf16_2x16x256_head_interleaved.comp",
        "sigmoid_multiply_batch16_bf16.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "const uint BATCH_TILE_WIDTH = 16u;" in source
        assert "layout(push_constant) uniform BatchControl" in source
        assert "gl_WorkGroupID.y * BATCH_TILE_WIDTH" in source
        if "fp8_e4m3" in shader_file:
            assert "#extension GL_EXT_float_e4m3 : require" in source
            assert "uintBitsToFloate4m3EXT" in source
        assert "{{" not in source
    assert required_vulkan_device_extensions(tmp_path, shader_files) == [
        "VK_EXT_shader_float8"
    ]


def test_compiler_renders_position_aware_temporal_head_norm_rope(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "parallel_head_norm_rope_2way_temporal_bf16_h16_4_d256_r64_"
        "eps1e-06_offset1_theta10000000_half.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "uint start_stream_tick_low;" in source
    assert "position < batch_control.batch_width" in source
    assert "start_stream_tick_low + position" in source
    assert "StreamControl" not in source
    assert "{{" not in source


def test_compiler_renders_cooperative_bfloat16_batch_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_batch64_cooperative_paired_bf16_1024x4096.comp",
        "linear_residual_batch64_cooperative_row_major_bf16_4096x1024.comp",
        "parallel_linear_batch64_cooperative_3way_paired_bf16_"
        "1024x1024_256_256.comp",
        "parallel_linear_silu_multiply_batch64_cooperative_paired_bf16_"
        "1024x4096.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "coopMatMulAdd" in source
        assert "coopmat<bfloat16_t" in source
        assert "#extension GL_EXT_bfloat16 : require" in source
        assert "#extension GL_KHR_cooperative_matrix : require" in source
        assert "layout(local_size_x = 256" in source
        assert "const uint OUTPUT_TILE = 64u;" in source
        assert "const uint BATCH_TILE = 64u;" in source
        assert "{{" not in source
    assert required_vulkan_device_extensions(tmp_path, shader_files) == [
        "VK_KHR_cooperative_matrix",
        "VK_KHR_shader_bfloat16",
    ]


@pytest.mark.parametrize(
    "head_width",
    [32, 64, 80, 96, 128, 192, 256, 320, 384, 512, 768, 1024],
)
def test_attention_tile_stays_within_portable_shared_memory_budget(
    head_width: int,
) -> None:
    local_size, tile_tokens = attention_workgroup_shape(head_width)
    padded_head_width = ((head_width + 63) // 64) * 64
    physical_tile_tokens = 1024 // padded_head_width
    shared_floats = (
        2 * head_width
        + tile_tokens * ((head_width + 31) // 32)
        + 3 * tile_tokens
        + tile_tokens * head_width
        + 4
    )

    assert local_size == padded_head_width * physical_tile_tokens
    assert local_size <= 1024
    assert tile_tokens > physical_tile_tokens
    assert shared_floats * 4 <= 32 * 1024


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


def test_quantization_scale_grids_remain_row_major() -> None:
    assert (
        compiled_tensor_layout(
            {"dtype": "BF16", "shape": [768, 16]},
            tensor_name="projection.weight_scale",
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
        "embedding_lookup_batch_paired_bf16_32000x768_scale12.comp",
        "tied_output_projection_paired_bf16_32000x768_scale0.166666667_to_f32.comp",
        "tied_output_projection_batch4_paired_bf16_32000x768_scale0.166666667_to_f32.comp",
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
        "gl_WorkGroupID.y"
        in (
            tmp_path / "embedding_lookup_batch_paired_bf16_32000x768_scale12.comp"
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
    batched_projection = (
        tmp_path
        / "tied_output_projection_batch4_paired_bf16_32000x768_scale0.166666667_to_f32.comp"
    ).read_text()
    assert "const uint BATCH_TILE_WIDTH = 4u;" in batched_projection
    assert "layout(push_constant) uniform BatchControl" in batched_projection
    assert "gl_WorkGroupID.y * BATCH_TILE_WIDTH" in batched_projection


def test_compiler_renders_direct_three_way_linear_split_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    paired = "linear_split_3way_paired_bf16_1024x1024_1024_1024.comp"
    row_major = "linear_split_3way_row_major_bf16_768x2048_1024_1024.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {paired, row_major})

    paired_source = (tmp_path / paired).read_text()
    row_major_source = (tmp_path / row_major).read_text()
    assert "const uint INPUT_SIZE = 1024u;" in paired_source
    assert "const uint PART_A_WIDTH = 1024u;" in paired_source
    assert "const bool PAIRED_WEIGHT_LAYOUT = true;" in paired_source
    assert "const bool PAIRED_WEIGHT_LAYOUT = false;" in row_major_source
    assert "binding = 4) readonly buffer Weight" in paired_source
    assert "output_c.words" in paired_source
    assert "{{" not in paired_source
    assert "{{" not in row_major_source


def test_compiler_renders_parallel_linear_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    pair = "parallel_linear_2way_paired_bf16_1024x2560_2560.comp"
    triple = "parallel_linear_3way_row_major_bf16_1024x1024_512_512.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {pair, triple})

    pair_source = (tmp_path / pair).read_text()
    triple_source = (tmp_path / triple).read_text()
    assert "binding = 3) readonly buffer WeightA" in pair_source
    assert "binding = 4) readonly buffer WeightB" in pair_source
    assert "const uint OUTPUT_A_WORDS = 2560u / 2u;" in pair_source
    assert "const bool PAIRED_WEIGHT_LAYOUT = true;" in pair_source
    assert "binding = 6) readonly buffer WeightC" in triple_source
    assert "const uint OUTPUT_C_WORDS = 512u / 2u;" in triple_source
    assert "const bool PAIRED_WEIGHT_LAYOUT = false;" in triple_source
    assert "{{" not in pair_source
    assert "{{" not in triple_source


def test_compiler_renders_fused_parallel_ffn_projection_shader(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "parallel_linear_silu_multiply_paired_bf16_1024x2560.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "binding = 2) readonly buffer GateWeight" in source
    assert "binding = 3) readonly buffer UpWeight" in source
    assert "const uint INPUT_SIZE = 1024u;" in source
    assert "const uint OUTPUT_SIZE = 2560u;" in source
    assert "const bool PAIRED_WEIGHT_LAYOUT = true;" in source
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
    assert "for (uint column = lane * 4u;" in source
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
        "linear_split_recurrent_depthwise_gate_paired_bf16_"
        "1024x1024_k3_ig0_2_og1.comp"
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
    assert "const bool PAIRED_WEIGHT_LAYOUT = true;" in source
    assert "{{" not in source


def test_parallel_linear_shader_selector_rejects_invalid_metadata_and_layout() -> None:
    node = {
        "id": "qkv",
        "op": "parallel_linear_3way",
        "inputs": ["hidden"],
        "outputs": ["q", "k", "v"],
        "params": ["q_weight", "k_weight", "v_weight"],
        "attrs": {"branch_count": 2},
    }
    circuit = {
        "parameters": {
            "refs": {
                parameter_id: {"tensor": parameter_id}
                for parameter_id in node["params"]
            }
        }
    }
    tensor_index = {
        "tensors": {
            parameter_id: {
                "dtype": "BF16",
                "shape": [512, 1024],
                "layout": VULKAN_BF16_ROW_PAIR_LAYOUT,
            }
            for parameter_id in node["params"]
        }
    }
    dimensions = {"hidden_size": 1024, "intermediate_size": 2560}

    with pytest.raises(ModelCompileError, match="inconsistent branch metadata"):
        shader_file_for_node(circuit, node, tensor_index, dimensions)

    node["attrs"]["branch_count"] = 3
    tensor_index["tensors"]["v_weight"]["layout"] = "unknown_layout"
    with pytest.raises(ModelCompileError, match="unsupported layouts"):
        shader_file_for_node(circuit, node, tensor_index, dimensions)


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
        assert "fp8_e4m3x4_to_f32" in shader
        assert "for (uint column = lane * 4u;" in shader
        assert "gl_NumSubgroups == 1u" in shader
        assert "WeightScaleInv" in shader
        assert "{{" not in shader
    assert (
        "binding = 4) readonly buffer Bias"
        in (tmp_path / "linear_bias_fp8_e4m3_b128x128_5120x17408.comp").read_text()
    )


def test_compiler_renders_native_auto_gptq_int4_linear_variants(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_int4_gptq_g128_512x768.comp",
        "linear_bias_int4_gptq_g128_512x768.comp",
        "linear_residual_int4_gptq_g128_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_int4_gptq_g128_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_int4_gptq_g128_512x768.comp").read_text()
    residual = (
        tmp_path / "linear_residual_int4_gptq_g128_512x768.comp"
    ).read_text()
    assert "const uint GROUP_SIZE = 128u;" in linear
    assert "const uint INPUT_SIZE = 512u;" in linear
    assert "const uint OUTPUT_SIZE = 768u;" in linear
    assert "& 15u) + 1u" in linear
    assert "unpackHalf2x16" in linear
    assert "readonly buffer Bias" in bias
    assert "readonly buffer ResidualFrame" in residual
    assert all("{{" not in source for source in (linear, bias, residual))


def test_compiler_renders_native_compressed_tensors_int4_linear_variants(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_int4_ct_g32_512x768.comp",
        "linear_bias_int4_ct_g32_512x768.comp",
        "linear_residual_int4_ct_g32_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_int4_ct_g32_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_int4_ct_g32_512x768.comp").read_text()
    residual = (
        tmp_path / "linear_residual_int4_ct_g32_512x768.comp"
    ).read_text()
    assert "const uint GROUP_SIZE = 32u;" in linear
    assert "row * PACKED_COLUMNS" in linear
    assert "int(quantized) - 8" in linear
    assert "read_scale(row * SCALE_COLUMNS" in bias
    assert "readonly buffer ResidualFrame" in residual
    assert all("{{" not in source for source in (linear, bias, residual))


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


def test_compiler_renders_model_owned_sampling_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    candidate_shader_file = "temperature_top_k_candidates_f32_248320_k20_g128_l256.comp"
    sampler_shader_file = (
        "temperature_top_k_top_p_sampler_f32_t0.6_k20_p0.95_g128_l256.comp"
    )

    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {candidate_shader_file, sampler_shader_file},
    )

    candidates = (tmp_path / candidate_shader_file).read_text()
    sampler = (tmp_path / sampler_shader_file).read_text()
    assert "const uint VOCAB_SIZE = 248320u;" in candidates
    assert "const uint TOP_K = 20u;" in candidates
    assert "const uint PARTITION_COUNT = 128u;" in candidates
    assert "subgroupMax(local_logit)" in candidates
    assert "const float TEMPERATURE = 0.6;" in sampler
    assert "const float TOP_P = 0.95;" in sampler
    assert "binding = 3) readonly buffer SamplerSeed" in sampler
    assert "partition_cursors" in sampler
    assert "{{" not in candidates
    assert "{{" not in sampler


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


def test_compiler_renders_hybrid_recurrent_and_gated_attention_pedals(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "causal_conv1d_silu_bf16_c6144_k4.comp",
        "gated_delta_step_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06.comp",
        "gated_delta_step_k16x128_v16x128_abf16_dtbf16_nbf16_eps1e-06.comp",
        "split_bf16_2x8x256_head_interleaved.comp",
        "sigmoid_multiply_bf16.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    convolution = (tmp_path / "causal_conv1d_silu_bf16_c6144_k4.comp").read_text()
    recurrence = (
        tmp_path
        / "gated_delta_step_k16x128_v16x128_af32_dtbf16_nf32_eps1e-06.comp"
    ).read_text()
    bf16_recurrence = (
        tmp_path
        / "gated_delta_step_k16x128_v16x128_abf16_dtbf16_nbf16_eps1e-06.comp"
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
