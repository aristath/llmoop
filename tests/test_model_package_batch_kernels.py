from model_package_layout_common import *

def test_compiler_selects_only_compatible_weight_shared_batch_kernels() -> None:
    assert SCALAR_BATCH_LANE_TILE_WIDTH == 16
    assert (
        weight_shared_batch_shader_file("rms_norm_bf16_h5120_eps1e-06_offset1.comp")
        == "rms_norm_batch16_bf16_h5120_eps1e-06_offset1.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_fp8_e4m3_b128x128_5120x17408.comp")
        == "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "linear_residual_fp8_e4m3_b128x128_17408x5120.comp"
        )
        == "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_int4_gptq_sf16_g128_5120x17408.comp")
        == "linear_batch16_int4_gptq_sf16_g128_5120x17408.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "linear_residual_int4_ct_sbf16_g32_16384x5376.comp"
        )
        == "linear_residual_batch16_int4_ct_sbf16_g32_16384x5376.comp"
    )
    assert (
        weight_shared_batch_shader_file("parallel_linear_2way_bf16_1024x2560_2560.comp")
        == "parallel_linear_batch16_2way_bf16_1024x2560_2560.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "parallel_linear_2way_fp8_e4m3_b128x128_5120x5120.comp"
        )
        == "parallel_linear_batch16_2way_fp8_e4m3_b128x128_5120x5120.comp"
    )
    assert weight_shared_batch_shader_file(
        "parallel_linear_silu_multiply_fp8_e4m3_b128x128_5120x17408.comp"
    ) == ("parallel_linear_silu_multiply_batch16_fp8_e4m3_b128x128_5120x17408.comp")
    assert (
        weight_shared_batch_shader_file("linear_bf16_1024x1024.comp")
        == "linear_batch16_bf16_1024x1024.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_bf16_1024x1024.comp", tile_width=4)
        == "linear_batch4_bf16_1024x1024.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_residual_bf16_1024x1024.comp")
        == "linear_residual_batch16_bf16_1024x1024.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "parallel_linear_silu_multiply_bf16_1024x4096.comp"
        )
        == "parallel_linear_silu_multiply_batch16_bf16_1024x4096.comp"
    )
    assert (
        weight_shared_batch_shader_file("split_bf16_2x16x256_head_interleaved.comp")
        == "split_batch16_bf16_2x16x256_head_interleaved.comp"
    )
    assert (
        weight_shared_batch_shader_file("sigmoid_multiply_bf16.comp")
        == "sigmoid_multiply_batch16_bf16.comp"
    )
    assert (
        weight_shared_batch_shader_file("softplus_multiply_bf16_q72_d128_per_head.comp")
        == "softplus_multiply_batch16_bf16_q72_d128_per_head.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_fp8_e4m3_b127x128_5120x17408.comp")
        is None
    )
    assert weight_shared_batch_shader_file("linear_bf16_1023x1024.comp") is None
    assert (
        frame_parallel_batch_shader_file(
            "rms_norm_batch16_bf16_h4096_eps1e-06_offset1.comp"
        )
        == "rms_norm_batch1_bf16_h4096_eps1e-06_offset1.comp"
    )
    assert (
        frame_parallel_batch_shader_file(
            "split_batch16_bf16_2x16x256_head_interleaved.comp"
        )
        == "split_batch1_bf16_2x16x256_head_interleaved.comp"
    )
    assert (
        frame_parallel_batch_shader_file("linear_batch16_bf16_4096x4096.comp") is None
    )
    assert (
        frame_parallel_batch_shader_file("moe_topk_bf16_e256_k8.comp")
        == "moe_topk_batch1_bf16_e256_k8.comp"
    )
    assert (
        frame_parallel_batch_shader_file(
            "sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
        )
        == "sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    )
    assert (
        frame_parallel_batch_shader_file("moe_reduce_bf16_h2048_k8_scale1.comp")
        == "moe_reduce_batch1_bf16_h2048_k8_scale1.comp"
    )


def test_compiler_orders_frame_parallel_before_portable_batch_implementation() -> None:
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "norm", "op": "rms_norm"},
        circuit={},
        shader_file="rms_norm_bf16_h4096_eps1e-06_offset1.comp",
        local_size_x=64,
        workgroup_count_x=1,
    )

    frame_parallel, *portable = spec["batch_implementations"]
    assert spec["execution_domain"] == "decode"
    assert frame_parallel["execution_domain"] == "prefill"
    assert frame_parallel["lane_tile_width"] == 1
    assert frame_parallel["exact_primary_equivalence"] is True
    assert frame_parallel["exact_causal_sequence_equivalence"] is True
    assert frame_parallel["device_requirements"] == {
        "vulkan_device_extensions": [],
        "vulkan_features": [],
        "subgroup_operations": [],
        "subgroup_size": 64,
    }
    assert frame_parallel["stages"][0]["shader_path"] == (
        "shaders/rms_norm_batch1_bf16_h4096_eps1e-06_offset1.comp"
    )
    assert [implementation["lane_tile_width"] for implementation in portable] == [
        2,
        4,
        8,
        16,
    ]
    assert all(
        implementation["exact_primary_equivalence"] is True
        for implementation in portable
    )
    assert all(
        implementation["exact_causal_sequence_equivalence"] is True
        for implementation in portable
    )


def test_compiler_selects_stateful_causal_scan_kernels() -> None:
    assert CAUSAL_SCAN_LANE_TILE_WIDTH == 64
    assert (
        causal_scan_batch_shader_file("causal_conv1d_silu_bf16_c8192_k4.comp")
        == "causal_conv1d_silu_temporal_bf16_c8192_k4.comp"
    )
    assert (
        causal_scan_batch_shader_file(
            "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
        )
        == "gated_delta_scan_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
    )
    assert causal_scan_batch_shader_file(
        "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
        "offset1_theta10000000_half__sc6.comp"
    ) == (
        "parallel_head_norm_rope_2way_temporal_bf16_h16_4_d256_r64_"
        "eps1e-06_offset1_theta10000000_half.comp"
    )
    assert causal_scan_batch_shader_file("linear_bf16_4096x4096.comp") is None
    assert causal_scan_workgroup_count_x("causal_conv1d_silu_bf16_c8192_k4.comp") == 64
    assert (
        causal_scan_workgroup_count_x(
            "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
        )
        == 32
    )
    assert (
        causal_scan_workgroup_count_x(
            "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
            "offset1_theta10000000_half__sc6.comp"
        )
        == 20
    )

    attention_local_size = attention_workgroup_shape(256)[0]
    assert causal_scan_batch_stages(
        "append_gqa_attention_bf16_q16_kv4_d256_scale0.0625__sc7.comp",
        attention_local_size,
    ) == [
        {
            "shader_path": (
                "shaders/append_gqa_attention_temporal_read_bf16_"
                "q16_kv4_d256_scale0.0625.comp"
            ),
            "local_size_x": attention_local_size,
            "workgroup_count_x": 16 * 64,
        },
        {
            "shader_path": "shaders/append_kv_temporal_commit_bf16_kv4_d256_w0.comp",
            "local_size_x": 64,
            "workgroup_count_x": 4,
        },
    ]
    attention_spec = component_kernel_spec(
        execution_index=0,
        node={"id": "attention", "op": "append_scaled_dot_product_attention"},
        circuit={},
        shader_file="append_gqa_attention_bf16_q16_kv4_d256_scale0.0625__sc7.comp",
        local_size_x=attention_local_size,
        workgroup_count_x=16,
    )
    temporal = attention_spec["batch_implementations"][0]
    assert temporal["execution_domain"] == "prefill"
    assert temporal["exact_primary_equivalence"] is False
    assert temporal["exact_causal_sequence_equivalence"] is True


def test_compiler_renders_temporal_attention_stages(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "append_gqa_attention_temporal_read_bf16_"
        "q16_kv4_d256_scale0.0625_w32768_sinks.comp",
        "append_kv_temporal_commit_bf16_kv4_d256_w32768_sinks.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    read_source = next(
        tmp_path.glob("append_gqa_attention_temporal_read_*.comp")
    ).read_text()
    assert "layout(set = 0, binding = 6) readonly buffer KvStateRead" in read_source
    assert "const uint ATTENTION_WINDOW = 32768u;" in read_source
    assert "absolute_tick >= batch_control.start_stream_tick_low" in read_source
    assert "uint query_head = gl_WorkGroupID.x % QUERY_HEADS;" in read_source
    assert "uint position = gl_WorkGroupID.x / QUERY_HEADS;" in read_source
    assert "if (position >= batch_control.batch_width) return;" in read_source
    commit_source = next(tmp_path.glob("append_kv_temporal_commit_*.comp")).read_text()
    assert "layout(set = 0, binding = 7) buffer KvStateWrite" in commit_source
    assert "const uint ATTENTION_WINDOW = 32768u;" in commit_source
    assert (
        "min(batch_control.dynamic_state_capacity, ATTENTION_WINDOW)" in commit_source
    )
    assert "position * KV_WORD_COUNT + head_word" in commit_source
    assert "{{" not in read_source
    assert "{{" not in commit_source


def test_compiler_selects_cooperative_bfloat16_projection_kernels() -> None:
    assert (
        cooperative_bfloat16_batch_shader_file("linear_bf16_1024x4096.comp")
        == "linear_batch64_cooperative_bf16_1024x4096.comp"
    )
    assert (
        cooperative_bfloat16_batch_shader_file("linear_residual_bf16_4096x1024.comp")
        == "linear_residual_batch64_cooperative_bf16_4096x1024.comp"
    )
    assert cooperative_bfloat16_batch_shader_file(
        "parallel_linear_3way_bf16_1024x1024_256_256.comp"
    ) == ("parallel_linear_batch64_cooperative_3way_bf16_1024x1024_256_256.comp")
    assert cooperative_bfloat16_batch_shader_file(
        "parallel_linear_silu_multiply_bf16_1024x4096.comp"
    ) == ("parallel_linear_silu_multiply_batch64_cooperative_bf16_1024x4096.comp")
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "parallel_linear_3way_bf16_1024x1024_256_256.comp"
        )
        == 24
    )
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "parallel_linear_2way_bf16_1024x1024_256.comp"
        )
        == 20
    )
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "parallel_linear_silu_multiply_bf16_1024x4096.comp"
        )
        == 64
    )
    assert (
        cooperative_bfloat16_batch_shader_file(
            "linear_fp8_e4m3_b128x128_1024x4096.comp"
        )
        is None
    )


def test_projection_component_compiles_ordered_target_native_and_scalar_implementations() -> (
    None
):
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "project", "op": "linear"},
        circuit={},
        shader_file="linear_bf16_1024x4096.comp",
        local_size_x=64,
        workgroup_count_x=2048,
    )

    assert spec["batch_mode"] == "weight_shared"
    assert "batch_shader_path" not in spec
    assert "batch_lane_tile_width" not in spec
    cooperative, *exact = spec["batch_implementations"]
    assert cooperative == {
        "execution_domain": "prefill",
        "lane_tile_width": 64,
        "exact_primary_equivalence": False,
        "exact_causal_sequence_equivalence": False,
        "device_requirements": {
            "vulkan_device_extensions": [],
            "vulkan_features": [],
            "subgroup_operations": [],
            "cooperative_bfloat16_shape": [16, 16, 16],
            "subgroup_size": 64,
        },
        "stages": [
            {
                "shader_path": (
                    "shaders/linear_batch64_cooperative_bf16_1024x4096.comp"
                ),
                "local_size_x": 256,
                "workgroup_count_x": 64,
            }
        ],
    }
    assert [implementation["lane_tile_width"] for implementation in exact] == [
        2,
        4,
        8,
        16,
    ]
    for implementation in exact:
        tile_width = implementation["lane_tile_width"]
        assert implementation == {
            "execution_domain": "decode_and_prefill",
            "lane_tile_width": tile_width,
            "exact_primary_equivalence": True,
            "exact_causal_sequence_equivalence": True,
            "device_requirements": {
                "vulkan_device_extensions": [],
                "vulkan_features": [],
                "subgroup_operations": [],
            },
            "stages": [
                {
                    "shader_path": (
                        f"shaders/linear_batch{tile_width}_bf16_1024x4096.comp"
                    ),
                    "local_size_x": 64,
                    "workgroup_count_x": 2048,
                }
            ],
        }


def test_compiler_renders_weight_shared_component_batch_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rms_norm_batch16_bf16_h5120_eps1e-06_offset1.comp",
        "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp",
        "parallel_linear_batch16_2way_bf16_1024x2560_2560.comp",
        "parallel_linear_silu_multiply_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_batch16_bf16_1024x4096.comp",
        "linear_residual_batch16_bf16_4096x1024.comp",
        "parallel_linear_silu_multiply_batch16_bf16_1024x4096.comp",
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
            assert "SPV_VALVE_mixed_float_dot_product" in source
            assert "fp8_dot4_acc32" in source
            assert "shared fe4m3vec4 quantized_input" in source
        assert "{{" not in source
    assert required_vulkan_device_extensions(tmp_path, shader_files) == [
        "VK_EXT_shader_float8",
        "VK_VALVE_shader_mixed_float_dot_product",
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
        "linear_batch64_cooperative_bf16_1024x4096.comp",
        "linear_residual_batch64_cooperative_bf16_4096x1024.comp",
        "parallel_linear_batch64_cooperative_3way_bf16_1024x1024_256_256.comp",
        "parallel_linear_silu_multiply_batch64_cooperative_bf16_1024x4096.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "coopMatMulAdd" in source
        assert "coopmat<bfloat16_t" in source
        assert "#extension GL_EXT_bfloat16 : require" in source
        assert "#extension GL_KHR_cooperative_matrix : require" in source
        assert "layout(local_size_x = 256" in source
        expected_output_tile = 64
        assert f"const uint OUTPUT_TILE = {expected_output_tile}u;" in source
        assert "const uint BATCH_TILE = 64u;" in source
        expected_result_tile = (
            "BRANCH_COUNT * OUTPUT_TILE * BATCH_TILE"
            if "silu_multiply" in shader_file
            else "OUTPUT_TILE * MATRIX_TILE"
        )
        assert f"shared float result_tile[{expected_result_tile}];" in source
        assert "{{" not in source
    direct_linear = (
        tmp_path / "linear_residual_batch64_cooperative_bf16_4096x1024.comp"
    ).read_text()
    direct_parallel = (
        tmp_path / "parallel_linear_batch64_cooperative_3way_bf16_"
        "1024x1024_256_256.comp"
    ).read_text()
    direct_fused = (
        tmp_path / "parallel_linear_silu_multiply_batch64_cooperative_bf16_"
        "1024x4096.comp"
    ).read_text()
    assert "weight.values," in direct_linear
    assert "weight_a.values," in direct_parallel
    assert "weight_b.values," in direct_parallel
    assert "weight_c.values," in direct_parallel
    assert "gate_weight.values," in direct_fused
    assert "up_weight.values," in direct_fused
    assert "const uint BRANCH_SUBGROUPS = 2u;" in direct_fused
    assert "sums[OUTPUT_SUBTILES_PER_SUBGROUP * BATCH_SUBTILES]" in direct_fused
    assert "branch * OUTPUT_TILE * BATCH_TILE" in direct_fused
    assert "coopmat<bfloat16_t" in direct_linear
    assert "uintBitsToBFloat16EXT(uint16_t(f32_to_bf16" in direct_linear
    assert "residual_frames.values," in direct_linear
    assert "gl_CooperativeMatrixLayoutColumnMajor" in direct_linear
    assert "uintBitsToBFloat16EXT" in direct_parallel
    assert "gl_CooperativeMatrixLayoutColumnMajor" in direct_parallel
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
