from model_package_layout_common import *

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
                "layout": ROW_MAJOR_LAYOUT,
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


def test_parallel_linear_shader_selector_supports_fp8_weight_scale_pairs() -> None:
    node = {
        "id": "qk",
        "op": "parallel_linear_2way",
        "inputs": ["hidden"],
        "outputs": ["q", "k"],
        "params": [
            "q_weight",
            "q_weight_scale_inv",
            "k_weight",
            "k_weight_scale_inv",
        ],
        "attrs": {"branch_count": 2, "branch_parameter_counts": [2, 2]},
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
            "q_weight": {
                "dtype": "F8_E4M3",
                "shape": [5120, 5120],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "k_weight": {
                "dtype": "F8_E4M3",
                "shape": [5120, 5120],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "q_weight_scale_inv": {
                "dtype": "BF16",
                "shape": [40, 40],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "k_weight_scale_inv": {
                "dtype": "BF16",
                "shape": [40, 40],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }

    dimensions = {"hidden_size": 5120, "intermediate_size": 5120}

    assert shader_file_for_node(circuit, node, tensor_index, dimensions) == (
        "parallel_linear_2way_fp8_e4m3_b128x128_5120x5120.comp"
    )
    assert workgroup_count_x_for_node(circuit, node, tensor_index) == 160


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

    expected_tile_rows = {
        "linear_fp8_e4m3_b128x128_5120x17408.comp": 64,
        "linear_bias_fp8_e4m3_b128x128_5120x17408.comp": 64,
        "linear_residual_fp8_e4m3_b128x128_17408x5120.comp": 32,
    }
    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "const uint BLOCK_ROWS = 128u;" in shader
        assert "const uint BLOCK_COLUMNS = 128u;" in shader
        assert (
            f"const uint OUTPUT_TILE_ROWS = {expected_tile_rows[shader_file]}u;"
            in shader
        )
        assert "#extension GL_EXT_spirv_intrinsics : require" in shader
        assert "SPV_VALVE_mixed_float_dot_product" in shader
        assert "fp8_dot4_acc32" in shader
        assert "shared fe4m3vec4 quantized_input" in shader
        assert "subgroupClusteredMax" in shader
        assert "for (uint word = lane;" in shader
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
        "linear_int4_gptq_sf16_g128_512x768.comp",
        "linear_bias_int4_gptq_sf16_g128_512x768.comp",
        "linear_residual_int4_gptq_sf16_g128_512x768.comp",
        "linear_batch16_int4_gptq_sf16_g128_512x768.comp",
        "linear_bias_batch16_int4_gptq_sf16_g128_512x768.comp",
        "linear_residual_batch16_int4_gptq_sf16_g128_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_int4_gptq_sf16_g128_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_int4_gptq_sf16_g128_512x768.comp").read_text()
    residual = (
        tmp_path / "linear_residual_int4_gptq_sf16_g128_512x768.comp"
    ).read_text()
    batch = (tmp_path / "linear_batch16_int4_gptq_sf16_g128_512x768.comp").read_text()
    assert "const uint GROUP_SIZE = 128u;" in linear
    assert "const uint INPUT_SIZE = 512u;" in linear
    assert "const uint OUTPUT_SIZE = 768u;" in linear
    assert "const uint OUTPUT_TILE_ROWS = 64u;" in linear
    assert "SPV_KHR_integer_dot_product" not in linear
    assert "int8_dot4" not in linear
    assert "quantized_input" not in linear
    assert "read_inputx4(batch_index, packed_column * 8u)" in linear
    assert "packed_column * OUTPUT_SIZE + row" in linear
    assert "unpackHalf2x16" in linear
    assert "readonly buffer Bias" in bias
    assert "readonly buffer ResidualFrames" in residual
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch
    assert "batch_control.batch_width" in batch
    assert all("{{" not in source for source in (linear, bias, residual, batch))


def test_compiler_renders_native_compressed_tensors_int4_linear_variants(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_int4_ct_sbf16_g32_512x768.comp",
        "linear_bias_int4_ct_sbf16_g32_512x768.comp",
        "linear_residual_int4_ct_sbf16_g32_512x768.comp",
        "linear_batch16_int4_ct_sbf16_g32_512x768.comp",
        "linear_bias_batch16_int4_ct_sbf16_g32_512x768.comp",
        "linear_residual_batch16_int4_ct_sbf16_g32_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_int4_ct_sbf16_g32_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_int4_ct_sbf16_g32_512x768.comp").read_text()
    residual = (tmp_path / "linear_residual_int4_ct_sbf16_g32_512x768.comp").read_text()
    batch = (tmp_path / "linear_batch16_int4_ct_sbf16_g32_512x768.comp").read_text()
    assert "const uint GROUP_SIZE = 32u;" in linear
    assert "const uint OUTPUT_TILE_ROWS = 16u;" in linear
    assert "row * PACKED_COLUMNS" in linear
    assert "int(packed & 15u) - 8" in linear
    assert "SPV_KHR_integer_dot_product" not in linear
    assert "int8_dot4" not in linear
    assert "quantized_input" not in linear
    assert "read_inputx4(batch_index, packed_column * 8u)" in linear
    assert "subgroupAdd" in linear
    assert "read_bf16_word(scales.words[index >> 1u], index)" in bias
    assert "readonly buffer ResidualFrames" in residual
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch
    assert "batch_control.batch_width" in batch
    assert all("{{" not in source for source in (linear, bias, residual, batch))


def test_compiler_renders_native_block_scaled_fp8_sparse_experts(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "moe_topk_bf16_e256_k8.comp",
        "moe_topk_batch1_bf16_e256_k8.comp",
        "sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_down_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_down_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "moe_reduce_bf16_h2048_k8_scale1.comp",
        "moe_reduce_batch1_bf16_h2048_k8_scale1.comp",
        "sigmoid_scalar_multiply_bf16_2048.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    gate_up_shader = (
        tmp_path / "sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    ).read_text()
    down_shader = (
        tmp_path / "sparse_moe_down_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    ).read_text()
    router_shader = (tmp_path / "moe_topk_bf16_e256_k8.comp").read_text()
    reduce_shader = (tmp_path / "moe_reduce_bf16_h2048_k8_scale1.comp").read_text()
    assert "const uint NUM_EXPERTS = 256u;" in gate_up_shader
    assert "const uint EXPERTS_PER_TOKEN = 8u;" in gate_up_shader
    assert "#extension GL_EXT_float_e4m3 : require" in gate_up_shader
    assert "uintBitsToFloate4m3EXT" in gate_up_shader
    assert "ExpertInputScaleInv" in gate_up_shader
    assert "ExpertOutputScaleInv" in down_shader
    assert "const uint TILE_ROWS = 32u;" in gate_up_shader
    assert "const uint TILE_ROWS = 64u;" in down_shader
    assert "shared fe4m3vec4 quantized_hidden" in gate_up_shader
    assert "shared fe4m3vec4 quantized_intermediate" in down_shader
    assert "SPV_VALVE_mixed_float_dot_product" in gate_up_shader
    assert "fp8_dot4_acc32" in gate_up_shader
    assert "subgroupClusteredMax" in gate_up_shader
    assert "expert_routes.words[route] = (weight << 16u) | expert;" in router_shader
    assert "route < EXPERTS_PER_TOKEN" in reduce_shader
    assert all("{{" not in source for source in (gate_up_shader, down_shader))
    assert (
        "gl_WorkGroupID.y"
        in (
            tmp_path
            / "sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
        ).read_text()
    )
    assert (
        "const uint HIDDEN_SIZE = 2048u;"
        in (tmp_path / "sigmoid_scalar_multiply_bf16_2048.comp").read_text()
    )


def test_compiler_parallelizes_only_selected_sparse_expert_routes() -> None:
    attrs = {
        "hidden_size": 2048,
        "intermediate_size": 512,
        "num_experts": 256,
        "experts_per_token": 8,
    }
    circuit = {"parameters": {"refs": {"expert_weight": {"tensor": "expert_weight"}}}}
    fp8_tensor_index = {"tensors": {"expert_weight": {"dtype": "F8_E4M3"}}}
    bf16_tensor_index = {"tensors": {"expert_weight": {"dtype": "BF16"}}}
    gate_up = {
        "op": "sparse_moe_gate_up",
        "attrs": attrs,
        "params": ["expert_weight"],
    }
    down = {
        "op": "sparse_moe_down",
        "attrs": attrs,
        "params": ["expert_weight"],
    }

    assert workgroup_count_x_for_node(circuit, gate_up, fp8_tensor_index) == 128
    assert workgroup_count_x_for_node(circuit, down, fp8_tensor_index) == 256
    assert workgroup_count_x_for_node(circuit, gate_up, bf16_tensor_index) == 2048
    assert workgroup_count_x_for_node(circuit, down, bf16_tensor_index) == 8192

    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "sparse_moe_gate_up", "op": "sparse_moe_gate_up"},
        shader_file=("sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"),
        local_size_x=64,
        workgroup_count_x=128,
    )
    assert spec["batch_mode"] == "weight_shared"
    assert [
        implementation["lane_tile_width"]
        for implementation in spec["batch_implementations"]
    ] == [1]
    assert spec["execution_domain"] == "decode"
    assert spec["batch_implementations"][0]["execution_domain"] == "prefill"
    assert spec["batch_implementations"][0]["stages"] == [
        {
            "shader_path": (
                "shaders/sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_"
                "h2048_i512_e256_k8.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": 128,
        }
    ]


def test_compiler_tiles_dense_fp8_dispatch_without_changing_bf16_dispatch() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "weight": {"tensor": "weight"},
                "gate_weight": {"tensor": "gate_weight"},
                "up_weight": {"tensor": "up_weight"},
            }
        }
    }
    fp8_tensor_index = {
        "tensors": {
            "weight": {"dtype": "F8_E4M3", "shape": [17408, 5120]},
            "gate_weight": {"dtype": "F8_E4M3", "shape": [17408, 5120]},
            "up_weight": {"dtype": "F8_E4M3", "shape": [17408, 5120]},
        }
    }
    bf16_tensor_index = {
        "tensors": {
            tensor_name: {"dtype": "BF16", "shape": [17408, 5120]}
            for tensor_name in ("weight", "gate_weight", "up_weight")
        }
    }
    linear = {"op": "linear", "params": ["weight"]}
    fused_ffn = {
        "op": "parallel_linear_silu_multiply",
        "params": ["gate_weight", "up_weight"],
    }

    assert workgroup_count_x_for_node(circuit, linear, fp8_tensor_index) == 272
    assert workgroup_count_x_for_node(circuit, fused_ffn, fp8_tensor_index) == 544
    assert workgroup_count_x_for_node(circuit, linear, bf16_tensor_index) == 8704
    assert workgroup_count_x_for_node(circuit, fused_ffn, bf16_tensor_index) == 8704


def test_compiler_tiles_int4_dispatch_by_physical_packing_format() -> None:
    circuit = {"parameters": {"refs": {"weight": {"tensor": "weight"}}}}
    node = {"id": "project", "op": "linear", "params": ["weight"]}
    auto_gptq = {
        "tensors": {
            "weight": {
                "dtype": "I32",
                "shape": [640, 17408],
                "logical_shape": [17408, 5120],
                "quantization": {"format": "auto_gptq"},
            }
        }
    }
    compressed_tensors = {
        "tensors": {
            "weight": {
                "dtype": "I32",
                "shape": [16384, 672],
                "logical_shape": [16384, 5376],
                "quantization": {"format": "compressed_tensors_pack_quantized"},
            }
        }
    }
    bf16 = {"tensors": {"weight": {"dtype": "BF16", "shape": [17408, 5120]}}}

    assert workgroup_count_x_for_node(circuit, node, auto_gptq) == 272
    assert workgroup_count_x_for_node(circuit, node, compressed_tensors) == 1024
    assert workgroup_count_x_for_node(circuit, node, bf16) == 8704


def test_compiler_rejects_fp8_sparse_expert_geometry_unsafe_for_native_dot() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "moe_input": {"tensor": "experts.gate_up"},
                "moe_input_scale_inv": {"tensor": "experts.gate_up_scale"},
            }
        }
    }
    node = {
        "id": "sparse_moe_gate_up",
        "op": "sparse_moe_gate_up",
        "params": ["moe_input", "moe_input_scale_inv"],
        "attrs": {
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "experts_per_token": 8,
        },
    }
    tensor_index = {
        "tensors": {
            "experts.gate_up": {
                "dtype": "F8_E4M3",
                "shape": [256, 1024, 2048],
                "layout": "row_major",
            },
            "experts.gate_up_scale": {
                "dtype": "BF16",
                "shape": [256, 8, 32],
                "layout": "row_major",
            },
        }
    }

    with pytest.raises(ModelCompileError, match="requires 128-column blocks"):
        fp8_moe_block_shape_for_stage(
            circuit,
            node,
            tensor_index,
            stage="gate_up",
        )

