from model_package_layout_common import *
from nerve.model_package_derived_tensors import (
    derive_internal_q8_linear_tensors,
    derive_output_projection_tensors,
)
from nerve.model_package_tensors import (
    e4m3fn_to_f32,
    f32_to_bf16_bytes,
    f32_to_e4m3fn,
    write_compiled_derived_fp8_e4m3_output_projection,
    write_compiled_derived_q8_0_from_fp8_e4m3,
)

import numpy as np


def test_write_compiled_tensor_preserves_canonical_row_major_order(
    tmp_path: Path,
) -> None:
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
        layout=ROW_MAJOR_LAYOUT,
    )

    compiled = destination.read_bytes()
    header_bytes = struct.unpack("<Q", compiled[:8])[0]
    payload = compiled[8 + header_bytes :]
    assert struct.unpack("<16H", payload) == values


def test_compiler_derives_fp8_output_projection_tensor_pair(tmp_path: Path) -> None:
    source_tensor = "lm_head.weight"
    source = tmp_path / "source.safetensors"
    values = tuple([0x3F80] * (16 * 128))
    source_header = {
        source_tensor: {
            "dtype": "BF16",
            "shape": [16, 128],
            "data_offsets": [0, len(values) * 2],
        }
    }
    source_header_payload = json.dumps(source_header).encode("utf-8")
    source.write_bytes(
        struct.pack("<Q", len(source_header_payload))
        + source_header_payload
        + struct.pack(f"<{len(values)}H", *values)
    )
    tensor_index = {
        "tensors": {
            source_tensor: {
                "dtype": "BF16",
                "shape": [16, 128],
                "source_file": str(source),
                "source_header_bytes": len(source_header_payload),
                "data_offsets": [0, len(values) * 2],
                "parameter_count": len(values),
                "byte_count": len(values) * 2,
            }
        }
    }
    model_graph = {
        "graph": {
            "output_transducer": {
                "components": [
                    {"id": "output_norm", "type": "rms_norm", "params": {}},
                    {
                        "id": "output_projection",
                        "type": "linear_projection",
                        "params": {"weight": {"tensor": source_tensor}},
                    },
                ]
            },
            "draft_execution_graphs": [
                {
                    "id": "draft_00",
                    "output_transducer": {
                        "params": {"projection": {"tensor": source_tensor}}
                    },
                }
            ],
        }
    }

    derive_output_projection_tensors(model_graph, tensor_index)

    projection = model_graph["graph"]["output_transducer"]["components"][1]
    weight = projection["params"]["weight"]["tensor"]
    scale = projection["params"]["weight_scale_inv"]["tensor"]
    assert weight == "lm_head.weight.__nerve_output_fp8_e4m3"
    assert scale == "lm_head.weight.__nerve_output_fp8_e4m3_scale_inv"
    assert tensor_index["tensors"][weight]["dtype"] == "F8_E4M3"
    assert tensor_index["tensors"][weight]["byte_count"] == 16 * 128
    assert tensor_index["tensors"][scale]["dtype"] == "BF16"
    assert tensor_index["tensors"][scale]["shape"] == [1, 1]
    draft_output = model_graph["graph"]["draft_execution_graphs"][0][
        "output_transducer"
    ]["params"]
    assert draft_output["projection"]["tensor"] == weight
    assert draft_output["weight_scale_inv"]["tensor"] == scale

    destinations = {
        weight: tmp_path / "weight.safetensors",
        scale: tmp_path / "scale.safetensors",
    }
    digests = write_compiled_derived_fp8_e4m3_output_projection(
        weight_tensor_name=weight,
        weight_info=tensor_index["tensors"][weight],
        weight_destination=destinations[weight],
        scale_tensor_name=scale,
        scale_info=tensor_index["tensors"][scale],
        scale_destination=destinations[scale],
        layout=ROW_MAJOR_LAYOUT,
    )
    assert set(digests) == {weight, scale}
    assert destinations[weight].stat().st_size > 16 * 128
    assert destinations[scale].stat().st_size > 2


def test_compiler_writes_internal_q8_0_blocks_from_block_scaled_fp8(
    tmp_path: Path,
) -> None:
    tensor_name = "layer.weight.__nerve_q8_0"
    source_tensor = "layer.weight"
    scale_tensor = "layer.weight_scale_inv"
    source_values = np.linspace(-4.0, 4.0, 64, dtype=np.float32).reshape(2, 32)
    fp8_bytes = f32_to_e4m3fn(source_values).tobytes(order="C")
    scale_bytes = f32_to_bf16_bytes(np.asarray([1.0], dtype=np.float32))
    source_header = {
        source_tensor: {
            "dtype": "F8_E4M3",
            "shape": [2, 32],
            "data_offsets": [0, len(fp8_bytes)],
        }
    }
    scale_header = {
        scale_tensor: {
            "dtype": "BF16",
            "shape": [1, 1],
            "data_offsets": [0, len(scale_bytes)],
        }
    }
    source_header_payload = json.dumps(source_header).encode("utf-8")
    scale_header_payload = json.dumps(scale_header).encode("utf-8")
    source = tmp_path / "source.safetensors"
    scale_source = tmp_path / "scale.safetensors"
    source.write_bytes(
        struct.pack("<Q", len(source_header_payload))
        + source_header_payload
        + fp8_bytes
    )
    scale_source.write_bytes(
        struct.pack("<Q", len(scale_header_payload))
        + scale_header_payload
        + scale_bytes
    )
    destination = tmp_path / "q8.safetensors"

    header_bytes, data_sha256 = write_compiled_derived_q8_0_from_fp8_e4m3(
        tensor_name=tensor_name,
        info={
            "dtype": "Q8_0",
            "shape": [2, 1, 9],
            "logical_shape": [2, 32],
            "byte_count": 2 * 36,
            "derived": {
                "kind": "fp8_e4m3_to_q8_0",
                "source_tensor": source_tensor,
                "source_file": str(source),
                "source_header_bytes": len(source_header_payload),
                "data_offsets": [0, len(fp8_bytes)],
                "source_shape": [2, 32],
                "scale_tensor": scale_tensor,
                "scale_source_file": str(scale_source),
                "scale_source_header_bytes": len(scale_header_payload),
                "scale_data_offsets": [0, len(scale_bytes)],
                "scale_shape": [1, 1],
            },
        },
        destination=destination,
        layout=ROW_MAJOR_LAYOUT,
    )

    compiled = destination.read_bytes()
    stored_header_bytes = struct.unpack("<Q", compiled[:8])[0]
    header = json.loads(compiled[8 : 8 + stored_header_bytes])
    payload = compiled[8 + stored_header_bytes :]
    assert header_bytes == stored_header_bytes
    assert len(data_sha256) == 64
    assert header[tensor_name]["dtype"] == "Q8_0"
    assert header[tensor_name]["shape"] == [2, 1, 9]
    assert len(payload) == 72

    expected = e4m3fn_to_f32(np.frombuffer(fp8_bytes, dtype=np.uint8)).reshape(2, 32)
    reconstructed = np.empty((2, 32), dtype=np.float32)
    for row in range(2):
        block = payload[row * 36 : (row + 1) * 36]
        scale_word = np.frombuffer(block[:2], dtype="<u2").astype(np.uint32) << 16
        scale_value = scale_word.view(np.float32)[0]
        quantized = np.frombuffer(block[4:], dtype=np.int8).astype(np.float32)
        reconstructed[row, :] = quantized * scale_value
    assert np.max(np.abs(reconstructed - expected)) < 0.04


def test_compiler_rewrites_eligible_fp8_linears_to_internal_q8(
    tmp_path: Path,
) -> None:
    lowered_dir = tmp_path / "lowered"
    circuit_dir = lowered_dir / "layer_00"
    circuit_dir.mkdir(parents=True)
    circuit = {
        "parameters": {
            "refs": {
                "projection": {
                    "tensor": "layer.proj.weight",
                    "role": "projection",
                },
                "projection_scale_inv": {
                    "tensor": "layer.proj.weight_scale_inv",
                    "role": "projection_scale",
                },
            }
        },
        "nodes": [
            {
                "id": "projection",
                "op": "linear",
                "inputs": ["hidden"],
                "outputs": ["projected"],
                "params": ["projection", "projection_scale_inv"],
            }
        ],
    }
    (circuit_dir / "circuit.json").write_text(json.dumps(circuit))
    lowered_index = {
        "graph": {
            "circuits": [
                {
                    "id": "layer_00",
                    "circuit": "layer_00/circuit.json",
                }
            ]
        }
    }
    tensor_index = {
        "tensors": {
            "layer.proj.weight": {
                "dtype": "F8_E4M3",
                "shape": [64, 128],
                "parameter_count": 64 * 128,
                "byte_count": 64 * 128,
                "source_file": "/models/source.safetensors",
                "source_header_bytes": 128,
                "data_offsets": [0, 64 * 128],
            },
            "layer.proj.weight_scale_inv": {
                "dtype": "BF16",
                "shape": [1, 1],
                "parameter_count": 1,
                "byte_count": 2,
                "source_file": "/models/source.safetensors",
                "source_header_bytes": 128,
                "data_offsets": [64 * 128, 64 * 128 + 2],
            },
        }
    }

    derive_internal_q8_linear_tensors(lowered_index, lowered_dir, tensor_index)

    rewritten = json.loads((circuit_dir / "circuit.json").read_text())
    q8_tensor = "layer.proj.weight.__nerve_q8_0"
    assert rewritten["nodes"][0]["params"] == ["projection"]
    assert rewritten["parameters"]["refs"] == {
        "projection": {"tensor": q8_tensor, "role": "projection"}
    }
    assert tensor_index["tensors"][q8_tensor]["dtype"] == "Q8_0"
    assert tensor_index["tensors"][q8_tensor]["shape"] == [64, 4, 9]
    assert tensor_index["tensors"][q8_tensor]["logical_shape"] == [64, 128]
    assert tensor_index["tensors"][q8_tensor]["byte_count"] == 64 * 4 * 36
    assert tensor_index["tensors"][q8_tensor]["derived"]["kind"] == (
        "fp8_e4m3_to_q8_0"
    )


def test_compiler_rewrites_parallel_and_fused_fp8_linears_to_internal_q8(
    tmp_path: Path,
) -> None:
    lowered_dir = tmp_path / "lowered"
    circuit_dir = lowered_dir / "layer_00"
    circuit_dir.mkdir(parents=True)
    circuit = {
        "parameters": {
            "refs": {
                parameter: {
                    "tensor": f"layer.{parameter}.weight"
                    if not parameter.endswith("_scale_inv")
                    else f"layer.{parameter.removesuffix('_scale_inv')}.weight_scale_inv"
                }
                for parameter in (
                    "q",
                    "q_scale_inv",
                    "k",
                    "k_scale_inv",
                    "gate",
                    "gate_scale_inv",
                    "up",
                    "up_scale_inv",
                )
            }
        },
        "nodes": [
            {
                "id": "qk",
                "op": "parallel_linear_2way",
                "inputs": ["hidden"],
                "outputs": ["q", "k"],
                "params": ["q", "q_scale_inv", "k", "k_scale_inv"],
                "attrs": {"branch_count": 2, "branch_parameter_counts": [2, 2]},
            },
            {
                "id": "ffn_gate_up",
                "op": "parallel_linear_silu_multiply",
                "inputs": ["hidden"],
                "outputs": ["ffn"],
                "params": ["gate", "gate_scale_inv", "up", "up_scale_inv"],
                "attrs": {
                    "branch_count": 2,
                    "element_count": 64,
                    "intermediate_rounding": "BF16",
                },
            },
        ],
    }
    (circuit_dir / "circuit.json").write_text(json.dumps(circuit))
    lowered_index = {
        "graph": {
            "circuits": [
                {
                    "id": "layer_00",
                    "circuit": "layer_00/circuit.json",
                }
            ]
        }
    }
    tensor_index = {"tensors": {}}
    for tensor in ("q", "k", "gate", "up"):
        tensor_index["tensors"][f"layer.{tensor}.weight"] = {
            "dtype": "F8_E4M3",
            "shape": [64, 128],
            "parameter_count": 64 * 128,
            "byte_count": 64 * 128,
            "source_file": "/models/source.safetensors",
            "source_header_bytes": 128,
            "data_offsets": [0, 64 * 128],
        }
        tensor_index["tensors"][f"layer.{tensor}.weight_scale_inv"] = {
            "dtype": "BF16",
            "shape": [1, 1],
            "parameter_count": 1,
            "byte_count": 2,
            "source_file": "/models/source.safetensors",
            "source_header_bytes": 128,
            "data_offsets": [64 * 128, 64 * 128 + 2],
        }

    derive_internal_q8_linear_tensors(lowered_index, lowered_dir, tensor_index)

    rewritten = json.loads((circuit_dir / "circuit.json").read_text())
    assert rewritten["nodes"][0]["params"] == ["q", "k"]
    assert rewritten["nodes"][0]["attrs"]["branch_parameter_counts"] == [1, 1]
    assert rewritten["nodes"][1]["params"] == ["gate", "up"]
    assert set(rewritten["parameters"]["refs"]) == {"q", "k", "gate", "up"}
    for parameter in ("q", "k", "gate", "up"):
        q8_tensor = f"layer.{parameter}.weight.__nerve_q8_0"
        assert rewritten["parameters"]["refs"][parameter]["tensor"] == q8_tensor
        assert tensor_index["tensors"][q8_tensor]["dtype"] == "Q8_0"
        assert tensor_index["tensors"][q8_tensor]["shape"] == [64, 4, 9]


def test_compiler_renders_row_major_matrix_and_transducer_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_bf16_768x2048.comp",
        "linear_residual_bf16_2048x768.comp",
        "embedding_lookup_bf16_32000x768_scale12.comp",
        "embedding_lookup_batch_bf16_32000x768_scale12.comp",
        "tied_output_projection_bf16_32000x768_scale0.166666667_to_f32.comp",
        "tied_output_projection_batch4_bf16_32000x768_scale0.166666667_to_f32.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "{{" not in shader
        assert "uint words[]" in shader
    assert (
        "const uint INPUT_SIZE = 768u;"
        in (tmp_path / "linear_bf16_768x2048.comp").read_text()
    )
    assert (
        "const uint VOCAB_SIZE = 32000u;"
        in (tmp_path / "embedding_lookup_bf16_32000x768_scale12.comp").read_text()
    )
    assert (
        "gl_WorkGroupID.y"
        in (tmp_path / "embedding_lookup_batch_bf16_32000x768_scale12.comp").read_text()
    )
    assert (
        "const float EMBEDDING_SCALE = 12;"
        in (tmp_path / "embedding_lookup_bf16_32000x768_scale12.comp").read_text()
    )
    assert (
        "const float OUTPUT_SCALE = 0.166666667;"
        in (
            tmp_path
            / "tied_output_projection_bf16_32000x768_scale0.166666667_to_f32.comp"
        ).read_text()
    )
    batched_projection = (
        tmp_path
        / "tied_output_projection_batch4_bf16_32000x768_scale0.166666667_to_f32.comp"
    ).read_text()
    assert "const uint BATCH_TILE_WIDTH = 4u;" in batched_projection
    assert "layout(push_constant) uniform BatchControl" in batched_projection
    assert "gl_WorkGroupID.y * BATCH_TILE_WIDTH" in batched_projection


def test_compiler_renders_direct_three_way_linear_split_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "linear_split_3way_bf16_1024x1024_1024_1024.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "const uint INPUT_SIZE = 1024u;" in source
    assert "const uint PART_A_WIDTH = 1024u;" in source
    assert "binding = 4) readonly buffer Weight" in source
    assert "output_c.words" in source
    assert "PAIRED_WEIGHT_LAYOUT" not in source
    assert "{{" not in source


def test_compiler_renders_row_major_per_layer_embedding_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "per_layer_embedding_bf16_v32000_h1024_p128_l2of8_c3r12000_"
        "eps1e-06_tes1_pes1_mps1_cs1__sc7.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "readonly buffer TokenEmbedding { uint words[]; }" in source
    assert "binding = 2) readonly buffer PerLayerEmbeddingChunk0" in source
    assert "binding = 4) readonly buffer PerLayerEmbeddingChunk2" in source
    assert "readonly buffer ModelProjection { uint words[]; }" in source
    assert "token_id * INPUT_WORDS + word" in source
    assert "uint chunk = token_id / EMBEDDING_CHUNK_ROWS;" in source
    assert "row * PACKED_WORDS + word" in source
    assert "row * INPUT_WORDS + word" in source
    assert "uvec2" not in source
    assert "layout(set = 0, binding = 7) readonly buffer StreamControl" in source
    assert "round_bf16(lo_projection + lo_identity)" in source
    assert "{{" not in source
