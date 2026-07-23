from model_structure_common import *
from model_structure_common import _tensor

def test_synthesizes_separate_experts_as_packed_circuit_parameters() -> None:
    tensors: dict[str, dict[str, object]] = {}

    def source_tensor(name: str, shape: list[int], dtype: str) -> None:
        byte_count = prod(shape) * (1 if dtype == "F8_E4M3" else 2)
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, byte_count],
            "parameter_count": prod(shape),
            "byte_count": byte_count,
            "source_file": "/tmp/source.safetensors",
            "source_header_bytes": 0,
        }

    prefix = "decoder.layers.0"
    for expert in range(2):
        base = f"{prefix}.mlp.experts.{expert}"
        for projection, shape in (
            ("gate_proj", [4, 8]),
            ("up_proj", [4, 8]),
            ("down_proj", [8, 4]),
        ):
            weight = f"{base}.{projection}.weight"
            source_tensor(weight, shape, "F8_E4M3")
            source_tensor(
                f"{weight}_scale_inv",
                [shape[0] // 2, shape[1] // 2],
                "BF16",
            )

    synthesize_packed_expert_tensors(tensors, prefix)

    assert tensors[f"{prefix}.mlp.experts.gate_up_proj"]["shape"] == [2, 8, 8]
    assert tensors[f"{prefix}.mlp.experts.down_proj"]["shape"] == [2, 8, 4]
    assert tensors[f"{prefix}.mlp.experts.gate_up_proj_scale_inv"]["shape"] == [
        2,
        4,
        4,
    ]
    assert len(tensors[f"{prefix}.mlp.experts.gate_up_proj"]["source_parts"]) == 4


def test_synthesizes_separate_compressed_tensors_experts() -> None:
    tensors: dict[str, dict[str, object]] = {}

    def source_tensor(name: str, shape: list[int], dtype: str) -> None:
        byte_width = 4 if dtype == "I32" else 2
        byte_count = prod(shape) * byte_width
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, byte_count],
            "parameter_count": prod(shape),
            "byte_count": byte_count,
            "source_file": "/tmp/source.safetensors",
            "source_header_bytes": 0,
        }

    prefix = "decoder.layers.0"
    for expert in range(2):
        base = f"{prefix}.mlp.experts.{expert}"
        for projection, logical_shape in (
            ("gate_proj", [32, 32]),
            ("up_proj", [32, 32]),
            ("down_proj", [32, 32]),
        ):
            weight = f"{base}.{projection}.weight_packed"
            scale = f"{base}.{projection}.weight_scale"
            source_tensor(weight, [logical_shape[0], logical_shape[1] // 8], "I32")
            source_tensor(scale, [logical_shape[0], logical_shape[1] // 32], "BF16")
            tensors[weight]["logical_shape"] = logical_shape
            tensors[weight]["quantization"] = {
                "format": "compressed_tensors_pack_quantized",
                "bits": 4,
                "group_size": 32,
                "symmetric": True,
                "signed_offset": 8,
                "scales": scale,
            }

    synthesize_packed_expert_tensors(tensors, prefix)

    packed_input = f"{prefix}.mlp.experts.gate_up_proj"
    packed_output = f"{prefix}.mlp.experts.down_proj"
    input_scales = f"{packed_input}_scales"
    output_scales = f"{packed_output}_scales"
    assert tensors[packed_input]["shape"] == [2, 64, 4]
    assert tensors[packed_input]["logical_shape"] == [2, 64, 32]
    assert tensors[packed_output]["shape"] == [2, 32, 4]
    assert tensors[packed_output]["logical_shape"] == [2, 32, 32]
    assert tensors[input_scales]["shape"] == [2, 64, 1]
    assert tensors[output_scales]["shape"] == [2, 32, 1]
    assert tensors[packed_input]["quantization"]["scales"] == input_scales
    assert tensors[packed_output]["quantization"]["scales"] == output_scales
    assert len(tensors[packed_input]["source_parts"]) == 4
    assert len(tensors[input_scales]["source_parts"]) == 4

