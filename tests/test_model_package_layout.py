from __future__ import annotations

import json
import struct
from pathlib import Path

from llmoop.model_package import (
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
        "embedding_lookup_paired_bf16_32000x768.comp",
        "tied_output_projection_paired_bf16_32000x768_to_f32.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "{{" not in shader
        assert "uvec2 words[]" in shader
    assert "const uint INPUT_SIZE = 768u;" in (
        tmp_path / "linear_paired_bf16_768x2048.comp"
    ).read_text()
    assert "const uint VOCAB_SIZE = 32000u;" in (
        tmp_path / "embedding_lookup_paired_bf16_32000x768.comp"
    ).read_text()
