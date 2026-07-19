from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from llmoop.compilation import PACKAGE_SCHEMA, ModelCompileError
from llmoop.model_package import (
    compile_shader_artifacts,
    copy_exact_bytes,
    copy_shader_templates,
    render_shader_source,
    validate_compiled_package,
)


def minimal_package(root: Path) -> dict[str, object]:
    (root / "tokenizer").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "shaders").mkdir()
    (root / "config.json").write_text("{}")
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    (root / "weights" / "model.safetensors").write_bytes(b"weights")
    (root / "shaders" / "kernel.spv").write_bytes(b"\x03\x02#\x07payload")
    (root / "tensors.json").write_text(
        json.dumps(
            {
                "tensors": {
                    "weight": {"source_file": "weights/model.safetensors"}
                }
            }
        )
    )
    return {
        "schema": PACKAGE_SCHEMA,
        "config_path": "config.json",
        "tensor_index_path": "tensors.json",
        "tokenizer": {"path": "tokenizer", "files": ["tokenizer.json"]},
        "pedals": [{"kernels": [{"shader_path": "shaders/kernel.spv"}]}],
    }


def test_package_integrity_accepts_a_complete_compiler_boundary(tmp_path: Path) -> None:
    manifest = minimal_package(tmp_path)

    validate_compiled_package(tmp_path, manifest)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schema", "unsupported schema"),
        ("config", "missing required artifact"),
        ("tokenizer", "missing tokenizer artifact"),
        ("tensor", "references missing artifact"),
        ("shader", "not valid SPIR-V"),
        ("shader_reference", "does not reference any shader"),
    ],
)
def test_package_integrity_rejects_corrupt_or_incomplete_artifacts(
    tmp_path: Path, corruption: str, message: str
) -> None:
    manifest = minimal_package(tmp_path)
    if corruption == "schema":
        manifest["schema"] = "broken"
    elif corruption == "config":
        (tmp_path / "config.json").unlink()
    elif corruption == "tokenizer":
        (tmp_path / "tokenizer" / "tokenizer.json").unlink()
    elif corruption == "tensor":
        (tmp_path / "weights" / "model.safetensors").unlink()
    elif corruption == "shader":
        (tmp_path / "shaders" / "kernel.spv").write_bytes(b"not spirv")
    elif corruption == "shader_reference":
        manifest["pedals"] = []

    with pytest.raises(ModelCompileError, match=message):
        validate_compiled_package(tmp_path, manifest)


def test_shader_templates_compile_to_vulkan_1_4_spirv(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_paired_bf16_768x2048.comp",
        "rms_norm_bf16_h768_eps1e-05_offset0.comp",
        "rotary_bf16_12x64_r64_theta10000_half__sc2.comp",
        "append_kv_state_bf16_4x64__sc9.comp",
        "gqa_attention_bf16_q12_kv4_d64_scale0.125__sc6.comp",
        "causal_conv1d_silu_bf16_c768_k4.comp",
        "rg_lru_step_bf16_h768_b6x128_k4__sc13.comp",
        "moe_topk_bf16_e64_k8.comp",
        "sparse_moe_experts_bf16_h768_i256_e64_k8.comp",
        "temperature_top_k_top_p_sampler_f32_32000_t0.7_k40_p0.95_l64.comp",
    }
    shader_dir = tmp_path / "shaders"
    copy_shader_templates(shader_source_dir, shader_dir, shader_files)

    compile_shader_artifacts(shader_dir)

    assert not list(shader_dir.glob("*.comp"))
    artifacts = sorted(shader_dir.glob("*.spv"))
    assert len(artifacts) == len(shader_files)
    assert all(path.read_bytes().startswith(b"\x03\x02#\x07") for path in artifacts)


def test_shader_renderer_rejects_unknown_and_partially_bound_templates(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelCompileError, match="missing shader source or template"):
        render_shader_source(tmp_path, "unknown_kernel.comp")

    (tmp_path / "linear_bf16.comp.template").write_text(
        "const uint input = {{INPUT_SIZE}};\n"
        "const uint output = {{OUTPUT_SIZE}};\n"
        "const uint forgotten = {{UNBOUND_VALUE}};\n"
    )
    with pytest.raises(ModelCompileError, match="UNBOUND_VALUE"):
        render_shader_source(tmp_path, "linear_bf16_8x16.comp")


def test_exact_tensor_copy_rejects_truncated_sources() -> None:
    destination = BytesIO()

    with pytest.raises(ModelCompileError, match="unexpected end"):
        copy_exact_bytes(BytesIO(b"short"), destination, 6)

    assert destination.getvalue() == b"short"
