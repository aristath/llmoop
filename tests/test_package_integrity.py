from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

from llmoop.compilation import PACKAGE_SCHEMA, ModelCompileError
from llmoop.behavioral_compiler import (
    CONTRACT_DIGEST_ALGORITHM,
    json_contract_digest,
)
from llmoop.model_package import (
    build_package_artifact_integrity,
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
                "schema": "llmoop.tensor_index.v1",
                "tensors": {
                    "weight": {
                        "source_file": "weights/model.safetensors",
                        "data_sha256": sha256(b"weights").hexdigest(),
                    }
                }
            }
        )
    )
    circuit = {
        "schema": "llmoop.stream_circuit.v1",
        "id": "fixture_circuit",
        "source": {
            "pedal_id": "fixture_pedal",
            "source_layer_index": 0,
            "source_operator_type": "fixture",
        },
        "behavioral_role": "fixture",
        "implementation": "exact_reference",
        "boundary": {
            "inputs": [
                {"id": "input_frame", "signal": "frame", "shape": [4]}
            ],
            "outputs": [
                {
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [4],
                    "source": "output_frame",
                }
            ],
            "controls": [],
        },
        "state_ports": [],
        "parameters": {
            "layout": "row_major",
            "storage": "safetensors",
            "refs": {"weight": {"tensor": "weight"}},
        },
        "nodes": [
            {
                "id": "project",
                "op": "linear",
                "inputs": ["input_frame"],
                "outputs": ["output_frame"],
                "params": ["weight"],
            }
        ],
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
    }
    (root / "behavioral_validation.json").write_text(
        json.dumps(
            {
                "schema": "llmoop.behavioral_validation.v1",
                "status": "passed",
                "candidate_kind": "exact_reference",
                "candidate_contract_digest_algorithm": CONTRACT_DIGEST_ALGORITHM,
                "source_oracle": {
                    "model_contract_digest": "a" * 64,
                    "tensor_count": 1,
                    "parameter_count": 1,
                    "byte_count": 7,
                },
                "teacher_forced": {"status": "not_required"},
                "free_running": {"status": "not_required"},
                "circuits": [
                    {
                        "pedal_id": "fixture_pedal",
                        "candidate_kind": "exact_reference",
                        "status": "passed",
                        "source_node_count": 1,
                        "candidate_node_count": 1,
                        "covered_source_node_count": 1,
                        "candidate_contract_digest": json_contract_digest(circuit),
                        "rewrite_count": 0,
                        "rewrites": [],
                    }
                ],
            }
        )
    )
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "package_id": "fixture_package",
        "device_id": "runtime_default",
        "max_context_activations": 1024,
        "placement": {
            "schema": "llmoop.stream_circuit_placement.v1",
            "default_device_id": "runtime_default",
            "pedal_devices": {},
        },
        "circuit_graph": {
            "wiring": "explicit_graph",
            "cables": [],
            "pedals": [
                {
                    "pedal_id": "fixture_pedal",
                    "operator_type": "fixture",
                    "implementation": "exact_reference",
                    "behavioral_role": "fixture",
                    "circuit": circuit,
                    "params": {
                        "schema": "llmoop.circuit_params.v1",
                        "circuit": "fixture_circuit",
                        "layout": "row_major",
                        "storage": "safetensors",
                        "refs": {"weight": {"tensor": "weight"}},
                    },
                    "state": {
                        "schema": "llmoop.circuit_state.v1",
                        "circuit": "fixture_circuit",
                        "state_ports": [],
                    },
                }
            ],
        },
        "config_path": "config.json",
        "tensor_index_path": "tensors.json",
        "behavioral_validation_path": "behavioral_validation.json",
        "tokenizer": {"path": "tokenizer", "files": ["tokenizer.json"]},
        "pedal_executions": [
            {
                "pedal_id": "fixture_pedal",
                "operator_type": "fixture",
                "implementation": "exact_reference",
                "kernels": [
                    {
                        "execution_index": 0,
                        "node_id": "project",
                        "op": "linear",
                        "shader_path": "shaders/kernel.spv",
                    }
                ],
            }
        ],
    }
    manifest["artifact_integrity"] = build_package_artifact_integrity(root)
    return manifest


def test_package_integrity_accepts_a_complete_compiler_boundary(tmp_path: Path) -> None:
    manifest = minimal_package(tmp_path)

    validate_compiled_package(tmp_path, manifest)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schema", "unsupported schema"),
        ("config", "missing required artifact"),
        ("behavioral", "missing behavioral validation artifact"),
        ("tokenizer", "missing tokenizer artifact"),
        ("tensor", "references missing artifact"),
        ("tensor_digest", "has no valid data SHA-256"),
        ("shader", "not valid SPIR-V"),
        ("shader_digest", "does not match its integrity contract"),
        ("behavioral_shallow", "incomplete source oracle"),
        ("stale_proof", "incomplete or stale proof"),
        ("kernel", "kernel 0 does not match"),
        ("execution_identity", "execution identity does not match"),
        ("placement_device", "invalid device id"),
        ("path_escape", "must stay inside the package"),
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
    elif corruption == "behavioral":
        (tmp_path / "behavioral_validation.json").unlink()
    elif corruption == "tokenizer":
        (tmp_path / "tokenizer" / "tokenizer.json").unlink()
    elif corruption == "tensor":
        (tmp_path / "weights" / "model.safetensors").unlink()
    elif corruption == "tensor_digest":
        tensor_index = json.loads((tmp_path / "tensors.json").read_text())
        tensor_index["tensors"]["weight"]["data_sha256"] = "not-a-digest"
        (tmp_path / "tensors.json").write_text(json.dumps(tensor_index))
    elif corruption == "shader":
        (tmp_path / "shaders" / "kernel.spv").write_bytes(b"not spirv")
    elif corruption == "shader_digest":
        (tmp_path / "shaders" / "kernel.spv").write_bytes(
            b"\x03\x02#\x07changed payload"
        )
    elif corruption == "behavioral_shallow":
        (tmp_path / "behavioral_validation.json").write_text(
            json.dumps(
                {
                        "schema": "llmoop.behavioral_validation.v1",
                        "status": "passed",
                        "candidate_kind": "exact_reference",
                        "candidate_contract_digest_algorithm": CONTRACT_DIGEST_ALGORITHM,
                    }
            )
        )
    elif corruption == "stale_proof":
        evidence = json.loads((tmp_path / "behavioral_validation.json").read_text())
        evidence["circuits"][0]["candidate_node_count"] = 2
        (tmp_path / "behavioral_validation.json").write_text(json.dumps(evidence))
    elif corruption == "kernel":
        manifest["pedal_executions"][0]["kernels"][0]["op"] = "multiply"
    elif corruption == "execution_identity":
        manifest["pedal_executions"][0]["implementation"] = "wrong"
    elif corruption == "placement_device":
        manifest["placement"]["pedal_devices"]["fixture_pedal"] = ""
    elif corruption == "path_escape":
        manifest["config_path"] = "../config.json"

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
        "temperature_top_k_candidates_f32_32000_k40_g128_l256.comp",
        "temperature_top_k_top_p_sampler_f32_t0.7_k40_p0.95_g128_l256.comp",
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
