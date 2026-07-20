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
                },
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
        "runtime_role": "signal_processor",
        "behavioral_role": "fixture",
        "implementation": "exact_reference",
        "boundary": {
            "inputs": [{"id": "input_frame", "signal": "frame", "shape": [4]}],
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
    input_circuit = {
        "schema": "llmoop.stream_circuit.v1",
        "id": "input_circuit",
        "source": {
            "pedal_id": "input",
            "source_layer_index": None,
            "source_operator_type": "input_transducer",
        },
        "runtime_role": "input_transducer",
        "behavioral_role": "token_to_frame",
        "implementation": "compiled_input_transducer_v1",
        "boundary": {
            "inputs": [{"id": "token", "signal": "token_id", "shape": [1]}],
            "outputs": [
                {
                    "id": "frame",
                    "signal": "frame",
                    "shape": [4],
                    "source": "frame",
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
                "id": "lookup",
                "op": "embedding_lookup",
                "inputs": ["token"],
                "outputs": ["frame"],
                "params": ["weight"],
            }
        ],
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
    }
    output_circuit = {
        "schema": "llmoop.stream_circuit.v1",
        "id": "output_circuit",
        "source": {
            "pedal_id": "output",
            "source_layer_index": None,
            "source_operator_type": "output_transducer",
        },
        "runtime_role": "output_transducer",
        "behavioral_role": "frame_to_logits",
        "implementation": "compiled_output_transducer_v1",
        "boundary": {
            "inputs": [{"id": "frame", "signal": "frame", "shape": [4]}],
            "outputs": [
                {
                    "id": "logits",
                    "signal": "logits",
                    "shape": [4],
                    "source": "logits",
                }
            ],
            "controls": [],
        },
        "state_ports": [],
        "parameters": {
            "layout": "row_major",
            "storage": "safetensors",
            "refs": {
                "output_norm.weight": {"tensor": "weight"},
                "output_projection.weight": {"tensor": "weight"},
            },
        },
        "nodes": [
            {
                "id": "norm",
                "op": "rms_norm",
                "inputs": ["frame"],
                "outputs": ["normalized"],
                "params": ["output_norm.weight"],
            },
            {
                "id": "project",
                "op": "linear_projection",
                "inputs": ["normalized"],
                "outputs": ["logits"],
                "params": ["output_projection.weight"],
            },
        ],
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
    }
    sampler_circuit = {
        "schema": "llmoop.stream_circuit.v1",
        "id": "sampler_circuit",
        "source": {
            "pedal_id": "sampler",
            "source_layer_index": None,
            "source_operator_type": "sampler",
        },
        "runtime_role": "sampler",
        "behavioral_role": "logits_to_token",
        "implementation": "compiled_sampler_v1",
        "boundary": {
            "inputs": [
                {"id": "logits", "signal": "logits", "shape": [4]},
                {"id": "random_seed", "signal": "random_seed", "shape": [1]},
            ],
            "outputs": [
                {
                    "id": "token",
                    "signal": "token_id",
                    "shape": [1],
                    "source": "token",
                }
            ],
            "controls": [],
        },
        "state_ports": [],
        "parameters": {
            "layout": "row_major",
            "storage": "safetensors",
            "refs": {},
        },
        "nodes": [
            {
                "id": "sample",
                "op": "sample_token",
                "inputs": ["logits", "random_seed"],
                "outputs": ["token"],
                "params": [],
                "attrs": {
                    "method": "greedy",
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "randomness": "seed_and_stream_tick",
                },
            }
        ],
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
    }
    circuits = {
        "input": input_circuit,
        "fixture_pedal": circuit,
        "output": output_circuit,
        "sampler": sampler_circuit,
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
                        "pedal_id": pedal_id,
                        "candidate_kind": "exact_reference",
                        "status": "passed",
                        "source_node_count": len(candidate["nodes"]),
                        "candidate_node_count": len(candidate["nodes"]),
                        "covered_source_node_count": len(candidate["nodes"]),
                        "candidate_contract_digest": json_contract_digest(candidate),
                        "rewrite_count": 0,
                        "rewrites": [],
                    }
                    for pedal_id, candidate in circuits.items()
                ],
            }
        )
    )
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "package_id": "fixture_package",
        "max_context_activations": 1024,
        "required_vulkan_device_extensions": [],
        "circuit_graph": {
            "wiring": "explicit_graph",
            "cables": [
                {
                    "id": "input_to_processor",
                    "connection": {"kind": "forward"},
                    "source": {"pedal_id": "input", "port_id": "frame"},
                    "destination": {
                        "pedal_id": "fixture_pedal",
                        "port_id": "input_frame",
                    },
                },
                {
                    "id": "processor_to_output",
                    "connection": {"kind": "forward"},
                    "source": {
                        "pedal_id": "fixture_pedal",
                        "port_id": "output_frame",
                    },
                    "destination": {"pedal_id": "output", "port_id": "frame"},
                },
                {
                    "id": "output_to_sampler",
                    "connection": {"kind": "forward"},
                    "source": {"pedal_id": "output", "port_id": "logits"},
                    "destination": {"pedal_id": "sampler", "port_id": "logits"},
                },
                {
                    "id": "feedback",
                    "connection": {
                        "kind": "temporal_feedback",
                        "delay_activations": 1,
                    },
                    "source": {"pedal_id": "sampler", "port_id": "token"},
                    "destination": {"pedal_id": "input", "port_id": "token"},
                },
            ],
            "boundary": {
                "external_inputs": [
                    {
                        "id": "model_input",
                        "endpoint": {
                            "pedal_id": "input",
                            "port_id": "token",
                        },
                    },
                    {
                        "id": "random_seed",
                        "endpoint": {
                            "pedal_id": "sampler",
                            "port_id": "random_seed",
                        },
                    },
                ],
                "public_outputs": [
                    {
                        "id": "model_output",
                        "endpoint": {
                            "pedal_id": "sampler",
                            "port_id": "token",
                        },
                    }
                ],
            },
            "pedals": [
                {
                    "pedal_id": pedal_id,
                    "operator_type": candidate["source"]["source_operator_type"],
                    "runtime_role": candidate["runtime_role"],
                    "implementation": candidate["implementation"],
                    "behavioral_role": candidate["behavioral_role"],
                    "circuit": candidate,
                    "params": {
                        "schema": "llmoop.circuit_params.v1",
                        "circuit": candidate["id"],
                        "layout": candidate["parameters"]["layout"],
                        "storage": candidate["parameters"]["storage"],
                        "refs": candidate["parameters"]["refs"],
                    },
                    "state": {
                        "schema": "llmoop.circuit_state.v1",
                        "circuit": candidate["id"],
                        "state_ports": [],
                    },
                }
                for pedal_id, candidate in circuits.items()
            ],
        },
        "config_path": "config.json",
        "tensor_index_path": "tensors.json",
        "behavioral_validation_path": "behavioral_validation.json",
        "tokenizer": {"path": "tokenizer", "files": ["tokenizer.json"]},
        "input_transducer": {
            "spec": {
                "parameter_tensor": "weight",
                "output_signal_id": "input_frame",
            },
            "shader_path": "shaders/kernel.spv",
            "batch_shader_path": "shaders/kernel.spv",
        },
        "output_transducer": {
            "spec": {
                "node_ids": ["norm", "project"],
                "norm_parameter_tensor": "weight",
                "projection_parameter_tensor": "weight",
                "input_signal_id": "output_frame",
            },
            "embedding_norm_shader_path": "shaders/kernel.spv",
            "projection_shader_path": "shaders/kernel.spv",
            "projection_batch_shader_path": "shaders/kernel.spv",
            "projection_batch_lane_tile_width": 4,
        },
        "sampler": {
            "spec": {
                "method": "greedy",
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 1.0,
            },
            "kernels": [{"shader_path": "shaders/kernel.spv"}],
        },
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
                        "batch_mode": "serial_lanes",
                        "batch_implementations": [],
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
        ("sampler_contract", "sampler execution does not match"),
        ("batch_contract", "invalid batch execution contract"),
        ("device_extensions", "required Vulkan device extensions"),
        ("generation_boundary", "boundaries must expose"),
        ("compiler_placement", "must not contain runtime placement fields"),
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
    elif corruption == "sampler_contract":
        manifest["sampler"]["spec"]["top_k"] = 2
    elif corruption == "batch_contract":
        manifest["pedal_executions"][0]["kernels"][0]["batch_mode"] = (
            "weight_shared"
        )
    elif corruption == "device_extensions":
        manifest["required_vulkan_device_extensions"] = [
            "VK_EXT_shader_float8",
            "VK_EXT_shader_float8",
        ]
    elif corruption == "generation_boundary":
        manifest["circuit_graph"]["boundary"]["public_outputs"][0]["endpoint"] = {
            "pedal_id": "output",
            "port_id": "logits",
        }
    elif corruption == "compiler_placement":
        manifest["placement"] = {
            "schema": "llmoop.stream_circuit_placement.v1",
            "default_device_id": "gpu0",
            "pedal_devices": {"fixture_pedal": "gpu0"},
        }
    elif corruption == "path_escape":
        manifest["config_path"] = "../config.json"

    with pytest.raises(ModelCompileError, match=message):
        validate_compiled_package(tmp_path, manifest)


def test_shader_templates_compile_to_vulkan_1_4_spirv(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_bf16_768x2048.comp",
        "embedding_lookup_batch_bf16_32000x768_scale12.comp",
        "linear_batch16_bf16_768x2048.comp",
        "linear_residual_batch16_bf16_2048x768.comp",
        "linear_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_bias_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_residual_fp8_e4m3_b128x128_17408x5120.comp",
        "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp",
        "parallel_linear_batch16_2way_bf16_1024x2560_2560.comp",
        "parallel_linear_silu_multiply_fp8_e4m3_b128x128_5120x17408.comp",
        "parallel_linear_silu_multiply_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "parallel_linear_silu_multiply_batch16_bf16_768x2048.comp",
        "per_layer_embedding_bf16_v32000_h768_p128_l2of6_eps1e-05_"
        "tes1_pes1_mps1_cs1__sc5.comp",
        "rms_norm_bf16_h768_eps1e-05_offset0.comp",
        "rotary_bf16_12x64_r64_theta10000_half__sc2.comp",
        "append_kv_state_bf16_4x64__sc9.comp",
        "gqa_attention_bf16_q12_kv4_d64_scale0.125__sc6.comp",
        "causal_conv1d_silu_bf16_c768_k4.comp",
        "causal_conv1d_silu_temporal_bf16_c768_k4.comp",
        "gated_delta_scan_k4x64_v4x64_af32_dtbf16_nf32_eps1e-06.comp",
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
