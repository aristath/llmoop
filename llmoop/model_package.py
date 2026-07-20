from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from copy import deepcopy
from hashlib import blake2s, sha256
from pathlib import Path
from typing import Any, Callable

from llmoop.behavioral_compiler import (
    build_behavioral_validation,
    validate_behavioral_validation_artifact,
)
from llmoop.circuit_ir import validate_circuit
from llmoop.circuit_lowering import lower_pedalboard
from llmoop.circuit_optimizer import optimize_circuit_for_vulkan
from llmoop.compilation import (
    PACKAGE_SCHEMA,
    CompiledModelReport,
    Json,
    ModelCompileError,
    check_compile_cancelled,
    emit_compile_event,
    read_json,
    relative_json_path,
    write_json,
)
from llmoop.model_transpiler import read_safetensors_header, transpile_model


TOKENIZER_PACKAGE_DIR = "tokenizer"
WEIGHTS_PACKAGE_DIR = "weights"
PACKAGE_ARTIFACT_INTEGRITY_SCHEMA = "llmoop.package_artifact_integrity.v1"
VULKAN_BF16_ROW_PAIR_LAYOUT = "vulkan_bf16_row_pair_u32"
ROW_MAJOR_LAYOUT = "row_major"
CONFIG_PACKAGE_FILE = "config.json"
PEDAL_BATCH_LANE_TILE_WIDTH = 4
GLSL_VULKAN_DEVICE_EXTENSION_REQUIREMENTS = {
    "GL_EXT_float_e4m3": "VK_EXT_shader_float8",
}
TOKENIZER_PACKAGE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def compile_model_package(
    model_dir: Path,
    *,
    transpiled_dir: Path | None,
    lowered_dir: Path | None,
    package_dir: Path | None,
    shader_source_dir: Path,
    event_sink: Callable[[Json], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> CompiledModelReport:
    slug = compiled_model_slug(model_dir)
    transpiled_dir = transpiled_dir or Path("transpiled") / slug
    lowered_dir = lowered_dir or Path("lowered") / slug
    package_dir = package_dir or Path("packages") / slug

    structure = transpile_model(
        model_dir,
        transpiled_dir,
        progress=lambda current, total, pedal_id: emit_compile_event(
            event_sink,
            "PedalTranspiled",
            current=current,
            total=total,
            pedal_id=pedal_id,
        ),
        cancel_requested=cancel_requested,
    )
    check_compile_cancelled(cancel_requested)
    if lowered_dir.exists():
        shutil.rmtree(lowered_dir)
    lowered = lower_pedalboard(
        transpiled_dir,
        lowered_dir,
        progress=lambda current, total, pedal_id: emit_compile_event(
            event_sink,
            "PedalLoweringStarted",
            current=current,
            total=total,
            pedal_id=pedal_id,
        ),
        cancel_requested=cancel_requested,
    )
    check_compile_cancelled(cancel_requested)
    tensor_index = read_json(transpiled_dir / "tensors.json")
    model_graph = read_json(transpiled_dir / "model.json")
    tensor_index = referenced_tensor_index(
        tensor_index,
        model_graph=model_graph,
        lowered_index=lowered["index"],
        lowered_dir=lowered_dir,
    )

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    emit_compile_event(
        event_sink, "ArtifactWritingStarted", package_dir=str(package_dir)
    )
    write_runtime_config_package(model_graph, package_dir)
    tokenizer_manifest = copy_tokenizer_package(
        model_dir, package_dir / TOKENIZER_PACKAGE_DIR
    )
    packaged_tensor_index = copy_tensor_package(
        tensor_index,
        package_dir,
        progress=lambda current, total, tensor_name: emit_compile_event(
            event_sink,
            "TensorPackagingStarted",
            current=current,
            total=total,
            tensor_name=tensor_name,
        ),
        cancel_requested=cancel_requested,
    )
    package_manifest = build_vulkan_resident_package_manifest(
        model_graph=model_graph,
        tensor_index=packaged_tensor_index,
        lowered_index=lowered["index"],
        lowered_dir=lowered_dir,
        package_dir=package_dir,
        package_id=f"{slug}_vulkan_resident",
        shader_source_dir=shader_source_dir,
        tokenizer_manifest=tokenizer_manifest,
        event_sink=event_sink,
        cancel_requested=cancel_requested,
    )
    package_manifest["artifact_integrity"] = build_package_artifact_integrity(
        package_dir
    )
    package_manifest_path = package_dir / "vulkan_resident_package.json"
    write_json(package_manifest_path, package_manifest)
    emit_compile_event(
        event_sink, "PackageValidationStarted", package=str(package_manifest_path)
    )
    validate_compiled_package(package_dir, package_manifest)
    check_compile_cancelled(cancel_requested)

    return CompiledModelReport(
        model_dir=model_dir,
        transpiled_dir=transpiled_dir,
        lowered_dir=lowered_dir,
        package_dir=package_dir,
        package_manifest=package_manifest_path,
        model_type=structure.model_type or "unknown",
        circuit_count=lowered["index"]["summary"]["circuit_count"],
        shader_count=len(list((package_dir / "shaders").glob("*.spv"))),
    )


def build_vulkan_resident_package_manifest(
    *,
    model_graph: Json,
    tensor_index: Json,
    lowered_index: Json,
    lowered_dir: Path,
    package_dir: Path,
    package_id: str,
    shader_source_dir: Path,
    tokenizer_manifest: Json,
    event_sink: Callable[[Json], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    dimensions = model_graph["dimensions"]
    hidden_size = int(dimensions["hidden_size"])
    vocab_size = int(dimensions["vocab_size"])
    norm_eps = float(model_graph["numerics"]["rms_norm_eps"])
    norm_weight_offset = float(model_graph["numerics"]["rms_norm_weight_offset"])
    embedding_scale = float(model_graph["numerics"]["embedding_scale"])
    logits_scale = float(model_graph["numerics"]["logits_scale"])
    configured_context = dimensions.get("max_position_embeddings")
    max_context_activations = int(
        configured_context
        if configured_context is not None
        else dimensions.get("attention_window_size") or 4096
    )
    dtype = "BF16"
    dtype_bytes = dtype_byte_count(dtype)
    frame_bytes = hidden_size * dtype_bytes
    logits_bytes = vocab_size * dtype_byte_count("F32")
    sampling = model_graph["sampling"]
    sampler_method = str(sampling["method"])
    if sampler_method == "greedy":
        sampler_id = "greedy_sampler"
        sampler_shader_file = f"greedy_sampler_f32_{vocab_size}.comp"
        sampler_temperature = 1.0
        sampler_top_k = 1
        sampler_top_p = 1.0
        sampler_scratch_byte_capacity = 0
        sampler_kernels = [
            {
                "role": "sample_logits",
                "shader_path": compiled_shader_path(f"shaders/{sampler_shader_file}"),
                "local_size_x": 1024,
                "workgroup_count_x": 1,
            }
        ]
    elif sampler_method == "temperature_top_k_top_p":
        sampler_id = "temperature_top_k_top_p_sampler"
        sampler_temperature = float(sampling["temperature"])
        sampler_top_k = int(sampling["top_k"])
        sampler_top_p = float(sampling["top_p"])
        sampler_partition_count = 128
        sampler_candidate_local_size_x = 256
        sampler_merge_local_size_x = 256
        sampler_scratch_byte_capacity = sampler_partition_count * sampler_top_k * 8
        sampler_candidate_shader_file = (
            f"temperature_top_k_candidates_f32_{vocab_size}"
            f"_k{sampler_top_k}_g{sampler_partition_count}"
            f"_l{sampler_candidate_local_size_x}.comp"
        )
        sampler_shader_file = (
            f"temperature_top_k_top_p_sampler_f32"
            f"_t{shader_float_token(sampler_temperature)}"
            f"_k{sampler_top_k}_p{shader_float_token(sampler_top_p)}"
            f"_g{sampler_partition_count}_l{sampler_merge_local_size_x}.comp"
        )
        sampler_kernels = [
            {
                "role": "partition_top_k",
                "shader_path": compiled_shader_path(
                    f"shaders/{sampler_candidate_shader_file}"
                ),
                "local_size_x": sampler_candidate_local_size_x,
                "workgroup_count_x": sampler_partition_count,
            },
            {
                "role": "sample_candidates",
                "shader_path": compiled_shader_path(f"shaders/{sampler_shader_file}"),
                "local_size_x": sampler_merge_local_size_x,
                "workgroup_count_x": 1,
            },
        ]
    else:
        raise ModelCompileError(f"unsupported sampling method {sampler_method!r}")

    embed_tensor = model_graph["graph"]["input_transducer"]["params"]["weight"][
        "tensor"
    ]
    output_components = model_graph["graph"]["output_transducer"]["components"]
    norm_tensor = next(
        component["params"]["weight"]["tensor"]
        for component in output_components
        if component["type"] == "rms_norm"
    )
    projection_tensor = next(
        component["params"]["weight"]["tensor"]
        for component in output_components
        if component["type"] == "linear_projection"
    )
    for role, tensor in (
        ("input embedding", embed_tensor),
        ("output normalization", norm_tensor),
        ("output projection", projection_tensor),
    ):
        actual_dtype = tensor_dtype(tensor_index, tensor)
        if actual_dtype != dtype:
            raise ModelCompileError(
                f"{role} tensor {tensor!r} has dtype {actual_dtype}; expected {dtype}"
            )
    embedding_layout = tensor_layout(tensor_index, embed_tensor)
    projection_layout = tensor_layout(tensor_index, projection_tensor)
    embedding_shader_file = (
        f"embedding_lookup_paired_bf16_{vocab_size}x{hidden_size}"
        f"_scale{shader_float_token(embedding_scale)}.comp"
        if embedding_layout == VULKAN_BF16_ROW_PAIR_LAYOUT
        else f"embedding_lookup_bf16_{vocab_size}x{hidden_size}"
        f"_scale{shader_float_token(embedding_scale)}.comp"
    )
    output_scale = 1.0 / logits_scale
    projection_shader_file = (
        f"tied_output_projection_paired_bf16_{vocab_size}x{hidden_size}"
        f"_scale{shader_float_token(output_scale)}_to_f32.comp"
        if projection_layout == VULKAN_BF16_ROW_PAIR_LAYOUT
        else f"tied_output_projection_bf16_{vocab_size}x{hidden_size}"
        f"_scale{shader_float_token(output_scale)}_to_f32.comp"
    )
    projection_batch_lane_tile_width = 4
    projection_batch_shader_file = (
        f"tied_output_projection_batch{projection_batch_lane_tile_width}_paired_bf16_"
        f"{vocab_size}x{hidden_size}_scale{shader_float_token(output_scale)}_to_f32.comp"
        if projection_layout == VULKAN_BF16_ROW_PAIR_LAYOUT
        else f"tied_output_projection_batch{projection_batch_lane_tile_width}_bf16_"
        f"{vocab_size}x{hidden_size}_scale{shader_float_token(output_scale)}_to_f32.comp"
    )
    norm_shader_file = rms_norm_shader_file(hidden_size, norm_eps, norm_weight_offset)

    source_circuits = {}
    compiled_circuits = {}
    for circuit_ref in all_lowered_circuit_refs(lowered_index):
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        source_circuits[circuit_ref["id"]] = circuit
        compiled_circuits[circuit_ref["id"]] = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda node, circuit=circuit: (
                can_fuse_bf16_linear_split(circuit, node, tensor_index)
            ),
            can_fuse_parallel_linears=lambda nodes, circuit=circuit: (
                can_fuse_bf16_parallel_linears(circuit, nodes, tensor_index)
            ),
            can_fuse_parallel_linear_silu_multiply=lambda projection, activation, circuit=circuit: (
                can_fuse_parallel_linear_silu_multiply(
                    circuit, projection, activation, tensor_index
                )
            ),
            can_fuse_parallel_head_norm_rope=lambda branches, circuit=circuit: (
                can_fuse_bf16_parallel_head_norm_rope(circuit, branches, tensor_index)
            ),
            can_fuse_multiply_rolling_depthwise=lambda multiply, rolling, depthwise, circuit=circuit: (
                can_fuse_bf16_multiply_rolling_depthwise(
                    circuit, multiply, rolling, depthwise, tensor_index
                )
            ),
            can_fuse_recurrent_output_gate=lambda recurrent, gate, circuit=circuit: (
                can_fuse_bf16_recurrent_output_gate(
                    circuit, recurrent, gate, tensor_index
                )
            ),
            can_fuse_linear_split_recurrent=lambda projection, recurrent, circuit=circuit: (
                can_fuse_bf16_linear_split_recurrent(
                    circuit, projection, recurrent, tensor_index
                )
            ),
            can_fuse_append_attention=lambda append, attention, circuit=circuit: (
                can_fuse_bf16_append_attention(circuit, append, attention, tensor_index)
            ),
        )
    behavioral_validation = build_behavioral_validation(
        model_graph=model_graph,
        tensor_index=tensor_index,
        lowered_index=lowered_index,
        source_circuits=source_circuits,
        candidate_circuits=compiled_circuits,
    )
    write_json(package_dir / "behavioral_validation.json", behavioral_validation)
    pedal_executions = pedal_execution_specs(
        lowered_index=lowered_index,
        compiled_circuits=compiled_circuits,
        tensor_index=tensor_index,
        dimensions=dimensions,
    )
    speculative_decoders = speculative_decoder_specs(
        lowered_index=lowered_index,
        lowered_dir=lowered_dir,
        compiled_circuits=compiled_circuits,
        tensor_index=tensor_index,
        dimensions=dimensions,
        projection_shader_file=projection_shader_file,
        norm_shader_file=norm_shader_file,
        frame_bytes=frame_bytes,
        logits_bytes=logits_bytes,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    all_pedal_executions = [
        *pedal_executions,
        *(
            execution
            for decoder in speculative_decoders
            for execution in decoder["pedal_executions"]
        ),
    ]
    shader_files = required_shader_files(
        all_pedal_executions,
        embedding_shader_file=embedding_shader_file,
        projection_shader_file=projection_shader_file,
        projection_batch_shader_file=projection_batch_shader_file,
        norm_shader_file=norm_shader_file,
        sampler_shader_files={
            kernel["shader_path"].removeprefix("shaders/").removesuffix(".spv")
            + ".comp"
            for kernel in sampler_kernels
        }
        | {
            decoder["output_transducer"]["norm_shader_path"]
            .removeprefix("shaders/")
            .removesuffix(".spv")
            + ".comp"
            for decoder in speculative_decoders
        }
        | {
            decoder["output_transducer"]["projection_shader_path"]
            .removeprefix("shaders/")
            .removesuffix(".spv")
            + ".comp"
            for decoder in speculative_decoders
        },
    )
    copy_shader_templates(
        shader_source_dir,
        package_dir / "shaders",
        shader_files,
    )
    required_device_extensions = required_vulkan_device_extensions(
        package_dir / "shaders", shader_files
    )
    compile_shader_artifacts(
        package_dir / "shaders",
        progress=lambda current, total, shader_name: emit_compile_event(
            event_sink,
            "ShaderCompilationStarted",
            current=current,
            total=total,
            shader_name=shader_name,
        ),
        cancel_requested=cancel_requested,
    )
    for execution in all_pedal_executions:
        for kernel in execution["kernels"]:
            kernel["shader_path"] = compiled_shader_path(kernel["shader_path"])
            if kernel.get("batch_shader_path") is not None:
                kernel["batch_shader_path"] = compiled_shader_path(
                    kernel["batch_shader_path"]
                )
    return {
        "schema": PACKAGE_SCHEMA,
        "package_id": package_id,
        "circuit_graph": package_circuit_graph(
            lowered_index, lowered_dir, compiled_circuits
        ),
        "tensor_index_path": "tensors.json",
        "behavioral_validation_path": "behavioral_validation.json",
        "config_path": CONFIG_PACKAGE_FILE,
        "tokenizer": tokenizer_manifest,
        "activation_element_bytes": dtype_bytes,
        "max_context_activations": max_context_activations,
        "required_vulkan_device_extensions": required_device_extensions,
        "pedal_batch_lane_tile_width": PEDAL_BATCH_LANE_TILE_WIDTH,
        "input_transducer": {
            "spec": {
                "transducer_id": "input_transducer.token_embedding",
                "parameter_tensor": embed_tensor,
                "parameter_dtype": dtype,
                "parameter_shape": tensor_shape(tensor_index, embed_tensor),
                "parameter_byte_capacity": tensor_byte_count(
                    tensor_index, embed_tensor
                ),
                "output_signal_id": "input_frame",
                "output_frame_byte_capacity": frame_bytes,
                "output_frame_word_count": frame_bytes // 4,
                "local_size_x": 256,
            },
            "shader_path": compiled_shader_path(f"shaders/{embedding_shader_file}"),
        },
        "output_transducer": {
            "spec": {
                "transducer_id": "output_transducer",
                "input_signal_id": "output_frame",
                "node_ids": [
                    "output_norm",
                    "output_projection",
                ],
                "norm_parameter_tensor": norm_tensor,
                "norm_parameter_dtype": dtype,
                "norm_parameter_shape": tensor_shape(tensor_index, norm_tensor),
                "norm_parameter_byte_capacity": tensor_byte_count(
                    tensor_index, norm_tensor
                ),
                "projection_parameter_tensor": projection_tensor,
                "projection_parameter_dtype": dtype,
                "projection_parameter_shape": tensor_shape(
                    tensor_index, projection_tensor
                ),
                "projection_parameter_byte_capacity": tensor_byte_count(
                    tensor_index, projection_tensor
                ),
                "input_frame_byte_capacity": frame_bytes,
                "normalized_frame_byte_capacity": frame_bytes,
                "logits_byte_capacity": logits_bytes,
                # The projection shader collaboratively computes two vocabulary
                # rows per workgroup. Dispatch geometry is part of the compiled
                # pedal, not something the runtime should infer from a model.
                "projection_workgroup_count_x": (vocab_size + 1) // 2,
                "norm_local_size_x": 64,
                "projection_local_size_x": 64,
            },
            "embedding_norm_shader_path": compiled_shader_path(
                f"shaders/{norm_shader_file}"
            ),
            "projection_shader_path": compiled_shader_path(
                f"shaders/{projection_shader_file}"
            ),
            "projection_batch_shader_path": compiled_shader_path(
                f"shaders/{projection_batch_shader_file}"
            ),
            "projection_batch_lane_tile_width": projection_batch_lane_tile_width,
        },
        "sampler": {
            "spec": {
                "sampler_id": sampler_id,
                "method": sampler_method,
                "temperature": sampler_temperature,
                "top_k": sampler_top_k,
                "top_p": sampler_top_p,
                "logits_byte_capacity": logits_bytes,
                "output_byte_capacity": 16,
                "scratch_byte_capacity": sampler_scratch_byte_capacity,
            },
            "kernels": sampler_kernels,
        },
        "pedal_executions": pedal_executions,
        "speculative_decoders": speculative_decoders,
    }


def package_circuit_graph(
    lowered_index: Json,
    lowered_dir: Path,
    compiled_circuits: dict[str, Json],
) -> Json:
    graph = lowered_index["graph"]
    pedals = []
    for circuit_ref in graph["circuits"]:
        pedals.append(
            {
                "pedal_id": circuit_ref["id"],
                "operator_type": circuit_ref["operator_type"],
                "runtime_role": circuit_ref["runtime_role"],
                "implementation": circuit_ref["implementation"],
                "behavioral_role": circuit_ref["behavioral_role"],
                "circuit": deepcopy(compiled_circuits[circuit_ref["id"]]),
                "params": read_json(lowered_dir / circuit_ref["params"]),
                "state": read_json(lowered_dir / circuit_ref["state"]),
            }
        )

    return {
        "wiring": graph["wiring"],
        "cables": deepcopy(graph["cables"]),
        "boundary": deepcopy(graph["boundary"]),
        "architecture": deepcopy(lowered_index.get("architecture", {})),
        "dimensions": deepcopy(lowered_index.get("dimensions", {})),
        "input_transducer": deepcopy(graph.get("input_transducer", {})),
        "output_transducer": deepcopy(graph.get("output_transducer", {})),
        "pedals": pedals,
    }


def pedal_execution_specs(
    *,
    lowered_index: Json,
    compiled_circuits: dict[str, Json],
    tensor_index: Json,
    dimensions: Json,
) -> list[Json]:
    executions: list[Json] = []

    for circuit_ref in lowered_index["graph"]["circuits"]:
        if circuit_ref["runtime_role"] != "signal_processor":
            continue
        circuit = compiled_circuits[circuit_ref["id"]]
        kernels = []
        for index, node in enumerate(circuit["nodes"]):
            shader_file = shader_file_for_node(
                circuit,
                node,
                tensor_index,
                dimensions,
            )
            kernels.append(
                pedal_kernel_spec(
                    execution_index=index,
                    node=node,
                    shader_file=shader_file,
                    local_size_x=local_size_x_for_node(node),
                    workgroup_count_x=workgroup_count_x_for_node(
                        circuit, node, tensor_index
                    ),
                )
            )
        executions.append(
            {
                "pedal_id": circuit_ref["id"],
                "operator_type": circuit_ref["operator_type"],
                "implementation": circuit_ref["implementation"],
                "kernels": kernels,
            }
        )

    return executions


def speculative_decoder_specs(
    *,
    lowered_index: Json,
    lowered_dir: Path,
    compiled_circuits: dict[str, Json],
    tensor_index: Json,
    dimensions: Json,
    projection_shader_file: str,
    norm_shader_file: str,
    frame_bytes: int,
    logits_bytes: int,
    vocab_size: int,
    hidden_size: int,
) -> list[Json]:
    decoders = []
    for draft in lowered_index.get("draft_pedalboards", []):
        circuit_refs = draft["circuits"]
        input_ref = next(
            ref for ref in circuit_refs if ref["runtime_role"] == "draft_input_adapter"
        )
        output_ref = next(
            ref
            for ref in circuit_refs
            if ref["runtime_role"] == "draft_output_transducer"
        )
        executable_refs = [
            ref
            for ref in circuit_refs
            if ref["runtime_role"] in {"draft_input_adapter", "draft_processor"}
        ]
        executions = [
            pedal_execution_spec(
                circuit_ref=ref,
                circuit=compiled_circuits[ref["id"]],
                tensor_index=tensor_index,
                dimensions=dimensions,
            )
            for ref in executable_refs
        ]
        output_circuit = compiled_circuits[output_ref["id"]]
        output_refs = output_circuit["parameters"]["refs"]
        norm_tensor = output_refs["norm"]["tensor"]
        projection_tensor = output_refs["projection"]["tensor"]
        decoders.append(
            {
                "id": draft["id"],
                "type": draft["type"],
                "source_prefix": draft["source_prefix"],
                "circuit_graph": package_auxiliary_circuit_graph(
                    draft, lowered_dir, compiled_circuits
                ),
                "input_adapter": {
                    "pedal_id": input_ref["id"],
                    "token_embedding_signal_id": "token_embedding",
                    "target_hidden_signal_id": "target_hidden",
                    "output_signal_id": "output_frame",
                    "input_frame_byte_capacity": frame_bytes,
                    "target_hidden_byte_capacity": frame_bytes,
                    "output_frame_byte_capacity": frame_bytes,
                },
                "output_transducer": {
                    "pedal_id": output_ref["id"],
                    "input_signal_id": "output_frame",
                    "hidden_signal_id": "output_hidden",
                    "logits_signal_id": "output_logits",
                    "norm_parameter_tensor": norm_tensor,
                    "norm_parameter_dtype": tensor_dtype(tensor_index, norm_tensor),
                    "norm_parameter_shape": tensor_shape(tensor_index, norm_tensor),
                    "norm_parameter_byte_capacity": tensor_byte_count(
                        tensor_index, norm_tensor
                    ),
                    "projection_parameter_tensor": projection_tensor,
                    "projection_parameter_dtype": tensor_dtype(
                        tensor_index, projection_tensor
                    ),
                    "projection_parameter_shape": tensor_shape(
                        tensor_index, projection_tensor
                    ),
                    "projection_parameter_byte_capacity": tensor_byte_count(
                        tensor_index, projection_tensor
                    ),
                    "input_frame_byte_capacity": frame_bytes,
                    "output_hidden_byte_capacity": frame_bytes,
                    "logits_byte_capacity": logits_bytes,
                    "vocabulary_size": vocab_size,
                    "hidden_size": hidden_size,
                    "projection_workgroup_count_x": (vocab_size + 1) // 2,
                    "norm_local_size_x": 64,
                    "projection_local_size_x": 64,
                    "norm_shader_path": compiled_shader_path(
                        f"shaders/{norm_shader_file}"
                    ),
                    "projection_shader_path": compiled_shader_path(
                        f"shaders/{projection_shader_file}"
                    ),
                },
                "pedal_executions": executions,
                "state_contract": deepcopy(draft["state_contract"]),
                "verification_contract": {
                    "target_execution": "multi_token",
                    "state_updates": "transactional",
                    "acceptance": "longest_matching_prefix",
                },
            }
        )
    return decoders


def pedal_execution_spec(
    *,
    circuit_ref: Json,
    circuit: Json,
    tensor_index: Json,
    dimensions: Json,
) -> Json:
    kernels = []
    for index, node in enumerate(circuit["nodes"]):
        shader_file = shader_file_for_node(circuit, node, tensor_index, dimensions)
        kernels.append(
            pedal_kernel_spec(
                execution_index=index,
                node=node,
                shader_file=shader_file,
                local_size_x=local_size_x_for_node(node),
                workgroup_count_x=workgroup_count_x_for_node(
                    circuit, node, tensor_index
                ),
            )
        )
    return {
        "pedal_id": circuit_ref["id"],
        "operator_type": circuit_ref["operator_type"],
        "implementation": circuit_ref["implementation"],
        "kernels": kernels,
    }


def pedal_kernel_spec(
    *,
    execution_index: int,
    node: Json,
    shader_file: str,
    local_size_x: int,
    workgroup_count_x: int,
) -> Json:
    batch_shader_file = weight_shared_batch_shader_file(shader_file)
    spec = {
        "execution_index": execution_index,
        "node_id": node["id"],
        "op": node["op"],
        "shader_path": f"shaders/{shader_file}",
        "local_size_x": local_size_x,
        "workgroup_count_x": workgroup_count_x,
        "batch_mode": "weight_shared" if batch_shader_file else "serial_lanes",
    }
    if batch_shader_file is not None:
        spec["batch_shader_path"] = f"shaders/{batch_shader_file}"
    return spec


def weight_shared_batch_shader_file(shader_file: str) -> str | None:
    tile = PEDAL_BATCH_LANE_TILE_WIDTH
    rms_norm = re.fullmatch(
        r"rms_norm_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if rms_norm is not None and int(rms_norm.group(1)) % 2 == 0:
        return shader_file.replace("rms_norm_bf16_", f"rms_norm_batch{tile}_bf16_", 1)
    fp8 = re.fullmatch(
        r"(linear|linear_residual)_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fp8 is not None:
        operation, block_rows, block_columns, input_size, _ = fp8.groups()
        if int(block_rows) % 2 == 0 and int(block_columns) % 4 == 0 and int(input_size) % 4 == 0:
            return shader_file.replace(
                f"{operation}_fp8_e4m3_",
                f"{operation}_batch{tile}_fp8_e4m3_",
                1,
            )
    bf16 = re.fullmatch(
        r"(linear|linear_residual)_(paired_)?bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if bf16 is not None:
        operation, paired, input_size, output_size = bf16.groups()
        if int(input_size) % 2 == 0 and int(output_size) % 2 == 0:
            layout = "paired" if paired else "row_major"
            return (
                f"{operation}_batch{tile}_{layout}_bf16_"
                f"{input_size}x{output_size}.comp"
            )
    parallel = re.fullmatch(
        r"parallel_linear_([23])way_(paired|row_major)_bf16_(\d+)x.+\.comp",
        shader_file,
    )
    if parallel is not None and int(parallel.group(3)) % 2 == 0:
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_ffn is not None:
        block_rows, block_columns, input_size, _ = map(int, fused_ffn.groups())
        if block_rows % 2 == 0 and block_columns % 4 == 0 and input_size % 4 == 0:
            return shader_file.replace(
                "parallel_linear_silu_multiply_fp8_e4m3_",
                f"parallel_linear_silu_multiply_batch{tile}_fp8_e4m3_",
                1,
            )
    fused_bf16_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_(paired|row_major)_bf16_"
        r"(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_bf16_ffn is not None:
        layout, input_size, output_size = fused_bf16_ffn.groups()
        if int(input_size) % 2 == 0 and int(output_size) % 2 == 0:
            return (
                f"parallel_linear_silu_multiply_batch{tile}_{layout}_bf16_"
                f"{input_size}x{output_size}.comp"
            )
    return None


def package_auxiliary_circuit_graph(
    draft: Json,
    lowered_dir: Path,
    compiled_circuits: dict[str, Json],
) -> Json:
    pedals = []
    for circuit_ref in draft["circuits"]:
        pedals.append(
            {
                "pedal_id": circuit_ref["id"],
                "operator_type": circuit_ref["operator_type"],
                "runtime_role": circuit_ref["runtime_role"],
                "implementation": circuit_ref["implementation"],
                "behavioral_role": circuit_ref["behavioral_role"],
                "circuit": deepcopy(compiled_circuits[circuit_ref["id"]]),
                "params": read_json(lowered_dir / circuit_ref["params"]),
                "state": read_json(lowered_dir / circuit_ref["state"]),
            }
        )
    return {
        "wiring": draft["wiring"],
        "cables": deepcopy(draft["cables"]),
        "boundary": deepcopy(draft["boundary"]),
        "pedals": pedals,
    }


def shader_file_for_node(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    dimensions: Json,
) -> str:
    hidden_size = int(dimensions["hidden_size"])
    intermediate_size = int(dimensions["intermediate_size"])
    op = node["op"]

    if op == "rms_norm":
        return rms_norm_shader_file(
            hidden_size,
            float(node["attrs"]["eps"]),
            float(node["attrs"]["weight_offset"]),
        )
    if op == "linear":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            if quantization_format == "auto_gptq":
                group_size = packed_int4_linear_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "gptq"
                has_bias = len(node.get("params", [])) == 4
            elif quantization_format == "compressed_tensors_pack_quantized":
                group_size = compressed_tensors_int4_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "ct"
                has_bias = len(node.get("params", [])) == 3
            else:
                raise ModelCompileError(
                    f"linear node {node['id']!r} has unsupported packed format "
                    f"{quantization_format!r}"
                )
            prefix = "linear_bias" if has_bias else "linear"
            return (
                f"{prefix}_int4_{format_token}_g{group_size}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_block_shape_for_node(
                circuit, node, tensor_index
            )
            has_bias = len(node.get("params", [])) == 3
            prefix = "linear_bias" if has_bias else "linear"
            return (
                f"{prefix}_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"linear node {node['id']!r} has unsupported weight dtype "
                f"{parameter_dtype}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        has_bias = len(node.get("params", [])) == 2
        prefix = "linear"
        if has_bias:
            prefix += "_bias"
        if layout == VULKAN_BF16_ROW_PAIR_LAYOUT:
            prefix += "_paired"
        prefix += "_bf16"
        return f"{prefix}_{in_features}x{out_features}.comp"
    if op in {"parallel_linear_2way", "parallel_linear_3way"}:
        expected_branch_count = 2 if op == "parallel_linear_2way" else 3
        branch_count = int(node["attrs"]["branch_count"])
        if (
            branch_count != expected_branch_count
            or branch_count != len(node["params"])
            or branch_count != len(node["outputs"])
        ):
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has inconsistent branch metadata"
            )
        shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        ]
        input_widths = {int(shape[1]) for shape in shapes if len(shape) == 2}
        dtypes = {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        if (
            len(shapes) != branch_count
            or any(len(shape) != 2 for shape in shapes)
            or len(input_widths) != 1
            or dtypes != {"BF16"}
        ):
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has incompatible shapes {shapes}"
            )
        output_widths = [int(shape[0]) for shape in shapes]
        layouts = {
            parameter_layout_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        supported_layouts = {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
        if len(layouts) != 1 or not layouts <= supported_layouts:
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has unsupported layouts "
                f"{sorted(layouts)}"
            )
        layout_token = (
            "paired" if layouts == {VULKAN_BF16_ROW_PAIR_LAYOUT} else "row_major"
        )
        input_width = input_widths.pop()
        return (
            f"parallel_linear_{branch_count}way_{layout_token}_bf16_{input_width}x"
            + "_".join(map(str, output_widths))
            + ".comp"
        )
    if op == "parallel_linear_silu_multiply":
        params = node.get("params", [])
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or int(node.get("attrs", {}).get("branch_count", 0)) != 2
        ):
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid bindings"
            )
        if len(params) == 2:
            weight_ids = params
            shapes = [
                parameter_shape_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            ]
            dtypes = {
                parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            layouts = {
                parameter_layout_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            if (
                len(shapes) != 2
                or shapes[0] != shapes[1]
                or len(shapes[0]) != 2
                or dtypes != {"BF16"}
                or len(layouts) != 1
                or not layouts <= {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
            ):
                raise ModelCompileError(
                    f"fused FFN projection node {node['id']!r} has incompatible "
                    f"parameters {shapes}"
                )
            shader_format = (
                "paired" if layouts == {VULKAN_BF16_ROW_PAIR_LAYOUT} else "row_major"
            )
            block_shape = None
        elif len(params) == 4:
            weight_ids = [params[0], params[2]]
            shapes = [
                parameter_shape_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            ]
            branch_params = [params[:2], params[2:]]
            block_shapes = {
                fp8_block_shape_for_node(
                    circuit,
                    {
                        "id": f"{node['id']}__branch_{index}",
                        "params": parameter_ids,
                    },
                    tensor_index,
                )
                for index, parameter_ids in enumerate(branch_params)
            }
            if (
                len(shapes) != 2
                or shapes[0] != shapes[1]
                or len(shapes[0]) != 2
                or len(block_shapes) != 1
                or any(
                    parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                    != "F8_E4M3"
                    or parameter_layout_for_id(circuit, parameter_id, tensor_index)
                    != ROW_MAJOR_LAYOUT
                    for parameter_id in weight_ids
                )
            ):
                raise ModelCompileError(
                    f"fused FFN projection node {node['id']!r} has incompatible "
                    f"parameters {shapes}"
                )
            shader_format = "fp8_e4m3"
            block_shape = block_shapes.pop()
        else:
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid parameter count "
                f"{len(params)}"
            )
        output_width, input_width = map(int, shapes[0])
        if (
            input_width <= 0
            or input_width % 2
            or output_width <= 0
            or output_width % 2
            or int(node["attrs"].get("element_count", 0)) != output_width
            or node["attrs"].get("intermediate_rounding") != "BF16"
        ):
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid geometry"
            )
        if block_shape is not None:
            block_rows, block_columns = block_shape
            return (
                "parallel_linear_silu_multiply_fp8_e4m3_"
                f"b{block_rows}x{block_columns}_{input_width}x{output_width}.comp"
            )
        return (
            f"parallel_linear_silu_multiply_{shader_format}_bf16_"
            f"{input_width}x{output_width}.comp"
        )
    if op == "linear_split_3way":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = map(int, parameter_shape)
        if parameter_dtype_for_node(circuit, node, tensor_index) != "BF16":
            raise ModelCompileError(
                f"linear-split node {node['id']!r} requires BF16 weights"
            )
        part_widths = [int(width) for width in node["attrs"]["part_widths"]]
        if (
            len(part_widths) != 3
            or any(width <= 0 or width % 2 for width in part_widths)
            or sum(part_widths) != out_features
        ):
            raise ModelCompileError(
                f"linear-split node {node['id']!r} cannot partition {out_features} "
                f"outputs into {part_widths}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        layout_token = (
            "paired" if layout == VULKAN_BF16_ROW_PAIR_LAYOUT else "row_major"
        )
        return (
            f"linear_split_3way_{layout_token}_bf16_{in_features}x"
            + "_".join(map(str, part_widths))
            + ".comp"
        )
    if op == "linear_residual":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            if quantization_format == "auto_gptq":
                group_size = packed_int4_linear_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "gptq"
            elif quantization_format == "compressed_tensors_pack_quantized":
                group_size = compressed_tensors_int4_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "ct"
            else:
                raise ModelCompileError(
                    f"linear-residual node {node['id']!r} has unsupported packed "
                    f"format {quantization_format!r}"
                )
            return (
                f"linear_residual_int4_{format_token}_g{group_size}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_block_shape_for_node(
                circuit, node, tensor_index
            )
            return (
                f"linear_residual_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"linear-residual node {node['id']!r} has unsupported weight dtype "
                f"{parameter_dtype}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        prefix = (
            "linear_residual_paired_bf16"
            if layout == VULKAN_BF16_ROW_PAIR_LAYOUT
            else "linear_residual_bf16"
        )
        return f"{prefix}_{in_features}x{out_features}.comp"
    if op == "split":
        if node["attrs"].get("layout") == "per_head_interleaved":
            return (
                f"split_bf16_2x{node['attrs']['blocks']}x{node['attrs']['block_part_width']}"
                "_head_interleaved.comp"
            )
        if node["attrs"].get("part_widths") is not None:
            part_widths = [int(width) for width in node["attrs"]["part_widths"]]
            if len(part_widths) != 3:
                raise ModelCompileError(
                    f"split node {node['id']!r} has unsupported unequal part widths {part_widths}"
                )
            return "split_bf16_3x" + "_".join(map(str, part_widths)) + ".comp"
        part_width = int(node["attrs"]["part_width"])
        return f"split_bf16_{len(node['outputs'])}x{part_width}.comp"
    if op == "concatenate":
        part_widths = [int(width) for width in node["attrs"]["part_widths"]]
        if (
            node["attrs"].get("axis") != "channel"
            or len(node.get("inputs", [])) != len(part_widths)
            or len(node.get("outputs", [])) != 1
            or any(width <= 0 or width % 2 for width in part_widths)
        ):
            raise ModelCompileError(
                f"concatenate node {node['id']!r} has unsupported geometry"
            )
        return "concatenate_bf16_" + "_".join(map(str, part_widths)) + ".comp"
    if op == "multiply":
        element_count = int(
            node.get("attrs", {}).get(
                "element_count",
                intermediate_size if node["id"] == "ffn_gate_multiply" else hidden_size,
            )
        )
        return f"multiply_bf16_{element_count}.comp"
    if op == "scalar_multiply":
        return f"scalar_multiply_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "rolling_state_update":
        temporal_memory = state_port(circuit, "temporal_memory")
        frames, state_hidden = temporal_memory["shape"]
        return f"rolling_state_update_bf16_{frames}x{state_hidden}.comp"
    if op == "depthwise_conv1d":
        temporal_memory = state_port(circuit, "temporal_memory")
        frames, state_hidden = temporal_memory["shape"]
        return f"depthwise_conv1d_bf16_{frames}x{state_hidden}.comp"
    if op in {"multiply_rolling_depthwise", "multiply_rolling_depthwise_gate"}:
        expected_input_count = 4 if op.endswith("_gate") else 3
        if (
            len(node.get("inputs", [])) != expected_input_count
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 1
            or len(node.get("state_reads", [])) != 1
            or node.get("state_reads") != node.get("state_writes")
        ):
            raise ModelCompileError(
                f"fused recurrent convolution node {node['id']!r} has invalid bindings"
            )
        temporal_memory = state_port(circuit, node["state_reads"][0])
        frames, state_hidden = map(int, temporal_memory["shape"])
        kernel_shape = parameter_shape_for_node(circuit, node, tensor_index)
        supported_kernel_shapes = ([state_hidden, frames], [state_hidden, 1, frames])
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames < 2
            or state_hidden <= 0
            or state_hidden % 2
            or kernel_shape not in supported_kernel_shapes
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"fused recurrent convolution node {node['id']!r} has incompatible "
                f"state {temporal_memory.get('shape')} or kernel {kernel_shape}"
            )
        shader_prefix = (
            "multiply_rolling_depthwise_gate"
            if op.endswith("_gate")
            else "multiply_rolling_depthwise"
        )
        return f"{shader_prefix}_bf16_{frames}x{state_hidden}.comp"
    if op == "linear_split_recurrent_depthwise_gate":
        if (
            len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 2
            or len(node.get("state_reads", [])) != 1
            or node.get("state_reads") != node.get("state_writes")
        ):
            raise ModelCompileError(
                f"projected recurrent convolution node {node['id']!r} has invalid bindings"
            )
        temporal_memory = state_port(circuit, node["state_reads"][0])
        frames, hidden_size = map(int, temporal_memory["shape"])
        projection_shape = parameter_shape_for_id(
            circuit, node["params"][0], tensor_index
        )
        kernel_shape = parameter_shape_for_id(circuit, node["params"][1], tensor_index)
        part_widths = [
            int(width) for width in node["attrs"]["projection"]["part_widths"]
        ]
        input_gate_indices = [
            int(index) for index in node["attrs"]["input_gate_branch_indices"]
        ]
        output_gate_index = int(node["attrs"]["output_gate_branch_index"])
        projection_layout = parameter_layout_for_id(
            circuit, node["params"][0], tensor_index
        )
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames < 2
            or hidden_size <= 0
            or hidden_size % 2
            or len(projection_shape) != 2
            or projection_shape[0] != 3 * hidden_size
            or projection_shape[1] <= 0
            or projection_shape[1] % 2
            or part_widths != [hidden_size] * 3
            or sorted([*input_gate_indices, output_gate_index]) != [0, 1, 2]
            or kernel_shape not in ([hidden_size, frames], [hidden_size, 1, frames])
            or any(
                parameter_dtype_for_id(circuit, parameter_id, tensor_index) != "BF16"
                for parameter_id in node["params"]
            )
            or projection_layout not in {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
            or parameter_layout_for_id(circuit, node["params"][1], tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"projected recurrent convolution node {node['id']!r} has "
                f"incompatible projection {projection_shape}, state "
                f"{temporal_memory.get('shape')}, or kernel {kernel_shape}"
            )
        layout_token = (
            "paired"
            if projection_layout == VULKAN_BF16_ROW_PAIR_LAYOUT
            else "row_major"
        )
        return (
            f"linear_split_recurrent_depthwise_gate_{layout_token}_bf16_"
            f"{projection_shape[1]}x{hidden_size}_k{frames}"
            f"_ig{input_gate_indices[0]}_{input_gate_indices[1]}"
            f"_og{output_gate_index}.comp"
        )
    if op == "residual_add":
        return f"add_bf16_{hidden_size}.comp"
    if op == "scaled_residual_add":
        return (
            f"scaled_add_bf16_{hidden_size}"
            f"_scale{shader_float_token(float(node['attrs']['scale']))}.comp"
        )
    if op == "silu":
        return f"silu_bf16_{intermediate_size}.comp"
    if op == "gelu_tanh":
        return f"gelu_tanh_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "silu_multiply":
        return f"silu_multiply_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "sigmoid_multiply":
        return "sigmoid_multiply_bf16.comp"
    if op == "sigmoid_scalar_multiply":
        return f"sigmoid_scalar_multiply_bf16_{hidden_size}.comp"
    if op == "rms_norm_per_head":
        return (
            f"rms_norm_per_head_bf16_{node['attrs']['head_count']}x"
            f"{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}"
            f"_offset{shader_float_token(float(node['attrs']['weight_offset']))}.comp"
        )
    if op == "rms_norm_per_head_unscaled":
        return (
            f"rms_norm_per_head_unscaled_bf16_{node['attrs']['head_count']}"
            f"x{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}.comp"
        )
    if op == "parallel_head_norm_rope_2way":
        branches = node.get("attrs", {}).get("branches", [])
        if (
            len(branches) != 2
            or len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 2
            or len(node.get("params", [])) != 2
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} has invalid branch metadata"
            )
        norms = [branch.get("norm", {}) for branch in branches]
        ropes = [branch.get("rope", {}) for branch in branches]
        head_counts = [int(norm["head_count"]) for norm in norms]
        common_fields = {
            "head_width": {int(norm["head_width"]) for norm in norms}
            | {int(rope["head_width"]) for rope in ropes},
            "eps": {float(norm["eps"]) for norm in norms},
            "weight_offset": {float(norm["weight_offset"]) for norm in norms},
            "rotary_width": {int(rope["rotary_width"]) for rope in ropes},
            "theta": {float(rope["theta"]) for rope in ropes},
            "rope_type": {str(rope.get("rope_type", "default")) for rope in ropes},
            "interleaved": {bool(rope["interleaved"]) for rope in ropes},
        }
        if any(len(values) != 1 for values in common_fields.values()) or any(
            int(norm["head_count"]) != int(rope["head_count"])
            for norm, rope in zip(norms, ropes, strict=True)
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} mixes incompatible branch geometry"
            )
        parameter_dtypes = {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        parameter_shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        ]
        head_width = common_fields["head_width"].pop()
        if parameter_dtypes != {"BF16"} or any(
            list(map(int, shape)) != [head_width] for shape in parameter_shapes
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} has incompatible "
                f"normalization parameters {parameter_shapes}"
            )
        rope_type = common_fields["rope_type"].pop()
        interleaved = common_fields["interleaved"].pop()
        rope_layout = (
            "proportional"
            if rope_type == "proportional"
            else "interleaved"
            if interleaved
            else "half"
        )
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"parallel_head_norm_rope_2way_bf16_h{head_counts[0]}_{head_counts[1]}"
            f"_d{head_width}_r{common_fields['rotary_width'].pop()}"
            f"_eps{shader_float_token(common_fields['eps'].pop())}"
            f"_offset{shader_float_token(common_fields['weight_offset'].pop())}"
            f"_theta{shader_float_token(common_fields['theta'].pop())}_{rope_layout}"
            f"__sc{binding}.comp"
        )
    if op == "per_layer_embedding":
        attrs = node["attrs"]
        token_shape = parameter_shape_for_id(circuit, "token_embedding", tensor_index)
        vocab_size = int(token_shape[0])
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"per_layer_embedding_paired_bf16_v{vocab_size}_h{attrs['hidden_size']}"
            f"_p{attrs['per_layer_width']}_l{attrs['layer_index']}of{attrs['layer_count']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}"
            f"_tes{shader_float_token(float(attrs['token_embedding_scale']))}"
            f"_pes{shader_float_token(float(attrs['per_layer_embedding_scale']))}"
            f"_mps{shader_float_token(float(attrs['model_projection_scale']))}"
            f"_cs{shader_float_token(float(attrs['combination_scale']))}__sc{binding}.comp"
        )
    if op == "rotary_position_embedding":
        binding = stream_control_binding_for_node(circuit, node)
        rope_layout = (
            "proportional"
            if node["attrs"].get("rope_type") == "proportional"
            else "interleaved"
            if node["attrs"]["interleaved"]
            else "half"
        )
        return (
            f"rotary_bf16_{node['attrs']['head_count']}x"
            f"{node['attrs']['head_width']}"
            f"_r{node['attrs']['rotary_width']}"
            f"_theta{shader_float_token(float(node['attrs']['theta']))}_{rope_layout}"
            f"__sc{binding}.comp"
        )
    if op == "append_state_update":
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"append_kv_state_bf16_{node['attrs']['key_value_heads']}"
            f"x{node['attrs']['head_width']}__sc{binding}.comp"
        )
    if op == "scaled_dot_product_attention":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        name = (
            "gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        if attrs.get("attention_sinks"):
            name += "_sinks"
        return f"{name}__sc{binding}.comp"
    if op == "append_scaled_dot_product_attention":
        attrs = node["attrs"]["attention"]
        binding = stream_control_binding_for_node(circuit, node)
        name = (
            "append_gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        if attrs.get("attention_sinks"):
            name += "_sinks"
        return f"{name}__sc{binding}.comp"
    if op == "causal_conv1d_silu":
        return (
            f"causal_conv1d_silu_bf16_c{node['attrs']['channels']}"
            f"_k{node['attrs']['kernel_width']}.comp"
        )
    if op == "gated_delta_step":
        attrs = node["attrs"]
        dtype_tokens = {"F32": "f32", "BF16": "bf16"}
        parameter_tokens: dict[str, str] = {}
        for parameter_id in ("delta_a_log", "delta_dt_bias", "delta_norm"):
            actual_dtype = parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            if actual_dtype not in dtype_tokens:
                raise ModelCompileError(
                    f"gated-delta parameter {parameter_id} has dtype {actual_dtype}; "
                    "expected F32 or BF16"
                )
            parameter_tokens[parameter_id] = dtype_tokens[actual_dtype]
        return (
            f"gated_delta_step_k{attrs['key_heads']}x{attrs['key_head_width']}"
            f"_v{attrs['value_heads']}x{attrs['value_head_width']}"
            f"_a{parameter_tokens['delta_a_log']}"
            f"_dt{parameter_tokens['delta_dt_bias']}"
            f"_n{parameter_tokens['delta_norm']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}.comp"
        )
    if op == "rg_lru_step":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"rg_lru_step_bf16_h{attrs['width']}_b{attrs['heads']}x{attrs['block_width']}"
            f"_k{attrs['conv_kernel_width']}__sc{binding}.comp"
        )
    if op == "moe_topk":
        attrs = node["attrs"]
        return (
            f"moe_topk_bf16_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        )
    if op == "sparse_moe_experts":
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_moe_block_shape_for_node(
                circuit, node, tensor_index
            )
            return (
                f"sparse_moe_experts_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"h{attrs['hidden_size']}_i{attrs['intermediate_size']}_"
                f"e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
            )
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"sparse MoE node {node['id']!r} has unsupported expert dtype "
                f"{parameter_dtype}"
            )
        return (
            f"sparse_moe_experts_bf16_h{attrs['hidden_size']}_i{attrs['intermediate_size']}"
            f"_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        )
    if op == "moe_reduce":
        attrs = node["attrs"]
        return f"moe_reduce_bf16_h{attrs['hidden_size']}_e{attrs['num_experts']}.comp"

    raise ModelCompileError(
        f"no Vulkan shader selector for op {op!r} in node {node['id']!r}"
    )


def workgroup_count_x_for_node(circuit: Json, node: Json, tensor_index: Json) -> int:
    if node["op"] == "linear_split_recurrent_depthwise_gate":
        hidden_size = int(state_port(circuit, node["state_reads"][0])["shape"][1])
        return hidden_size // 2
    if node["op"] == "parallel_head_norm_rope_2way":
        return sum(
            int(branch["norm"]["head_count"]) for branch in node["attrs"]["branches"]
        )
    if node["op"] in {"parallel_linear_2way", "parallel_linear_3way"}:
        return sum(
            (int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0]) + 1)
            // 2
            for parameter_id in node["params"]
        )
    if node["op"] == "parallel_linear_silu_multiply":
        out_features, _ = parameter_shape_for_id(
            circuit, node["params"][0], tensor_index
        )
        return (int(out_features) + 1) // 2
    if node["op"] in {"linear", "linear_residual", "linear_split_3way"}:
        out_features, _ = parameter_shape_for_node(circuit, node, tensor_index)
        # One workgroup collaboratively computes and packs two BF16 output rows.
        return (int(out_features) + 1) // 2
    if node["op"] in {
        "scaled_dot_product_attention",
        "append_scaled_dot_product_attention",
    }:
        attrs = (
            node["attrs"]["attention"]
            if node["op"] == "append_scaled_dot_product_attention"
            else node["attrs"]
        )
        return int(attrs["query_heads"])
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_heads"])
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["heads"])
    if node["op"] == "sparse_moe_experts":
        return int(node["attrs"]["num_experts"])
    if node["op"] in {
        "rms_norm_per_head",
        "rms_norm_per_head_unscaled",
        "rotary_position_embedding",
    }:
        return int(node["attrs"]["head_count"])
    return 1


def local_size_x_for_node(node: Json) -> int:
    # The tiled attention kernel maps sixteen 64-wide head reductions onto one
    # workgroup. This execution geometry belongs to the compiled pedal package.
    if node["op"] in {
        "scaled_dot_product_attention",
        "append_scaled_dot_product_attention",
    }:
        attrs = (
            node["attrs"]["attention"]
            if node["op"] == "append_scaled_dot_product_attention"
            else node["attrs"]
        )
        return attention_workgroup_shape(int(attrs["head_width"]))[0]
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_head_width"])
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["block_width"])
    if node["op"] == "sparse_moe_experts":
        return 256
    return 64


def attention_workgroup_shape(head_width: int) -> tuple[int, int]:
    padded_head_width = ((head_width + 63) // 64) * 64
    physical_tile_tokens = 1024 // padded_head_width
    if physical_tile_tokens == 0:
        return 0, 0
    # Keep attention scratch below the 32 KiB Vulkan floor while amortizing the
    # four workgroup barriers over more KV tokens than the physical tile holds.
    shared_float_budget = (32 * 1024) // 4
    fixed_shared_floats = 2 * head_width + 4
    tile_shared_floats = head_width + ((head_width + 31) // 32) + 3
    max_token_batches = (shared_float_budget - fixed_shared_floats) // (
        physical_tile_tokens * tile_shared_floats
    )
    token_batches = max(1, min(7, max_token_batches))
    return (
        padded_head_width * physical_tile_tokens,
        physical_tile_tokens * token_batches,
    )


def required_shader_files(
    pedal_executions: list[Json],
    *,
    embedding_shader_file: str,
    projection_shader_file: str,
    projection_batch_shader_file: str,
    norm_shader_file: str,
    sampler_shader_files: set[str],
) -> set[str]:
    return {
        norm_shader_file,
        *sampler_shader_files,
        embedding_shader_file,
        projection_shader_file,
        projection_batch_shader_file,
        *(
            kernel["shader_path"].removeprefix("shaders/")
            for pedal in pedal_executions
            for kernel in pedal["kernels"]
        ),
        *(
            kernel["batch_shader_path"].removeprefix("shaders/")
            for pedal in pedal_executions
            for kernel in pedal["kernels"]
            if kernel.get("batch_shader_path") is not None
        ),
    }


def required_vulkan_device_extensions(
    shader_dir: Path, shader_files: set[str]
) -> list[str]:
    required_glsl_extensions = set()
    for shader_file in shader_files:
        source = (shader_dir / shader_file).read_text()
        required_glsl_extensions.update(
            re.findall(r"^\s*#extension\s+(\S+)\s*:\s*require\s*$", source, re.MULTILINE)
        )
    return sorted(
        {
            vulkan_extension
            for glsl_extension, vulkan_extension in (
                GLSL_VULKAN_DEVICE_EXTENSION_REQUIREMENTS.items()
            )
            if glsl_extension in required_glsl_extensions
        }
    )


def rms_norm_shader_file(hidden_size: int, eps: float, weight_offset: float) -> str:
    return (
        f"rms_norm_bf16_h{hidden_size}_eps{shader_float_token(eps)}"
        f"_offset{shader_float_token(weight_offset)}.comp"
    )


def shader_float_token(value: float) -> str:
    return format(value, ".9g")


def copy_shader_templates(
    source_dir: Path, dest_dir: Path, shader_files: set[str]
) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for shader_file in sorted(shader_files):
        destination = dest_dir / shader_file
        destination.write_text(render_shader_source(source_dir, shader_file))


def render_shader_source(source_dir: Path, shader_file: str) -> str:
    source = source_dir / shader_file
    if source.exists():
        return source.read_text()

    stream_control_variant = re.fullmatch(r"(.+)__sc(\d+)\.comp", shader_file)
    if stream_control_variant is not None:
        source_name, binding = stream_control_variant.groups()
        rendered = render_shader_source(source_dir, f"{source_name}.comp")
        rendered, replacement_count = re.subn(
            r"layout\(set = 0, binding = \d+\) readonly buffer StreamControl",
            f"layout(set = 0, binding = {binding}) readonly buffer StreamControl",
            rendered,
        )
        if replacement_count != 1:
            raise ModelCompileError(
                f"shader {shader_file} has {replacement_count} stream-control bindings; expected one"
            )
        return rendered

    parallel_linear = re.fullmatch(
        r"parallel_linear_(?:batch(\d+)_)?([23])way_(paired|row_major)_bf16_"
        r"(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel_linear is not None:
        batch_tile_width = (
            int(parallel_linear.group(1))
            if parallel_linear.group(1) is not None
            else None
        )
        branch_count = int(parallel_linear.group(2))
        weight_layout = parallel_linear.group(3)
        input_size = int(parallel_linear.group(4))
        output_widths = [
            int(width) for width in parallel_linear.groups()[4:] if width is not None
        ]
        if (
            len(output_widths) != branch_count
            or (batch_tile_width is not None and batch_tile_width <= 0)
            or input_size <= 0
            or input_size % 2
            or any(width <= 0 or width % 2 for width in output_widths)
        ):
            raise ModelCompileError(
                f"invalid parallel-linear shader shape {shader_file!r}"
            )
        labels = [chr(ord("A") + index) for index in range(branch_count)]
        output_bindings = "\n\n".join(
            f"layout(set = 0, binding = {index + 1}) buffer Output{label} {{\n"
            "    uint words[];\n"
            f"}} output_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_bindings = "\n\n".join(
            f"layout(set = 0, binding = {branch_count + index + 1}) "
            f"readonly buffer Weight{label} {{\n"
            "    uint words[];\n"
            f"}} weight_{label.lower()};"
            for index, label in enumerate(labels)
        )
        output_constants = "\n".join(
            f"const uint OUTPUT_{label}_WORDS = {width}u / 2u;"
            for label, width in zip(labels, output_widths, strict=True)
        )
        branch_lines = []
        consumed_words = []
        for index, label in enumerate(labels):
            prefix = "if" if index == 0 else "else if"
            threshold = " + ".join([*consumed_words, f"OUTPUT_{label}_WORDS"])
            offset = " + ".join(consumed_words) or "0u"
            branch_lines.append(
                f"    {prefix} (word_index < {threshold}) {{\n"
                f"        branch = {index}u;\n"
                f"        local_word_index = word_index - ({offset});\n"
                "    }"
            )
            consumed_words.append(f"OUTPUT_{label}_WORDS")
        weight_reads = "\n".join(
            f"    if (branch == {index}u) return weight_{label.lower()}.words[weight_index];"
            for index, label in enumerate(labels[:-1])
        )
        weight_reads += (
            "\n    return weight_" + labels[-1].lower() + ".words[weight_index];"
        )
        output_index = (
            "batch_index * OUTPUT_{label}_WORDS + local_word_index"
            if batch_tile_width is not None
            else "local_word_index"
        )
        output_writes = "\n".join(
            f"    if (branch == {index}u) {{ output_{label.lower()}.words["
            + output_index.format(label=label)
            + "] = packed; return; }"
            for index, label in enumerate(labels[:-1])
        )
        output_writes += (
            "\n    output_"
            + labels[-1].lower()
            + ".words["
            + output_index.format(label=labels[-1])
            + "] = packed;"
        )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_batch_bf16.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_bf16.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "OUTPUT_BINDINGS": output_bindings,
                "WEIGHT_BINDINGS": weight_bindings,
                "INPUT_SIZE": str(input_size),
                "OUTPUT_WORD_CONSTANTS": output_constants,
                "TOTAL_OUTPUT_WORDS": " + ".join(
                    f"OUTPUT_{label}_WORDS" for label in labels
                ),
                "PAIRED_WEIGHT_LAYOUT": (
                    "true" if weight_layout == "paired" else "false"
                ),
                "BRANCH_SELECTION": "\n".join(branch_lines),
                "WEIGHT_READS": weight_reads,
                "OUTPUT_WRITES": output_writes,
            },
        )

    batched_bf16_linear = re.fullmatch(
        r"(linear|linear_residual)_batch(\d+)_(paired|row_major)_bf16_"
        r"(\d+)x(\d+)\.comp",
        shader_file,
    )
    if batched_bf16_linear is not None:
        operation = batched_bf16_linear.group(1)
        batch_tile_width = int(batched_bf16_linear.group(2))
        weight_layout = batched_bf16_linear.group(3)
        input_size = int(batched_bf16_linear.group(4))
        output_size = int(batched_bf16_linear.group(5))
        if (
            batch_tile_width <= 0
            or input_size <= 0
            or input_size % 2
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid batched BF16 linear shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            f"{operation}_batch_bf16.comp.template",
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "PAIRED_WEIGHT_LAYOUT": (
                    "true" if weight_layout == "paired" else "false"
                ),
            },
        )

    fused_ffn_projection = re.fullmatch(
        r"parallel_linear_silu_multiply_(?:batch(\d+)_)?"
        r"(paired|row_major)_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_ffn_projection is not None:
        batch_tile_width = (
            int(fused_ffn_projection.group(1))
            if fused_ffn_projection.group(1) is not None
            else None
        )
        weight_layout = fused_ffn_projection.group(2)
        input_size = int(fused_ffn_projection.group(3))
        output_size = int(fused_ffn_projection.group(4))
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or input_size <= 0
            or input_size % 2
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid fused FFN projection shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_silu_multiply_batch_bf16.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_silu_multiply_bf16.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "PAIRED_WEIGHT_LAYOUT": (
                    "true" if weight_layout == "paired" else "false"
                ),
            },
        )

    concatenate = re.fullmatch(r"concatenate_bf16_(\d+)_(\d+)\.comp", shader_file)
    if concatenate is not None:
        width_a = int(concatenate.group(1))
        width_b = int(concatenate.group(2))
        if width_a <= 0 or width_a % 2 or width_b <= 0 or width_b % 2:
            raise ModelCompileError(f"invalid concatenate shader shape {shader_file!r}")
        return render_shader_template(
            source_dir,
            "concatenate_bf16.comp.template",
            {"WIDTH_A": str(width_a), "WIDTH_B": str(width_b)},
        )

    fused_fp8_ffn_projection = re.fullmatch(
        r"parallel_linear_silu_multiply_(?:batch(\d+)_)?fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_fp8_ffn_projection is not None:
        batch_tile_width = (
            int(fused_fp8_ffn_projection.group(1))
            if fused_fp8_ffn_projection.group(1) is not None
            else None
        )
        block_rows = int(fused_fp8_ffn_projection.group(2))
        block_columns = int(fused_fp8_ffn_projection.group(3))
        input_size = int(fused_fp8_ffn_projection.group(4))
        output_size = int(fused_fp8_ffn_projection.group(5))
        if any(
            value <= 0 for value in (block_rows, block_columns, input_size, output_size)
        ) or (batch_tile_width is not None and batch_tile_width <= 0):
            raise ModelCompileError(
                f"invalid fused FP8 FFN projection shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_silu_multiply_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_silu_multiply_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
            },
        )

    recurrent_depthwise = re.fullmatch(
        r"multiply_rolling_depthwise(_gate)?_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if recurrent_depthwise is not None:
        has_output_gate = recurrent_depthwise.group(1) is not None
        output_gate_binding = (
            "layout(set = 0, binding = 3) readonly buffer OutputGate {\n"
            "    uint words[];\n"
            "} output_gate;"
            if has_output_gate
            else ""
        )
        finalize_output = (
            "uint finalize_output(uint word_index, uint conv_pair) {\n"
            "    uint gate_pair = output_gate.words[word_index];\n"
            "    uint lo = f32_to_bf16(\n"
            "        bf16_to_f32(conv_pair) * bf16_to_f32(gate_pair)\n"
            "    );\n"
            "    uint hi = f32_to_bf16(\n"
            "        bf16_to_f32(conv_pair >> 16) * bf16_to_f32(gate_pair >> 16)\n"
            "    );\n"
            "    return (hi << 16) | lo;\n"
            "}"
            if has_output_gate
            else (
                "uint finalize_output(uint word_index, uint conv_pair) {\n"
                "    return conv_pair;\n"
                "}"
            )
        )
        binding_offset = 1 if has_output_gate else 0
        return render_shader_template(
            source_dir,
            "multiply_rolling_depthwise_bf16.comp.template",
            {
                "OUTPUT_GATE_BINDING": output_gate_binding,
                "OUTPUT_BINDING": str(3 + binding_offset),
                "KERNEL_BINDING": str(4 + binding_offset),
                "STATE_READ_BINDING": str(5 + binding_offset),
                "STATE_WRITE_BINDING": str(6 + binding_offset),
                "FRAME_COUNT": recurrent_depthwise.group(2),
                "HIDDEN_SIZE": recurrent_depthwise.group(3),
                "FINALIZE_OUTPUT_FUNCTION": finalize_output,
            },
        )

    shaped_templates = (
        (
            r"rms_norm_batch(\d+)_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_batch_bf16.comp.template",
            ("BATCH_TILE_WIDTH", "HIDDEN_SIZE", "NORM_EPS", "WEIGHT_OFFSET"),
        ),
        (
            r"linear_batch(\d+)_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
            "linear_batch_fp8_e4m3.comp.template",
            (
                "BATCH_TILE_WIDTH",
                "BLOCK_ROWS",
                "BLOCK_COLUMNS",
                "INPUT_SIZE",
                "OUTPUT_SIZE",
            ),
        ),
        (
            r"linear_residual_batch(\d+)_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
            "linear_residual_batch_fp8_e4m3.comp.template",
            (
                "BATCH_TILE_WIDTH",
                "BLOCK_ROWS",
                "BLOCK_COLUMNS",
                "INPUT_SIZE",
                "OUTPUT_SIZE",
            ),
        ),
        (
            r"silu_multiply_bf16_(\d+)\.comp",
            "silu_multiply_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"linear_int4_ct_g(\d+)_(\d+)x(\d+)\.comp",
            "linear_int4_ct.comp.template",
            ("GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_bias_int4_ct_g(\d+)_(\d+)x(\d+)\.comp",
            "linear_bias_int4_ct.comp.template",
            ("GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_residual_int4_ct_g(\d+)_(\d+)x(\d+)\.comp",
            "linear_residual_int4_ct.comp.template",
            ("GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_int4_gptq_g(\d+)_(\d+)x(\d+)\.comp",
            "linear_int4_gptq.comp.template",
            ("GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_bias_int4_gptq_g(\d+)_(\d+)x(\d+)\.comp",
            "linear_bias_int4_gptq.comp.template",
            ("GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_residual_int4_gptq_g(\d+)_(\d+)x(\d+)\.comp",
            "linear_residual_int4_gptq.comp.template",
            ("GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
            "linear_fp8_e4m3.comp.template",
            ("BLOCK_ROWS", "BLOCK_COLUMNS", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_bias_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
            "linear_bias_fp8_e4m3.comp.template",
            ("BLOCK_ROWS", "BLOCK_COLUMNS", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_residual_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
            "linear_residual_fp8_e4m3.comp.template",
            ("BLOCK_ROWS", "BLOCK_COLUMNS", "INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"sigmoid_scalar_multiply_bf16_(\d+)\.comp",
            "sigmoid_scalar_multiply_bf16.comp.template",
            ("HIDDEN_SIZE",),
        ),
        (
            r"linear_bf16_(\d+)x(\d+)\.comp",
            "linear_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_paired_bf16_(\d+)x(\d+)\.comp",
            "linear_paired_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_split_3way_(paired|row_major)_bf16_(\d+)x(\d+)_(\d+)_(\d+)\.comp",
            "linear_split_3way_bf16.comp.template",
            (
                "WEIGHT_LAYOUT",
                "INPUT_SIZE",
                "PART_A_WIDTH",
                "PART_B_WIDTH",
                "PART_C_WIDTH",
            ),
        ),
        (
            r"linear_split_recurrent_depthwise_gate_(paired|row_major)_bf16_"
            r"(\d+)x(\d+)_k(\d+)_ig([012])_([012])_og([012])\.comp",
            "linear_split_recurrent_depthwise_gate_bf16.comp.template",
            (
                "WEIGHT_LAYOUT",
                "INPUT_SIZE",
                "HIDDEN_SIZE",
                "FRAME_COUNT",
                "INPUT_GATE_A_INDEX",
                "INPUT_GATE_B_INDEX",
                "OUTPUT_GATE_INDEX",
            ),
        ),
        (
            r"linear_bias_bf16_(\d+)x(\d+)\.comp",
            "linear_bias_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_bias_paired_bf16_(\d+)x(\d+)\.comp",
            "linear_bias_paired_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_residual_bf16_(\d+)x(\d+)\.comp",
            "linear_residual_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_residual_paired_bf16_(\d+)x(\d+)\.comp",
            "linear_residual_paired_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"embedding_lookup_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)\.comp",
            "embedding_lookup_bf16.comp.template",
            ("VOCAB_SIZE", "HIDDEN_SIZE", "EMBEDDING_SCALE"),
        ),
        (
            r"embedding_lookup_paired_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)\.comp",
            "embedding_lookup_paired_bf16.comp.template",
            ("VOCAB_SIZE", "HIDDEN_SIZE", "EMBEDDING_SCALE"),
        ),
        (
            r"tied_output_projection_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_bf16.comp.template",
            ("VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"tied_output_projection_paired_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_paired_bf16.comp.template",
            ("VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"tied_output_projection_batch(\d+)_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_batch_bf16.comp.template",
            ("BATCH_TILE_WIDTH", "VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"tied_output_projection_batch(\d+)_paired_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_batch_paired_bf16.comp.template",
            ("BATCH_TILE_WIDTH", "VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"rms_norm_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_bf16.comp.template",
            ("HIDDEN_SIZE", "NORM_EPS", "WEIGHT_OFFSET"),
        ),
        (
            r"rms_norm_per_head_bf16_(\d+)x(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_per_head_bf16.comp.template",
            ("HEAD_COUNT", "HEAD_WIDTH", "NORM_EPS", "WEIGHT_OFFSET"),
        ),
        (
            r"rms_norm_per_head_unscaled_bf16_(\d+)x(\d+)_eps([0-9eE+.-]+)\.comp",
            "rms_norm_per_head_unscaled_bf16.comp.template",
            ("HEAD_COUNT", "HEAD_WIDTH", "NORM_EPS"),
        ),
        (
            r"parallel_head_norm_rope_2way_bf16_h(\d+)_(\d+)_d(\d+)_r(\d+)"
            r"_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)_theta([0-9eE+.-]+)"
            r"_(half|interleaved|proportional)\.comp",
            "parallel_head_norm_rope_2way_bf16.comp.template",
            (
                "BRANCH_A_HEADS",
                "BRANCH_B_HEADS",
                "HEAD_WIDTH",
                "ROTARY_WIDTH",
                "NORM_EPS",
                "WEIGHT_OFFSET",
                "ROPE_THETA",
                "ROPE_LAYOUT",
            ),
        ),
        (
            r"rotary_bf16_(\d+)x(\d+)_r(\d+)_theta([0-9eE+.-]+)_(half|interleaved|proportional)\.comp",
            "rotary_bf16.comp.template",
            ("HEAD_COUNT", "HEAD_WIDTH", "ROTARY_WIDTH", "ROPE_THETA", "ROPE_LAYOUT"),
        ),
        (
            r"append_kv_state_bf16_(\d+)x(\d+)\.comp",
            "append_kv_state_bf16.comp.template",
            ("KV_HEADS", "HEAD_WIDTH"),
        ),
        (
            r"greedy_sampler_f32_(\d+)\.comp",
            "greedy_sampler_f32.comp.template",
            ("VOCAB_SIZE",),
        ),
        (
            r"temperature_top_k_candidates_f32_(\d+)_k(\d+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_candidates_f32.comp.template",
            ("VOCAB_SIZE", "TOP_K", "PARTITION_COUNT", "LOCAL_SIZE_X"),
        ),
        (
            r"temperature_top_k_top_p_sampler_f32_t([0-9eE+.-]+)_k(\d+)_p([0-9eE+.-]+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_top_p_sampler_f32.comp.template",
            ("TEMPERATURE", "TOP_K", "TOP_P", "PARTITION_COUNT", "LOCAL_SIZE_X"),
        ),
        (
            r"split_bf16_2x(\d+)\.comp",
            "split_bf16_2way.comp.template",
            ("PART_WIDTH",),
        ),
        (
            r"split_bf16_3x(\d+)\.comp",
            "split_bf16_3way.comp.template",
            ("PART_WIDTH",),
        ),
        (
            r"split_bf16_3x(\d+)_(\d+)_(\d+)\.comp",
            "split_bf16_3way_widths.comp.template",
            ("PART_A_WIDTH", "PART_B_WIDTH", "PART_C_WIDTH"),
        ),
        (
            r"split_bf16_2x(\d+)x(\d+)_head_interleaved\.comp",
            "split_bf16_2way_head_interleaved.comp.template",
            ("BLOCKS", "BLOCK_PART_WIDTH"),
        ),
        (
            r"causal_conv1d_silu_bf16_c(\d+)_k(\d+)\.comp",
            "causal_conv1d_silu_bf16.comp.template",
            ("CHANNELS", "KERNEL_WIDTH"),
        ),
        (
            r"rolling_state_update_bf16_(\d+)x(\d+)\.comp",
            "rolling_state_update_bf16.comp.template",
            ("FRAME_COUNT", "HIDDEN_SIZE"),
        ),
        (
            r"depthwise_conv1d_bf16_(\d+)x(\d+)\.comp",
            "depthwise_conv1d_bf16.comp.template",
            ("FRAME_COUNT", "HIDDEN_SIZE"),
        ),
        (
            r"scaled_add_bf16_(\d+)_scale([0-9eE+.-]+)\.comp",
            "scaled_add_bf16.comp.template",
            ("ELEMENT_COUNT", "RESIDUAL_SCALE"),
        ),
        (
            r"add_bf16_(\d+)\.comp",
            "add_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"multiply_bf16_(\d+)\.comp",
            "multiply_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"scalar_multiply_bf16_(\d+)\.comp",
            "scalar_multiply_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"gelu_tanh_bf16_(\d+)\.comp",
            "gelu_tanh_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"silu_bf16_(\d+)\.comp",
            "silu_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
    )
    for pattern, template, names in shaped_templates:
        match = re.fullmatch(pattern, shader_file)
        if match is not None:
            replacements = dict(zip(names, match.groups(), strict=True))
            if template == "causal_conv1d_silu_bf16.comp.template":
                channels = int(replacements["CHANNELS"])
                kernel_width = int(replacements["KERNEL_WIDTH"])
                if channels % 2 != 0 or kernel_width % 2 != 0:
                    raise ModelCompileError(
                        "packed BF16 causal convolution requires even channel and kernel widths, "
                        f"got {channels} channels and kernel width {kernel_width}"
                    )
            if "ROPE_LAYOUT" in replacements:
                rope_layout = replacements.pop("ROPE_LAYOUT")
                replacements["ROPE_INTERLEAVED"] = (
                    "true" if rope_layout == "interleaved" else "false"
                )
                replacements["ROPE_PROPORTIONAL"] = (
                    "true" if rope_layout == "proportional" else "false"
                )
            if "WEIGHT_LAYOUT" in replacements:
                weight_layout = replacements.pop("WEIGHT_LAYOUT")
                replacements["PAIRED_WEIGHT_LAYOUT"] = (
                    "true" if weight_layout == "paired" else "false"
                )
            return render_shader_template(source_dir, template, replacements)

    per_layer_embedding_shape = re.fullmatch(
        r"per_layer_embedding_paired_bf16_v(\d+)_h(\d+)_p(\d+)_l(\d+)of(\d+)"
        r"_eps([0-9eE+.-]+)_tes([0-9eE+.-]+)_pes([0-9eE+.-]+)"
        r"_mps([0-9eE+.-]+)_cs([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if per_layer_embedding_shape is not None:
        (
            vocab_size,
            hidden_size,
            per_layer_width,
            layer_index,
            layer_count,
        ) = map(int, per_layer_embedding_shape.groups()[:5])
        if hidden_size % 2 or per_layer_width % 2:
            raise ModelCompileError(
                "paired per-layer embeddings require even hidden and per-layer widths"
            )
        if not 0 <= layer_index < layer_count:
            raise ModelCompileError(
                f"per-layer embedding index {layer_index} is outside {layer_count} layers"
            )
        return render_shader_template(
            source_dir,
            "per_layer_embedding_paired_bf16.comp.template",
            {
                "VOCAB_SIZE": str(vocab_size),
                "HIDDEN_SIZE": str(hidden_size),
                "PER_LAYER_WIDTH": str(per_layer_width),
                "LAYER_INDEX": str(layer_index),
                "LAYER_COUNT": str(layer_count),
                "NORM_EPS": per_layer_embedding_shape.group(6),
                "TOKEN_EMBEDDING_SCALE": per_layer_embedding_shape.group(7),
                "PER_LAYER_EMBEDDING_SCALE": per_layer_embedding_shape.group(8),
                "MODEL_PROJECTION_SCALE": per_layer_embedding_shape.group(9),
                "COMBINATION_SCALE": per_layer_embedding_shape.group(10),
            },
        )

    attention_shape = re.fullmatch(
        r"gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)(?:_w(\d+))?(_sinks)?\.comp",
        shader_file,
    )
    if attention_shape is not None:
        query_heads, kv_heads, head_width = map(int, attention_shape.groups()[:3])
        if query_heads % kv_heads != 0:
            raise ModelCompileError(
                f"query head count {query_heads} is not divisible by KV head count {kv_heads}"
            )
        local_size, tile_tokens = attention_workgroup_shape(head_width)
        if head_width < 2 or head_width % 2 != 0 or tile_tokens == 0:
            raise ModelCompileError(
                f"attention head width {head_width} cannot be tiled into a Vulkan workgroup"
            )
        return render_shader_template(
            source_dir,
            "gqa_attention_bf16.comp.template",
            {
                "QUERY_HEADS": str(query_heads),
                "KV_HEADS": str(kv_heads),
                "QUERY_GROUPS_PER_KV_HEAD": str(query_heads // kv_heads),
                "HEAD_WIDTH": str(head_width),
                "LOCAL_SIZE": str(local_size),
                "TILE_TOKENS": str(tile_tokens),
                "ATTENTION_SCALE": attention_shape.group(4),
                "ATTENTION_WINDOW": attention_shape.group(5) or "0",
                "HAS_SINKS": "1" if attention_shape.group(6) else "0",
            },
        )

    append_attention_shape = re.fullmatch(
        r"append_gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)"
        r"(?:_w(\d+))?(_sinks)?\.comp",
        shader_file,
    )
    if append_attention_shape is not None:
        query_heads, kv_heads, head_width = map(
            int, append_attention_shape.groups()[:3]
        )
        if query_heads % kv_heads != 0:
            raise ModelCompileError(
                f"query head count {query_heads} is not divisible by KV head count {kv_heads}"
            )
        local_size, tile_tokens = attention_workgroup_shape(head_width)
        if head_width < 2 or head_width % 2 != 0 or tile_tokens == 0:
            raise ModelCompileError(
                f"attention head width {head_width} cannot be tiled into a Vulkan workgroup"
            )
        has_sinks = append_attention_shape.group(6) is not None
        return render_shader_template(
            source_dir,
            "append_gqa_attention_bf16.comp.template",
            {
                "QUERY_HEADS": str(query_heads),
                "KV_HEADS": str(kv_heads),
                "QUERY_GROUPS_PER_KV_HEAD": str(query_heads // kv_heads),
                "HEAD_WIDTH": str(head_width),
                "LOCAL_SIZE": str(local_size),
                "TILE_TOKENS": str(tile_tokens),
                "ATTENTION_SCALE": append_attention_shape.group(4),
                "ATTENTION_WINDOW": append_attention_shape.group(5) or "0",
                "HAS_SINKS": "1" if has_sinks else "0",
                "ATTENTION_SINK_BINDING": "5",
                "STATE_READ_BINDING": "6" if has_sinks else "5",
                "STATE_WRITE_BINDING": "7" if has_sinks else "6",
            },
        )

    gated_delta_shape = re.fullmatch(
        r"gated_delta_step_k(\d+)x(\d+)_v(\d+)x(\d+)"
        r"_a(f32|bf16)_dt(f32|bf16)_n(f32|bf16)_eps([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if gated_delta_shape is not None:
        key_heads, key_width, value_heads, value_width = map(
            int, gated_delta_shape.groups()[:4]
        )
        if value_heads % key_heads != 0:
            raise ModelCompileError(
                f"gated-delta value head count {value_heads} is not divisible by key head count {key_heads}"
            )
        if value_width > 1024 or value_width < 2 or value_width % 2 != 0:
            raise ModelCompileError(
                f"gated-delta value head width {value_width} is not a supported workgroup width"
            )
        return render_shader_template(
            source_dir,
            "gated_delta_step.comp.template",
            {
                "KEY_HEADS": str(key_heads),
                "KEY_HEAD_WIDTH": str(key_width),
                "VALUE_HEADS": str(value_heads),
                "VALUE_HEAD_WIDTH": str(value_width),
                "KEY_HEAD_REPEAT": str(value_heads // key_heads),
                "READ_A_LOG": scalar_parameter_read_expression(
                    "a_log", gated_delta_shape.group(5)
                ),
                "READ_DT_BIAS": scalar_parameter_read_expression(
                    "dt_bias", gated_delta_shape.group(6)
                ),
                "READ_NORM_WEIGHT": scalar_parameter_read_expression(
                    "norm_weight", gated_delta_shape.group(7)
                ),
                "NORM_EPS": gated_delta_shape.group(8),
            },
        )

    rg_lru_shape = re.fullmatch(
        r"rg_lru_step_bf16_h(\d+)_b(\d+)x(\d+)_k(\d+)\.comp", shader_file
    )
    if rg_lru_shape is not None:
        width, heads, block_width, kernel_width = map(int, rg_lru_shape.groups())
        if heads * block_width != width:
            raise ModelCompileError(
                f"RG-LRU block shape {heads}x{block_width} does not equal width {width}"
            )
        if block_width > 1024 or block_width % 2:
            raise ModelCompileError(
                f"RG-LRU block width {block_width} is not a supported workgroup width"
            )
        return render_shader_template(
            source_dir,
            "rg_lru_step_bf16.comp.template",
            {
                "WIDTH": str(width),
                "HEADS": str(heads),
                "BLOCK_WIDTH": str(block_width),
                "KERNEL_WIDTH": str(kernel_width),
            },
        )

    moe_topk_shape = re.fullmatch(r"moe_topk_bf16_e(\d+)_k(\d+)\.comp", shader_file)
    if moe_topk_shape is not None:
        num_experts, experts_per_token = map(int, moe_topk_shape.groups())
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_shader_template(
            source_dir,
            "moe_topk_bf16.comp.template",
            {
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
            },
        )

    sparse_moe_fp8_shape = re.fullmatch(
        r"sparse_moe_experts_fp8_e4m3_b(\d+)x(\d+)_h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_fp8_shape is not None:
        (
            block_rows,
            block_columns,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = map(int, sparse_moe_fp8_shape.groups())
        if hidden_size % 2 or intermediate_size % 2:
            raise ModelCompileError(
                "packed BF16 activations for FP8 sparse experts require even dimensions"
            )
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_shader_template(
            source_dir,
            "sparse_moe_experts_fp8_e4m3.comp.template",
            {
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
            },
        )

    sparse_moe_shape = re.fullmatch(
        r"sparse_moe_experts_bf16_h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_shape is not None:
        hidden_size, intermediate_size, num_experts, experts_per_token = map(
            int, sparse_moe_shape.groups()
        )
        if hidden_size % 2 or intermediate_size % 2:
            raise ModelCompileError(
                "packed BF16 sparse experts require even dimensions"
            )
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_shader_template(
            source_dir,
            "sparse_moe_experts_bf16.comp.template",
            {
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
            },
        )

    moe_reduce_shape = re.fullmatch(r"moe_reduce_bf16_h(\d+)_e(\d+)\.comp", shader_file)
    if moe_reduce_shape is not None:
        hidden_size, num_experts = map(int, moe_reduce_shape.groups())
        return render_shader_template(
            source_dir,
            "moe_reduce_bf16.comp.template",
            {
                "HIDDEN_SIZE": str(hidden_size),
                "NUM_EXPERTS": str(num_experts),
            },
        )

    raise ModelCompileError(f"missing shader source or template for {shader_file}")


def render_shader_template(
    source_dir: Path, template_file: str, replacements: dict[str, str]
) -> str:
    template_path = source_dir / template_file
    if not template_path.exists():
        raise ModelCompileError(f"missing shader template {template_path}")
    rendered = template_path.read_text()
    for name, value in replacements.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
    unresolved = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", rendered)))
    if unresolved:
        raise ModelCompileError(
            f"shader template {template_path} has unresolved values: {', '.join(unresolved)}"
        )
    return rendered


def compile_shader_artifacts(
    shader_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        raise ModelCompileError(
            "compiling a Vulkan model package requires glslangValidator"
        )

    sources = sorted(shader_dir.glob("*.comp"))
    if not sources:
        raise ModelCompileError(
            f"no Vulkan shader sources were rendered in {shader_dir}"
        )
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        check_compile_cancelled(cancel_requested)
        if progress is not None:
            progress(index, total, source.name)
        destination = source.with_suffix(".spv")
        completed = subprocess.run(
            [
                compiler,
                "-V",
                "--target-env",
                "vulkan1.4",
                str(source),
                "-o",
                str(destination),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise ModelCompileError(
                f"failed to compile Vulkan shader {source}: {diagnostic}"
            )
        compiled = destination.read_bytes()
        if len(compiled) < 4 or compiled[:4] != b"\x03\x02#\x07":
            raise ModelCompileError(
                f"shader compiler produced invalid SPIR-V artifact {destination}"
            )
        source.unlink()


def compiled_shader_path(source_path: str) -> str:
    if not source_path.endswith(".comp"):
        raise ModelCompileError(
            f"compiled Vulkan shader source path must end in .comp: {source_path!r}"
        )
    return f"{source_path[:-5]}.spv"


def stream_control_binding_for_node(circuit: Json, node: Json) -> int:
    state_view_signals = {
        output
        for producer in circuit["nodes"]
        if producer.get("op") in {"append_state_update", "rolling_state_update"}
        for output in producer.get("outputs", [])
    }
    signal_bindings = [*node.get("inputs", []), *node.get("outputs", [])]
    state_view_binding_count = sum(
        signal in state_view_signals for signal in signal_bindings
    )
    return (
        len(node.get("inputs", []))
        + len(node.get("outputs", []))
        + len(node.get("params", []))
        + len(node.get("state_reads", []))
        + len(node.get("state_writes", []))
        + state_view_binding_count
    )


def copy_tokenizer_package(model_dir: Path, dest_dir: Path) -> Json:
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ModelCompileError(
            f"source model does not contain required tokenizer file {tokenizer_json}"
        )

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for filename in TOKENIZER_PACKAGE_FILES:
        source = model_dir / filename
        if source.is_file():
            shutil.copy2(source, dest_dir / filename)
            copied_files.append(filename)

    if "chat_template.jinja" not in copied_files:
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = read_json(tokenizer_config_path)
            inline_template = tokenizer_config.get("chat_template")
            if isinstance(inline_template, str):
                (dest_dir / "chat_template.jinja").write_text(inline_template)
                copied_files.append("chat_template.jinja")

    return {
        "path": TOKENIZER_PACKAGE_DIR,
        "files": copied_files,
    }


def write_runtime_config_package(model_graph: Json, package_dir: Path) -> None:
    token_ids = model_graph["token_ids"]
    write_json(
        package_dir / CONFIG_PACKAGE_FILE,
        {
            "schema": "llmoop.runtime_model_config.v1",
            "bos_token_id": token_ids["bos"],
            "eos_token_id": token_ids["eos"],
            "pad_token_id": token_ids["pad"],
            "dimensions": model_graph["dimensions"],
            "numerics": model_graph["numerics"],
        },
    )


def referenced_tensor_index(
    tensor_index: Json,
    *,
    model_graph: Json,
    lowered_index: Json,
    lowered_dir: Path,
) -> Json:
    referenced = {
        model_graph["graph"]["input_transducer"]["params"]["weight"]["tensor"]
    }
    for component in model_graph["graph"]["output_transducer"]["components"]:
        referenced.update(ref["tensor"] for ref in component.get("params", {}).values())
    for circuit_ref in all_lowered_circuit_refs(lowered_index):
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        referenced.update(
            ref["tensor"] for ref in circuit["parameters"]["refs"].values()
        )

    missing = sorted(referenced - set(tensor_index["tensors"]))
    if missing:
        raise ModelCompileError(
            f"compiled circuit graph references missing tensors: {', '.join(missing)}"
        )
    selected = deepcopy(tensor_index)
    selected["tensors"] = {
        name: deepcopy(tensor_index["tensors"][name]) for name in sorted(referenced)
    }
    selected["totals"] = {
        "tensor_count": len(selected["tensors"]),
        "parameter_count": sum(
            int(info["parameter_count"]) for info in selected["tensors"].values()
        ),
        "byte_count": sum(
            int(info["byte_count"]) for info in selected["tensors"].values()
        ),
    }
    return selected


def all_lowered_circuit_refs(lowered_index: Json) -> list[Json]:
    refs = list(lowered_index["graph"]["circuits"])
    refs.extend(
        circuit_ref
        for draft in lowered_index.get("draft_pedalboards", [])
        for circuit_ref in draft["circuits"]
    )
    return refs


def copy_tensor_package(
    tensor_index: Json,
    package_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    weights_dir = package_dir / WEIGHTS_PACKAGE_DIR
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not tensor_index["tensors"]:
        raise ModelCompileError("tensor index does not declare any source_file entries")

    packaged = deepcopy(tensor_index)
    compiled_sources = []
    tensors = sorted(packaged["tensors"].items())
    total = len(tensors)
    for index, (tensor_name, info) in enumerate(tensors, start=1):
        check_compile_cancelled(cancel_requested)
        if progress is not None:
            progress(index, total, tensor_name)
        layout = compiled_tensor_layout(info, tensor_name=tensor_name)
        digest = blake2s(tensor_name.encode("utf-8"), digest_size=8).hexdigest()
        destination = weights_dir / f"tensor_{digest}.safetensors"
        if info.get("source_parts"):
            header_bytes, data_sha256 = write_compiled_composite_tensor(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
        else:
            source = Path(info["source_file"])
            if not source.is_file():
                raise ModelCompileError(f"tensor source file does not exist: {source}")
            header_bytes, data_sha256 = write_compiled_tensor(
                tensor_name=tensor_name,
                info=info,
                source=source,
                destination=destination,
                layout=layout,
            )
        relative_destination = relative_json_path(package_dir, destination)
        info["source_file"] = relative_destination
        info["data_offsets"] = [0, int(info["byte_count"])]
        info["data_sha256"] = data_sha256
        info["layout"] = layout
        info.pop("source_parts", None)
        info.pop("source_header_bytes", None)
        info.pop("layout_hint", None)
        compiled_sources.append(
            {
                "path": relative_destination,
                "safetensors_header_bytes": header_bytes,
                "metadata": {
                    "format": "llmoop",
                    "layout": layout,
                },
            }
        )

    packaged["source"] = {
        "packaged": True,
        "compiled": True,
        "weights_dir": WEIGHTS_PACKAGE_DIR,
        "weights_file": compiled_sources[0]["path"],
        "weights_files": compiled_sources,
    }

    write_json(package_dir / "tensors.json", packaged)
    return packaged


def validate_compiled_package(package_dir: Path, manifest: Json) -> None:
    if manifest.get("schema") != PACKAGE_SCHEMA:
        raise ModelCompileError(
            f"compiled package has unsupported schema {manifest.get('schema')!r}"
        )
    if not isinstance(manifest.get("package_id"), str) or not manifest["package_id"]:
        raise ModelCompileError("compiled package has no package id")
    if (
        not isinstance(manifest.get("max_context_activations"), int)
        or isinstance(manifest.get("max_context_activations"), bool)
        or manifest["max_context_activations"] <= 0
    ):
        raise ModelCompileError(
            "compiled package max context activation capacity must be positive"
        )
    if (
        not isinstance(manifest.get("pedal_batch_lane_tile_width"), int)
        or isinstance(manifest.get("pedal_batch_lane_tile_width"), bool)
        or manifest["pedal_batch_lane_tile_width"] <= 0
    ):
        raise ModelCompileError(
            "compiled package pedal batch lane tile width must be positive"
        )
    required_device_extensions = manifest.get("required_vulkan_device_extensions")
    if (
        not isinstance(required_device_extensions, list)
        or any(
            not isinstance(extension, str) or not extension
            for extension in required_device_extensions
        )
        or len(required_device_extensions) != len(set(required_device_extensions))
        or required_device_extensions != sorted(required_device_extensions)
    ):
        raise ModelCompileError(
            "compiled package required Vulkan device extensions must be unique sorted names"
        )
    required_files = (
        package_artifact_path(package_dir, manifest.get("config_path"), "config"),
        package_artifact_path(
            package_dir, manifest.get("tensor_index_path"), "tensor index"
        ),
    )
    for path in required_files:
        if not path.is_file():
            raise ModelCompileError(
                f"compiled package is missing required artifact {path}"
            )

    behavioral_path = package_artifact_path(
        package_dir,
        manifest.get("behavioral_validation_path"),
        "behavioral validation",
    )
    if not behavioral_path.is_file():
        raise ModelCompileError(
            f"compiled package is missing behavioral validation artifact {behavioral_path}"
        )
    behavioral = read_json(behavioral_path)
    candidate_circuits = validate_compiled_circuit_graph(manifest)
    auxiliary_circuits = validate_compiled_speculative_decoders(manifest)
    duplicate_circuits = set(candidate_circuits).intersection(auxiliary_circuits)
    if duplicate_circuits:
        raise ModelCompileError(
            "compiled package repeats circuit ids across target and draft graphs: "
            f"{sorted(duplicate_circuits)}"
        )
    all_candidate_circuits = {**candidate_circuits, **auxiliary_circuits}
    validate_compiled_generation_contract(manifest, candidate_circuits)
    validate_behavioral_validation_artifact(behavioral, all_candidate_circuits)
    validate_compiled_pedal_executions(manifest, candidate_circuits)

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not tokenizer.get("path"):
        raise ModelCompileError("compiled package does not declare tokenizer artifacts")
    tokenizer_dir = package_artifact_path(
        package_dir, tokenizer["path"], "tokenizer directory"
    )
    tokenizer_files = tokenizer.get("files")
    if (
        not isinstance(tokenizer_files, list)
        or not tokenizer_files
        or any(
            not isinstance(filename, str) or not filename
            for filename in tokenizer_files
        )
    ):
        raise ModelCompileError(
            "compiled package tokenizer must declare at least one artifact"
        )
    for filename in tokenizer_files:
        path = package_artifact_path(tokenizer_dir, filename, "tokenizer artifact")
        if not path.is_file():
            raise ModelCompileError(
                f"compiled package is missing tokenizer artifact {path}"
            )

    tensor_index = read_json(required_files[1])
    if tensor_index.get("schema") != "llmoop.tensor_index.v1":
        raise ModelCompileError("compiled package tensor index schema is invalid")
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ModelCompileError("compiled package tensor index contains no tensors")
    for tensor_name, info in tensors.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(info, dict)
        ):
            raise ModelCompileError(
                "compiled package tensor index contains an invalid tensor"
            )
        source = package_artifact_path(
            package_dir,
            info.get("source_file"),
            f"tensor {tensor_name!r} source",
        )
        if not source.is_file():
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} references missing artifact {source}"
            )
        data_digest = info.get("data_sha256")
        if (
            not isinstance(data_digest, str)
            or len(data_digest) != 64
            or any(character not in "0123456789abcdef" for character in data_digest)
        ):
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} has no valid data SHA-256"
            )

    shader_paths: set[str] = set()

    def collect_shader_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("shader_path") and isinstance(child, str):
                    shader_paths.add(child)
                else:
                    collect_shader_paths(child)
        elif isinstance(value, list):
            for child in value:
                collect_shader_paths(child)

    collect_shader_paths(manifest)
    if not shader_paths:
        raise ModelCompileError(
            "compiled package does not reference any shader artifacts"
        )
    for relative_path in sorted(shader_paths):
        shader = package_artifact_path(package_dir, relative_path, "shader")
        if not shader.is_file():
            raise ModelCompileError(
                f"compiled package references missing shader {shader}"
            )
        payload = shader.read_bytes()
        if len(payload) < 4 or payload[:4] != b"\x03\x02#\x07":
            raise ModelCompileError(
                f"compiled package shader is not valid SPIR-V: {shader}"
            )
    validate_package_artifact_integrity(package_dir, manifest)


def build_package_artifact_integrity(package_dir: Path) -> Json:
    files = {}
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir)
        if (
            relative.parts[0] == WEIGHTS_PACKAGE_DIR
            or relative.name == "vulkan_resident_package.json"
        ):
            continue
        payload = path.read_bytes()
        files[relative.as_posix()] = {
            "byte_count": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    return {
        "schema": PACKAGE_ARTIFACT_INTEGRITY_SCHEMA,
        "algorithm": "sha256",
        "files": files,
    }


def validate_package_artifact_integrity(package_dir: Path, manifest: Json) -> None:
    integrity = manifest.get("artifact_integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("schema") != PACKAGE_ARTIFACT_INTEGRITY_SCHEMA
        or integrity.get("algorithm") != "sha256"
        or not isinstance(integrity.get("files"), dict)
        or not integrity["files"]
    ):
        raise ModelCompileError(
            "compiled package artifact integrity contract is invalid"
        )

    actual_files = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
        and path.relative_to(package_dir).parts[0] != WEIGHTS_PACKAGE_DIR
        and path.name != "vulkan_resident_package.json"
    }
    if set(integrity["files"]) != actual_files:
        raise ModelCompileError(
            "compiled package artifact integrity contract does not cover every non-weight artifact"
        )
    for relative_path, contract in integrity["files"].items():
        path = package_artifact_path(package_dir, relative_path, "integrity artifact")
        if (
            not isinstance(contract, dict)
            or not isinstance(contract.get("byte_count"), int)
            or isinstance(contract.get("byte_count"), bool)
            or contract["byte_count"] < 0
            or not isinstance(contract.get("sha256"), str)
            or len(contract["sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in contract["sha256"]
            )
        ):
            raise ModelCompileError(
                f"compiled package artifact integrity entry for {relative_path!r} is invalid"
            )
        payload = path.read_bytes()
        if (
            len(payload) != contract["byte_count"]
            or sha256(payload).hexdigest() != contract["sha256"]
        ):
            raise ModelCompileError(
                f"compiled package artifact {relative_path!r} does not match its integrity contract"
            )


def package_artifact_path(package_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"compiled package has no {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCompileError(
            f"compiled package {label} path must stay inside the package: {value!r}"
        )
    return package_dir / relative


def validate_compiled_circuit_graph(manifest: Json) -> dict[str, Json]:
    graph = manifest.get("circuit_graph")
    if not isinstance(graph, dict) or graph.get("wiring") != "explicit_graph":
        raise ModelCompileError(
            "compiled package must contain an explicit circuit graph"
        )
    pedals = graph.get("pedals")
    if not isinstance(pedals, list) or not pedals:
        raise ModelCompileError("compiled package circuit graph contains no pedals")

    candidates: dict[str, Json] = {}
    for pedal in pedals:
        pedal_id = pedal.get("pedal_id") if isinstance(pedal, dict) else None
        if not isinstance(pedal_id, str) or not pedal_id:
            raise ModelCompileError(
                "compiled package circuit graph contains a pedal without an id"
            )
        if pedal_id in candidates:
            raise ModelCompileError(
                f"compiled package circuit graph repeats pedal {pedal_id!r}"
            )
        circuit = pedal.get("circuit")
        if not isinstance(circuit, dict):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} has no circuit"
            )
        report = validate_circuit(circuit)
        if not report.ok:
            try:
                report.raise_for_errors()
            except ValueError as error:
                raise ModelCompileError(str(error)) from error
        source = circuit.get("source")
        if not isinstance(source, dict) or source.get("pedal_id") != pedal_id:
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} circuit identity does not match"
            )
        if pedal.get("operator_type") != source.get("source_operator_type"):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} operator identity does not match"
            )
        if pedal.get("runtime_role") != circuit.get("runtime_role"):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} runtime role does not match"
            )
        if pedal.get("implementation") != circuit.get("implementation"):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} implementation does not match"
            )
        if pedal.get("behavioral_role") != circuit.get("behavioral_role"):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} behavioral role does not match"
            )
        params = pedal.get("params")
        state = pedal.get("state")
        if (
            not isinstance(params, dict)
            or params.get("schema") != "llmoop.circuit_params.v1"
            or params.get("circuit") != circuit.get("id")
            or params.get("layout") != circuit.get("parameters", {}).get("layout")
            or params.get("storage") != circuit.get("parameters", {}).get("storage")
            or params.get("refs") != circuit.get("parameters", {}).get("refs")
        ):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} parameter artifact does not match its circuit"
            )
        if (
            not isinstance(state, dict)
            or state.get("schema") != "llmoop.circuit_state.v1"
            or state.get("circuit") != circuit.get("id")
            or state.get("state_ports", []) != circuit.get("state_ports", [])
        ):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} state artifact does not match its circuit"
            )
        candidates[pedal_id] = circuit

    compiler_owned_placement = {"device_id", "placement"}.intersection(manifest)
    if compiler_owned_placement:
        raise ModelCompileError(
            "compiled package must not contain runtime placement fields "
            f"{sorted(compiler_owned_placement)}"
        )

    cables = graph.get("cables")
    if not isinstance(cables, list):
        raise ModelCompileError("compiled package circuit graph cables must be a list")
    cable_ids: set[str] = set()
    connected_outputs: set[tuple[str, str]] = set()
    connected_inputs: set[tuple[str, str]] = set()
    forward_inputs: set[tuple[str, str]] = set()
    feedback_inputs: set[tuple[str, str]] = set()
    forward_indegree = {pedal_id: 0 for pedal_id in candidates}
    forward_destinations: dict[str, list[str]] = {
        pedal_id: [] for pedal_id in candidates
    }
    for cable in cables:
        cable_id = cable.get("id") if isinstance(cable, dict) else None
        source = cable.get("source") if isinstance(cable, dict) else None
        destination = cable.get("destination") if isinstance(cable, dict) else None
        if not isinstance(cable_id, str) or not cable_id or cable_id in cable_ids:
            raise ModelCompileError(
                f"compiled package circuit graph contains invalid cable id {cable_id!r}"
            )
        cable_ids.add(cable_id)
        if not isinstance(source, dict) or not isinstance(destination, dict):
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} has invalid endpoints"
            )
        source_id = source.get("pedal_id")
        destination_id = destination.get("pedal_id")
        if source_id not in candidates or destination_id not in candidates:
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} references an unknown pedal"
            )
        connection = cable.get("connection")
        if not isinstance(connection, dict):
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} has no connection contract"
            )
        connection_kind = connection.get("kind")
        if connection_kind not in {"forward", "temporal_feedback"}:
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} has unsupported connection kind {connection_kind!r}"
            )
        if connection_kind == "temporal_feedback":
            delay = connection.get("delay_activations")
            if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
                raise ModelCompileError(
                    f"compiled package temporal feedback cable {cable_id!r} must delay at least one activation"
                )
        if source_id == destination_id and connection_kind == "forward":
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} creates an instantaneous self-loop"
            )
        output = _port_by_id(
            candidates[source_id]["boundary"]["outputs"], source.get("port_id")
        )
        input_port = _port_by_id(
            candidates[destination_id]["boundary"]["inputs"],
            destination.get("port_id"),
        )
        if output is None or input_port is None:
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} references an unknown port"
            )
        if output.get("signal") != input_port.get("signal") or output.get(
            "shape"
        ) != input_port.get("shape"):
            raise ModelCompileError(
                f"compiled package cable {cable_id!r} connects incompatible ports"
            )
        source_endpoint = (source_id, source["port_id"])
        destination_endpoint = (destination_id, destination["port_id"])
        destination_set = (
            forward_inputs if connection_kind == "forward" else feedback_inputs
        )
        if destination_endpoint in destination_set:
            raise ModelCompileError(
                f"compiled package input {destination_id}.{destination['port_id']} has multiple {connection_kind} cables"
            )
        destination_set.add(destination_endpoint)
        connected_outputs.add(source_endpoint)
        connected_inputs.add(destination_endpoint)
        if connection_kind == "forward":
            forward_indegree[destination_id] += 1
            forward_destinations[source_id].append(destination_id)

    remaining = set(candidates)
    while remaining:
        ready = next(
            (
                pedal_id
                for pedal_id in candidates
                if pedal_id in remaining and forward_indegree[pedal_id] == 0
            ),
            None,
        )
        if ready is None:
            raise ModelCompileError(
                "compiled package circuit graph contains an instantaneous cycle"
            )
        remaining.remove(ready)
        for destination_id in forward_destinations[ready]:
            forward_indegree[destination_id] -= 1

    boundary = graph.get("boundary")
    if not isinstance(boundary, dict):
        raise ModelCompileError("compiled package circuit graph has no boundary")
    external_inputs = _validate_package_graph_boundary_ports(
        boundary.get("external_inputs"),
        candidates,
        kind="external input",
        direction="inputs",
    )
    public_outputs = _validate_package_graph_boundary_ports(
        boundary.get("public_outputs"),
        candidates,
        kind="public output",
        direction="outputs",
    )

    unrouted_inputs = []
    unrouted_outputs = []
    for pedal_id, circuit in candidates.items():
        unrouted_inputs.extend(
            (pedal_id, port["id"])
            for port in circuit["boundary"]["inputs"]
            if (pedal_id, port["id"]) not in connected_inputs
            and (pedal_id, port["id"]) not in external_inputs
        )
        unrouted_outputs.extend(
            (pedal_id, port["id"])
            for port in circuit["boundary"]["outputs"]
            if (pedal_id, port["id"]) not in connected_outputs
            and (pedal_id, port["id"]) not in public_outputs
        )
    if unrouted_inputs or unrouted_outputs:
        raise ModelCompileError(
            "compiled package circuit graph has unrouted ports; "
            f"inputs={unrouted_inputs}, outputs={unrouted_outputs}"
        )
    return candidates


def _validate_package_graph_boundary_ports(
    ports: Any,
    candidates: dict[str, Json],
    *,
    kind: str,
    direction: str,
) -> set[tuple[str, str]]:
    if not isinstance(ports, list) or not ports:
        raise ModelCompileError(
            f"compiled package circuit graph must declare at least one {kind}"
        )
    ids: set[str] = set()
    endpoints: set[tuple[str, str]] = set()
    for port in ports:
        port_id = port.get("id") if isinstance(port, dict) else None
        endpoint = port.get("endpoint") if isinstance(port, dict) else None
        if not isinstance(port_id, str) or not port_id or port_id in ids:
            raise ModelCompileError(
                f"compiled package circuit graph has invalid or duplicate {kind} id {port_id!r}"
            )
        ids.add(port_id)
        if not isinstance(endpoint, dict):
            raise ModelCompileError(
                f"compiled package circuit graph {kind} {port_id!r} has no endpoint"
            )
        pedal_id = endpoint.get("pedal_id")
        endpoint_port_id = endpoint.get("port_id")
        circuit = candidates.get(pedal_id)
        if (
            circuit is None
            or _port_by_id(circuit["boundary"][direction], endpoint_port_id) is None
        ):
            raise ModelCompileError(
                f"compiled package circuit graph {kind} {port_id!r} references an unknown {direction[:-1]}"
            )
        key = (pedal_id, endpoint_port_id)
        if key in endpoints:
            raise ModelCompileError(
                f"compiled package circuit graph repeats {kind} endpoint {pedal_id}.{endpoint_port_id}"
            )
        endpoints.add(key)
    return endpoints


def _port_by_id(ports: list[Json], port_id: Any) -> Json | None:
    return next((port for port in ports if port.get("id") == port_id), None)


def validate_compiled_speculative_decoders(manifest: Json) -> dict[str, Json]:
    raw_decoders = manifest.get("speculative_decoders", [])
    if not isinstance(raw_decoders, list):
        raise ModelCompileError("compiled package speculative decoders must be a list")
    decoder_ids: set[str] = set()
    candidates: dict[str, Json] = {}
    for decoder in raw_decoders:
        decoder_id = decoder.get("id") if isinstance(decoder, dict) else None
        if (
            not isinstance(decoder_id, str)
            or not decoder_id
            or decoder_id in decoder_ids
        ):
            raise ModelCompileError(
                f"compiled package contains invalid or duplicate speculative decoder {decoder_id!r}"
            )
        decoder_ids.add(decoder_id)
        if decoder.get("type") != "multi_token_prediction":
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} has unsupported type {decoder.get('type')!r}"
            )
        graph = decoder.get("circuit_graph")
        graph_candidates = validate_compiled_circuit_graph({"circuit_graph": graph})
        duplicate = set(candidates).intersection(graph_candidates)
        if duplicate:
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} repeats pedal ids {sorted(duplicate)}"
            )
        roles: dict[str, list[str]] = {}
        for pedal_id, circuit in graph_candidates.items():
            roles.setdefault(str(circuit.get("runtime_role")), []).append(pedal_id)
        if (
            len(roles.get("draft_input_adapter", [])) != 1
            or not roles.get("draft_processor")
            or len(roles.get("draft_output_transducer", [])) != 1
            or set(roles)
            != {
                "draft_input_adapter",
                "draft_processor",
                "draft_output_transducer",
            }
        ):
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} must contain one input adapter, "
                "at least one draft processor, and one output transducer"
            )
        execution_by_pedal = {
            execution.get("pedal_id"): execution
            for execution in decoder.get("pedal_executions", [])
            if isinstance(execution, dict)
        }
        executable_ids = set(roles["draft_input_adapter"]) | set(
            roles["draft_processor"]
        )
        if set(execution_by_pedal) != executable_ids:
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} executions do not cover its executable pedals"
            )
        for pedal_id in executable_ids:
            circuit = graph_candidates[pedal_id]
            execution = execution_by_pedal[pedal_id]
            kernels = execution.get("kernels")
            if not isinstance(kernels, list) or len(kernels) != len(circuit["nodes"]):
                raise ModelCompileError(
                    f"speculative decoder {decoder_id!r} execution for {pedal_id!r} "
                    "does not cover every circuit node"
                )
            for index, (kernel, node) in enumerate(
                zip(kernels, circuit["nodes"], strict=True)
            ):
                if (
                    kernel.get("execution_index") != index
                    or kernel.get("node_id") != node.get("id")
                    or kernel.get("op") != node.get("op")
                    or not isinstance(kernel.get("shader_path"), str)
                    or not kernel["shader_path"]
                ):
                    raise ModelCompileError(
                        f"speculative decoder {decoder_id!r} kernel {pedal_id}.{index} "
                        "does not match its circuit node"
                    )
        output_id = roles["draft_output_transducer"][0]
        output_spec = decoder.get("output_transducer")
        output_refs = graph_candidates[output_id]["parameters"]["refs"]
        if (
            not isinstance(output_spec, dict)
            or output_spec.get("pedal_id") != output_id
            or output_spec.get("norm_parameter_tensor")
            != output_refs.get("norm", {}).get("tensor")
            or output_spec.get("projection_parameter_tensor")
            != output_refs.get("projection", {}).get("tensor")
            or any(
                not isinstance(output_spec.get(field), str) or not output_spec[field]
                for field in ("norm_shader_path", "projection_shader_path")
            )
        ):
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} output execution does not match its circuit"
            )
        expected_state_contract = {
            "ownership": "per_stream_per_pedal_instance",
            "draft_updates": "tentative",
            "acceptance": "commit_accepted_prefix",
            "rejection": "restore_last_committed_state",
        }
        if decoder.get("state_contract") != expected_state_contract:
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} has no transactional state contract"
            )
        candidates.update(graph_candidates)
    return candidates


def validate_compiled_pedal_executions(
    manifest: Json,
    candidate_circuits: dict[str, Json],
) -> None:
    executions = manifest.get("pedal_executions")
    if not isinstance(executions, list):
        raise ModelCompileError("compiled package has no pedal execution list")
    execution_by_pedal: dict[str, Json] = {}
    for execution in executions:
        pedal_id = execution.get("pedal_id") if isinstance(execution, dict) else None
        if (
            not isinstance(pedal_id, str)
            or not pedal_id
            or pedal_id in execution_by_pedal
        ):
            raise ModelCompileError(
                f"compiled package contains invalid or duplicate pedal execution {pedal_id!r}"
            )
        execution_by_pedal[pedal_id] = execution
    executable_circuits = {
        pedal_id: circuit
        for pedal_id, circuit in candidate_circuits.items()
        if circuit.get("runtime_role") == "signal_processor"
    }
    if set(execution_by_pedal) != set(executable_circuits):
        raise ModelCompileError(
            "compiled package pedal executions do not match its signal-processing circuits"
        )
    for pedal_id, circuit in executable_circuits.items():
        execution = execution_by_pedal[pedal_id]
        source = circuit.get("source", {})
        if execution.get("operator_type") != source.get(
            "source_operator_type"
        ) or execution.get("implementation") != circuit.get("implementation"):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} execution identity does not match its circuit"
            )
        kernels = execution.get("kernels")
        nodes = circuit.get("nodes", [])
        if not isinstance(kernels, list) or len(kernels) != len(nodes):
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} execution does not cover every circuit node"
            )
        for index, (kernel, node) in enumerate(zip(kernels, nodes, strict=True)):
            if (
                not isinstance(kernel, dict)
                or kernel.get("execution_index") != index
                or kernel.get("node_id") != node.get("id")
                or kernel.get("op") != node.get("op")
                or not isinstance(kernel.get("shader_path"), str)
                or not kernel["shader_path"]
            ):
                raise ModelCompileError(
                    f"compiled package pedal {pedal_id!r} kernel {index} does not match its circuit node"
                )
            batch_mode = kernel.get("batch_mode")
            batch_shader_path = kernel.get("batch_shader_path")
            if batch_mode == "serial_lanes" and batch_shader_path is None:
                continue
            if (
                batch_mode == "weight_shared"
                and isinstance(batch_shader_path, str)
                and batch_shader_path
            ):
                continue
            raise ModelCompileError(
                f"compiled package pedal {pedal_id!r} kernel {index} has an invalid batch execution contract"
            )


def validate_compiled_generation_contract(
    manifest: Json,
    candidate_circuits: dict[str, Json],
) -> None:
    role_ids: dict[str, list[str]] = {}
    for pedal_id, circuit in candidate_circuits.items():
        role_ids.setdefault(str(circuit.get("runtime_role")), []).append(pedal_id)
    for role in ("input_transducer", "output_transducer", "sampler"):
        if len(role_ids.get(role, [])) != 1:
            raise ModelCompileError(
                f"compiled generation graph must contain exactly one {role} pedal"
            )
    if not role_ids.get("signal_processor"):
        raise ModelCompileError(
            "compiled generation graph must contain at least one signal processor"
        )

    input_id = role_ids["input_transducer"][0]
    output_id = role_ids["output_transducer"][0]
    sampler_id = role_ids["sampler"][0]
    processor_ids = set(role_ids["signal_processor"])
    graph = manifest["circuit_graph"]
    forward = [
        cable for cable in graph["cables"] if cable["connection"]["kind"] == "forward"
    ]
    feedback = [
        cable
        for cable in graph["cables"]
        if cable["connection"]["kind"] == "temporal_feedback"
    ]
    input_edges = [
        cable
        for cable in forward
        if cable["source"]["pedal_id"] == input_id
        and cable["destination"]["pedal_id"] in processor_ids
    ]
    output_edges = [
        cable
        for cable in forward
        if cable["source"]["pedal_id"] in processor_ids
        and cable["destination"]["pedal_id"] == output_id
    ]
    sampler_edges = [
        cable
        for cable in forward
        if cable["source"]["pedal_id"] == output_id
        and cable["destination"]["pedal_id"] == sampler_id
    ]
    generation_feedback = [
        cable
        for cable in feedback
        if cable["source"]["pedal_id"] == sampler_id
        and cable["destination"]["pedal_id"] == input_id
    ]
    if any(
        len(edges) != 1
        for edges in (
            input_edges,
            output_edges,
            sampler_edges,
            generation_feedback,
        )
    ):
        raise ModelCompileError(
            "compiled generation graph must wire input transducer -> processors -> "
            "output transducer -> sampler with one delayed sampler feedback edge"
        )

    input_circuit = candidate_circuits[input_id]
    output_circuit = candidate_circuits[output_id]
    sampler_circuit = candidate_circuits[sampler_id]
    input_nodes = input_circuit.get("nodes", [])
    output_nodes = output_circuit.get("nodes", [])
    sampler_nodes = sampler_circuit.get("nodes", [])
    if (
        len(input_nodes) != 1
        or len(input_nodes[0].get("inputs", [])) != 1
        or len(input_nodes[0].get("outputs", [])) != 1
        or len(output_nodes) != 2
        or len(output_nodes[0].get("inputs", [])) != 1
        or len(output_nodes[-1].get("outputs", [])) != 1
        or len(sampler_nodes) != 1
        or len(sampler_nodes[0].get("inputs", [])) != 2
        or len(sampler_nodes[0].get("outputs", [])) != 1
    ):
        raise ModelCompileError(
            "compiled generation system pedals have invalid node boundaries"
        )
    input_token_port = input_nodes[0]["inputs"][0]
    input_frame_port = input_nodes[0]["outputs"][0]
    output_frame_port = output_nodes[0]["inputs"][0]
    output_logits_port = output_nodes[-1]["outputs"][0]
    sampler_logits_port, sampler_random_port = sampler_nodes[0]["inputs"]
    sampler_token_port = sampler_nodes[0]["outputs"][0]
    if (
        input_edges[0]["source"]["port_id"] != input_frame_port
        or output_edges[0]["destination"]["port_id"] != output_frame_port
        or sampler_edges[0]["source"]["port_id"] != output_logits_port
        or sampler_edges[0]["destination"]["port_id"] != sampler_logits_port
        or generation_feedback[0]["source"]["port_id"] != sampler_token_port
        or generation_feedback[0]["destination"]["port_id"] != input_token_port
    ):
        raise ModelCompileError(
            "compiled generation graph cables do not match system-pedal ports"
        )

    boundary = graph["boundary"]
    external_endpoints = {
        (port["endpoint"]["pedal_id"], port["endpoint"]["port_id"])
        for port in boundary["external_inputs"]
    }
    public_endpoints = {
        (port["endpoint"]["pedal_id"], port["endpoint"]["port_id"])
        for port in boundary["public_outputs"]
    }
    if (
        len(boundary["external_inputs"]) != 2
        or external_endpoints
        != {(input_id, input_token_port), (sampler_id, sampler_random_port)}
        or len(boundary["public_outputs"]) != 1
        or public_endpoints != {(sampler_id, sampler_token_port)}
    ):
        raise ModelCompileError(
            "compiled generation graph boundaries must expose one user input, one "
            "sampler random seed, and one sampler public output"
        )

    input_package = manifest.get("input_transducer")
    output_package = manifest.get("output_transducer")
    sampler_package = manifest.get("sampler")
    if not all(
        isinstance(value, dict)
        for value in (input_package, output_package, sampler_package)
    ):
        raise ModelCompileError(
            "compiled generation package is missing a system-pedal execution spec"
        )

    input_spec = input_package.get("spec")
    input_refs = input_circuit.get("parameters", {}).get("refs", {})
    if (
        not isinstance(input_spec, dict)
        or len(input_nodes) != 1
        or input_nodes[0].get("op") != "embedding_lookup"
        or input_spec.get("parameter_tensor")
        != input_refs.get("weight", {}).get("tensor")
        or input_spec.get("output_signal_id")
        != input_edges[0]["destination"]["port_id"]
        or not isinstance(input_package.get("shader_path"), str)
        or not input_package["shader_path"]
    ):
        raise ModelCompileError(
            "compiled input-transducer execution does not match its circuit pedal"
        )

    output_spec = output_package.get("spec")
    output_refs = output_circuit.get("parameters", {}).get("refs", {})
    if (
        not isinstance(output_spec, dict)
        or [node.get("id") for node in output_nodes] != output_spec.get("node_ids")
        or [node.get("op") for node in output_nodes]
        != ["rms_norm", "linear_projection"]
        or output_spec.get("norm_parameter_tensor")
        != output_refs.get("output_norm.weight", {}).get("tensor")
        or output_spec.get("projection_parameter_tensor")
        != output_refs.get("output_projection.weight", {}).get("tensor")
        or output_spec.get("input_signal_id") != output_edges[0]["source"]["port_id"]
        or any(
            not isinstance(output_package.get(field), str) or not output_package[field]
            for field in (
                "embedding_norm_shader_path",
                "projection_shader_path",
                "projection_batch_shader_path",
            )
        )
        or not isinstance(output_package.get("projection_batch_lane_tile_width"), int)
        or isinstance(output_package.get("projection_batch_lane_tile_width"), bool)
        or output_package["projection_batch_lane_tile_width"] <= 0
    ):
        raise ModelCompileError(
            "compiled output-transducer execution does not match its circuit pedal"
        )

    sampler_spec = sampler_package.get("spec")
    sampler_attrs = sampler_nodes[0].get("attrs", {}) if len(sampler_nodes) == 1 else {}
    if (
        not isinstance(sampler_spec, dict)
        or len(sampler_nodes) != 1
        or sampler_nodes[0].get("op") != "sample_token"
        or sampler_attrs.get("randomness") != "seed_and_stream_tick"
        or any(
            sampler_spec.get(field) != sampler_attrs.get(field)
            for field in ("method", "temperature", "top_k", "top_p")
        )
        or not isinstance(sampler_package.get("kernels"), list)
        or not sampler_package["kernels"]
    ):
        raise ModelCompileError(
            "compiled sampler execution does not match its circuit pedal"
        )


def compiled_tensor_layout(info: Json, *, tensor_name: str | None = None) -> str:
    shape = [int(value) for value in info.get("shape", [])]
    if info.get("layout_hint") == ROW_MAJOR_LAYOUT:
        return ROW_MAJOR_LAYOUT
    if (
        info.get("dtype") == "BF16"
        and not (tensor_name or "").endswith(
            (".weight_scale_inv", ".weight_scale", ".scales")
        )
        and len(shape) == 2
        and shape[0] % 2 == 0
        and shape[1] % 2 == 0
    ):
        return VULKAN_BF16_ROW_PAIR_LAYOUT
    return ROW_MAJOR_LAYOUT


def write_compiled_tensor(
    *,
    tensor_name: str,
    info: Json,
    source: Path,
    destination: Path,
    layout: str,
) -> tuple[int, str]:
    byte_count = int(info["byte_count"])
    header = {
        "__metadata__": {"format": "llmoop", "layout": layout},
        tensor_name: {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [0, byte_count],
        },
    }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    source_header_bytes = int(
        info.get("source_header_bytes") or read_safetensors_header(source)[0]
    )
    source_start = 8 + source_header_bytes + int(info["data_offsets"][0])
    data_digest = sha256()

    with (
        source.open("rb") as source_handle,
        destination.open("wb") as destination_handle,
    ):
        destination_handle.write(struct.pack("<Q", len(header_payload)))
        destination_handle.write(header_payload)
        source_handle.seek(source_start)
        if layout == VULKAN_BF16_ROW_PAIR_LAYOUT:
            write_bf16_row_pair_tensor(
                source_handle,
                destination_handle,
                rows=int(info["shape"][0]),
                columns=int(info["shape"][1]),
                digest=data_digest,
            )
        else:
            copy_exact_bytes(
                source_handle, destination_handle, byte_count, digest=data_digest
            )
    return len(header_payload), data_digest.hexdigest()


def write_compiled_composite_tensor(
    *,
    tensor_name: str,
    info: Json,
    destination: Path,
    layout: str,
) -> tuple[int, str]:
    if layout != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"composite tensor {tensor_name!r} requires unsupported layout {layout!r}"
        )
    byte_count = int(info["byte_count"])
    header = {
        "__metadata__": {"format": "llmoop", "layout": layout},
        tensor_name: {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [0, byte_count],
        },
    }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    written = 0
    data_digest = sha256()
    with destination.open("wb") as destination_handle:
        destination_handle.write(struct.pack("<Q", len(header_payload)))
        destination_handle.write(header_payload)
        for part in info["source_parts"]:
            source = Path(part["source_file"])
            if not source.is_file():
                raise ModelCompileError(
                    f"composite tensor {tensor_name!r} source does not exist: {source}"
                )
            source_header_bytes = int(part["source_header_bytes"])
            offsets = [int(value) for value in part["data_offsets"]]
            part_bytes = int(part["byte_count"])
            if offsets[1] - offsets[0] != part_bytes:
                raise ModelCompileError(
                    f"composite tensor {tensor_name!r} part {part['tensor']!r} "
                    "has inconsistent byte offsets"
                )
            with source.open("rb") as source_handle:
                source_handle.seek(8 + source_header_bytes + offsets[0])
                copy_exact_bytes(
                    source_handle,
                    destination_handle,
                    part_bytes,
                    digest=data_digest,
                )
            written += part_bytes
    if written != byte_count:
        raise ModelCompileError(
            f"composite tensor {tensor_name!r} wrote {written} bytes; expected {byte_count}"
        )
    return len(header_payload), data_digest.hexdigest()


def write_bf16_row_pair_tensor(
    source_handle: Any,
    destination_handle: Any,
    *,
    rows: int,
    columns: int,
    digest: Any | None = None,
) -> None:
    try:
        import numpy
    except ImportError as error:
        raise ModelCompileError(
            "compiling Vulkan BF16 matrix layouts requires numpy"
        ) from error

    row_bytes = columns * 2
    word_count = columns // 2
    for _row_pair in range(rows // 2):
        row_0 = source_handle.read(row_bytes)
        row_1 = source_handle.read(row_bytes)
        if len(row_0) != row_bytes or len(row_1) != row_bytes:
            raise ModelCompileError(
                "unexpected end of BF16 tensor while compiling row pairs"
            )
        words_0 = numpy.frombuffer(row_0, dtype="<u4", count=word_count)
        words_1 = numpy.frombuffer(row_1, dtype="<u4", count=word_count)
        paired = numpy.empty((word_count, 2), dtype="<u4")
        paired[:, 0] = words_0
        paired[:, 1] = words_1
        payload = paired.tobytes()
        destination_handle.write(payload)
        if digest is not None:
            digest.update(payload)


def copy_exact_bytes(
    source_handle: Any,
    destination_handle: Any,
    byte_count: int,
    *,
    digest: Any | None = None,
) -> None:
    remaining = byte_count
    while remaining:
        chunk = source_handle.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ModelCompileError(
                "unexpected end of tensor source while compiling package"
            )
        destination_handle.write(chunk)
        if digest is not None:
            digest.update(chunk)
        remaining -= len(chunk)


def parameter_shape_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> list[int]:
    parameter_id = node["params"][0]
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_shape(tensor_index, parameter["tensor"])


def can_fuse_bf16_linear_split(circuit: Json, node: Json, tensor_index: Json) -> bool:
    shape = parameter_shape_for_node(circuit, node, tensor_index)
    return (
        len(shape) == 2
        and all(int(dimension) > 0 and int(dimension) % 2 == 0 for dimension in shape)
        and parameter_dtype_for_node(circuit, node, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, node, tensor_index)
        in {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
    )


def can_fuse_bf16_parallel_linears(
    circuit: Json, nodes: list[Json], tensor_index: Json
) -> bool:
    shapes = [parameter_shape_for_node(circuit, node, tensor_index) for node in nodes]
    layouts = {parameter_layout_for_node(circuit, node, tensor_index) for node in nodes}
    return (
        len(nodes) in {2, 3}
        and all(
            len(shape) == 2
            and all(
                int(dimension) > 0 and int(dimension) % 2 == 0 for dimension in shape
            )
            for shape in shapes
        )
        and len({int(shape[1]) for shape in shapes}) == 1
        and all(
            parameter_dtype_for_node(circuit, node, tensor_index) == "BF16"
            for node in nodes
        )
        and len(layouts) == 1
        and layouts <= {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
    )


def can_fuse_parallel_linear_silu_multiply(
    circuit: Json,
    projection: Json,
    activation: Json,
    tensor_index: Json,
) -> bool:
    if (
        projection.get("op") != "parallel_linear_2way"
        or activation.get("op") != "silu_multiply"
    ):
        return False
    try:
        params = projection["params"]
        element_count = int(activation.get("attrs", {})["element_count"])
    except (KeyError, TypeError, ValueError, ModelCompileError):
        return False

    if len(params) == 2:
        weight_ids = params
        try:
            layouts = {
                parameter_layout_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            supported_parameters = (
                {
                    parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                    for parameter_id in weight_ids
                }
                == {"BF16"}
                and len(layouts) == 1
                and layouts <= {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
            )
        except ModelCompileError:
            return False
    elif len(params) == 4:
        weight_ids = [params[0], params[2]]
        branch_params = [params[:2], params[2:]]
        try:
            supported_parameters = (
                all(
                    parameter_dtype_for_id(circuit, weight_id, tensor_index)
                    == "F8_E4M3"
                    and parameter_layout_for_id(circuit, weight_id, tensor_index)
                    == ROW_MAJOR_LAYOUT
                    for weight_id in weight_ids
                )
                and len(
                    {
                        fp8_block_shape_for_node(
                            circuit,
                            {
                                "id": f"{projection['id']}__branch_{index}",
                                "params": parameter_ids,
                            },
                            tensor_index,
                        )
                        for index, parameter_ids in enumerate(branch_params)
                    }
                )
                == 1
            )
        except ModelCompileError:
            return False
    else:
        return False

    try:
        shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in weight_ids
        ]
    except ModelCompileError:
        return False
    return (
        supported_parameters
        and len(shapes) == 2
        and shapes[0] == shapes[1]
        and len(shapes[0]) == 2
        and all(
            int(dimension) > 0 and int(dimension) % 2 == 0 for dimension in shapes[0]
        )
        and element_count == int(shapes[0][0])
        and activation.get("attrs", {}).get("intermediate_rounding") == "BF16"
    )


def can_fuse_bf16_parallel_head_norm_rope(
    circuit: Json,
    branches: list[tuple[Json, Json]],
    tensor_index: Json,
) -> bool:
    if len(branches) != 2:
        return False
    norms = [norm for norm, _rope in branches]
    ropes = [rope for _norm, rope in branches]
    try:
        head_counts = [int(norm["attrs"]["head_count"]) for norm in norms]
        head_widths = {
            int(node["attrs"]["head_width"]) for branch in branches for node in branch
        }
        rotary_widths = {int(rope["attrs"]["rotary_width"]) for rope in ropes}
        common_values = (
            {float(norm["attrs"]["eps"]) for norm in norms},
            {float(norm["attrs"]["weight_offset"]) for norm in norms},
            {float(rope["attrs"]["theta"]) for rope in ropes},
            {str(rope["attrs"].get("rope_type", "default")) for rope in ropes},
            {bool(rope["attrs"]["interleaved"]) for rope in ropes},
            {str(rope["attrs"]["position_source"]) for rope in ropes},
        )
        parameter_shapes = [
            parameter_shape_for_node(circuit, norm, tensor_index) for norm in norms
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if len(head_widths) != 1 or len(rotary_widths) != 1:
        return False
    head_width = next(iter(head_widths))
    rotary_width = next(iter(rotary_widths))
    return (
        all(count > 0 for count in head_counts)
        and head_width > 0
        and head_width % 2 == 0
        and rotary_width > 0
        and rotary_width % 2 == 0
        and rotary_width <= head_width
        and all(len(values) == 1 for values in common_values)
        and common_values[-1] == {"stream_tick"}
        and all(
            int(norm["attrs"]["head_count"]) == int(rope["attrs"]["head_count"])
            for norm, rope in branches
        )
        and all(shape == [head_width] for shape in parameter_shapes)
        and all(
            parameter_dtype_for_node(circuit, norm, tensor_index) == "BF16"
            for norm in norms
        )
        and all(
            parameter_layout_for_node(circuit, norm, tensor_index) == ROW_MAJOR_LAYOUT
            for norm in norms
        )
    )


def can_fuse_bf16_multiply_rolling_depthwise(
    circuit: Json,
    multiply: Json,
    rolling: Json,
    depthwise: Json,
    tensor_index: Json,
) -> bool:
    if (
        multiply.get("op") != "multiply"
        or len(multiply.get("inputs", [])) != 2
        or rolling.get("attrs", {}).get("update") != "shift_append"
        or len(rolling.get("state_reads", [])) != 1
        or rolling.get("state_reads") != rolling.get("state_writes")
        or len(depthwise.get("params", [])) != 1
    ):
        return False
    temporal_memory = state_port(circuit, rolling["state_reads"][0])
    state_shape = list(map(int, temporal_memory.get("shape", [])))
    if len(state_shape) != 2:
        return False
    frames, hidden_size = state_shape
    kernel_shape = parameter_shape_for_node(circuit, depthwise, tensor_index)
    return (
        temporal_memory.get("dtype") == "BF16"
        and temporal_memory.get("update") == "shift_append"
        and frames >= 2
        and hidden_size > 0
        and hidden_size % 2 == 0
        and depthwise.get("attrs", {}).get("groups") == hidden_size
        and kernel_shape in ([hidden_size, frames], [hidden_size, 1, frames])
        and parameter_dtype_for_node(circuit, depthwise, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, depthwise, tensor_index)
        == ROW_MAJOR_LAYOUT
    )


def can_fuse_bf16_recurrent_output_gate(
    circuit: Json,
    recurrent: Json,
    gate: Json,
    tensor_index: Json,
) -> bool:
    if (
        recurrent.get("op") != "multiply_rolling_depthwise"
        or gate.get("op") != "multiply"
        or len(recurrent.get("state_reads", [])) != 1
        or recurrent.get("state_reads") != recurrent.get("state_writes")
        or len(recurrent.get("params", [])) != 1
    ):
        return False
    temporal_memory = state_port(circuit, recurrent["state_reads"][0])
    state_shape = list(map(int, temporal_memory.get("shape", [])))
    if len(state_shape) != 2:
        return False
    frames, hidden_size = state_shape
    kernel_shape = parameter_shape_for_node(circuit, recurrent, tensor_index)
    return (
        temporal_memory.get("dtype") == "BF16"
        and temporal_memory.get("update") == "shift_append"
        and frames >= 2
        and hidden_size > 0
        and hidden_size % 2 == 0
        and kernel_shape in ([hidden_size, frames], [hidden_size, 1, frames])
        and parameter_dtype_for_node(circuit, recurrent, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, recurrent, tensor_index)
        == ROW_MAJOR_LAYOUT
    )


def can_fuse_bf16_linear_split_recurrent(
    circuit: Json,
    projection: Json,
    recurrent: Json,
    tensor_index: Json,
) -> bool:
    if (
        len(projection.get("params", [])) != 1
        or len(recurrent.get("params", [])) != 1
        or len(recurrent.get("state_reads", [])) != 1
        or recurrent.get("state_reads") != recurrent.get("state_writes")
    ):
        return False
    temporal_memory = state_port(circuit, recurrent["state_reads"][0])
    state_shape = list(map(int, temporal_memory.get("shape", [])))
    if len(state_shape) != 2:
        return False
    frames, hidden_size = state_shape
    projection_shape = parameter_shape_for_node(circuit, projection, tensor_index)
    kernel_shape = parameter_shape_for_node(circuit, recurrent, tensor_index)
    part_widths = [
        int(width) for width in projection.get("attrs", {}).get("part_widths", [])
    ]
    return (
        temporal_memory.get("dtype") == "BF16"
        and temporal_memory.get("update") == "shift_append"
        and frames >= 2
        and hidden_size > 0
        and hidden_size % 2 == 0
        and len(projection_shape) == 2
        and projection_shape[0] == 3 * hidden_size
        and projection_shape[1] > 0
        and projection_shape[1] % 2 == 0
        and part_widths == [hidden_size] * 3
        and kernel_shape in ([hidden_size, frames], [hidden_size, 1, frames])
        and parameter_dtype_for_node(circuit, projection, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, projection, tensor_index)
        in {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
        and parameter_dtype_for_node(circuit, recurrent, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, recurrent, tensor_index)
        == ROW_MAJOR_LAYOUT
    )


def can_fuse_bf16_append_attention(
    circuit: Json,
    append: Json,
    attention: Json,
    tensor_index: Json,
) -> bool:
    if (
        append.get("op") != "append_state_update"
        or attention.get("op") != "scaled_dot_product_attention"
        or len(append.get("inputs", [])) != 3
        or len(append.get("state_reads", [])) != 1
        or append.get("state_reads") != append.get("state_writes")
        or append["inputs"][2] != append["state_reads"][0]
        or attention.get("attrs", {}).get("causal") is not True
    ):
        return False
    append_attrs = append.get("attrs", {})
    attention_attrs = attention.get("attrs", {})
    geometry_keys = (
        "query_heads",
        "key_value_heads",
        "head_width",
        "query_groups_per_kv_head",
    )
    try:
        append_geometry = tuple(int(append_attrs[key]) for key in geometry_keys)
        attention_geometry = tuple(int(attention_attrs[key]) for key in geometry_keys)
        query_heads, kv_heads, head_width, query_groups = attention_geometry
        memory = state_port(circuit, append["state_reads"][0])
        key_shape = list(map(int, memory.get("key_shape_per_token", [])))
        value_shape = list(map(int, memory.get("value_shape_per_token", [])))
        window_size = attention_attrs.get("window_size")
        if window_size is not None:
            window_size = int(window_size)
        scale = float(attention_attrs["scale"])
    except (KeyError, TypeError, ValueError, ModelCompileError):
        return False
    params = attention.get("params", [])
    has_sinks = bool(attention_attrs.get("attention_sinks"))
    if len(params) != int(has_sinks):
        return False
    if params:
        try:
            sink_shape = parameter_shape_for_id(circuit, params[0], tensor_index)
            sink_dtype = parameter_dtype_for_id(circuit, params[0], tensor_index)
            sink_layout = parameter_layout_for_id(circuit, params[0], tensor_index)
        except (KeyError, ModelCompileError):
            return False
        if (
            sink_shape != [query_heads]
            or sink_dtype != "BF16"
            or sink_layout != ROW_MAJOR_LAYOUT
        ):
            return False
    return (
        append_geometry == attention_geometry
        and append_attrs.get("growth") == "per_activation"
        and query_heads > 0
        and kv_heads > 0
        and query_heads % kv_heads == 0
        and query_groups == query_heads // kv_heads
        and head_width >= 2
        and head_width % 2 == 0
        and attention_workgroup_shape(head_width)[1] > 0
        and scale > 0.0
        and (window_size is None or window_size > 0)
        and memory.get("type") == "append_only_attention_memory"
        and memory.get("dtype") == "BF16"
        and memory.get("growth") == "per_activation"
        and memory.get("layout") == "append_only_kv"
        and memory.get("source_layout") == "batch_kvheads_seq_headdim"
        and key_shape == [kv_heads, head_width]
        and value_shape == [kv_heads, head_width]
    )


def parameter_shape_for_id(
    circuit: Json, parameter_id: str, tensor_index: Json
) -> list[int]:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_shape(tensor_index, parameter["tensor"])


def parameter_dtype_for_node(circuit: Json, node: Json, tensor_index: Json) -> str:
    return parameter_dtype_for_id(circuit, node["params"][0], tensor_index)


def parameter_layout_for_node(circuit: Json, node: Json, tensor_index: Json) -> str:
    parameter_id = node["params"][0]
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_layout(tensor_index, parameter["tensor"])


def parameter_layout_for_id(
    circuit: Json, parameter_id: str, tensor_index: Json
) -> str:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_layout(tensor_index, parameter["tensor"])


def parameter_dtype_for_id(circuit: Json, parameter_id: str, tensor_index: Json) -> str:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return str(tensor_index["tensors"][parameter["tensor"]]["dtype"])


def scalar_parameter_read_expression(buffer: str, dtype_token: str) -> str:
    if dtype_token == "f32":
        return f"uintBitsToFloat({buffer}.words[index])"
    if dtype_token == "bf16":
        return f"unpack_bf16({buffer}.words[index >> 1u], index)"
    raise ModelCompileError(
        f"unsupported scalar parameter shader dtype token {dtype_token!r}"
    )


def packed_int4_linear_group_size_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> int:
    weight_id = str(node["params"][0])
    qzeros_id = f"{weight_id}_qzeros"
    scales_id = f"{weight_id}_scales"
    expected_params = [weight_id, qzeros_id, scales_id]
    actual_params = list(node.get("params", []))
    if len(actual_params) == 4:
        expected_params.append(f"{weight_id}_bias")
    if actual_params != expected_params:
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} must bind {expected_params}; "
            f"got {actual_params}"
        )
    weight_ref = circuit["parameters"]["refs"][weight_id]
    weight_info = tensor_index["tensors"][weight_ref["tensor"]]
    quantization = weight_info.get("quantization")
    if not isinstance(quantization, dict):
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} has no compiled quantization metadata"
        )
    if (
        quantization.get("format") != "auto_gptq"
        or int(quantization.get("bits") or 0) != 4
        or int(quantization.get("zero_point_add") or 0) != 1
    ):
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} has unsupported quantization "
            f"{quantization}"
        )
    out_features, in_features = parameter_shape_for_id(circuit, weight_id, tensor_index)
    packed_shape = [int(value) for value in weight_info.get("shape", [])]
    qzeros_shape = parameter_shape_for_id(circuit, qzeros_id, tensor_index)
    scales_shape = parameter_shape_for_id(circuit, scales_id, tensor_index)
    group_size = int(quantization.get("group_size") or 0)
    group_count = (in_features + group_size - 1) // group_size if group_size else 0
    if packed_shape != [in_features // 8, out_features]:
        raise ModelCompileError(
            f"packed INT4 weight shape {packed_shape} does not encode "
            f"{[out_features, in_features]}"
        )
    if qzeros_shape != [group_count, (out_features + 7) // 8]:
        raise ModelCompileError(
            f"packed INT4 zero-point shape {qzeros_shape} is incompatible with "
            f"{[out_features, in_features]}"
        )
    if scales_shape != [group_count, out_features]:
        raise ModelCompileError(
            f"packed INT4 scale shape {scales_shape} is incompatible with "
            f"{[out_features, in_features]}"
        )
    if parameter_dtype_for_id(circuit, qzeros_id, tensor_index) != "I32":
        raise ModelCompileError("packed INT4 zero points must use I32 storage")
    if parameter_dtype_for_id(circuit, scales_id, tensor_index) != "F16":
        raise ModelCompileError("packed INT4 scales must use F16 storage")
    if any(
        parameter_layout_for_id(circuit, parameter_id, tensor_index) != ROW_MAJOR_LAYOUT
        for parameter_id in (weight_id, qzeros_id, scales_id)
    ):
        raise ModelCompileError("packed INT4 parameters must use row-major storage")
    if (
        len(actual_params) == 4
        and parameter_dtype_for_id(circuit, actual_params[3], tensor_index) != "BF16"
    ):
        raise ModelCompileError("packed INT4 linear bias must use BF16 storage")
    return group_size


def packed_linear_quantization_format_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> str:
    weight_id = str(node["params"][0])
    weight_ref = circuit["parameters"]["refs"][weight_id]
    quantization = tensor_index["tensors"][weight_ref["tensor"]].get("quantization")
    if not isinstance(quantization, dict) or not quantization.get("format"):
        raise ModelCompileError(
            f"packed linear node {node['id']!r} has no quantization format"
        )
    return str(quantization["format"])


def compressed_tensors_int4_group_size_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> int:
    weight_id = str(node["params"][0])
    scales_id = f"{weight_id}_scales"
    expected_params = [weight_id, scales_id]
    actual_params = list(node.get("params", []))
    if node["op"] == "linear" and len(actual_params) == 3:
        expected_params.append(f"{weight_id}_bias")
    if actual_params != expected_params:
        raise ModelCompileError(
            f"compressed-tensors INT4 node {node['id']!r} must bind "
            f"{expected_params}; got {actual_params}"
        )
    weight_ref = circuit["parameters"]["refs"][weight_id]
    weight_info = tensor_index["tensors"][weight_ref["tensor"]]
    quantization = weight_info.get("quantization")
    if (
        not isinstance(quantization, dict)
        or quantization.get("format") != "compressed_tensors_pack_quantized"
        or int(quantization.get("bits") or 0) != 4
        or int(quantization.get("signed_offset") or 0) != 8
        or not bool(quantization.get("symmetric"))
    ):
        raise ModelCompileError(
            f"compressed-tensors INT4 node {node['id']!r} has unsupported "
            f"quantization {quantization}"
        )
    out_features, in_features = parameter_shape_for_id(circuit, weight_id, tensor_index)
    packed_shape = [int(value) for value in weight_info.get("shape", [])]
    scales_shape = parameter_shape_for_id(circuit, scales_id, tensor_index)
    group_size = int(quantization.get("group_size") or 0)
    if packed_shape != [out_features, (in_features + 7) // 8]:
        raise ModelCompileError(
            f"compressed-tensors INT4 weight shape {packed_shape} does not encode "
            f"{[out_features, in_features]}"
        )
    if group_size <= 0 or scales_shape != [
        out_features,
        (in_features + group_size - 1) // group_size,
    ]:
        raise ModelCompileError(
            f"compressed-tensors INT4 scale shape {scales_shape} is incompatible "
            f"with {[out_features, in_features]}"
        )
    if parameter_dtype_for_id(circuit, scales_id, tensor_index) != "BF16":
        raise ModelCompileError("compressed-tensors INT4 scales must use BF16 storage")
    if any(
        parameter_layout_for_id(circuit, parameter_id, tensor_index) != ROW_MAJOR_LAYOUT
        for parameter_id in (weight_id, scales_id)
    ):
        raise ModelCompileError(
            "compressed-tensors INT4 parameters must use row-major storage"
        )
    if (
        len(actual_params) == 3
        and parameter_dtype_for_id(circuit, actual_params[2], tensor_index) != "BF16"
    ):
        raise ModelCompileError(
            "compressed-tensors INT4 linear bias must use BF16 storage"
        )
    return group_size


def fp8_block_shape_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> tuple[int, int]:
    weight_id = str(node["params"][0])
    scale_id = f"{weight_id}_scale_inv"
    if len(node.get("params", [])) < 2 or node["params"][1] != scale_id:
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} does not bind {scale_id!r} "
            "immediately after its weight"
        )
    out_features, in_features = parameter_shape_for_id(circuit, weight_id, tensor_index)
    scale_shape = parameter_shape_for_id(circuit, scale_id, tensor_index)
    if len(scale_shape) != 2 or any(value <= 0 for value in scale_shape):
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} has invalid scale shape {scale_shape}"
        )
    if parameter_dtype_for_id(circuit, scale_id, tensor_index) != "BF16":
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} requires a BF16 block scale"
        )
    if parameter_layout_for_id(circuit, scale_id, tensor_index) != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} requires row-major block scales"
        )
    block_rows = (out_features + scale_shape[0] - 1) // scale_shape[0]
    block_columns = (in_features + scale_shape[1] - 1) // scale_shape[1]
    expected_scale_shape = [
        (out_features + block_rows - 1) // block_rows,
        (in_features + block_columns - 1) // block_columns,
    ]
    if scale_shape != expected_scale_shape:
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} scale shape {scale_shape} is not "
            f"a regular block grid for weight shape {[out_features, in_features]}"
        )
    return block_rows, block_columns


def fp8_moe_block_shape_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> tuple[int, int]:
    expected_params = [
        "moe_input",
        "moe_input_scale_inv",
        "moe_output",
        "moe_output_scale_inv",
    ]
    if node.get("params") != expected_params:
        raise ModelCompileError(
            f"FP8 sparse MoE node {node['id']!r} must bind {expected_params}; "
            f"got {node.get('params')}"
        )
    attrs = node["attrs"]
    experts = int(attrs["num_experts"])
    hidden = int(attrs["hidden_size"])
    intermediate = int(attrs["intermediate_size"])
    input_shape = parameter_shape_for_id(circuit, "moe_input", tensor_index)
    output_shape = parameter_shape_for_id(circuit, "moe_output", tensor_index)
    input_scale_shape = parameter_shape_for_id(
        circuit, "moe_input_scale_inv", tensor_index
    )
    output_scale_shape = parameter_shape_for_id(
        circuit, "moe_output_scale_inv", tensor_index
    )
    if input_shape != [experts, intermediate * 2, hidden]:
        raise ModelCompileError(
            f"FP8 sparse MoE input shape {input_shape} does not match "
            f"{[experts, intermediate * 2, hidden]}"
        )
    if output_shape != [experts, hidden, intermediate]:
        raise ModelCompileError(
            f"FP8 sparse MoE output shape {output_shape} does not match "
            f"{[experts, hidden, intermediate]}"
        )
    for parameter_id in ("moe_input_scale_inv", "moe_output_scale_inv"):
        if parameter_dtype_for_id(circuit, parameter_id, tensor_index) != "BF16":
            raise ModelCompileError(
                f"FP8 sparse MoE scale {parameter_id!r} must be BF16"
            )
        if (
            parameter_layout_for_id(circuit, parameter_id, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"FP8 sparse MoE scale {parameter_id!r} must be row-major"
            )
    if len(input_scale_shape) != 3 or input_scale_shape[0] != experts:
        raise ModelCompileError(
            f"FP8 sparse MoE input scale shape is invalid: {input_scale_shape}"
        )
    block_rows, block_columns = regular_block_shape(
        [intermediate * 2, hidden], input_scale_shape[1:]
    )
    expected_output_scale = [
        experts,
        (hidden + block_rows - 1) // block_rows,
        (intermediate + block_columns - 1) // block_columns,
    ]
    if output_scale_shape != expected_output_scale:
        raise ModelCompileError(
            f"FP8 sparse MoE output scale shape {output_scale_shape} does not "
            f"match {expected_output_scale}"
        )
    return block_rows, block_columns


def regular_block_shape(
    matrix_shape: list[int], scale_shape: list[int]
) -> tuple[int, int]:
    if (
        len(matrix_shape) != 2
        or len(scale_shape) != 2
        or any(value <= 0 for value in scale_shape)
    ):
        raise ModelCompileError(
            f"invalid block-scaled matrix shape {matrix_shape} / {scale_shape}"
        )
    rows, columns = matrix_shape
    block_rows = (rows + scale_shape[0] - 1) // scale_shape[0]
    block_columns = (columns + scale_shape[1] - 1) // scale_shape[1]
    expected_scale_shape = [
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    ]
    if scale_shape != expected_scale_shape:
        raise ModelCompileError(
            f"scale shape {scale_shape} is not a regular block grid for {matrix_shape}"
        )
    return block_rows, block_columns


def state_port(circuit: Json, state_id: str) -> Json:
    for port in circuit.get("state_ports", []):
        if port["id"] == state_id:
            return port
    raise ModelCompileError(f"circuit {circuit['id']} has no state port {state_id!r}")


def tensor_shape(tensor_index: Json, tensor: str) -> list[int]:
    info = tensor_index["tensors"][tensor]
    return [int(dim) for dim in info.get("logical_shape", info["shape"])]


def tensor_dtype(tensor_index: Json, tensor: str) -> str:
    return str(tensor_index["tensors"][tensor]["dtype"])


def tensor_byte_count(tensor_index: Json, tensor: str) -> int:
    return int(tensor_index["tensors"][tensor]["byte_count"])


def tensor_layout(tensor_index: Json, tensor: str) -> str:
    return str(tensor_index["tensors"][tensor].get("layout", ROW_MAJOR_LAYOUT))


def dtype_byte_count(dtype: str) -> int:
    byte_counts = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }
    try:
        return byte_counts[dtype]
    except KeyError as error:
        raise ModelCompileError(f"unsupported dtype {dtype!r}") from error


def compiled_model_slug(model_dir: Path) -> str:
    digest = blake2s(
        str(model_dir.resolve()).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"model_{digest}"
