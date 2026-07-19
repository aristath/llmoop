from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from copy import deepcopy
from hashlib import blake2s
from pathlib import Path
from typing import Any, Callable

from llmoop.circuit_lowering import lower_pedalboard
from llmoop.circuit_optimizer import optimize_circuit_for_vulkan
from llmoop.model_compiler import (
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
VULKAN_BF16_ROW_PAIR_LAYOUT = "vulkan_bf16_row_pair_u32"
ROW_MAJOR_LAYOUT = "row_major"
CONFIG_PACKAGE_FILE = "config.json"
RUNTIME_DEFAULT_LOGICAL_DEVICE_ID = "runtime_default"
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
    clean: bool,
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
        clean=clean,
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
    if clean and lowered_dir.exists():
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

    if clean and package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    emit_compile_event(event_sink, "ArtifactWritingStarted", package_dir=str(package_dir))
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
        sampler_local_size_x = 1024
        sampler_shader_file = f"greedy_sampler_f32_{vocab_size}.comp"
        sampler_temperature = 1.0
        sampler_top_k = 1
        sampler_top_p = 1.0
    elif sampler_method == "temperature_top_k_top_p":
        sampler_id = "temperature_top_k_top_p_sampler"
        sampler_temperature = float(sampling["temperature"])
        sampler_top_k = int(sampling["top_k"])
        sampler_top_p = float(sampling["top_p"])
        sampler_local_size_x = max(1, min(64, 4096 // sampler_top_k))
        sampler_shader_file = (
            f"temperature_top_k_top_p_sampler_f32_{vocab_size}"
            f"_t{shader_float_token(sampler_temperature)}"
            f"_k{sampler_top_k}_p{shader_float_token(sampler_top_p)}"
            f"_l{sampler_local_size_x}.comp"
        )
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
    norm_shader_file = rms_norm_shader_file(hidden_size, norm_eps, norm_weight_offset)

    compiled_circuits = {}
    for circuit_ref in lowered_index["graph"]["circuits"]:
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        compiled_circuits[circuit_ref["id"]] = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda node, circuit=circuit: (
                can_fuse_bf16_linear_split(circuit, node, tensor_index)
            ),
            can_fuse_parallel_linears=lambda nodes, circuit=circuit: (
                can_fuse_bf16_parallel_linears(circuit, nodes, tensor_index)
            ),
            can_fuse_parallel_head_norm_rope=lambda branches, circuit=circuit: (
                can_fuse_bf16_parallel_head_norm_rope(
                    circuit, branches, tensor_index
                )
            ),
            can_fuse_dual_linear_silu_multiply=lambda projection, multiply, circuit=circuit: (
                can_fuse_bf16_dual_linear_silu_multiply(
                    circuit, projection, multiply, tensor_index
                )
            ),
        )
    pedal_executions = pedal_execution_specs(
        lowered_index=lowered_index,
        compiled_circuits=compiled_circuits,
        tensor_index=tensor_index,
        dimensions=dimensions,
    )
    copy_shader_templates(
        shader_source_dir,
        package_dir / "shaders",
        required_shader_files(
            pedal_executions,
            embedding_shader_file=embedding_shader_file,
            projection_shader_file=projection_shader_file,
            norm_shader_file=norm_shader_file,
            sampler_shader_file=sampler_shader_file,
        ),
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
    for execution in pedal_executions:
        for kernel in execution["kernels"]:
            kernel["shader_path"] = compiled_shader_path(kernel["shader_path"])
    placement = package_placement()

    return {
        "schema": PACKAGE_SCHEMA,
        "package_id": package_id,
        "device_id": RUNTIME_DEFAULT_LOGICAL_DEVICE_ID,
        "placement": placement,
        "circuit_graph": package_circuit_graph(
            lowered_index, lowered_dir, compiled_circuits
        ),
        "tensor_index_path": "tensors.json",
        "config_path": CONFIG_PACKAGE_FILE,
        "tokenizer": tokenizer_manifest,
        "activation_element_bytes": dtype_bytes,
        "max_context_activations": max_context_activations,
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
                    "output_transducer.embedding_norm",
                    "output_transducer.tied_output_projection",
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
                "local_size_x": sampler_local_size_x,
            },
            "shader_path": compiled_shader_path(f"shaders/{sampler_shader_file}"),
        },
        "pedal_executions": pedal_executions,
    }


def package_placement() -> Json:
    return {
        "schema": "llmoop.stream_circuit_placement.v1",
        "default_device_id": RUNTIME_DEFAULT_LOGICAL_DEVICE_ID,
        "pedal_devices": {},
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
                "implementation": circuit_ref["implementation"],
                "behavioral_role": circuit_ref["behavioral_role"],
                "circuit": deepcopy(compiled_circuits[circuit_ref["id"]]),
                "params": read_json(lowered_dir / circuit_ref["params"]),
                "state": read_json(lowered_dir / circuit_ref["state"]),
            }
        )

    return {
        "wiring": graph["wiring"],
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
                {
                    "execution_index": index,
                    "node_id": node["id"],
                    "op": node["op"],
                    "shader_path": f"shaders/{shader_file}",
                    "local_size_x": local_size_x_for_node(node),
                    "workgroup_count_x": workgroup_count_x_for_node(
                        circuit, node, tensor_index
                    ),
                }
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
            "paired"
            if layouts == {VULKAN_BF16_ROW_PAIR_LAYOUT}
            else "row_major"
        )
        input_width = input_widths.pop()
        return (
            f"parallel_linear_{branch_count}way_{layout_token}_bf16_{input_width}x"
            + "_".join(map(str, output_widths))
            + ".comp"
        )
    if op == "dual_linear_silu_multiply":
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("params", [])) != 2
            or len(node.get("outputs", [])) != 1
        ):
            raise ModelCompileError(
                f"dual-linear SiLU node {node['id']!r} has invalid bindings"
            )
        shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        ]
        dtypes = {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        layouts = {
            parameter_layout_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        supported_layouts = {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
        activated_input_index = int(node["attrs"]["activated_input_index"])
        if (
            len(shapes) != 2
            or shapes[0] != shapes[1]
            or len(shapes[0]) != 2
            or any(
                int(dimension) <= 0 or int(dimension) % 2
                for dimension in shapes[0]
            )
            or dtypes != {"BF16"}
            or len(layouts) != 1
            or not layouts <= supported_layouts
            or activated_input_index not in {0, 1}
        ):
            raise ModelCompileError(
                f"dual-linear SiLU node {node['id']!r} has incompatible projections"
            )
        output_width, input_width = map(int, shapes[0])
        layout_token = (
            "paired"
            if layouts == {VULKAN_BF16_ROW_PAIR_LAYOUT}
            else "row_major"
        )
        return (
            f"dual_linear_silu_multiply_{layout_token}_bf16_"
            f"{input_width}x{output_width}_a{activated_input_index}.comp"
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
        return "silu_multiply_bf16.comp"
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
        if (
            any(len(values) != 1 for values in common_fields.values())
            or any(
                int(norm["head_count"]) != int(rope["head_count"])
                for norm, rope in zip(norms, ropes, strict=True)
            )
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
    if node["op"] == "dual_linear_silu_multiply":
        output_width = int(
            parameter_shape_for_id(circuit, node["params"][0], tensor_index)[0]
        )
        return (output_width + 1) // 2
    if node["op"] == "parallel_head_norm_rope_2way":
        return sum(
            int(branch["norm"]["head_count"])
            for branch in node["attrs"]["branches"]
        )
    if node["op"] in {"parallel_linear_2way", "parallel_linear_3way"}:
        return sum(
            (int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0]) + 1)
            // 2
            for parameter_id in node["params"]
        )
    if node["op"] in {"linear", "linear_residual", "linear_split_3way"}:
        out_features, _ = parameter_shape_for_node(circuit, node, tensor_index)
        # One workgroup collaboratively computes and packs two BF16 output rows.
        return (int(out_features) + 1) // 2
    if node["op"] == "scaled_dot_product_attention":
        return int(node["attrs"]["query_heads"])
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
    if node["op"] == "scaled_dot_product_attention":
        return attention_workgroup_shape(int(node["attrs"]["head_width"]))[0]
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_head_width"])
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["block_width"])
    if node["op"] == "sparse_moe_experts":
        return 256
    return 64


def attention_workgroup_shape(head_width: int) -> tuple[int, int]:
    padded_head_width = ((head_width + 63) // 64) * 64
    tile_tokens = 1024 // padded_head_width
    return padded_head_width * tile_tokens, tile_tokens


def required_shader_files(
    pedal_executions: list[Json],
    *,
    embedding_shader_file: str,
    projection_shader_file: str,
    norm_shader_file: str,
    sampler_shader_file: str,
) -> set[str]:
    return {
        norm_shader_file,
        sampler_shader_file,
        embedding_shader_file,
        projection_shader_file,
        *(
            kernel["shader_path"].removeprefix("shaders/")
            for pedal in pedal_executions
            for kernel in pedal["kernels"]
        ),
    }


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
        r"parallel_linear_([23])way_(paired|row_major)_bf16_(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel_linear is not None:
        branch_count = int(parallel_linear.group(1))
        weight_layout = parallel_linear.group(2)
        input_size = int(parallel_linear.group(3))
        output_widths = [
            int(width)
            for width in parallel_linear.groups()[3:]
            if width is not None
        ]
        if (
            len(output_widths) != branch_count
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
            threshold = " + ".join(
                [*consumed_words, f"OUTPUT_{label}_WORDS"]
            )
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
            "\n    return weight_"
            + labels[-1].lower()
            + ".words[weight_index];"
        )
        output_writes = "\n".join(
            f"    if (branch == {index}u) {{ output_{label.lower()}.words[local_word_index] = packed; return; }}"
            for index, label in enumerate(labels[:-1])
        )
        output_writes += (
            "\n    output_"
            + labels[-1].lower()
            + ".words[local_word_index] = packed;"
        )
        return render_shader_template(
            source_dir,
            "parallel_linear_bf16.comp.template",
            {
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

    shaped_templates = (
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
            r"dual_linear_silu_multiply_(paired|row_major)_bf16_"
            r"(\d+)x(\d+)_a([01])\.comp",
            "dual_linear_silu_multiply_bf16.comp.template",
            (
                "WEIGHT_LAYOUT",
                "INPUT_SIZE",
                "OUTPUT_SIZE",
                "ACTIVATED_INPUT_INDEX",
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
            r"temperature_top_k_top_p_sampler_f32_(\d+)_t([0-9eE+.-]+)_k(\d+)_p([0-9eE+.-]+)_l(\d+)\.comp",
            "temperature_top_k_top_p_sampler_f32.comp.template",
            ("VOCAB_SIZE", "TEMPERATURE", "TOP_K", "TOP_P", "LOCAL_SIZE_X"),
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
    for circuit_ref in lowered_index["graph"]["circuits"]:
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
            header_bytes = write_compiled_composite_tensor(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
        else:
            source = Path(info["source_file"])
            if not source.is_file():
                raise ModelCompileError(f"tensor source file does not exist: {source}")
            header_bytes = write_compiled_tensor(
                tensor_name=tensor_name,
                info=info,
                source=source,
                destination=destination,
                layout=layout,
            )
        relative_destination = relative_json_path(package_dir, destination)
        info["source_file"] = relative_destination
        info["data_offsets"] = [0, int(info["byte_count"])]
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
    required_files = (
        package_dir / str(manifest.get("config_path", "")),
        package_dir / str(manifest.get("tensor_index_path", "")),
    )
    for path in required_files:
        if not path.is_file():
            raise ModelCompileError(f"compiled package is missing required artifact {path}")

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not tokenizer.get("path"):
        raise ModelCompileError("compiled package does not declare tokenizer artifacts")
    tokenizer_dir = package_dir / str(tokenizer["path"])
    for filename in tokenizer.get("files", []):
        path = tokenizer_dir / str(filename)
        if not path.is_file():
            raise ModelCompileError(f"compiled package is missing tokenizer artifact {path}")

    tensor_index = read_json(package_dir / str(manifest["tensor_index_path"]))
    for tensor_name, info in tensor_index.get("tensors", {}).items():
        source = package_dir / str(info.get("source_file", ""))
        if not source.is_file():
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} references missing artifact {source}"
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
        raise ModelCompileError("compiled package does not reference any shader artifacts")
    for relative_path in sorted(shader_paths):
        shader = package_dir / relative_path
        if not shader.is_file():
            raise ModelCompileError(f"compiled package references missing shader {shader}")
        payload = shader.read_bytes()
        if len(payload) < 4 or payload[:4] != b"\x03\x02#\x07":
            raise ModelCompileError(f"compiled package shader is not valid SPIR-V: {shader}")


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
) -> int:
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
            )
        else:
            copy_exact_bytes(source_handle, destination_handle, byte_count)
    return len(header_payload)


def write_compiled_composite_tensor(
    *,
    tensor_name: str,
    info: Json,
    destination: Path,
    layout: str,
) -> int:
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
                copy_exact_bytes(source_handle, destination_handle, part_bytes)
            written += part_bytes
    if written != byte_count:
        raise ModelCompileError(
            f"composite tensor {tensor_name!r} wrote {written} bytes; expected {byte_count}"
        )
    return len(header_payload)


def write_bf16_row_pair_tensor(
    source_handle: Any,
    destination_handle: Any,
    *,
    rows: int,
    columns: int,
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
        destination_handle.write(paired.tobytes())


def copy_exact_bytes(
    source_handle: Any, destination_handle: Any, byte_count: int
) -> None:
    remaining = byte_count
    while remaining:
        chunk = source_handle.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ModelCompileError(
                "unexpected end of tensor source while compiling package"
            )
        destination_handle.write(chunk)
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
    layouts = {
        parameter_layout_for_node(circuit, node, tensor_index) for node in nodes
    }
    return (
        len(nodes) in {2, 3}
        and all(
            len(shape) == 2
            and all(
                int(dimension) > 0 and int(dimension) % 2 == 0
                for dimension in shape
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
            int(node["attrs"]["head_width"])
            for branch in branches
            for node in branch
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
            int(norm["attrs"]["head_count"])
            == int(rope["attrs"]["head_count"])
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


def can_fuse_bf16_dual_linear_silu_multiply(
    circuit: Json,
    projection: Json,
    multiply: Json,
    tensor_index: Json,
) -> bool:
    if multiply.get("inputs") is None or len(projection.get("params", [])) != 2:
        return False
    shapes = [
        parameter_shape_for_id(circuit, parameter_id, tensor_index)
        for parameter_id in projection["params"]
    ]
    layouts = {
        parameter_layout_for_id(circuit, parameter_id, tensor_index)
        for parameter_id in projection["params"]
    }
    return (
        len(shapes) == 2
        and shapes[0] == shapes[1]
        and len(shapes[0]) == 2
        and all(
            int(dimension) > 0 and int(dimension) % 2 == 0
            for dimension in shapes[0]
        )
        and all(
            parameter_dtype_for_id(circuit, parameter_id, tensor_index) == "BF16"
            for parameter_id in projection["params"]
        )
        and len(layouts) == 1
        and layouts <= {ROW_MAJOR_LAYOUT, VULKAN_BF16_ROW_PAIR_LAYOUT}
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


def parameter_layout_for_id(circuit: Json, parameter_id: str, tensor_index: Json) -> str:
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
    out_features, in_features = parameter_shape_for_id(
        circuit, weight_id, tensor_index
    )
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
        parameter_layout_for_id(circuit, parameter_id, tensor_index)
        != ROW_MAJOR_LAYOUT
        for parameter_id in (weight_id, qzeros_id, scales_id)
    ):
        raise ModelCompileError("packed INT4 parameters must use row-major storage")
    if len(actual_params) == 4 and parameter_dtype_for_id(
        circuit, actual_params[3], tensor_index
    ) != "BF16":
        raise ModelCompileError("packed INT4 linear bias must use BF16 storage")
    return group_size


def packed_linear_quantization_format_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> str:
    weight_id = str(node["params"][0])
    weight_ref = circuit["parameters"]["refs"][weight_id]
    quantization = tensor_index["tensors"][weight_ref["tensor"]].get(
        "quantization"
    )
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
    out_features, in_features = parameter_shape_for_id(
        circuit, weight_id, tensor_index
    )
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
        parameter_layout_for_id(circuit, parameter_id, tensor_index)
        != ROW_MAJOR_LAYOUT
        for parameter_id in (weight_id, scales_id)
    ):
        raise ModelCompileError(
            "compressed-tensors INT4 parameters must use row-major storage"
        )
    if len(actual_params) == 3 and parameter_dtype_for_id(
        circuit, actual_params[2], tensor_index
    ) != "BF16":
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
    out_features, in_features = parameter_shape_for_id(
        circuit, weight_id, tensor_index
    )
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
        if parameter_layout_for_id(circuit, parameter_id, tensor_index) != ROW_MAJOR_LAYOUT:
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


def regular_block_shape(matrix_shape: list[int], scale_shape: list[int]) -> tuple[int, int]:
    if len(matrix_shape) != 2 or len(scale_shape) != 2 or any(
        value <= 0 for value in scale_shape
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
    return [
        int(dim) for dim in info.get("logical_shape", info["shape"])
    ]


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
