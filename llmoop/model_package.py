from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from copy import deepcopy
from hashlib import blake2s
from pathlib import Path
from typing import Any

from llmoop.circuit_lowering import lower_pedalboard
from llmoop.circuit_optimizer import optimize_circuit_for_vulkan
from llmoop.model_compiler import (
    PACKAGE_SCHEMA,
    CompiledModelReport,
    Json,
    ModelCompileError,
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
) -> CompiledModelReport:
    slug = compiled_model_slug(model_dir)
    transpiled_dir = transpiled_dir or Path("transpiled") / slug
    lowered_dir = lowered_dir or Path("lowered") / slug
    package_dir = package_dir or Path("packages") / slug

    structure = transpile_model(model_dir, transpiled_dir, clean=clean)
    if clean and lowered_dir.exists():
        shutil.rmtree(lowered_dir)
    lowered = lower_pedalboard(transpiled_dir, lowered_dir)
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
    write_runtime_config_package(model_graph, package_dir)
    tokenizer_manifest = copy_tokenizer_package(
        model_dir, package_dir / TOKENIZER_PACKAGE_DIR
    )
    packaged_tensor_index = copy_tensor_package(tensor_index, package_dir)
    package_manifest = build_vulkan_resident_greedy_package_manifest(
        model_graph=model_graph,
        tensor_index=packaged_tensor_index,
        lowered_index=lowered["index"],
        lowered_dir=lowered_dir,
        package_dir=package_dir,
        package_id=f"{slug}_vulkan_resident_greedy",
        shader_source_dir=shader_source_dir,
        tokenizer_manifest=tokenizer_manifest,
    )
    package_manifest_path = package_dir / "vulkan_resident_greedy_package.json"
    write_json(package_manifest_path, package_manifest)

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


def build_vulkan_resident_greedy_package_manifest(
    *,
    model_graph: Json,
    tensor_index: Json,
    lowered_index: Json,
    lowered_dir: Path,
    package_dir: Path,
    package_id: str,
    shader_source_dir: Path,
    tokenizer_manifest: Json,
) -> Json:
    dimensions = model_graph["dimensions"]
    hidden_size = int(dimensions["hidden_size"])
    vocab_size = int(dimensions["vocab_size"])
    norm_eps = float(model_graph["numerics"]["rms_norm_eps"])
    norm_weight_offset = float(model_graph["numerics"]["rms_norm_weight_offset"])
    embedding_scale = float(model_graph["numerics"]["embedding_scale"])
    logits_scale = float(model_graph["numerics"]["logits_scale"])
    max_context_activations = int(dimensions["max_position_embeddings"])
    dtype = "BF16"
    dtype_bytes = dtype_byte_count(dtype)
    frame_bytes = hidden_size * dtype_bytes
    logits_bytes = vocab_size * dtype_byte_count("F32")

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

    compiled_circuits = {
        circuit_ref["id"]: optimize_circuit_for_vulkan(
            read_json(lowered_dir / circuit_ref["circuit"])
        )
        for circuit_ref in lowered_index["graph"]["circuits"]
    }
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
            dimensions,
            pedal_executions,
            embedding_shader_file=embedding_shader_file,
            projection_shader_file=projection_shader_file,
            norm_shader_file=norm_shader_file,
        ),
    )
    compile_shader_artifacts(package_dir / "shaders")
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
                "sampler_id": "greedy_sampler",
                "logits_byte_capacity": logits_bytes,
                "output_byte_capacity": 16,
                "local_size_x": 1024,
            },
            "shader_path": compiled_shader_path(
                f"shaders/greedy_sampler_f32_{vocab_size}.comp"
            ),
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
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        prefix = (
            "linear_paired_bf16"
            if layout == VULKAN_BF16_ROW_PAIR_LAYOUT
            else "linear_bf16"
        )
        return f"{prefix}_{in_features}x{out_features}.comp"
    if op == "linear_residual":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
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
        part_width = int(node["attrs"]["part_width"])
        return f"split_bf16_{len(node['outputs'])}x{part_width}.comp"
    if op == "multiply":
        element_count = (
            intermediate_size if node["id"] == "ffn_gate_multiply" else hidden_size
        )
        return f"multiply_bf16_{element_count}.comp"
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
    if op == "silu_multiply":
        return "silu_multiply_bf16.comp"
    if op == "sigmoid_multiply":
        return "sigmoid_multiply_bf16.comp"
    if op == "rms_norm_per_head":
        heads = (
            node["attrs"]["query_heads"]
            if node["id"].startswith("q_")
            else node["attrs"]["key_value_heads"]
        )
        return (
            f"rms_norm_per_head_bf16_{heads}x{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}"
            f"_offset{shader_float_token(float(node['attrs']['weight_offset']))}.comp"
        )
    if op == "rotary_position_embedding":
        heads = (
            node["attrs"]["query_heads"]
            if node["id"].startswith("q_")
            else node["attrs"]["key_value_heads"]
        )
        binding = stream_control_binding_for_node(circuit, node)
        rope_layout = "interleaved" if node["attrs"]["interleaved"] else "half"
        return (
            f"rotary_bf16_{heads}x{node['attrs']['head_width']}"
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
        return (
            "gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
            f"__sc{binding}.comp"
        )
    if op == "causal_conv1d_silu":
        return (
            f"causal_conv1d_silu_bf16_c{node['attrs']['channels']}"
            f"_k{node['attrs']['kernel_width']}.comp"
        )
    if op == "gated_delta_step":
        attrs = node["attrs"]
        expected_dtypes = {
            "delta_a_log": "F32",
            "delta_dt_bias": "BF16",
            "delta_norm": "F32",
        }
        for parameter_id, expected_dtype in expected_dtypes.items():
            actual_dtype = parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            if actual_dtype != expected_dtype:
                raise ModelCompileError(
                    f"gated-delta parameter {parameter_id} has dtype {actual_dtype}; "
                    f"expected {expected_dtype}"
                )
        return (
            f"gated_delta_step_k{attrs['key_heads']}x{attrs['key_head_width']}"
            f"_v{attrs['value_heads']}x{attrs['value_head_width']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}.comp"
        )
    if op == "moe_topk":
        attrs = node["attrs"]
        return (
            f"moe_topk_bf16_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        )
    if op == "sparse_moe_experts":
        attrs = node["attrs"]
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
    if node["op"] in {"linear", "linear_residual"}:
        out_features, _ = parameter_shape_for_node(circuit, node, tensor_index)
        # One workgroup collaboratively computes and packs two BF16 output rows.
        return (int(out_features) + 1) // 2
    if node["op"] == "scaled_dot_product_attention":
        return int(node["attrs"]["query_heads"])
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_heads"])
    if node["op"] == "sparse_moe_experts":
        return int(node["attrs"]["num_experts"])
    if node["op"] in {"rms_norm_per_head", "rotary_position_embedding"}:
        return int(
            node["attrs"]["query_heads"]
            if node["id"].startswith("q_")
            else node["attrs"]["key_value_heads"]
        )
    return 1


def local_size_x_for_node(node: Json) -> int:
    # The tiled attention kernel maps sixteen 64-wide head reductions onto one
    # workgroup. This execution geometry belongs to the compiled pedal package.
    if node["op"] == "scaled_dot_product_attention":
        return 1024
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_head_width"])
    if node["op"] == "sparse_moe_experts":
        return 256
    return 64


def required_shader_files(
    dimensions: Json,
    pedal_executions: list[Json],
    *,
    embedding_shader_file: str,
    projection_shader_file: str,
    norm_shader_file: str,
) -> set[str]:
    vocab_size = int(dimensions["vocab_size"])

    return {
        norm_shader_file,
        f"greedy_sampler_f32_{vocab_size}.comp",
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

    shaped_templates = (
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
            r"rotary_bf16_(\d+)x(\d+)_r(\d+)_theta([0-9eE+.-]+)_(half|interleaved)\.comp",
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
                replacements["ROPE_INTERLEAVED"] = (
                    "true"
                    if replacements.pop("ROPE_LAYOUT") == "interleaved"
                    else "false"
                )
            return render_shader_template(source_dir, template, replacements)

    attention_shape = re.fullmatch(
        r"gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if attention_shape is not None:
        query_heads, kv_heads, head_width = map(int, attention_shape.groups()[:3])
        if query_heads % kv_heads != 0:
            raise ModelCompileError(
                f"query head count {query_heads} is not divisible by KV head count {kv_heads}"
            )
        if head_width < 2 or head_width % 2 != 0 or 1024 % head_width != 0:
            raise ModelCompileError(
                f"attention head width {head_width} cannot be tiled into a 1024-invocation pedal"
            )
        return render_shader_template(
            source_dir,
            "gqa_attention_bf16.comp.template",
            {
                "QUERY_HEADS": str(query_heads),
                "KV_HEADS": str(kv_heads),
                "QUERY_GROUPS_PER_KV_HEAD": str(query_heads // kv_heads),
                "HEAD_WIDTH": str(head_width),
                "TILE_TOKENS": str(1024 // head_width),
                "ATTENTION_SCALE": attention_shape.group(4),
            },
        )

    gated_delta_shape = re.fullmatch(
        r"gated_delta_step_k(\d+)x(\d+)_v(\d+)x(\d+)_eps([0-9eE+.-]+)\.comp",
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
                "NORM_EPS": gated_delta_shape.group(5),
            },
        )

    moe_topk_shape = re.fullmatch(r"moe_topk_bf16_e(\d+)_k(\d+)\.comp", shader_file)
    if moe_topk_shape is not None:
        num_experts, experts_per_token = map(int, moe_topk_shape.groups())
        if not 0 < experts_per_token <= num_experts <= 64:
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
        if not 0 < experts_per_token <= num_experts <= 64:
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


def compile_shader_artifacts(shader_dir: Path) -> None:
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
    for source in sources:
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
        if producer.get("state_writes")
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


def copy_tensor_package(tensor_index: Json, package_dir: Path) -> Json:
    weights_dir = package_dir / WEIGHTS_PACKAGE_DIR
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not tensor_index["tensors"]:
        raise ModelCompileError("tensor index does not declare any source_file entries")

    packaged = deepcopy(tensor_index)
    compiled_sources = []
    for tensor_name, info in sorted(packaged["tensors"].items()):
        source = Path(info["source_file"])
        if not source.is_file():
            raise ModelCompileError(f"tensor source file does not exist: {source}")
        layout = compiled_tensor_layout(info)
        digest = blake2s(tensor_name.encode("utf-8"), digest_size=8).hexdigest()
        destination = weights_dir / f"tensor_{digest}.safetensors"
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


def compiled_tensor_layout(info: Json) -> str:
    shape = [int(value) for value in info.get("shape", [])]
    if (
        info.get("dtype") == "BF16"
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
    source_header_bytes, _source_header = read_safetensors_header(source)
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


def parameter_layout_for_node(circuit: Json, node: Json, tensor_index: Json) -> str:
    parameter_id = node["params"][0]
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_layout(tensor_index, parameter["tensor"])


def parameter_dtype_for_id(circuit: Json, parameter_id: str, tensor_index: Json) -> str:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return str(tensor_index["tensors"][parameter["tensor"]]["dtype"])


def state_port(circuit: Json, state_id: str) -> Json:
    for port in circuit.get("state_ports", []):
        if port["id"] == state_id:
            return port
    raise ModelCompileError(f"circuit {circuit['id']} has no state port {state_id!r}")


def tensor_shape(tensor_index: Json, tensor: str) -> list[int]:
    return [int(dim) for dim in tensor_index["tensors"][tensor]["shape"]]


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
