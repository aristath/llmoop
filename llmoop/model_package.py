from __future__ import annotations

import shutil
from copy import deepcopy
from hashlib import blake2s
from pathlib import Path

from llmoop.circuit_lowering import lower_pedalboard
from llmoop.model_compiler import (
    PACKAGE_SCHEMA,
    CompiledModelReport,
    Json,
    ModelCompileError,
    read_json,
    relative_json_path,
    write_json,
)
from llmoop.model_transpiler import transpile_model


TOKENIZER_PACKAGE_DIR = "tokenizer"
WEIGHTS_PACKAGE_DIR = "weights"
CONFIG_PACKAGE_FILE = "config.json"
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
    clean: bool,
    shader_source_dir: Path,
    default_dynamic_state_capacity_activations: int,
) -> CompiledModelReport:
    slug = compiled_model_slug(model_dir)
    transpiled_dir = transpiled_dir or Path("transpiled") / slug
    lowered_dir = lowered_dir or Path("lowered") / slug

    structure = transpile_model(model_dir, transpiled_dir, clean=clean)
    lowered = lower_pedalboard(transpiled_dir, lowered_dir)
    tensor_index = read_json(transpiled_dir / "tensors.json")
    model_graph = read_json(transpiled_dir / "model.json")
    copy_config_package(model_dir, lowered_dir)
    tokenizer_manifest = copy_tokenizer_package(model_dir, lowered_dir / TOKENIZER_PACKAGE_DIR)
    packaged_tensor_index = copy_tensor_package(tensor_index, lowered_dir)
    package_manifest = build_vulkan_resident_greedy_package_manifest(
        model_graph=model_graph,
        tensor_index=packaged_tensor_index,
        lowered_index=lowered["index"],
        lowered_dir=lowered_dir,
        package_id=f"{slug}_vulkan_resident_greedy",
        shader_source_dir=shader_source_dir,
        default_dynamic_state_capacity_activations=default_dynamic_state_capacity_activations,
        tokenizer_manifest=tokenizer_manifest,
    )
    package_manifest_path = lowered_dir / "vulkan_resident_greedy_package.json"
    write_json(package_manifest_path, package_manifest)

    return CompiledModelReport(
        model_dir=model_dir,
        transpiled_dir=transpiled_dir,
        lowered_dir=lowered_dir,
        package_manifest=package_manifest_path,
        model_type=structure.model_type or "unknown",
        circuit_count=lowered["index"]["summary"]["circuit_count"],
        shader_count=len(list((lowered_dir / "shaders").glob("*.comp"))),
    )


def build_vulkan_resident_greedy_package_manifest(
    *,
    model_graph: Json,
    tensor_index: Json,
    lowered_index: Json,
    lowered_dir: Path,
    package_id: str,
    shader_source_dir: Path,
    default_dynamic_state_capacity_activations: int,
    tokenizer_manifest: Json,
) -> Json:
    dimensions = model_graph["dimensions"]
    hidden_size = int(dimensions["hidden_size"])
    vocab_size = int(dimensions["vocab_size"])
    dtype = "BF16"
    dtype_bytes = dtype_byte_count(dtype)
    frame_bytes = hidden_size * dtype_bytes
    logits_bytes = vocab_size * dtype_byte_count("F32")

    embed_tensor = model_graph["graph"]["input_transducer"]["params"]["weight"]["tensor"]
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

    copy_shader_templates(
        shader_source_dir,
        lowered_dir / "shaders",
        required_shader_files(dimensions),
    )

    reusable_kernel_shaders, cap8_overrides = reusable_shader_refs(
        lowered_index=lowered_index,
        lowered_dir=lowered_dir,
        tensor_index=tensor_index,
        dimensions=dimensions,
    )

    return {
        "schema": PACKAGE_SCHEMA,
        "package_id": package_id,
        "device_id": "gpu0",
        "circuit_index_path": "pedalboard.circuits.json",
        "tensor_index_path": "tensors.json",
        "config_path": CONFIG_PACKAGE_FILE,
        "tokenizer": tokenizer_manifest,
        "activation_element_bytes": dtype_bytes,
        "dynamic_state_capacity_activations": default_dynamic_state_capacity_activations,
        "input_transducer": {
            "spec": {
                "transducer_id": "input_transducer.token_embedding",
                "parameter_tensor": embed_tensor,
                "parameter_dtype": dtype,
                "parameter_shape": tensor_shape(tensor_index, embed_tensor),
                "parameter_byte_capacity": tensor_byte_count(tensor_index, embed_tensor),
                "output_signal_id": "input_frame",
                "output_frame_byte_capacity": frame_bytes,
                "output_frame_word_count": frame_bytes // 4,
                "local_size_x": 256,
            },
            "shader_path": f"shaders/embedding_lookup_bf16_{vocab_size}x{hidden_size}.comp",
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
                "norm_parameter_byte_capacity": tensor_byte_count(tensor_index, norm_tensor),
                "projection_parameter_tensor": projection_tensor,
                "projection_parameter_dtype": dtype,
                "projection_parameter_shape": tensor_shape(tensor_index, projection_tensor),
                "projection_parameter_byte_capacity": tensor_byte_count(
                    tensor_index, projection_tensor
                ),
                "input_frame_byte_capacity": frame_bytes,
                "normalized_frame_byte_capacity": frame_bytes,
                "logits_byte_capacity": logits_bytes,
                "projection_work_items": vocab_size,
                "norm_local_size_x": 64,
                "projection_local_size_x": 64,
            },
            "embedding_norm_shader_path": "shaders/rms_norm_bf16_serial.comp",
            "projection_shader_path": f"shaders/tied_output_projection_bf16_{vocab_size}x{hidden_size}_to_f32.comp",
        },
        "sampler": {
            "spec": {
                "sampler_id": "greedy_sampler",
                "logits_byte_capacity": logits_bytes,
                "output_byte_capacity": 16,
                "local_size_x": 64,
            },
            "shader_path": f"shaders/greedy_sampler_f32_{vocab_size}.comp",
        },
        "reusable_kernel_shaders": reusable_kernel_shaders,
        "capacity_profiles": [
            {
                "min_dynamic_state_capacity_activations": 5,
                "max_dynamic_state_capacity_activations": 8,
                "reusable_kernel_shader_overrides": cap8_overrides,
            }
        ],
    }


def reusable_shader_refs(
    *,
    lowered_index: Json,
    lowered_dir: Path,
    tensor_index: Json,
    dimensions: Json,
) -> tuple[list[Json], list[Json]]:
    refs: list[Json] = []
    cap8_overrides: list[Json] = []

    for circuit_ref in lowered_index["graph"]["circuits"]:
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        for node in circuit["nodes"]:
            shader_file = shader_file_for_node(
                circuit,
                node,
                tensor_index,
                dimensions,
                attention_capacity=4,
            )
            refs.append(
                {
                    "pedal_id": circuit_ref["id"],
                    "node_id": node["id"],
                    "shader_path": f"shaders/{shader_file}",
                }
            )
            if node["op"] == "scaled_dot_product_attention":
                cap8_overrides.append(
                    {
                        "pedal_id": circuit_ref["id"],
                        "node_id": node["id"],
                        "shader_path": (
                            "shaders/"
                            + shader_file_for_node(
                                circuit,
                                node,
                                tensor_index,
                                dimensions,
                                attention_capacity=8,
                            )
                        ),
                    }
                )

    return refs, cap8_overrides


def shader_file_for_node(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    dimensions: Json,
    *,
    attention_capacity: int,
) -> str:
    hidden_size = int(dimensions["hidden_size"])
    intermediate_size = int(dimensions["intermediate_size"])
    op = node["op"]

    if op == "rms_norm":
        return "rms_norm_bf16_serial.comp"
    if op == "linear":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        return f"linear_bf16_{in_features}x{out_features}.comp"
    if op == "split":
        return f"split_bf16_{hidden_size * 3}_to_3x{hidden_size}.comp"
    if op == "multiply":
        element_count = intermediate_size if node["id"] == "ffn_gate_multiply" else hidden_size
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
    if op == "silu":
        return f"silu_bf16_{intermediate_size}.comp"
    if op == "rms_norm_per_head":
        heads = (
            node["attrs"]["query_heads"]
            if node["id"].startswith("q_")
            else node["attrs"]["key_value_heads"]
        )
        return f"rms_norm_per_head_bf16_{heads}x{node['attrs']['head_width']}.comp"
    if op == "rotary_position_embedding":
        heads = (
            node["attrs"]["query_heads"]
            if node["id"].startswith("q_")
            else node["attrs"]["key_value_heads"]
        )
        return f"rotary_bf16_{heads}x{node['attrs']['head_width']}.comp"
    if op == "append_state_update":
        return f"append_kv_state_bf16_{node['attrs']['key_value_heads']}x{node['attrs']['head_width']}.comp"
    if op == "scaled_dot_product_attention":
        attrs = node["attrs"]
        bucket = 4 if attention_capacity <= 4 else 8
        return (
            "gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}_cap{bucket}.comp"
        )

    raise ModelCompileError(f"no Vulkan shader selector for op {op!r} in node {node['id']!r}")


def required_shader_files(dimensions: Json) -> set[str]:
    hidden_size = int(dimensions["hidden_size"])
    intermediate_size = int(dimensions["intermediate_size"])
    vocab_size = int(dimensions["vocab_size"])
    query_heads = int(dimensions["num_attention_heads"])
    kv_heads = int(dimensions["num_key_value_heads"])
    head_width = hidden_size // query_heads
    conv_l_cache = int(dimensions["conv_l_cache"])

    return {
        f"embedding_lookup_bf16_{vocab_size}x{hidden_size}.comp",
        "rms_norm_bf16_serial.comp",
        f"tied_output_projection_bf16_{vocab_size}x{hidden_size}_to_f32.comp",
        f"greedy_sampler_f32_{vocab_size}.comp",
        f"linear_bf16_{hidden_size}x{hidden_size}.comp",
        f"linear_bf16_{hidden_size}x{hidden_size * 3}.comp",
        f"linear_bf16_{hidden_size}x{hidden_size // 2}.comp",
        f"linear_bf16_{hidden_size}x{intermediate_size}.comp",
        f"linear_bf16_{intermediate_size}x{hidden_size}.comp",
        f"split_bf16_{hidden_size * 3}_to_3x{hidden_size}.comp",
        f"multiply_bf16_{hidden_size}.comp",
        f"multiply_bf16_{intermediate_size}.comp",
        f"rolling_state_update_bf16_{conv_l_cache}x{hidden_size}.comp",
        f"depthwise_conv1d_bf16_{conv_l_cache}x{hidden_size}.comp",
        f"add_bf16_{hidden_size}.comp",
        f"silu_bf16_{intermediate_size}.comp",
        f"rms_norm_per_head_bf16_{query_heads}x{head_width}.comp",
        f"rms_norm_per_head_bf16_{kv_heads}x{head_width}.comp",
        f"rotary_bf16_{query_heads}x{head_width}.comp",
        f"rotary_bf16_{kv_heads}x{head_width}.comp",
        f"append_kv_state_bf16_{kv_heads}x{head_width}.comp",
        f"gqa_attention_bf16_q{query_heads}_kv{kv_heads}_d{head_width}_cap4.comp",
        f"gqa_attention_bf16_q{query_heads}_kv{kv_heads}_d{head_width}_cap8.comp",
    }


def copy_shader_templates(source_dir: Path, dest_dir: Path, shader_files: set[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for shader_file in sorted(shader_files):
        source = source_dir / shader_file
        if not source.exists():
            raise ModelCompileError(f"missing shader template {source}")
        shutil.copy2(source, dest_dir / shader_file)


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

    return {
        "path": TOKENIZER_PACKAGE_DIR,
        "files": copied_files,
    }


def copy_config_package(model_dir: Path, lowered_dir: Path) -> None:
    source = model_dir / CONFIG_PACKAGE_FILE
    if not source.is_file():
        raise ModelCompileError(f"source model does not contain required config file {source}")
    shutil.copy2(source, lowered_dir / CONFIG_PACKAGE_FILE)


def copy_tensor_package(tensor_index: Json, lowered_dir: Path) -> Json:
    weights_dir = lowered_dir / WEIGHTS_PACKAGE_DIR
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        {
            Path(info["source_file"])
            for info in tensor_index["tensors"].values()
            if info.get("source_file")
        }
    )
    if not source_files:
        raise ModelCompileError("tensor index does not declare any source_file entries")

    dest_by_source: dict[Path, Path] = {}
    used_names: set[str] = set()
    for source in source_files:
        if not source.is_file():
            raise ModelCompileError(f"tensor source file does not exist: {source}")
        dest_name = source.name
        if dest_name in used_names:
            digest = blake2s(str(source.resolve()).encode("utf-8"), digest_size=4).hexdigest()
            dest_name = f"{source.stem}-{digest}{source.suffix}"
        used_names.add(dest_name)
        dest = weights_dir / dest_name
        shutil.copy2(source, dest)
        dest_by_source[source] = dest

    packaged = deepcopy(tensor_index)
    source_records = {
        Path(source_record["path"]): source_record
        for source_record in tensor_index.get("source", {}).get("weights_files", [])
    }
    packaged["source"] = {
        "packaged": True,
        "weights_dir": WEIGHTS_PACKAGE_DIR,
        "weights_file": relative_json_path(lowered_dir, dest_by_source[source_files[0]]),
        "weights_files": [
            {
                **{
                    key: value
                    for key, value in source_records.get(source, {}).items()
                    if key != "path"
                },
                "path": relative_json_path(lowered_dir, dest_by_source[source]),
            }
            for source in source_files
        ],
    }
    for info in packaged["tensors"].values():
        source = Path(info["source_file"])
        info["source_file"] = relative_json_path(lowered_dir, dest_by_source[source])

    write_json(lowered_dir / "tensors.json", packaged)
    return packaged


def parameter_shape_for_node(circuit: Json, node: Json, tensor_index: Json) -> list[int]:
    parameter_id = node["params"][0]
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_shape(tensor_index, parameter["tensor"])


def state_port(circuit: Json, state_id: str) -> Json:
    for port in circuit.get("state_ports", []):
        if port["id"] == state_id:
            return port
    raise ModelCompileError(f"circuit {circuit['id']} has no state port {state_id!r}")


def tensor_shape(tensor_index: Json, tensor: str) -> list[int]:
    return [int(dim) for dim in tensor_index["tensors"][tensor]["shape"]]


def tensor_byte_count(tensor_index: Json, tensor: str) -> int:
    return int(tensor_index["tensors"][tensor]["byte_count"])


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
    digest = blake2s(str(model_dir.resolve()).encode("utf-8"), digest_size=4).hexdigest()
    return f"model_{digest}"
