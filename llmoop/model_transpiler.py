from __future__ import annotations

import json
import math
import re
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Json = dict[str, Any]


class ModelTranspileError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayerStructure:
    index: int
    prefix: str
    operator_type: str
    tensors: dict[str, str]


@dataclass(frozen=True)
class ModelStructure:
    model_dir: Path
    model_type: str | None
    architectures: tuple[str, ...]
    dtype: str | None
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    conv_l_cache: int
    vocab_size: int
    max_position_embeddings: int | None
    norm_eps: float
    rope_theta: float
    rope_interleaved: bool
    token_ids: Json
    tensors: dict[str, str]
    layers: tuple[LayerStructure, ...]


TOKEN_EMBEDDING_CANDIDATES = (
    "model.embed_tokens.weight",
    "model.tok_embeddings.weight",
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
)

OUTPUT_NORM_CANDIDATES = (
    "model.embedding_norm.weight",
    "model.norm.weight",
    "model.final_layernorm.weight",
    "transformer.ln_f.weight",
    "gpt_neox.final_layer_norm.weight",
)

OUTPUT_PROJECTION_CANDIDATES = (
    "lm_head.weight",
    "output.weight",
    "embed_out.weight",
)

OPERATOR_NORM_SUFFIXES = (
    "operator_norm.weight",
    "input_layernorm.weight",
    "self_attn_layer_norm.weight",
)

FFN_NORM_SUFFIXES = (
    "ffn_norm.weight",
    "post_attention_layernorm.weight",
    "post_attention_layer_norm.weight",
)

FFN_GATE_SUFFIXES = (
    "feed_forward.w1.weight",
    "mlp.gate_proj.weight",
    "feed_forward.gate_proj.weight",
)

FFN_DOWN_SUFFIXES = (
    "feed_forward.w2.weight",
    "mlp.down_proj.weight",
    "feed_forward.down_proj.weight",
)

FFN_UP_SUFFIXES = (
    "feed_forward.w3.weight",
    "mlp.up_proj.weight",
    "feed_forward.up_proj.weight",
)

CONV_IN_PROJECTION_SUFFIXES = ("conv.in_proj.weight",)
CONV_KERNEL_SUFFIXES = ("conv.conv.weight", "conv.depthwise.weight")
CONV_OUT_PROJECTION_SUFFIXES = ("conv.out_proj.weight",)

ATTENTION_Q_PROJECTION_SUFFIXES = ("self_attn.q_proj.weight", "attention.wq.weight")
ATTENTION_K_PROJECTION_SUFFIXES = ("self_attn.k_proj.weight", "attention.wk.weight")
ATTENTION_V_PROJECTION_SUFFIXES = ("self_attn.v_proj.weight", "attention.wv.weight")
ATTENTION_OUT_PROJECTION_SUFFIXES = (
    "self_attn.out_proj.weight",
    "self_attn.o_proj.weight",
    "attention.wo.weight",
)
ATTENTION_Q_NORM_SUFFIXES = ("self_attn.q_layernorm.weight", "self_attn.q_norm.weight")
ATTENTION_K_NORM_SUFFIXES = ("self_attn.k_layernorm.weight", "self_attn.k_norm.weight")

LAYER_ROOT_PATTERNS = (
    re.compile(r"^(?P<root>.+?\.layers)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<root>transformer\.h)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<root>gpt_neox\.layers)\.(?P<index>\d+)\."),
)


def transpile_model(model_dir: Path, output_dir: Path, *, clean: bool) -> ModelStructure:
    model_dir = model_dir.expanduser()
    config = read_json(model_dir / "config.json")
    tensor_index = make_tensor_index(model_dir)
    structure = discover_model_structure(model_dir, config, tensor_index["tensors"])

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "tensors.json", tensor_index)
    write_json(output_dir / "model.json", make_model_graph(structure, output_dir, tensor_index))

    for layer in structure.layers:
        write_json(
            output_dir / "layers" / f"layer_{layer.index:02d}.json",
            make_layer(structure, layer),
        )

    return structure


def discover_model_structure(
    model_dir: Path,
    config: Json,
    tensors: dict[str, Json],
) -> ModelStructure:
    token_embedding = find_first_tensor(tensors, TOKEN_EMBEDDING_CANDIDATES, role="token embedding")
    output_norm = find_first_tensor(tensors, OUTPUT_NORM_CANDIDATES, role="output norm")
    output_projection = find_first_existing_tensor(tensors, OUTPUT_PROJECTION_CANDIDATES) or token_embedding

    hidden_size = int(config.get("hidden_size") or tensors[token_embedding]["shape"][1])
    vocab_size = int(config.get("vocab_size") or tensors[token_embedding]["shape"][0])
    layer_root, layer_indices = discover_layer_root(tensors)
    layer_count = int(config.get("num_hidden_layers") or (max(layer_indices) + 1))
    layers = tuple(
        discover_layer_structure(
            tensors=tensors,
            configured_layer_types=config.get("layer_types"),
            layer_root=layer_root,
            layer_index=index,
        )
        for index in range(layer_count)
    )

    intermediate_size = discover_intermediate_size(config, tensors, layers)
    num_attention_heads = discover_int_config(
        config,
        "num_attention_heads",
        "n_head",
        "num_heads",
        role="number of attention heads",
    )
    num_key_value_heads = int(
        config.get("num_key_value_heads")
        or config.get("num_kv_heads")
        or config.get("multi_query_group_num")
        or num_attention_heads
    )
    conv_l_cache = discover_conv_l_cache(config, tensors, layers)

    return ModelStructure(
        model_dir=model_dir,
        model_type=config.get("model_type"),
        architectures=tuple(config.get("architectures", ())),
        dtype=config.get("dtype") or config.get("torch_dtype"),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layer_count,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        conv_l_cache=conv_l_cache,
        vocab_size=vocab_size,
        max_position_embeddings=config.get("max_position_embeddings"),
        norm_eps=discover_norm_eps(config),
        rope_theta=discover_rope_theta(config),
        rope_interleaved=bool(config.get("rope_interleaved", False)),
        token_ids={
            "bos": config.get("bos_token_id"),
            "eos": config.get("eos_token_id"),
            "pad": config.get("pad_token_id"),
        },
        tensors={
            "token_embedding": token_embedding,
            "output_norm": output_norm,
            "output_projection": output_projection,
        },
        layers=layers,
    )


def discover_layer_structure(
    *,
    tensors: dict[str, Json],
    configured_layer_types: list[str] | None,
    layer_root: str,
    layer_index: int,
) -> LayerStructure:
    prefix = f"{layer_root}.{layer_index}"
    layer_tensors = {
        "operator_norm": find_layer_tensor(tensors, prefix, OPERATOR_NORM_SUFFIXES, role="operator norm"),
        "ffn_norm": find_layer_tensor(tensors, prefix, FFN_NORM_SUFFIXES, role="feed-forward norm"),
        "ffn_gate": find_layer_tensor(tensors, prefix, FFN_GATE_SUFFIXES, role="feed-forward gate projection"),
        "ffn_down": find_layer_tensor(tensors, prefix, FFN_DOWN_SUFFIXES, role="feed-forward down projection"),
        "ffn_up": find_layer_tensor(tensors, prefix, FFN_UP_SUFFIXES, role="feed-forward up projection"),
    }

    configured = configured_layer_types[layer_index] if configured_layer_types else None
    if configured in ("conv", "short_conv"):
        operator_type = "conv"
    elif configured in ("full_attention", "attention", "gqa_attention"):
        operator_type = "full_attention"
    else:
        operator_type = infer_operator_type(tensors, prefix)

    if operator_type == "conv":
        layer_tensors.update(
            {
                "conv_in_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    CONV_IN_PROJECTION_SUFFIXES,
                    role="short-conv input projection",
                ),
                "conv_depthwise_kernel": find_layer_tensor(
                    tensors,
                    prefix,
                    CONV_KERNEL_SUFFIXES,
                    role="short-conv depthwise kernel",
                ),
                "conv_out_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    CONV_OUT_PROJECTION_SUFFIXES,
                    role="short-conv output projection",
                ),
            }
        )
    elif operator_type == "full_attention":
        layer_tensors.update(
            {
                "q_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    ATTENTION_Q_PROJECTION_SUFFIXES,
                    role="attention query projection",
                ),
                "k_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    ATTENTION_K_PROJECTION_SUFFIXES,
                    role="attention key projection",
                ),
                "v_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    ATTENTION_V_PROJECTION_SUFFIXES,
                    role="attention value projection",
                ),
                "attention_out_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    ATTENTION_OUT_PROJECTION_SUFFIXES,
                    role="attention output projection",
                ),
            }
        )
        optional_q_norm = find_optional_layer_tensor(
            tensors, prefix, ATTENTION_Q_NORM_SUFFIXES
        )
        optional_k_norm = find_optional_layer_tensor(
            tensors, prefix, ATTENTION_K_NORM_SUFFIXES
        )
        if optional_q_norm is not None:
            layer_tensors["q_norm"] = optional_q_norm
        if optional_k_norm is not None:
            layer_tensors["k_norm"] = optional_k_norm
    else:
        raise ModelTranspileError(f"unsupported operator type {operator_type!r} for layer {layer_index}")

    return LayerStructure(
        index=layer_index,
        prefix=prefix,
        operator_type=operator_type,
        tensors=layer_tensors,
    )


def infer_operator_type(tensors: dict[str, Json], prefix: str) -> str:
    if find_first_existing_tensor(tensors, (f"{prefix}.{suffix}" for suffix in CONV_IN_PROJECTION_SUFFIXES)):
        return "conv"
    if find_first_existing_tensor(tensors, (f"{prefix}.{suffix}" for suffix in ATTENTION_Q_PROJECTION_SUFFIXES)):
        return "full_attention"
    raise ModelTranspileError(f"could not infer operator type for layer prefix {prefix!r}")


def discover_layer_root(tensors: dict[str, Json]) -> tuple[str, tuple[int, ...]]:
    roots: Counter[str] = Counter()
    root_indices: dict[str, set[int]] = {}
    for name in tensors:
        for pattern in LAYER_ROOT_PATTERNS:
            match = pattern.match(name)
            if match:
                root = match.group("root")
                index = int(match.group("index"))
                roots[root] += 1
                root_indices.setdefault(root, set()).add(index)
                break
    if not roots:
        raise ModelTranspileError("could not discover a repeated decoder-layer root in checkpoint tensors")
    root = roots.most_common(1)[0][0]
    return root, tuple(sorted(root_indices[root]))


def discover_intermediate_size(config: Json, tensors: dict[str, Json], layers: tuple[LayerStructure, ...]) -> int:
    if "intermediate_size" in config:
        return int(config["intermediate_size"])
    if "ffn_hidden_size" in config:
        return int(config["ffn_hidden_size"])
    first_gate = tensors[layers[0].tensors["ffn_gate"]]["shape"]
    return int(first_gate[0])


def discover_conv_l_cache(config: Json, tensors: dict[str, Json], layers: tuple[LayerStructure, ...]) -> int:
    for key in ("conv_L_cache", "conv_l_cache", "short_conv_kernel_size"):
        if key in config:
            return int(config[key])
    for layer in layers:
        if layer.operator_type == "conv":
            return int(tensors[layer.tensors["conv_depthwise_kernel"]]["shape"][-1])
    return 0


def discover_int_config(config: Json, *keys: str, role: str) -> int:
    for key in keys:
        if key in config:
            return int(config[key])
    raise ModelTranspileError(f"could not discover {role} from config keys {keys}")


def discover_norm_eps(config: Json) -> float:
    for key in ("rms_norm_eps", "norm_eps", "block_norm_eps"):
        if key in config:
            return float(config[key])
    raise ModelTranspileError(
        "could not discover RMS normalization epsilon from model config"
    )


def discover_rope_theta(config: Json) -> float:
    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    if "rope_theta" in config:
        return float(config["rope_theta"])
    raise ModelTranspileError("could not discover RoPE theta from model config")


def make_layer(structure: ModelStructure, layer: LayerStructure) -> Json:
    hidden_size = structure.hidden_size
    tensor_refs = list(layer.tensors.values())
    operator = (
        make_conv_operator(structure, layer)
        if layer.operator_type == "conv"
        else make_attention_operator(structure, layer)
    )

    return {
        "schema": "llmoop.pedal_instance.v1",
        "id": f"layer_{layer.index:02d}",
        "source_layer_index": layer.index,
        "type": "pedal_instance",
        "pedal_class": make_pedal_class(structure, layer.operator_type),
        "operator_type": layer.operator_type,
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": structure.rope_theta,
            "rope_interleaved": structure.rope_interleaved,
        },
        "ports": {
            "inputs": [{"id": "input", "signal": "frame", "shape": [hidden_size]}],
            "outputs": [{"id": "output", "signal": "frame", "shape": [hidden_size]}],
            "controls": [{"id": "control", "type": "pedal_control", "optional": True}],
        },
        "state_ports": make_state_ports(structure, layer.operator_type),
        "parameter_block": make_parameter_block(layer.operator_type, layer.tensors),
        "transition_contract": {
            "type": "stateful_frame_transform",
            "equation": "(output_frame, next_state, events) = pedal(input_frame, state, params, control)",
            "reference_behavior": f"source_transformers_layer_{layer.index}",
            "behavioral_error_contract": "not_defined_yet",
        },
        "runtime_boundary": {
            "opaque_to_pedalboard": True,
            "compiler_may_fuse_internal_operations": True,
            "compiler_may_replace_reference_decomposition": True,
        },
        "reference_decomposition": make_reference_decomposition(structure, layer, operator),
        "tensor_refs": tensor_refs,
    }


def make_model_graph(structure: ModelStructure, output_dir: Path, tensor_index: Json) -> Json:
    pedals = [
        {
            "id": f"layer_{layer.index:02d}",
            "type": "pedal_instance",
            "pedal_class": make_pedal_class(structure, layer.operator_type),
            "operator_type": layer.operator_type,
            "file": f"layers/layer_{layer.index:02d}.json",
        }
        for layer in structure.layers
    ]

    output_projection = {
        "id": "output_projection",
        "type": "linear_projection",
        "params": {"weight": tensor_ref(structure.tensors["output_projection"])},
    }
    if structure.tensors["output_projection"] == structure.tensors["token_embedding"]:
        output_projection["sharing"] = "same_parameter_object_as_token_embedding"

    return {
        "schema": "llmoop.model_graph.v1",
        "source": tensor_index["source"],
        "architecture": {
            "family": "decoder_only_transformer",
            "model_type": structure.model_type,
            "architectures": list(structure.architectures),
            "dtype": structure.dtype,
        },
        "dimensions": {
            "hidden_size": structure.hidden_size,
            "intermediate_size": structure.intermediate_size,
            "num_hidden_layers": structure.num_hidden_layers,
            "num_attention_heads": structure.num_attention_heads,
            "num_key_value_heads": structure.num_key_value_heads,
            "conv_l_cache": structure.conv_l_cache,
            "vocab_size": structure.vocab_size,
            "max_position_embeddings": structure.max_position_embeddings,
        },
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": structure.rope_theta,
            "rope_interleaved": structure.rope_interleaved,
        },
        "token_ids": structure.token_ids,
        "files": {
            "tensor_index": "tensors.json",
            "pedals_dir": "layers/",
        },
        "graph": {
            "input_transducer": {
                "id": "token_embedding",
                "type": "embedding_lookup",
                "output": "stream_frame",
                "params": {"weight": tensor_ref(structure.tensors["token_embedding"])},
            },
            "pedalboard": {
                "wiring": "series",
                "pedals": pedals,
            },
            "output_transducer": {
                "components": [
                    {
                        "id": "output_norm",
                        "type": "rms_norm",
                        "attrs": {"eps": structure.norm_eps},
                        "params": {"weight": tensor_ref(structure.tensors["output_norm"])},
                    },
                    output_projection,
                ]
            },
        },
        "component_templates": {
            "shortconv_layer": "opaque layer pedal with fixed rolling temporal state",
            "gqa_attention_layer": "opaque layer pedal with append-only KV state",
            "swiglu_feed_forward": "dense gated feed-forward operator",
            "rms_norm": "stateless normalization operator",
            "residual_add": "stateless signal mixer",
        },
        "output_dir": str(output_dir),
    }


def make_reference_decomposition(
    structure: ModelStructure,
    layer: LayerStructure,
    operator: Json,
) -> Json:
    hidden_size = structure.hidden_size
    return {
        "source": "source_transformers_layer",
        "wiring": [
            {
                "id": "operator_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "input",
                "output": "operator_norm.output",
                "params": {"weight": tensor_ref(layer.tensors["operator_norm"])},
            },
            operator,
            {
                "id": "operator_residual",
                "type": "residual_add",
                "circuit_template": f"add_h{hidden_size}_v1",
                "inputs": ["input", "operator.output"],
                "output": "operator_residual.output",
            },
            {
                "id": "ffn_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "operator_residual.output",
                "output": "ffn_norm.output",
                "params": {"weight": tensor_ref(layer.tensors["ffn_norm"])},
            },
            make_ffn_component(structure, layer),
            {
                "id": "ffn_residual",
                "type": "residual_add",
                "circuit_template": f"add_h{hidden_size}_v1",
                "inputs": ["operator_residual.output", "ffn.output"],
                "output": "output",
            },
        ],
    }


def make_ffn_component(structure: ModelStructure, layer: LayerStructure) -> Json:
    return {
        "id": "feed_forward",
        "type": "swiglu_feed_forward",
        "circuit_template": f"swiglu_ffn_{structure.hidden_size}_{structure.intermediate_size}_v1",
        "input": "ffn_norm.output",
        "output": "ffn.output",
        "params": {
            "gate": tensor_ref(layer.tensors["ffn_gate"]),
            "down": tensor_ref(layer.tensors["ffn_down"]),
            "up": tensor_ref(layer.tensors["ffn_up"]),
        },
    }


def make_conv_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    return {
        "id": "operator",
        "type": "short_conv_operator",
        "circuit_template": f"short_conv_h{structure.hidden_size}_k{structure.conv_l_cache}_v1",
        "input": "operator_norm.output",
        "output": "operator.output",
        "state_ports": make_state_ports(structure, "conv"),
        "params": {
            "in_projection": tensor_ref(layer.tensors["conv_in_projection"]),
            "depthwise_kernel": tensor_ref(layer.tensors["conv_depthwise_kernel"]),
            "out_projection": tensor_ref(layer.tensors["conv_out_projection"]),
        },
        "internal_pedals": [
            {"id": "in_projection", "type": "linear"},
            {"id": "split_b_c_x", "type": "split", "parts": ["b", "c", "x"]},
            {"id": "input_gate", "type": "multiply", "expression": "b * x"},
            {"id": "temporal_memory", "type": "stateful_delay_line"},
            {"id": "depthwise_conv", "type": "depthwise_temporal_convolution"},
            {"id": "output_gate", "type": "multiply", "expression": "c * conv_out"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_attention_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    head_width = structure.hidden_size // structure.num_attention_heads
    heads = {
        "query_heads": structure.num_attention_heads,
        "key_value_heads": structure.num_key_value_heads,
        "head_width": head_width,
        "query_groups_per_kv_head": structure.num_attention_heads // structure.num_key_value_heads,
    }
    params = {
        "q_projection": tensor_ref(layer.tensors["q_projection"]),
        "k_projection": tensor_ref(layer.tensors["k_projection"]),
        "v_projection": tensor_ref(layer.tensors["v_projection"]),
        "out_projection": tensor_ref(layer.tensors["attention_out_projection"]),
    }
    internal_pedals = [
        {"id": "q_projection", "type": "linear"},
        {"id": "k_projection", "type": "linear"},
        {"id": "v_projection", "type": "linear"},
    ]
    if "q_norm" in layer.tensors:
        params["q_norm"] = tensor_ref(layer.tensors["q_norm"])
        internal_pedals.append({"id": "q_norm", "type": "rms_norm_per_head"})
    if "k_norm" in layer.tensors:
        params["k_norm"] = tensor_ref(layer.tensors["k_norm"])
        internal_pedals.append({"id": "k_norm", "type": "rms_norm_per_head"})
    internal_pedals.extend(
        [
            {"id": "rope", "type": "rotary_position_embedding"},
            {"id": "kv_memory", "type": "stateful_append_memory"},
            {"id": "attention_read", "type": "scaled_dot_product_attention"},
            {"id": "out_projection", "type": "linear"},
        ]
    )
    return {
        "id": "operator",
        "type": "gqa_attention_operator",
        "circuit_template": (
            "gqa_attention_"
            f"h{structure.hidden_size}_q{structure.num_attention_heads}_"
            f"kv{structure.num_key_value_heads}_d{head_width}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "heads": heads,
        "state_ports": make_state_ports(structure, "full_attention"),
        "params": params,
        "internal_pedals": internal_pedals,
    }


def make_parameter_block(operator_type: str, tensors: dict[str, str]) -> Json:
    if operator_type == "conv":
        layout = "shortconv_layer_params_v1"
    elif operator_type == "full_attention":
        layout = "gqa_attention_layer_params_v1"
    else:
        raise ModelTranspileError(f"unsupported parameter layout for operator {operator_type!r}")
    return {
        "layout": layout,
        "storage": "source_tensor_refs",
        "params": {name: tensor_ref(tensor) for name, tensor in tensors.items()},
        "tensor_refs": list(tensors.values()),
    }


def make_state_ports(structure: ModelStructure, operator_type: str) -> list[Json]:
    if operator_type == "conv":
        return [
            {
                "id": "temporal_memory",
                "type": "rolling_frame_memory",
                "shape": [structure.conv_l_cache, structure.hidden_size],
                "update": "shift_append",
                "sharing": "per_stream_per_pedal_instance",
            }
        ]

    if operator_type == "full_attention":
        head_width = structure.hidden_size // structure.num_attention_heads
        return [
            {
                "id": "kv_memory",
                "type": "append_only_attention_memory",
                "key_shape_per_token": [structure.num_key_value_heads, head_width],
                "value_shape_per_token": [structure.num_key_value_heads, head_width],
                "growth": "per_activation",
                "sharing": "per_stream_per_pedal_instance",
            }
        ]

    raise ModelTranspileError(f"unsupported state ports for operator {operator_type!r}")


def make_pedal_class(structure: ModelStructure, operator_type: str) -> str:
    if operator_type == "conv":
        return (
            f"shortconv_layer_h{structure.hidden_size}_"
            f"k{structure.conv_l_cache}_ffn{structure.intermediate_size}_v1"
        )

    if operator_type == "full_attention":
        head_width = structure.hidden_size // structure.num_attention_heads
        return (
            "gqa_attention_layer_"
            f"h{structure.hidden_size}_q{structure.num_attention_heads}_"
            f"kv{structure.num_key_value_heads}_d{head_width}_"
            f"ffn{structure.intermediate_size}_v1"
        )

    raise ModelTranspileError(f"unsupported pedal class for operator {operator_type!r}")


def make_tensor_index(model_dir: Path) -> Json:
    tensor_entries: Json = {}
    total_params = 0
    total_bytes = 0
    source_files: list[Json] = []

    for weights_file in discover_safetensor_files(model_dir):
        header_len, header = read_safetensors_header(weights_file)
        source_files.append(
            {
                "path": str(weights_file),
                "safetensors_header_bytes": header_len,
                "metadata": header.get("__metadata__", {}),
            }
        )
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            shape = info["shape"]
            offsets = info["data_offsets"]
            params = math.prod(shape)
            byte_count = offsets[1] - offsets[0]
            total_params += params
            total_bytes += byte_count
            tensor_entries[name] = {
                "dtype": info["dtype"],
                "shape": shape,
                "data_offsets": offsets,
                "parameter_count": params,
                "byte_count": byte_count,
                "source_file": str(weights_file),
            }

    return {
        "schema": "llmoop.tensor_index.v1",
        "source": {
            "model_dir": str(model_dir),
            "weights_file": source_files[0]["path"],
            "weights_files": source_files,
        },
        "totals": {
            "tensor_count": len(tensor_entries),
            "parameter_count": total_params,
            "byte_count": total_bytes,
        },
        "tensors": tensor_entries,
    }


def discover_safetensor_files(model_dir: Path) -> tuple[Path, ...]:
    single = model_dir / "model.safetensors"
    if single.exists():
        return (single,)

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = read_json(index_file)
        files = sorted({model_dir / filename for filename in index.get("weight_map", {}).values()})
        if files:
            return tuple(files)

    files = tuple(sorted(model_dir.glob("*.safetensors")))
    if files:
        return files

    raise ModelTranspileError(f"no safetensors checkpoint files found in {model_dir}")


def read_safetensors_header(path: Path) -> tuple[int, Json]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header_len, header


def find_first_tensor(tensors: dict[str, Json], candidates: Iterable[str], *, role: str) -> str:
    match = find_first_existing_tensor(tensors, candidates)
    if match is None:
        raise ModelTranspileError(f"could not discover {role} tensor")
    return match


def find_first_existing_tensor(tensors: dict[str, Json], candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in tensors:
            return name
    return None


def find_layer_tensor(
    tensors: dict[str, Json],
    prefix: str,
    suffixes: Iterable[str],
    *,
    role: str,
) -> str:
    return find_first_tensor(tensors, (f"{prefix}.{suffix}" for suffix in suffixes), role=role)


def find_optional_layer_tensor(
    tensors: dict[str, Json], prefix: str, suffixes: Iterable[str]
) -> str | None:
    return find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in suffixes)
    )


def tensor_ref(name: str) -> dict[str, str]:
    return {"tensor": name}


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
