#!/usr/bin/env python3
"""Transpile a local LFM2 checkpoint into modular graph JSON files."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("/home/aristath/models/lfm2.5/230m")
DEFAULT_OUTPUT_DIR = Path("transpiled/lfm2_5_230m")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header_len, header


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def tensor_ref(name: str) -> dict[str, str]:
    return {"tensor": name}


def require_tensor(tensors: dict[str, Any], name: str) -> None:
    if name not in tensors:
        raise KeyError(f"missing tensor: {name}")


def make_tensor_index(model_dir: Path, header_len: int, header: dict[str, Any]) -> dict[str, Any]:
    model_file = model_dir / "model.safetensors"
    tensor_entries = {}
    total_params = 0
    total_bytes = 0

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
            "source_file": str(model_file),
        }

    return {
        "schema": "llmoop.tensor_index.v1",
        "source": {
            "model_dir": str(model_dir),
            "weights_file": str(model_file),
            "safetensors_header_bytes": header_len,
            "metadata": header.get("__metadata__", {}),
        },
        "totals": {
            "tensor_count": len(tensor_entries),
            "parameter_count": total_params,
            "byte_count": total_bytes,
        },
        "tensors": tensor_entries,
    }


def make_common_layer_prefix(layer_index: int) -> str:
    return f"model.layers.{layer_index}"


def make_ffn_component(prefix: str) -> dict[str, Any]:
    return {
        "id": "feed_forward",
        "type": "swiglu_feed_forward",
        "circuit_template": "swiglu_ffn_1024_2560_v1",
        "input": "ffn_norm.output",
        "output": "ffn.output",
        "params": {
            "gate": tensor_ref(f"{prefix}.feed_forward.w1.weight"),
            "down": tensor_ref(f"{prefix}.feed_forward.w2.weight"),
            "up": tensor_ref(f"{prefix}.feed_forward.w3.weight"),
        },
    }


def make_conv_operator(prefix: str, hidden_size: int, conv_l_cache: int) -> dict[str, Any]:
    return {
        "id": "operator",
        "type": "short_conv_operator",
        "circuit_template": f"short_conv_h{hidden_size}_k{conv_l_cache}_v1",
        "input": "operator_norm.output",
        "output": "operator.output",
        "state_ports": [
            {
                "id": "temporal_memory",
                "type": "rolling_frame_memory",
                "shape": [conv_l_cache, hidden_size],
                "update": "shift_append",
                "sharing": "per_stream_per_layer_instance",
            }
        ],
        "params": {
            "in_projection": tensor_ref(f"{prefix}.conv.in_proj.weight"),
            "depthwise_kernel": tensor_ref(f"{prefix}.conv.conv.weight"),
            "out_projection": tensor_ref(f"{prefix}.conv.out_proj.weight"),
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


def make_attention_operator(prefix: str, hidden_size: int, n_heads: int, n_kv_heads: int) -> dict[str, Any]:
    head_width = hidden_size // n_heads
    return {
        "id": "operator",
        "type": "gqa_attention_operator",
        "circuit_template": f"gqa_attention_h{hidden_size}_q{n_heads}_kv{n_kv_heads}_d{head_width}_v1",
        "input": "operator_norm.output",
        "output": "operator.output",
        "heads": {
            "query_heads": n_heads,
            "key_value_heads": n_kv_heads,
            "head_width": head_width,
            "query_groups_per_kv_head": n_heads // n_kv_heads,
        },
        "state_ports": [
            {
                "id": "kv_memory",
                "type": "append_only_attention_memory",
                "key_shape_per_token": [n_kv_heads, head_width],
                "value_shape_per_token": [n_kv_heads, head_width],
                "growth": "per_activation",
                "sharing": "per_stream_per_layer_instance",
            }
        ],
        "params": {
            "q_projection": tensor_ref(f"{prefix}.self_attn.q_proj.weight"),
            "k_projection": tensor_ref(f"{prefix}.self_attn.k_proj.weight"),
            "v_projection": tensor_ref(f"{prefix}.self_attn.v_proj.weight"),
            "out_projection": tensor_ref(f"{prefix}.self_attn.out_proj.weight"),
            "q_norm": tensor_ref(f"{prefix}.self_attn.q_layernorm.weight"),
            "k_norm": tensor_ref(f"{prefix}.self_attn.k_layernorm.weight"),
        },
        "internal_pedals": [
            {"id": "q_projection", "type": "linear"},
            {"id": "k_projection", "type": "linear"},
            {"id": "v_projection", "type": "linear"},
            {"id": "q_norm", "type": "rms_norm_per_head"},
            {"id": "k_norm", "type": "rms_norm_per_head"},
            {"id": "rope", "type": "rotary_position_embedding"},
            {"id": "kv_memory", "type": "stateful_append_memory"},
            {"id": "attention_read", "type": "scaled_dot_product_attention"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_reference_decomposition(
    prefix: str,
    hidden_size: int,
    operator: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "lfm2_reference_layer",
        "wiring": [
            {
                "id": "operator_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "input",
                "output": "operator_norm.output",
                "params": {"weight": tensor_ref(f"{prefix}.operator_norm.weight")},
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
                "params": {"weight": tensor_ref(f"{prefix}.ffn_norm.weight")},
            },
            make_ffn_component(prefix),
            {
                "id": "ffn_residual",
                "type": "residual_add",
                "circuit_template": f"add_h{hidden_size}_v1",
                "inputs": ["operator_residual.output", "ffn.output"],
                "output": "output",
            },
        ],
    }


def make_parameter_block(prefix: str, layer_type: str, tensor_refs: list[str]) -> dict[str, Any]:
    params = {
        "operator_norm": tensor_ref(f"{prefix}.operator_norm.weight"),
        "ffn_norm": tensor_ref(f"{prefix}.ffn_norm.weight"),
        "ffn_gate": tensor_ref(f"{prefix}.feed_forward.w1.weight"),
        "ffn_down": tensor_ref(f"{prefix}.feed_forward.w2.weight"),
        "ffn_up": tensor_ref(f"{prefix}.feed_forward.w3.weight"),
    }
    if layer_type == "conv":
        layout = "lfm2_shortconv_layer_params_v1"
        params.update(
            {
                "conv_in_projection": tensor_ref(f"{prefix}.conv.in_proj.weight"),
                "conv_depthwise_kernel": tensor_ref(f"{prefix}.conv.conv.weight"),
                "conv_out_projection": tensor_ref(f"{prefix}.conv.out_proj.weight"),
            }
        )
    else:
        layout = "lfm2_gqa_attention_layer_params_v1"
        params.update(
            {
                "q_projection": tensor_ref(f"{prefix}.self_attn.q_proj.weight"),
                "k_projection": tensor_ref(f"{prefix}.self_attn.k_proj.weight"),
                "v_projection": tensor_ref(f"{prefix}.self_attn.v_proj.weight"),
                "attention_out_projection": tensor_ref(f"{prefix}.self_attn.out_proj.weight"),
                "q_norm": tensor_ref(f"{prefix}.self_attn.q_layernorm.weight"),
                "k_norm": tensor_ref(f"{prefix}.self_attn.k_layernorm.weight"),
            }
        )
    return {
        "layout": layout,
        "storage": "source_tensor_refs",
        "params": params,
        "tensor_refs": tensor_refs,
    }


def make_state_ports(config: dict[str, Any], layer_type: str) -> list[dict[str, Any]]:
    hidden_size = config["hidden_size"]
    if layer_type == "conv":
        return [
            {
                "id": "temporal_memory",
                "type": "rolling_frame_memory",
                "shape": [config["conv_L_cache"], hidden_size],
                "update": "shift_append",
                "sharing": "per_stream_per_pedal_instance",
            }
        ]

    head_width = hidden_size // config["num_attention_heads"]
    return [
        {
            "id": "kv_memory",
            "type": "append_only_attention_memory",
            "key_shape_per_token": [config["num_key_value_heads"], head_width],
            "value_shape_per_token": [config["num_key_value_heads"], head_width],
            "growth": "per_activation",
            "sharing": "per_stream_per_pedal_instance",
        }
    ]


def make_pedal_class(config: dict[str, Any], layer_type: str) -> str:
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    if layer_type == "conv":
        return f"lfm2_shortconv_layer_h{hidden_size}_k{config['conv_L_cache']}_ffn{intermediate_size}_v1"

    head_width = hidden_size // config["num_attention_heads"]
    return (
        "lfm2_gqa_attention_layer_"
        f"h{hidden_size}_q{config['num_attention_heads']}_kv{config['num_key_value_heads']}_"
        f"d{head_width}_ffn{intermediate_size}_v1"
    )


def make_layer(config: dict[str, Any], tensors: dict[str, Any], layer_index: int) -> dict[str, Any]:
    layer_type = config["layer_types"][layer_index]
    hidden_size = config["hidden_size"]
    prefix = make_common_layer_prefix(layer_index)

    common_tensors = [
        f"{prefix}.operator_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.feed_forward.w1.weight",
        f"{prefix}.feed_forward.w2.weight",
        f"{prefix}.feed_forward.w3.weight",
    ]
    if layer_type == "conv":
        operator = make_conv_operator(prefix, hidden_size, config["conv_L_cache"])
        operator_tensors = [
            f"{prefix}.conv.in_proj.weight",
            f"{prefix}.conv.conv.weight",
            f"{prefix}.conv.out_proj.weight",
        ]
    elif layer_type == "full_attention":
        operator = make_attention_operator(
            prefix,
            hidden_size,
            config["num_attention_heads"],
            config["num_key_value_heads"],
        )
        operator_tensors = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.out_proj.weight",
            f"{prefix}.self_attn.q_layernorm.weight",
            f"{prefix}.self_attn.k_layernorm.weight",
        ]
    else:
        raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_index}")

    for name in common_tensors + operator_tensors:
        require_tensor(tensors, name)

    tensor_refs = common_tensors + operator_tensors
    pedal_class = make_pedal_class(config, layer_type)

    return {
        "schema": "llmoop.pedal_instance.v1",
        "id": f"layer_{layer_index:02d}",
        "source_layer_index": layer_index,
        "type": "pedal_instance",
        "pedal_class": pedal_class,
        "operator_type": layer_type,
        "ports": {
            "inputs": [{"id": "input", "signal": "frame", "shape": [hidden_size]}],
            "outputs": [{"id": "output", "signal": "frame", "shape": [hidden_size]}],
            "controls": [{"id": "control", "type": "pedal_control", "optional": True}],
        },
        "state_ports": make_state_ports(config, layer_type),
        "parameter_block": make_parameter_block(prefix, layer_type, tensor_refs),
        "transition_contract": {
            "type": "stateful_frame_transform",
            "equation": "(output_frame, next_state, events) = pedal(input_frame, state, params, control)",
            "reference_behavior": f"source_lfm2_layer_{layer_index}",
            "behavioral_error_contract": "not_defined_yet",
        },
        "runtime_boundary": {
            "opaque_to_pedalboard": True,
            "compiler_may_fuse_internal_operations": True,
            "compiler_may_replace_reference_decomposition": True,
        },
        "reference_decomposition": make_reference_decomposition(prefix, hidden_size, operator),
        "tensor_refs": tensor_refs,
    }


def make_model_graph(config: dict[str, Any], output_dir: Path, tensor_index: dict[str, Any]) -> dict[str, Any]:
    hidden_size = config["hidden_size"]
    pedals = [
        {
            "id": f"layer_{index:02d}",
            "type": "pedal_instance",
            "pedal_class": make_pedal_class(config, layer_type),
            "operator_type": layer_type,
            "file": f"layers/layer_{index:02d}.json",
        }
        for index, layer_type in enumerate(config["layer_types"])
    ]

    return {
        "schema": "llmoop.model_graph.v1",
        "source": tensor_index["source"],
        "architecture": {
            "model_type": config["model_type"],
            "architectures": config.get("architectures", []),
            "dtype": config.get("dtype"),
        },
        "dimensions": {
            "hidden_size": hidden_size,
            "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "conv_l_cache": config["conv_L_cache"],
            "vocab_size": config["vocab_size"],
            "max_position_embeddings": config["max_position_embeddings"],
        },
        "token_ids": {
            "bos": config.get("bos_token_id"),
            "eos": config.get("eos_token_id"),
            "pad": config.get("pad_token_id"),
        },
        "files": {
            "tensor_index": "tensors.json",
            "pedals_dir": "layers/",
        },
        "graph": {
            "input_transducer": {
                "id": "token_embedding",
                "type": "embedding_lookup",
                "output": "stream_frame",
                "params": {"weight": tensor_ref("model.embed_tokens.weight")},
            },
            "pedalboard": {
                "wiring": "series",
                "pedals": pedals,
            },
            "output_transducer": {
                "components": [
                    {
                        "id": "embedding_norm",
                        "type": "rms_norm",
                        "params": {"weight": tensor_ref("model.embedding_norm.weight")},
                    },
                    {
                        "id": "tied_output_projection",
                        "type": "linear_projection",
                        "params": {"weight": tensor_ref("model.embed_tokens.weight")},
                        "sharing": "same_parameter_object_as_token_embedding",
                    },
                ]
            },
        },
        "component_templates": {
            "lfm2_shortconv_layer": "opaque layer pedal with fixed rolling temporal state",
            "lfm2_gqa_attention_layer": "opaque layer pedal with append-only KV state",
            "swiglu_feed_forward": "dense gated feed-forward operator",
            "rms_norm": "stateless normalization operator",
            "residual_add": "stateless signal mixer",
        },
        "output_dir": str(output_dir),
    }


def transpile(model_dir: Path, output_dir: Path, clean: bool) -> None:
    config = read_json(model_dir / "config.json")
    header_len, header = read_safetensors_header(model_dir / "model.safetensors")
    tensor_index = make_tensor_index(model_dir, header_len, header)
    tensors = tensor_index["tensors"]

    if config["model_type"] != "lfm2":
        raise ValueError(f"expected model_type lfm2, got {config['model_type']!r}")

    require_tensor(tensors, "model.embed_tokens.weight")
    require_tensor(tensors, "model.embedding_norm.weight")

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "tensors.json", tensor_index)
    write_json(output_dir / "model.json", make_model_graph(config, output_dir, tensor_index))

    for layer_index in range(config["num_hidden_layers"]):
        layer = make_layer(config, tensors, layer_index)
        write_json(output_dir / "layers" / f"layer_{layer_index:02d}.json", layer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-clean", action="store_true", help="do not delete an existing output directory first")
    args = parser.parse_args()

    transpile(args.model_dir, args.output_dir, clean=not args.no_clean)
    print(f"transpiled {args.model_dir} -> {args.output_dir}")


if __name__ == "__main__":
    main()
