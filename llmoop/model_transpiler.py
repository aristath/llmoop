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
    attention_window_size: int | None
    feed_forward_type: str
    shared_intermediate_size: int | None
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
    head_width: int
    rotary_width: int
    attention_window_size: int | None
    attention_output_gate: bool
    conv_l_cache: int
    vocab_size: int
    max_position_embeddings: int | None
    norm_eps: float
    rope_theta: float
    rope_interleaved: bool
    rms_norm_weight_offset: float
    embedding_scale: float
    residual_scale: float
    attention_scale: float
    logits_scale: float
    logits_soft_cap: float | None
    activation: str
    num_experts: int | None
    experts_per_token: int | None
    recurrent_mixer: Json | None
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
    "model.final_norm.weight",
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
    "temporal_pre_norm.weight",
)

FFN_NORM_SUFFIXES = (
    "ffn_norm.weight",
    "post_attention_layernorm.weight",
    "post_attention_layer_norm.weight",
    "channel_pre_norm.weight",
)

FFN_GATE_SUFFIXES = (
    "feed_forward.w1.weight",
    "mlp.gate_proj.weight",
    "feed_forward.gate_proj.weight",
    "mlp_block.gate_proj.weight",
)

FFN_DOWN_SUFFIXES = (
    "feed_forward.w2.weight",
    "mlp.down_proj.weight",
    "feed_forward.down_proj.weight",
    "mlp_block.down_proj.weight",
)

FFN_UP_SUFFIXES = (
    "feed_forward.w3.weight",
    "mlp.up_proj.weight",
    "feed_forward.up_proj.weight",
    "mlp_block.up_proj.weight",
)

MOE_INPUT_SUFFIXES = (
    "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.experts.gate_up_proj",
)
MOE_OUTPUT_SUFFIXES = (
    "block_sparse_moe.output_linear.weight",
    "block_sparse_moe.experts.down_proj",
)
MOE_ROUTER_SUFFIXES = (
    "block_sparse_moe.router.layer.weight",
    "block_sparse_moe.router.weight",
    "block_sparse_moe.gate.weight",
)

SHARED_MLP_INPUT_SUFFIXES = ("shared_mlp.input_linear.weight",)
SHARED_MLP_OUTPUT_SUFFIXES = ("shared_mlp.output_linear.weight",)

CONV_IN_PROJECTION_SUFFIXES = ("conv.in_proj.weight",)
CONV_KERNEL_SUFFIXES = ("conv.conv.weight", "conv.depthwise.weight")
CONV_OUT_PROJECTION_SUFFIXES = ("conv.out_proj.weight",)

ATTENTION_Q_PROJECTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "attention.wq.weight",
    "temporal_block.q_proj.weight",
)
ATTENTION_K_PROJECTION_SUFFIXES = (
    "self_attn.k_proj.weight",
    "attention.wk.weight",
    "temporal_block.k_proj.weight",
)
ATTENTION_V_PROJECTION_SUFFIXES = (
    "self_attn.v_proj.weight",
    "attention.wv.weight",
    "temporal_block.v_proj.weight",
)
ATTENTION_OUT_PROJECTION_SUFFIXES = (
    "self_attn.out_proj.weight",
    "self_attn.o_proj.weight",
    "attention.wo.weight",
    "temporal_block.o_proj.weight",
)
ATTENTION_Q_NORM_SUFFIXES = ("self_attn.q_layernorm.weight", "self_attn.q_norm.weight")
ATTENTION_K_NORM_SUFFIXES = ("self_attn.k_layernorm.weight", "self_attn.k_norm.weight")
ATTENTION_SINK_SUFFIXES = ("self_attn.sinks",)

GATED_DELTA_QKV_SUFFIXES = ("linear_attn.in_proj_qkv.weight",)
GATED_DELTA_Z_SUFFIXES = ("linear_attn.in_proj_z.weight",)
GATED_DELTA_B_SUFFIXES = ("linear_attn.in_proj_b.weight",)
GATED_DELTA_A_SUFFIXES = ("linear_attn.in_proj_a.weight",)
GATED_DELTA_CONV_SUFFIXES = ("linear_attn.conv1d.weight",)
GATED_DELTA_A_LOG_SUFFIXES = ("linear_attn.A_log",)
GATED_DELTA_DT_BIAS_SUFFIXES = ("linear_attn.dt_bias",)
GATED_DELTA_NORM_SUFFIXES = ("linear_attn.norm.weight",)
GATED_DELTA_OUT_SUFFIXES = ("linear_attn.out_proj.weight",)

RG_LRU_X_SUFFIXES = ("temporal_block.linear_x.weight",)
RG_LRU_Y_SUFFIXES = ("temporal_block.linear_y.weight",)
RG_LRU_OUT_SUFFIXES = ("temporal_block.linear_out.weight",)
RG_LRU_CONV_SUFFIXES = ("temporal_block.conv_1d.weight",)
RG_LRU_INPUT_GATE_WEIGHT_SUFFIXES = ("temporal_block.rg_lru.input_gate_weight",)
RG_LRU_INPUT_GATE_BIAS_SUFFIXES = ("temporal_block.rg_lru.input_gate_bias",)
RG_LRU_RECURRENT_GATE_WEIGHT_SUFFIXES = ("temporal_block.rg_lru.recurrent_gate_weight",)
RG_LRU_RECURRENT_GATE_BIAS_SUFFIXES = ("temporal_block.rg_lru.recurrent_gate_bias",)
RG_LRU_RECURRENT_PARAM_SUFFIXES = ("temporal_block.rg_lru.recurrent_param",)

LAYER_ROOT_PATTERNS = (
    re.compile(r"^(?P<root>.+?\.layers)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<root>transformer\.h)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<root>gpt_neox\.layers)\.(?P<index>\d+)\."),
)


def transpile_model(
    model_dir: Path, output_dir: Path, *, clean: bool
) -> ModelStructure:
    model_dir = model_dir.expanduser()
    config = read_json(model_dir / "config.json")
    generation_config_path = model_dir / "generation_config.json"
    generation_config = (
        read_json(generation_config_path) if generation_config_path.is_file() else {}
    )
    tensor_index = make_tensor_index(model_dir)
    structure = discover_model_structure(
        model_dir,
        config,
        tensor_index["tensors"],
        generation_config=generation_config,
    )

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "tensors.json", tensor_index)
    write_json(
        output_dir / "model.json", make_model_graph(structure, output_dir, tensor_index)
    )

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
    *,
    generation_config: Json | None = None,
) -> ModelStructure:
    layer_root, layer_indices = discover_layer_root(tensors)
    decoder_config = discover_decoder_config(config, max(layer_indices) + 1)
    model_prefix = layer_root.removesuffix(".layers")
    token_embedding = find_first_tensor(
        tensors,
        (f"{model_prefix}.embed_tokens.weight", *TOKEN_EMBEDDING_CANDIDATES),
        role="token embedding",
    )
    output_norm = find_first_tensor(
        tensors,
        (f"{model_prefix}.norm.weight", *OUTPUT_NORM_CANDIDATES),
        role="output norm",
    )
    output_projection = (
        find_first_existing_tensor(
            tensors,
            (f"{model_prefix}.lm_head.weight", *OUTPUT_PROJECTION_CANDIDATES),
        )
        or token_embedding
    )

    hidden_size = int(
        decoder_config.get("hidden_size") or tensors[token_embedding]["shape"][1]
    )
    vocab_size = int(
        decoder_config.get("vocab_size") or tensors[token_embedding]["shape"][0]
    )
    layer_count = int(
        decoder_config.get("num_hidden_layers") or (max(layer_indices) + 1)
    )
    configured_layer_types = discover_configured_layer_types(
        decoder_config, layer_count
    )
    attention_window_size = discover_attention_window_size(decoder_config)
    layers = tuple(
        discover_layer_structure(
            tensors=tensors,
            configured_layer_types=configured_layer_types,
            configured_attention_window_size=attention_window_size,
            layer_root=layer_root,
            layer_index=index,
        )
        for index in range(layer_count)
    )

    intermediate_size = discover_intermediate_size(decoder_config, tensors, layers)
    num_attention_heads = discover_int_config(
        decoder_config,
        "num_attention_heads",
        "n_head",
        "num_heads",
        role="number of attention heads",
    )
    num_key_value_heads = int(
        decoder_config.get("num_key_value_heads")
        or decoder_config.get("num_kv_heads")
        or decoder_config.get("multi_query_group_num")
        or num_attention_heads
    )
    head_width = int(
        decoder_config.get("head_dim") or hidden_size // num_attention_heads
    )
    rope_parameters = decoder_config.get("rope_parameters")
    partial_rotary_factor = float(
        rope_parameters.get("partial_rotary_factor", 1.0)
        if isinstance(rope_parameters, dict)
        else decoder_config.get("partial_rotary_factor", 1.0)
    )
    rotary_width = int(head_width * partial_rotary_factor)
    attention_output_gate = discover_attention_output_gate(
        decoder_config,
        tensors,
        layers,
        num_attention_heads=num_attention_heads,
        head_width=head_width,
    )
    recurrent_mixer = discover_recurrent_mixer(decoder_config, tensors, layers)
    conv_l_cache = discover_conv_l_cache(decoder_config, tensors, layers)
    rms_norm_weight_offset = discover_outer_norm_weight_offset(
        recurrent_mixer=recurrent_mixer,
        attention_output_gate=attention_output_gate,
        output_norm=output_norm,
        layers=layers,
    )
    has_sparse_experts = any(
        layer.feed_forward_type == "sparse_moe" for layer in layers
    )
    if has_sparse_experts:
        num_experts = discover_int_config(
            decoder_config, "num_local_experts", "num_experts", role="number of experts"
        )
        experts_per_token = discover_int_config(
            decoder_config,
            "num_experts_per_tok",
            "experts_per_token",
            role="experts per token",
        )
    else:
        num_experts = None
        experts_per_token = None

    return ModelStructure(
        model_dir=model_dir,
        model_type=decoder_config.get("model_type") or config.get("model_type"),
        architectures=tuple(config.get("architectures", ())),
        dtype=(
            decoder_config.get("dtype")
            or decoder_config.get("torch_dtype")
            or config.get("dtype")
            or config.get("torch_dtype")
        ),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layer_count,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_width=head_width,
        rotary_width=rotary_width,
        attention_window_size=attention_window_size,
        attention_output_gate=attention_output_gate,
        conv_l_cache=conv_l_cache,
        vocab_size=vocab_size,
        max_position_embeddings=decoder_config.get("max_position_embeddings"),
        norm_eps=discover_norm_eps(decoder_config),
        rope_theta=discover_rope_theta(decoder_config),
        rope_interleaved=bool(decoder_config.get("rope_interleaved", False)),
        rms_norm_weight_offset=rms_norm_weight_offset,
        embedding_scale=discover_embedding_scale(decoder_config, hidden_size),
        residual_scale=float(decoder_config.get("residual_multiplier", 1.0)),
        attention_scale=float(
            decoder_config.get("attention_multiplier", head_width**-0.5)
        ),
        logits_scale=float(decoder_config.get("logits_scaling", 1.0)),
        logits_soft_cap=(
            float(decoder_config["logits_soft_cap"])
            if decoder_config.get("logits_soft_cap") is not None
            else None
        ),
        activation=discover_activation(decoder_config),
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        recurrent_mixer=recurrent_mixer,
        token_ids={
            "bos": decoder_config.get("bos_token_id"),
            "eos": (generation_config or {}).get(
                "eos_token_id", decoder_config.get("eos_token_id")
            ),
            "pad": decoder_config.get("pad_token_id"),
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
    configured_attention_window_size: int | None,
    layer_root: str,
    layer_index: int,
) -> LayerStructure:
    prefix = f"{layer_root}.{layer_index}"
    layer_tensors = {
        "operator_norm": find_layer_tensor(
            tensors, prefix, OPERATOR_NORM_SUFFIXES, role="operator norm"
        ),
        "ffn_norm": find_layer_tensor(
            tensors, prefix, FFN_NORM_SUFFIXES, role="feed-forward norm"
        ),
    }
    dense_gate = find_optional_layer_tensor(tensors, prefix, FFN_GATE_SUFFIXES)
    moe_input = find_optional_layer_tensor(tensors, prefix, MOE_INPUT_SUFFIXES)
    if dense_gate is not None:
        feed_forward_type = "dense_swiglu"
        layer_tensors.update(
            {
                "ffn_gate": dense_gate,
                "ffn_down": find_layer_tensor(
                    tensors,
                    prefix,
                    FFN_DOWN_SUFFIXES,
                    role="feed-forward down projection",
                ),
                "ffn_up": find_layer_tensor(
                    tensors, prefix, FFN_UP_SUFFIXES, role="feed-forward up projection"
                ),
            }
        )
        add_optional_linear_biases(
            tensors,
            layer_tensors,
            ("ffn_gate", "ffn_down", "ffn_up"),
        )
    elif moe_input is not None:
        feed_forward_type = "sparse_moe"
        layer_tensors.update(
            {
                "moe_input": moe_input,
                "moe_output": find_layer_tensor(
                    tensors,
                    prefix,
                    MOE_OUTPUT_SUFFIXES,
                    role="expert output projection",
                ),
                "moe_router": find_layer_tensor(
                    tensors,
                    prefix,
                    MOE_ROUTER_SUFFIXES,
                    role="expert router projection",
                ),
            }
        )
        shared_input = find_optional_layer_tensor(
            tensors, prefix, SHARED_MLP_INPUT_SUFFIXES
        )
        shared_output = find_optional_layer_tensor(
            tensors, prefix, SHARED_MLP_OUTPUT_SUFFIXES
        )
        if (shared_input is None) != (shared_output is None):
            raise ModelTranspileError(
                f"layer prefix {prefix!r} has an incomplete shared feed-forward expert"
            )
        if shared_input is not None and shared_output is not None:
            layer_tensors["shared_mlp_input"] = shared_input
            layer_tensors["shared_mlp_output"] = shared_output
    else:
        raise ModelTranspileError(
            f"could not discover feed-forward structure for layer prefix {prefix!r}"
        )

    configured = configured_layer_types[layer_index] if configured_layer_types else None
    if configured in ("conv", "short_conv"):
        operator_type = "conv"
    elif configured in ("full_attention", "attention", "gqa_attention"):
        operator_type = "full_attention"
    elif configured in ("sliding_attention", "window_attention"):
        operator_type = "full_attention"
    elif configured in ("linear_attention", "gated_delta"):
        operator_type = "gated_delta"
    elif configured in ("recurrent", "rg_lru"):
        operator_type = "rg_lru"
    else:
        operator_type = infer_operator_type(tensors, prefix)

    if configured == "full_attention":
        layer_attention_window_size = None
    elif operator_type == "full_attention":
        layer_attention_window_size = configured_attention_window_size
    else:
        layer_attention_window_size = None

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
        optional_sinks = find_optional_layer_tensor(
            tensors, prefix, ATTENTION_SINK_SUFFIXES
        )
        if optional_sinks is not None:
            layer_tensors["attention_sinks"] = optional_sinks
        add_optional_linear_biases(
            tensors,
            layer_tensors,
            (
                "q_projection",
                "k_projection",
                "v_projection",
                "attention_out_projection",
            ),
        )
    elif operator_type == "gated_delta":
        layer_tensors.update(
            {
                "delta_qkv_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_QKV_SUFFIXES,
                    role="gated-delta QKV projection",
                ),
                "delta_z_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_Z_SUFFIXES,
                    role="gated-delta output gate projection",
                ),
                "delta_b_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_B_SUFFIXES,
                    role="gated-delta beta projection",
                ),
                "delta_a_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_A_SUFFIXES,
                    role="gated-delta decay projection",
                ),
                "delta_conv_kernel": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_CONV_SUFFIXES,
                    role="gated-delta convolution kernel",
                ),
                "delta_a_log": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_A_LOG_SUFFIXES,
                    role="gated-delta decay parameter",
                ),
                "delta_dt_bias": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_DT_BIAS_SUFFIXES,
                    role="gated-delta time bias",
                ),
                "delta_norm": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_NORM_SUFFIXES,
                    role="gated-delta output norm",
                ),
                "delta_out_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    GATED_DELTA_OUT_SUFFIXES,
                    role="gated-delta output projection",
                ),
            }
        )
    elif operator_type == "rg_lru":
        layer_tensors.update(
            {
                "rg_lru_x_projection": find_layer_tensor(
                    tensors, prefix, RG_LRU_X_SUFFIXES, role="RG-LRU x projection"
                ),
                "rg_lru_y_projection": find_layer_tensor(
                    tensors, prefix, RG_LRU_Y_SUFFIXES, role="RG-LRU y projection"
                ),
                "rg_lru_out_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_OUT_SUFFIXES,
                    role="RG-LRU output projection",
                ),
                "rg_lru_conv_kernel": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_CONV_SUFFIXES,
                    role="RG-LRU depthwise convolution kernel",
                ),
                "rg_lru_input_gate_weight": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_INPUT_GATE_WEIGHT_SUFFIXES,
                    role="RG-LRU input gate weight",
                ),
                "rg_lru_input_gate_bias": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_INPUT_GATE_BIAS_SUFFIXES,
                    role="RG-LRU input gate bias",
                ),
                "rg_lru_recurrent_gate_weight": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_RECURRENT_GATE_WEIGHT_SUFFIXES,
                    role="RG-LRU recurrent gate weight",
                ),
                "rg_lru_recurrent_gate_bias": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_RECURRENT_GATE_BIAS_SUFFIXES,
                    role="RG-LRU recurrent gate bias",
                ),
                "rg_lru_recurrent_param": find_layer_tensor(
                    tensors,
                    prefix,
                    RG_LRU_RECURRENT_PARAM_SUFFIXES,
                    role="RG-LRU recurrence parameter",
                ),
            }
        )
        add_optional_linear_biases(
            tensors,
            layer_tensors,
            (
                "rg_lru_x_projection",
                "rg_lru_y_projection",
                "rg_lru_out_projection",
            ),
        )
        conv_bias = find_bias_for_weight(tensors, layer_tensors["rg_lru_conv_kernel"])
        if conv_bias is not None:
            layer_tensors["rg_lru_conv_bias"] = conv_bias
    else:
        raise ModelTranspileError(
            f"unsupported operator type {operator_type!r} for layer {layer_index}"
        )

    return LayerStructure(
        index=layer_index,
        prefix=prefix,
        operator_type=operator_type,
        attention_window_size=layer_attention_window_size,
        feed_forward_type=feed_forward_type,
        shared_intermediate_size=(
            int(tensors[layer_tensors["shared_mlp_input"]]["shape"][0]) // 2
            if "shared_mlp_input" in layer_tensors
            else None
        ),
        tensors=layer_tensors,
    )


def infer_operator_type(tensors: dict[str, Json], prefix: str) -> str:
    if find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in CONV_IN_PROJECTION_SUFFIXES)
    ):
        return "conv"
    if find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in ATTENTION_Q_PROJECTION_SUFFIXES)
    ):
        return "full_attention"
    if find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in GATED_DELTA_QKV_SUFFIXES)
    ):
        return "gated_delta"
    if find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in RG_LRU_X_SUFFIXES)
    ):
        return "rg_lru"
    raise ModelTranspileError(
        f"could not infer operator type for layer prefix {prefix!r}"
    )


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
        raise ModelTranspileError(
            "could not discover a repeated decoder-layer root in checkpoint tensors"
        )
    root = roots.most_common(1)[0][0]
    return root, tuple(sorted(root_indices[root]))


def discover_decoder_config(config: Json, discovered_layer_count: int) -> Json:
    candidates: list[Json] = []

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        if int(value.get("num_hidden_layers", -1)) == discovered_layer_count:
            candidates.append(value)
        for nested in value.values():
            if isinstance(nested, dict):
                visit(nested)

    visit(config)
    if not candidates:
        raise ModelTranspileError(
            f"could not discover decoder config for {discovered_layer_count} repeated layers"
        )
    candidates.sort(
        key=lambda candidate: sum(
            key in candidate
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "vocab_size",
                "layer_types",
            )
        ),
        reverse=True,
    )
    return candidates[0]


def discover_configured_layer_types(config: Json, layer_count: int) -> list[str] | None:
    configured = config.get("layer_types") or config.get("layers_block_type")
    if configured is None:
        configured = config.get("_block_types")
    if not isinstance(configured, list) or not configured:
        return None
    values = [str(value) for value in configured]
    if len(values) == layer_count:
        return values
    return [values[index % len(values)] for index in range(layer_count)]


def discover_attention_output_gate(
    config: Json,
    tensors: dict[str, Json],
    layers: tuple[LayerStructure, ...],
    *,
    num_attention_heads: int,
    head_width: int,
) -> bool:
    configured = config.get("attn_output_gate")
    expected_width = num_attention_heads * head_width
    for layer in layers:
        if layer.operator_type != "full_attention":
            continue
        projection_width = int(tensors[layer.tensors["q_projection"]]["shape"][0])
        discovered = projection_width == expected_width * 2
        if projection_width not in (expected_width, expected_width * 2):
            raise ModelTranspileError(
                f"attention query projection width {projection_width} is incompatible with "
                f"{num_attention_heads} heads of width {head_width}"
            )
        if configured is not None and bool(configured) != discovered:
            raise ModelTranspileError(
                "attention output-gate config disagrees with query projection shape"
            )
        return discovered
    return False


def discover_recurrent_mixer(
    config: Json,
    tensors: dict[str, Json],
    layers: tuple[LayerStructure, ...],
) -> Json | None:
    rg_lru_layer = next(
        (layer for layer in layers if layer.operator_type == "rg_lru"), None
    )
    if rg_lru_layer is not None:
        gate_shape = tensors[rg_lru_layer.tensors["rg_lru_input_gate_weight"]]["shape"]
        conv_shape = tensors[rg_lru_layer.tensors["rg_lru_conv_kernel"]]["shape"]
        x_shape = tensors[rg_lru_layer.tensors["rg_lru_x_projection"]]["shape"]
        return {
            "type": "rg_lru",
            "width": int(x_shape[0]),
            "heads": int(gate_shape[0]),
            "block_width": int(gate_shape[1]),
            "conv_kernel_width": int(conv_shape[-1]),
            "state_dtype": "F32",
        }
    if not any(layer.operator_type == "gated_delta" for layer in layers):
        return None
    keys = (
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
    )
    missing = [key for key in keys if key not in config]
    if missing:
        raise ModelTranspileError(
            f"gated-delta structure is missing config dimensions: {', '.join(missing)}"
        )
    return {
        "type": "gated_delta",
        "conv_kernel_width": int(config["linear_conv_kernel_dim"]),
        "key_head_width": int(config["linear_key_head_dim"]),
        "key_heads": int(config["linear_num_key_heads"]),
        "value_heads": int(config["linear_num_value_heads"]),
        "value_head_width": int(config["linear_value_head_dim"]),
        "state_dtype": str(config.get("mamba_ssm_dtype", "float32"))
        .upper()
        .replace("FLOAT", "F"),
    }


def discover_outer_norm_weight_offset(
    *,
    recurrent_mixer: Json | None,
    attention_output_gate: bool,
    output_norm: str,
    layers: tuple[LayerStructure, ...],
) -> float:
    # This recurrent/full-attention graph stores its outer RMS scales around
    # zero and applies (1 + weight). The gated-delta mixer norm itself is a
    # conventional direct scale and is encoded separately in its circuit.
    offset_norm_suffixes = (
        ".temporal_pre_norm.weight",
        ".channel_pre_norm.weight",
        ".final_norm.weight",
    )
    stores_offset_weights = output_norm.endswith(offset_norm_suffixes) or any(
        layer.tensors["operator_norm"].endswith(offset_norm_suffixes)
        or layer.tensors["ffn_norm"].endswith(offset_norm_suffixes)
        for layer in layers
    )
    return (
        1.0
        if stores_offset_weights
        or (recurrent_mixer is not None and attention_output_gate)
        else 0.0
    )


def discover_intermediate_size(
    config: Json, tensors: dict[str, Json], layers: tuple[LayerStructure, ...]
) -> int:
    discovered: set[int] = set()
    for layer in layers:
        if layer.feed_forward_type == "dense_swiglu":
            discovered.add(int(tensors[layer.tensors["ffn_gate"]]["shape"][0]))
        elif layer.feed_forward_type == "sparse_moe":
            shape = tensors[layer.tensors["moe_input"]]["shape"]
            discovered.add(int(shape[-2]) // 2)
    if len(discovered) != 1:
        raise ModelTranspileError(
            f"feed-forward tensor shapes disagree on intermediate width: {sorted(discovered)}"
        )
    return discovered.pop()


def discover_attention_window_size(config: Json) -> int | None:
    for key in ("sliding_window", "attention_window_size"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def discover_embedding_scale(config: Json, hidden_size: int) -> float:
    if "embedding_multiplier" in config:
        return float(config["embedding_multiplier"])
    if bool(config.get("embeddings_scale_by_sqrt_dim", False)):
        return round_float_to_bf16(math.sqrt(hidden_size))
    return 1.0


def discover_activation(config: Json) -> str:
    configured = str(
        config.get("hidden_activation") or config.get("hidden_act") or "silu"
    )
    if configured in ("silu", "swish"):
        return "silu"
    if configured in ("gelu_pytorch_tanh", "gelu_new", "gelu_fast"):
        return "gelu_tanh"
    raise ModelTranspileError(f"unsupported feed-forward activation {configured!r}")


def round_float_to_bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def discover_conv_l_cache(
    config: Json, tensors: dict[str, Json], layers: tuple[LayerStructure, ...]
) -> int:
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
        if layer.operator_type == "full_attention"
        else make_gated_delta_operator(structure, layer)
        if layer.operator_type == "gated_delta"
        else make_rg_lru_operator(structure, layer)
    )

    return {
        "schema": "llmoop.pedal_instance.v1",
        "id": f"layer_{layer.index:02d}",
        "source_layer_index": layer.index,
        "type": "pedal_instance",
        "pedal_class": make_pedal_class(structure, layer),
        "operator_type": layer.operator_type,
        "feed_forward": make_feed_forward_descriptor(structure, layer),
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": structure.rope_theta,
            "rope_interleaved": structure.rope_interleaved,
            "rotary_width": structure.rotary_width,
            "rms_norm_weight_offset": structure.rms_norm_weight_offset,
            "attention_output_gate": structure.attention_output_gate,
            "residual_scale": structure.residual_scale,
            "attention_scale": structure.attention_scale,
            "attention_window_size": layer.attention_window_size,
        },
        "ports": {
            "inputs": [{"id": "input", "signal": "frame", "shape": [hidden_size]}],
            "outputs": [{"id": "output", "signal": "frame", "shape": [hidden_size]}],
            "controls": [{"id": "control", "type": "pedal_control", "optional": True}],
        },
        "state_ports": make_state_ports(
            structure,
            layer.operator_type,
            attention_window_size=layer.attention_window_size,
        ),
        "parameter_block": make_parameter_block(
            layer.operator_type, layer.feed_forward_type, layer.tensors
        ),
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
        "reference_decomposition": make_reference_decomposition(
            structure, layer, operator
        ),
        "tensor_refs": tensor_refs,
    }


def make_model_graph(
    structure: ModelStructure, output_dir: Path, tensor_index: Json
) -> Json:
    pedals = [
        {
            "id": f"layer_{layer.index:02d}",
            "type": "pedal_instance",
            "pedal_class": make_pedal_class(structure, layer),
            "operator_type": layer.operator_type,
            "file": f"layers/layer_{layer.index:02d}.json",
        }
        for layer in structure.layers
    ]

    output_projection = {
        "id": "output_projection",
        "type": "linear_projection",
        "attrs": {
            "scale": 1.0 / structure.logits_scale,
            "soft_cap": structure.logits_soft_cap,
        },
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
            "head_width": structure.head_width,
            "rotary_width": structure.rotary_width,
            "attention_window_size": structure.attention_window_size,
            "conv_l_cache": structure.conv_l_cache,
            "vocab_size": structure.vocab_size,
            "max_position_embeddings": structure.max_position_embeddings,
            "num_experts": structure.num_experts,
            "experts_per_token": structure.experts_per_token,
        },
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": structure.rope_theta,
            "rope_interleaved": structure.rope_interleaved,
            "rms_norm_weight_offset": structure.rms_norm_weight_offset,
            "embedding_scale": structure.embedding_scale,
            "residual_scale": structure.residual_scale,
            "attention_scale": structure.attention_scale,
            "logits_scale": structure.logits_scale,
            "logits_soft_cap": structure.logits_soft_cap,
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
                "attrs": {"scale": structure.embedding_scale},
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
                        "attrs": {
                            "eps": structure.norm_eps,
                            "weight_offset": structure.rms_norm_weight_offset,
                        },
                        "params": {
                            "weight": tensor_ref(structure.tensors["output_norm"])
                        },
                    },
                    output_projection,
                ]
            },
        },
        "component_templates": {
            "shortconv_layer": "opaque layer pedal with fixed rolling temporal state",
            "rg_lru_layer": "opaque recurrent layer pedal with fixed convolution and recurrent state",
            "gqa_attention_layer": "opaque layer pedal with append-only KV state",
            "swiglu_feed_forward": "dense gated feed-forward operator",
            "rms_norm": "stateless normalization operator",
            "residual_add": "stateless signal mixer",
        },
        "output_dir": str(output_dir),
    }


def make_feed_forward_descriptor(
    structure: ModelStructure, layer: LayerStructure
) -> Json:
    descriptor: Json = {
        "type": layer.feed_forward_type,
        "hidden_size": structure.hidden_size,
        "intermediate_size": structure.intermediate_size,
        "activation": structure.activation,
    }
    if layer.feed_forward_type == "sparse_moe":
        descriptor.update(
            {
                "num_experts": structure.num_experts,
                "experts_per_token": structure.experts_per_token,
                "shared_intermediate_size": layer.shared_intermediate_size,
            }
        )
    return descriptor


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
    if layer.feed_forward_type == "sparse_moe":
        params = {
            "router": tensor_ref(layer.tensors["moe_router"]),
            "input": tensor_ref(layer.tensors["moe_input"]),
            "output": tensor_ref(layer.tensors["moe_output"]),
        }
        if layer.shared_intermediate_size is not None:
            params.update(
                {
                    "shared_input": tensor_ref(layer.tensors["shared_mlp_input"]),
                    "shared_output": tensor_ref(layer.tensors["shared_mlp_output"]),
                }
            )
        return {
            "id": "feed_forward",
            "type": "sparse_moe_feed_forward",
            "input": "ffn_norm.output",
            "output": "ffn.output",
            "dimensions": make_feed_forward_descriptor(structure, layer),
            "params": params,
        }
    params = {
        "gate": tensor_ref(layer.tensors["ffn_gate"]),
        "down": tensor_ref(layer.tensors["ffn_down"]),
        "up": tensor_ref(layer.tensors["ffn_up"]),
    }
    for source_id, target_id in (
        ("ffn_gate_bias", "gate_bias"),
        ("ffn_down_bias", "down_bias"),
        ("ffn_up_bias", "up_bias"),
    ):
        if source_id in layer.tensors:
            params[target_id] = tensor_ref(layer.tensors[source_id])
    return {
        "id": "feed_forward",
        "type": "swiglu_feed_forward",
        "circuit_template": f"swiglu_ffn_{structure.hidden_size}_{structure.intermediate_size}_v1",
        "input": "ffn_norm.output",
        "output": "ffn.output",
        "activation": structure.activation,
        "params": params,
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
    head_width = structure.head_width
    heads = {
        "query_heads": structure.num_attention_heads,
        "key_value_heads": structure.num_key_value_heads,
        "head_width": head_width,
        "query_groups_per_kv_head": structure.num_attention_heads
        // structure.num_key_value_heads,
    }
    params = {
        "q_projection": tensor_ref(layer.tensors["q_projection"]),
        "k_projection": tensor_ref(layer.tensors["k_projection"]),
        "v_projection": tensor_ref(layer.tensors["v_projection"]),
        "out_projection": tensor_ref(layer.tensors["attention_out_projection"]),
    }
    for source_id, target_id in (
        ("q_projection_bias", "q_projection_bias"),
        ("k_projection_bias", "k_projection_bias"),
        ("v_projection_bias", "v_projection_bias"),
        ("attention_out_projection_bias", "out_projection_bias"),
    ):
        if source_id in layer.tensors:
            params[target_id] = tensor_ref(layer.tensors[source_id])
    internal_pedals = [
        {"id": "q_projection", "type": "linear"},
        {"id": "k_projection", "type": "linear"},
        {"id": "v_projection", "type": "linear"},
    ]
    if structure.attention_output_gate:
        internal_pedals.append({"id": "q_gate_split", "type": "split"})
    if "q_norm" in layer.tensors:
        params["q_norm"] = tensor_ref(layer.tensors["q_norm"])
        internal_pedals.append({"id": "q_norm", "type": "rms_norm_per_head"})
    if "k_norm" in layer.tensors:
        params["k_norm"] = tensor_ref(layer.tensors["k_norm"])
        internal_pedals.append({"id": "k_norm", "type": "rms_norm_per_head"})
    if "attention_sinks" in layer.tensors:
        params["attention_sinks"] = tensor_ref(layer.tensors["attention_sinks"])
    internal_pedals.extend(
        [
            {"id": "rope", "type": "rotary_position_embedding"},
            {"id": "kv_memory", "type": "stateful_append_memory"},
            {"id": "attention_read", "type": "scaled_dot_product_attention"},
            *(
                [{"id": "attention_output_gate", "type": "sigmoid_multiply"}]
                if structure.attention_output_gate
                else []
            ),
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
        "rotary_width": structure.rotary_width,
        "output_gate": structure.attention_output_gate,
        "window_size": layer.attention_window_size,
        "state_ports": make_state_ports(
            structure,
            "full_attention",
            attention_window_size=layer.attention_window_size,
        ),
        "params": params,
        "internal_pedals": internal_pedals,
    }


def make_gated_delta_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    mixer = structure.recurrent_mixer
    if mixer is None:
        raise ModelTranspileError("gated-delta layer has no recurrent mixer dimensions")
    return {
        "id": "operator",
        "type": "gated_delta_operator",
        "circuit_template": (
            f"gated_delta_k{mixer['key_heads']}x{mixer['key_head_width']}_"
            f"v{mixer['value_heads']}x{mixer['value_head_width']}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "dimensions": mixer,
        "state_ports": make_state_ports(structure, "gated_delta"),
        "params": {
            "qkv_projection": tensor_ref(layer.tensors["delta_qkv_projection"]),
            "z_projection": tensor_ref(layer.tensors["delta_z_projection"]),
            "b_projection": tensor_ref(layer.tensors["delta_b_projection"]),
            "a_projection": tensor_ref(layer.tensors["delta_a_projection"]),
            "conv_kernel": tensor_ref(layer.tensors["delta_conv_kernel"]),
            "a_log": tensor_ref(layer.tensors["delta_a_log"]),
            "dt_bias": tensor_ref(layer.tensors["delta_dt_bias"]),
            "norm": tensor_ref(layer.tensors["delta_norm"]),
            "out_projection": tensor_ref(layer.tensors["delta_out_projection"]),
        },
        "internal_pedals": [
            {"id": "qkv_projection", "type": "linear"},
            {"id": "z_projection", "type": "linear"},
            {"id": "b_projection", "type": "linear"},
            {"id": "a_projection", "type": "linear"},
            {"id": "causal_conv", "type": "stateful_depthwise_convolution"},
            {"id": "delta_update", "type": "gated_delta_recurrence"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_rg_lru_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    mixer = structure.recurrent_mixer
    if mixer is None or mixer.get("type") != "rg_lru":
        raise ModelTranspileError("RG-LRU layer has no recurrent mixer dimensions")
    params = {
        name: tensor_ref(layer.tensors[name])
        for name in (
            "rg_lru_x_projection",
            "rg_lru_y_projection",
            "rg_lru_out_projection",
            "rg_lru_conv_kernel",
            "rg_lru_input_gate_weight",
            "rg_lru_input_gate_bias",
            "rg_lru_recurrent_gate_weight",
            "rg_lru_recurrent_gate_bias",
            "rg_lru_recurrent_param",
        )
    }
    for name in (
        "rg_lru_x_projection_bias",
        "rg_lru_y_projection_bias",
        "rg_lru_out_projection_bias",
        "rg_lru_conv_bias",
    ):
        if name in layer.tensors:
            params[name] = tensor_ref(layer.tensors[name])
    return {
        "id": "operator",
        "type": "rg_lru_operator",
        "circuit_template": (
            f"rg_lru_h{structure.hidden_size}_b{mixer['heads']}x{mixer['block_width']}"
            f"_k{mixer['conv_kernel_width']}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "dimensions": mixer,
        "activation": structure.activation,
        "state_ports": make_state_ports(structure, "rg_lru"),
        "params": params,
        "internal_pedals": [
            {"id": "x_projection", "type": "linear"},
            {"id": "y_projection", "type": "linear"},
            {"id": "y_activation", "type": structure.activation},
            {"id": "depthwise_convolution", "type": "stateful_depthwise_convolution"},
            {"id": "real_gated_recurrence", "type": "rg_lru_recurrence"},
            {"id": "output_gate", "type": "multiply"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_parameter_block(
    operator_type: str, feed_forward_type: str, tensors: dict[str, str]
) -> Json:
    if operator_type == "conv":
        layout = "shortconv_layer_params_v1"
    elif operator_type == "full_attention":
        layout = "gqa_attention_layer_params_v1"
    elif operator_type == "gated_delta":
        layout = "gated_delta_layer_params_v1"
    elif operator_type == "rg_lru":
        layout = "rg_lru_layer_params_v1"
    else:
        raise ModelTranspileError(
            f"unsupported parameter layout for operator {operator_type!r}"
        )
    return {
        "layout": f"{layout}_{feed_forward_type}",
        "storage": "source_tensor_refs",
        "params": {name: tensor_ref(tensor) for name, tensor in tensors.items()},
        "tensor_refs": list(tensors.values()),
    }


def make_state_ports(
    structure: ModelStructure,
    operator_type: str,
    *,
    attention_window_size: int | None = None,
) -> list[Json]:
    if operator_type == "conv":
        return [
            {
                "id": "temporal_memory",
                "type": "rolling_frame_memory",
                "shape": [structure.conv_l_cache, structure.hidden_size],
                "dtype": "BF16",
                "update": "shift_append",
                "sharing": "per_stream_per_pedal_instance",
            }
        ]

    if operator_type == "full_attention":
        head_width = structure.head_width
        return [
            {
                "id": "kv_memory",
                "type": "append_only_attention_memory",
                "query_heads": structure.num_attention_heads,
                "key_shape_per_token": [structure.num_key_value_heads, head_width],
                "value_shape_per_token": [structure.num_key_value_heads, head_width],
                "dtype": "BF16",
                "growth": "per_activation",
                "window_size": attention_window_size,
                "sharing": "per_stream_per_pedal_instance",
            }
        ]

    if operator_type == "gated_delta":
        mixer = structure.recurrent_mixer
        if mixer is None:
            raise ModelTranspileError(
                "gated-delta layer has no recurrent mixer dimensions"
            )
        key_width = int(mixer["key_heads"]) * int(mixer["key_head_width"])
        value_width = int(mixer["value_heads"]) * int(mixer["value_head_width"])
        conv_width = key_width * 2 + value_width
        return [
            {
                "id": "conv_state",
                "type": "rolling_channel_memory",
                "shape": [conv_width, int(mixer["conv_kernel_width"])],
                "dtype": "BF16",
                "update": "shift_append",
                "sharing": "per_stream_per_pedal_instance",
            },
            {
                "id": "recurrent_state",
                "type": "gated_delta_matrix_memory",
                "shape": [
                    int(mixer["value_heads"]),
                    int(mixer["key_head_width"]),
                    int(mixer["value_head_width"]),
                ],
                "dtype": mixer["state_dtype"],
                "update": "decay_delta_outer_product",
                "sharing": "per_stream_per_pedal_instance",
            },
        ]

    if operator_type == "rg_lru":
        mixer = structure.recurrent_mixer
        if mixer is None or mixer.get("type") != "rg_lru":
            raise ModelTranspileError("RG-LRU layer has no recurrent mixer dimensions")
        return [
            {
                "id": "conv_state",
                "type": "rolling_channel_memory",
                "shape": [
                    int(mixer["width"]),
                    int(mixer["conv_kernel_width"]),
                ],
                "dtype": "BF16",
                "update": "shift_append",
                "sharing": "per_stream_per_pedal_instance",
            },
            {
                "id": "recurrent_state",
                "type": "diagonal_recurrent_memory",
                "shape": [int(mixer["width"])],
                "dtype": str(mixer["state_dtype"]),
                "update": "real_gated_linear_recurrence",
                "sharing": "per_stream_per_pedal_instance",
            },
        ]

    raise ModelTranspileError(f"unsupported state ports for operator {operator_type!r}")


def make_pedal_class(structure: ModelStructure, layer: LayerStructure) -> str:
    operator_type = layer.operator_type
    feed_forward = (
        f"moe{structure.num_experts}x{structure.experts_per_token}i{structure.intermediate_size}"
        if layer.feed_forward_type == "sparse_moe"
        else f"ffn{structure.intermediate_size}"
    )
    if operator_type == "conv":
        return (
            f"shortconv_layer_h{structure.hidden_size}_"
            f"k{structure.conv_l_cache}_{feed_forward}_v1"
        )

    if operator_type == "rg_lru":
        mixer = structure.recurrent_mixer
        if mixer is None or mixer.get("type") != "rg_lru":
            raise ModelTranspileError("RG-LRU layer has no recurrent mixer dimensions")
        return (
            "rg_lru_layer_"
            f"h{structure.hidden_size}_b{mixer['heads']}x{mixer['block_width']}_"
            f"k{mixer['conv_kernel_width']}_{feed_forward}_v1"
        )

    if operator_type == "full_attention":
        head_width = structure.head_width
        return (
            "gqa_attention_layer_"
            f"h{structure.hidden_size}_q{structure.num_attention_heads}_"
            f"kv{structure.num_key_value_heads}_d{head_width}_"
            f"{feed_forward}_v1"
        )

    if operator_type == "gated_delta":
        mixer = structure.recurrent_mixer
        if mixer is None:
            raise ModelTranspileError(
                "gated-delta layer has no recurrent mixer dimensions"
            )
        return (
            "gated_delta_layer_"
            f"h{structure.hidden_size}_k{mixer['key_heads']}x{mixer['key_head_width']}_"
            f"v{mixer['value_heads']}x{mixer['value_head_width']}_"
            f"{feed_forward}_v1"
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
        files = sorted(
            {model_dir / filename for filename in index.get("weight_map", {}).values()}
        )
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


def find_first_tensor(
    tensors: dict[str, Json], candidates: Iterable[str], *, role: str
) -> str:
    match = find_first_existing_tensor(tensors, candidates)
    if match is None:
        raise ModelTranspileError(f"could not discover {role} tensor")
    return match


def find_first_existing_tensor(
    tensors: dict[str, Json], candidates: Iterable[str]
) -> str | None:
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
    return find_first_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in suffixes), role=role
    )


def find_optional_layer_tensor(
    tensors: dict[str, Json], prefix: str, suffixes: Iterable[str]
) -> str | None:
    return find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in suffixes)
    )


def find_bias_for_weight(tensors: dict[str, Json], weight: str) -> str | None:
    if not weight.endswith(".weight"):
        return None
    bias = f"{weight[: -len('.weight')]}.bias"
    return bias if bias in tensors else None


def add_optional_linear_biases(
    tensors: dict[str, Json],
    layer_tensors: dict[str, str],
    weight_ids: Iterable[str],
) -> None:
    for weight_id in weight_ids:
        bias = find_bias_for_weight(tensors, layer_tensors[weight_id])
        if bias is not None:
            layer_tensors[f"{weight_id}_bias"] = bias


def tensor_ref(name: str) -> dict[str, str]:
    return {"tensor": name}


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
