from nerve.circuit_lowering_common import *
from nerve.circuit_lowering_helpers import *

def _conv_nodes(
    hidden_size: int, numerics: Json, feed_forward: Json, parameters: Json
) -> list[Json]:
    return [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": _norm_attrs(numerics),
        },
        {
            "id": "conv_in_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["conv_projected"],
            "params": _linear_params("conv_in_projection", parameters),
        },
        {
            "id": "split_b_c_x",
            "op": "split",
            "inputs": ["conv_projected"],
            "outputs": ["gate_b", "gate_c", "projected_x"],
            "attrs": {
                "axis": "channel",
                "parts": ["b", "c", "x"],
                "part_width": hidden_size,
            },
        },
        {
            "id": "input_gate",
            "op": "multiply",
            "inputs": ["gate_b", "projected_x"],
            "outputs": ["gated_x"],
        },
        {
            "id": "temporal_memory_update",
            "op": "rolling_state_update",
            "inputs": ["gated_x", "temporal_memory"],
            "outputs": ["temporal_window"],
            "state_reads": ["temporal_memory"],
            "state_writes": ["temporal_memory"],
            "attrs": {"update": "shift_append", "logical_layout": "time_hidden"},
        },
        {
            "id": "depthwise_temporal_conv",
            "op": "depthwise_conv1d",
            "inputs": ["temporal_window"],
            "outputs": ["conv_out"],
            "params": ["conv_depthwise_kernel"],
            "attrs": {"groups": hidden_size, "padding": "conv_l_cache_minus_1"},
        },
        {
            "id": "output_gate",
            "op": "multiply",
            "inputs": ["gate_c", "conv_out"],
            "outputs": ["gated_conv_out"],
        },
        {
            "id": "conv_out_projection",
            "op": "linear",
            "inputs": ["gated_conv_out"],
            "outputs": ["operator_out"],
            "params": _linear_params("conv_out_projection", parameters),
        },
        *_ffn_tail(
            operator_output="operator_out",
            numerics=numerics,
            feed_forward=feed_forward,
            parameters=parameters,
        ),
    ]


def _attention_nodes(
    heads: Json,
    numerics: Json,
    *,
    has_q_norm: bool,
    has_k_norm: bool,
    has_value_norm: bool,
    feed_forward: Json,
    parameters: Json,
) -> list[Json]:
    nodes = [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": _norm_attrs(numerics),
        }
    ]
    q_output = (
        "q_and_gate_projected"
        if numerics.get("attention_output_gate")
        else "q_projected"
    )
    if "qkv_projection" in parameters:
        query_width = int(heads["query_heads"]) * int(heads["head_width"])
        if numerics.get("attention_output_gate"):
            query_width *= 2
        kv_width = int(heads["key_value_heads"]) * int(heads["head_width"])
        nodes.extend(
            [
                {
                    "id": "qkv_projection",
                    "op": "linear",
                    "inputs": ["operator_norm_out"],
                    "outputs": ["qkv_projected"],
                    "params": _linear_params("qkv_projection", parameters),
                },
                {
                    "id": "qkv_split",
                    "op": "split",
                    "inputs": ["qkv_projected"],
                    "outputs": [q_output, "k_projected", "v_projected"],
                    "attrs": {"part_widths": [query_width, kv_width, kv_width]},
                },
            ]
        )
    else:
        nodes.extend(
            [
                {
                    "id": "q_projection",
                    "op": "linear",
                    "inputs": ["operator_norm_out"],
                    "outputs": [q_output],
                    "params": _linear_params("q_projection", parameters),
                },
                *(
                    [
                        {
                            "id": "k_projection",
                            "op": "linear",
                            "inputs": ["operator_norm_out"],
                            "outputs": ["k_projected"],
                            "params": _linear_params("k_projection", parameters),
                        },
                        *(
                            [
                                {
                                    "id": "v_projection",
                                    "op": "linear",
                                    "inputs": ["operator_norm_out"],
                                    "outputs": ["v_projected"],
                                    "params": _linear_params(
                                        "v_projection", parameters
                                    ),
                                }
                            ]
                            if "v_projection" in parameters
                            else []
                        ),
                    ]
                    if "k_projection" in parameters
                    else []
                ),
            ]
        )
    attention_gate = None
    attention_gate_op = None
    if numerics.get("attention_output_gate"):
        attention_width = int(heads["query_heads"]) * int(heads["head_width"])
        nodes.append(
            {
                "id": "q_gate_split",
                "op": "split",
                "inputs": ["q_and_gate_projected"],
                "outputs": ["q_projected", "attention_gate"],
                "attrs": {
                    "axis": "channel",
                    "parts": 2,
                    "part_width": attention_width,
                    "layout": "per_head_interleaved",
                    "blocks": int(heads["query_heads"]),
                    "block_part_width": int(heads["head_width"]),
                },
            }
        )
        attention_gate = "attention_gate"
        attention_gate_op = "sigmoid_multiply"
    elif "attention_gate_projection" in parameters:
        nodes.append(
            {
                "id": "attention_gate_projection",
                "op": "linear",
                "inputs": ["operator_norm_out"],
                "outputs": ["attention_gate"],
                "params": _linear_params("attention_gate_projection", parameters),
            }
        )
        attention_gate = "attention_gate"
        activation = str(numerics.get("attention_gate_activation"))
        if activation not in {"sigmoid", "softplus"}:
            raise ValueError(f"unsupported attention gate activation {activation!r}")
        attention_gate_op = f"{activation}_multiply"
    q_rope_input = "q_projected"
    if has_q_norm:
        nodes.append(
            {
                "id": "q_head_norm",
                "op": "rms_norm_per_head",
                "inputs": ["q_projected"],
                "outputs": ["q_normed"],
                "params": ["q_norm"],
                "attrs": {
                    **_norm_attrs(numerics),
                    **heads,
                    "head_count": int(heads["query_heads"]),
                },
            }
        )
        q_rope_input = "q_normed"
    k_rope_input = "k_projected"
    if has_k_norm:
        nodes.append(
            {
                "id": "k_head_norm",
                "op": "rms_norm_per_head",
                "inputs": ["k_projected"],
                "outputs": ["k_normed"],
                "params": ["k_norm"],
                "attrs": {
                    **_norm_attrs(numerics),
                    **heads,
                    "head_count": int(heads["key_value_heads"]),
                },
            }
        )
        k_rope_input = "k_normed"
    value_input = (
        "k_projected" if numerics.get("attention_key_equals_value") else "v_projected"
    )
    if has_value_norm:
        nodes.append(
            {
                "id": "v_head_norm",
                "op": "rms_norm_per_head_unscaled",
                "inputs": [value_input],
                "outputs": ["v_normed"],
                "attrs": {
                    **_norm_attrs(numerics),
                    **heads,
                    "head_count": int(heads["key_value_heads"]),
                },
            }
        )
        value_input = "v_normed"
    rope_attrs = {
        "position_source": "stream_tick",
        "theta": float(numerics["rope_theta"]),
        "rope_type": str(numerics.get("rope_type", "default")),
        "scaling": numerics.get("rope_scaling"),
        "interleaved": bool(numerics["rope_interleaved"]),
        "rotary_width": int(numerics["rotary_width"]),
        **heads,
    }
    shared_kv = "k_projection" not in parameters and "qkv_projection" not in parameters
    attention_tail: list[Json] = [
        {
            "id": "q_rope",
            "op": "rotary_position_embedding",
            "inputs": [q_rope_input],
            "outputs": ["q_positioned"],
            "attrs": {
                **rope_attrs,
                "head_count": int(heads["query_heads"]),
            },
        },
        *(
            [
                {
                    "id": "k_rope",
                    "op": "rotary_position_embedding",
                    "inputs": [k_rope_input],
                    "outputs": ["k_positioned"],
                    "attrs": {
                        **rope_attrs,
                        "head_count": int(heads["key_value_heads"]),
                    },
                },
                {
                    "id": "kv_memory_append",
                    "op": "append_state_update",
                    "inputs": ["k_positioned", value_input, "kv_memory"],
                    "outputs": ["k_memory", "v_memory"],
                    "state_reads": ["kv_memory"],
                    "state_writes": ["kv_memory"],
                    "attrs": {"growth": "per_activation", **heads},
                },
            ]
            if "k_projection" in parameters or "qkv_projection" in parameters
            else []
        ),
        {
            "id": "attention_read",
            "op": "scaled_dot_product_attention",
            "inputs": [
                "q_positioned",
                "kv_memory" if shared_kv else "k_memory",
                "kv_memory" if shared_kv else "v_memory",
            ],
            "outputs": ["attention_out"],
            "params": (["attention_sinks"] if "attention_sinks" in parameters else []),
            "attrs": {
                "causal": True,
                "scale": float(numerics["attention_scale"]),
                "window_size": numerics.get("attention_window_size"),
                "attention_sinks": "attention_sinks" in parameters,
                **heads,
            },
        },
        *(
            [
                {
                    "id": "attention_output_gate",
                    "op": attention_gate_op,
                    "inputs": ["attention_out", attention_gate],
                    "outputs": ["attention_gated"],
                    "attrs": {
                        "query_heads": int(heads["query_heads"]),
                        "head_width": int(heads["head_width"]),
                        "per_head": bool(
                            numerics.get("attention_gate_per_head", False)
                        ),
                    },
                }
            ]
            if attention_gate is not None
            else []
        ),
        {
            "id": "attention_out_projection",
            "op": "linear",
            "inputs": ["attention_gated" if attention_gate else "attention_out"],
            "outputs": ["operator_out"],
            "params": _linear_params("attention_out_projection", parameters),
        },
        *_ffn_tail(
            operator_output="operator_out",
            numerics=numerics,
            feed_forward=feed_forward,
            parameters=parameters,
        ),
    ]
    nodes.extend(attention_tail)
    return nodes


def _gated_delta_nodes(
    dimensions: Json, numerics: Json, feed_forward: Json, parameters: Json
) -> list[Json]:
    key_width = int(dimensions["key_heads"]) * int(dimensions["key_head_width"])
    value_width = int(dimensions["value_heads"]) * int(dimensions["value_head_width"])
    conv_width = key_width * 2 + value_width
    return [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": _norm_attrs(numerics),
        },
        {
            "id": "delta_qkv_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_qkv_projected"],
            "params": _linear_params("delta_qkv_projection", parameters),
        },
        {
            "id": "delta_z_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_z"],
            "params": _linear_params("delta_z_projection", parameters),
        },
        {
            "id": "delta_b_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_b"],
            "params": _linear_params("delta_b_projection", parameters),
        },
        {
            "id": "delta_a_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_a"],
            "params": _linear_params("delta_a_projection", parameters),
        },
        {
            "id": "delta_causal_conv",
            "op": "causal_conv1d_silu",
            "inputs": ["delta_qkv_projected"],
            "outputs": ["delta_qkv_convolved"],
            "params": ["delta_conv_kernel"],
            "state_reads": ["conv_state"],
            "state_writes": ["conv_state"],
            "attrs": {
                "channels": conv_width,
                "kernel_width": int(dimensions["conv_kernel_width"]),
            },
        },
        {
            "id": "gated_delta_update",
            "op": "gated_delta_step",
            "inputs": ["delta_qkv_convolved", "delta_z", "delta_b", "delta_a"],
            "outputs": ["delta_mixed"],
            "params": ["delta_a_log", "delta_dt_bias", "delta_norm"],
            "state_reads": ["recurrent_state"],
            "state_writes": ["recurrent_state"],
            "attrs": {
                **dimensions,
                "key_width": key_width,
                "value_width": value_width,
                "norm_eps": float(numerics["rms_norm_eps"]),
                "norm_weight_offset": 0.0,
            },
        },
        {
            "id": "delta_out_projection",
            "op": "linear",
            "inputs": ["delta_mixed"],
            "outputs": ["operator_out"],
            "params": _linear_params("delta_out_projection", parameters),
        },
        *_ffn_tail(
            operator_output="operator_out",
            numerics=numerics,
            feed_forward=feed_forward,
            parameters=parameters,
        ),
    ]


def _rg_lru_nodes(
    dimensions: Json, numerics: Json, feed_forward: Json, parameters: Json
) -> list[Json]:
    recurrent_params = [
        "rg_lru_conv_kernel",
        "rg_lru_input_gate_weight",
        "rg_lru_input_gate_bias",
        "rg_lru_recurrent_gate_weight",
        "rg_lru_recurrent_gate_bias",
        "rg_lru_recurrent_param",
    ]
    if "rg_lru_conv_bias" not in parameters:
        raise ValueError("RG-LRU circuit requires a depthwise convolution bias")
    recurrent_params.insert(1, "rg_lru_conv_bias")
    return [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": _norm_attrs(numerics),
        },
        {
            "id": "rg_lru_y_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["rg_lru_y"],
            "params": _linear_params("rg_lru_y_projection", parameters),
        },
        {
            "id": "rg_lru_y_activation",
            "op": str(feed_forward["activation"]),
            "inputs": ["rg_lru_y"],
            "outputs": ["rg_lru_y_activated"],
            "attrs": {"element_count": int(dimensions["width"])},
        },
        {
            "id": "rg_lru_x_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["rg_lru_x"],
            "params": _linear_params("rg_lru_x_projection", parameters),
        },
        {
            "id": "rg_lru_step",
            "op": "rg_lru_step",
            "inputs": ["rg_lru_x"],
            "outputs": ["rg_lru_recurrent_out"],
            "params": recurrent_params,
            "state_reads": ["conv_state", "recurrent_state"],
            "state_writes": ["conv_state", "recurrent_state"],
            "attrs": dimensions,
        },
        {
            "id": "rg_lru_output_gate",
            "op": "multiply",
            "inputs": ["rg_lru_recurrent_out", "rg_lru_y_activated"],
            "outputs": ["rg_lru_gated"],
        },
        {
            "id": "rg_lru_out_projection",
            "op": "linear",
            "inputs": ["rg_lru_gated"],
            "outputs": ["operator_out"],
            "params": _linear_params("rg_lru_out_projection", parameters),
        },
        *_ffn_tail(
            operator_output="operator_out",
            numerics=numerics,
            feed_forward=feed_forward,
            parameters=parameters,
        ),
    ]


def _ffn_tail(
    operator_output: str, numerics: Json, feed_forward: Json, parameters: Json
) -> list[Json]:
    residual_scale = float(numerics["residual_scale"])
    operator_residual_update = operator_output
    operator_post_norm: list[Json] = []
    if "operator_post_norm" in parameters:
        operator_post_norm = [
            {
                "id": "operator_post_norm",
                "op": "rms_norm",
                "inputs": [operator_output],
                "outputs": ["operator_post_norm_out"],
                "params": ["operator_post_norm"],
                "attrs": _norm_attrs(numerics),
            }
        ]
        operator_residual_update = "operator_post_norm_out"
    prefix = [
        *operator_post_norm,
        _residual_node(
            node_id="operator_residual",
            residual="input_frame",
            update=operator_residual_update,
            output="operator_residual_out",
            scale=residual_scale,
        ),
        {
            "id": "ffn_norm",
            "op": "rms_norm",
            "inputs": ["operator_residual_out"],
            "outputs": ["ffn_norm_out"],
            "params": ["ffn_norm"],
            "attrs": _norm_attrs(numerics),
        },
    ]
    if feed_forward["type"] == "sparse_moe":
        shared_intermediate_size = feed_forward.get("shared_intermediate_size")
        has_shared_expert = shared_intermediate_size is not None
        routing = feed_forward["routing"]
        body = [
            {
                "id": "moe_router_projection",
                "op": "linear",
                "inputs": ["ffn_norm_out"],
                "outputs": ["moe_router_logits"],
                "params": _linear_params("moe_router", parameters),
            },
            {
                "id": "moe_topk",
                "op": "moe_topk",
                "inputs": ["moe_router_logits"],
                "outputs": ["moe_routes"],
                "params": (
                    ["moe_router_correction_bias"]
                    if "moe_router_correction_bias" in parameters
                    else []
                ),
                "attrs": {
                    "num_experts": int(feed_forward["num_experts"]),
                    "experts_per_token": int(feed_forward["experts_per_token"]),
                    "activation": str(routing["activation"]),
                    "normalize_selected": bool(routing["normalize_selected"]),
                    "logit_softcap": float(routing["logit_softcap"]),
                    "selection_bias": "moe_router_correction_bias" in parameters,
                },
            },
            {
                "id": "sparse_moe_gate_up",
                "op": "sparse_moe_gate_up",
                "inputs": ["ffn_norm_out", "moe_routes"],
                "outputs": ["moe_expert_intermediates"],
                "params": [
                    "moe_input",
                    *(
                        ["moe_input_scale_inv"]
                        if "moe_input_scale_inv" in parameters
                        else []
                    ),
                    *(["moe_input_scales"] if "moe_input_scales" in parameters else []),
                ],
                "attrs": {
                    "hidden_size": int(feed_forward.get("hidden_size", 0)),
                    "intermediate_size": int(feed_forward["intermediate_size"]),
                    "num_experts": int(feed_forward["num_experts"]),
                    "experts_per_token": int(feed_forward["experts_per_token"]),
                },
            },
            {
                "id": "sparse_moe_down",
                "op": "sparse_moe_down",
                "inputs": ["moe_expert_intermediates", "moe_routes"],
                "outputs": ["moe_expert_outputs"],
                "params": [
                    "moe_output",
                    *(
                        ["moe_output_scale_inv"]
                        if "moe_output_scale_inv" in parameters
                        else []
                    ),
                    *(
                        ["moe_output_scales"]
                        if "moe_output_scales" in parameters
                        else []
                    ),
                ],
                "attrs": {
                    "hidden_size": int(feed_forward.get("hidden_size", 0)),
                    "intermediate_size": int(feed_forward["intermediate_size"]),
                    "num_experts": int(feed_forward["num_experts"]),
                    "experts_per_token": int(feed_forward["experts_per_token"]),
                },
            },
            {
                "id": "moe_reduce",
                "op": "moe_reduce",
                "inputs": ["moe_expert_outputs"],
                "outputs": ["moe_out" if has_shared_expert else "ffn_out"],
                "attrs": {
                    "hidden_size": int(feed_forward["hidden_size"]),
                    "experts_per_token": int(feed_forward["experts_per_token"]),
                    "routed_scaling_factor": float(routing["routed_scaling_factor"]),
                },
            },
        ]
        if has_shared_expert:
            shared_width = int(shared_intermediate_size)
            body.extend(
                [
                    {
                        "id": "shared_mlp_input_projection",
                        "op": "linear",
                        "inputs": ["ffn_norm_out"],
                        "outputs": ["shared_gate_up"],
                        "params": _linear_params("shared_mlp_input", parameters),
                    },
                    {
                        "id": "shared_mlp_split",
                        "op": "split",
                        "inputs": ["shared_gate_up"],
                        "outputs": ["shared_gate", "shared_up"],
                        "attrs": {"part_width": shared_width},
                    },
                    {
                        "id": "shared_mlp_activation",
                        "op": "silu_multiply",
                        "inputs": ["shared_gate", "shared_up"],
                        "outputs": ["shared_hidden"],
                        "attrs": {"element_count": shared_width},
                    },
                    {
                        "id": "shared_mlp_output_projection",
                        "op": "linear",
                        "inputs": ["shared_hidden"],
                        "outputs": ["shared_out"],
                        "params": _linear_params("shared_mlp_output", parameters),
                    },
                    *(
                        [
                            {
                                "id": "shared_expert_gate_projection",
                                "op": "linear",
                                "inputs": ["ffn_norm_out"],
                                "outputs": ["shared_gate_logit"],
                                "params": _linear_params("shared_mlp_gate", parameters),
                            },
                            {
                                "id": "shared_expert_gate",
                                "op": "sigmoid_scalar_multiply",
                                "inputs": ["shared_out", "shared_gate_logit"],
                                "outputs": ["gated_shared_out"],
                            },
                        ]
                        if "shared_mlp_gate" in parameters
                        else []
                    ),
                    {
                        "id": "shared_and_sparse_expert_add",
                        "op": "residual_add",
                        "inputs": [
                            "moe_out",
                            (
                                "gated_shared_out"
                                if "shared_mlp_gate" in parameters
                                else "shared_out"
                            ),
                        ],
                        "outputs": ["ffn_out"],
                    },
                ]
            )
    else:
        if "ffn_gate_up" in parameters:
            body = [
                {
                    "id": "ffn_gate_up_projection",
                    "op": "linear",
                    "inputs": ["ffn_norm_out"],
                    "outputs": ["ffn_gate_up"],
                    "params": _linear_params("ffn_gate_up", parameters),
                },
                {
                    "id": "ffn_gate_up_split",
                    "op": "split",
                    "inputs": ["ffn_gate_up"],
                    "outputs": ["ffn_gate", "ffn_up"],
                    "attrs": {"part_width": int(feed_forward["intermediate_size"])},
                },
            ]
        else:
            body = [
                {
                    "id": "ffn_gate_projection",
                    "op": "linear",
                    "inputs": ["ffn_norm_out"],
                    "outputs": ["ffn_gate"],
                    "params": _linear_params("ffn_gate", parameters),
                },
                {
                    "id": "ffn_up_projection",
                    "op": "linear",
                    "inputs": ["ffn_norm_out"],
                    "outputs": ["ffn_up"],
                    "params": _linear_params("ffn_up", parameters),
                },
            ]
        body.extend(
            [
                {
                    "id": "ffn_gate_activation",
                    "op": str(feed_forward["activation"]),
                    "inputs": ["ffn_gate"],
                    "outputs": ["ffn_gate_activated"],
                    "attrs": {"element_count": int(feed_forward["intermediate_size"])},
                },
                {
                    "id": "ffn_gate_multiply",
                    "op": "multiply",
                    "inputs": ["ffn_gate_activated", "ffn_up"],
                    "outputs": ["ffn_hidden"],
                },
                {
                    "id": "ffn_down_projection",
                    "op": "linear",
                    "inputs": ["ffn_hidden"],
                    "outputs": ["ffn_out"],
                    "params": _linear_params("ffn_down", parameters),
                },
            ]
        )
    ffn_residual_update = "ffn_out"
    ffn_post_norm: list[Json] = []
    if "ffn_post_norm" in parameters:
        ffn_post_norm = [
            {
                "id": "ffn_post_norm",
                "op": "rms_norm",
                "inputs": ["ffn_out"],
                "outputs": ["ffn_post_norm_out"],
                "params": ["ffn_post_norm"],
                "attrs": _norm_attrs(numerics),
            }
        ]
        ffn_residual_update = "ffn_post_norm_out"

    per_layer_width = numerics.get("per_layer_input_width")
    has_layer_scalar = "layer_scalar" in parameters
    ffn_residual_output = (
        "ffn_residual_out"
        if per_layer_width is not None or has_layer_scalar
        else "output_frame"
    )
    tail: list[Json] = [
        *prefix,
        *body,
        *ffn_post_norm,
        _residual_node(
            node_id="ffn_residual",
            residual="operator_residual_out",
            update=ffn_residual_update,
            output=ffn_residual_output,
            scale=residual_scale,
        ),
    ]
    if per_layer_width is None and not has_layer_scalar:
        return tail

    hidden_size = int(feed_forward["hidden_size"])
    if per_layer_width is not None:
        width = int(per_layer_width)
        per_layer_embedding_chunks = sorted(
            (
                name
                for name in parameters
                if name.startswith("per_layer_embedding_chunk_")
            ),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if not per_layer_embedding_chunks:
            raise ValueError(
                "per-layer input requires compiled per-layer embedding chunks"
            )
        tail.extend(
            [
                {
                    "id": "per_layer_embedding",
                    "op": "per_layer_embedding",
                    "inputs": [],
                    "outputs": ["per_layer_input"],
                    "params": [
                        "token_embedding",
                        *per_layer_embedding_chunks,
                        "per_layer_model_projection",
                        "per_layer_projection_norm",
                    ],
                    "attrs": {
                        "hidden_size": hidden_size,
                        "per_layer_width": width,
                        "layer_index": int(numerics["per_layer_input_layer_index"]),
                        "layer_count": int(numerics["per_layer_input_layer_count"]),
                        "embedding_chunk_count": int(
                            numerics["per_layer_embedding_chunk_count"]
                        ),
                        "embedding_chunk_rows": int(
                            numerics["per_layer_embedding_chunk_rows"]
                        ),
                        "norm_eps": float(numerics["rms_norm_eps"]),
                        "token_embedding_scale": float(
                            numerics["token_embedding_scale"]
                        ),
                        "per_layer_embedding_scale": float(
                            numerics["per_layer_embedding_scale"]
                        ),
                        "model_projection_scale": float(
                            numerics["per_layer_model_projection_scale"]
                        ),
                        "combination_scale": float(numerics["per_layer_input_scale"]),
                    },
                },
                {
                    "id": "per_layer_input_gate",
                    "op": "linear",
                    "inputs": ["ffn_residual_out"],
                    "outputs": ["per_layer_gate"],
                    "params": ["per_layer_input_gate"],
                },
                {
                    "id": "per_layer_gate_activation",
                    "op": str(feed_forward["activation"]),
                    "inputs": ["per_layer_gate"],
                    "outputs": ["per_layer_gate_activated"],
                    "attrs": {"element_count": width},
                },
                {
                    "id": "per_layer_gate_multiply",
                    "op": "multiply",
                    "inputs": ["per_layer_gate_activated", "per_layer_input"],
                    "outputs": ["per_layer_gated"],
                    "attrs": {"element_count": width},
                },
                {
                    "id": "per_layer_projection",
                    "op": "linear",
                    "inputs": ["per_layer_gated"],
                    "outputs": ["per_layer_projected"],
                    "params": ["per_layer_projection"],
                },
                {
                    "id": "per_layer_post_norm",
                    "op": "rms_norm",
                    "inputs": ["per_layer_projected"],
                    "outputs": ["per_layer_normed"],
                    "params": ["per_layer_post_norm"],
                    "attrs": _norm_attrs(numerics),
                },
                {
                    "id": "per_layer_residual",
                    "op": "residual_add",
                    "inputs": ["ffn_residual_out", "per_layer_normed"],
                    "outputs": [
                        "per_layer_residual_out" if has_layer_scalar else "output_frame"
                    ],
                },
            ]
        )
    if has_layer_scalar:
        tail.append(
            {
                "id": "layer_scale",
                "op": "scalar_multiply",
                "inputs": [
                    "per_layer_residual_out"
                    if per_layer_width is not None
                    else "ffn_residual_out"
                ],
                "outputs": ["output_frame"],
                "params": ["layer_scalar"],
                "attrs": {"element_count": hidden_size},
            }
        )
    return tail

__all__ = [name for name in globals() if not name.startswith("__")]
