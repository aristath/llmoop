from nerve.circuit_lowering_common import *

def _residual_node(
    *, node_id: str, residual: str, update: str, output: str, scale: float
) -> Json:
    node: Json = {
        "id": node_id,
        "op": "residual_add" if scale == 1.0 else "scaled_residual_add",
        "inputs": [residual, update],
        "outputs": [output],
    }
    if scale != 1.0:
        node["attrs"] = {"scale": scale}
    return node


def _linear_params(weight_id: str, parameters: Json) -> list[str]:
    result = [weight_id]
    scale_id = f"{weight_id}_scale_inv"
    if scale_id in parameters:
        result.append(scale_id)
    qzeros_id = f"{weight_id}_qzeros"
    scales_id = f"{weight_id}_scales"
    if qzeros_id in parameters:
        if scales_id not in parameters:
            raise ValueError(
                f"packed linear parameter {weight_id!r} has incomplete quantization metadata"
            )
        result.extend((qzeros_id, scales_id))
    elif scales_id in parameters:
        result.append(scales_id)
    bias_id = f"{weight_id}_bias"
    if bias_id in parameters:
        result.append(bias_id)
    return result


def _attention_heads_from_state(pedal: Json) -> Json:
    state = pedal["state_ports"][0]
    kv_heads, head_width = state["key_shape_per_token"]
    hidden_size = pedal["ports"]["inputs"][0]["shape"][0]
    query_heads = int(state.get("query_heads") or hidden_size // head_width)
    return {
        "query_heads": query_heads,
        "key_value_heads": kv_heads,
        "head_width": head_width,
        "query_groups_per_kv_head": query_heads // kv_heads,
    }


def _state_port_for_circuit(port: Json, operator_type: str) -> Json:
    state = dict(port)
    state.setdefault("owner", "stream")
    if operator_type == "conv":
        state.setdefault("layout", "time_hidden")
        state.setdefault("source_layout", "batch_hidden_time")
    elif operator_type == "full_attention":
        state.setdefault("layout", "append_only_kv")
        state.setdefault("source_layout", "batch_kvheads_seq_headdim")
    elif operator_type == "gated_delta":
        state.setdefault(
            "layout",
            "channel_time" if state["id"] == "conv_state" else "head_key_value",
        )
        state.setdefault("source_layout", state["layout"])
    elif operator_type == "rg_lru":
        state.setdefault(
            "layout", "channel_time" if state["id"] == "conv_state" else "channel"
        )
        state.setdefault("source_layout", state["layout"])
    return state


def _param_ref(name: str, ref: Json) -> Json:
    result = dict(ref)
    result["role"] = _param_role(name)
    return result


def _param_role(name: str) -> str:
    roles = {
        "operator_norm": "operator_normalization_weight",
        "operator_post_norm": "operator_post_normalization_weight",
        "ffn_norm": "feed_forward_normalization_weight",
        "ffn_post_norm": "feed_forward_post_normalization_weight",
        "ffn_gate": "feed_forward_swiglu_gate_projection",
        "ffn_gate_bias": "feed_forward_swiglu_gate_projection_bias",
        "ffn_gate_up": "feed_forward_fused_gate_up_projection",
        "ffn_gate_up_bias": "feed_forward_fused_gate_up_projection_bias",
        "ffn_down": "feed_forward_down_projection",
        "ffn_down_bias": "feed_forward_down_projection_bias",
        "ffn_up": "feed_forward_up_projection",
        "ffn_up_bias": "feed_forward_up_projection_bias",
        "moe_router": "mixture_of_experts_router_projection",
        "moe_router_correction_bias": "mixture_of_experts_router_selection_bias",
        "moe_input": "mixture_of_experts_gate_up_weights",
        "moe_output": "mixture_of_experts_down_weights",
        "shared_mlp_input": "shared_expert_gate_up_projection",
        "shared_mlp_output": "shared_expert_down_projection",
        "shared_mlp_gate": "shared_expert_output_gate_projection",
        "conv_in_projection": "short_convolution_input_projection",
        "conv_depthwise_kernel": "short_convolution_depthwise_temporal_kernel",
        "conv_out_projection": "short_convolution_output_projection",
        "q_projection": "attention_query_projection",
        "q_projection_bias": "attention_query_projection_bias",
        "qkv_projection": "attention_fused_query_key_value_projection",
        "qkv_projection_bias": "attention_fused_query_key_value_projection_bias",
        "k_projection": "attention_key_projection",
        "k_projection_bias": "attention_key_projection_bias",
        "v_projection": "attention_value_projection",
        "v_projection_bias": "attention_value_projection_bias",
        "attention_out_projection": "attention_output_projection",
        "attention_out_projection_bias": "attention_output_projection_bias",
        "attention_gate_projection": "attention_gate_projection",
        "attention_gate_projection_bias": "attention_gate_projection_bias",
        "attention_sinks": "attention_sink_logits",
        "q_norm": "attention_query_head_normalization",
        "k_norm": "attention_key_head_normalization",
        "token_embedding": "token_embedding_for_per_layer_input",
        "per_layer_model_projection": "packed_per_layer_context_projection",
        "per_layer_projection_norm": "per_layer_context_projection_normalization",
        "per_layer_input_gate": "per_layer_residual_gate_projection",
        "per_layer_projection": "per_layer_residual_output_projection",
        "per_layer_post_norm": "per_layer_residual_post_normalization",
        "layer_scalar": "layer_output_scalar",
        "delta_qkv_projection": "gated_delta_query_key_value_projection",
        "delta_z_projection": "gated_delta_output_gate_projection",
        "delta_b_projection": "gated_delta_beta_projection",
        "delta_a_projection": "gated_delta_decay_projection",
        "delta_conv_kernel": "gated_delta_depthwise_convolution_kernel",
        "delta_a_log": "gated_delta_decay_parameter",
        "delta_dt_bias": "gated_delta_time_bias",
        "delta_norm": "gated_delta_output_normalization_weight",
        "delta_out_projection": "gated_delta_output_projection",
        "rg_lru_x_projection": "real_gated_recurrence_x_projection",
        "rg_lru_x_projection_bias": "real_gated_recurrence_x_projection_bias",
        "rg_lru_y_projection": "real_gated_recurrence_y_projection",
        "rg_lru_y_projection_bias": "real_gated_recurrence_y_projection_bias",
        "rg_lru_out_projection": "real_gated_recurrence_output_projection",
        "rg_lru_out_projection_bias": "real_gated_recurrence_output_projection_bias",
        "rg_lru_conv_kernel": "real_gated_recurrence_depthwise_convolution_kernel",
        "rg_lru_conv_bias": "real_gated_recurrence_depthwise_convolution_bias",
        "rg_lru_input_gate_weight": "real_gated_recurrence_input_gate_weight",
        "rg_lru_input_gate_bias": "real_gated_recurrence_input_gate_bias",
        "rg_lru_recurrent_gate_weight": "real_gated_recurrence_recurrent_gate_weight",
        "rg_lru_recurrent_gate_bias": "real_gated_recurrence_recurrent_gate_bias",
        "rg_lru_recurrent_param": "real_gated_recurrence_parameter",
    }
    if name.endswith("_scale_inv"):
        weight_id = name.removesuffix("_scale_inv")
        return f"{roles[weight_id]}_block_scale_inverse"
    if name.endswith("_qzeros"):
        weight_id = name.removesuffix("_qzeros")
        return f"{roles[weight_id]}_packed_zero_points"
    if name.endswith("_scales"):
        weight_id = name.removesuffix("_scales")
        return f"{roles[weight_id]}_group_scales"
    if name.startswith("per_layer_embedding_chunk_"):
        return "packed_per_layer_token_embedding_chunk"
    return roles[name]


def _norm_attrs(numerics: Json) -> Json:
    return {
        "eps": float(numerics["rms_norm_eps"]),
        "weight_offset": float(numerics["rms_norm_weight_offset"]),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
