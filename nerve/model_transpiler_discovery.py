from nerve.model_transpiler_types import *
from nerve.model_transpiler_tensor_index import *
from nerve.model_transpiler_quantization import (
    add_optional_linear_biases,
    attach_block_quantization_scales,
    attach_packed_linear_quantization,
    synthesize_packed_expert_tensors,
)

def discover_model_structure(
    model_dir: Path,
    config: Json,
    tensors: dict[str, Json],
    *,
    generation_config: Json | None = None,
) -> ModelStructure:
    layer_root, layer_indices = discover_layer_root(tensors, config=config)
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
    num_attention_heads = discover_int_config(
        decoder_config,
        "num_attention_heads",
        "n_head",
        "num_heads",
        role="number of attention heads",
    )
    configured_num_key_value_heads = int(
        decoder_config.get("num_key_value_heads")
        or decoder_config.get("num_kv_heads")
        or decoder_config.get("multi_query_group_num")
        or num_attention_heads
    )
    configured_head_width = int(
        decoder_config.get("head_dim") or hidden_size // num_attention_heads
    )
    per_layer_attention_heads = discover_per_layer_attention_heads(
        decoder_config,
        layer_count=layer_count,
        default=num_attention_heads,
    )
    attention_gate_activation = discover_attention_gate_activation(
        model_dir, decoder_config
    )
    configured_layer_types = discover_configured_layer_types(
        decoder_config, layer_count
    )
    attention_window_size = discover_attention_window_size(decoder_config)
    per_layer_input = discover_per_layer_input_structure(
        decoder_config, tensors, model_prefix, layer_count
    )
    shared_kv_sources = discover_shared_kv_sources(
        decoder_config, configured_layer_types, layer_count
    )
    layers = tuple(
        discover_layer_structure(
            tensors=tensors,
            decoder_config=decoder_config,
            configured_layer_types=configured_layer_types,
            configured_attention_window_size=attention_window_size,
            num_attention_heads=per_layer_attention_heads[index],
            configured_num_key_value_heads=configured_num_key_value_heads,
            configured_head_width=configured_head_width,
            configured_attention_gate_activation=attention_gate_activation,
            shared_kv_source_layer=shared_kv_sources.get(index),
            per_layer_input=per_layer_input,
            token_embedding=token_embedding,
            layer_root=layer_root,
            layer_index=index,
        )
        for index in range(layer_count)
    )
    draft_pedalboards = discover_draft_pedalboards(
        tensors=tensors,
        decoder_config=decoder_config,
        primary_layer_root=layer_root,
        hidden_size=hidden_size,
        token_embedding=token_embedding,
        output_projection=output_projection,
        num_attention_heads=num_attention_heads,
        configured_num_key_value_heads=configured_num_key_value_heads,
        configured_head_width=configured_head_width,
        configured_attention_window_size=attention_window_size,
    )

    first_attention = next(
        (layer for layer in layers if layer.operator_type == "full_attention"), None
    )
    num_key_value_heads = (
        first_attention.num_key_value_heads
        if first_attention is not None
        else configured_num_key_value_heads
    )
    head_width = (
        first_attention.head_width
        if first_attention is not None
        else configured_head_width
    )
    rotary_width = (
        first_attention.rotary_width
        if first_attention is not None
        else configured_head_width
    )
    attention_output_gate = discover_attention_output_gate(
        decoder_config,
        tensors,
        layers,
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
    moe_routing = (
        discover_moe_routing(model_dir, decoder_config) if has_sparse_experts else None
    )

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
        rope_theta=(
            first_attention.rope_theta
            if first_attention is not None
            else discover_rope_theta(decoder_config)
        ),
        rope_interleaved=bool(decoder_config.get("rope_interleaved", False)),
        rms_norm_weight_offset=rms_norm_weight_offset,
        embedding_scale=discover_embedding_scale(
            decoder_config,
            hidden_size,
            scaled_by_structure=per_layer_input is not None
            or any("layer_scalar" in layer.tensors for layer in layers),
        ),
        residual_scale=float(decoder_config.get("residual_multiplier", 1.0)),
        attention_scale=(
            first_attention.attention_scale
            if first_attention is not None
            else float(decoder_config.get("attention_multiplier", head_width**-0.5))
        ),
        logits_scale=float(decoder_config.get("logits_scaling", 1.0)),
        logits_soft_cap=(
            float(
                decoder_config.get("logits_soft_cap")
                if decoder_config.get("logits_soft_cap") is not None
                else decoder_config["final_logit_softcapping"]
            )
            if decoder_config.get("logits_soft_cap") is not None
            or decoder_config.get("final_logit_softcapping") is not None
            else None
        ),
        activation=discover_activation(decoder_config),
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        moe_routing=moe_routing,
        recurrent_mixer=recurrent_mixer,
        quantization=discover_quantization_policy(config),
        sampling=discover_sampling_policy(generation_config or {}),
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
        draft_pedalboards=draft_pedalboards,
    )


def discover_layer_structure(
    *,
    tensors: dict[str, Json],
    decoder_config: Json,
    configured_layer_types: list[str] | None,
    configured_attention_window_size: int | None,
    num_attention_heads: int,
    configured_num_key_value_heads: int,
    configured_head_width: int,
    configured_attention_gate_activation: str | None,
    shared_kv_source_layer: int | None,
    per_layer_input: Json | None,
    token_embedding: str,
    layer_root: str,
    layer_index: int,
) -> LayerStructure:
    prefix = f"{layer_root}.{layer_index}"
    configured = configured_layer_types[layer_index] if configured_layer_types else None
    has_explicit_feed_forward_pre_norm = (
        find_optional_layer_tensor(tensors, prefix, FFN_PRE_NORM_SUFFIXES) is not None
    )
    layer_tensors = {
        "operator_norm": find_layer_tensor(
            tensors, prefix, OPERATOR_NORM_SUFFIXES, role="operator norm"
        ),
        "ffn_norm": find_layer_tensor(
            tensors,
            prefix,
            FFN_PRE_NORM_SUFFIXES
            if has_explicit_feed_forward_pre_norm
            else FFN_NORM_SUFFIXES,
            role="feed-forward norm",
        ),
    }
    if has_explicit_feed_forward_pre_norm:
        layer_tensors["operator_post_norm"] = find_layer_tensor(
            tensors,
            prefix,
            OPERATOR_POST_NORM_SUFFIXES,
            role="post-attention norm",
        )
        optional_ffn_post_norm = find_optional_layer_tensor(
            tensors, prefix, FFN_POST_NORM_SUFFIXES
        )
        if optional_ffn_post_norm is not None:
            layer_tensors["ffn_post_norm"] = optional_ffn_post_norm

    per_layer_input_width = None
    if per_layer_input is not None:
        per_layer_input_width = int(per_layer_input["width"])
        layer_tensors.update(
            {
                "token_embedding": token_embedding,
                "per_layer_embedding": str(per_layer_input["embedding"]),
                "per_layer_model_projection": str(per_layer_input["model_projection"]),
                "per_layer_projection_norm": str(per_layer_input["projection_norm"]),
                "per_layer_input_gate": find_layer_tensor(
                    tensors,
                    prefix,
                    PER_LAYER_INPUT_GATE_SUFFIXES,
                    role="per-layer input gate",
                ),
                "per_layer_projection": find_layer_tensor(
                    tensors,
                    prefix,
                    PER_LAYER_PROJECTION_SUFFIXES,
                    role="per-layer residual projection",
                ),
                "per_layer_post_norm": find_layer_tensor(
                    tensors,
                    prefix,
                    PER_LAYER_POST_NORM_SUFFIXES,
                    role="per-layer residual post norm",
                ),
                "layer_scalar": find_layer_tensor(
                    tensors,
                    prefix,
                    LAYER_SCALAR_SUFFIXES,
                    role="layer output scalar",
                ),
            }
        )
    else:
        optional_layer_scalar = find_optional_layer_tensor(
            tensors, prefix, LAYER_SCALAR_SUFFIXES
        )
        if optional_layer_scalar is not None:
            layer_tensors["layer_scalar"] = optional_layer_scalar
    synthesize_packed_expert_tensors(tensors, prefix)
    dense_gate = find_optional_layer_tensor(tensors, prefix, FFN_GATE_SUFFIXES)
    fused_gate_up = find_optional_layer_tensor(
        tensors, prefix, FFN_FUSED_GATE_UP_SUFFIXES
    )
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
    elif fused_gate_up is not None:
        feed_forward_type = "dense_swiglu"
        layer_tensors.update(
            {
                "ffn_gate_up": fused_gate_up,
                "ffn_down": find_layer_tensor(
                    tensors,
                    prefix,
                    FFN_DOWN_SUFFIXES,
                    role="feed-forward down projection",
                ),
            }
        )
        add_optional_linear_biases(
            tensors,
            layer_tensors,
            ("ffn_gate_up", "ffn_down"),
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
        router_correction_bias = find_optional_layer_tensor(
            tensors, prefix, MOE_ROUTER_CORRECTION_BIAS_SUFFIXES
        )
        if router_correction_bias is not None:
            layer_tensors["moe_router_correction_bias"] = router_correction_bias
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
            shared_gate = find_optional_layer_tensor(
                tensors, prefix, SHARED_MLP_GATE_SUFFIXES
            )
            if shared_gate is not None:
                layer_tensors["shared_mlp_gate"] = shared_gate
    else:
        raise ModelTranspileError(
            f"could not discover feed-forward structure for layer prefix {prefix!r}"
        )

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

    attention_key_equals_value = False
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
        head_width = discover_layer_head_width(
            decoder_config,
            configured,
            configured_head_width=configured_head_width,
        )
        fused_qkv = find_optional_layer_tensor(
            tensors, prefix, ATTENTION_FUSED_QKV_SUFFIXES
        )
        if fused_qkv is not None:
            layer_tensors["qkv_projection"] = fused_qkv
        else:
            layer_tensors["q_projection"] = find_layer_tensor(
                tensors,
                prefix,
                ATTENTION_Q_PROJECTION_SUFFIXES,
                role="attention query projection",
            )
            if shared_kv_source_layer is None:
                layer_tensors["k_projection"] = find_layer_tensor(
                    tensors,
                    prefix,
                    ATTENTION_K_PROJECTION_SUFFIXES,
                    role="attention key projection",
                )
                value_projection = find_optional_layer_tensor(
                    tensors, prefix, ATTENTION_V_PROJECTION_SUFFIXES
                )
                if value_projection is None:
                    if not bool(decoder_config.get("attention_k_eq_v", False)):
                        raise ModelTranspileError(
                            f"attention layer prefix {prefix!r} has no value projection"
                        )
                    attention_key_equals_value = True
                else:
                    layer_tensors["v_projection"] = value_projection
        layer_tensors["attention_out_projection"] = find_layer_tensor(
            tensors,
            prefix,
            ATTENTION_OUT_PROJECTION_SUFFIXES,
            role="attention output projection",
        )
        optional_q_norm = find_optional_layer_tensor(
            tensors, prefix, ATTENTION_Q_NORM_SUFFIXES
        )
        optional_k_norm = (
            find_optional_layer_tensor(tensors, prefix, ATTENTION_K_NORM_SUFFIXES)
            if shared_kv_source_layer is None
            else None
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
        optional_gate_projection = find_optional_layer_tensor(
            tensors, prefix, ATTENTION_GATE_PROJECTION_SUFFIXES
        )
        if optional_gate_projection is not None:
            layer_tensors["attention_gate_projection"] = optional_gate_projection
        attention_linear_ids: tuple[str, ...] = (
            ("qkv_projection", "attention_out_projection")
            if fused_qkv is not None
            else ("q_projection", "attention_out_projection")
            if shared_kv_source_layer is not None
            else tuple(
                parameter_id
                for parameter_id in (
                    "q_projection",
                    "k_projection",
                    "v_projection",
                    "attention_out_projection",
                    "attention_gate_projection",
                )
                if parameter_id in layer_tensors
            )
        )
        if (
            "attention_gate_projection" in layer_tensors
            and "attention_gate_projection" not in attention_linear_ids
        ):
            attention_linear_ids = (
                *attention_linear_ids,
                "attention_gate_projection",
            )
        add_optional_linear_biases(tensors, layer_tensors, attention_linear_ids)
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

    if operator_type == "full_attention":
        head_width = discover_layer_head_width(
            decoder_config,
            configured,
            configured_head_width=configured_head_width,
        )
        q_projection = layer_tensors.get("q_projection")
        if q_projection is not None:
            q_width = tensor_matrix_shape(tensors, q_projection)[0]
            configured_output_gate = bool(decoder_config.get("attn_output_gate", False))
            expected_q_width = num_attention_heads * head_width
            if q_width == expected_q_width * 2:
                configured_output_gate = True
            if q_width != expected_q_width * (2 if configured_output_gate else 1):
                raise ModelTranspileError(
                    f"attention query projection width {q_width} is incompatible with "
                    f"{num_attention_heads} heads of width {head_width}"
                )
        num_key_value_heads = configured_num_key_value_heads
        if shared_kv_source_layer is None and "k_projection" in layer_tensors:
            k_width = tensor_matrix_shape(tensors, layer_tensors["k_projection"])[0]
            if k_width % head_width:
                raise ModelTranspileError(
                    f"attention key projection width {k_width} is not divisible by head width {head_width}"
                )
            num_key_value_heads = k_width // head_width
        rope_parameters = discover_layer_rope_parameters(decoder_config, configured)
        rope_theta = float(rope_parameters["rope_theta"])
        rope_type = str(rope_parameters.get("rope_type", "default"))
        rotary_width = int(
            head_width * float(rope_parameters.get("partial_rotary_factor", 1.0))
        )
        rope_scaling = compile_rope_scaling(rope_parameters, rotary_width)
        attention_scale = float(
            decoder_config.get(
                "attention_multiplier",
                1.0
                if per_layer_input is not None or "attention_k_eq_v" in decoder_config
                else head_width**-0.5,
            )
        )
        value_head_norm = (
            (per_layer_input is not None or "attention_k_eq_v" in decoder_config)
            and shared_kv_source_layer is None
            and "q_norm" in layer_tensors
            and "k_norm" in layer_tensors
        )
        attention_gate_activation = None
        attention_gate_per_head = False
        if "attention_gate_projection" in layer_tensors:
            if configured_attention_gate_activation is None:
                raise ModelTranspileError(
                    f"could not discover activation for attention gate in layer {layer_index}"
                )
            gate_width = tensor_matrix_shape(
                tensors, layer_tensors["attention_gate_projection"]
            )[0]
            attention_gate_per_head = gate_width == num_attention_heads
            if not attention_gate_per_head and gate_width != expected_q_width:
                raise ModelTranspileError(
                    f"attention gate width {gate_width} is incompatible with "
                    f"{num_attention_heads} heads of width {head_width}"
                )
            attention_gate_activation = configured_attention_gate_activation
    else:
        head_width = configured_head_width
        num_key_value_heads = configured_num_key_value_heads
        rotary_width = configured_head_width
        rope_theta = discover_rope_theta(decoder_config)
        rope_type = "default"
        rope_scaling = None
        attention_scale = float(
            decoder_config.get("attention_multiplier", configured_head_width**-0.5)
        )
        value_head_norm = False
        attention_gate_activation = None
        attention_gate_per_head = False

    attach_block_quantization_scales(tensors, layer_tensors)
    attach_packed_linear_quantization(tensors, layer_tensors)

    return LayerStructure(
        index=layer_index,
        prefix=prefix,
        operator_type=operator_type,
        attention_window_size=layer_attention_window_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_width=head_width,
        rotary_width=rotary_width,
        rope_theta=rope_theta,
        rope_type=rope_type,
        rope_scaling=rope_scaling,
        attention_scale=attention_scale,
        attention_gate_activation=attention_gate_activation,
        attention_gate_per_head=attention_gate_per_head,
        attention_key_equals_value=attention_key_equals_value,
        value_head_norm=value_head_norm,
        shared_kv_source_layer=shared_kv_source_layer,
        per_layer_input_width=per_layer_input_width,
        feed_forward_type=feed_forward_type,
        intermediate_size=discover_layer_intermediate_size(tensors, layer_tensors),
        shared_intermediate_size=(
            tensor_matrix_shape(tensors, layer_tensors["shared_mlp_input"])[0] // 2
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
    ) or find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in ATTENTION_FUSED_QKV_SUFFIXES)
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


def discover_layer_root(
    tensors: dict[str, Json], *, config: Json | None = None
) -> tuple[str, tuple[int, ...]]:
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
    if config is None:
        root = roots.most_common(1)[0][0]
        return root, tuple(sorted(root_indices[root]))

    decoder_layer_counts = discover_configured_layer_counts(config)

    def score(root: str) -> tuple[int, int]:
        indices = root_indices[root]
        model_prefix = root.removesuffix(".layers")
        contiguous = indices == set(range(max(indices) + 1))
        boundary_score = 0
        if f"{model_prefix}.embed_tokens.weight" in tensors:
            boundary_score += 200
        if f"{model_prefix}.norm.weight" in tensors:
            boundary_score += 50
        layer_count_score = (
            100 if len(indices) in decoder_layer_counts and contiguous else 0
        )
        first_prefix = f"{root}.{min(indices)}"
        decoder_operator_score = 0
        if find_first_existing_tensor(
            tensors,
            (f"{first_prefix}.{suffix}" for suffix in FFN_GATE_SUFFIXES),
        ) or find_first_existing_tensor(
            tensors,
            (f"{first_prefix}.{suffix}" for suffix in FFN_FUSED_GATE_UP_SUFFIXES),
        ):
            decoder_operator_score += 25
        if infer_optional_operator_type(tensors, first_prefix) is not None:
            decoder_operator_score += 25
        return (
            boundary_score + layer_count_score + decoder_operator_score,
            roots[root],
        )

    root = max(roots, key=score)
    return root, tuple(sorted(root_indices[root]))


def discover_draft_pedalboards(
    *,
    tensors: dict[str, Json],
    decoder_config: Json,
    primary_layer_root: str,
    hidden_size: int,
    token_embedding: str,
    output_projection: str,
    num_attention_heads: int,
    configured_num_key_value_heads: int,
    configured_head_width: int,
    configured_attention_window_size: int | None,
) -> tuple[DraftPedalboardStructure, ...]:
    """Discover auxiliary next-token predictors from their tensor topology.

    The checkpoint family name is deliberately irrelevant. A candidate is an
    auxiliary repeated layer stack whose parent also owns the two normalizers
    and 2H-to-H projection needed to combine a token embedding with a target
    hidden frame.
    """
    roots: dict[str, set[int]] = {}
    for name in tensors:
        for pattern in LAYER_ROOT_PATTERNS:
            match = pattern.match(name)
            if match is not None:
                roots.setdefault(match.group("root"), set()).add(
                    int(match.group("index"))
                )
                break

    discovered: list[DraftPedalboardStructure] = []
    for root in sorted(roots):
        if root == primary_layer_root:
            continue
        indices = sorted(roots[root])
        if indices != list(range(len(indices))):
            continue
        prefix = root.removesuffix(".layers")
        input_projection = find_first_existing_tensor(
            tensors,
            (f"{prefix}.{suffix}" for suffix in DRAFT_INPUT_PROJECTION_SUFFIXES),
        )
        embedding_norm = find_first_existing_tensor(
            tensors,
            (f"{prefix}.{suffix}" for suffix in DRAFT_EMBEDDING_NORM_SUFFIXES),
        )
        hidden_norm = find_first_existing_tensor(
            tensors,
            (f"{prefix}.{suffix}" for suffix in DRAFT_HIDDEN_NORM_SUFFIXES),
        )
        output_norm = find_first_existing_tensor(
            tensors,
            (f"{prefix}.{suffix}" for suffix in DRAFT_OUTPUT_NORM_SUFFIXES),
        )
        if None in (input_projection, embedding_norm, hidden_norm, output_norm):
            continue
        assert input_projection is not None
        assert embedding_norm is not None
        assert hidden_norm is not None
        assert output_norm is not None
        if tensor_matrix_shape(tensors, input_projection) != [
            hidden_size,
            hidden_size * 2,
        ]:
            continue
        if any(
            tensors[name].get("shape") != [hidden_size]
            for name in (embedding_norm, hidden_norm, output_norm)
        ):
            continue

        layers = tuple(
            discover_layer_structure(
                tensors=tensors,
                decoder_config=decoder_config,
                configured_layer_types=None,
                configured_attention_window_size=configured_attention_window_size,
                num_attention_heads=num_attention_heads,
                configured_num_key_value_heads=configured_num_key_value_heads,
                configured_head_width=configured_head_width,
                configured_attention_gate_activation=None,
                shared_kv_source_layer=None,
                per_layer_input=None,
                token_embedding=token_embedding,
                layer_root=root,
                layer_index=index,
            )
            for index in indices
        )
        adapter_tensors = {
            "embedding_norm": embedding_norm,
            "hidden_norm": hidden_norm,
            "input_projection": input_projection,
            "output_norm": output_norm,
            "output_projection": output_projection,
        }
        attach_block_quantization_scales(tensors, adapter_tensors)
        attach_packed_linear_quantization(tensors, adapter_tensors)
        discovered.append(
            DraftPedalboardStructure(
                id=f"draft_{len(discovered):02d}",
                prefix=prefix,
                tensors=adapter_tensors,
                layers=layers,
            )
        )
    return tuple(discovered)


def discover_configured_layer_counts(config: Json) -> set[int]:
    counts: set[int] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        configured = value.get("num_hidden_layers")
        if configured is not None:
            counts.add(int(configured))
        for nested in value.values():
            if isinstance(nested, dict):
                visit(nested)

    visit(config)
    return counts


def infer_optional_operator_type(tensors: dict[str, Json], prefix: str) -> str | None:
    try:
        return infer_operator_type(tensors, prefix)
    except ModelTranspileError:
        return None


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


def discover_per_layer_input_structure(
    config: Json,
    tensors: dict[str, Json],
    model_prefix: str,
    layer_count: int,
) -> Json | None:
    names = {
        "embedding": f"{model_prefix}.embed_tokens_per_layer.weight",
        "model_projection": f"{model_prefix}.per_layer_model_projection.weight",
        "projection_norm": f"{model_prefix}.per_layer_projection_norm.weight",
    }
    present = {key: name in tensors for key, name in names.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = ", ".join(key for key, exists in present.items() if not exists)
        raise ModelTranspileError(
            f"per-layer input structure is incomplete; missing {missing}"
        )
    width = int(config.get("hidden_size_per_layer_input") or 0)
    if width <= 0:
        raise ModelTranspileError(
            "per-layer input tensors require a positive hidden_size_per_layer_input"
        )
    packed_width = layer_count * width
    embedding_shape = tensors[names["embedding"]]["shape"]
    projection_shape = tensors[names["model_projection"]]["shape"]
    norm_shape = tensors[names["projection_norm"]]["shape"]
    if int(embedding_shape[-1]) != packed_width:
        raise ModelTranspileError(
            f"packed per-layer embedding width {embedding_shape[-1]} does not equal {layer_count}x{width}"
        )
    if int(projection_shape[0]) != packed_width:
        raise ModelTranspileError(
            f"packed per-layer projection width {projection_shape[0]} does not equal {layer_count}x{width}"
        )
    if list(map(int, norm_shape)) != [width]:
        raise ModelTranspileError(
            f"per-layer projection norm shape {norm_shape} does not equal [{width}]"
        )
    return {**names, "width": width, "packed_width": packed_width}


def discover_shared_kv_sources(
    config: Json,
    configured_layer_types: list[str] | None,
    layer_count: int,
) -> dict[int, int]:
    shared_count = int(config.get("num_kv_shared_layers") or 0)
    if shared_count <= 0:
        return {}
    if shared_count >= layer_count:
        raise ModelTranspileError(
            f"num_kv_shared_layers {shared_count} leaves no source layer"
        )
    if configured_layer_types is None:
        raise ModelTranspileError(
            "shared KV layers require structural layer_types to identify their source pedals"
        )
    first_shared = layer_count - shared_count
    source_by_type: dict[str, int] = {}
    for index in range(first_shared):
        source_by_type[configured_layer_types[index]] = index
    result: dict[int, int] = {}
    for index in range(first_shared, layer_count):
        layer_type = configured_layer_types[index]
        if layer_type not in source_by_type:
            raise ModelTranspileError(
                f"shared KV layer {index} has no earlier source for type {layer_type!r}"
            )
        result[index] = source_by_type[layer_type]
    return result


def discover_layer_head_width(
    config: Json,
    configured_layer_type: str | None,
    *,
    configured_head_width: int,
) -> int:
    if configured_layer_type == "full_attention" and config.get("global_head_dim"):
        return int(config["global_head_dim"])
    return configured_head_width


def discover_per_layer_attention_heads(
    config: Json, *, layer_count: int, default: int
) -> tuple[int, ...]:
    configured = config.get("num_attention_heads_per_layer")
    if configured is None:
        return (default,) * layer_count
    if not isinstance(configured, list) or len(configured) != layer_count:
        raise ModelTranspileError(
            "num_attention_heads_per_layer must contain one entry per layer"
        )
    result = tuple(int(value) for value in configured)
    if any(value <= 0 for value in result):
        raise ModelTranspileError(
            "num_attention_heads_per_layer entries must be positive"
        )
    return result


def discover_attention_gate_activation(model_dir: Path, config: Json) -> str | None:
    configured = config.get("attention_gate_activation") or config.get(
        "attn_gate_activation"
    )
    if configured is not None:
        activation = str(configured).lower()
        if activation not in {"sigmoid", "softplus"}:
            raise ModelTranspileError(
                f"unsupported attention gate activation {configured!r}"
            )
        return activation

    discovered: set[str] = set()
    patterns = {
        "softplus": re.compile(
            r"(?:F|torch|nn\.functional)\.softplus\s*\(\s*self\.g_proj\s*\("
        ),
        "sigmoid": re.compile(
            r"(?:F|torch|nn\.functional)\.sigmoid\s*\(\s*self\.g_proj\s*\("
        ),
    }
    for source_file in sorted(model_dir.glob("*.py")):
        source = source_file.read_text(errors="replace")
        discovered.update(
            activation
            for activation, pattern in patterns.items()
            if pattern.search(source) is not None
        )
    if len(discovered) > 1:
        raise ModelTranspileError(
            f"model source contains ambiguous attention gate activations {sorted(discovered)}"
        )
    return next(iter(discovered), None)


def discover_moe_routing(model_dir: Path, config: Json) -> Json:
    configured = config.get("moe_router_activation") or config.get("scoring_func")
    activation = str(configured).lower() if configured is not None else None
    if activation is None:
        discovered: set[str] = set()
        patterns = {
            "sigmoid": re.compile(
                r"(?:F|torch|nn\.functional)\.sigmoid\s*\(\s*router_logits"
            ),
            "softmax": re.compile(
                r"(?:F|torch|nn\.functional)\.softmax\s*\(\s*router_logits"
            ),
        }
        for source_file in sorted(model_dir.glob("*.py")):
            source = source_file.read_text(errors="replace")
            discovered.update(
                candidate
                for candidate, pattern in patterns.items()
                if pattern.search(source) is not None
            )
        if len(discovered) > 1:
            raise ModelTranspileError(
                f"model source contains ambiguous MoE router activations {sorted(discovered)}"
            )
        activation = next(iter(discovered), "softmax")
    if activation not in {"sigmoid", "softmax"}:
        raise ModelTranspileError(f"unsupported MoE router activation {activation!r}")

    logit_softcap = float(config.get("moe_router_logit_softcapping") or 0.0)
    routed_scale = float(
        config.get("moe_routed_scaling_factor")
        or config.get("routed_scaling_factor")
        or 1.0
    )
    if logit_softcap < 0.0 or not math.isfinite(logit_softcap):
        raise ModelTranspileError(
            f"MoE router logit softcap must be finite and non-negative, got {logit_softcap}"
        )
    if routed_scale <= 0.0 or not math.isfinite(routed_scale):
        raise ModelTranspileError(
            f"MoE routed scaling factor must be finite and positive, got {routed_scale}"
        )
    return {
        "activation": activation,
        "normalize_selected": bool(config.get("norm_topk_prob", True)),
        "routed_scaling_factor": routed_scale,
        "logit_softcap": logit_softcap,
    }


def discover_layer_rope_parameters(
    config: Json, configured_layer_type: str | None
) -> Json:
    configured = config.get("rope_parameters")
    if isinstance(configured, dict):
        nested = (
            configured.get(configured_layer_type) if configured_layer_type else None
        )
        if isinstance(nested, dict):
            return nested
        if "rope_theta" in configured:
            return configured
    if config.get("rope_theta") is not None:
        return {
            "rope_theta": config["rope_theta"],
            "partial_rotary_factor": config.get("partial_rotary_factor", 1.0),
            "rope_type": "default",
        }
    raise ModelTranspileError(
        f"could not discover RoPE parameters for layer type {configured_layer_type!r}"
    )


def compile_rope_scaling(parameters: Json, rotary_width: int) -> Json | None:
    rope_type = str(parameters.get("rope_type", "default"))
    if rope_type in {"default", "proportional"}:
        return None
    if rope_type != "yarn":
        raise ModelTranspileError(f"unsupported RoPE type {rope_type!r}")
    if rotary_width <= 0 or rotary_width % 2:
        raise ModelTranspileError(
            f"YaRN rotary width must be positive and even, got {rotary_width}"
        )

    theta = float(parameters.get("rope_theta") or 0.0)
    factor = float(parameters.get("factor") or 0.0)
    original_context = int(parameters.get("original_max_position_embeddings") or 0)
    beta_fast = float(parameters.get("beta_fast") or 32.0)
    beta_slow = float(parameters.get("beta_slow") or 1.0)
    truncate = bool(parameters.get("truncate", True))
    if (
        not math.isfinite(theta)
        or theta <= 0.0
        or not math.isfinite(factor)
        or factor <= 0.0
        or original_context <= 0
        or not math.isfinite(beta_fast)
        or not math.isfinite(beta_slow)
        or beta_fast < beta_slow
    ):
        raise ModelTranspileError(
            "YaRN requires positive finite theta/factor/context and beta_fast >= beta_slow"
        )

    attention_factor_value = parameters.get("attention_factor")
    attention_factor = (
        float(attention_factor_value)
        if attention_factor_value is not None
        else 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0
    )
    if not math.isfinite(attention_factor) or attention_factor <= 0.0:
        raise ModelTranspileError(
            f"YaRN attention factor must be finite and positive, got {attention_factor}"
        )

    def correction_dimension(rotations: float) -> float:
        return (
            rotary_width
            * math.log(original_context / (rotations * 2.0 * math.pi))
            / (2.0 * math.log(theta))
        )

    correction_low = correction_dimension(beta_fast)
    correction_high = correction_dimension(beta_slow)
    if truncate:
        correction_low = math.floor(correction_low)
        correction_high = math.ceil(correction_high)
    correction_low = max(float(correction_low), 0.0)
    correction_high = min(float(correction_high), float(rotary_width - 1))
    if correction_high <= correction_low:
        raise ModelTranspileError(
            "YaRN correction range must have positive width, got "
            f"{correction_low}..{correction_high}"
        )
    return {
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_context,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
        "truncate": truncate,
        "attention_factor": attention_factor,
        "correction_low": correction_low,
        "correction_high": correction_high,
    }


def discover_attention_output_gate(
    config: Json,
    tensors: dict[str, Json],
    layers: tuple[LayerStructure, ...],
) -> bool:
    configured = config.get("attn_output_gate")
    for layer in layers:
        if layer.operator_type != "full_attention":
            continue
        expected_width = layer.num_attention_heads * layer.head_width
        kv_width = layer.num_key_value_heads * layer.head_width
        if "qkv_projection" in layer.tensors:
            projection_width = tensor_matrix_shape(
                tensors, layer.tensors["qkv_projection"]
            )[0]
            ordinary_width = expected_width + 2 * kv_width
            discovered = projection_width == ordinary_width + expected_width
            if projection_width not in (
                ordinary_width,
                ordinary_width + expected_width,
            ):
                raise ModelTranspileError(
                    f"fused attention QKV width {projection_width} is incompatible with "
                    f"{layer.num_attention_heads} query and {layer.num_key_value_heads} KV heads of "
                    f"width {layer.head_width}"
                )
        else:
            projection_width = tensor_matrix_shape(
                tensors, layer.tensors["q_projection"]
            )[0]
            discovered = projection_width == expected_width * 2
            if projection_width not in (expected_width, expected_width * 2):
                raise ModelTranspileError(
                    f"attention query projection width {projection_width} is incompatible with "
                    f"{layer.num_attention_heads} heads of width {layer.head_width}"
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


def discover_quantization_policy(config: Json) -> Json | None:
    """Normalize execution-relevant quantization facts from source metadata.

    Checkpoint families place ``quantization_config`` at different nesting
    levels.  The compiler records the numerical contract itself instead of
    coupling execution to a model name or configuration path.
    """

    candidates: list[Json] = []

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        quantization = value.get("quantization_config")
        if isinstance(quantization, dict):
            candidates.append(quantization)
        for nested in value.values():
            if isinstance(nested, dict):
                visit(nested)

    visit(config)
    fp8_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("quant_method", "")).lower() == "fp8"
    ]
    if not fp8_candidates:
        return None

    policies: list[Json] = []
    for candidate in fp8_candidates:
        block_shape = candidate.get("weight_block_size")
        if block_shape is None:
            continue
        if (
            not isinstance(block_shape, list)
            or len(block_shape) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in block_shape
            )
        ):
            raise ModelTranspileError(
                f"FP8 weight_block_size must contain two positive integers; got {block_shape!r}"
            )
        activation_scheme = str(candidate.get("activation_scheme", "")).lower()
        if activation_scheme != "dynamic":
            continue
        weight_per_tensor = bool(candidate.get("weight_per_tensor", False))
        activation_per_tensor = bool(candidate.get("act_per_tensor", False))
        if weight_per_tensor or activation_per_tensor:
            continue
        policies.append(
            {
                "weight": {
                    "format": "block_scaled_fp8_e4m3",
                    "block_shape": [int(value) for value in block_shape],
                    "per_tensor": False,
                },
                "activation": {
                    "format": "dynamic_block_fp8_e4m3",
                    "group_size": int(block_shape[1]),
                    "per_tensor": False,
                },
            }
        )

    if not policies:
        return None
    first = policies[0]
    if any(policy != first for policy in policies[1:]):
        raise ModelTranspileError(
            "source contains conflicting dynamic block-FP8 quantization contracts"
        )
    return first


def discover_sampling_policy(generation_config: Json) -> Json:
    repetition_penalty = float(generation_config.get("repetition_penalty", 1.0))
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0.0:
        raise ModelTranspileError(
            "sampling repetition_penalty must be finite and positive, "
            f"got {repetition_penalty}"
        )
    presence_penalty = float(generation_config.get("presence_penalty", 0.0))
    if not math.isfinite(presence_penalty):
        raise ModelTranspileError(
            f"sampling presence_penalty must be finite, got {presence_penalty}"
        )
    min_p = float(generation_config.get("min_p", 0.0))
    if not math.isfinite(min_p) or not 0.0 <= min_p <= 1.0:
        raise ModelTranspileError(f"sampling min_p must be in [0, 1], got {min_p}")
    if not bool(generation_config.get("do_sample", False)):
        return {
            "method": "greedy",
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        }

    temperature = float(generation_config.get("temperature", 1.0))
    top_k = int(generation_config.get("top_k", 0))
    top_p = float(generation_config.get("top_p", 1.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ModelTranspileError(
            f"sampling temperature must be finite and positive, got {temperature}"
        )
    if top_k <= 0:
        raise ModelTranspileError(
            "sampled generation requires a positive top_k for the resident Vulkan sampler"
        )
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ModelTranspileError(f"sampling top_p must be in (0, 1], got {top_p}")
    return {
        "method": "temperature_top_k_top_p",
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }


def discover_attention_window_size(config: Json) -> int | None:
    for key in ("sliding_window", "attention_window_size"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def discover_layer_intermediate_size(
    tensors: dict[str, Json], layer_tensors: dict[str, str]
) -> int:
    if "ffn_gate_up" in layer_tensors:
        return tensor_matrix_shape(tensors, layer_tensors["ffn_gate_up"])[0] // 2
    if "ffn_gate" in layer_tensors:
        return tensor_matrix_shape(tensors, layer_tensors["ffn_gate"])[0]
    info = tensors[layer_tensors["moe_input"]]
    shape = [int(value) for value in info.get("logical_shape", info["shape"])]
    if len(shape) != 3:
        raise ModelTranspileError(
            f"sparse expert input {layer_tensors['moe_input']!r} must be rank 3"
        )
    return shape[1] // 2


def discover_embedding_scale(
    config: Json, hidden_size: int, *, scaled_by_structure: bool = False
) -> float:
    if "embedding_multiplier" in config:
        return float(config["embedding_multiplier"])
    if bool(config.get("embeddings_scale_by_sqrt_dim", False)) or scaled_by_structure:
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
