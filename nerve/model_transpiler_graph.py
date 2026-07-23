from nerve.model_transpiler_types import *
from nerve.model_transpiler_tensor_index import *
from nerve.model_transpiler_quantization import *

def make_layer(
    structure: ModelStructure,
    layer: LayerStructure,
    *,
    pedal_id: str | None = None,
    runtime_role: str = "signal_processor",
) -> Json:
    hidden_size = structure.hidden_size
    pedal_id = pedal_id or f"layer_{layer.index:02d}"
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
        "schema": "nerve.pedal_instance.v1",
        "id": pedal_id,
        "source_layer_index": layer.index,
        "type": "pedal_instance",
        "runtime_role": runtime_role,
        "pedal_class": make_pedal_class(structure, layer),
        "operator_type": layer.operator_type,
        "feed_forward": make_feed_forward_descriptor(structure, layer),
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": layer.rope_theta,
            "rope_type": layer.rope_type,
            "rope_scaling": deepcopy(layer.rope_scaling),
            "rope_interleaved": structure.rope_interleaved,
            "rotary_width": layer.rotary_width,
            "rms_norm_weight_offset": structure.rms_norm_weight_offset,
            "attention_output_gate": structure.attention_output_gate,
            "attention_gate_activation": layer.attention_gate_activation,
            "attention_gate_per_head": layer.attention_gate_per_head,
            "attention_key_equals_value": layer.attention_key_equals_value,
            "residual_scale": structure.residual_scale,
            "attention_scale": layer.attention_scale,
            "attention_window_size": layer.attention_window_size,
            "value_head_norm": layer.value_head_norm,
            "per_layer_input_width": layer.per_layer_input_width,
            "per_layer_input_layer_index": layer.index,
            "per_layer_input_layer_count": structure.num_hidden_layers,
            "per_layer_embedding_chunk_count": len(
                [
                    name
                    for name in layer.tensors
                    if name.startswith("per_layer_embedding_chunk_")
                ]
            )
            or None,
            "per_layer_embedding_chunk_rows": (
                MAX_SHADER_PARAMETER_CHUNK_BYTES
                // (structure.num_hidden_layers * layer.per_layer_input_width * 2)
                if layer.per_layer_input_width is not None
                else None
            ),
            "token_embedding_scale": structure.embedding_scale,
            "per_layer_embedding_scale": (
                round_float_to_bf16(math.sqrt(layer.per_layer_input_width))
                if layer.per_layer_input_width is not None
                else None
            ),
            "per_layer_model_projection_scale": hidden_size**-0.5,
            "per_layer_input_scale": 2.0**-0.5,
        },
        "ports": {
            "inputs": [{"id": "input", "signal": "frame", "shape": [hidden_size]}],
            "outputs": [{"id": "output", "signal": "frame", "shape": [hidden_size]}],
            "controls": [],
        },
        "state_ports": make_state_ports(structure, layer),
        "parameter_block": make_parameter_block(
            layer.operator_type, layer.feed_forward_type, layer.tensors
        ),
        "transition_contract": {
            "type": "stateful_frame_transform",
            "equation": "(output_frame, next_state, events) = pedal(input_frame, state, params, control)",
            "reference_behavior": f"source_checkpoint_entity:{layer.prefix}",
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
        "schema": "nerve.model_graph.v1",
        "source": tensor_index["source"],
        "architecture": {
            "family": "decoder_only_transformer",
            "model_type": structure.model_type,
            "architectures": list(structure.architectures),
            "dtype": structure.dtype,
        },
        "dimensions": {
            "hidden_size": structure.hidden_size,
            "intermediate_sizes": [
                layer.intermediate_size for layer in structure.layers
            ],
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
            "attention_layer_shapes": [
                {
                    "layer": layer.index,
                    "query_heads": layer.num_attention_heads,
                    "key_value_heads": layer.num_key_value_heads,
                    "head_width": layer.head_width,
                    "rotary_width": layer.rotary_width,
                    "rope_theta": layer.rope_theta,
                    "rope_type": layer.rope_type,
                    "shared_kv_source_layer": layer.shared_kv_source_layer,
                }
                for layer in structure.layers
                if layer.operator_type == "full_attention"
            ],
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
        "quantization": structure.quantization,
        "sampling": structure.sampling,
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
            "draft_pedalboards": [
                make_draft_pedalboard_descriptor(structure, draft)
                for draft in structure.draft_pedalboards
            ],
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


def make_draft_pedalboard_descriptor(
    structure: ModelStructure,
    draft: DraftPedalboardStructure,
) -> Json:
    hidden_size = structure.hidden_size
    adapter_params = {
        name: tensor_ref(tensor)
        for name, tensor in draft.tensors.items()
        if name not in {"output_norm", "output_projection"}
    }
    return {
        "id": draft.id,
        "type": "multi_token_prediction",
        "source_prefix": draft.prefix,
        "input_adapter": {
            "type": "normalized_embedding_hidden_projection",
            "inputs": [
                {"id": "token_embedding", "signal": "frame", "shape": [hidden_size]},
                {"id": "target_hidden", "signal": "frame", "shape": [hidden_size]},
            ],
            "output": {"id": "output_frame", "signal": "frame", "shape": [hidden_size]},
            "attrs": {
                "eps": structure.norm_eps,
                "weight_offset": structure.rms_norm_weight_offset,
                "concatenation_order": ["token_embedding", "target_hidden"],
            },
            "params": adapter_params,
        },
        "pedalboard": {
            "wiring": "series",
            "pedals": [
                {
                    "id": f"{draft.id}_layer_{layer.index:02d}",
                    "type": "pedal_instance",
                    "pedal_class": make_pedal_class(structure, layer),
                    "operator_type": layer.operator_type,
                    "file": (
                        f"drafts/{draft.id}/layers/"
                        f"{draft.id}_layer_{layer.index:02d}.json"
                    ),
                }
                for layer in draft.layers
            ],
        },
        "output_transducer": {
            "type": "normalized_hidden_projection",
            "inputs": [
                {"id": "input_frame", "signal": "frame", "shape": [hidden_size]}
            ],
            "outputs": [
                {"id": "output_hidden", "signal": "frame", "shape": [hidden_size]},
                {
                    "id": "output_logits",
                    "signal": "logits",
                    "shape": [structure.vocab_size],
                },
            ],
            "attrs": {
                "eps": structure.norm_eps,
                "weight_offset": structure.rms_norm_weight_offset,
                "scale": 1.0 / structure.logits_scale,
                "soft_cap": structure.logits_soft_cap,
            },
            "params": {
                "norm": tensor_ref(draft.tensors["output_norm"]),
                "projection": tensor_ref(draft.tensors["output_projection"]),
            },
        },
        "state_contract": {
            "ownership": "per_stream_per_pedal_instance",
            "draft_updates": "tentative",
            "acceptance": "commit_accepted_prefix",
            "rejection": "restore_last_committed_state",
        },
    }


def make_feed_forward_descriptor(
    structure: ModelStructure, layer: LayerStructure
) -> Json:
    descriptor: Json = {
        "type": layer.feed_forward_type,
        "hidden_size": structure.hidden_size,
        "intermediate_size": layer.intermediate_size,
        "activation": structure.activation,
    }
    if layer.feed_forward_type == "sparse_moe":
        descriptor.update(
            {
                "num_experts": structure.num_experts,
                "experts_per_token": structure.experts_per_token,
                "routing": structure.moe_routing,
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
        if "moe_router_correction_bias" in layer.tensors:
            params["router_correction_bias"] = tensor_ref(
                layer.tensors["moe_router_correction_bias"]
            )
        if layer.shared_intermediate_size is not None:
            params.update(
                {
                    "shared_input": tensor_ref(layer.tensors["shared_mlp_input"]),
                    "shared_output": tensor_ref(layer.tensors["shared_mlp_output"]),
                }
            )
            if "shared_mlp_gate" in layer.tensors:
                params["shared_gate"] = tensor_ref(layer.tensors["shared_mlp_gate"])
        return {
            "id": "feed_forward",
            "type": "sparse_moe_feed_forward",
            "input": "ffn_norm.output",
            "output": "ffn.output",
            "dimensions": make_feed_forward_descriptor(structure, layer),
            "params": params,
        }
    if "ffn_gate_up" in layer.tensors:
        params = {
            "gate_up": tensor_ref(layer.tensors["ffn_gate_up"]),
            "down": tensor_ref(layer.tensors["ffn_down"]),
        }
        if "ffn_gate_up_bias" in layer.tensors:
            params["gate_up_bias"] = tensor_ref(layer.tensors["ffn_gate_up_bias"])
    else:
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
        "circuit_template": (
            f"swiglu_ffn_{structure.hidden_size}_{layer.intermediate_size}_v1"
        ),
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
        "state_ports": make_state_ports(structure, layer),
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
    head_width = layer.head_width
    heads = {
        "query_heads": layer.num_attention_heads,
        "key_value_heads": layer.num_key_value_heads,
        "head_width": head_width,
        "query_groups_per_kv_head": layer.num_attention_heads
        // layer.num_key_value_heads,
    }
    if "qkv_projection" in layer.tensors:
        params = {
            "qkv_projection": tensor_ref(layer.tensors["qkv_projection"]),
            "out_projection": tensor_ref(layer.tensors["attention_out_projection"]),
        }
        if "qkv_projection_bias" in layer.tensors:
            params["qkv_projection_bias"] = tensor_ref(
                layer.tensors["qkv_projection_bias"]
            )
    else:
        params = {
            "q_projection": tensor_ref(layer.tensors["q_projection"]),
            "out_projection": tensor_ref(layer.tensors["attention_out_projection"]),
        }
        if layer.shared_kv_source_layer is None:
            params["k_projection"] = tensor_ref(layer.tensors["k_projection"])
            if "v_projection" in layer.tensors:
                params["v_projection"] = tensor_ref(layer.tensors["v_projection"])
    for source_id, target_id in (
        ("q_projection_bias", "q_projection_bias"),
        ("k_projection_bias", "k_projection_bias"),
        ("v_projection_bias", "v_projection_bias"),
        ("attention_out_projection_bias", "out_projection_bias"),
        ("attention_gate_projection_bias", "attention_gate_projection_bias"),
    ):
        if source_id in layer.tensors:
            params[target_id] = tensor_ref(layer.tensors[source_id])
    internal_pedals = (
        [
            {"id": "qkv_projection", "type": "linear"},
            {"id": "qkv_split", "type": "split"},
        ]
        if "qkv_projection" in layer.tensors
        else [
            {"id": "q_projection", "type": "linear"},
            *(
                [
                    {"id": "k_projection", "type": "linear"},
                    *(
                        [{"id": "v_projection", "type": "linear"}]
                        if "v_projection" in layer.tensors
                        else []
                    ),
                ]
                if layer.shared_kv_source_layer is None
                else []
            ),
        ]
    )
    if structure.attention_output_gate:
        internal_pedals.append({"id": "q_gate_split", "type": "split"})
    if "attention_gate_projection" in layer.tensors:
        params["attention_gate_projection"] = tensor_ref(
            layer.tensors["attention_gate_projection"]
        )
        internal_pedals.append({"id": "attention_gate_projection", "type": "linear"})
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
            {
                "id": "kv_memory",
                "type": (
                    "shared_state_read"
                    if layer.shared_kv_source_layer is not None
                    else "stateful_append_memory"
                ),
            },
            {"id": "attention_read", "type": "scaled_dot_product_attention"},
            *(
                [
                    {
                        "id": "attention_output_gate",
                        "type": (
                            "sigmoid_multiply"
                            if structure.attention_output_gate
                            else f"{layer.attention_gate_activation}_multiply"
                        ),
                    }
                ]
                if structure.attention_output_gate
                or layer.attention_gate_activation is not None
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
            f"h{structure.hidden_size}_q{layer.num_attention_heads}_"
            f"kv{layer.num_key_value_heads}_d{head_width}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "heads": heads,
        "rotary_width": layer.rotary_width,
        "rope_type": layer.rope_type,
        "output_gate": structure.attention_output_gate,
        "attention_gate": (
            {
                "activation": layer.attention_gate_activation,
                "per_head": layer.attention_gate_per_head,
            }
            if layer.attention_gate_activation is not None
            else None
        ),
        "window_size": layer.attention_window_size,
        "shared_kv_source_layer": layer.shared_kv_source_layer,
        "state_ports": make_state_ports(structure, layer),
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
        "state_ports": make_state_ports(structure, layer),
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
        "state_ports": make_state_ports(structure, layer),
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
    layer: LayerStructure,
) -> list[Json]:
    operator_type = layer.operator_type
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
        head_width = layer.head_width
        sharing = (
            f"shared_from:layer_{layer.shared_kv_source_layer:02d}.kv_memory"
            if layer.shared_kv_source_layer is not None
            else "per_stream_per_pedal_instance"
        )
        return [
            {
                "id": "kv_memory",
                "type": "append_only_attention_memory",
                "query_heads": layer.num_attention_heads,
                "key_shape_per_token": [layer.num_key_value_heads, head_width],
                "value_shape_per_token": [layer.num_key_value_heads, head_width],
                "dtype": "BF16",
                "growth": "per_activation",
                "max_dynamic_activations": layer.attention_window_size,
                "sharing": sharing,
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
        f"moe{structure.num_experts}x{structure.experts_per_token}i{layer.intermediate_size}"
        if layer.feed_forward_type == "sparse_moe"
        else f"ffn{layer.intermediate_size}"
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
        head_width = layer.head_width
        return (
            "gqa_attention_layer_"
            f"h{structure.hidden_size}_q{layer.num_attention_heads}_"
            f"kv{layer.num_key_value_heads}_d{head_width}_"
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


