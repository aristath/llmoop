from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from llmoop.circuit_ir import validate_circuit_against_pedal
from llmoop.model_compiler import check_compile_cancelled


Json = dict[str, Any]


def lower_pedal(pedal_path: Path, out_dir: Path) -> Json:
    pedal = read_json(pedal_path)
    circuit = build_pedal_circuit(pedal, pedal_path)
    validation = validate_circuit_against_pedal(circuit, pedal)
    validation.raise_for_errors()

    out_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = out_dir / "circuit.json"
    params_path = out_dir / "params.json"
    state_path = out_dir / "state.json"
    write_json(circuit_path, circuit)
    write_json(params_path, build_params_artifact(circuit))
    write_json(state_path, build_state_artifact(circuit))

    return {
        "pedal": pedal,
        "circuit": circuit,
        "validation": validation.to_json(),
        "circuit_path": circuit_path,
        "params_path": params_path,
        "state_path": state_path,
    }


def build_pedal_circuit(pedal: Json, pedal_path: Path) -> Json:
    operator_type = pedal.get("operator_type")
    if operator_type == "conv":
        return build_conv_circuit(pedal, pedal_path)
    if operator_type == "full_attention":
        return build_attention_circuit(pedal, pedal_path)
    if operator_type == "gated_delta":
        return build_gated_delta_circuit(pedal, pedal_path)
    if operator_type == "rg_lru":
        return build_rg_lru_circuit(pedal, pedal_path)
    raise ValueError(f"unsupported pedal operator type {operator_type!r}")


def lower_pedalboard(
    pedalboard_dir: Path,
    out_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    model = read_json(pedalboard_dir / "model.json")
    source_pedals = model["graph"]["pedalboard"]["pedals"]

    lowered: list[Json] = []
    operator_counts: Counter[str] = Counter()
    total = len(source_pedals)
    for current, source_pedal in enumerate(source_pedals, start=1):
        check_compile_cancelled(cancel_requested)
        if progress is not None:
            progress(current, total, source_pedal["id"])
        pedal_path = pedalboard_dir / source_pedal["file"]
        pedal_out_dir = out_dir / source_pedal["id"]
        result = lower_pedal(pedal_path, pedal_out_dir)
        circuit_rel = result["circuit_path"].relative_to(out_dir)
        params_rel = result["params_path"].relative_to(out_dir)
        state_rel = result["state_path"].relative_to(out_dir)
        operator_counts[source_pedal["operator_type"]] += 1
        lowered.append(
            {
                "id": source_pedal["id"],
                "operator_type": source_pedal["operator_type"],
                "circuit": str(circuit_rel),
                "params": str(params_rel),
                "state": str(state_rel),
                "implementation": result["circuit"]["implementation"],
                "behavioral_role": result["circuit"]["behavioral_role"],
            }
        )

    index = {
        "schema": "llmoop.lowered_pedalboard.v1",
        "source": {
            "format": "llmoop.compiled_pedalboard_artifact.v1",
            "artifact_root": ".",
        },
        "architecture": model["architecture"],
        "dimensions": model["dimensions"],
        "numerics": model["numerics"],
        "token_ids": model["token_ids"],
        "graph": {
            "wiring": model["graph"]["pedalboard"]["wiring"],
            "circuits": lowered,
            "input_transducer": model["graph"]["input_transducer"],
            "output_transducer": model["graph"]["output_transducer"],
        },
        "summary": {
            "circuit_count": len(lowered),
            "operator_counts": dict(sorted(operator_counts.items())),
        },
        "notes": [
            "This index maps the source pedalboard to stream-circuit artifacts.",
            "The artifacts preserve pedal boundaries for now; a backend may later fuse or replace connected regions.",
            "No layer receives privileged treatment; every pedal is addressed through the same boundary contract.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "pedalboard.circuits.json"
    write_json(index_path, index)
    return {
        "index": index,
        "index_path": index_path,
        "circuits": lowered,
    }


def build_conv_circuit(pedal: Json, pedal_path: Path) -> Json:
    hidden_size = pedal["ports"]["inputs"][0]["shape"][0]
    return _base_circuit(
        pedal=pedal,
        pedal_path=pedal_path,
        behavioral_role="source_reference_circuit",
        implementation="reference_shortconv_layer_circuit_v1",
        circuit_id=f"{pedal['id']}_shortconv_circuit_v1",
        nodes=_conv_nodes(
            hidden_size,
            pedal["numerics"],
            pedal["feed_forward"],
            pedal["parameter_block"]["params"],
        ),
        behavioral_notes=(
            "This circuit preserves the source short-convolution layer decomposition.",
            "It is a structural lowering artifact; a backend may replace this source pedal with an executable implementation.",
        ),
    )


def build_attention_circuit(pedal: Json, pedal_path: Path) -> Json:
    heads = _attention_heads_from_state(pedal)
    parameters = pedal["parameter_block"]["params"]
    return _base_circuit(
        pedal=pedal,
        pedal_path=pedal_path,
        behavioral_role="source_reference_circuit",
        implementation="reference_gqa_attention_layer_circuit_v1",
        circuit_id=f"{pedal['id']}_gqa_attention_circuit_v1",
        nodes=_attention_nodes(
            heads,
            pedal["numerics"],
            has_q_norm="q_norm" in parameters,
            has_k_norm="k_norm" in parameters,
            has_value_norm=bool(pedal["numerics"].get("value_head_norm")),
            feed_forward=pedal["feed_forward"],
            parameters=parameters,
        ),
        behavioral_notes=(
            "This circuit preserves the source grouped-query attention layer decomposition.",
            "KV is represented as stream-owned append-only transient state, not as a disposable host cache.",
        ),
    )


def build_gated_delta_circuit(pedal: Json, pedal_path: Path) -> Json:
    dimensions = pedal["reference_decomposition"]["wiring"][1]["dimensions"]
    return _base_circuit(
        pedal=pedal,
        pedal_path=pedal_path,
        behavioral_role="source_reference_circuit",
        implementation="reference_gated_delta_layer_circuit_v1",
        circuit_id=f"{pedal['id']}_gated_delta_circuit_v1",
        nodes=_gated_delta_nodes(
            dimensions,
            pedal["numerics"],
            pedal["feed_forward"],
            pedal["parameter_block"]["params"],
        ),
        behavioral_notes=(
            "This circuit preserves a recurrent gated-delta token mixer with fixed per-stream state.",
            "The recurrent matrix is transient pedal-owned DSP state, not a global cache.",
        ),
    )


def build_rg_lru_circuit(pedal: Json, pedal_path: Path) -> Json:
    dimensions = pedal["reference_decomposition"]["wiring"][1]["dimensions"]
    return _base_circuit(
        pedal=pedal,
        pedal_path=pedal_path,
        behavioral_role="source_reference_circuit",
        implementation="reference_rg_lru_layer_circuit_v1",
        circuit_id=f"{pedal['id']}_rg_lru_circuit_v1",
        nodes=_rg_lru_nodes(
            dimensions,
            pedal["numerics"],
            pedal["feed_forward"],
            pedal["parameter_block"]["params"],
        ),
        behavioral_notes=(
            "This circuit preserves a real-gated linear recurrent token mixer with fixed per-stream state.",
            "The convolution and diagonal recurrence are transient pedal-owned DSP state.",
        ),
    )


def build_params_artifact(circuit: Json) -> Json:
    return {
        "schema": "llmoop.circuit_params.v1",
        "circuit": circuit["id"],
        "layout": circuit["parameters"]["layout"],
        "storage": circuit["parameters"]["storage"],
        "refs": circuit["parameters"]["refs"],
    }


def build_state_artifact(circuit: Json) -> Json:
    return {
        "schema": "llmoop.circuit_state.v1",
        "circuit": circuit["id"],
        "state_ports": circuit["state_ports"],
    }


def _base_circuit(
    *,
    pedal: Json,
    pedal_path: Path,
    behavioral_role: str,
    implementation: str,
    circuit_id: str,
    nodes: list[Json],
    behavioral_notes: tuple[str, ...],
) -> Json:
    input_port = pedal["ports"]["inputs"][0]
    output_port = pedal["ports"]["outputs"][0]
    params = pedal["parameter_block"]["params"]
    operator_type = pedal["operator_type"]
    return {
        "schema": "llmoop.stream_circuit.v1",
        "id": circuit_id,
        "source": {
            "pedal_id": pedal["id"],
            "source_layer_index": pedal["source_layer_index"],
            "source_operator_type": operator_type,
        },
        "behavioral_role": behavioral_role,
        "implementation": implementation,
        "boundary": {
            "inputs": [
                {
                    "id": "input_frame",
                    "signal": input_port["signal"],
                    "shape": input_port["shape"],
                    "pedal_port": input_port["id"],
                }
            ],
            "outputs": [
                {
                    "id": "output_frame",
                    "signal": output_port["signal"],
                    "shape": output_port["shape"],
                    "source": "output_frame",
                    "pedal_port": output_port["id"],
                }
            ],
            "controls": pedal["ports"].get("controls", []),
        },
        "state_ports": [
            _state_port_for_circuit(port, operator_type)
            for port in pedal.get("state_ports", [])
        ],
        "parameters": {
            "layout": pedal["parameter_block"]["layout"],
            "storage": pedal["parameter_block"]["storage"],
            "refs": {name: _param_ref(name, ref) for name, ref in params.items()},
        },
        "nodes": nodes,
        "behavioral_error_contract": {
            "mode": behavioral_role,
            "reference": pedal["transition_contract"]["reference_behavior"],
            "current_tolerance": {
                "atol": 1e-6,
                "rtol": 1e-5,
            },
            "notes": list(behavioral_notes),
        },
        "lowering_notes": [
            "Layer is represented as one pedal-level circuit with explicit internal nodes.",
            "Transient memory belongs to the stream instance.",
            "The graph is ordered topologically so a backend can fuse or replace regions without changing the boundary contract.",
        ],
    }


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
    q_rope_input = "q_projected"
    if has_q_norm:
        nodes.append(
            {
                "id": "q_head_norm",
                "op": "rms_norm_per_head",
                "inputs": ["q_projected"],
                "outputs": ["q_normed"],
                "params": ["q_norm"],
                "attrs": {**_norm_attrs(numerics), **heads},
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
                "attrs": {**_norm_attrs(numerics), **heads},
            }
        )
        k_rope_input = "k_normed"
    value_input = (
        "k_projected"
        if numerics.get("attention_key_equals_value")
        else "v_projected"
    )
    if has_value_norm:
        nodes.append(
            {
                "id": "v_head_norm",
                "op": "rms_norm_per_head_unscaled",
                "inputs": [value_input],
                "outputs": ["v_normed"],
                "attrs": {**_norm_attrs(numerics), **heads},
            }
        )
        value_input = "v_normed"
    rope_attrs = {
        "position_source": "stream_tick",
        "theta": float(numerics["rope_theta"]),
        "rope_type": str(numerics.get("rope_type", "default")),
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
            "attrs": rope_attrs,
        },
        *(
            [
                {
                    "id": "k_rope",
                    "op": "rotary_position_embedding",
                    "inputs": [k_rope_input],
                    "outputs": ["k_positioned"],
                    "attrs": rope_attrs,
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
    if attention_gate:
        attention_tail.insert(
            4,
            {
                "id": "attention_output_gate",
                "op": "sigmoid_multiply",
                "inputs": ["attention_out", attention_gate],
                "outputs": ["attention_gated"],
            },
        )
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
                "outputs": ["moe_routing_weights"],
                "attrs": {
                    "num_experts": int(feed_forward["num_experts"]),
                    "experts_per_token": int(feed_forward["experts_per_token"]),
                },
            },
            {
                "id": "sparse_moe_experts",
                "op": "sparse_moe_experts",
                "inputs": ["ffn_norm_out", "moe_routing_weights"],
                "outputs": ["moe_expert_outputs"],
                "params": [
                    "moe_input",
                    *(
                        ["moe_input_scale_inv"]
                        if "moe_input_scale_inv" in parameters
                        else []
                    ),
                    "moe_output",
                    *(
                        ["moe_output_scale_inv"]
                        if "moe_output_scale_inv" in parameters
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
                    "num_experts": int(feed_forward["num_experts"]),
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
                                "params": _linear_params(
                                    "shared_mlp_gate", parameters
                                ),
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
        tail.extend(
            [
                {
                    "id": "per_layer_embedding",
                    "op": "per_layer_embedding",
                    "inputs": [],
                    "outputs": ["per_layer_input"],
                    "params": [
                        "token_embedding",
                        "per_layer_embedding",
                        "per_layer_model_projection",
                        "per_layer_projection_norm",
                    ],
                    "attrs": {
                        "hidden_size": hidden_size,
                        "per_layer_width": width,
                        "layer_index": int(numerics["per_layer_input_layer_index"]),
                        "layer_count": int(numerics["per_layer_input_layer_count"]),
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
            raise CircuitLoweringError(
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
        "attention_sinks": "attention_sink_logits",
        "q_norm": "attention_query_head_normalization",
        "k_norm": "attention_key_head_normalization",
        "token_embedding": "token_embedding_for_per_layer_input",
        "per_layer_embedding": "packed_per_layer_token_embedding",
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
    return roles[name]


def _norm_attrs(numerics: Json) -> Json:
    return {
        "eps": float(numerics["rms_norm_eps"]),
        "weight_offset": float(numerics["rms_norm_weight_offset"]),
    }


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
