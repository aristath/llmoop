from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from llmoop.circuit_ir import validate_circuit_against_pedal


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
    raise ValueError(f"unsupported pedal operator type {operator_type!r}")


def lower_pedalboard(pedalboard_dir: Path, out_dir: Path) -> Json:
    model = read_json(pedalboard_dir / "model.json")
    source_pedals = model["graph"]["pedalboard"]["pedals"]

    lowered: list[Json] = []
    operator_counts: Counter[str] = Counter()
    for source_pedal in source_pedals:
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
        nodes=_conv_nodes(hidden_size, pedal["numerics"]),
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
        nodes=_gated_delta_nodes(dimensions, pedal["numerics"]),
        behavioral_notes=(
            "This circuit preserves a recurrent gated-delta token mixer with fixed per-stream state.",
            "The recurrent matrix is transient pedal-owned DSP state, not a global cache.",
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
        "state_ports": [_state_port_for_circuit(port, operator_type) for port in pedal.get("state_ports", [])],
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


def _conv_nodes(hidden_size: int, numerics: Json) -> list[Json]:
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
            "params": ["conv_in_projection"],
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
            "params": ["conv_out_projection"],
        },
        *_ffn_tail(
            operator_output="operator_out",
            norm_eps=float(numerics["rms_norm_eps"]),
            norm_weight_offset=float(numerics["rms_norm_weight_offset"]),
        ),
    ]


def _attention_nodes(
    heads: Json,
    numerics: Json,
    *,
    has_q_norm: bool,
    has_k_norm: bool,
) -> list[Json]:
    nodes = [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": _norm_attrs(numerics),
        },
        {
            "id": "q_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["q_and_gate_projected" if numerics.get("attention_output_gate") else "q_projected"],
            "params": ["q_projection"],
        },
        {
            "id": "k_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["k_projected"],
            "params": ["k_projection"],
        },
        {
            "id": "v_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["v_projected"],
            "params": ["v_projection"],
        },
    ]
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
    rope_attrs = {
        "position_source": "stream_tick",
        "theta": float(numerics["rope_theta"]),
        "interleaved": bool(numerics["rope_interleaved"]),
        "rotary_width": int(numerics["rotary_width"]),
        **heads,
    }
    attention_tail: list[Json] = [
        {
                "id": "q_rope",
                "op": "rotary_position_embedding",
                "inputs": [q_rope_input],
                "outputs": ["q_positioned"],
                "attrs": rope_attrs,
        },
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
                "inputs": ["k_positioned", "v_projected", "kv_memory"],
                "outputs": ["k_memory", "v_memory"],
                "state_reads": ["kv_memory"],
                "state_writes": ["kv_memory"],
                "attrs": {"growth": "per_activation", **heads},
        },
        {
                "id": "attention_read",
                "op": "scaled_dot_product_attention",
                "inputs": ["q_positioned", "k_memory", "v_memory"],
                "outputs": ["attention_out"],
                "attrs": {"causal": True, **heads},
        },
        {
                "id": "attention_out_projection",
                "op": "linear",
                "inputs": ["attention_gated" if attention_gate else "attention_out"],
                "outputs": ["operator_out"],
                "params": ["attention_out_projection"],
        },
        *_ffn_tail(
            operator_output="operator_out",
            norm_eps=float(numerics["rms_norm_eps"]),
            norm_weight_offset=float(numerics["rms_norm_weight_offset"]),
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


def _gated_delta_nodes(dimensions: Json, numerics: Json) -> list[Json]:
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
            "params": ["delta_qkv_projection"],
        },
        {
            "id": "delta_z_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_z"],
            "params": ["delta_z_projection"],
        },
        {
            "id": "delta_b_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_b"],
            "params": ["delta_b_projection"],
        },
        {
            "id": "delta_a_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["delta_a"],
            "params": ["delta_a_projection"],
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
            "params": ["delta_out_projection"],
        },
        *_ffn_tail(
            operator_output="operator_out",
            norm_eps=float(numerics["rms_norm_eps"]),
            norm_weight_offset=float(numerics["rms_norm_weight_offset"]),
        ),
    ]


def _ffn_tail(
    operator_output: str, norm_eps: float, norm_weight_offset: float
) -> list[Json]:
    return [
        {
            "id": "operator_residual",
            "op": "residual_add",
            "inputs": ["input_frame", operator_output],
            "outputs": ["operator_residual_out"],
        },
        {
            "id": "ffn_norm",
            "op": "rms_norm",
            "inputs": ["operator_residual_out"],
            "outputs": ["ffn_norm_out"],
            "params": ["ffn_norm"],
            "attrs": {"eps": norm_eps, "weight_offset": norm_weight_offset},
        },
        {
            "id": "ffn_gate_projection",
            "op": "linear",
            "inputs": ["ffn_norm_out"],
            "outputs": ["ffn_gate"],
            "params": ["ffn_gate"],
        },
        {
            "id": "ffn_up_projection",
            "op": "linear",
            "inputs": ["ffn_norm_out"],
            "outputs": ["ffn_up"],
            "params": ["ffn_up"],
        },
        {
            "id": "ffn_gate_activation",
            "op": "silu",
            "inputs": ["ffn_gate"],
            "outputs": ["ffn_gate_activated"],
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
            "params": ["ffn_down"],
        },
        {
            "id": "ffn_residual",
            "op": "residual_add",
            "inputs": ["operator_residual_out", "ffn_out"],
            "outputs": ["output_frame"],
        },
    ]


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
        state.setdefault("layout", "channel_time" if state["id"] == "conv_state" else "head_key_value")
        state.setdefault("source_layout", state["layout"])
    return state


def _param_ref(name: str, ref: Json) -> Json:
    result = dict(ref)
    result["role"] = _param_role(name)
    return result


def _param_role(name: str) -> str:
    roles = {
        "operator_norm": "operator_normalization_weight",
        "ffn_norm": "feed_forward_normalization_weight",
        "ffn_gate": "feed_forward_swiglu_gate_projection",
        "ffn_down": "feed_forward_down_projection",
        "ffn_up": "feed_forward_up_projection",
        "conv_in_projection": "short_convolution_input_projection",
        "conv_depthwise_kernel": "short_convolution_depthwise_temporal_kernel",
        "conv_out_projection": "short_convolution_output_projection",
        "q_projection": "attention_query_projection",
        "k_projection": "attention_key_projection",
        "v_projection": "attention_value_projection",
        "attention_out_projection": "attention_output_projection",
        "q_norm": "attention_query_head_normalization",
        "k_norm": "attention_key_head_normalization",
        "delta_qkv_projection": "gated_delta_query_key_value_projection",
        "delta_z_projection": "gated_delta_output_gate_projection",
        "delta_b_projection": "gated_delta_beta_projection",
        "delta_a_projection": "gated_delta_decay_projection",
        "delta_conv_kernel": "gated_delta_depthwise_convolution_kernel",
        "delta_a_log": "gated_delta_decay_parameter",
        "delta_dt_bias": "gated_delta_time_bias",
        "delta_norm": "gated_delta_output_normalization_weight",
        "delta_out_projection": "gated_delta_output_projection",
    }
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
