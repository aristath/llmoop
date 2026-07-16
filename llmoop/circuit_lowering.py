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
        nodes=_conv_nodes(hidden_size),
        behavioral_notes=(
            "This circuit preserves the source short-convolution layer decomposition.",
            "It is a structural lowering artifact; a backend may replace this source pedal with an executable implementation.",
        ),
    )


def build_attention_circuit(pedal: Json, pedal_path: Path) -> Json:
    heads = _attention_heads_from_state(pedal)
    return _base_circuit(
        pedal=pedal,
        pedal_path=pedal_path,
        behavioral_role="source_reference_circuit",
        implementation="reference_gqa_attention_layer_circuit_v1",
        circuit_id=f"{pedal['id']}_gqa_attention_circuit_v1",
        nodes=_attention_nodes(heads),
        behavioral_notes=(
            "This circuit preserves the source grouped-query attention layer decomposition.",
            "KV is represented as stream-owned append-only transient state, not as a disposable host cache.",
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


def _conv_nodes(hidden_size: int) -> list[Json]:
    return [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": {"eps_source": "model.config.norm_eps"},
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
            "attrs": {"axis": "channel", "parts": ["b", "c", "x"]},
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
        *_ffn_tail(operator_output="operator_out"),
    ]


def _attention_nodes(heads: Json) -> list[Json]:
    return [
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm"],
            "attrs": {"eps_source": "model.config.norm_eps"},
        },
        {
            "id": "q_projection",
            "op": "linear",
            "inputs": ["operator_norm_out"],
            "outputs": ["q_projected"],
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
        {
            "id": "q_head_norm",
            "op": "rms_norm_per_head",
            "inputs": ["q_projected"],
            "outputs": ["q_normed"],
            "params": ["q_norm"],
            "attrs": heads,
        },
        {
            "id": "k_head_norm",
            "op": "rms_norm_per_head",
            "inputs": ["k_projected"],
            "outputs": ["k_normed"],
            "params": ["k_norm"],
            "attrs": heads,
        },
        {
            "id": "q_rope",
            "op": "rotary_position_embedding",
            "inputs": ["q_normed"],
            "outputs": ["q_positioned"],
            "attrs": {"position_source": "stream_tick", **heads},
        },
        {
            "id": "k_rope",
            "op": "rotary_position_embedding",
            "inputs": ["k_normed"],
            "outputs": ["k_positioned"],
            "attrs": {"position_source": "stream_tick", **heads},
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
            "inputs": ["attention_out"],
            "outputs": ["operator_out"],
            "params": ["attention_out_projection"],
        },
        *_ffn_tail(operator_output="operator_out"),
    ]


def _ffn_tail(operator_output: str) -> list[Json]:
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
            "attrs": {"eps_source": "model.config.norm_eps"},
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
    query_heads = hidden_size // head_width
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
    }
    return roles[name]


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
