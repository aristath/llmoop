from nerve.circuit_lowering_common import *
from nerve.circuit_lowering_helpers import *
from nerve.circuit_lowering_nodes import *

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
        "schema": "nerve.circuit_params.v1",
        "circuit": circuit["id"],
        "layout": circuit["parameters"]["layout"],
        "storage": circuit["parameters"]["storage"],
        "refs": circuit["parameters"]["refs"],
    }


def build_state_artifact(circuit: Json) -> Json:
    return {
        "schema": "nerve.circuit_state.v1",
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
    input_ports = pedal["ports"]["inputs"]
    output_ports = pedal["ports"]["outputs"]
    if len(input_ports) != 1 or len(output_ports) != 1:
        raise ValueError(
            f"layer pedal {pedal.get('id')!r} must expose exactly one frame input and "
            f"one frame output; found {len(input_ports)} inputs and {len(output_ports)} outputs"
        )
    input_port = input_ports[0]
    output_port = output_ports[0]
    params = pedal["parameter_block"]["params"]
    operator_type = pedal["operator_type"]
    return {
        "schema": "nerve.stream_circuit.v1",
        "id": circuit_id,
        "source": {
            "pedal_id": pedal["id"],
            "source_layer_index": pedal["source_layer_index"],
            "source_operator_type": operator_type,
        },
        "runtime_role": pedal.get("runtime_role", "signal_processor"),
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

__all__ = [name for name in globals() if not name.startswith("__")]
