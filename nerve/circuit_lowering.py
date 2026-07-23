from __future__ import annotations

from nerve.circuit_lowering_common import *
from nerve.circuit_lowering_helpers import *
from nerve.circuit_lowering_nodes import *
from nerve.circuit_lowering_operators import *
from nerve.circuit_lowering_system import *

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
    source_drafts = model["graph"].get("draft_pedalboards", [])
    draft_source_pedals = [
        pedal for draft in source_drafts for pedal in draft["pedalboard"]["pedals"]
    ]

    lowered: list[Json] = []
    operator_counts: Counter[str] = Counter()
    total = len(source_pedals) + len(draft_source_pedals)
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
                "runtime_role": result["circuit"]["runtime_role"],
                "circuit": str(circuit_rel),
                "params": str(params_rel),
                "state": str(state_rel),
                "implementation": result["circuit"]["implementation"],
                "behavioral_role": result["circuit"]["behavioral_role"],
            }
        )

    draft_pedalboards: list[Json] = []
    lowered_count = len(source_pedals)
    for draft in source_drafts:
        draft_refs: list[Json] = []
        for source_pedal in draft["pedalboard"]["pedals"]:
            check_compile_cancelled(cancel_requested)
            lowered_count += 1
            if progress is not None:
                progress(lowered_count, total, source_pedal["id"])
            pedal_path = pedalboard_dir / source_pedal["file"]
            pedal_out_dir = out_dir / "drafts" / draft["id"] / source_pedal["id"]
            result = lower_pedal(pedal_path, pedal_out_dir)
            operator_counts[source_pedal["operator_type"]] += 1
            draft_refs.append(
                {
                    "id": source_pedal["id"],
                    "operator_type": source_pedal["operator_type"],
                    "runtime_role": result["circuit"]["runtime_role"],
                    "circuit": str(result["circuit_path"].relative_to(out_dir)),
                    "params": str(result["params_path"].relative_to(out_dir)),
                    "state": str(result["state_path"].relative_to(out_dir)),
                    "implementation": result["circuit"]["implementation"],
                    "behavioral_role": result["circuit"]["behavioral_role"],
                }
            )
        lowered_draft = lower_draft_pedalboard(model, draft, draft_refs, out_dir)
        operator_counts["draft_input_adapter"] += 1
        operator_counts["draft_output_transducer"] += 1
        draft_pedalboards.append(lowered_draft)

    if not lowered:
        raise ValueError("cannot lower an empty pedalboard")

    system_circuits = build_system_circuits(model)
    system_refs: dict[str, Json] = {}
    for circuit in system_circuits:
        circuit_id = circuit["source"]["pedal_id"]
        circuit_out_dir = out_dir / circuit_id
        circuit_out_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_circuit(circuit)
        validation.raise_for_errors()
        circuit_path = circuit_out_dir / "circuit.json"
        params_path = circuit_out_dir / "params.json"
        state_path = circuit_out_dir / "state.json"
        write_json(circuit_path, circuit)
        write_json(params_path, build_params_artifact(circuit))
        write_json(state_path, build_state_artifact(circuit))
        operator_counts[circuit["source"]["source_operator_type"]] += 1
        system_refs[circuit["runtime_role"]] = {
            "id": circuit_id,
            "operator_type": circuit["source"]["source_operator_type"],
            "runtime_role": circuit["runtime_role"],
            "circuit": str(circuit_path.relative_to(out_dir)),
            "params": str(params_path.relative_to(out_dir)),
            "state": str(state_path.relative_to(out_dir)),
            "implementation": circuit["implementation"],
            "behavioral_role": circuit["behavioral_role"],
        }

    input_ref = system_refs["input_transducer"]
    output_ref = system_refs["output_transducer"]
    sampler_ref = system_refs["sampler"]
    all_circuits = [input_ref, *lowered, output_ref, sampler_ref]
    forward_chain = [input_ref, *lowered, output_ref, sampler_ref]

    index = {
        "schema": "nerve.lowered_pedalboard.v1",
        "source": {
            "format": "nerve.compiled_pedalboard_artifact.v1",
            "artifact_root": ".",
        },
        "architecture": model["architecture"],
        "dimensions": model["dimensions"],
        "numerics": model["numerics"],
        "token_ids": model["token_ids"],
        "graph": {
            "wiring": "explicit_graph",
            "circuits": all_circuits,
            "cables": [
                {
                    "id": f"cable_{index:04d}",
                    "connection": {"kind": "forward"},
                    "source": {
                        "pedal_id": source["id"],
                        "port_id": _canonical_output_port(source["runtime_role"]),
                    },
                    "destination": {
                        "pedal_id": destination["id"],
                        "port_id": _canonical_input_port(destination["runtime_role"]),
                    },
                }
                for index, (source, destination) in enumerate(
                    zip(forward_chain, forward_chain[1:])
                )
            ]
            + [
                {
                    "id": "generation_feedback",
                    "connection": {
                        "kind": "temporal_feedback",
                        "delay_activations": 1,
                    },
                    "source": {
                        "pedal_id": sampler_ref["id"],
                        "port_id": "sampled_token",
                    },
                    "destination": {
                        "pedal_id": input_ref["id"],
                        "port_id": "input_token",
                    },
                }
            ],
            "boundary": {
                "external_inputs": [
                    {
                        "id": "user_input",
                        "endpoint": {
                            "pedal_id": input_ref["id"],
                            "port_id": "input_token",
                        },
                    },
                    {
                        "id": "random_seed",
                        "endpoint": {
                            "pedal_id": sampler_ref["id"],
                            "port_id": "random_seed",
                        },
                    },
                ],
                "public_outputs": [
                    {
                        "id": "model_output",
                        "endpoint": {
                            "pedal_id": sampler_ref["id"],
                            "port_id": "sampled_token",
                        },
                    }
                ],
            },
            "input_transducer": model["graph"]["input_transducer"],
            "output_transducer": model["graph"]["output_transducer"],
        },
        "draft_pedalboards": draft_pedalboards,
        "summary": {
            "circuit_count": len(all_circuits)
            + sum(len(draft["circuits"]) for draft in draft_pedalboards),
            "generation_circuit_count": len(all_circuits),
            "draft_pedalboard_count": len(draft_pedalboards),
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
        "draft_pedalboards": draft_pedalboards,
    }


def _canonical_input_port(runtime_role: str) -> str:
    return {
        "input_transducer": "input_token",
        "signal_processor": "input_frame",
        "output_transducer": "input_frame",
        "sampler": "input_logits",
    }[runtime_role]


def _canonical_output_port(runtime_role: str) -> str:
    return {
        "input_transducer": "output_frame",
        "signal_processor": "output_frame",
        "output_transducer": "output_logits",
        "sampler": "sampled_token",
    }[runtime_role]


def lower_draft_pedalboard(
    model: Json,
    draft: Json,
    layer_refs: list[Json],
    out_dir: Path,
) -> Json:
    if not layer_refs:
        raise ValueError(f"draft pedalboard {draft['id']!r} contains no layer pedals")
    system_circuits = build_draft_system_circuits(model, draft)
    system_refs = []
    for circuit in system_circuits:
        circuit_id = circuit["source"]["pedal_id"]
        circuit_out_dir = out_dir / "drafts" / draft["id"] / circuit_id
        circuit_out_dir.mkdir(parents=True, exist_ok=True)
        validate_circuit(circuit).raise_for_errors()
        circuit_path = circuit_out_dir / "circuit.json"
        params_path = circuit_out_dir / "params.json"
        state_path = circuit_out_dir / "state.json"
        write_json(circuit_path, circuit)
        write_json(params_path, build_params_artifact(circuit))
        write_json(state_path, build_state_artifact(circuit))
        system_refs.append(
            {
                "id": circuit_id,
                "operator_type": circuit["source"]["source_operator_type"],
                "runtime_role": circuit["runtime_role"],
                "circuit": str(circuit_path.relative_to(out_dir)),
                "params": str(params_path.relative_to(out_dir)),
                "state": str(state_path.relative_to(out_dir)),
                "implementation": circuit["implementation"],
                "behavioral_role": circuit["behavioral_role"],
            }
        )

    input_ref, output_ref = system_refs
    forward_chain = [input_ref, *layer_refs, output_ref]
    return {
        "id": draft["id"],
        "type": draft["type"],
        "source_prefix": draft["source_prefix"],
        "wiring": "explicit_graph",
        "circuits": forward_chain,
        "cables": [
            {
                "id": f"{draft['id']}_cable_{index:04d}",
                "connection": {"kind": "forward"},
                "source": {
                    "pedal_id": source["id"],
                    "port_id": (
                        "output_frame"
                        if source["runtime_role"] != "draft_output_transducer"
                        else "output_hidden"
                    ),
                },
                "destination": {
                    "pedal_id": destination["id"],
                    "port_id": "input_frame",
                },
            }
            for index, (source, destination) in enumerate(
                zip(forward_chain, forward_chain[1:])
            )
        ],
        "boundary": {
            "external_inputs": [
                {
                    "id": "token_embedding",
                    "endpoint": {
                        "pedal_id": input_ref["id"],
                        "port_id": "token_embedding",
                    },
                },
                {
                    "id": "target_hidden",
                    "endpoint": {
                        "pedal_id": input_ref["id"],
                        "port_id": "target_hidden",
                    },
                },
            ],
            "public_outputs": [
                {
                    "id": "draft_hidden",
                    "endpoint": {
                        "pedal_id": output_ref["id"],
                        "port_id": "output_hidden",
                    },
                },
                {
                    "id": "draft_logits",
                    "endpoint": {
                        "pedal_id": output_ref["id"],
                        "port_id": "output_logits",
                    },
                },
            ],
        },
        "state_contract": dict(draft["state_contract"]),
    }


