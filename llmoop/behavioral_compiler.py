from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from llmoop.compilation import Json, ModelCompileError


BEHAVIORAL_VALIDATION_SCHEMA = "llmoop.behavioral_validation.v1"
EXACT_REWRITE_CONTRACTS = {
    "append_scaled_dot_product_attention": "append_attention_exact_bf16.v1",
    "dual_linear_silu_multiply": "dual_linear_silu_multiply_exact_bf16.v1",
    "linear_residual": "linear_residual_exact_bf16.v1",
    "linear_split_3way": "linear_split_exact_bf16.v1",
    "linear_split_recurrent_depthwise_gate": "linear_recurrent_exact_bf16.v1",
    "multiply_rolling_depthwise": "rolling_depthwise_exact_bf16.v1",
    "multiply_rolling_depthwise_gate": "rolling_depthwise_gate_exact_bf16.v1",
    "parallel_head_norm_rope_2way": "parallel_head_norm_rope_exact_bf16.v1",
    "parallel_linear_2way": "parallel_linear_exact_bf16.v1",
    "parallel_linear_3way": "parallel_linear_exact_bf16.v1",
    "silu_multiply": "silu_multiply_exact_bf16.v1",
}
EXACT_REWRITE_SOURCE_OPS = {
    "append_scaled_dot_product_attention": {
        ("append_state_update", "scaled_dot_product_attention")
    },
    "dual_linear_silu_multiply": {
        ("linear", "linear", "silu", "multiply")
    },
    "linear_residual": {("linear", "residual_add")},
    "linear_split_3way": {("linear", "split")},
    "linear_split_recurrent_depthwise_gate": {
        (
            "linear",
            "split",
            "multiply",
            "rolling_state_update",
            "depthwise_conv1d",
            "multiply",
        )
    },
    "multiply_rolling_depthwise": {
        ("multiply", "rolling_state_update", "depthwise_conv1d")
    },
    "multiply_rolling_depthwise_gate": {
        ("multiply", "rolling_state_update", "depthwise_conv1d", "multiply")
    },
    "parallel_head_norm_rope_2way": {
        (
            "rms_norm_per_head",
            "rotary_position_embedding",
            "rms_norm_per_head",
            "rotary_position_embedding",
        )
    },
    "parallel_linear_2way": {("linear", "linear")},
    "parallel_linear_3way": {("linear", "linear", "linear")},
    "silu_multiply": {("silu", "multiply")},
}


def build_behavioral_validation(
    *,
    model_graph: Json,
    tensor_index: Json,
    lowered_index: Json,
    source_circuits: dict[str, Json],
    candidate_circuits: dict[str, Json],
    empirical_evidence: Json | None = None,
) -> Json:
    circuit_evidence = []
    for circuit_ref in lowered_index["graph"]["circuits"]:
        pedal_id = circuit_ref["id"]
        try:
            source = source_circuits[pedal_id]
            candidate = candidate_circuits[pedal_id]
        except KeyError as error:
            raise ModelCompileError(
                f"behavioral compiler is missing circuit candidate {pedal_id!r}"
            ) from error
        circuit_evidence.append(
            prove_exact_circuit_candidate(
                pedal_id=pedal_id,
                source=source,
                candidate=candidate,
                empirical_evidence=empirical_evidence,
            )
        )

    exact = all(item["candidate_kind"] == "exact_reference" for item in circuit_evidence)
    if not exact:
        _require_closed_loop_evidence(empirical_evidence)
    return {
        "schema": BEHAVIORAL_VALIDATION_SCHEMA,
        "status": "passed",
        "candidate_kind": "exact_reference" if exact else "approximate",
        "source_oracle": {
            "kind": "source_checkpoint_contract",
            "model_contract_digest": model_contract_digest(model_graph, tensor_index),
            "tensor_count": len(tensor_index["tensors"]),
            "parameter_count": int(tensor_index["totals"]["parameter_count"]),
            "byte_count": int(tensor_index["totals"]["byte_count"]),
        },
        "admission_gate": {
            "exact_candidates": "proof_carrying_rewrite",
            "approximate_candidates": [
                "teacher_forced_oracle_validation",
                "free_running_oracle_validation",
            ],
        },
        "teacher_forced": (
            {"status": "not_required", "reason": "exact_reference_proof"}
            if exact
            else empirical_evidence["teacher_forced"]
        ),
        "free_running": (
            {"status": "not_required", "reason": "exact_reference_proof"}
            if exact
            else empirical_evidence["free_running"]
        ),
        "circuits": circuit_evidence,
    }


def prove_exact_circuit_candidate(
    *,
    pedal_id: str,
    source: Json,
    candidate: Json,
    empirical_evidence: Json | None = None,
) -> Json:
    contract_fields = (
        "schema",
        "source",
        "boundary",
        "state_ports",
        "parameters",
        "behavioral_error_contract",
    )
    drift = [field for field in contract_fields if source.get(field) != candidate.get(field)]
    if drift:
        _require_closed_loop_evidence(empirical_evidence)
        return _approximate_candidate_evidence(pedal_id, drift)

    source_nodes = source.get("nodes", [])
    candidate_nodes = candidate.get("nodes", [])
    source_by_id = {node["id"]: node for node in source_nodes}
    if len(source_by_id) != len(source_nodes):
        raise ModelCompileError(f"source circuit {pedal_id!r} contains duplicate node ids")
    source_positions = {node["id"]: index for index, node in enumerate(source_nodes)}
    consumers = _signal_consumers(source_nodes)
    boundary_outputs = {
        port.get("source", port["id"])
        for port in source.get("boundary", {}).get("outputs", [])
    }
    covered: list[str] = []
    rewrites = []
    for node in candidate_nodes:
        source_node = source_by_id.get(node["id"])
        compiled_from = node.get("attrs", {}).get("compiled_from")
        if source_node is not None and compiled_from is None:
            if node != source_node:
                _require_closed_loop_evidence(empirical_evidence)
                return _approximate_candidate_evidence(
                    pedal_id, [f"node:{node['id']}"]
                )
            covered.append(node["id"])
            continue

        if not isinstance(compiled_from, list) or not compiled_from:
            _require_closed_loop_evidence(empirical_evidence)
            return _approximate_candidate_evidence(
                pedal_id, [f"unproven_node:{node.get('id', '<missing>')}"]
            )
        if any(source_id not in source_by_id for source_id in compiled_from):
            unknown = [source_id for source_id in compiled_from if source_id not in source_by_id]
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {node['id']!r} references "
                f"unknown source nodes {unknown}"
            )
        proof_contract = EXACT_REWRITE_CONTRACTS.get(node.get("op"))
        if proof_contract is None:
            _require_closed_loop_evidence(empirical_evidence)
            return _approximate_candidate_evidence(
                pedal_id, [f"unproven_rewrite:{node.get('op', '<missing>')}"]
            )
        region = [source_by_id[source_id] for source_id in compiled_from]
        source_ops = tuple(node["op"] for node in region)
        if source_ops not in EXACT_REWRITE_SOURCE_OPS[node["op"]]:
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {node['id']!r} cannot use "
                f"proof contract {proof_contract} for source ops {source_ops}"
            )
        if node["op"] == "silu_multiply":
            source_element_count = region[0].get("attrs", {}).get("element_count")
            if (
                not isinstance(source_element_count, int)
                or source_element_count <= 0
                or node.get("attrs", {}).get("element_count") != source_element_count
            ):
                raise ModelCompileError(
                    f"candidate circuit {pedal_id!r} rewrite {node['id']!r} does not "
                    "preserve the SiLU signal extent"
                )
        _validate_rewrite_interface(
            pedal_id=pedal_id,
            candidate_node=node,
            region=region,
            region_ids=set(compiled_from),
            consumers=consumers,
            boundary_outputs=boundary_outputs,
        )
        covered.extend(compiled_from)
        rewrites.append(
            {
                "candidate_node": node["id"],
                "candidate_op": node["op"],
                "source_nodes": compiled_from,
                "source_positions": [source_positions[source_id] for source_id in compiled_from],
                "proof_contract": proof_contract,
            }
        )

    duplicates = sorted(node_id for node_id, count in Counter(covered).items() if count != 1)
    missing = sorted(set(source_by_id) - set(covered))
    if duplicates or missing:
        raise ModelCompileError(
            f"candidate circuit {pedal_id!r} does not exactly cover its source graph; "
            f"missing={missing}, duplicate={duplicates}"
        )
    return {
        "pedal_id": pedal_id,
        "candidate_kind": "exact_reference",
        "status": "passed",
        "source_node_count": len(source_nodes),
        "candidate_node_count": len(candidate_nodes),
        "covered_source_node_count": len(covered),
        "rewrite_count": len(rewrites),
        "rewrites": rewrites,
    }


def model_contract_digest(model_graph: Json, tensor_index: Json) -> str:
    tensor_contracts = {
        name: {
            key: metadata.get(key)
            for key in ("dtype", "shape", "logical_shape", "parameter_count", "byte_count")
        }
        for name, metadata in sorted(tensor_index["tensors"].items())
    }
    contract = {
        "architecture": model_graph.get("architecture"),
        "dimensions": model_graph.get("dimensions"),
        "numerics": model_graph.get("numerics"),
        "graph": model_graph.get("graph"),
        "tensors": tensor_contracts,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_rewrite_interface(
    *,
    pedal_id: str,
    candidate_node: Json,
    region: list[Json],
    region_ids: set[str],
    consumers: dict[str, set[str]],
    boundary_outputs: set[str],
) -> None:
    produced = {signal for node in region for signal in node.get("outputs", [])}
    inputs = {
        signal
        for node in region
        for signal in node.get("inputs", [])
        if signal not in produced
    }
    outputs = {
        signal
        for node in region
        for signal in node.get("outputs", [])
        if signal in boundary_outputs
        or any(consumer not in region_ids for consumer in consumers.get(signal, set()))
    }
    params = {param for node in region for param in node.get("params", [])}
    state_reads = {state for node in region for state in node.get("state_reads", [])}
    state_writes = {state for node in region for state in node.get("state_writes", [])}
    comparisons = {
        "inputs": (inputs, set(candidate_node.get("inputs", []))),
        "outputs": (outputs, set(candidate_node.get("outputs", []))),
        "params": (params, set(candidate_node.get("params", []))),
        "state_reads": (state_reads, set(candidate_node.get("state_reads", []))),
        "state_writes": (state_writes, set(candidate_node.get("state_writes", []))),
    }
    drift = {
        name: {"source": sorted(source), "candidate": sorted(candidate)}
        for name, (source, candidate) in comparisons.items()
        if source != candidate
    }
    if drift:
        raise ModelCompileError(
            f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
            f"changes its observable region interface: {drift}"
        )


def _signal_consumers(nodes: list[Json]) -> dict[str, set[str]]:
    consumers: dict[str, set[str]] = {}
    for node in nodes:
        for signal in node.get("inputs", []):
            consumers.setdefault(signal, set()).add(node["id"])
    return consumers


def _require_closed_loop_evidence(evidence: Json | None) -> None:
    if evidence is None:
        raise ModelCompileError(
            "behavioral compiler rejected a non-exact candidate without source-oracle evidence"
        )
    for mode in ("teacher_forced", "free_running"):
        result = evidence.get(mode)
        if not isinstance(result, dict) or result.get("status") != "passed":
            raise ModelCompileError(
                f"behavioral compiler requires passing {mode.replace('_', '-')} oracle evidence"
            )


def _approximate_candidate_evidence(pedal_id: str, drift: list[str]) -> Json:
    return {
        "pedal_id": pedal_id,
        "candidate_kind": "approximate",
        "status": "passed",
        "drift": drift,
    }
