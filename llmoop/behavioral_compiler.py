from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from copy import deepcopy
from typing import Any

from llmoop.compilation import Json, ModelCompileError


BEHAVIORAL_VALIDATION_SCHEMA = "llmoop.behavioral_validation.v1"
BEHAVIORAL_EMPIRICAL_EVIDENCE_SCHEMA = "llmoop.behavioral_empirical_evidence.v1"
CONTRACT_DIGEST_ALGORITHM = "llmoop.json_tree_sha256.v1"
EXACT_REWRITE_CONTRACTS = {
    "append_scaled_dot_product_attention": "append_attention_exact_bf16.v1",
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
    oracle_digest = model_contract_digest(model_graph, tensor_index)
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
                expected_model_contract_digest=oracle_digest,
            )
        )

    exact = all(item["candidate_kind"] == "exact_reference" for item in circuit_evidence)
    if not exact:
        _require_closed_loop_evidence(empirical_evidence)
    return {
        "schema": BEHAVIORAL_VALIDATION_SCHEMA,
        "status": "passed",
        "candidate_kind": "exact_reference" if exact else "approximate",
        "candidate_contract_digest_algorithm": CONTRACT_DIGEST_ALGORITHM,
        "source_oracle": {
            "kind": "source_checkpoint_contract",
            "model_contract_digest": oracle_digest,
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
    expected_model_contract_digest: str | None = None,
) -> Json:
    candidate_digest = json_contract_digest(candidate)
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
        _require_closed_loop_evidence(
            empirical_evidence, expected_model_contract_digest
        )
        return _approximate_candidate_evidence(
            pedal_id, drift, candidate_digest, len(candidate.get("nodes", []))
        )

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
                _require_closed_loop_evidence(
                    empirical_evidence, expected_model_contract_digest
                )
                return _approximate_candidate_evidence(
                    pedal_id,
                    [f"node:{node['id']}"],
                    candidate_digest,
                    len(candidate_nodes),
                )
            covered.append(node["id"])
            continue

        if not isinstance(compiled_from, list) or not compiled_from:
            _require_closed_loop_evidence(
                empirical_evidence, expected_model_contract_digest
            )
            return _approximate_candidate_evidence(
                pedal_id,
                [f"unproven_node:{node.get('id', '<missing>')}"],
                candidate_digest,
                len(candidate_nodes),
            )
        if len(set(compiled_from)) != len(compiled_from):
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {node['id']!r} repeats "
                "a source node in compiled_from"
            )
        if any(source_id not in source_by_id for source_id in compiled_from):
            unknown = [source_id for source_id in compiled_from if source_id not in source_by_id]
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {node['id']!r} references "
                f"unknown source nodes {unknown}"
            )
        proof_contract = EXACT_REWRITE_CONTRACTS.get(node.get("op"))
        if proof_contract is None:
            _require_closed_loop_evidence(
                empirical_evidence, expected_model_contract_digest
            )
            return _approximate_candidate_evidence(
                pedal_id,
                [f"unproven_rewrite:{node.get('op', '<missing>')}"],
                candidate_digest,
                len(candidate_nodes),
            )
        region = [source_by_id[source_id] for source_id in compiled_from]
        source_ops = tuple(node["op"] for node in region)
        if source_ops not in EXACT_REWRITE_SOURCE_OPS[node["op"]]:
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {node['id']!r} cannot use "
                f"proof contract {proof_contract} for source ops {source_ops}"
            )
        _validate_exact_rewrite_semantics(
            pedal_id=pedal_id,
            candidate_node=node,
            region=region,
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
        "candidate_contract_digest": candidate_digest,
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


def json_contract_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_json_tree_digest(digest, value)
    return digest.hexdigest()


def _update_json_tree_digest(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"n")
    elif value is False:
        digest.update(b"f")
    elif value is True:
        digest.update(b"t")
    elif isinstance(value, int):
        digest.update(b"i")
        _update_length_prefixed(digest, str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ModelCompileError("contract digest cannot encode a non-finite number")
        digest.update(b"d")
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        digest.update(b"s")
        _update_length_prefixed(digest, value.encode("utf-8"))
    elif isinstance(value, list):
        digest.update(b"l")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_json_tree_digest(digest, item)
    elif isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ModelCompileError("contract digest object keys must be strings")
        digest.update(b"o")
        digest.update(len(value).to_bytes(8, "big"))
        for key in sorted(value):
            _update_json_tree_digest(digest, key)
            _update_json_tree_digest(digest, value[key])
    else:
        raise ModelCompileError(
            f"contract digest cannot encode value of type {type(value).__name__}"
        )


def _update_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def validate_behavioral_validation_artifact(
    evidence: Json,
    candidate_circuits: dict[str, Json],
) -> None:
    if (
        evidence.get("schema") != BEHAVIORAL_VALIDATION_SCHEMA
        or evidence.get("status") != "passed"
    ):
        raise ModelCompileError("behavioral validation artifact has not passed")
    candidate_kind = evidence.get("candidate_kind")
    if candidate_kind not in {"exact_reference", "approximate"}:
        raise ModelCompileError(
            f"behavioral validation artifact has unsupported candidate kind {candidate_kind!r}"
        )
    if evidence.get("candidate_contract_digest_algorithm") != CONTRACT_DIGEST_ALGORITHM:
        raise ModelCompileError(
            "behavioral validation artifact has an unsupported candidate contract digest algorithm"
        )
    source_oracle = evidence.get("source_oracle")
    if (
        not isinstance(source_oracle, dict)
        or not _is_sha256_digest(source_oracle.get("model_contract_digest"))
        or any(
            not isinstance(source_oracle.get(field), int)
            or isinstance(source_oracle.get(field), bool)
            or source_oracle[field] < 0
            for field in ("tensor_count", "parameter_count", "byte_count")
        )
    ):
        raise ModelCompileError(
            "behavioral validation artifact has an incomplete source oracle contract"
        )

    for mode in ("teacher_forced", "free_running"):
        result = evidence.get(mode)
        if candidate_kind == "exact_reference":
            if not isinstance(result, dict) or result.get("status") != "not_required":
                raise ModelCompileError(
                    f"exact behavioral validation must mark {mode} evidence not_required"
                )
        elif (
            not isinstance(result, dict)
            or result.get("status") != "passed"
            or not isinstance(result.get("sample_count"), int)
            or isinstance(result.get("sample_count"), bool)
            or result["sample_count"] <= 0
            or not isinstance(result.get("metrics"), dict)
            or not result["metrics"]
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in result["metrics"].values()
            )
        ):
            raise ModelCompileError(
                f"approximate behavioral validation lacks measured passing {mode} evidence"
            )

    circuits = evidence.get("circuits")
    if not isinstance(circuits, list):
        raise ModelCompileError("behavioral validation artifact has no circuit proofs")
    proof_by_pedal: dict[str, Json] = {}
    for proof in circuits:
        pedal_id = proof.get("pedal_id") if isinstance(proof, dict) else None
        if not isinstance(pedal_id, str) or not pedal_id:
            raise ModelCompileError(
                "behavioral validation artifact contains a proof without a pedal id"
            )
        if pedal_id in proof_by_pedal:
            raise ModelCompileError(
                f"behavioral validation artifact repeats pedal {pedal_id!r}"
            )
        proof_by_pedal[pedal_id] = proof
    if set(proof_by_pedal) != set(candidate_circuits):
        raise ModelCompileError(
            "behavioral validation artifact does not prove every packaged pedal"
        )
    approximate_proof_count = 0
    for pedal_id, candidate in candidate_circuits.items():
        proof = proof_by_pedal[pedal_id]
        proof_kind = proof.get("candidate_kind")
        if (
            proof.get("status") != "passed"
            or proof_kind not in {"exact_reference", "approximate"}
            or (candidate_kind == "exact_reference" and proof_kind != "exact_reference")
            or proof.get("candidate_node_count") != len(candidate.get("nodes", []))
            or proof.get("candidate_contract_digest")
            != json_contract_digest(candidate)
        ):
            raise ModelCompileError(
                f"behavioral validation artifact has an incomplete or stale proof for pedal {pedal_id!r}"
            )
        if proof_kind == "approximate":
            approximate_proof_count += 1
        elif (
            proof.get("source_node_count")
            != proof.get("covered_source_node_count")
        ):
            raise ModelCompileError(
                f"behavioral validation artifact does not completely cover pedal {pedal_id!r}"
            )
    if candidate_kind == "approximate" and approximate_proof_count == 0:
        raise ModelCompileError(
            "approximate behavioral validation artifact contains no approximate pedal proof"
        )


def _validate_exact_rewrite_semantics(
    *,
    pedal_id: str,
    candidate_node: Json,
    region: list[Json],
) -> None:
    op = candidate_node["op"]
    source_ids = [node["id"] for node in region]
    attrs = candidate_node.get("attrs", {})
    expected_attrs: Json

    if op in {"parallel_linear_2way", "parallel_linear_3way"}:
        _require_empty_attrs(pedal_id, op, region)
        expected_attrs = {
            "compiled_from": source_ids,
            "branch_count": len(region),
        }
    elif op == "linear_residual":
        _require_empty_attrs(pedal_id, op, region)
        expected_attrs = {
            "compiled_from": source_ids,
            "intermediate_rounding": "BF16",
        }
    elif op == "silu_multiply":
        activation_attrs = region[0].get("attrs", {})
        element_count = activation_attrs.get("element_count")
        if (
            not isinstance(element_count, int)
            or element_count <= 0
            or set(activation_attrs) != {"element_count"}
            or region[1].get("attrs", {})
        ):
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
                "cannot prove the SiLU/multiply source attributes"
            )
        expected_attrs = {
            "compiled_from": source_ids,
            "intermediate_rounding": "BF16",
            "element_count": element_count,
        }
    elif op == "linear_split_3way":
        _require_empty_attrs(pedal_id, op, region[:1])
        split_attrs = deepcopy(region[1].get("attrs", {}))
        part_widths = split_attrs.get("part_widths")
        if part_widths is None:
            part_width = split_attrs.get("part_width")
            if not isinstance(part_width, int):
                raise ModelCompileError(
                    f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
                    "does not preserve a provable split width"
                )
            part_widths = [part_width] * 3
        if (
            not isinstance(part_widths, list)
            or len(part_widths) != 3
            or any(not isinstance(width, int) or width <= 0 for width in part_widths)
        ):
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
                "does not preserve provable split widths"
            )
        split_attrs["part_widths"] = part_widths
        expected_attrs = {
            **split_attrs,
            "compiled_from": source_ids,
            "intermediate_rounding": "BF16",
        }
    elif op in {"multiply_rolling_depthwise", "multiply_rolling_depthwise_gate"}:
        multiply, rolling, depthwise = region[:3]
        expected_attrs = {
            "compiled_from": source_ids,
            "multiply": deepcopy(multiply.get("attrs", {})),
            "rolling": deepcopy(rolling.get("attrs", {})),
            "depthwise": deepcopy(depthwise.get("attrs", {})),
            "intermediate_rounding": "BF16",
        }
        if op == "multiply_rolling_depthwise_gate":
            if region[3].get("attrs", {}):
                raise ModelCompileError(
                    f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
                    "drops output-gate attributes"
                )
            expected_attrs["output_gate_rounding"] = "BF16"
    elif op == "parallel_head_norm_rope_2way":
        expected_attrs = {
            "compiled_from": source_ids,
            "branches": [
                {
                    "norm": deepcopy(region[0].get("attrs", {})),
                    "rope": deepcopy(region[1].get("attrs", {})),
                },
                {
                    "norm": deepcopy(region[2].get("attrs", {})),
                    "rope": deepcopy(region[3].get("attrs", {})),
                },
            ],
            "intermediate_rounding": "BF16",
        }
    elif op == "append_scaled_dot_product_attention":
        expected_attrs = {
            "compiled_from": source_ids,
            "append": deepcopy(region[0].get("attrs", {})),
            "attention": deepcopy(region[1].get("attrs", {})),
            "current_kv_source": "direct_bf16_input",
        }
    elif op == "linear_split_recurrent_depthwise_gate":
        linear, split, multiply, rolling, depthwise, output_gate = region
        _require_empty_attrs(pedal_id, op, [linear, output_gate])
        split_attrs = deepcopy(split.get("attrs", {}))
        part_widths = split_attrs.get("part_widths")
        if part_widths is None:
            part_width = split_attrs.get("part_width")
            if not isinstance(part_width, int):
                raise ModelCompileError(
                    f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
                    "does not preserve a provable projection split width"
                )
            part_widths = [part_width] * 3
        split_attrs["part_widths"] = part_widths
        projection_attrs = {
            **split_attrs,
            "compiled_from": [linear["id"], split["id"]],
            "intermediate_rounding": "BF16",
        }
        recurrent_attrs = {
            "compiled_from": [
                multiply["id"],
                rolling["id"],
                depthwise["id"],
                output_gate["id"],
            ],
            "multiply": deepcopy(multiply.get("attrs", {})),
            "rolling": deepcopy(rolling.get("attrs", {})),
            "depthwise": deepcopy(depthwise.get("attrs", {})),
            "intermediate_rounding": "BF16",
            "output_gate_rounding": "BF16",
        }
        projection_outputs = split.get("outputs", [])
        recurrent_inputs = [*multiply.get("inputs", []), *rolling.get("inputs", [])]
        try:
            input_gate_indices = [
                projection_outputs.index(multiply["inputs"][0]),
                projection_outputs.index(multiply["inputs"][1]),
            ]
            recurrent_output = depthwise["outputs"][0]
            output_gate_signal = next(
                signal for signal in output_gate["inputs"] if signal != recurrent_output
            )
            output_gate_index = projection_outputs.index(output_gate_signal)
        except (IndexError, KeyError, StopIteration, ValueError) as error:
            raise ModelCompileError(
                f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
                f"has an unprovable recurrent branch mapping: {recurrent_inputs}"
            ) from error
        expected_attrs = {
            "compiled_from": source_ids,
            "projection": projection_attrs,
            "recurrent": recurrent_attrs,
            "input_gate_branch_indices": input_gate_indices,
            "output_gate_branch_index": output_gate_index,
            "projection_rounding": "BF16",
        }
    else:
        raise ModelCompileError(
            f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
            f"has no semantic proof implementation for {op!r}"
        )

    if attrs != expected_attrs:
        raise ModelCompileError(
            f"candidate circuit {pedal_id!r} rewrite {candidate_node['id']!r} "
            f"changes exact rewrite attributes: expected={expected_attrs!r}, candidate={attrs!r}"
        )


def _require_empty_attrs(pedal_id: str, op: str, nodes: list[Json]) -> None:
    with_attrs = [node["id"] for node in nodes if node.get("attrs", {})]
    if with_attrs:
        raise ModelCompileError(
            f"candidate circuit {pedal_id!r} rewrite {op!r} drops source attributes "
            f"from {with_attrs}"
        )


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
    inputs = _ordered_unique(
        signal
        for node in region
        for signal in node.get("inputs", [])
        if signal not in produced
    )
    if candidate_node["op"] == "append_scaled_dot_product_attention":
        inputs = [region[1]["inputs"][0], *region[0]["inputs"]]
    outputs = _ordered_unique(
        signal
        for node in region
        for signal in node.get("outputs", [])
        if signal in boundary_outputs
        or any(consumer not in region_ids for consumer in consumers.get(signal, set()))
    )
    params = [param for node in region for param in node.get("params", [])]
    state_reads = _ordered_unique(
        state for node in region for state in node.get("state_reads", [])
    )
    state_writes = _ordered_unique(
        state for node in region for state in node.get("state_writes", [])
    )
    comparisons = {
        "inputs": (inputs, candidate_node.get("inputs", [])),
        "outputs": (outputs, candidate_node.get("outputs", [])),
        "params": (params, candidate_node.get("params", [])),
        "state_reads": (state_reads, candidate_node.get("state_reads", [])),
        "state_writes": (state_writes, candidate_node.get("state_writes", [])),
    }
    drift = {
        name: {"source": source, "candidate": candidate}
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


def _ordered_unique(values: Any) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _require_closed_loop_evidence(
    evidence: Json | None,
    expected_model_contract_digest: str | None = None,
) -> None:
    if evidence is None:
        raise ModelCompileError(
            "behavioral compiler rejected a non-exact candidate without source-oracle evidence"
        )
    if evidence.get("schema") != BEHAVIORAL_EMPIRICAL_EVIDENCE_SCHEMA:
        raise ModelCompileError(
            "behavioral compiler requires versioned source-oracle evidence"
        )
    digest = evidence.get("model_contract_digest")
    if not _is_sha256_digest(digest):
        raise ModelCompileError(
            "behavioral compiler evidence lacks a valid model contract digest"
        )
    if expected_model_contract_digest is not None and digest != expected_model_contract_digest:
        raise ModelCompileError(
            "behavioral compiler evidence targets a different source model contract"
        )
    for mode in ("teacher_forced", "free_running"):
        result = evidence.get(mode)
        if not isinstance(result, dict) or result.get("status") != "passed":
            raise ModelCompileError(
                f"behavioral compiler requires passing {mode.replace('_', '-')} oracle evidence"
            )
        sample_count = result.get("sample_count")
        metrics = result.get("metrics")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
            raise ModelCompileError(
                f"behavioral compiler requires positive {mode.replace('_', '-')} sample evidence"
            )
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise ModelCompileError(
                f"behavioral compiler requires finite {mode.replace('_', '-')} validation metrics"
            )


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _approximate_candidate_evidence(
    pedal_id: str,
    drift: list[str],
    candidate_digest: str,
    candidate_node_count: int,
) -> Json:
    return {
        "pedal_id": pedal_id,
        "candidate_kind": "approximate",
        "status": "passed",
        "drift": drift,
        "candidate_contract_digest": candidate_digest,
        "candidate_node_count": candidate_node_count,
    }
