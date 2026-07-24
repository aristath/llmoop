from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.behavioral_compiler import (
    build_behavioral_validation,
    model_contract_digest,
    prove_exact_circuit_candidate,
    validate_behavioral_validation_artifact,
)
from nerve.compilation import ModelCompileError


def source_circuit() -> dict:
    return {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input_frame", "source": "x"}],
            "outputs": [{"id": "output_frame", "source": "y"}],
        },
        "state_ports": [],
        "parameters": {"refs": {}},
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "activation",
                "op": "silu",
                "inputs": ["x"],
                "outputs": ["activated"],
                "attrs": {"element_count": 4},
            },
            {
                "id": "multiply",
                "op": "multiply",
                "inputs": ["activated", "gate"],
                "outputs": ["y"],
            },
        ],
    }


def empirical_evidence(*, free_running_status: str = "passed") -> dict:
    return {
        "schema": "nerve.behavioral_empirical_evidence.v1",
        "model_contract_digest": "a" * 64,
        "teacher_forced": {
            "status": "passed",
            "sample_count": 128,
            "metrics": {"maximum_logit_error": 0.01},
        },
        "free_running": {
            "status": free_running_status,
            "sample_count": 64,
            "metrics": {"distribution_similarity": 0.99},
        },
    }


def fused_candidate(source: dict) -> dict:
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "activation__multiply",
            "op": "silu_multiply",
            "inputs": ["x", "gate"],
            "outputs": ["y"],
            "attrs": {
                "compiled_from": ["activation", "multiply"],
                "intermediate_rounding": "BF16",
                "element_count": 4,
            },
        }
    ]
    return candidate


def test_exact_candidate_gate_proves_complete_fusion_coverage() -> None:
    source = source_circuit()
    candidate = fused_candidate(source)

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert evidence["covered_source_node_count"] == 2
    assert evidence["rewrites"][0]["proof_contract"] == "silu_multiply_exact_bf16.v1"


def test_exact_candidate_gate_proves_fused_parallel_ffn_projection() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input", "source": "x"}],
            "outputs": [{"id": "output", "source": "y"}],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "gate_weight": {"tensor": "gate.weight"},
                "up_weight": {"tensor": "up.weight"},
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "gate",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["gate_projection"],
                "params": ["gate_weight"],
            },
            {
                "id": "up",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["up_projection"],
                "params": ["up_weight"],
            },
            {
                "id": "activation",
                "op": "silu",
                "inputs": ["gate_projection"],
                "outputs": ["activated"],
                "attrs": {"element_count": 4},
            },
            {
                "id": "multiply",
                "op": "multiply",
                "inputs": ["activated", "up_projection"],
                "outputs": ["y"],
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "fused_ffn",
            "op": "parallel_linear_silu_multiply",
            "inputs": ["x"],
            "outputs": ["y"],
            "params": ["gate_weight", "up_weight"],
            "attrs": {
                "compiled_from": ["gate", "up", "activation", "multiply"],
                "branch_count": 2,
                "intermediate_rounding": "BF16",
                "element_count": 4,
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert (
        evidence["rewrites"][0]["proof_contract"]
        == "parallel_linear_silu_multiply_exact_bf16.v1"
    )


def test_exact_candidate_gate_proves_fp8_parallel_linear_parameter_pairs() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input", "source": "x"}],
            "outputs": [
                {"id": "query", "source": "q"},
                {"id": "key", "source": "k"},
            ],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "q_weight": {"tensor": "q.weight"},
                "q_weight_scale_inv": {"tensor": "q.weight_scale_inv"},
                "k_weight": {"tensor": "k.weight"},
                "k_weight_scale_inv": {"tensor": "k.weight_scale_inv"},
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "q_projection",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["q"],
                "params": ["q_weight", "q_weight_scale_inv"],
            },
            {
                "id": "k_projection",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["k"],
                "params": ["k_weight", "k_weight_scale_inv"],
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "q_projection__k_projection__quantize_input",
            "op": "quantize_fp8_e4m3",
            "inputs": ["x"],
            "outputs": ["x_fp8", "x_scale"],
            "attrs": {
                "physical_representation_contract": (
                    "bf16_blockwise_fp8_e4m3_f32_scale.v1"
                ),
                "consumer_node_ids": ["q_projection__k_projection"],
                "semantic_source_node_ids": ["q_projection", "k_projection"],
                "element_count": 5120,
                "block_columns": 128,
                "output_element_bytes": [1, 4],
            },
        },
        {
            "id": "q_projection__k_projection",
            "op": "parallel_linear_2way",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["q", "k"],
            "params": [
                "q_weight",
                "q_weight_scale_inv",
                "k_weight",
                "k_weight_scale_inv",
            ],
            "attrs": {
                "compiled_from": ["q_projection", "k_projection"],
                "branch_count": 2,
                "branch_parameter_counts": [2, 2],
                "physical_input_contract": (
                    "bf16_blockwise_fp8_e4m3_f32_scale.v1"
                ),
                "physical_input_helper_id": (
                    "q_projection__k_projection__quantize_input"
                ),
                "physical_logical_inputs": ["x"],
                "output_element_bytes": [2, 2],
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert evidence["physical_helper_count"] == 1
    assert evidence["rewrites"][0]["proof_contract"] == "parallel_linear_exact_bf16.v1"


def test_exact_candidate_gate_rejects_dropped_source_behavior() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["nodes"] = [deepcopy(source["nodes"][0])]

    with pytest.raises(ModelCompileError, match="does not exactly cover"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_exact_candidate_gate_rejects_reordered_interface_and_specialization() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "activation__multiply",
            "op": "silu_multiply",
            "inputs": ["gate", "x"],
            "outputs": ["y"],
            "attrs": {
                "compiled_from": ["activation", "multiply"],
                "intermediate_rounding": "BF16",
                "element_count": 4,
            },
        }
    ]
    with pytest.raises(ModelCompileError, match="observable region interface"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )

    candidate["nodes"][0]["inputs"] = ["x", "gate"]
    candidate["nodes"][0]["attrs"]["intermediate_rounding"] = "F32"
    with pytest.raises(ModelCompileError, match="exact rewrite attributes"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_approximate_candidate_requires_both_closed_loop_evidence_modes() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["boundary"]["outputs"][0]["source"] = "approximate_y"

    with pytest.raises(ModelCompileError, match="without source-oracle evidence"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )
    with pytest.raises(ModelCompileError, match="versioned source-oracle"):
        prove_exact_circuit_candidate(
            component_id="layer_00",
            source=source,
            candidate=candidate,
            empirical_evidence={
                "teacher_forced": {"status": "passed"},
                "free_running": {"status": "failed"},
            },
        )

    with pytest.raises(ModelCompileError, match="free-running"):
        prove_exact_circuit_candidate(
            component_id="layer_00",
            source=source,
            candidate=candidate,
            empirical_evidence=empirical_evidence(free_running_status="failed"),
        )

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00",
        source=source,
        candidate=candidate,
        empirical_evidence=empirical_evidence(),
    )
    assert evidence["candidate_kind"] == "approximate"
    assert len(evidence["candidate_contract_digest"]) == 64


def test_behavioral_validation_accepts_mixed_exact_and_approximate_components() -> None:
    model_graph = {
        "architecture": {"family": "fixture"},
        "dimensions": {"hidden_size": 4},
        "numerics": {"activation_dtype": "BF16"},
        "graph": {"topology": "series"},
    }
    tensor_index = {
        "tensors": {},
        "totals": {"parameter_count": 0, "byte_count": 0},
    }
    source_exact = source_circuit()
    source_approximate = deepcopy(source_exact)
    source_approximate["source"]["component_id"] = "layer_01"
    candidate_exact = fused_candidate(source_exact)
    candidate_approximate = deepcopy(source_approximate)
    candidate_approximate["boundary"]["outputs"][0]["source"] = "approximate_y"
    empirical = empirical_evidence()
    empirical["model_contract_digest"] = model_contract_digest(
        model_graph, tensor_index
    )

    validation = build_behavioral_validation(
        model_graph=model_graph,
        tensor_index=tensor_index,
        lowered_index={
            "graph": {
                "circuits": [
                    {"id": "layer_00"},
                    {"id": "layer_01"},
                ]
            }
        },
        source_circuits={
            "layer_00": source_exact,
            "layer_01": source_approximate,
        },
        candidate_circuits={
            "layer_00": candidate_exact,
            "layer_01": candidate_approximate,
        },
        empirical_evidence=empirical,
    )

    assert validation["candidate_kind"] == "approximate"
    assert [proof["candidate_kind"] for proof in validation["circuits"]] == [
        "exact_reference",
        "approximate",
    ]
    validate_behavioral_validation_artifact(
        validation,
        {"layer_00": candidate_exact, "layer_01": candidate_approximate},
    )

    mislabeled = deepcopy(validation)
    mislabeled["circuits"][1].update(
        {
            "candidate_kind": "exact_reference",
            "source_node_count": 2,
            "covered_source_node_count": 2,
        }
    )
    with pytest.raises(ModelCompileError, match="no approximate component proof"):
        validate_behavioral_validation_artifact(
            mislabeled,
            {"layer_00": candidate_exact, "layer_01": candidate_approximate},
        )
