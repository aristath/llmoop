from __future__ import annotations

from copy import deepcopy

import pytest

from llmoop.behavioral_compiler import prove_exact_circuit_candidate
from llmoop.compilation import ModelCompileError


def source_circuit() -> dict:
    return {
        "schema": "llmoop.stream_circuit.v1",
        "source": {"pedal_id": "layer_00"},
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


def test_exact_candidate_gate_proves_complete_fusion_coverage() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "activation__multiply",
            "op": "silu_multiply",
            "inputs": ["x", "gate"],
            "outputs": ["y"],
            "attrs": {
                "compiled_from": ["activation", "multiply"],
                "element_count": 4,
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        pedal_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert evidence["covered_source_node_count"] == 2
    assert evidence["rewrites"][0]["proof_contract"] == "silu_multiply_exact_bf16.v1"


def test_exact_candidate_gate_rejects_dropped_source_behavior() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["nodes"] = [deepcopy(source["nodes"][0])]

    with pytest.raises(ModelCompileError, match="does not exactly cover"):
        prove_exact_circuit_candidate(
            pedal_id="layer_00", source=source, candidate=candidate
        )


def test_approximate_candidate_requires_both_closed_loop_evidence_modes() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["boundary"]["outputs"][0]["source"] = "approximate_y"

    with pytest.raises(ModelCompileError, match="without source-oracle evidence"):
        prove_exact_circuit_candidate(
            pedal_id="layer_00", source=source, candidate=candidate
        )
    with pytest.raises(ModelCompileError, match="free-running"):
        prove_exact_circuit_candidate(
            pedal_id="layer_00",
            source=source,
            candidate=candidate,
            empirical_evidence={
                "teacher_forced": {"status": "passed"},
                "free_running": {"status": "failed"},
            },
        )

    evidence = prove_exact_circuit_candidate(
        pedal_id="layer_00",
        source=source,
        candidate=candidate,
        empirical_evidence={
            "teacher_forced": {"status": "passed"},
            "free_running": {"status": "passed"},
        },
    )
    assert evidence["candidate_kind"] == "approximate"
