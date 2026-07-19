from __future__ import annotations

from copy import deepcopy
from typing import Callable

import pytest

from llmoop.circuit_ir import validate_circuit, validate_circuit_against_pedal


def valid_circuit() -> dict[str, object]:
    return {
        "schema": "llmoop.stream_circuit.v1",
        "id": "test_circuit",
        "boundary": {
            "inputs": [{"id": "input_frame", "signal": "frame", "shape": [8]}],
            "outputs": [
                {
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                }
            ],
        },
        "state_ports": [
            {
                "id": "memory",
                "type": "rolling_memory",
                "shape": [2, 8],
                "update": "replace",
            }
        ],
        "parameters": {"refs": {"weight": {"tensor": "layer.weight"}}},
        "nodes": [
            {
                "id": "project",
                "op": "linear",
                "inputs": ["input_frame"],
                "outputs": ["projected"],
                "params": ["weight"],
                "state_reads": ["memory"],
                "state_writes": ["memory"],
            },
            {
                "id": "residual",
                "op": "add",
                "inputs": ["projected", "input_frame"],
                "outputs": ["output_frame"],
                "params": [],
                "state_reads": [],
                "state_writes": [],
            },
        ],
    }


def matching_pedal() -> dict[str, object]:
    return {
        "ports": {
            "inputs": [{"id": "input", "signal": "frame", "shape": [8]}],
            "outputs": [{"id": "output", "signal": "frame", "shape": [8]}],
        },
        "state_ports": [
            {
                "id": "memory",
                "type": "rolling_memory",
                "shape": [2, 8],
                "update": "replace",
            }
        ],
        "parameter_block": {
            "params": {"weight": {"tensor": "layer.weight"}}
        },
    }


def duplicate_node_id(circuit: dict[str, object]) -> None:
    circuit["nodes"][1]["id"] = "project"  # type: ignore[index]


def consume_unknown_signal(circuit: dict[str, object]) -> None:
    circuit["nodes"][0]["inputs"] = ["not_produced"]  # type: ignore[index]


def produce_signal_twice(circuit: dict[str, object]) -> None:
    circuit["nodes"][1]["outputs"] = ["projected"]  # type: ignore[index]


def reference_unknown_parameter(circuit: dict[str, object]) -> None:
    circuit["nodes"][0]["params"] = ["missing_weight"]  # type: ignore[index]


def reference_unknown_state(circuit: dict[str, object]) -> None:
    circuit["nodes"][0]["state_writes"] = ["missing_state"]  # type: ignore[index]


def expose_unknown_output(circuit: dict[str, object]) -> None:
    circuit["boundary"]["outputs"][0]["source"] = "not_produced"  # type: ignore[index]


@pytest.mark.parametrize(
    ("corrupt", "message", "path"),
    [
        (duplicate_node_id, "duplicate node id", "nodes[1].id"),
        (consume_unknown_signal, "has not been produced", "nodes[0].inputs"),
        (produce_signal_twice, "already produced", "nodes[1].outputs"),
        (reference_unknown_parameter, "is not declared", "nodes[0].params"),
        (reference_unknown_state, "is not declared", "nodes[0].state_writes"),
        (expose_unknown_output, "is not produced", "boundary.outputs[0].source"),
    ],
)
def test_circuit_validation_rejects_broken_dataflow_contracts(
    corrupt: Callable[[dict[str, object]], None], message: str, path: str
) -> None:
    circuit = valid_circuit()
    corrupt(circuit)

    report = validate_circuit(circuit)

    assert not report.ok
    assert any(message in issue.message and issue.path == path for issue in report.errors)
    with pytest.raises(ValueError, match=message):
        report.raise_for_errors()


def test_valid_circuit_and_pedal_contract_has_no_false_positive_errors() -> None:
    report = validate_circuit_against_pedal(valid_circuit(), matching_pedal())

    assert report.ok, report.to_json()
    assert not report.errors


def test_pedal_contract_validation_catches_interface_state_and_tensor_drift() -> None:
    pedal = matching_pedal()
    pedal["ports"]["outputs"][0]["shape"] = [16]  # type: ignore[index]
    pedal["state_ports"][0]["update"] = "append"  # type: ignore[index]
    pedal["parameter_block"]["params"]["weight"]["tensor"] = "other.weight"  # type: ignore[index]

    report = validate_circuit_against_pedal(deepcopy(valid_circuit()), pedal)
    errors = {(issue.path, issue.message) for issue in report.errors}

    assert any(path == "boundary.outputs[0].shape" for path, _message in errors)
    assert any(path == "state_ports.memory.update" for path, _message in errors)
    assert any(path == "parameters.refs.weight.tensor" for path, _message in errors)
