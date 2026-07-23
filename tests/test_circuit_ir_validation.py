from __future__ import annotations

from copy import deepcopy
from typing import Callable

import pytest

from nerve.circuit_ir import validate_circuit, validate_circuit_against_component


def valid_circuit() -> dict[str, object]:
    return {
        "schema": "nerve.stream_circuit.v1",
        "id": "test_circuit",
        "runtime_role": "signal_processor",
        "boundary": {
            "inputs": [
                {
                    "id": "input_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "input",
                }
            ],
            "outputs": [
                {
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                    "component_port": "output",
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


def matching_component() -> dict[str, object]:
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
        "parameter_block": {"params": {"weight": {"tensor": "layer.weight"}}},
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
    assert any(
        message in issue.message and issue.path == path for issue in report.errors
    )
    with pytest.raises(ValueError, match=message):
        report.raise_for_errors()


def test_valid_circuit_and_component_contract_has_no_false_positive_errors() -> None:
    report = validate_circuit_against_component(valid_circuit(), matching_component())

    assert report.ok, report.to_json()
    assert not report.errors


def test_component_contract_validation_catches_interface_state_and_tensor_drift() -> None:
    component = matching_component()
    component["ports"]["outputs"][0]["shape"] = [16]  # type: ignore[index]
    component["state_ports"][0]["update"] = "append"  # type: ignore[index]
    component["parameter_block"]["params"]["weight"]["tensor"] = "other.weight"  # type: ignore[index]

    report = validate_circuit_against_component(deepcopy(valid_circuit()), component)
    errors = {(issue.path, issue.message) for issue in report.errors}

    assert any(path == "boundary.outputs[0].shape" for path, _message in errors)
    assert any(path == "state_ports.memory.update" for path, _message in errors)
    assert any(path == "parameters.refs.weight.tensor" for path, _message in errors)


@pytest.mark.parametrize(
    ("boundary_side", "shape", "message"),
    [
        ("inputs", [], "non-empty list of positive integers"),
        ("inputs", [0, 8], "non-empty list of positive integers"),
        ("outputs", [True, 8], "non-empty list of positive integers"),
    ],
)
def test_circuit_validation_rejects_non_tensor_boundary_shapes(
    boundary_side: str, shape: list[object], message: str
) -> None:
    circuit = valid_circuit()
    circuit["boundary"][boundary_side][0]["shape"] = shape  # type: ignore[index]

    report = validate_circuit(circuit)

    assert not report.ok
    assert any(
        issue.path == f"boundary.{boundary_side}[0].shape" and message in issue.message
        for issue in report.errors
    )


@pytest.mark.parametrize("boundary_side", ["inputs", "outputs"])
def test_circuit_validation_rejects_duplicate_boundary_port_ids(
    boundary_side: str,
) -> None:
    circuit = valid_circuit()
    duplicate = deepcopy(circuit["boundary"][boundary_side][0])  # type: ignore[index]
    circuit["boundary"][boundary_side].append(duplicate)  # type: ignore[index]

    report = validate_circuit(circuit)

    assert not report.ok
    assert any(
        issue.path == f"boundary.{boundary_side}[1].id"
        and "duplicate boundary" in issue.message
        for issue in report.errors
    )


def test_component_contract_validation_checks_every_boundary_port() -> None:
    circuit = valid_circuit()
    component = matching_component()
    circuit["boundary"]["inputs"].append(  # type: ignore[index]
        {
            "id": "sidechain_frame",
            "signal": "frame",
            "shape": [4],
            "component_port": "sidechain",
        }
    )
    component["ports"]["inputs"].append(  # type: ignore[index]
        {"id": "sidechain", "signal": "frame", "shape": [8]}
    )

    report = validate_circuit_against_component(circuit, component)

    assert not report.ok
    assert any(issue.path == "boundary.inputs[1].shape" for issue in report.errors)


def test_component_contract_validation_rejects_missing_and_wrong_port_mappings() -> None:
    circuit = valid_circuit()
    del circuit["boundary"]["inputs"][0]["component_port"]  # type: ignore[index]
    circuit["boundary"]["outputs"][0]["component_port"] = "input"  # type: ignore[index]

    report = validate_circuit_against_component(circuit, matching_component())

    assert not report.ok
    assert {
        issue.path
        for issue in report.errors
        if issue.path and issue.path.endswith(".component_port")
    } == {
        "boundary.inputs[0].component_port",
        "boundary.outputs[0].component_port",
    }
