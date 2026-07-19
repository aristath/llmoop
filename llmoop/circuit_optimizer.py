from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


Json = dict[str, Any]


def optimize_circuit_for_vulkan(circuit: Json) -> Json:
    """Compile discoverable node regions without changing the pedal boundary."""
    optimized = deepcopy(circuit)
    nodes = optimized["nodes"]
    consumer_counts = Counter(
        signal
        for node in nodes
        for signal in node.get("inputs", [])
    )

    compiled_nodes: list[Json] = []
    index = 0
    while index < len(nodes):
        current = nodes[index]
        following = nodes[index + 1] if index + 1 < len(nodes) else None

        fused = _fuse_silu_multiply(current, following, consumer_counts)
        if fused is None:
            fused = _fuse_linear_residual(current, following, consumer_counts)
        if fused is not None:
            compiled_nodes.append(fused)
            index += 2
            continue

        compiled_nodes.append(deepcopy(current))
        index += 1

    optimized["nodes"] = compiled_nodes
    return optimized


def _fuse_silu_multiply(
    activation: Json,
    multiply: Json | None,
    consumer_counts: Counter[str],
) -> Json | None:
    if multiply is None or activation.get("op") != "silu" or multiply.get("op") != "multiply":
        return None
    if not _plain_single_input_output_node(activation):
        return None
    activation_output = activation["outputs"][0]
    multiply_inputs = multiply.get("inputs", [])
    if (
        len(multiply_inputs) != 2
        or multiply_inputs.count(activation_output) != 1
        or consumer_counts[activation_output] != 1
        or multiply.get("params")
        or multiply.get("state_reads")
        or multiply.get("state_writes")
    ):
        return None

    other_input = next(signal for signal in multiply_inputs if signal != activation_output)
    return {
        "id": f"{activation['id']}__{multiply['id']}",
        "op": "silu_multiply",
        "inputs": [activation["inputs"][0], other_input],
        "outputs": deepcopy(multiply.get("outputs", [])),
        "attrs": {
            "compiled_from": [activation["id"], multiply["id"]],
            "intermediate_rounding": "BF16",
        },
    }


def _fuse_linear_residual(
    linear: Json,
    residual: Json | None,
    consumer_counts: Counter[str],
) -> Json | None:
    if residual is None or linear.get("op") != "linear" or residual.get("op") != "residual_add":
        return None
    if (
        len(linear.get("inputs", [])) != 1
        or len(linear.get("outputs", [])) != 1
        or not _linear_params_are_fusible(linear.get("params", []))
        or linear.get("state_reads")
        or linear.get("state_writes")
    ):
        return None
    linear_output = linear["outputs"][0]
    residual_inputs = residual.get("inputs", [])
    if (
        len(residual_inputs) != 2
        or residual_inputs.count(linear_output) != 1
        or consumer_counts[linear_output] != 1
        or residual.get("params")
        or residual.get("state_reads")
        or residual.get("state_writes")
    ):
        return None

    residual_input = next(signal for signal in residual_inputs if signal != linear_output)
    return {
        "id": f"{linear['id']}__{residual['id']}",
        "op": "linear_residual",
        "inputs": [linear["inputs"][0], residual_input],
        "outputs": deepcopy(residual.get("outputs", [])),
        "params": deepcopy(linear["params"]),
        "attrs": {
            "compiled_from": [linear["id"], residual["id"]],
            "intermediate_rounding": "BF16",
        },
    }


def _plain_single_input_output_node(node: Json) -> bool:
    return (
        len(node.get("inputs", [])) == 1
        and len(node.get("outputs", [])) == 1
        and not node.get("params")
        and not node.get("state_reads")
        and not node.get("state_writes")
    )


def _linear_params_are_fusible(parameters: list[str]) -> bool:
    return len(parameters) == 1 or (
        len(parameters) == 2 and parameters[1] == f"{parameters[0]}_scale_inv"
    )
