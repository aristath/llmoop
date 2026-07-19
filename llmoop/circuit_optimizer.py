from __future__ import annotations

from collections import Counter
from copy import deepcopy
from collections.abc import Callable
from typing import Any


Json = dict[str, Any]


def optimize_circuit_for_vulkan(
    circuit: Json,
    *,
    can_fuse_linear_split: Callable[[Json], bool] | None = None,
    can_fuse_parallel_linears: Callable[[list[Json]], bool] | None = None,
) -> Json:
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
        parallel_fusion = _fuse_parallel_linears(
            nodes,
            index,
            can_fuse_parallel_linears,
        )
        if parallel_fusion is not None:
            fused, consumed_node_count = parallel_fusion
            compiled_nodes.append(fused)
            index += consumed_node_count
            continue

        current = nodes[index]
        following = nodes[index + 1] if index + 1 < len(nodes) else None

        fused = _fuse_linear_split(
            current,
            following,
            consumer_counts,
            can_fuse_linear_split,
        )
        if fused is None:
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


def _fuse_parallel_linears(
    nodes: list[Json],
    start: int,
    can_fuse: Callable[[list[Json]], bool] | None,
) -> tuple[Json, int] | None:
    if can_fuse is None:
        return None
    candidates = nodes[start : start + 3]
    for count in range(min(3, len(candidates)), 1, -1):
        group = candidates[:count]
        shared_inputs = group[0].get("inputs", [])
        if (
            len(shared_inputs) != 1
            or any(
                node.get("op") != "linear"
                or node.get("inputs") != shared_inputs
                or len(node.get("outputs", [])) != 1
                or len(node.get("params", [])) != 1
                or node.get("state_reads")
                or node.get("state_writes")
                for node in group
            )
            or not can_fuse(group)
        ):
            continue
        return (
            {
                "id": "__".join(node["id"] for node in group),
                "op": f"parallel_linear_{count}way",
                "inputs": deepcopy(shared_inputs),
                "outputs": [node["outputs"][0] for node in group],
                "params": [node["params"][0] for node in group],
                "attrs": {
                    "compiled_from": [node["id"] for node in group],
                    "branch_count": count,
                },
            },
            count,
        )
    return None


def _fuse_linear_split(
    linear: Json,
    split: Json | None,
    consumer_counts: Counter[str],
    can_fuse: Callable[[Json], bool] | None,
) -> Json | None:
    if (
        split is None
        or can_fuse is None
        or linear.get("op") != "linear"
        or split.get("op") != "split"
        or not can_fuse(linear)
    ):
        return None
    if (
        len(linear.get("inputs", [])) != 1
        or len(linear.get("outputs", [])) != 1
        or len(linear.get("params", [])) != 1
        or linear.get("state_reads")
        or linear.get("state_writes")
    ):
        return None
    linear_output = linear["outputs"][0]
    split_attrs = split.get("attrs", {})
    split_outputs = split.get("outputs", [])
    if (
        split.get("inputs") != [linear_output]
        or consumer_counts[linear_output] != 1
        or len(split_outputs) != 3
        or split_attrs.get("layout") not in {None, "contiguous"}
        or split.get("params")
        or split.get("state_reads")
        or split.get("state_writes")
    ):
        return None
    if split_attrs.get("part_widths") is not None:
        part_widths = [int(width) for width in split_attrs["part_widths"]]
    else:
        part_width = split_attrs.get("part_width")
        if part_width is None:
            return None
        part_widths = [int(part_width)] * 3
    if len(part_widths) != 3 or any(width <= 0 or width % 2 for width in part_widths):
        return None

    attrs = deepcopy(split_attrs)
    attrs["part_widths"] = part_widths
    attrs["compiled_from"] = [linear["id"], split["id"]]
    attrs["intermediate_rounding"] = "BF16"
    return {
        "id": f"{linear['id']}__{split['id']}",
        "op": "linear_split_3way",
        "inputs": deepcopy(linear["inputs"]),
        "outputs": deepcopy(split_outputs),
        "params": deepcopy(linear["params"]),
        "attrs": attrs,
    }


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
