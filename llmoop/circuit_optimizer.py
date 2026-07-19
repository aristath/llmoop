from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from collections.abc import Callable
from typing import Any


Json = dict[str, Any]


def optimize_circuit_for_vulkan(
    circuit: Json,
    *,
    can_fuse_linear_split: Callable[[Json], bool] | None = None,
    can_fuse_parallel_linears: Callable[[list[Json]], bool] | None = None,
    can_fuse_parallel_linear_silu_multiply: (
        Callable[[Json, Json], bool] | None
    ) = None,
    can_fuse_parallel_head_norm_rope: (
        Callable[[list[tuple[Json, Json]]], bool] | None
    ) = None,
    can_fuse_multiply_rolling_depthwise: (
        Callable[[Json, Json, Json], bool] | None
    ) = None,
    can_fuse_recurrent_output_gate: Callable[[Json, Json], bool] | None = None,
    can_fuse_linear_split_recurrent: Callable[[Json, Json], bool] | None = None,
    can_fuse_append_attention: Callable[[Json, Json], bool] | None = None,
) -> Json:
    """Compile discoverable node regions without changing the pedal boundary."""
    optimized = deepcopy(circuit)
    nodes = _fuse_parallel_head_norm_rope_regions(
        optimized["nodes"], can_fuse_parallel_head_norm_rope
    )
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

    compiled_nodes = _fuse_multiply_rolling_depthwise_regions(
        compiled_nodes,
        can_fuse_multiply_rolling_depthwise,
    )
    compiled_nodes = _fuse_recurrent_output_gate_regions(
        compiled_nodes,
        can_fuse_recurrent_output_gate,
    )
    compiled_nodes = _fuse_parallel_linear_silu_multiply_regions(
        compiled_nodes,
        can_fuse_parallel_linear_silu_multiply,
        {
            output.get("source", output["id"])
            for output in optimized.get("boundary", {}).get("outputs", [])
        },
    )
    compiled_nodes = _fuse_linear_split_recurrent_regions(
        compiled_nodes,
        can_fuse_linear_split_recurrent,
    )
    optimized["nodes"] = _fuse_append_attention_regions(
        compiled_nodes,
        can_fuse_append_attention,
        {
            output.get("source", output["id"])
            for output in optimized.get("boundary", {}).get("outputs", [])
        },
    )
    return optimized


def _fuse_parallel_linear_silu_multiply_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
    protected_signals: set[str],
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        projection = nodes[index]
        activation = nodes[index + 1] if index + 1 < len(nodes) else None
        outputs = projection.get("outputs", [])
        if (
            activation is None
            or projection.get("op") != "parallel_linear_2way"
            or activation.get("op") != "silu_multiply"
            or len(projection.get("inputs", [])) != 1
            or len(outputs) != 2
            or len(projection.get("params", [])) != 2
            or projection.get("state_reads")
            or projection.get("state_writes")
            or activation.get("inputs") != outputs
            or len(activation.get("outputs", [])) != 1
            or activation.get("params")
            or activation.get("state_reads")
            or activation.get("state_writes")
            or any(consumer_counts[output] != 1 for output in outputs)
            or any(output in protected_signals for output in outputs)
            or not can_fuse(projection, activation)
        ):
            compiled.append(deepcopy(projection))
            index += 1
            continue

        projection_sources = projection.get("attrs", {}).get("compiled_from")
        activation_sources = activation.get("attrs", {}).get("compiled_from")
        if not isinstance(projection_sources, list) or not isinstance(
            activation_sources, list
        ):
            compiled.append(deepcopy(projection))
            index += 1
            continue
        compiled.append(
            {
                "id": f"{projection['id']}__{activation['id']}",
                "op": "parallel_linear_silu_multiply",
                "inputs": deepcopy(projection["inputs"]),
                "outputs": deepcopy(activation["outputs"]),
                "params": deepcopy(projection["params"]),
                "attrs": {
                    "compiled_from": [*projection_sources, *activation_sources],
                    "branch_count": 2,
                    "intermediate_rounding": "BF16",
                    "element_count": activation.get("attrs", {}).get(
                        "element_count"
                    ),
                },
            }
        )
        index += 2
    return compiled


def _fuse_append_attention_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
    protected_signals: set[str],
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        append = nodes[index]
        attention = nodes[index + 1] if index + 1 < len(nodes) else None
        append_outputs = append.get("outputs", [])
        if (
            attention is None
            or append.get("op") != "append_state_update"
            or attention.get("op") != "scaled_dot_product_attention"
            or len(append.get("inputs", [])) != 3
            or len(append_outputs) != 2
            or append.get("params")
            or len(append.get("state_reads", [])) != 1
            or append.get("state_reads") != append.get("state_writes")
            or len(attention.get("inputs", [])) != 3
            or attention["inputs"][1:] != append_outputs
            or len(attention.get("outputs", [])) != 1
            or attention.get("state_reads")
            or attention.get("state_writes")
            or any(consumer_counts[output] != 1 for output in append_outputs)
            or any(output in protected_signals for output in append_outputs)
            or not can_fuse(append, attention)
        ):
            compiled.append(deepcopy(append))
            index += 1
            continue

        compiled.append(
            {
                "id": f"{append['id']}__{attention['id']}",
                "op": "append_scaled_dot_product_attention",
                "inputs": [
                    attention["inputs"][0],
                    append["inputs"][0],
                    append["inputs"][1],
                    append["inputs"][2],
                ],
                "outputs": deepcopy(attention["outputs"]),
                "params": deepcopy(attention.get("params", [])),
                "state_reads": deepcopy(append["state_reads"]),
                "state_writes": deepcopy(append["state_writes"]),
                "attrs": {
                    "compiled_from": [append["id"], attention["id"]],
                    "append": deepcopy(append.get("attrs", {})),
                    "attention": deepcopy(attention.get("attrs", {})),
                    "current_kv_source": "direct_bf16_input",
                },
            }
        )
        index += 2
    return compiled


def _fuse_linear_split_recurrent_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        projection = nodes[index]
        recurrent = nodes[index + 1] if index + 1 < len(nodes) else None
        projection_outputs = projection.get("outputs", [])
        recurrent_inputs = recurrent.get("inputs", []) if recurrent is not None else []
        state_reads = recurrent.get("state_reads", []) if recurrent is not None else []
        if (
            recurrent is None
            or projection.get("op") != "linear_split_3way"
            or recurrent.get("op") != "multiply_rolling_depthwise_gate"
            or len(projection.get("inputs", [])) != 1
            or len(projection_outputs) != 3
            or len(projection.get("params", [])) != 1
            or projection.get("state_reads")
            or projection.get("state_writes")
            or len(recurrent_inputs) != 4
            or len(recurrent.get("outputs", [])) != 1
            or len(recurrent.get("params", [])) != 1
            or len(state_reads) != 1
            or recurrent.get("state_writes") != state_reads
            or recurrent_inputs[2] != state_reads[0]
            or set([recurrent_inputs[0], recurrent_inputs[1], recurrent_inputs[3]])
            != set(projection_outputs)
            or any(consumer_counts[output] != 1 for output in projection_outputs)
            or not can_fuse(projection, recurrent)
        ):
            compiled.append(deepcopy(projection))
            index += 1
            continue

        input_gate_indices = [
            projection_outputs.index(recurrent_inputs[0]),
            projection_outputs.index(recurrent_inputs[1]),
        ]
        output_gate_index = projection_outputs.index(recurrent_inputs[3])
        projection_attrs = deepcopy(projection.get("attrs", {}))
        recurrent_attrs = deepcopy(recurrent.get("attrs", {}))
        compiled.append(
            {
                "id": f"{projection['id']}__{recurrent['id']}",
                "op": "linear_split_recurrent_depthwise_gate",
                "inputs": [projection["inputs"][0], state_reads[0]],
                "outputs": deepcopy(recurrent["outputs"]),
                "params": [projection["params"][0], recurrent["params"][0]],
                "state_reads": deepcopy(state_reads),
                "state_writes": deepcopy(state_reads),
                "attrs": {
                    "compiled_from": [
                        *projection_attrs.get("compiled_from", [projection["id"]]),
                        *recurrent_attrs.get("compiled_from", [recurrent["id"]]),
                    ],
                    "projection": projection_attrs,
                    "recurrent": recurrent_attrs,
                    "input_gate_branch_indices": input_gate_indices,
                    "output_gate_branch_index": output_gate_index,
                    "projection_rounding": "BF16",
                },
            }
        )
        index += 2
    return compiled


def _fuse_recurrent_output_gate_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        recurrent = nodes[index]
        gate = nodes[index + 1] if index + 1 < len(nodes) else None
        recurrent_outputs = recurrent.get("outputs", [])
        if (
            gate is None
            or recurrent.get("op") != "multiply_rolling_depthwise"
            or gate.get("op") != "multiply"
            or len(recurrent.get("inputs", [])) != 3
            or len(recurrent_outputs) != 1
            or len(recurrent.get("params", [])) != 1
            or len(recurrent.get("state_reads", [])) != 1
            or recurrent.get("state_reads") != recurrent.get("state_writes")
            or len(gate.get("inputs", [])) != 2
            or gate["inputs"].count(recurrent_outputs[0]) != 1
            or len(gate.get("outputs", [])) != 1
            or gate.get("params")
            or gate.get("state_reads")
            or gate.get("state_writes")
            or consumer_counts[recurrent_outputs[0]] != 1
            or not can_fuse(recurrent, gate)
        ):
            compiled.append(deepcopy(recurrent))
            index += 1
            continue

        output_gate = next(
            signal for signal in gate["inputs"] if signal != recurrent_outputs[0]
        )
        attrs = deepcopy(recurrent.get("attrs", {}))
        attrs["compiled_from"] = [
            *attrs.get("compiled_from", [recurrent["id"]]),
            gate["id"],
        ]
        attrs["output_gate_rounding"] = "BF16"
        compiled.append(
            {
                "id": f"{recurrent['id']}__{gate['id']}",
                "op": "multiply_rolling_depthwise_gate",
                "inputs": [*deepcopy(recurrent["inputs"]), output_gate],
                "outputs": deepcopy(gate["outputs"]),
                "params": deepcopy(recurrent["params"]),
                "state_reads": deepcopy(recurrent["state_reads"]),
                "state_writes": deepcopy(recurrent["state_writes"]),
                "attrs": attrs,
            }
        )
        index += 2
    return compiled


def _fuse_multiply_rolling_depthwise_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        multiply = nodes[index]
        rolling = nodes[index + 1] if index + 1 < len(nodes) else None
        depthwise = nodes[index + 2] if index + 2 < len(nodes) else None
        multiply_outputs = multiply.get("outputs", [])
        rolling_outputs = rolling.get("outputs", []) if rolling is not None else []
        if (
            rolling is None
            or depthwise is None
            or multiply.get("op") != "multiply"
            or rolling.get("op") != "rolling_state_update"
            or depthwise.get("op") != "depthwise_conv1d"
            or len(multiply.get("inputs", [])) != 2
            or len(multiply_outputs) != 1
            or multiply.get("params")
            or multiply.get("state_reads")
            or multiply.get("state_writes")
            or len(rolling.get("inputs", [])) != 2
            or rolling["inputs"].count(multiply_outputs[0]) != 1
            or len(rolling_outputs) != 1
            or rolling.get("params")
            or len(rolling.get("state_reads", [])) != 1
            or len(rolling.get("state_writes", [])) != 1
            or rolling["state_reads"] != rolling["state_writes"]
            or depthwise.get("inputs") != rolling_outputs
            or len(depthwise.get("outputs", [])) != 1
            or len(depthwise.get("params", [])) != 1
            or depthwise.get("state_reads")
            or depthwise.get("state_writes")
            or consumer_counts[multiply_outputs[0]] != 1
            or consumer_counts[rolling_outputs[0]] != 1
            or not can_fuse(multiply, rolling, depthwise)
        ):
            compiled.append(deepcopy(multiply))
            index += 1
            continue

        state_input = next(
            signal for signal in rolling["inputs"] if signal != multiply_outputs[0]
        )
        compiled.append(
            {
                "id": f"{multiply['id']}__{rolling['id']}__{depthwise['id']}",
                "op": "multiply_rolling_depthwise",
                "inputs": [*deepcopy(multiply["inputs"]), state_input],
                "outputs": deepcopy(depthwise["outputs"]),
                "params": deepcopy(depthwise["params"]),
                "state_reads": deepcopy(rolling["state_reads"]),
                "state_writes": deepcopy(rolling["state_writes"]),
                "attrs": {
                    "compiled_from": [
                        multiply["id"],
                        rolling["id"],
                        depthwise["id"],
                    ],
                    "multiply": deepcopy(multiply.get("attrs", {})),
                    "rolling": deepcopy(rolling.get("attrs", {})),
                    "depthwise": deepcopy(depthwise.get("attrs", {})),
                    "intermediate_rounding": "BF16",
                },
            }
        )
        index += 3
    return compiled


def _fuse_parallel_head_norm_rope_regions(
    nodes: list[Json],
    can_fuse: Callable[[list[tuple[Json, Json]]], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes

    consumers: dict[str, list[tuple[int, Json]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for signal in node.get("inputs", []):
            consumers[signal].append((index, node))

    skipped: set[int] = set()
    compiled: list[Json] = []
    for index, node in enumerate(nodes):
        if index in skipped:
            continue
        following_index = index + 1
        if following_index >= len(nodes) or following_index in skipped:
            compiled.append(deepcopy(node))
            continue

        first = _head_norm_rope_branch(nodes, index, consumers)
        second = _head_norm_rope_branch(nodes, following_index, consumers)
        branches = [first, second] if first is not None and second is not None else []
        if (
            len(branches) != 2
            or branches[0][1] == branches[1][1]
            or any(rope_index <= following_index for _, rope_index, _, _ in branches)
            or branches[1][2].get("inputs") == branches[0][2].get("outputs")
            or not can_fuse([(norm, rope) for _, _, norm, rope in branches])
        ):
            compiled.append(deepcopy(node))
            continue

        first_norm = branches[0][2]
        first_rope = branches[0][3]
        second_norm = branches[1][2]
        second_rope = branches[1][3]
        compiled.append(
            {
                "id": "__".join(
                    item["id"]
                    for item in (first_norm, first_rope, second_norm, second_rope)
                ),
                "op": "parallel_head_norm_rope_2way",
                "inputs": [first_norm["inputs"][0], second_norm["inputs"][0]],
                "outputs": [first_rope["outputs"][0], second_rope["outputs"][0]],
                "params": [first_norm["params"][0], second_norm["params"][0]],
                "attrs": {
                    "compiled_from": [
                        item["id"]
                        for item in (first_norm, first_rope, second_norm, second_rope)
                    ],
                    "branches": [
                        {
                            "norm": deepcopy(first_norm.get("attrs", {})),
                            "rope": deepcopy(first_rope.get("attrs", {})),
                        },
                        {
                            "norm": deepcopy(second_norm.get("attrs", {})),
                            "rope": deepcopy(second_rope.get("attrs", {})),
                        },
                    ],
                    "intermediate_rounding": "BF16",
                },
            }
        )
        skipped.update(
            {
                following_index,
                branches[0][1],
                branches[1][1],
            }
        )

    return compiled


def _head_norm_rope_branch(
    nodes: list[Json],
    norm_index: int,
    consumers: dict[str, list[tuple[int, Json]]],
) -> tuple[int, int, Json, Json] | None:
    norm = nodes[norm_index]
    if (
        norm.get("op") != "rms_norm_per_head"
        or len(norm.get("inputs", [])) != 1
        or len(norm.get("outputs", [])) != 1
        or len(norm.get("params", [])) != 1
        or norm.get("state_reads")
        or norm.get("state_writes")
    ):
        return None
    norm_output = norm["outputs"][0]
    output_consumers = consumers.get(norm_output, [])
    if len(output_consumers) != 1:
        return None
    rope_index, rope = output_consumers[0]
    if (
        rope_index <= norm_index
        or rope.get("op") != "rotary_position_embedding"
        or rope.get("inputs") != [norm_output]
        or len(rope.get("outputs", [])) != 1
        or rope.get("params")
        or rope.get("state_reads")
        or rope.get("state_writes")
    ):
        return None
    return norm_index, rope_index, norm, rope


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
    element_count = activation.get("attrs", {}).get("element_count")
    if not isinstance(element_count, int) or element_count <= 0:
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
            "element_count": element_count,
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
