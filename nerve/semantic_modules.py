from __future__ import annotations

from collections.abc import Iterable
from typing import Any


Json = dict[str, Any]
SEMANTIC_MODULE_TREE_SCHEMA = "nerve.semantic_module_tree.v1"


def build_layer_semantic_module_tree(component: Json, nodes: list[Json]) -> Json:
    """Build the exact semantic anatomy while the source layer is lowered."""
    operator_type = str(component["operator_type"])
    definitions = [
        _module("layer", "layer", "Editable source-layer component"),
        _module(
            "layer.token_mixer",
            "token_mixer",
            f"{operator_type.replace('_', ' ')} temporal/token mixing block",
            parent="layer",
        ),
        _module(
            "layer.feature_transform",
            "feature_transform",
            f"{component['feed_forward']['type'].replace('_', ' ')} feature transform",
            parent="layer",
        ),
    ]
    definitions.extend(_token_mixer_modules(operator_type))
    definitions.extend(_feature_transform_modules(component))
    return _materialize_tree(component, nodes, definitions)


def semantic_module_ids_for_source_nodes(
    circuit: Json, source_node_ids: Iterable[str]
) -> list[str]:
    tree = circuit.get("semantic_module_tree")
    if not isinstance(tree, dict):
        return []
    wanted = set(source_node_ids)
    return [
        module["id"]
        for module in tree.get("modules", [])
        if wanted.intersection(module.get("source_node_ids", []))
    ]


def normalized_source_node_ids(node: Json) -> list[str]:
    semantic_source_node_ids = node.get("attrs", {}).get("semantic_source_node_ids")
    if isinstance(semantic_source_node_ids, list) and semantic_source_node_ids:
        return list(
            dict.fromkeys(str(node_id) for node_id in semantic_source_node_ids)
        )
    compiled_from = node.get("attrs", {}).get("compiled_from")
    if isinstance(compiled_from, list) and compiled_from:
        return list(dict.fromkeys(str(node_id) for node_id in compiled_from))
    return [str(node["id"])]


def _token_mixer_modules(operator_type: str) -> list[Json]:
    common = [
        _module(
            "layer.token_mixer.normalization",
            "normalization",
            "Normalize the layer input for the token mixer",
            parent="layer.token_mixer",
            nodes=["operator_norm"],
        ),
        _module(
            "layer.token_mixer.output_projection",
            "projection",
            "Project the mixed representation back to the layer width",
            parent="layer.token_mixer",
            nodes={
                "conv": ["conv_out_projection"],
                "full_attention": ["attention_out_projection"],
                "gated_delta": ["delta_out_projection"],
                "rg_lru": ["rg_lru_out_projection"],
            }[operator_type],
        ),
        _module(
            "layer.token_mixer.post_normalization",
            "normalization",
            "Normalize the token-mixer update before its residual path",
            parent="layer.token_mixer",
            nodes=["operator_post_norm"],
            optional=True,
        ),
        _module(
            "layer.token_mixer.residual",
            "residual_path",
            "Merge the token-mixer update with the incoming frame",
            parent="layer.token_mixer",
            nodes=["operator_residual"],
        ),
    ]
    if operator_type == "conv":
        return [
            *common,
            _module(
                "layer.token_mixer.input_projection",
                "projection",
                "Project and split input, input-gate, and output-gate channels",
                parent="layer.token_mixer",
                nodes=["conv_in_projection", "split_b_c_x"],
            ),
            _module(
                "layer.token_mixer.gates",
                "gate",
                "Apply input and output gates around temporal convolution",
                parent="layer.token_mixer",
                nodes=["input_gate", "output_gate"],
            ),
            _module(
                "layer.token_mixer.temporal_state",
                "state_attachment",
                "Own and update the rolling local temporal memory",
                parent="layer.token_mixer",
                nodes=["temporal_memory_update"],
                state=["temporal_memory"],
            ),
            _module(
                "layer.token_mixer.temporal_convolution",
                "temporal_mixer",
                "Apply the learned depthwise temporal filter",
                parent="layer.token_mixer",
                nodes=["depthwise_temporal_conv"],
            ),
        ]
    if operator_type == "full_attention":
        return [
            *common,
            _module(
                "layer.token_mixer.input_projections",
                "projection",
                "Produce query, key, value, and optional gate channels",
                parent="layer.token_mixer",
                nodes=[
                    "qkv_projection",
                    "qkv_split",
                    "q_projection",
                    "k_projection",
                    "v_projection",
                ],
                optional=True,
            ),
            _module(
                "layer.token_mixer.head_normalizations",
                "normalization",
                "Normalize query, key, and value heads",
                parent="layer.token_mixer",
                nodes=["q_head_norm", "k_head_norm", "v_head_norm"],
                optional=True,
            ),
            _module(
                "layer.token_mixer.position",
                "position_operation",
                "Apply rotary position transformations",
                parent="layer.token_mixer",
                nodes=["q_rope", "k_rope"],
                optional=True,
            ),
            _module(
                "layer.token_mixer.attention_state",
                "state_attachment",
                "Own, append, and read the stream-specific attention memory",
                parent="layer.token_mixer",
                nodes=["kv_memory_append", "attention_read"],
                state=["kv_memory"],
            ),
            _module(
                "layer.token_mixer.gates",
                "gate",
                "Construct and apply the optional attention output gate",
                parent="layer.token_mixer",
                nodes=[
                    "q_gate_split",
                    "attention_gate_projection",
                    "attention_output_gate",
                ],
                optional=True,
            ),
        ]
    if operator_type == "gated_delta":
        return [
            *common,
            _module(
                "layer.token_mixer.input_projections",
                "projection",
                "Produce recurrent query, key, value, gate, beta, and decay channels",
                parent="layer.token_mixer",
                nodes=[
                    "delta_qkv_projection",
                    "delta_z_projection",
                    "delta_b_projection",
                    "delta_a_projection",
                ],
            ),
            _module(
                "layer.token_mixer.convolution_state",
                "state_attachment",
                "Own and update the local gated-delta convolution history",
                parent="layer.token_mixer",
                nodes=["delta_causal_conv"],
                state=["conv_state"],
            ),
            _module(
                "layer.token_mixer.recurrent_state",
                "state_attachment",
                "Own and update the gated-delta recurrent matrix",
                parent="layer.token_mixer",
                nodes=["gated_delta_update"],
                state=["recurrent_state"],
            ),
        ]
    if operator_type == "rg_lru":
        return [
            *common,
            _module(
                "layer.token_mixer.input_projections",
                "projection",
                "Produce recurrent input and output-gate channels",
                parent="layer.token_mixer",
                nodes=["rg_lru_y_projection", "rg_lru_x_projection"],
            ),
            _module(
                "layer.token_mixer.gates",
                "gate",
                "Activate and apply the recurrent output gate",
                parent="layer.token_mixer",
                nodes=["rg_lru_y_activation", "rg_lru_output_gate"],
            ),
            _module(
                "layer.token_mixer.recurrent_state",
                "state_attachment",
                "Own and update convolutional and diagonal recurrent memory",
                parent="layer.token_mixer",
                nodes=["rg_lru_step"],
                state=["conv_state", "recurrent_state"],
            ),
        ]
    raise ValueError(f"unsupported semantic token-mixer family {operator_type!r}")


def _feature_transform_modules(component: Json) -> list[Json]:
    modules = [
        _module(
            "layer.feature_transform.normalization",
            "normalization",
            "Normalize the token-mixer residual for feature transformation",
            parent="layer.feature_transform",
            nodes=["ffn_norm"],
        ),
        _module(
            "layer.feature_transform.post_normalization",
            "normalization",
            "Normalize the feature-transform update before its residual path",
            parent="layer.feature_transform",
            nodes=["ffn_post_norm"],
            optional=True,
        ),
        _module(
            "layer.feature_transform.residual",
            "residual_path",
            "Merge the feature-transform update with the token-mixer residual",
            parent="layer.feature_transform",
            nodes=["ffn_residual"],
        ),
        _module(
            "layer.feature_transform.per_layer_input",
            "auxiliary_input",
            "Build and merge the optional per-layer contextual input",
            parent="layer.feature_transform",
            nodes=[
                "per_layer_embedding",
                "per_layer_input_gate",
                "per_layer_gate_activation",
                "per_layer_gate_multiply",
                "per_layer_projection",
                "per_layer_post_norm",
                "per_layer_residual",
                "layer_scale",
            ],
            optional=True,
        ),
    ]
    feed_forward = component["feed_forward"]
    if feed_forward["type"] != "sparse_moe":
        modules.extend(
            [
                _module(
                    "layer.feature_transform.projections",
                    "projection",
                    "Project into and out of the dense gated feature space",
                    parent="layer.feature_transform",
                    nodes=[
                        "ffn_gate_up_projection",
                        "ffn_gate_up_split",
                        "ffn_gate_projection",
                        "ffn_up_projection",
                        "ffn_down_projection",
                    ],
                    optional=True,
                ),
                _module(
                    "layer.feature_transform.gate",
                    "gate",
                    "Activate and combine the dense feature branches",
                    parent="layer.feature_transform",
                    nodes=["ffn_gate_activation", "ffn_gate_multiply"],
                ),
            ]
        )
        return modules

    modules.extend(
        [
            _module(
                "layer.feature_transform.routing",
                "expert_routing",
                "Select and weight sparse experts",
                parent="layer.feature_transform",
                nodes=["moe_router_projection", "moe_topk"],
            ),
            _module(
                "layer.feature_transform.expert_bank",
                "selected_expert_bank",
                "Execute only the routed experts from the packed expert bank",
                parent="layer.feature_transform",
                nodes=["sparse_moe_gate_up", "sparse_moe_down"],
            ),
            _module(
                "layer.feature_transform.reduction",
                "expert_reduction",
                "Reduce routed expert results and combine optional shared output",
                parent="layer.feature_transform",
                nodes=["moe_reduce", "shared_and_sparse_expert_add"],
                optional=True,
            ),
            _module(
                "layer.feature_transform.shared_expert",
                "shared_expert",
                "Execute and optionally gate the shared expert",
                parent="layer.feature_transform",
                nodes=[
                    "shared_mlp_input_projection",
                    "shared_mlp_split",
                    "shared_mlp_activation",
                    "shared_mlp_output_projection",
                    "shared_expert_gate_projection",
                    "shared_expert_gate",
                ],
                optional=True,
            ),
        ]
    )
    for expert_index in range(int(feed_forward["num_experts"])):
        modules.append(
            _module(
                f"layer.feature_transform.expert_bank.expert_{expert_index:03d}",
                "expert",
                f"Packed sparse expert {expert_index}",
                parent="layer.feature_transform.expert_bank",
                params=["moe_input", "moe_output"],
                virtual=True,
                attrs={
                    "expert_index": expert_index,
                    "parameter_slices": [
                        {"parameter_ref_id": "moe_input", "axis": 0, "index": expert_index},
                        {"parameter_ref_id": "moe_output", "axis": 0, "index": expert_index},
                    ],
                },
            )
        )
    return modules


def _module(
    module_id: str,
    role: str,
    responsibility: str,
    *,
    parent: str | None = None,
    nodes: list[str] | None = None,
    params: list[str] | None = None,
    state: list[str] | None = None,
    optional: bool = False,
    virtual: bool = False,
    attrs: Json | None = None,
) -> Json:
    return {
        "id": module_id,
        "role": role,
        "responsibility": responsibility,
        "parent_id": parent,
        "source_node_ids": nodes or [],
        "parameter_ref_ids": params or [],
        "owned_state_port_ids": state or [],
        "optional": optional,
        "virtual": virtual,
        "attrs": attrs or {},
    }


def _materialize_tree(component: Json, nodes: list[Json], definitions: list[Json]) -> Json:
    node_by_id = {str(node["id"]): node for node in nodes}
    node_order = {node_id: index for index, node_id in enumerate(node_by_id)}
    parameters = set(component["parameter_block"]["params"])
    state_ports = {str(state["id"]) for state in component.get("state_ports", [])}

    materialized = []
    for definition in definitions:
        declared = definition.pop("source_node_ids")
        optional = definition.pop("optional")
        present = [node_id for node_id in declared if node_id in node_by_id]
        if not present and optional:
            continue
        missing = [node_id for node_id in declared if node_id not in node_by_id]
        if missing and not definition["virtual"] and not optional:
            raise ValueError(
                f"semantic module {definition['id']!r} references missing source nodes {missing}"
            )
        direct_params = list(definition["parameter_ref_ids"])
        for node_id in present:
            for parameter in node_by_id[node_id].get("params", []):
                if parameter not in direct_params:
                    direct_params.append(parameter)
        unknown_params = [parameter for parameter in direct_params if parameter not in parameters]
        if unknown_params:
            raise ValueError(
                f"semantic module {definition['id']!r} references unknown parameters "
                f"{unknown_params}"
            )
        unknown_state = [
            state
            for state in definition["owned_state_port_ids"]
            if state not in state_ports
        ]
        if unknown_state:
            raise ValueError(
                f"semantic module {definition['id']!r} owns unknown state {unknown_state}"
            )
        materialized.append(
            {
                **definition,
                "source_node_ids": sorted(present, key=node_order.__getitem__),
                "parameter_ref_ids": direct_params,
                "child_ids": [],
                "input_signals": [],
                "output_signals": [],
            }
        )

    by_id = {module["id"]: module for module in materialized}
    for module in materialized:
        parent_id = module["parent_id"]
        if parent_id is not None:
            if parent_id not in by_id:
                raise ValueError(
                    f"semantic module {module['id']!r} has missing parent {parent_id!r}"
                )
            by_id[parent_id]["child_ids"].append(module["id"])

    module_order = {module["id"]: index for index, module in enumerate(materialized)}

    def subtree_order(module_id: str) -> tuple[int, int]:
        subtree = _subtree_node_ids(module_id, by_id)
        earliest_node = min(
            (node_order[node_id] for node_id in subtree),
            default=len(node_order),
        )
        return earliest_node, module_order[module_id]

    for module in materialized:
        module["child_ids"].sort(key=subtree_order)

    for module in materialized:
        subtree_nodes = _subtree_node_ids(module["id"], by_id)
        module["input_signals"], module["output_signals"] = _module_boundary_signals(
            subtree_nodes, nodes, state_ports
        )

    tree = {
        "schema": SEMANTIC_MODULE_TREE_SCHEMA,
        "root_module_id": "layer",
        "modules": materialized,
    }
    _validate_materialized_tree(tree, nodes, parameters, state_ports)
    return tree


def _subtree_node_ids(module_id: str, by_id: dict[str, Json]) -> set[str]:
    module = by_id[module_id]
    result = set(module["source_node_ids"])
    for child_id in module["child_ids"]:
        result.update(_subtree_node_ids(child_id, by_id))
    return result


def _module_boundary_signals(
    subtree_node_ids: set[str], nodes: list[Json], state_ports: set[str]
) -> tuple[list[str], list[str]]:
    produced_inside = {
        output
        for node in nodes
        if node["id"] in subtree_node_ids
        for output in node.get("outputs", [])
    }
    consumed_outside = {
        signal
        for node in nodes
        if node["id"] not in subtree_node_ids
        for signal in node.get("inputs", [])
    }
    inputs = []
    outputs = []
    for node in nodes:
        if node["id"] not in subtree_node_ids:
            continue
        for signal in node.get("inputs", []):
            if (
                signal not in produced_inside
                and signal not in state_ports
                and signal not in inputs
            ):
                inputs.append(signal)
        for signal in node.get("outputs", []):
            if (
                (signal in consumed_outside or signal == "output_frame")
                and signal not in outputs
            ):
                outputs.append(signal)
    return inputs, outputs


def _validate_materialized_tree(
    tree: Json, nodes: list[Json], parameters: set[str], state_ports: set[str]
) -> None:
    modules = tree["modules"]
    by_id = {module["id"]: module for module in modules}
    if len(by_id) != len(modules):
        raise ValueError("semantic module ids must be unique")
    root_id = tree["root_module_id"]
    if root_id not in by_id or by_id[root_id]["parent_id"] is not None:
        raise ValueError("semantic module tree root is missing or has a parent")

    visited: set[str] = set()

    def visit(module_id: str, ancestry: set[str]) -> None:
        if module_id in ancestry:
            raise ValueError(f"semantic module tree contains a cycle at {module_id!r}")
        if module_id in visited:
            return
        visited.add(module_id)
        for child_id in by_id[module_id]["child_ids"]:
            child = by_id.get(child_id)
            if child is None or child["parent_id"] != module_id:
                raise ValueError(
                    f"semantic module {module_id!r} has invalid child {child_id!r}"
                )
            visit(child_id, ancestry | {module_id})

    visit(root_id, set())
    if visited != set(by_id):
        raise ValueError("semantic module tree contains modules unreachable from the root")

    owners: dict[str, str] = {}
    for module in modules:
        for node_id in module["source_node_ids"]:
            if node_id in owners:
                raise ValueError(
                    f"source node {node_id!r} belongs to both {owners[node_id]!r} "
                    f"and {module['id']!r}"
                )
            owners[node_id] = module["id"]
        if not set(module["parameter_ref_ids"]).issubset(parameters):
            raise ValueError(f"semantic module {module['id']!r} has invalid parameters")
        if not set(module["owned_state_port_ids"]).issubset(state_ports):
            raise ValueError(f"semantic module {module['id']!r} has invalid state")
    node_ids = {str(node["id"]) for node in nodes}
    if set(owners) != node_ids:
        missing = sorted(node_ids - set(owners))
        extra = sorted(set(owners) - node_ids)
        raise ValueError(
            f"semantic module source-node coverage is incomplete: missing={missing}, extra={extra}"
        )

    state_owners = [
        state
        for module in modules
        for state in module["owned_state_port_ids"]
    ]
    if len(state_owners) != len(set(state_owners)) or set(state_owners) != state_ports:
        raise ValueError("every layer state port must have exactly one semantic owner")
