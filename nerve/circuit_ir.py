from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Json = dict[str, Any]


@dataclass(frozen=True)
class CircuitIssue:
    severity: str
    message: str
    path: str | None = None

    def to_json(self) -> Json:
        return {
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class CircuitValidationReport:
    checks: tuple[str, ...]
    issues: tuple[CircuitIssue, ...]

    @property
    def errors(self) -> tuple[CircuitIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[CircuitIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        messages = "\n".join(f"- {issue.path + ': ' if issue.path else ''}{issue.message}" for issue in self.errors)
        raise ValueError(f"circuit validation failed:\n{messages}")

    def to_json(self) -> Json:
        return {
            "ok": self.ok,
            "check_count": len(self.checks),
            "checks": list(self.checks),
            "issues": [issue.to_json() for issue in self.issues],
        }


def load_circuit(path: Path) -> Json:
    return json.loads(path.read_text())


def validate_circuit(circuit: Json) -> CircuitValidationReport:
    checks: list[str] = []
    issues: list[CircuitIssue] = []

    _check(
        circuit.get("schema") == "nerve.stream_circuit.v1",
        checks,
        issues,
        "schema is nerve.stream_circuit.v1",
        f"unsupported circuit schema {circuit.get('schema')!r}",
        "schema",
    )
    _check(isinstance(circuit.get("id"), str) and bool(circuit["id"]), checks, issues, "circuit id exists", "missing circuit id", "id")
    _check(
        circuit.get("runtime_role")
        in {
            "signal_processor",
            "input_transducer",
            "output_transducer",
            "sampler",
            "draft_processor",
            "draft_input_adapter",
            "draft_output_transducer",
        },
        checks,
        issues,
        "runtime role is supported",
        f"unsupported runtime role {circuit.get('runtime_role')!r}",
        "runtime_role",
    )

    boundary = circuit.get("boundary")
    _check(isinstance(boundary, dict), checks, issues, "boundary exists", "boundary must be an object", "boundary")
    if not isinstance(boundary, dict):
        return CircuitValidationReport(checks=tuple(checks), issues=tuple(issues))

    inputs = boundary.get("inputs", [])
    outputs = boundary.get("outputs", [])
    _check(isinstance(inputs, list) and bool(inputs), checks, issues, "boundary has inputs", "boundary.inputs must be a non-empty list", "boundary.inputs")
    _check(isinstance(outputs, list) and bool(outputs), checks, issues, "boundary has outputs", "boundary.outputs must be a non-empty list", "boundary.outputs")

    produced: set[str] = set()
    input_port_ids: set[str] = set()
    for index, port in enumerate(inputs if isinstance(inputs, list) else []):
        path = f"boundary.inputs[{index}]"
        if _port_has_id_signal_shape(port, checks, issues, path):
            if port["id"] in input_port_ids:
                issues.append(CircuitIssue("error", f"duplicate boundary input port id {port['id']!r}", f"{path}.id"))
                continue
            input_port_ids.add(port["id"])
            produced.add(port["id"])

    declared_state = _ids_by_path(circuit.get("state_ports", []), checks, issues, "state_ports")
    declared_params = set()
    parameters = circuit.get("parameters", {})
    if isinstance(parameters, dict) and isinstance(parameters.get("refs"), dict):
        declared_params = set(parameters["refs"])
        checks.append("parameter refs are declared")
    else:
        issues.append(CircuitIssue("error", "parameters.refs must be an object", "parameters.refs"))

    nodes = circuit.get("nodes", [])
    _check(isinstance(nodes, list) and bool(nodes), checks, issues, "circuit has nodes", "nodes must be a non-empty list", "nodes")
    if not isinstance(nodes, list):
        return CircuitValidationReport(checks=tuple(checks), issues=tuple(issues))

    node_ids: set[str] = set()
    produced_by: dict[str, str] = {signal: "boundary.input" for signal in produced}
    for index, node in enumerate(nodes):
        path = f"nodes[{index}]"
        if not isinstance(node, dict):
            issues.append(CircuitIssue("error", "node must be an object", path))
            continue

        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            issues.append(CircuitIssue("error", "node id must be a non-empty string", f"{path}.id"))
            node_id = f"<node:{index}>"
        elif node_id in node_ids:
            issues.append(CircuitIssue("error", f"duplicate node id {node_id!r}", f"{path}.id"))
        else:
            checks.append(f"{node_id} node id is unique")
            node_ids.add(node_id)

        _check(isinstance(node.get("op"), str) and bool(node["op"]), checks, issues, f"{node_id} op exists", "node op must be a non-empty string", f"{path}.op")

        for signal in _string_list(node.get("inputs", []), issues, f"{path}.inputs"):
            if signal not in produced and signal not in declared_state:
                issues.append(CircuitIssue("error", f"input signal {signal!r} has not been produced or declared as state", f"{path}.inputs"))
            else:
                checks.append(f"{node_id} input {signal} resolves")

        for param in _string_list(node.get("params", []), issues, f"{path}.params"):
            if param not in declared_params:
                issues.append(CircuitIssue("error", f"parameter ref {param!r} is not declared", f"{path}.params"))
            else:
                checks.append(f"{node_id} parameter {param} resolves")

        for state in _string_list(node.get("state_reads", []), issues, f"{path}.state_reads"):
            if state not in declared_state:
                issues.append(CircuitIssue("error", f"state read {state!r} is not declared", f"{path}.state_reads"))
            else:
                checks.append(f"{node_id} state read {state} resolves")

        for state in _string_list(node.get("state_writes", []), issues, f"{path}.state_writes"):
            if state not in declared_state:
                issues.append(CircuitIssue("error", f"state write {state!r} is not declared", f"{path}.state_writes"))
            else:
                checks.append(f"{node_id} state write {state} resolves")

        node_outputs = _string_list(node.get("outputs", []), issues, f"{path}.outputs")
        if not node_outputs:
            issues.append(CircuitIssue("error", "node must declare at least one output signal", f"{path}.outputs"))
        for signal in node_outputs:
            if signal in produced_by:
                issues.append(CircuitIssue("error", f"output signal {signal!r} already produced by {produced_by[signal]}", f"{path}.outputs"))
                continue
            produced.add(signal)
            produced_by[signal] = str(node_id)
            checks.append(f"{node_id} output {signal} is unique")

    output_port_ids: set[str] = set()
    for index, port in enumerate(outputs if isinstance(outputs, list) else []):
        path = f"boundary.outputs[{index}]"
        if not _port_has_id_signal_shape(port, checks, issues, path):
            continue
        if port["id"] in output_port_ids:
            issues.append(CircuitIssue("error", f"duplicate boundary output port id {port['id']!r}", f"{path}.id"))
            continue
        output_port_ids.add(port["id"])
        source = port.get("source", port["id"])
        if not isinstance(source, str):
            issues.append(CircuitIssue("error", "boundary output source must be a string", f"{path}.source"))
        elif source not in produced:
            issues.append(CircuitIssue("error", f"boundary output source {source!r} is not produced", f"{path}.source"))
        else:
            checks.append(f"boundary output {port['id']} source resolves")

    _validate_semantic_module_tree(
        circuit.get("semantic_module_tree"),
        circuit.get("semantic_execution_nodes", nodes),
        declared_params,
        declared_state,
        checks,
        issues,
    )

    return CircuitValidationReport(checks=tuple(checks), issues=tuple(issues))


def _validate_semantic_module_tree(
    tree: Any,
    nodes: list[Json],
    declared_params: set[str],
    declared_state: set[str],
    checks: list[str],
    issues: list[CircuitIssue],
) -> None:
    if tree is None:
        return
    if not isinstance(tree, dict):
        issues.append(
            CircuitIssue("error", "semantic module tree must be an object", "semantic_module_tree")
        )
        return
    if tree.get("schema") != "nerve.semantic_module_tree.v1":
        issues.append(
            CircuitIssue(
                "error",
                f"unsupported semantic module tree schema {tree.get('schema')!r}",
                "semantic_module_tree.schema",
            )
        )
    else:
        checks.append("semantic module tree schema is supported")

    modules = tree.get("modules")
    if not isinstance(modules, list) or not modules:
        issues.append(
            CircuitIssue(
                "error",
                "semantic module tree modules must be a non-empty list",
                "semantic_module_tree.modules",
            )
        )
        return

    module_by_id: dict[str, Json] = {}
    for index, module in enumerate(modules):
        path = f"semantic_module_tree.modules[{index}]"
        if not isinstance(module, dict):
            issues.append(CircuitIssue("error", "semantic module must be an object", path))
            continue
        module_id = module.get("id")
        if not isinstance(module_id, str) or not module_id:
            issues.append(
                CircuitIssue("error", "semantic module id must be non-empty", f"{path}.id")
            )
            continue
        if module_id in module_by_id:
            issues.append(
                CircuitIssue(
                    "error", f"duplicate semantic module id {module_id!r}", f"{path}.id"
                )
            )
            continue
        module_by_id[module_id] = module
        if not isinstance(module.get("role"), str) or not module["role"]:
            issues.append(
                CircuitIssue("error", "semantic module role must be non-empty", f"{path}.role")
            )
        if not isinstance(module.get("responsibility"), str) or not module["responsibility"]:
            issues.append(
                CircuitIssue(
                    "error",
                    "semantic module responsibility must be non-empty",
                    f"{path}.responsibility",
                )
            )
        for field, declared in (
            ("parameter_ref_ids", declared_params),
            ("owned_state_port_ids", declared_state),
        ):
            values = _string_list(module.get(field, []), issues, f"{path}.{field}")
            unknown = sorted(set(values) - declared)
            if unknown:
                issues.append(
                    CircuitIssue(
                        "error",
                        f"semantic module references unknown {field}: {unknown}",
                        f"{path}.{field}",
                    )
                )

    root_id = tree.get("root_module_id")
    root = module_by_id.get(root_id) if isinstance(root_id, str) else None
    if root is None:
        issues.append(
            CircuitIssue(
                "error",
                f"semantic module root {root_id!r} does not resolve",
                "semantic_module_tree.root_module_id",
            )
        )
        return
    if root.get("parent_id") is not None:
        issues.append(
            CircuitIssue(
                "error", "semantic module root must not have a parent", "semantic_module_tree.root_module_id"
            )
        )

    visited: set[str] = set()
    active: set[str] = set()

    def visit(module_id: str) -> None:
        if module_id in active:
            issues.append(
                CircuitIssue(
                    "error",
                    f"semantic module tree contains a cycle at {module_id!r}",
                    "semantic_module_tree.modules",
                )
            )
            return
        if module_id in visited:
            return
        visited.add(module_id)
        active.add(module_id)
        module = module_by_id[module_id]
        for child_id in _string_list(
            module.get("child_ids", []),
            issues,
            f"semantic_module_tree.modules[{modules.index(module)}].child_ids",
        ):
            child = module_by_id.get(child_id)
            if child is None:
                issues.append(
                    CircuitIssue(
                        "error",
                        f"semantic module child {child_id!r} does not resolve",
                        "semantic_module_tree.modules",
                    )
                )
                continue
            if child.get("parent_id") != module_id:
                issues.append(
                    CircuitIssue(
                        "error",
                        f"semantic module child {child_id!r} does not point back to {module_id!r}",
                        "semantic_module_tree.modules",
                    )
                )
            visit(child_id)
        active.remove(module_id)

    visit(root_id)
    if visited != set(module_by_id):
        issues.append(
            CircuitIssue(
                "error",
                f"semantic modules are unreachable from root: {sorted(set(module_by_id) - visited)}",
                "semantic_module_tree.modules",
            )
        )

    source_node_ids = {str(node.get("id")) for node in nodes}
    node_owner: dict[str, str] = {}
    state_owner: dict[str, str] = {}
    node_by_id = {str(node.get("id")): node for node in nodes}
    for index, module in enumerate(modules):
        if not isinstance(module, dict) or not isinstance(module.get("id"), str):
            continue
        module_id = module["id"]
        for node_id in _string_list(
            module.get("source_node_ids", []),
            issues,
            f"semantic_module_tree.modules[{index}].source_node_ids",
        ):
            if node_id not in source_node_ids:
                issues.append(
                    CircuitIssue(
                        "error",
                        f"semantic module references unknown source node {node_id!r}",
                        f"semantic_module_tree.modules[{index}].source_node_ids",
                    )
                )
            elif node_id in node_owner:
                issues.append(
                    CircuitIssue(
                        "error",
                        f"source node {node_id!r} belongs to both {node_owner[node_id]!r} and {module_id!r}",
                        f"semantic_module_tree.modules[{index}].source_node_ids",
                    )
                )
            else:
                node_owner[node_id] = module_id
            if node_id in node_by_id:
                missing_params = sorted(
                    set(node_by_id[node_id].get("params", []))
                    - set(module.get("parameter_ref_ids", []))
                )
                if missing_params:
                    issues.append(
                        CircuitIssue(
                            "error",
                            f"semantic module omits source-node parameters {missing_params}",
                            f"semantic_module_tree.modules[{index}].parameter_ref_ids",
                        )
                    )
        for state_id in _string_list(
            module.get("owned_state_port_ids", []),
            issues,
            f"semantic_module_tree.modules[{index}].owned_state_port_ids",
        ):
            if state_id in state_owner:
                issues.append(
                    CircuitIssue(
                        "error",
                        f"state {state_id!r} is owned by both {state_owner[state_id]!r} and {module_id!r}",
                        f"semantic_module_tree.modules[{index}].owned_state_port_ids",
                    )
                )
            else:
                state_owner[state_id] = module_id

    if set(node_owner) != source_node_ids:
        issues.append(
            CircuitIssue(
                "error",
                f"semantic module source-node coverage mismatch: missing={sorted(source_node_ids - set(node_owner))}",
                "semantic_module_tree.modules",
            )
        )
    if set(state_owner) != declared_state:
        issues.append(
            CircuitIssue(
                "error",
                f"semantic module state ownership mismatch: missing={sorted(declared_state - set(state_owner))}",
                "semantic_module_tree.modules",
            )
        )
    if not any(issue.path and issue.path.startswith("semantic_module_tree") for issue in issues):
        checks.append("semantic module tree has exact node and state ownership")


def validate_circuit_against_component(circuit: Json, component: Json) -> CircuitValidationReport:
    base = validate_circuit(circuit)
    checks = list(base.checks)
    issues = list(base.issues)

    boundary = circuit.get("boundary", {})
    circuit_inputs = boundary.get("inputs", []) if isinstance(boundary, dict) else []
    circuit_outputs = boundary.get("outputs", []) if isinstance(boundary, dict) else []
    component_inputs = component.get("ports", {}).get("inputs", [])
    component_outputs = component.get("ports", {}).get("outputs", [])

    _compare_ports(circuit_inputs, component_inputs, checks, issues, "inputs")
    _compare_ports(circuit_outputs, component_outputs, checks, issues, "outputs")

    circuit_state = {port.get("id"): port for port in circuit.get("state_ports", []) if isinstance(port, dict)}
    component_state = {port.get("id"): port for port in component.get("state_ports", []) if isinstance(port, dict)}
    _check(
        set(circuit_state) == set(component_state),
        checks,
        issues,
        "state port ids match component contract",
        f"expected state ports {sorted(component_state)}, found {sorted(circuit_state)}",
        "state_ports",
    )
    for state_id in sorted(set(circuit_state) & set(component_state)):
        c_state = circuit_state[state_id]
        p_state = component_state[state_id]
        _check(c_state.get("type") == p_state.get("type"), checks, issues, f"{state_id} state type matches component", f"expected {p_state.get('type')!r}, found {c_state.get('type')!r}", f"state_ports.{state_id}.type")
        _check(c_state.get("shape") == p_state.get("shape"), checks, issues, f"{state_id} state shape matches component", f"expected {p_state.get('shape')!r}, found {c_state.get('shape')!r}", f"state_ports.{state_id}.shape")
        _check(c_state.get("update") == p_state.get("update"), checks, issues, f"{state_id} state update matches component", f"expected {p_state.get('update')!r}, found {c_state.get('update')!r}", f"state_ports.{state_id}.update")

    circuit_params = circuit.get("parameters", {}).get("refs", {}) if isinstance(circuit.get("parameters"), dict) else {}
    component_params = component.get("parameter_block", {}).get("params", {})
    _check(
        set(circuit_params) == set(component_params),
        checks,
        issues,
        "parameter ids match component contract",
        f"expected parameter refs {sorted(component_params)}, found {sorted(circuit_params)}",
        "parameters.refs",
    )
    for param_id in sorted(set(circuit_params) & set(component_params)):
        c_tensor = circuit_params[param_id].get("tensor") if isinstance(circuit_params[param_id], dict) else None
        p_tensor = component_params[param_id].get("tensor") if isinstance(component_params[param_id], dict) else None
        _check(c_tensor == p_tensor, checks, issues, f"{param_id} tensor ref matches component", f"expected tensor {p_tensor!r}, found {c_tensor!r}", f"parameters.refs.{param_id}.tensor")

    return CircuitValidationReport(checks=tuple(checks), issues=tuple(issues))


def _check(
    condition: bool,
    checks: list[str],
    issues: list[CircuitIssue],
    check_name: str,
    error_message: str,
    path: str | None,
) -> None:
    if condition:
        checks.append(check_name)
        return
    issues.append(CircuitIssue(severity="error", message=error_message, path=path))


def _port_has_id_signal_shape(port: Any, checks: list[str], issues: list[CircuitIssue], path: str) -> bool:
    if not isinstance(port, dict):
        issues.append(CircuitIssue("error", "port must be an object", path))
        return False
    ok = True
    if not isinstance(port.get("id"), str) or not port["id"]:
        issues.append(CircuitIssue("error", "port id must be a non-empty string", f"{path}.id"))
        ok = False
    if not isinstance(port.get("signal"), str) or not port["signal"]:
        issues.append(CircuitIssue("error", "port signal must be a non-empty string", f"{path}.signal"))
        ok = False
    if not _is_shape(port.get("shape")):
        issues.append(CircuitIssue("error", "port shape must be a non-empty list of positive integers", f"{path}.shape"))
        ok = False
    if ok:
        checks.append(f"{path} has id/signal/shape")
    return ok


def _compare_port(
    circuit_port: Json,
    component_port: Json,
    checks: list[str],
    issues: list[CircuitIssue],
    circuit_path: str,
    component_path: str,
) -> None:
    _check(circuit_port.get("component_port") == component_port.get("id"), checks, issues, f"{circuit_path} maps to {component_path}", f"expected component port {component_port.get('id')!r}, found {circuit_port.get('component_port')!r}", f"{circuit_path}.component_port")
    _check(circuit_port.get("signal") == component_port.get("signal"), checks, issues, f"{circuit_path} signal matches {component_path}", f"expected signal {component_port.get('signal')!r}, found {circuit_port.get('signal')!r}", f"{circuit_path}.signal")
    _check(circuit_port.get("shape") == component_port.get("shape"), checks, issues, f"{circuit_path} shape matches {component_path}", f"expected shape {component_port.get('shape')!r}, found {circuit_port.get('shape')!r}", f"{circuit_path}.shape")


def _compare_ports(
    circuit_ports: Any,
    component_ports: Any,
    checks: list[str],
    issues: list[CircuitIssue],
    direction: str,
) -> None:
    if not isinstance(circuit_ports, list) or not isinstance(component_ports, list):
        issues.append(CircuitIssue("error", f"circuit and component {direction} must be lists", f"boundary.{direction}"))
        return
    _check(
        len(circuit_ports) == len(component_ports),
        checks,
        issues,
        f"boundary {direction} count matches component contract",
        f"expected {len(component_ports)} {direction}, found {len(circuit_ports)}",
        f"boundary.{direction}",
    )
    for index, (circuit_port, component_port) in enumerate(zip(circuit_ports, component_ports)):
        if not isinstance(circuit_port, dict) or not isinstance(component_port, dict):
            continue
        _compare_port(
            circuit_port,
            component_port,
            checks,
            issues,
            f"boundary.{direction}[{index}]",
            f"component.ports.{direction}[{index}]",
        )


def _ids_by_path(values: Any, checks: list[str], issues: list[CircuitIssue], path: str) -> set[str]:
    ids: set[str] = set()
    if values is None:
        checks.append(f"{path} absent")
        return ids
    if not isinstance(values, list):
        issues.append(CircuitIssue("error", f"{path} must be a list", path))
        return ids
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        if not isinstance(value, dict):
            issues.append(CircuitIssue("error", "state port must be an object", item_path))
            continue
        state_id = value.get("id")
        if not isinstance(state_id, str) or not state_id:
            issues.append(CircuitIssue("error", "state port id must be a non-empty string", f"{item_path}.id"))
            continue
        if state_id in ids:
            issues.append(CircuitIssue("error", f"duplicate state port {state_id!r}", f"{item_path}.id"))
            continue
        ids.add(state_id)
        checks.append(f"{state_id} state port is declared")
    return ids


def _string_list(value: Any, issues: list[CircuitIssue], path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        issues.append(CircuitIssue("error", "must be a list of strings", path))
        return []
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            issues.append(CircuitIssue("error", "must be a non-empty string", f"{path}[{index}]"))
            continue
        strings.append(item)
    return strings


def _is_shape(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    )
