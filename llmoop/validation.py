from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.pedalboard import Json, Pedalboard


@dataclass(frozen=True)
class ValidationIssue:
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
class ValidationReport:
    checks: tuple[str, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            messages = "\n".join(
                f"- {issue.path + ': ' if issue.path else ''}{issue.message}" for issue in self.errors
            )
            raise ValueError(f"pedalboard validation failed:\n{messages}")

    def to_json(self) -> Json:
        return {
            "ok": self.ok,
            "check_count": len(self.checks),
            "checks": list(self.checks),
            "issues": [issue.to_json() for issue in self.issues],
        }


def validate_pedalboard(pedalboard: Pedalboard) -> ValidationReport:
    checks: list[str] = []
    issues: list[ValidationIssue] = []

    model = pedalboard.model_graph
    dimensions = model["dimensions"]
    expected_layers = dimensions["num_hidden_layers"]
    hidden_size = dimensions["hidden_size"]
    conv_l_cache = dimensions["conv_l_cache"]
    head_width = hidden_size // dimensions["num_attention_heads"]
    attention_elements_per_token = 2 * dimensions["num_key_value_heads"] * head_width

    _check(
        len(pedalboard.pedals) == expected_layers,
        checks,
        issues,
        "model layer count matches pedal count",
        f"expected {expected_layers} pedals, found {len(pedalboard.pedals)}",
        "model.json",
    )

    pedal_ids = [pedal.id for pedal in pedalboard.pedals]
    _check(
        len(pedal_ids) == len(set(pedal_ids)),
        checks,
        issues,
        "pedal ids are unique",
        "duplicate pedal ids found",
        "model.json",
    )

    _check(
        "input_transducer" in model["graph"],
        checks,
        issues,
        "input transducer exists",
        "missing graph.input_transducer",
        "model.json",
    )
    _check(
        "output_transducer" in model["graph"],
        checks,
        issues,
        "output transducer exists",
        "missing graph.output_transducer",
        "model.json",
    )

    for index, pedal in enumerate(pedalboard.pedals):
        prefix = f"{pedal.source_file.name}"
        _check(
            pedal.id == f"layer_{index:02d}",
            checks,
            issues,
            f"{pedal.id} id matches serial index",
            f"expected layer_{index:02d}, found {pedal.id}",
            prefix,
        )
        _check(
            pedal.input_port.signal == "frame" and pedal.output_port.signal == "frame",
            checks,
            issues,
            f"{pedal.id} uses frame signal ports",
            "pedal input/output signal must be frame",
            prefix,
        )
        _check(
            pedal.input_port.shape == (hidden_size,) and pedal.output_port.shape == (hidden_size,),
            checks,
            issues,
            f"{pedal.id} frame width matches hidden size",
            f"pedal frame shape must be [{hidden_size}]",
            prefix,
        )
        _check(
            bool(pedal.parameter_block.tensor_refs),
            checks,
            issues,
            f"{pedal.id} has parameter tensor refs",
            "pedal has no parameter tensor refs",
            prefix,
        )

        if pedal.operator_type == "conv":
            _check(
                len(pedal.state_ports) == 1 and pedal.state_ports[0].type == "rolling_frame_memory",
                checks,
                issues,
                f"{pedal.id} declares one rolling temporal state",
                "conv pedal must declare one rolling_frame_memory state port",
                prefix,
            )
            if pedal.state_ports:
                _check(
                    pedal.state_ports[0].static_shape() == (conv_l_cache, hidden_size),
                    checks,
                    issues,
                    f"{pedal.id} temporal state shape matches conv cache",
                    f"expected temporal state shape [{conv_l_cache}, {hidden_size}]",
                    prefix,
                )
        elif pedal.operator_type == "full_attention":
            _check(
                len(pedal.state_ports) == 1 and pedal.state_ports[0].type == "append_only_attention_memory",
                checks,
                issues,
                f"{pedal.id} declares one append-only attention state",
                "attention pedal must declare one append_only_attention_memory state port",
                prefix,
            )
            if pedal.state_ports:
                _check(
                    pedal.state_ports[0].elements_per_token() == attention_elements_per_token,
                    checks,
                    issues,
                    f"{pedal.id} KV transient elements per token match dimensions",
                    f"expected {attention_elements_per_token} KV elements per token",
                    prefix,
                )
        else:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"unsupported operator type {pedal.operator_type!r}",
                    path=prefix,
                )
            )

    tensor_index_path = pedalboard.root / pedalboard.model_graph["files"]["tensor_index"]
    if tensor_index_path.exists():
        tensor_index = json.loads(tensor_index_path.read_text())
        tensors = set(tensor_index["tensors"])
        tensor_refs = set(_collect_tensor_refs(pedalboard.model_graph))
        for pedal in pedalboard.pedals:
            tensor_refs.update(_collect_tensor_refs(json.loads(pedal.source_file.read_text())))
        missing = sorted(tensor_refs - tensors)
        _check(
            not missing,
            checks,
            issues,
            "all graph tensor refs resolve in tensor index",
            f"missing tensor refs: {missing}",
            str(tensor_index_path.relative_to(pedalboard.root)),
        )
    else:
        issues.append(
            ValidationIssue(
                severity="error",
                message="missing tensor index",
                path=str(tensor_index_path),
            )
        )

    return ValidationReport(checks=tuple(checks), issues=tuple(issues))


def _check(
    condition: bool,
    checks: list[str],
    issues: list[ValidationIssue],
    check_name: str,
    error_message: str,
    path: str | None,
) -> None:
    if condition:
        checks.append(check_name)
        return
    issues.append(ValidationIssue(severity="error", message=error_message, path=path))


def _collect_tensor_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        tensor = value.get("tensor")
        if isinstance(tensor, str):
            refs.add(tensor)
        for child in value.values():
            refs.update(_collect_tensor_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_collect_tensor_refs(child))
    return refs
