#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.source_oracle import _oracle_imports


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the direct circuit model runtime against the source model oracle.")
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-ids", type=str, default=None, help="comma-separated token ids for full-sequence mode")
    parser.add_argument("--stream-input-ids", type=str, default=None, help="comma-separated teacher-forced token ids")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    input_ids = _parse_ids(args.stream_input_ids or args.input_ids or "1,2,3,4")
    torch, auto_model, dynamic_cache = _oracle_imports()
    runtime = CircuitModelRuntime.from_dirs(circuit_dir=args.circuit_dir, torch=torch)
    source = auto_model.from_pretrained(args.model_dir, dtype=torch.float32)
    source.eval()

    with torch.no_grad():
        if args.stream_input_ids:
            report = _check_stream(torch, runtime, source, dynamic_cache, input_ids)
        else:
            report = _check_full(torch, runtime, source, input_ids)

    if args.summary:
        report = _summary(report)
    print(json.dumps(report, indent=2))

    comparisons = _collect_comparisons(report)
    if not all(comparison["allclose"] for comparison in comparisons):
        raise SystemExit("circuit model runtime diverged from source oracle")


def _check_full(torch: Any, runtime: CircuitModelRuntime, source: Any, input_ids: tuple[int, ...]) -> dict[str, Any]:
    input_tensor = torch.tensor([list(input_ids)], dtype=torch.long)
    candidate = runtime.forward_input_ids(input_ids)
    source_model = source.model(input_ids=input_tensor, use_cache=True).last_hidden_state
    source_logits = source(input_ids=input_tensor, use_cache=True).logits

    return {
        "mode": "full_sequence",
        "input_ids": list(input_ids),
        "runtime": {
            "tensor_store": runtime.tensor_store.summary(),
            "implementation_count": len(candidate.steps),
            "implementations": {step.pedal_id: step.implementation for step in candidate.steps},
            "last_attention_state": next(
                step.state for step in candidate.steps if step.operator_type == "full_attention"
            ),
        },
        "hidden_comparison": _compare_tensors(
            torch,
            candidate.hidden_states,
            source_model,
            reference="source_model_hidden",
            candidate="circuit_model_hidden",
            atol=1e-5,
            rtol=1e-5,
        ),
        "logits_comparison": _compare_tensors(
            torch,
            candidate.logits,
            source_logits,
            reference="source_model_logits",
            candidate="circuit_model_logits",
            atol=1e-4,
            rtol=1e-4,
        ),
        "candidate_shapes": {
            "hidden": list(candidate.hidden_states.shape),
            "logits": list(candidate.logits.shape),
        },
    }


def _check_stream(
    torch: Any,
    runtime: CircuitModelRuntime,
    source: Any,
    dynamic_cache: Any,
    input_ids: tuple[int, ...],
) -> dict[str, Any]:
    stream = runtime.open_stream()
    source_cache = dynamic_cache(config=source.config)
    ticks = []

    for token_id in input_ids:
        tick = stream.tick(token_id)
        source_out = source(
            input_ids=torch.tensor([[int(token_id)]], dtype=torch.long),
            past_key_values=source_cache,
            use_cache=True,
        )
        ticks.append(
            {
                "tick": tick.tick,
                "token_id": tick.token_id,
                "implementations": {step.pedal_id: step.implementation for step in tick.output.steps},
                "last_attention_state": next(
                    step.state for step in tick.output.steps if step.operator_type == "full_attention"
                ),
                "logits_comparison": _compare_tensors(
                    torch,
                    tick.output.logits,
                    source_out.logits,
                    reference="source_incremental_logits",
                    candidate="circuit_incremental_logits",
                    atol=1e-4,
                    rtol=1e-4,
                ),
            }
        )

    source_full_hidden = source.model(
        input_ids=torch.tensor([list(input_ids)], dtype=torch.long),
        use_cache=True,
    ).last_hidden_state
    source_full_logits = source(
        input_ids=torch.tensor([list(input_ids)], dtype=torch.long),
        use_cache=True,
    ).logits

    return {
        "mode": "stream",
        "input_ids": list(input_ids),
        "tick_count": len(ticks),
        "runtime": {
            "tensor_store": runtime.tensor_store.summary(),
            "implementation_count": len(stream.ticks[-1].output.steps),
            "implementations": {step.pedal_id: step.implementation for step in stream.ticks[-1].output.steps},
            "last_attention_state": next(
                step.state for step in stream.ticks[-1].output.steps if step.operator_type == "full_attention"
            ),
        },
        "ticks": ticks,
        "hidden_comparison": _compare_tensors(
            torch,
            stream.hidden_states,
            source_full_hidden,
            reference="source_full_hidden",
            candidate="circuit_stream_hidden",
            atol=1e-4,
            rtol=1e-4,
        ),
        "logits_comparison": _compare_tensors(
            torch,
            stream.logits,
            source_full_logits,
            reference="source_full_logits",
            candidate="circuit_stream_logits",
            atol=1e-3,
            rtol=1e-4,
        ),
        "candidate_shapes": {
            "hidden": list(stream.hidden_states.shape),
            "logits": list(stream.logits.shape),
        },
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "mode": report["mode"],
        "input_ids": report["input_ids"],
        "candidate_shapes": report["candidate_shapes"],
        "hidden_comparison": report["hidden_comparison"],
        "logits_comparison": report["logits_comparison"],
        "implementations": report["runtime"]["implementations"],
        "last_attention_state": report["runtime"]["last_attention_state"],
        "tensor_store": report["runtime"]["tensor_store"],
    }
    if report["mode"] == "stream":
        summary["tick_count"] = report["tick_count"]
        summary["incremental_logits_max_abs_diff"] = max(
            tick["logits_comparison"]["max_abs_diff"] for tick in report["ticks"]
        )
        summary["incremental_logits_allclose"] = all(
            tick["logits_comparison"]["allclose"] for tick in report["ticks"]
        )
    return summary


def _parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not ids:
        raise ValueError("at least one token id is required")
    return ids


def _compare_tensors(
    torch: Any,
    candidate_tensor: Any,
    reference_tensor: Any,
    reference: str,
    candidate: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    diff = (candidate_tensor - reference_tensor).abs()
    return {
        "reference": reference,
        "candidate": candidate,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(torch.allclose(candidate_tensor, reference_tensor, atol=atol, rtol=rtol)),
    }


def _collect_comparisons(value: Any) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"max_abs_diff", "allclose", "reference", "candidate"} <= set(value):
            comparisons.append(value)
        for child in value.values():
            comparisons.extend(_collect_comparisons(child))
    elif isinstance(value, list):
        for child in value:
            comparisons.extend(_collect_comparisons(child))
    return comparisons


if __name__ == "__main__":
    main()
