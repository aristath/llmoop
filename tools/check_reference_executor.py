#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.circuit_executors import (
    install_all_circuit_pedals,
    install_attention_circuit_pedals,
    install_shortconv_circuit_pedals,
)
from llmoop.pedalboard import Pedalboard
from llmoop.reference_runtime import ReferencePedalExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the source-backed reference pedal executor.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pedalboard-dir", type=Path, required=True)
    parser.add_argument("--circuit-dir", type=Path)
    parser.add_argument("--token-id", type=int, default=None)
    parser.add_argument("--stream-input-ids", type=str, default=None, help="comma-separated teacher-forced token ids")
    parser.add_argument("--circuit-conv-pedals", action="store_true", help="replace every conv pedal with the reusable executable short-conv circuit")
    parser.add_argument("--circuit-attention-pedals", action="store_true", help="replace every attention pedal with the reusable executable GQA attention circuit")
    parser.add_argument("--all-circuit-pedals", action="store_true", help="replace every layer pedal with executable circuit implementations")
    parser.add_argument("--custom-stream-state", action="store_true", help="use llmoop per-pedal stream state instead of Transformers DynamicCache for executable pedals")
    parser.add_argument("--summary", action="store_true", help="print a compact summary instead of every pedal step")
    args = parser.parse_args()

    pedalboard = Pedalboard.from_dir(args.pedalboard_dir)
    executor = ReferencePedalExecutor.from_model_dir(pedalboard=pedalboard, model_dir=args.model_dir)
    if (args.all_circuit_pedals or args.circuit_attention_pedals or args.circuit_conv_pedals) and args.circuit_dir is None:
        raise SystemExit("--circuit-dir is required when installing executable circuit pedals")
    if args.all_circuit_pedals:
        install_all_circuit_pedals(executor, args.circuit_dir)
    else:
        if args.circuit_attention_pedals:
            install_attention_circuit_pedals(executor, args.circuit_dir)
    if args.circuit_conv_pedals:
        install_shortconv_circuit_pedals(executor, args.circuit_dir)
    if args.custom_stream_state:
        executor.use_pedal_stream_state()

    if args.stream_input_ids:
        input_ids = tuple(int(part.strip()) for part in args.stream_input_ids.split(",") if part.strip())
        run = executor.open_stream().run_teacher_forced(input_ids)
        report = run.to_json()
        if args.summary:
            report = {
                "input_ids": report["input_ids"],
                "tick_count": report["tick_count"],
                "output_tensor_shape": report["output_tensor_shape"],
                "incremental_max_abs_diff": max(
                    tick["incremental_comparison"]["max_abs_diff"] for tick in report["ticks"]
                ),
                "incremental_allclose": all(tick["incremental_comparison"]["allclose"] for tick in report["ticks"]),
                "full_sequence_comparison": report["comparison"],
                "implementations": {
                    step["pedal_id"]: step["implementation"] for step in report["ticks"][0]["activation"]["steps"]
                },
                "last_attention_state": next(
                    step["state"]
                    for step in report["ticks"][-1]["activation"]["steps"]
                    if step["operator_type"] == "full_attention"
                ),
            }
        print(json.dumps(report, indent=2))
        if not all(tick.incremental_comparison["allclose"] for tick in run.ticks):
            raise SystemExit("reference pedal stream diverged from source incremental forward")
        if not run.comparison["allclose"]:
            raise SystemExit("reference pedal stream diverged from source full-sequence forward beyond tolerance")
        return

    activation = executor.activate_token(args.token_id)

    report = activation.to_json()
    if args.summary:
        report = {
            "input_ids": report["input_ids"],
            "step_count": len(report["steps"]),
            "pedal_output_shape": report["pedal_output_frame"]["tensor_shape"],
            "normalized_output_shape": report["normalized_output_frame"]["tensor_shape"],
            "comparison": report["comparison"],
            "implementations": {step["pedal_id"]: step["implementation"] for step in report["steps"]},
            "first_state": report["steps"][0]["state"],
            "first_attention_state": next(
                step["state"] for step in report["steps"] if step["operator_type"] == "full_attention"
            ),
        }

    print(json.dumps(report, indent=2))

    if not activation.comparison["allclose"]:
        raise SystemExit("reference pedal executor diverged from source model forward")


if __name__ == "__main__":
    main()
