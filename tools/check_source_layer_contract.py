#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.source_oracle import check_lfm2_source_layer_contract, check_lfm2_source_model_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a transpiled pedal against the real LFM2 source layer.")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/aristath/models/lfm2.5/230m"))
    parser.add_argument("--pedals-dir", type=Path, default=Path("transpiled/lfm2_5_230m/layers"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="check every transpiled layer pedal")
    parser.add_argument("--summary", action="store_true", help="print a compact summary instead of the full JSON report")
    args = parser.parse_args()

    if args.all:
        report = check_lfm2_source_model_contract(model_dir=args.model_dir, pedals_dir=args.pedals_dir)
    else:
        report = check_lfm2_source_layer_contract(
            model_dir=args.model_dir,
            pedal_file=args.pedals_dir / f"layer_{args.layer:02d}.json",
            layer_index=args.layer,
        )
    report.raise_for_errors()
    if args.summary:
        print(json.dumps(_summary(report.to_json()), indent=2))
    else:
        print(json.dumps(report.to_json(), indent=2))


def _summary(report: dict) -> dict:
    if "layer_reports" not in report:
        return {
            "ok": report["ok"],
            "layer_id": report["layer_id"],
            "operator_type": report["operator_type"],
            "state": report["details"].get("state", {}),
        }

    operator_counts = {}
    state_shapes = {}
    for layer in report["layer_reports"]:
        operator_type = layer["operator_type"]
        operator_counts[operator_type] = operator_counts.get(operator_type, 0) + 1
        state = layer["details"].get("state", {})
        if operator_type == "conv":
            state_shapes.setdefault("conv_temporal_memory", state.get("declared_pedal_shape"))
        elif operator_type == "full_attention":
            state_shapes.setdefault("attention_key_per_token", state.get("declared_key_shape_per_token"))
            state_shapes.setdefault("attention_value_per_token", state.get("declared_value_shape_per_token"))

    return {
        "ok": report["ok"],
        "model_dir": report["model_dir"],
        "layer_count": report["layer_count"],
        "operator_counts": operator_counts,
        "state_shapes": state_shapes,
    }


if __name__ == "__main__":
    main()
