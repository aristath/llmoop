#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.circuit_lowering import lower_pedalboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower every model pedal into explicit stream-circuit IR artifacts.")
    parser.add_argument("--pedalboard-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    result = lower_pedalboard(args.pedalboard_dir, args.out_dir)
    if args.summary:
        report = {
            "ok": True,
            "index": str(result["index_path"]),
            "circuit_count": result["index"]["summary"]["circuit_count"],
            "operator_counts": result["index"]["summary"]["operator_counts"],
            "circuits": [
                {
                    "id": circuit["id"],
                    "operator_type": circuit["operator_type"],
                    "circuit": circuit["circuit"],
                    "implementation": circuit["implementation"],
                    "behavioral_role": circuit["behavioral_role"],
                }
                for circuit in result["circuits"]
            ],
        }
    else:
        report = result["index"]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
