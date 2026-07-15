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

from llmoop.circuit_lowering import build_pedal_circuit, lower_pedal


Json = dict[str, Any]


DEFAULT_PEDAL = Path("transpiled/lfm2_5_230m/layers/layer_00.json")
DEFAULT_OUT_DIR = Path("lowered/lfm2_5_230m/layer_00")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower LFM2 layer_00 into an explicit stream-circuit IR artifact.")
    parser.add_argument("--pedal", type=Path, default=DEFAULT_PEDAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    result = lower_layer00(args.pedal, args.out_dir)
    if args.summary:
        print(
            json.dumps(
                {
                    "ok": result["validation"]["ok"],
                    "circuit": str(result["circuit_path"]),
                    "params": str(result["params_path"]),
                    "state": str(result["state_path"]),
                    "node_count": len(result["circuit"]["nodes"]),
                    "check_count": result["validation"]["check_count"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(_stringify_paths(result), indent=2))

    if not result["validation"]["ok"]:
        raise SystemExit("layer_00 circuit failed validation")


def lower_layer00(pedal_path: Path = DEFAULT_PEDAL, out_dir: Path = DEFAULT_OUT_DIR) -> Json:
    result = lower_pedal(pedal_path, out_dir)
    if result["pedal"]["id"] != "layer_00":
        raise ValueError(f"expected layer_00 pedal, got {result['pedal']['id']!r}")
    return result


def build_layer00_circuit(pedal: Json, pedal_path: Path = DEFAULT_PEDAL) -> Json:
    if pedal.get("id") != "layer_00":
        raise ValueError(f"expected layer_00 pedal, got {pedal.get('id')!r}")
    return build_pedal_circuit(pedal, pedal_path)


def _stringify_paths(data: Json) -> Json:
    result: Json = {}
    for key, value in data.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif key == "pedal":
            result[key] = value["id"]
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
