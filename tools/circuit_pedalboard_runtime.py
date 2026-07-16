#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.circuit_pedalboard import CircuitPedalboard
from llmoop.pedalboard import PedalboardRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run symbolic checks from the lowered circuit pedalboard.")
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    board = CircuitPedalboard.from_dir(args.circuit_dir)
    runtime = PedalboardRuntime.symbolic(board)  # type: ignore[arg-type]
    activation = runtime.activate()
    stream = runtime.open_stream()
    stream.enqueue(frame_id="frame_0")
    stream.enqueue(frame_id="frame_1")
    ticks = stream.run_until_idle()

    report = {
        "ok": True,
        "summary": board.summary(),
        "activation": {
            "steps": len(activation.steps),
            "output_shape": list(activation.output_frame.shape),
            "history": list(activation.output_frame.history),
        },
        "stream": {
            "statuses": [tick.status for tick in ticks],
            "output_frames": len(stream.output_queue),
            "state_versions": dict(stream.state_versions),
        },
    }
    if args.trace:
        report["trace"] = board.activation_trace()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
