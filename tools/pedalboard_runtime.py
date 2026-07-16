#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.pedalboard import Pedalboard, PedalboardRuntime
from llmoop.stream_engine import SymbolicStreamingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and symbolically wire an llmoop pedalboard.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true", help="include the series activation trace")
    parser.add_argument("--step", action="store_true", help="run one symbolic frame through the board")
    parser.add_argument("--stream", type=int, default=0, help="enqueue N symbolic frames and tick the stream loop")
    parser.add_argument("--engine", type=int, default=0, help="enqueue N symbolic token packets through the full stream shell")
    parser.add_argument("--feedback-depth", type=int, default=0, help="allow symbolic output-to-input feedback for this many hops")
    args = parser.parse_args()

    pedalboard = Pedalboard.from_dir(args.model_dir)
    output = {
        "summary": pedalboard.summary(),
    }
    if args.trace:
        output["activation_trace"] = pedalboard.activation_trace()
    if args.step:
        runtime = PedalboardRuntime.symbolic(pedalboard)
        output["symbolic_activation"] = runtime.activate().to_json()
    if args.stream:
        runtime = PedalboardRuntime.symbolic(pedalboard)
        stream = runtime.open_stream()
        for index in range(args.stream):
            stream.enqueue(frame_id=f"frame_{index}")
        output["stream_ticks"] = [tick.to_json() for tick in stream.run_until_idle()]
        output["stream_outputs"] = [frame.to_json() for frame in stream.output_queue]
    if args.engine:
        engine = SymbolicStreamingEngine.from_pedalboard(pedalboard)
        for index in range(args.engine):
            engine.enqueue_token(token_id=index, packet_id=f"token_{index}")
        engine_ticks = engine.run_until_idle(
            feedback=args.feedback_depth > 0,
            max_feedback_depth=args.feedback_depth,
        )
        output["engine"] = engine.to_json()
        output["engine_ticks"] = [tick.to_json() for tick in engine_ticks]
        output["engine_outputs"] = [packet.to_json() for packet in engine.output_queue]

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
