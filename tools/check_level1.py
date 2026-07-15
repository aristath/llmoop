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
from llmoop.validation import validate_pedalboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the level-1 symbolic llmoop acceptance checks.")
    parser.add_argument("--model-dir", type=Path, default=Path("transpiled/lfm2_5_230m"))
    args = parser.parse_args()

    pedalboard = Pedalboard.from_dir(args.model_dir)
    validation = validate_pedalboard(pedalboard)
    validation.raise_for_errors()

    runtime = PedalboardRuntime.symbolic(pedalboard)
    activation = runtime.activate()
    _require(len(activation.steps) == 14, "expected one activation step per layer pedal")
    _require(activation.output_frame.shape == (1024,), "expected final symbolic frame width 1024")

    stream = runtime.open_stream()
    stream.enqueue(frame_id="frame_0")
    stream.enqueue(frame_id="frame_1")
    stream_ticks = stream.run_until_idle()
    _require([tick.status for tick in stream_ticks] == ["processed", "processed", "idle"], "stream tick statuses diverged")
    _require(len(stream.output_queue) == 2, "stream should produce two output frames")
    _require(stream.state_versions["layer_02.kv_memory"] == 2, "attention transient state should persist across stream ticks")

    engine = SymbolicStreamingEngine.from_pedalboard(pedalboard)
    engine.enqueue_token(42, packet_id="seed")
    engine_ticks = engine.run_until_idle(feedback=True, max_feedback_depth=2)
    _require([tick.status for tick in engine_ticks] == ["processed", "processed", "processed", "idle"], "engine feedback tick statuses diverged")
    _require([packet.id for packet in engine.output_queue] == ["public_0", "public_1", "public_2"], "public output packets diverged")
    _require(engine.pedal_stream.state_versions["layer_02.kv_memory"] == 3, "feedback loop should reuse the same stream state")

    feedback_packets = [tick.feedback_packet for tick in engine_ticks if tick.feedback_packet is not None]
    _require([packet.id for packet in feedback_packets] == ["feedback_0", "feedback_1"], "private feedback packets diverged")
    _require(all(packet.origin == "insert_out" for packet in feedback_packets), "feedback packets must originate at insert_out")
    _require(
        all(tick.output_packet is None or tick.output_packet.route == "external_output" for tick in engine_ticks),
        "public output packets must remain external-output routed",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "model_dir": str(args.model_dir),
                "validation": validation.to_json(),
                "pedalboard": {
                    "pedal_count": len(pedalboard.pedals),
                    "operator_counts": pedalboard.summary()["operator_counts"],
                    "activation_steps": len(activation.steps),
                    "output_shape": list(activation.output_frame.shape),
                },
                "stream": {
                    "statuses": [tick.status for tick in stream_ticks],
                    "output_frames": len(stream.output_queue),
                    "layer_02_kv_version": stream.state_versions["layer_02.kv_memory"],
                },
                "engine": {
                    "statuses": [tick.status for tick in engine_ticks],
                    "public_outputs": [packet.id for packet in engine.output_queue],
                    "private_feedback": [packet.id for packet in feedback_packets],
                    "layer_02_kv_version": engine.pedal_stream.state_versions["layer_02.kv_memory"],
                },
            },
            indent=2,
        )
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
