from __future__ import annotations

import unittest
from pathlib import Path

from llmoop.pedalboard import Pedalboard, PedalboardRuntime
from llmoop.stream_engine import FeedbackPacket, SymbolicOutputPacket, SymbolicStreamingEngine, TokenPacket
from llmoop.validation import validate_pedalboard


MODEL_DIR = Path("transpiled/lfm2_5_230m")


class Level1RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pedalboard = Pedalboard.from_dir(MODEL_DIR)

    def test_transpiled_pedalboard_validates(self) -> None:
        report = validate_pedalboard(self.pedalboard)
        self.assertTrue(report.ok, [issue.to_json() for issue in report.errors])

    def test_symbolic_activation_walks_every_layer(self) -> None:
        runtime = PedalboardRuntime.symbolic(self.pedalboard)
        activation = runtime.activate()

        self.assertEqual(14, len(activation.steps))
        self.assertEqual((1024,), activation.output_frame.shape)
        self.assertEqual(tuple(f"layer_{index:02d}" for index in range(14)), activation.output_frame.history)

    def test_stream_state_persists_across_ticks(self) -> None:
        stream = PedalboardRuntime.symbolic(self.pedalboard).open_stream()
        stream.enqueue(frame_id="frame_0")
        stream.enqueue(frame_id="frame_1")

        ticks = stream.run_until_idle()

        self.assertEqual(["processed", "processed", "idle"], [tick.status for tick in ticks])
        self.assertEqual(2, len(stream.output_queue))
        self.assertEqual(2, stream.state_versions["layer_00.temporal_memory"])
        self.assertEqual(2, stream.state_versions["layer_02.kv_memory"])

    def test_engine_keeps_public_output_and_feedback_separate(self) -> None:
        engine = SymbolicStreamingEngine.from_pedalboard(self.pedalboard)
        engine.enqueue_token(42, packet_id="seed")

        ticks = engine.run_until_idle(feedback=True, max_feedback_depth=2)

        self.assertEqual(["processed", "processed", "processed", "idle"], [tick.status for tick in ticks])
        self.assertEqual(["public_0", "public_1", "public_2"], [packet.id for packet in engine.output_queue])
        self.assertTrue(all(isinstance(packet, SymbolicOutputPacket) for packet in engine.output_queue))

        feedback_packets = [tick.feedback_packet for tick in ticks if tick.feedback_packet is not None]
        self.assertEqual(["feedback_0", "feedback_1"], [packet.id for packet in feedback_packets])
        self.assertTrue(all(isinstance(packet, FeedbackPacket) for packet in feedback_packets))
        self.assertTrue(all(packet.origin == "insert_out" for packet in feedback_packets))
        self.assertTrue(all(packet.signal == "frame" for packet in feedback_packets))
        self.assertTrue(all(packet.shape == (1024,) for packet in feedback_packets))
        self.assertTrue(all(isinstance(tick.input_packet, (TokenPacket, FeedbackPacket)) for tick in ticks[:-1]))
        self.assertEqual(3, engine.pedal_stream.state_versions["layer_02.kv_memory"])


if __name__ == "__main__":
    unittest.main()
