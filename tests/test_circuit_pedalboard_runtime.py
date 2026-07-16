from __future__ import annotations

import unittest

from llmoop.circuit_pedalboard import CircuitPedalboard
from llmoop.pedalboard import PedalboardRuntime
from tests.fixtures import compiled_model_or_skip


class CircuitPedalboardRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.board = CircuitPedalboard.from_dir(cls.fixture.lowered_dir)

    def test_circuit_pedalboard_loads_all_lowered_pedals(self) -> None:
        summary = self.board.summary()

        self.assertEqual(14, summary["circuit_count"])
        self.assertEqual({"conv": 8, "full_attention": 6}, summary["operator_counts"])
        self.assertEqual({"source_reference_circuit": 14}, summary["behavioral_roles"])
        self.assertEqual([1024], summary["input_shape"])
        self.assertEqual([1024], summary["output_shape"])
        self.assertEqual(14, summary["stream_state_count"])

    def test_symbolic_runtime_can_walk_circuit_pedalboard(self) -> None:
        activation = PedalboardRuntime.symbolic(self.board).activate()  # type: ignore[arg-type]

        self.assertEqual(14, len(activation.steps))
        self.assertEqual((1024,), activation.output_frame.shape)
        self.assertEqual(tuple(f"layer_{index:02d}" for index in range(14)), activation.output_frame.history)

    def test_stream_state_persists_when_runtime_uses_circuit_pedalboard(self) -> None:
        stream = PedalboardRuntime.symbolic(self.board).open_stream()  # type: ignore[arg-type]
        stream.enqueue(frame_id="frame_0")
        stream.enqueue(frame_id="frame_1")

        ticks = stream.run_until_idle()

        self.assertEqual(["processed", "processed", "idle"], [tick.status for tick in ticks])
        self.assertEqual(2, len(stream.output_queue))
        self.assertEqual(2, stream.state_versions["layer_00.temporal_memory"])
        self.assertEqual(2, stream.state_versions["layer_02.kv_memory"])


if __name__ == "__main__":
    unittest.main()
