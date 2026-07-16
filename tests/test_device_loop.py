from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.device_loop import StreamDevice
from llmoop.source_oracle import _oracle_imports
from llmoop.stream_processor import PrivateFeedbackSignal, StreamProcessor
from tests.fixtures import compiled_model_or_skip


DEVICE_LOOP_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(DEVICE_LOOP_DEPS_AVAILABLE, "device loop tests require torch, transformers, and safetensors")
class DeviceLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, _, _ = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            model_dir=cls.fixture.source_model_dir,
            torch=cls.torch,
        )
        cls.processor = StreamProcessor(runtime=cls.runtime)

    def test_device_dispatch_owns_feedback_loop(self) -> None:
        device = StreamDevice(processor=self.processor)
        stream = device.create_stream("s0")

        with self.torch.no_grad():
            device.inject_prompt("s0", prompt_ids=(1,), max_new_tokens=4, eos_token_id=None)
            run = device.dispatch_until_idle()

        self.assertEqual("idle", run.status)
        self.assertEqual(5, len(run.ticks))
        self.assertEqual(4, len(run.outputs))
        self.assertEqual(4, len(device.output_queue))
        self.assertEqual([f"public_{index}" for index in range(4)], [event.output.id for event in run.outputs])
        self.assertEqual([f"feedback_{index}" for index in range(4)], [signal.id for signal in stream.private_feedback_history])
        self.assertEqual([], list(device.active_queue))

        last_attention = stream.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual(5, last_attention["updates"])
        self.assertEqual([1, 8, 5, 64], last_attention["source_key_shape"])

    def test_dispatch_budget_yields_and_resumes(self) -> None:
        device = StreamDevice(processor=self.processor)
        device.create_stream("s0")

        with self.torch.no_grad():
            device.inject_prompt("s0", prompt_ids=(1,), max_new_tokens=4, eos_token_id=None)
            first = device.dispatch(max_ticks=2)
            second = device.dispatch_until_idle()

        self.assertEqual("budget_exhausted", first.status)
        self.assertEqual(2, len(first.ticks))
        self.assertEqual(("s0",), first.active_streams)
        self.assertEqual("idle", second.status)
        self.assertEqual(3, len(second.ticks))
        self.assertEqual(4, len(device.output_queue))

    def test_round_robin_dispatches_multiple_streams(self) -> None:
        device = StreamDevice(processor=self.processor)
        device.create_stream("a")
        device.create_stream("b")

        with self.torch.no_grad():
            device.inject_prompt("a", prompt_ids=(1,), max_new_tokens=2, eos_token_id=None)
            device.inject_prompt("b", prompt_ids=(1,), max_new_tokens=2, eos_token_id=None)
            run = device.dispatch_until_idle()

        self.assertEqual("idle", run.status)
        self.assertEqual(["a", "b", "a", "b", "a", "b"], [tick.stream_id for tick in run.ticks])
        self.assertEqual(2, len([event for event in run.outputs if event.stream_id == "a"]))
        self.assertEqual(2, len([event for event in run.outputs if event.stream_id == "b"]))
        self.assertEqual(4, len(device.output_queue))

    def test_device_fork_registers_child_and_schedules_clone(self) -> None:
        device = StreamDevice(processor=self.processor)
        parent = device.create_stream("parent")

        with self.torch.no_grad():
            device.inject_prompt("parent", prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            device.dispatch(max_ticks=1)
            child = device.fork_stream("parent", "child", policy="clone")
            run = device.dispatch(max_ticks=2)

        self.assertIn("child", device.streams)
        self.assertIs(child, device.get_stream("child"))
        self.assertIsInstance(parent.private_feedback_queue[0], PrivateFeedbackSignal)
        self.assertEqual(["parent", "child"], [tick.stream_id for tick in run.ticks])
        self.assertEqual(1, len([event for event in run.outputs if event.stream_id == "parent"]))
        self.assertEqual(1, len([event for event in run.outputs if event.stream_id == "child"]))
        self.assertEqual(
            parent.model_stream.state.summary_for(2, "full_attention")["source_key_shape"],
            child.model_stream.state.summary_for(2, "full_attention")["source_key_shape"],
        )

    def test_device_controls_schedule_or_deschedule_stream_work(self) -> None:
        device = StreamDevice(processor=self.processor)
        stream = device.create_stream("s0")

        with self.torch.no_grad():
            device.inject_prompt("s0", prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            device.dispatch(max_ticks=1)
            stop_event = device.stop_after_current("s0", reason="device_cut_signal")
            closing = device.dispatch_until_idle()
            stop_reason = stream.last_stop_reason
            reset_event = device.reset_stream("s0", reason="device_reset")

        self.assertEqual("control_stop_after_current", stop_event.type)
        self.assertEqual("idle", closing.status)
        self.assertEqual(1, len(device.output_queue))
        self.assertEqual([], stream.private_feedback_queue)
        self.assertEqual("device_cut_signal", stop_reason)
        self.assertEqual("control_reset_state", reset_event.type)
        self.assertEqual("device_reset", stream.last_stop_reason)
        self.assertEqual([], list(device.active_queue))
        self.assertEqual(0, stream.model_stream.state.summary_for(2, "full_attention")["updates"])


if __name__ == "__main__":
    unittest.main()
