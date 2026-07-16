from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.source_oracle import _oracle_imports
from llmoop.stream_processor import ExternalInputSignal, PrivateFeedbackSignal, PublicOutputSignal, StreamProcessor
from tests.fixtures import compiled_model_or_skip


STREAM_PROCESSOR_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(STREAM_PROCESSOR_DEPS_AVAILABLE, "stream processor dependencies are not installed")
class StreamProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, _, _ = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            torch=cls.torch,
        )
        cls.processor = StreamProcessor(runtime=cls.runtime)

    def test_running_stream_generation_matches_runtime_generation(self) -> None:
        prompt_ids = (1,)
        max_new_tokens = 4
        eos_token_id = int(self.runtime.config["eos_token_id"])

        with self.torch.no_grad():
            stream_run = self.processor.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            old_loop = self.runtime.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )

        self.assertEqual(old_loop.generated_ids, stream_run.generated_ids)
        self.assertEqual(old_loop.output_ids, stream_run.output_ids)
        self.assertEqual(old_loop.stop_reason, stream_run.stop_reason)
        self.assertEqual("greedy_sampler", stream_run.sampler)

    def test_public_output_and_private_feedback_are_separate_ports(self) -> None:
        prompt_ids = (1,)
        max_new_tokens = 4
        stream = self.processor.open_stream()

        with self.torch.no_grad():
            run = stream.generate(prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=None)

        self.assertEqual(max_new_tokens, len(run.public_outputs))
        self.assertEqual(max_new_tokens, len(run.private_feedback))
        self.assertTrue(all(isinstance(signal, PublicOutputSignal) for signal in run.public_outputs))
        self.assertTrue(all(isinstance(signal, PrivateFeedbackSignal) for signal in run.private_feedback))
        self.assertEqual([f"public_{index}" for index in range(max_new_tokens)], [signal.id for signal in run.public_outputs])
        self.assertEqual([f"feedback_{index}" for index in range(max_new_tokens)], [signal.id for signal in run.private_feedback])
        self.assertEqual([signal.token_id for signal in run.public_outputs], [signal.token_id for signal in run.private_feedback])
        self.assertTrue(all(signal.route == "public_output" for signal in run.public_outputs))
        self.assertTrue(all(signal.route == "insert_in" for signal in run.private_feedback))
        self.assertTrue(all(signal.origin == "insert_out" for signal in run.private_feedback))

        feedback_processing_ticks = [
            tick
            for tick in run.ticks
            if isinstance(tick.input_signal, PrivateFeedbackSignal)
        ]
        self.assertEqual(max_new_tokens, len(feedback_processing_ticks))
        self.assertIsNone(feedback_processing_ticks[-1].public_output)
        self.assertTrue(run.private_feedback[-1].closes_loop_after_processing)
        self.assertEqual("max_new_tokens", run.private_feedback[-1].stop_reason)
        self.assertEqual("idle", run.ticks[-1].status)

        last_attention = stream.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual("pedal", last_attention["owner"])
        self.assertEqual("layer_02", last_attention["pedal_id"])
        self.assertEqual("kv_memory", last_attention["state_id"])
        self.assertEqual(len(prompt_ids) + max_new_tokens, last_attention["updates"])
        self.assertEqual([1, 8, len(prompt_ids) + max_new_tokens, 64], last_attention["source_key_shape"])

    def test_mid_stream_external_input_has_priority_over_private_feedback(self) -> None:
        stream = self.processor.open_stream()

        with self.torch.no_grad():
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=4, eos_token_id=None)
            first = stream.tick()
            stream.inject_token(36309, origin="mid_stream_user", signal_id="mid_stream_user_token")
            second = stream.tick()

        self.assertIsInstance(first.private_feedback, PrivateFeedbackSignal)
        self.assertEqual(["feedback_0"], [signal.id for signal in stream.private_feedback_queue[:1]])
        self.assertIsInstance(second.input_signal, ExternalInputSignal)
        self.assertEqual("mid_stream_user_token", second.input_signal.id)
        self.assertEqual("mid_stream_user", second.input_signal.origin)
        self.assertIsInstance(second.public_output, PublicOutputSignal)
        self.assertEqual("public_output", second.public_output.route)
        self.assertIsInstance(second.private_feedback, PrivateFeedbackSignal)
        self.assertEqual("insert_in", second.private_feedback.route)

        last_attention = stream.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual(2, last_attention["updates"])
        self.assertEqual([1, 8, 2, 64], last_attention["source_key_shape"])

    def test_interrupt_closes_feedback_loop_without_resetting_state(self) -> None:
        stream = self.processor.open_stream()

        with self.torch.no_grad():
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=4, eos_token_id=None)
            stream.tick()
            before_interrupt = stream.model_stream.state.summary_for(2, "full_attention")
            interrupt_event = stream.interrupt(reason="user_interrupt")
            idle_tick = stream.tick()
            interrupt_stop_reason = stream.last_stop_reason
            stream.inject_prompt(prompt_ids=(36309,), max_new_tokens=1, eos_token_id=None)
            stream.run_until_idle()

        self.assertEqual("control_interrupt", interrupt_event.type)
        self.assertEqual("user_interrupt", interrupt_stop_reason)
        self.assertEqual([], stream.private_feedback_queue)
        self.assertEqual("idle", idle_tick.status)
        self.assertTrue(any(event.type == "control_interrupt" for event in idle_tick.events))
        self.assertEqual(1, before_interrupt["updates"])

        after_resume = stream.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual(3, after_resume["updates"])
        self.assertEqual([1, 8, 3, 64], after_resume["source_key_shape"])

    def test_stop_after_current_processes_one_feedback_then_goes_idle(self) -> None:
        stream = self.processor.open_stream()

        with self.torch.no_grad():
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=4, eos_token_id=None)
            stream.tick()
            stop_event = stream.stop_after_current(reason="user_stop")
            closing_tick = stream.tick()
            idle_tick = stream.tick()

        self.assertEqual("control_stop_after_current", stop_event.type)
        self.assertIsInstance(closing_tick.input_signal, PrivateFeedbackSignal)
        self.assertEqual("feedback_0", closing_tick.input_signal.id)
        self.assertIsNone(closing_tick.public_output)
        self.assertTrue(any(event.type == "loop_closed" for event in closing_tick.events))
        self.assertEqual("user_stop", stream.last_stop_reason)
        self.assertEqual("idle", idle_tick.status)
        self.assertEqual(1, len(stream.public_output_queue))

        last_attention = stream.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual(2, last_attention["updates"])
        self.assertEqual([1, 8, 2, 64], last_attention["source_key_shape"])

    def test_reset_state_replaces_transient_circuit_explicitly(self) -> None:
        stream = self.processor.open_stream()

        with self.torch.no_grad():
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=1, eos_token_id=None)
            stream.run_until_idle()
            before_reset = stream.model_stream.state.summary_for(2, "full_attention")
            reset_event = stream.reset_state(reason="user_reset")
            after_reset = stream.model_stream.state.summary_for(2, "full_attention")
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=1, eos_token_id=None)
            stream.run_until_idle()

        self.assertEqual("control_reset_state", reset_event.type)
        self.assertEqual(2, before_reset["updates"])
        self.assertEqual(0, after_reset["updates"])
        self.assertIsNone(after_reset["source_key_shape"])

        after_resume = stream.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual(2, after_resume["updates"])
        self.assertEqual([1, 8, 2, 64], after_resume["source_key_shape"])

    def test_clone_fork_copies_transient_state_without_sharing_tensors(self) -> None:
        parent = self.processor.open_stream(stream_id="parent")

        with self.torch.no_grad():
            parent.inject_prompt(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            parent_first = parent.tick()
            child = parent.fork(stream_id="child", policy="clone")

        self.assertIsInstance(parent_first.private_feedback, PrivateFeedbackSignal)
        self.assertEqual("feedback_0", parent.private_feedback_queue[0].id)
        self.assertEqual("feedback_0", child.private_feedback_queue[0].id)
        self.assertEqual(parent.model_stream.position, child.model_stream.position)

        parent_attention = parent.model_stream.state.attention_states[2]
        child_attention = child.model_stream.state.attention_states[2]
        self.assertIsNotNone(parent_attention.key)
        self.assertIsNotNone(child_attention.key)
        self.assertEqual(list(parent_attention.key.shape), list(child_attention.key.shape))
        self.assertNotEqual(parent_attention.key.data_ptr(), child_attention.key.data_ptr())

        with self.torch.no_grad():
            parent_second = parent.tick()
            child.inject_token(36309, origin="branch_external_input", signal_id="branch_token")
            child_second = child.tick()

        self.assertIsInstance(parent_second.input_signal, PrivateFeedbackSignal)
        self.assertEqual("feedback_0", parent_second.input_signal.id)
        self.assertIsInstance(child_second.input_signal, ExternalInputSignal)
        self.assertEqual("branch_token", child_second.input_signal.id)
        self.assertEqual("branch_external_input", child_second.input_signal.origin)
        self.assertEqual(["feedback_1"], [signal.id for signal in parent.private_feedback_queue])
        self.assertEqual(["feedback_0", "feedback_1"], [signal.id for signal in child.private_feedback_queue])

        parent_summary = parent.model_stream.state.summary_for(2, "full_attention")
        child_summary = child.model_stream.state.summary_for(2, "full_attention")
        self.assertEqual(2, parent_summary["updates"])
        self.assertEqual(2, child_summary["updates"])
        self.assertEqual([1, 8, 2, 64], parent_summary["source_key_shape"])
        self.assertEqual([1, 8, 2, 64], child_summary["source_key_shape"])

        child.reset_state(reason="child_reset")
        self.assertEqual(2, parent.model_stream.state.summary_for(2, "full_attention")["updates"])
        self.assertEqual(0, child.model_stream.state.summary_for(2, "full_attention")["updates"])

    def test_fresh_fork_starts_with_empty_transient_state(self) -> None:
        parent = self.processor.open_stream(stream_id="parent")

        with self.torch.no_grad():
            parent.inject_prompt(prompt_ids=(1,), max_new_tokens=2, eos_token_id=None)
            parent.tick()
            child = parent.fork(stream_id="fresh_child", policy="fresh")

        self.assertEqual(1, parent.model_stream.state.summary_for(2, "full_attention")["updates"])
        self.assertEqual(0, child.model_stream.state.summary_for(2, "full_attention")["updates"])
        self.assertEqual([], child.private_feedback_queue)
        self.assertTrue(child.pending_control_events)
        self.assertEqual("control_fork_fresh", child.pending_control_events[-1].type)

    def test_restore_snapshot_rewinds_stream_state_and_pending_feedback(self) -> None:
        stream = self.processor.open_stream(stream_id="restore_source")

        with self.torch.no_grad():
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            stream.tick()
            snapshot = stream.snapshot_state(snapshot_id="after_first_tick")
            original_second_tick = stream.tick()
            advanced_summary = stream.model_stream.state.summary_for(2, "full_attention")
            restore_event = stream.restore_snapshot(snapshot, reason="rewind")
            restored_summary = stream.model_stream.state.summary_for(2, "full_attention")
            restored_second_tick = stream.tick()

        self.assertEqual("control_restore_snapshot", restore_event.type)
        self.assertEqual(2, advanced_summary["updates"])
        self.assertEqual(1, restored_summary["updates"])
        self.assertEqual(["feedback_0"], [signal.id for signal in snapshot.pending_private_feedback])
        self.assertIsInstance(original_second_tick.input_signal, PrivateFeedbackSignal)
        self.assertIsInstance(restored_second_tick.input_signal, PrivateFeedbackSignal)
        self.assertEqual(original_second_tick.input_signal.id, restored_second_tick.input_signal.id)
        self.assertIsNotNone(original_second_tick.public_output)
        self.assertIsNotNone(restored_second_tick.public_output)
        self.assertEqual(original_second_tick.public_output.token_id, restored_second_tick.public_output.token_id)
        self.assertTrue(any(event.type == "control_restore_snapshot" for event in restored_second_tick.events))


if __name__ == "__main__":
    unittest.main()
