from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.randomness import RandomSignal, RandomSource
from llmoop.samplers import TemperatureSamplerPedal
from llmoop.source_oracle import _oracle_imports
from llmoop.stream_processor import PrivateFeedbackSignal, StreamProcessor
from tests.fixtures import compiled_model_or_skip


RANDOMNESS_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


class ExplicitRandomSignalTest(unittest.TestCase):
    def test_random_source_is_replayable_from_snapshot(self) -> None:
        source = RandomSource(source_id="test_random", seed=123)
        first = source.next_signal()
        snapshot = source.snapshot()
        second = source.next_signal()
        restored_second = snapshot.restore().next_signal()

        self.assertEqual(0, first.counter)
        self.assertEqual(1, second.counter)
        self.assertEqual(second.value, restored_second.value)
        self.assertEqual(second.to_json(), restored_second.to_json())


@unittest.skipUnless(RANDOMNESS_DEPS_AVAILABLE, "randomness tests require torch, transformers, and safetensors")
class TemperatureSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.torch, _, _ = _oracle_imports()

    def test_temperature_sampler_requires_explicit_random_signal(self) -> None:
        sampler = TemperatureSamplerPedal()
        logits = self.torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=self.torch.float32)

        with self.assertRaises(ValueError):
            sampler.sample(logits, self.torch)

    def test_random_signal_selects_categorical_token(self) -> None:
        sampler = TemperatureSamplerPedal()
        logits = self.torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=self.torch.float32)

        low = sampler.sample(
            logits,
            self.torch,
            random_signal=RandomSignal(id="r.0", source_id="r", seed=1, counter=0, value=0.10),
        )
        high = sampler.sample(
            logits,
            self.torch,
            random_signal=RandomSignal(id="r.1", source_id="r", seed=1, counter=1, value=0.90),
        )

        self.assertEqual(0, low.token_id)
        self.assertEqual(3, high.token_id)
        self.assertTrue(low.state["uses_randomness"])
        self.assertEqual(0.10, low.state["random_signal"]["value"])


@unittest.skipUnless(RANDOMNESS_DEPS_AVAILABLE, "stream randomness tests require torch, transformers, and safetensors")
class StreamRandomnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, _, _ = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            package_dir=cls.fixture.package_dir,
            torch=cls.torch,
        )

    def test_same_seed_and_state_replay_same_stochastic_tokens(self) -> None:
        sampler = TemperatureSamplerPedal(temperature=2.0, top_k=16)
        processor = StreamProcessor(runtime=self.runtime, sampler=sampler, random_seed=123)

        with self.torch.no_grad():
            left = processor.generate(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None, stream_id="left")
            right = processor.generate(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None, stream_id="right")

        self.assertEqual(left.generated_ids, right.generated_ids)
        left_random = [signal.sampler["state"]["random_signal"] for signal in left.public_outputs]
        right_random = [signal.sampler["state"]["random_signal"] for signal in right.public_outputs]
        self.assertEqual([item["counter"] for item in left_random], [item["counter"] for item in right_random])
        self.assertEqual([item["value"] for item in left_random], [item["value"] for item in right_random])

    def test_clone_fork_replays_random_branch(self) -> None:
        sampler = TemperatureSamplerPedal(temperature=2.0, top_k=16)
        processor = StreamProcessor(runtime=self.runtime, sampler=sampler, random_seed=456)
        parent = processor.open_stream(stream_id="parent")

        with self.torch.no_grad():
            parent.inject_prompt(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            parent.tick()
            child = parent.fork(stream_id="child", policy="clone", random_policy="clone")
            parent_next = parent.tick()
            child_next = child.tick()

        self.assertIsInstance(parent_next.input_signal, PrivateFeedbackSignal)
        self.assertIsInstance(child_next.input_signal, PrivateFeedbackSignal)
        self.assertEqual(parent_next.public_output.token_id, child_next.public_output.token_id)
        parent_random = parent_next.public_output.sampler["state"]["random_signal"]
        child_random = child_next.public_output.sampler["state"]["random_signal"]
        self.assertEqual(parent_random, child_random)

    def test_reseeded_fork_uses_independent_random_branch(self) -> None:
        sampler = TemperatureSamplerPedal(temperature=2.0, top_k=16)
        processor = StreamProcessor(runtime=self.runtime, sampler=sampler, random_seed=456)
        parent = processor.open_stream(stream_id="parent")

        with self.torch.no_grad():
            parent.inject_prompt(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            parent.tick()
            child = parent.fork(stream_id="child", policy="clone", random_policy="fresh", random_seed=789)
            parent_next = parent.tick()
            child_next = child.tick()

        parent_random = parent_next.public_output.sampler["state"]["random_signal"]
        child_random = child_next.public_output.sampler["state"]["random_signal"]
        self.assertEqual(456, parent_random["seed"])
        self.assertEqual(789, child_random["seed"])
        self.assertNotEqual(parent_random["value"], child_random["value"])
        self.assertEqual("child.random", child_random["source_id"])

    def test_restore_snapshot_rewinds_random_source(self) -> None:
        sampler = TemperatureSamplerPedal(temperature=2.0, top_k=16)
        processor = StreamProcessor(runtime=self.runtime, sampler=sampler, random_seed=321)
        stream = processor.open_stream(stream_id="restore_random")

        with self.torch.no_grad():
            stream.inject_prompt(prompt_ids=(1,), max_new_tokens=3, eos_token_id=None)
            stream.tick()
            snapshot = stream.snapshot_state(snapshot_id="after_random_0")
            original_second = stream.tick()
            stream.restore_snapshot(snapshot, reason="rewind_random")
            restored_second = stream.tick()

        original_random = original_second.public_output.sampler["state"]["random_signal"]
        restored_random = restored_second.public_output.sampler["state"]["random_signal"]
        self.assertEqual(original_second.public_output.token_id, restored_second.public_output.token_id)
        self.assertEqual(original_random, restored_random)


if __name__ == "__main__":
    unittest.main()
