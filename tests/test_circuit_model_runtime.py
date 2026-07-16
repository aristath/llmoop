from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.source_oracle import _oracle_imports
from tests.fixtures import compiled_model_or_skip


RUNTIME_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(RUNTIME_DEPS_AVAILABLE, "runtime oracle dependencies are not installed")
class CircuitModelRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, cls.auto_model, cls.dynamic_cache = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            model_dir=cls.fixture.source_model_dir,
            torch=cls.torch,
        )
        cls.source = cls.auto_model.from_pretrained(cls.fixture.source_model_dir, dtype=cls.torch.float32)
        cls.source.eval()

    def test_circuit_model_runtime_matches_source_hidden_and_logits(self) -> None:
        input_ids = (1, 2, 3, 4)
        input_tensor = self.torch.tensor([list(input_ids)], dtype=self.torch.long)

        with self.torch.no_grad():
            candidate = self.runtime.forward_input_ids(input_ids)
            source_hidden = self.source.model(input_ids=input_tensor, use_cache=True).last_hidden_state
            source_logits = self.source(input_ids=input_tensor, use_cache=True).logits

        self.assertEqual((1, 4, 1024), tuple(candidate.hidden_states.shape))
        self.assertEqual((1, 4, 65536), tuple(candidate.logits.shape))
        self.assertTrue(
            all(step.implementation.startswith("executable_") for step in candidate.steps)
        )
        self.assertTrue(self.torch.allclose(candidate.hidden_states, source_hidden, atol=1e-6, rtol=1e-6))
        self.assertTrue(self.torch.allclose(candidate.logits, source_logits, atol=1e-6, rtol=1e-6))

    def test_circuit_model_stream_uses_per_pedal_state_and_matches_source(self) -> None:
        input_ids = (1, 2, 3, 4)
        stream = self.runtime.open_stream()

        with self.torch.no_grad():
            stream.run_teacher_forced(input_ids)
            source_hidden = self.source.model(
                input_ids=self.torch.tensor([list(input_ids)], dtype=self.torch.long),
                use_cache=True,
            ).last_hidden_state
            source_logits = self.source(
                input_ids=self.torch.tensor([list(input_ids)], dtype=self.torch.long),
                use_cache=True,
            ).logits

        self.assertEqual((1, 4, 1024), tuple(stream.hidden_states.shape))
        self.assertEqual((1, 4, 65536), tuple(stream.logits.shape))
        self.assertTrue(self.torch.allclose(stream.hidden_states, source_hidden, atol=1e-4, rtol=1e-4))
        self.assertTrue(self.torch.allclose(stream.logits, source_logits, atol=1e-3, rtol=1e-4))

        last_attention = next(
            step.state for step in stream.ticks[-1].output.steps if step.operator_type == "full_attention"
        )
        self.assertEqual("pedal", last_attention["owner"])
        self.assertEqual("layer_02", last_attention["pedal_id"])
        self.assertEqual("kv_memory", last_attention["state_id"])
        self.assertEqual([1, 8, 4, 64], last_attention["source_key_shape"])
        self.assertEqual([1, 8, 4, 64], last_attention["source_value_shape"])
        self.assertEqual(4, last_attention["updates"])


if __name__ == "__main__":
    unittest.main()
