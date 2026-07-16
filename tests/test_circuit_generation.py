from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.source_oracle import _oracle_imports
from tests.fixtures import compiled_model_or_skip
from tools.check_circuit_generation import _source_greedy_generate


GENERATION_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(GENERATION_DEPS_AVAILABLE, "generation oracle dependencies are not installed")
class CircuitGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, cls.auto_model, cls.dynamic_cache = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            package_dir=cls.fixture.package_dir,
            torch=cls.torch,
        )
        cls.source = cls.auto_model.from_pretrained(cls.fixture.source_model_dir, dtype=cls.torch.float32)
        cls.source.eval()

    def test_greedy_generation_matches_source_oracle(self) -> None:
        prompt_ids = (1,)
        max_new_tokens = 4
        eos_token_id = int(self.runtime.config["eos_token_id"])

        with self.torch.no_grad():
            circuit = self.runtime.generate(prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
            source = _source_greedy_generate(
                torch=self.torch,
                source=self.source,
                dynamic_cache=self.dynamic_cache,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )

        self.assertEqual(tuple(source["generated_ids"]), circuit.generated_ids)
        self.assertEqual(tuple(source["output_ids"]), circuit.output_ids)
        self.assertEqual(source["stop_reason"], circuit.stop_reason)
        self.assertEqual(max_new_tokens, len(circuit.generated_steps))

    def test_generation_feeds_tokens_back_through_same_stream_state(self) -> None:
        prompt_ids = (1,)
        max_new_tokens = 4

        with self.torch.no_grad():
            circuit = self.runtime.generate(prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=None)

        total_ticks = len(prompt_ids) + len(circuit.generated_ids)
        last_steps = circuit.generated_steps[-1].tick.output.steps
        self.assertTrue(all(step.implementation.startswith("executable_") for step in last_steps))

        last_attention = next(step.state for step in last_steps if step.operator_type == "full_attention")
        self.assertEqual("pedal", last_attention["owner"])
        self.assertEqual("layer_02", last_attention["pedal_id"])
        self.assertEqual("kv_memory", last_attention["state_id"])
        self.assertEqual([1, 8, total_ticks, 64], last_attention["source_key_shape"])
        self.assertEqual([1, 8, total_ticks, 64], last_attention["source_value_shape"])
        self.assertEqual(total_ticks, last_attention["updates"])


if __name__ == "__main__":
    unittest.main()
