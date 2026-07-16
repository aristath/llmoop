from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.source_oracle import _oracle_imports
from llmoop.text_generation import generate_text, load_tokenizer
from tests.fixtures import compiled_model_or_skip
from tools.check_circuit_generation import _source_greedy_generate


TEXT_GENERATION_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(TEXT_GENERATION_DEPS_AVAILABLE, "text generation dependencies are not installed")
class TextGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, cls.auto_model, cls.dynamic_cache = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            model_dir=cls.fixture.source_model_dir,
            torch=cls.torch,
        )
        cls.tokenizer = load_tokenizer(cls.fixture.source_model_dir)
        cls.source = cls.auto_model.from_pretrained(cls.fixture.source_model_dir, dtype=cls.torch.float32)
        cls.source.eval()

    def test_text_prompt_generation_matches_source_oracle(self) -> None:
        prompt_text = "Hello"
        max_new_tokens = 4
        eos_token_id = int(self.runtime.config["eos_token_id"])
        prompt_ids = tuple(self.tokenizer.encode(prompt_text, add_special_tokens=True))

        with self.torch.no_grad():
            circuit = generate_text(
                runtime=self.runtime,
                tokenizer=self.tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            source = _source_greedy_generate(
                torch=self.torch,
                source=self.source,
                dynamic_cache=self.dynamic_cache,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )

        self.assertEqual(prompt_ids, circuit.prompt_ids)
        self.assertEqual(tuple(source["generated_ids"]), circuit.generated_ids)
        self.assertEqual(tuple(source["output_ids"]), circuit.output_ids)
        self.assertEqual(
            self.tokenizer.decode(source["generated_ids"], skip_special_tokens=True),
            circuit.generated_text,
        )
        self.assertEqual(
            self.tokenizer.decode(source["output_ids"], skip_special_tokens=True),
            circuit.output_text,
        )


if __name__ == "__main__":
    unittest.main()
