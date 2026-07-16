from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest

from tests.fixtures import compiled_model_or_skip


CLI_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)


@unittest.skipUnless(CLI_DEPS_AVAILABLE, "CLI generation dependencies are not installed")
class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()

    def test_run_model_generates_text_from_compiled_package(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llmoop",
                "--run-model",
                str(self.fixture.lowered_dir),
                "--model-dir",
                str(self.fixture.source_model_dir),
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
                "--ignore-eos",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual("Hello", payload["prompt_text"])
        self.assertGreaterEqual(len(payload["prompt_ids"]), 1)
        self.assertEqual(4, len(payload["generated_ids"]))
        self.assertEqual(payload["prompt_ids"] + payload["generated_ids"], payload["output_ids"])
        self.assertIsInstance(payload["generated_text"], str)
        self.assertIsInstance(payload["output_text"], str)
        self.assertEqual("greedy_sampler", payload["sampler"])
        self.assertEqual("max_new_tokens", payload["stop_reason"])
        self.assertEqual(4, payload["generated_count"])


if __name__ == "__main__":
    unittest.main()
