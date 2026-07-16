from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from llmoop.cli import build_runtime_command, resolve_runtime_package_manifest
from tests.fixtures import compiled_model_or_skip


CLI_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)


class RuntimeCliCommandTest(unittest.TestCase):
    def test_resolve_runtime_package_manifest_accepts_lowered_dir_or_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest = root / "vulkan_resident_greedy_package.json"
            manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(manifest, resolve_runtime_package_manifest(root))
            self.assertEqual(manifest, resolve_runtime_package_manifest(manifest))

    def test_build_runtime_command_prefers_explicit_runtime_binary(self) -> None:
        package = Path("lowered/model_x/vulkan_resident_greedy_package.json")
        args = Namespace(
            prompt="Hello",
            max_new_tokens=4,
            capacity=8,
            no_special_tokens=True,
            keep_special_tokens=True,
            generated_only=True,
            json=True,
            runtime_bin=Path("/tmp/llmoop-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/llmoop-runtime",
                "--package",
                str(package),
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
                "--capacity",
                "8",
                "--no-special-tokens",
                "--keep-special-tokens",
                "--generated-only",
                "--json",
            ],
            build_runtime_command(args, package),
        )


class CompiledPackageTest(unittest.TestCase):
    def test_compiled_package_contains_tokenizer_files(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())

        self.assertEqual("tokenizer", manifest["tokenizer"]["path"])
        self.assertEqual("config.json", manifest["config_path"])
        self.assertTrue((fixture.lowered_dir / "config.json").is_file())
        self.assertIn("tokenizer.json", manifest["tokenizer"]["files"])
        self.assertTrue((fixture.lowered_dir / "tokenizer" / "tokenizer.json").is_file())

    def test_compiled_package_contains_weight_files_and_local_tensor_index(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())
        tensor_index_path = fixture.lowered_dir / manifest["tensor_index_path"]
        tensor_index = json.loads(tensor_index_path.read_text())

        self.assertEqual("tensors.json", manifest["tensor_index_path"])
        self.assertEqual("weights", tensor_index["source"]["weights_dir"])
        self.assertTrue(tensor_index["source"]["packaged"])
        self.assertNotIn("model_dir", tensor_index["source"])
        self.assertFalse(Path(tensor_index["source"]["weights_file"]).is_absolute())
        for source_record in tensor_index["source"]["weights_files"]:
            self.assertFalse(Path(source_record["path"]).is_absolute())
        for info in tensor_index["tensors"].values():
            source_file = Path(info["source_file"])
            self.assertFalse(source_file.is_absolute())
            self.assertEqual("weights", source_file.parts[0])
            self.assertTrue((fixture.lowered_dir / source_file).is_file())

    def test_compiled_package_does_not_reference_source_or_transpiled_paths(self) -> None:
        fixture = compiled_model_or_skip()

        for artifact in fixture.lowered_dir.rglob("*.json"):
            payload = artifact.read_text()
            self.assertNotIn(str(fixture.source_model_dir), payload, artifact)
            self.assertNotIn("transpiled/", payload, artifact)
            self.assertNotIn("source_model_dir", payload, artifact)


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
