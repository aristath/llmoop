from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from llmoop.cli import (
    build_runtime_command,
    parse_pedal_device_overrides,
    resolve_runtime_package_manifest,
)
from llmoop.model_compiler import ModelCompileError
from llmoop.model_package import package_placement
from tests.fixtures import compiled_model_or_skip


CLI_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)


class RuntimeCliCommandTest(unittest.TestCase):
    def test_resolve_runtime_package_manifest_accepts_package_dir_or_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest = root / "vulkan_resident_greedy_package.json"
            manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(manifest, resolve_runtime_package_manifest(root))
            self.assertEqual(manifest, resolve_runtime_package_manifest(manifest))

    def test_build_runtime_command_prefers_explicit_runtime_binary(self) -> None:
        package = Path("packages/model_x/vulkan_resident_greedy_package.json")
        args = Namespace(
            prompt="Hello",
            inspect_placement=False,
            inspect_device_slice=None,
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

    def test_build_runtime_command_can_inspect_device_slice_without_prompt(self) -> None:
        package = Path("packages/model_x/vulkan_resident_greedy_package.json")
        args = Namespace(
            prompt=None,
            inspect_placement=False,
            inspect_device_slice="gpu1",
            max_new_tokens=4,
            capacity=4,
            no_special_tokens=False,
            keep_special_tokens=False,
            generated_only=False,
            json=True,
            runtime_bin=Path("/tmp/llmoop-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/llmoop-runtime",
                "--package",
                str(package),
                "--inspect-device-slice",
                "gpu1",
                "--capacity",
                "4",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_placement_without_prompt(self) -> None:
        package = Path("packages/model_x/vulkan_resident_greedy_package.json")
        args = Namespace(
            prompt=None,
            inspect_placement=True,
            inspect_device_slice=None,
            max_new_tokens=4,
            capacity=4,
            no_special_tokens=False,
            keep_special_tokens=False,
            generated_only=False,
            json=True,
            runtime_bin=Path("/tmp/llmoop-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/llmoop-runtime",
                "--package",
                str(package),
                "--inspect-placement",
                "--capacity",
                "4",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_parse_pedal_device_overrides_requires_explicit_pedal_device_pairs(self) -> None:
        self.assertEqual(
            {"layer_02": "gpu1", "layer_03": "lan:worker-a"},
            parse_pedal_device_overrides(["layer_02=gpu1", " layer_03 = lan:worker-a "]),
        )

        with self.assertRaisesRegex(ValueError, "expected PEDAL=DEVICE"):
            parse_pedal_device_overrides(["layer_02"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_pedal_device_overrides(["layer_02=gpu1", "layer_02=gpu2"])


class PackagePlacementTest(unittest.TestCase):
    def test_package_placement_records_default_device_and_overrides(self) -> None:
        lowered_index = {"graph": {"circuits": [{"id": "layer_00"}, {"id": "layer_02"}]}}

        self.assertEqual(
            {
                "schema": "llmoop.stream_circuit_placement.v1",
                "default_device_id": "gpu0",
                "pedal_devices": {"layer_00": "cpu0", "layer_02": "gpu1"},
            },
            package_placement(
                lowered_index,
                default_device_id="gpu0",
                pedal_devices={"layer_02": "gpu1", "layer_00": "cpu0"},
            ),
        )

    def test_package_placement_rejects_unknown_pedals(self) -> None:
        lowered_index = {"graph": {"circuits": [{"id": "layer_00"}]}}

        with self.assertRaisesRegex(ModelCompileError, "unknown pedal 'layer_99'"):
            package_placement(
                lowered_index,
                default_device_id="gpu0",
                pedal_devices={"layer_99": "gpu1"},
            )


class CompiledPackageTest(unittest.TestCase):
    def test_compiled_package_contains_tokenizer_files(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())

        self.assertEqual("tokenizer", manifest["tokenizer"]["path"])
        self.assertEqual("config.json", manifest["config_path"])
        self.assertTrue((fixture.package_dir / "config.json").is_file())
        self.assertIn("tokenizer.json", manifest["tokenizer"]["files"])
        self.assertTrue((fixture.package_dir / "tokenizer" / "tokenizer.json").is_file())

    def test_compiled_package_contains_weight_files_and_local_tensor_index(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())
        tensor_index_path = fixture.package_dir / manifest["tensor_index_path"]
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
            self.assertTrue((fixture.package_dir / source_file).is_file())

    def test_compiled_package_declares_pedal_executions(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())

        self.assertNotIn("reusable_kernel_shaders", manifest)
        executions = manifest["pedal_executions"]
        self.assertEqual(14, len(executions))
        layer_00 = executions[0]
        self.assertEqual("layer_00", layer_00["pedal_id"])
        self.assertEqual("conv", layer_00["operator_type"])
        self.assertEqual(
            [
                "operator_norm",
                "conv_in_projection",
                "split_b_c_x",
                "input_gate",
                "temporal_memory_update",
                "depthwise_temporal_conv",
                "output_gate",
                "conv_out_projection",
                "operator_residual",
                "ffn_norm",
                "ffn_gate_projection",
                "ffn_up_projection",
                "ffn_gate_activation",
                "ffn_gate_multiply",
                "ffn_down_projection",
                "ffn_residual",
            ],
            [kernel["node_id"] for kernel in layer_00["kernels"]],
        )
        self.assertEqual(list(range(16)), [kernel["execution_index"] for kernel in layer_00["kernels"]])
        for profile in manifest["capacity_profiles"]:
            self.assertIn("pedal_execution_shader_overrides", profile)
            self.assertNotIn("reusable_kernel_shader_overrides", profile)

    def test_compiled_package_embeds_runtime_circuit_graph(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())

        self.assertNotIn("circuit_index_path", manifest)
        self.assertEqual(
            {
                "schema": "llmoop.stream_circuit_placement.v1",
                "default_device_id": "gpu0",
                "pedal_devices": {},
            },
            manifest["placement"],
        )
        circuit_graph = manifest["circuit_graph"]
        self.assertEqual("series", circuit_graph["wiring"])
        self.assertEqual(14, len(circuit_graph["pedals"]))
        layer_00 = circuit_graph["pedals"][0]
        self.assertEqual("layer_00", layer_00["pedal_id"])
        self.assertEqual("conv", layer_00["operator_type"])
        self.assertEqual("layer_00_shortconv_circuit_v1", layer_00["circuit"]["id"])
        self.assertEqual("llmoop.circuit_params.v1", layer_00["params"]["schema"])
        self.assertEqual("llmoop.circuit_state.v1", layer_00["state"]["schema"])

    def test_compiled_package_does_not_reference_source_or_transpiled_paths(self) -> None:
        fixture = compiled_model_or_skip()

        for root in (fixture.lowered_dir, fixture.package_dir):
            for artifact in root.rglob("*.json"):
                payload = artifact.read_text()
                self.assertNotIn(str(fixture.source_model_dir), payload, artifact)
                self.assertNotIn("transpiled/", payload, artifact)
                self.assertNotIn("source_model_dir", payload, artifact)

    def test_runtime_package_is_separate_from_lowered_workspace(self) -> None:
        fixture = compiled_model_or_skip()

        self.assertEqual(fixture.package_dir, fixture.package_manifest.parent)
        self.assertNotEqual(fixture.lowered_dir, fixture.package_dir)
        self.assertFalse((fixture.lowered_dir / "vulkan_resident_greedy_package.json").exists())
        self.assertFalse(
            any(
                artifact.name == "vulkan_resident_greedy_package.json"
                for artifact in fixture.lowered_dir.rglob("*.json")
            )
        )
        self.assertTrue((fixture.lowered_dir / "pedalboard.circuits.json").is_file())
        self.assertTrue(fixture.package_manifest.is_file())


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
                "--package-dir",
                str(self.fixture.package_dir),
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
