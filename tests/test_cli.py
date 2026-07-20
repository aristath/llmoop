from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from llmoop.cli import (
    build_runtime_command,
    resolve_runtime_package_manifest,
)
from llmoop.model_package import copy_shader_templates
from tests.fixtures import compiled_model_or_skip


def runtime_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "prompt": None,
        "chat": False,
        "chat_template_var": [],
        "inspect_runtime": False,
        "inspect_package": False,
        "inspect_patch": False,
        "inspect_placement": False,
        "inspect_device_slice": None,
        "device": None,
        "place_pedal": [],
        "bind_device": [],
        "duplicate_after": [],
        "chain": None,
        "max_new_tokens": 4,
        "context_size": None,
        "vulkan_device_index": None,
        "seed": 0,
        "no_special_tokens": False,
        "keep_special_tokens": False,
        "generated_only": False,
        "profile": False,
        "profile_runs": 1,
        "json": False,
        "runtime_bin": Path("/tmp/llmoop-runtime"),
    }
    values.update(overrides)
    return Namespace(**values)


class RuntimeCliCommandTest(unittest.TestCase):
    def test_build_runtime_command_forwards_model_chat_template_variables(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            chat=True,
            chat_template_var=[
                "enable_thinking=false",
                "preserve_thinking=true",
            ],
        )

        command = build_runtime_command(args, package)

        self.assertIn(
            ["--chat-template-var", "enable_thinking=false"],
            [command[index : index + 2] for index in range(len(command) - 1)],
        )
        self.assertIn(
            ["--chat-template-var", "preserve_thinking=true"],
            [command[index : index + 2] for index in range(len(command) - 1)],
        )

    def test_build_runtime_command_forwards_non_default_random_seed(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(prompt="Hello", seed=42)

        self.assertIn(
            ["--seed", "42"],
            [
                build_runtime_command(args, package)[index : index + 2]
                for index in range(len(build_runtime_command(args, package)) - 1)
            ],
        )

    def test_model_compiler_renders_linear_shader_for_discovered_shape(self) -> None:
        shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
        with tempfile.TemporaryDirectory() as raw_dest:
            destination = Path(raw_dest)
            copy_shader_templates(
                shader_source_dir,
                destination,
                {"linear_bf16_768x2048.comp"},
            )
            shader = (destination / "linear_bf16_768x2048.comp").read_text()

        self.assertIn("const uint INPUT_SIZE = 768u;", shader)
        self.assertIn("const uint OUTPUT_SIZE = 2048u;", shader)
        self.assertIn("uint word_index = gl_WorkGroupID.x;", shader)
        self.assertNotIn("{{INPUT_SIZE}}", shader)

    def test_resolve_runtime_package_manifest_accepts_package_dir_or_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest = root / "vulkan_resident_package.json"
            manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(manifest, resolve_runtime_package_manifest(root))
            self.assertEqual(manifest, resolve_runtime_package_manifest(manifest))

    def test_build_runtime_command_prefers_explicit_runtime_binary(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=8,
            vulkan_device_index=None,
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
                "--context-size",
                "8",
                "--no-special-tokens",
                "--keep-special-tokens",
                "--generated-only",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_forwards_chat_mode_without_prompt(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            chat=True,
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
            no_special_tokens=False,
            keep_special_tokens=False,
            generated_only=False,
            profile=False,
            profile_runs=1,
            json=False,
            runtime_bin=Path("/tmp/llmoop-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/llmoop-runtime",
                "--package",
                str(package),
                "--chat",
                "--max-new-tokens",
                "4",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_forwards_profile_flag(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
            no_special_tokens=False,
            keep_special_tokens=False,
            generated_only=False,
            profile=True,
            json=False,
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
                "--profile",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_forwards_profile_runs(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
            no_special_tokens=False,
            keep_special_tokens=False,
            generated_only=False,
            profile=False,
            profile_runs=3,
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
                "--profile-runs",
                "3",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_device_slice_without_prompt(
        self,
    ) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice="gpu1",
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=4,
            vulkan_device_index=None,
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
                "--context-size",
                "4",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_placement_without_prompt(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=True,
            inspect_device_slice=None,
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=4,
            vulkan_device_index=None,
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
                "--context-size",
                "4",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_package_without_prompt(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=True,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_pedal=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=5,
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
                "--inspect-package",
                "--vulkan-device-index",
                "5",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_patch_without_prompt(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=True,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_pedal=["layer_05_repeat=gpu1"],
            bind_device=["gpu1=vulkan:5"],
            duplicate_after=[],
            chain="layer_00,layer_05_repeat=layer_05,layer_13",
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
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
                "--inspect-patch",
                "--place-pedal",
                "layer_05_repeat=gpu1",
                "--bind-device",
                "gpu1=vulkan:5",
                "--chain",
                "layer_00,layer_05_repeat=layer_05,layer_13",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_preserves_cpu_logical_placement(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=True,
            inspect_placement=False,
            inspect_device_slice=None,
            device="gpu0",
            place_pedal=["layer_01=cpu0"],
            bind_device=[],
            duplicate_after=[],
            chain="layer_00,layer_01,layer_02",
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
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
                "--inspect-patch",
                "--device",
                "gpu0",
                "--place-pedal",
                "layer_01=cpu0",
                "--chain",
                "layer_00,layer_01,layer_02",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_runtime_topology_without_prompt(
        self,
    ) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=True,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device="gpu0",
            place_pedal=["layer_05_repeat=gpu1"],
            bind_device=["gpu0=vulkan:5", "gpu1=vulkan:5"],
            duplicate_after=[],
            chain="layer_00,layer_05_repeat=layer_05,layer_13",
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
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
                "--inspect-runtime",
                "--device",
                "gpu0",
                "--place-pedal",
                "layer_05_repeat=gpu1",
                "--bind-device",
                "gpu0=vulkan:5",
                "--bind-device",
                "gpu1=vulkan:5",
                "--chain",
                "layer_00,layer_05_repeat=layer_05,layer_13",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_forwards_runtime_patch_overrides(self) -> None:
        package = Path("packages/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            inspect_runtime=False,
            inspect_package=False,
            inspect_patch=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device="gpu0",
            place_pedal=["layer_02=gpu1", "layer_07=lan:worker-a"],
            bind_device=["gpu0=vulkan:0", "gpu1=vulkan:5"],
            duplicate_after=["layer_05=layer_05_repeat"],
            chain="layer_00,layer_01,layer_05,layer_05_repeat=layer_05,layer_06",
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
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
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
                "--device",
                "gpu0",
                "--place-pedal",
                "layer_02=gpu1",
                "--place-pedal",
                "layer_07=lan:worker-a",
                "--bind-device",
                "gpu0=vulkan:0",
                "--bind-device",
                "gpu1=vulkan:5",
                "--duplicate-after",
                "layer_05=layer_05_repeat",
                "--chain",
                "layer_00,layer_01,layer_05,layer_05_repeat=layer_05,layer_06",
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
        self.assertTrue((fixture.package_dir / "config.json").is_file())
        self.assertIn("tokenizer.json", manifest["tokenizer"]["files"])
        self.assertTrue(
            (fixture.package_dir / "tokenizer" / "tokenizer.json").is_file()
        )

    def test_compiled_package_contains_weight_files_and_local_tensor_index(
        self,
    ) -> None:
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
        processor_pedals = [
            pedal
            for pedal in manifest["circuit_graph"]["pedals"]
            if pedal["runtime_role"] == "signal_processor"
        ]
        self.assertTrue(executions)
        self.assertEqual(len(processor_pedals), len(executions))
        execution_by_pedal = {
            execution["pedal_id"]: execution for execution in executions
        }
        self.assertEqual(
            {pedal["pedal_id"] for pedal in processor_pedals},
            set(execution_by_pedal),
        )
        for pedal in processor_pedals:
            execution = execution_by_pedal[pedal["pedal_id"]]
            nodes = pedal["circuit"]["nodes"]
            self.assertEqual(
                [node["id"] for node in nodes],
                [kernel["node_id"] for kernel in execution["kernels"]],
            )
            self.assertEqual(
                list(range(len(execution["kernels"]))),
                [kernel["execution_index"] for kernel in execution["kernels"]],
            )
            self.assertEqual(
                [node["op"] for node in nodes],
                [kernel["op"] for kernel in execution["kernels"]],
            )
            for node, kernel in zip(nodes, execution["kernels"], strict=True):
                self.assertIn(
                    kernel["batch_mode"],
                    {"serial_lanes", "weight_shared", "causal_scan"},
                )
                if kernel["batch_mode"] in {"weight_shared", "causal_scan"}:
                    self.assertGreaterEqual(len(kernel["batch_implementations"]), 1)
                    for implementation in kernel["batch_implementations"]:
                        self.assertGreater(implementation["lane_tile_width"], 0)
                        self.assertIn("device_requirements", implementation)
                        self.assertGreaterEqual(len(implementation["stages"]), 1)
                        for stage in implementation["stages"]:
                            self.assertTrue(stage["shader_path"].endswith(".spv"))
                            self.assertGreater(stage["local_size_x"], 0)
                            self.assertGreater(stage["workgroup_count_x"], 0)
                else:
                    self.assertEqual([], kernel["batch_implementations"])
                if node["op"] in {
                    "scaled_dot_product_attention",
                    "append_scaled_dot_product_attention",
                }:
                    attrs = (
                        node["attrs"]["attention"]
                        if node["op"] == "append_scaled_dot_product_attention"
                        else node["attrs"]
                    )
                    self.assertEqual(
                        int(attrs["query_heads"]), kernel["workgroup_count_x"]
                    )
                    self.assertNotIn("_cap", kernel["shader_path"])
        self.assertNotIn("capacity_profiles", manifest)
        self.assertTrue(
            all(
                kernel["workgroup_count_x"] >= 1
                for execution in executions
                for kernel in execution["kernels"]
            )
        )

    def test_compiled_package_contains_only_precompiled_spirv_shaders(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())
        shader_paths = [
            manifest["input_transducer"]["shader_path"],
            manifest["input_transducer"]["batch_shader_path"],
            manifest["output_transducer"]["embedding_norm_shader_path"],
            manifest["output_transducer"]["projection_shader_path"],
            manifest["output_transducer"]["projection_batch_shader_path"],
            *(kernel["shader_path"] for kernel in manifest["sampler"]["kernels"]),
            *(
                kernel["shader_path"]
                for execution in manifest["pedal_executions"]
                for kernel in execution["kernels"]
            ),
            *(
                stage["shader_path"]
                for execution in manifest["pedal_executions"]
                for kernel in execution["kernels"]
                for implementation in kernel["batch_implementations"]
                for stage in implementation["stages"]
            ),
        ]

        self.assertTrue(all(path.endswith(".spv") for path in shader_paths))
        self.assertTrue(
            all((fixture.package_dir / path).is_file() for path in shader_paths)
        )
        self.assertFalse(any((fixture.package_dir / "shaders").glob("*.comp")))

    def test_compiled_package_embeds_runtime_circuit_graph(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())

        self.assertNotIn("circuit_index_path", manifest)
        self.assertNotIn("placement", manifest)
        self.assertNotIn("device_id", manifest)
        circuit_graph = manifest["circuit_graph"]
        self.assertEqual("explicit_graph", circuit_graph["wiring"])
        roles = [pedal["runtime_role"] for pedal in circuit_graph["pedals"]]
        self.assertEqual(1, roles.count("input_transducer"))
        self.assertEqual(1, roles.count("output_transducer"))
        self.assertEqual(1, roles.count("sampler"))
        self.assertGreaterEqual(roles.count("signal_processor"), 1)
        self.assertEqual(len(circuit_graph["pedals"]), len(circuit_graph["cables"]))
        self.assertEqual(
            1,
            sum(
                cable["connection"]["kind"] == "temporal_feedback"
                for cable in circuit_graph["cables"]
            ),
        )
        for pedal in circuit_graph["pedals"]:
            self.assertEqual("llmoop.circuit_params.v1", pedal["params"]["schema"])
            self.assertEqual("llmoop.circuit_state.v1", pedal["state"]["schema"])
        behavioral = json.loads(
            (fixture.package_dir / manifest["behavioral_validation_path"]).read_text()
        )
        self.assertEqual("passed", behavioral["status"])
        self.assertEqual("exact_reference", behavioral["candidate_kind"])
        self.assertEqual(len(circuit_graph["pedals"]), len(behavioral["circuits"]))

    def test_compiled_package_does_not_reference_source_or_transpiled_paths(
        self,
    ) -> None:
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
        self.assertFalse(
            (fixture.lowered_dir / "vulkan_resident_package.json").exists()
        )
        self.assertFalse(
            any(
                artifact.name == "vulkan_resident_package.json"
                for artifact in fixture.lowered_dir.rglob("*.json")
            )
        )
        self.assertTrue((fixture.lowered_dir / "pedalboard.circuits.json").is_file())
        self.assertTrue(fixture.package_manifest.is_file())


if __name__ == "__main__":
    unittest.main()
