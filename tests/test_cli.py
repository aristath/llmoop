from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from nerve.cli import (
    build_runtime_command,
    resolve_runtime_package_manifest,
)
from nerve.model_package import copy_shader_templates
from tests.fixtures import compiled_model_or_skip


def runtime_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "prompt": None,
        "chat": False,
        "chat_template_var": [],
        "inspect_runtime": False,
        "inspect_package": False,
        "inspect_graph": False,
        "inspect_placement": False,
        "inspect_device_slice": None,
        "device": None,
        "place_node": [],
        "bind_device": [],
        "duplicate_after": [],
        "chain": None,
        "max_new_tokens": 4,
        "speculative_draft_tokens": 0,
        "context_size": None,
        "vulkan_device_index": None,
        "seed": 0,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "min_p": None,
        "presence_penalty": None,
        "repetition_penalty": None,
        "no_special_tokens": False,
        "keep_special_tokens": False,
        "generated_only": False,
        "json": False,
        "runtime_bin": Path("/tmp/nerve-runtime"),
    }
    values.update(overrides)
    return Namespace(**values)


class RuntimeCliCommandTest(unittest.TestCase):
    def test_build_runtime_command_forwards_model_chat_template_variables(self) -> None:
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(prompt="Hello", seed=42)

        self.assertIn(
            ["--seed", "42"],
            [
                build_runtime_command(args, package)[index : index + 2]
                for index in range(len(build_runtime_command(args, package)) - 1)
            ],
        )

    def test_build_runtime_command_forwards_sampler_overrides(self) -> None:
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            min_p=0.02,
            presence_penalty=1.5,
            repetition_penalty=1.05,
        )

        command = build_runtime_command(args, package)

        for expected in (
            ["--temperature", "1.0"],
            ["--top-k", "20"],
            ["--top-p", "0.95"],
            ["--min-p", "0.02"],
            ["--presence-penalty", "1.5"],
            ["--repetition-penalty", "1.05"],
        ):
            self.assertIn(
                expected,
                [command[index : index + 2] for index in range(len(command) - 1)],
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_node=[],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            chat=True,
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_node=[],
            bind_device=[],
            duplicate_after=[],
            chain=None,
            max_new_tokens=4,
            context_size=None,
            vulkan_device_index=None,
            no_special_tokens=False,
            keep_special_tokens=False,
            generated_only=False,
            json=False,
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--chat",
                "--max-new-tokens",
                "4",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_forwards_explicit_mtp_window(self) -> None:
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(prompt="Hello", speculative_draft_tokens=5)

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
                "--speculative-draft-tokens",
                "5",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_device_slice_without_prompt(
        self,
    ) -> None:
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=False,
            inspect_placement=False,
            inspect_device_slice="gpu1",
            device=None,
            place_node=[],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=False,
            inspect_placement=True,
            inspect_device_slice=None,
            device=None,
            place_node=[],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=True,
            inspect_graph=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_node=[],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--inspect-package",
                "--vulkan-device-index",
                "5",
                "--json",
            ],
            build_runtime_command(args, package),
        )

    def test_build_runtime_command_can_inspect_graph_without_prompt(self) -> None:
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=True,
            inspect_placement=False,
            inspect_device_slice=None,
            device=None,
            place_node=["layer_05_repeat=gpu1"],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--inspect-graph",
                "--place-node",
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=True,
            inspect_placement=False,
            inspect_device_slice=None,
            device="gpu0",
            place_node=["layer_01=cpu0"],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--inspect-graph",
                "--device",
                "gpu0",
                "--place-node",
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
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt=None,
            inspect_runtime=True,
            inspect_package=False,
            inspect_graph=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device="gpu0",
            place_node=["layer_05_repeat=gpu1"],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--inspect-runtime",
                "--device",
                "gpu0",
                "--place-node",
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

    def test_build_runtime_command_forwards_runtime_graph_overrides(self) -> None:
        package = Path("compiled_models/model_x/vulkan_resident_package.json")
        args = runtime_args(
            prompt="Hello",
            inspect_runtime=False,
            inspect_package=False,
            inspect_graph=False,
            inspect_placement=False,
            inspect_device_slice=None,
            device="gpu0",
            place_node=["layer_02=gpu1", "layer_07=lan:worker-a"],
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
            runtime_bin=Path("/tmp/nerve-runtime"),
        )

        self.assertEqual(
            [
                "/tmp/nerve-runtime",
                "--package",
                str(package),
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
                "--device",
                "gpu0",
                "--place-node",
                "layer_02=gpu1",
                "--place-node",
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
    def test_compiled_package_contains_both_runtime_sampler_families(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())
        runtime_roles = {
            kernel["role"]
            for kernel in manifest["sampler"]["kernels"]
            if kernel["role"].startswith("runtime_")
        }

        self.assertEqual(
            {
                "runtime_record_current_token",
                "runtime_record_token_batch",
                "runtime_sample_logits",
                "runtime_partition_top_k",
                "runtime_sample_candidates",
            },
            runtime_roles,
        )
        self.assertGreaterEqual(manifest["sampler"]["spec"]["top_k_capacity"], 1)
        self.assertGreater(manifest["sampler"]["spec"]["scratch_byte_capacity"], 0)

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
            self.assertEqual("row_major", info["layout"])

    def test_compiled_package_declares_component_executions(self) -> None:
        fixture = compiled_model_or_skip()
        manifest = json.loads(fixture.package_manifest.read_text())

        self.assertNotIn("reusable_kernel_shaders", manifest)
        executions = manifest["component_executions"]
        processor_components = [
            component
            for component in manifest["circuit_graph"]["components"]
            if component["runtime_role"] == "signal_processor"
        ]
        self.assertTrue(executions)
        self.assertEqual(len(processor_components), len(executions))
        execution_by_component = {
            execution["component_id"]: execution for execution in executions
        }
        self.assertEqual(
            {component["component_id"] for component in processor_components},
            set(execution_by_component),
        )
        for component in processor_components:
            execution = execution_by_component[component["component_id"]]
            nodes = component["circuit"]["nodes"]
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
                    kernel["execution_domain"],
                    {"decode", "decode_and_prefill"},
                )
                self.assertIn(
                    kernel["batch_mode"],
                    {"serial_lanes", "weight_shared", "causal_scan"},
                )
                if kernel["batch_mode"] in {"weight_shared", "causal_scan"}:
                    self.assertGreaterEqual(len(kernel["batch_implementations"]), 1)
                    for implementation in kernel["batch_implementations"]:
                        self.assertIn(
                            implementation["execution_domain"],
                            {"decode", "prefill", "decode_and_prefill"},
                        )
                        self.assertGreater(implementation["lane_tile_width"], 0)
                        self.assertIsInstance(
                            implementation["exact_primary_equivalence"], bool
                        )
                        self.assertIsInstance(
                            implementation["exact_causal_sequence_equivalence"], bool
                        )
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
            manifest["output_transducer"]["embedding_norm_batch_shader_path"],
            manifest["output_transducer"]["projection_shader_path"],
            manifest["output_transducer"]["projection_batch_shader_path"],
            *(kernel["shader_path"] for kernel in manifest["sampler"]["kernels"]),
            *(
                kernel["shader_path"]
                for execution in manifest["component_executions"]
                for kernel in execution["kernels"]
            ),
            *(
                stage["shader_path"]
                for execution in manifest["component_executions"]
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
        self.assertEqual("explicit_graph", circuit_graph["topology"])
        roles = [component["runtime_role"] for component in circuit_graph["components"]]
        self.assertEqual(1, roles.count("input_transducer"))
        self.assertEqual(1, roles.count("output_transducer"))
        self.assertEqual(1, roles.count("sampler"))
        self.assertGreaterEqual(roles.count("signal_processor"), 1)
        self.assertEqual(len(circuit_graph["components"]), len(circuit_graph["edges"]))
        self.assertEqual(
            1,
            sum(
                edge["connection"]["kind"] == "temporal_feedback"
                for edge in circuit_graph["edges"]
            ),
        )
        for component in circuit_graph["components"]:
            self.assertEqual("nerve.circuit_params.v1", component["params"]["schema"])
            self.assertEqual("nerve.circuit_state.v1", component["state"]["schema"])
        behavioral = json.loads(
            (fixture.package_dir / manifest["behavioral_validation_path"]).read_text()
        )
        self.assertEqual("passed", behavioral["status"])
        self.assertEqual("exact_reference", behavioral["candidate_kind"])
        self.assertEqual(len(circuit_graph["components"]), len(behavioral["circuits"]))

    def test_compiled_model_does_not_reference_source_paths(
        self,
    ) -> None:
        fixture = compiled_model_or_skip()

        for artifact in fixture.compiled_model_dir.rglob("*.json"):
            payload = artifact.read_text()
            self.assertNotIn(str(fixture.source_model_dir), payload, artifact)
            self.assertNotIn("source_model_dir", payload, artifact)

    def test_compiled_model_contains_runtime_and_intermediate_artifacts(self) -> None:
        fixture = compiled_model_or_skip()

        self.assertEqual(fixture.compiled_model_dir, fixture.package_dir)
        self.assertEqual(fixture.package_dir, fixture.package_manifest.parent)
        self.assertEqual(
            fixture.transpiled_dir, fixture.compiled_model_dir / "transpiled"
        )
        self.assertEqual(fixture.lowered_dir, fixture.compiled_model_dir / "lowered")
        self.assertTrue((fixture.transpiled_dir / "model.json").is_file())
        self.assertTrue((fixture.transpiled_dir / "tensors.json").is_file())
        self.assertTrue((fixture.lowered_dir / "execution_graph.circuits.json").is_file())
        self.assertTrue((fixture.package_dir / "weights").is_dir())
        self.assertTrue((fixture.package_dir / "shaders").is_dir())
        self.assertTrue((fixture.package_dir / "tokenizer").is_dir())
        self.assertTrue(fixture.package_manifest.is_file())


if __name__ == "__main__":
    unittest.main()
