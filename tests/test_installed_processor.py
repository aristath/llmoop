from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.device_backend import PythonDeviceBackend
from llmoop.installed_processor import InstalledStreamProcessor
from llmoop.source_oracle import _oracle_imports
from llmoop.stream_processor import StreamProcessor
from tests.fixtures import compiled_model_or_skip


INSTALLED_PROCESSOR_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(INSTALLED_PROCESSOR_DEPS_AVAILABLE, "installed processor tests require torch, transformers, and safetensors")
class InstalledProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()
        cls.torch, _, _ = _oracle_imports()
        cls.runtime = CircuitModelRuntime.from_dirs(
            circuit_dir=cls.fixture.lowered_dir,
            package_dir=cls.fixture.package_dir,
            torch=cls.torch,
        )

    def test_run_prompt_uses_device_owned_feedback_loop(self) -> None:
        installed = InstalledStreamProcessor.from_runtime(self.runtime)
        reference = StreamProcessor(runtime=self.runtime)

        with self.torch.no_grad():
            run = installed.run_prompt(
                stream_id="s0",
                prompt_ids=(1,),
                max_new_tokens=4,
                eos_token_id=None,
            )
            expected = reference.generate(
                prompt_ids=(1,),
                max_new_tokens=4,
                eos_token_id=None,
            )

        self.assertEqual(expected.generated_ids, run.generated_ids)
        self.assertEqual(expected.output_ids, run.output_ids)
        self.assertEqual("idle", run.dispatch.status)
        self.assertEqual(5, len(run.dispatch.ticks))
        self.assertEqual(4, len(run.outputs))
        self.assertEqual(4, len(installed.drain_outputs()))

    def test_host_controls_route_through_installed_device(self) -> None:
        installed = InstalledStreamProcessor.from_runtime(self.runtime)

        with self.torch.no_grad():
            installed.inject_prompt("s0", prompt_ids=(1,), max_new_tokens=4, eos_token_id=None)
            installed.dispatch(max_ticks=1)
            stop_event = installed.stop_after_current("s0", reason="host_cut_signal")
            run = installed.run_until_idle()

        self.assertEqual("control_stop_after_current", stop_event.type)
        self.assertEqual("idle", run.status)
        self.assertEqual(1, len(installed.drain_outputs()))
        stream = installed.get_stream("s0")
        self.assertEqual("host_cut_signal", stream.last_stop_reason)
        self.assertEqual([], stream.private_feedback_queue)

    def test_installation_manifest_names_ports_and_transient_template(self) -> None:
        installed = InstalledStreamProcessor.from_runtime(self.runtime, device_id="gpuish_0")
        manifest = installed.to_json()

        self.assertIsInstance(installed.device, PythonDeviceBackend)
        self.assertEqual("python_device_loop", installed.device.backend_id)
        self.assertEqual("python_device_loop", manifest["backend"])
        self.assertEqual("device_owned_insert_loop", manifest["host_ports"]["private_feedback"])
        self.assertIn("external_input", manifest["host_ports"]["inputs"])
        self.assertIn("public_output", manifest["host_ports"]["outputs"])
        self.assertEqual(14, manifest["permanent_circuit"]["pedal_count"])
        self.assertNotIn("source_model_dir", manifest["permanent_circuit"])
        self.assertEqual("gpuish_0", manifest["device"]["device_id"])
        self.assertEqual("python_device_loop", manifest["device"]["backend_id"])
        self.assertEqual(14, len(manifest["stream_template"]["state_allocations"]))


if __name__ == "__main__":
    unittest.main()
