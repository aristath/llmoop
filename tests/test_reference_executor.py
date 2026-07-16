from __future__ import annotations

import importlib.util
import unittest

from llmoop.circuit_executors import (
    install_all_circuit_pedals,
    install_attention_circuit_pedals,
    install_shortconv_circuit_pedals,
)
from llmoop.pedalboard import Pedalboard
from llmoop.reference_runtime import ReferencePedalExecutor
from tests.fixtures import compiled_model_or_skip


ORACLE_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(ORACLE_DEPS_AVAILABLE, "source oracle dependencies are not installed")
class ReferenceExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = compiled_model_or_skip()

    def pedalboard(self) -> Pedalboard:
        return Pedalboard.from_dir(self.fixture.transpiled_dir)

    def executor(self) -> ReferencePedalExecutor:
        return ReferencePedalExecutor.from_model_dir(
            pedalboard=self.pedalboard(),
            model_dir=self.fixture.source_model_dir,
        )

    def test_reference_pedal_executor_matches_source_model_forward(self) -> None:
        executor = self.executor()

        activation = executor.activate_token()

        self.assertEqual(14, len(activation.steps))
        self.assertEqual((1, 1, 1024), tuple(activation.pedal_output_frame.tensor.shape))
        self.assertEqual((1, 1, 1024), tuple(activation.normalized_output_frame.tensor.shape))
        self.assertTrue(activation.comparison["allclose"], activation.comparison)
        self.assertLessEqual(activation.comparison["max_abs_diff"], 1e-6)

        self.assertEqual("layer_00", activation.steps[0].pedal_id)
        self.assertEqual("conv", activation.steps[0].operator_type)
        self.assertEqual([1, 1024, 3], activation.steps[0].state["source_shape"])

        first_attention = next(step for step in activation.steps if step.operator_type == "full_attention")
        self.assertEqual("layer_02", first_attention.pedal_id)
        self.assertEqual([1, 8, 1, 64], first_attention.state["source_key_shape"])
        self.assertEqual([1, 8, 1, 64], first_attention.state["source_value_shape"])

    def test_reference_pedal_stream_matches_source_incremental_mode(self) -> None:
        executor = self.executor()

        run = executor.open_stream().run_teacher_forced((1, 2, 3, 4))

        self.assertEqual(4, len(run.ticks))
        self.assertEqual((1, 4, 1024), tuple(run.output_tensor.shape))
        self.assertTrue(all(tick.incremental_comparison["allclose"] for tick in run.ticks))
        self.assertLessEqual(max(tick.incremental_comparison["max_abs_diff"] for tick in run.ticks), 1e-6)

        self.assertTrue(run.comparison["allclose"], run.comparison)
        self.assertLessEqual(run.comparison["max_abs_diff"], 1e-4)

        last_attention = next(step for step in run.ticks[-1].activation.steps if step.operator_type == "full_attention")
        self.assertEqual([1, 8, 4, 64], last_attention.state["source_key_shape"])
        self.assertEqual([1, 8, 4, 64], last_attention.state["source_value_shape"])

    def test_reusable_shortconv_circuit_can_replace_all_conv_pedals(self) -> None:
        executor = self.executor()
        installed = install_shortconv_circuit_pedals(executor, self.fixture.lowered_dir)

        activation = executor.activate_input_ids((1, 2, 3, 4))

        self.assertEqual(("layer_00", "layer_01", "layer_03", "layer_05", "layer_07", "layer_09", "layer_11", "layer_13"), installed)
        self.assertTrue(
            all(
                step.implementation == "executable_shortconv_circuit_v1"
                for step in activation.steps
                if step.operator_type == "conv"
            )
        )
        self.assertTrue(
            all(
                step.implementation == "source_transformers_layer"
                for step in activation.steps
                if step.operator_type == "full_attention"
            )
        )
        self.assertTrue(activation.comparison["allclose"], activation.comparison)
        self.assertLessEqual(activation.comparison["max_abs_diff"], 1e-5)

    def test_reusable_shortconv_circuit_stream_reuses_all_conv_transient_state(self) -> None:
        executor = self.executor()
        install_shortconv_circuit_pedals(executor, self.fixture.lowered_dir)

        run = executor.open_stream().run_teacher_forced((1, 2, 3, 4))

        self.assertTrue(
            all(
                step.implementation == "executable_shortconv_circuit_v1"
                for step in run.ticks[-1].activation.steps
                if step.operator_type == "conv"
            )
        )
        self.assertTrue(all(tick.incremental_comparison["allclose"] for tick in run.ticks))
        self.assertLessEqual(max(tick.incremental_comparison["max_abs_diff"] for tick in run.ticks), 1e-5)
        self.assertTrue(run.comparison["allclose"], run.comparison)
        self.assertLessEqual(run.comparison["max_abs_diff"], 1e-4)

    def test_reusable_attention_circuit_can_replace_all_attention_pedals(self) -> None:
        executor = self.executor()
        installed = install_attention_circuit_pedals(executor, self.fixture.lowered_dir)

        activation = executor.activate_input_ids((1, 2, 3, 4))

        self.assertEqual(("layer_02", "layer_04", "layer_06", "layer_08", "layer_10", "layer_12"), installed)
        self.assertTrue(
            all(
                step.implementation == "executable_gqa_attention_circuit_v1"
                for step in activation.steps
                if step.operator_type == "full_attention"
            )
        )
        self.assertTrue(
            all(
                step.implementation == "source_transformers_layer"
                for step in activation.steps
                if step.operator_type == "conv"
            )
        )
        self.assertTrue(activation.comparison["allclose"], activation.comparison)
        self.assertLessEqual(activation.comparison["max_abs_diff"], 1e-5)

    def test_reusable_attention_circuit_stream_reuses_all_kv_transient_state(self) -> None:
        executor = self.executor()
        install_attention_circuit_pedals(executor, self.fixture.lowered_dir)

        run = executor.open_stream().run_teacher_forced((1, 2, 3, 4))

        self.assertTrue(
            all(
                step.implementation == "executable_gqa_attention_circuit_v1"
                for step in run.ticks[-1].activation.steps
                if step.operator_type == "full_attention"
            )
        )
        self.assertTrue(all(tick.incremental_comparison["allclose"] for tick in run.ticks))
        self.assertLessEqual(max(tick.incremental_comparison["max_abs_diff"] for tick in run.ticks), 1e-5)
        self.assertTrue(run.comparison["allclose"], run.comparison)
        self.assertLessEqual(run.comparison["max_abs_diff"], 1e-4)

    def test_all_layer_pedals_can_run_as_executable_circuits(self) -> None:
        executor = self.executor()
        installed = install_all_circuit_pedals(executor, self.fixture.lowered_dir)

        activation = executor.activate_input_ids((1, 2, 3, 4))

        self.assertEqual(tuple(f"layer_{index:02d}" for index in range(14)), installed)
        self.assertTrue(all(step.implementation.startswith("executable_") for step in activation.steps))
        self.assertTrue(activation.comparison["allclose"], activation.comparison)
        self.assertLessEqual(activation.comparison["max_abs_diff"], 1e-5)

    def test_all_layer_pedals_run_as_executable_circuits_in_stream_mode(self) -> None:
        executor = self.executor()
        install_all_circuit_pedals(executor, self.fixture.lowered_dir)

        run = executor.open_stream().run_teacher_forced((1, 2, 3, 4))

        self.assertTrue(all(step.implementation.startswith("executable_") for step in run.ticks[-1].activation.steps))
        self.assertTrue(all(tick.incremental_comparison["allclose"] for tick in run.ticks))
        self.assertLessEqual(max(tick.incremental_comparison["max_abs_diff"] for tick in run.ticks), 1e-5)
        self.assertTrue(run.comparison["allclose"], run.comparison)
        self.assertLessEqual(run.comparison["max_abs_diff"], 1e-4)

    def test_all_layer_circuits_can_use_per_pedal_stream_state(self) -> None:
        executor = self.executor()
        install_all_circuit_pedals(executor, self.fixture.lowered_dir)
        executor.use_pedal_stream_state()

        activation = executor.activate_input_ids((1, 2, 3, 4))

        self.assertTrue(all(step.implementation.startswith("executable_") for step in activation.steps))
        self.assertTrue(activation.comparison["allclose"], activation.comparison)
        self.assertLessEqual(activation.comparison["max_abs_diff"], 1e-5)

        first_conv = activation.steps[0]
        first_attention = next(step for step in activation.steps if step.operator_type == "full_attention")
        self.assertEqual("pedal", first_conv.state["owner"])
        self.assertEqual("layer_00", first_conv.state["pedal_id"])
        self.assertEqual("temporal_memory", first_conv.state["state_id"])
        self.assertEqual([1, 1024, 3], first_conv.state["source_shape"])
        self.assertEqual("pedal", first_attention.state["owner"])
        self.assertEqual("layer_02", first_attention.state["pedal_id"])
        self.assertEqual("kv_memory", first_attention.state["state_id"])
        self.assertEqual([1, 8, 4, 64], first_attention.state["source_key_shape"])
        self.assertEqual([1, 8, 4, 64], first_attention.state["source_value_shape"])

    def test_per_pedal_stream_state_persists_across_stream_ticks(self) -> None:
        executor = self.executor()
        install_all_circuit_pedals(executor, self.fixture.lowered_dir)
        executor.use_pedal_stream_state()

        run = executor.open_stream().run_teacher_forced((1, 2, 3, 4))

        self.assertTrue(all(step.implementation.startswith("executable_") for step in run.ticks[-1].activation.steps))
        self.assertTrue(all(tick.incremental_comparison["allclose"] for tick in run.ticks))
        self.assertLessEqual(max(tick.incremental_comparison["max_abs_diff"] for tick in run.ticks), 1e-5)
        self.assertTrue(run.comparison["allclose"], run.comparison)
        self.assertLessEqual(run.comparison["max_abs_diff"], 1e-4)

        last_conv = run.ticks[-1].activation.steps[0]
        last_attention = next(step for step in run.ticks[-1].activation.steps if step.operator_type == "full_attention")
        self.assertEqual("pedal", last_conv.state["owner"])
        self.assertEqual("layer_00", last_conv.state["pedal_id"])
        self.assertEqual([1, 1024, 3], last_conv.state["source_shape"])
        self.assertEqual(4, last_conv.state["updates"])
        self.assertEqual("pedal", last_attention.state["owner"])
        self.assertEqual("layer_02", last_attention.state["pedal_id"])
        self.assertEqual([1, 8, 4, 64], last_attention.state["source_key_shape"])
        self.assertEqual([1, 8, 4, 64], last_attention.state["source_value_shape"])
        self.assertEqual(4, last_attention.state["updates"])


if __name__ == "__main__":
    unittest.main()
