from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nerve.circuit_ir import load_circuit, validate_circuit
from nerve.circuit_lowering import lower_pedalboard
from nerve.compilation import read_json
from tests.fixtures import compiled_model_or_skip


class PedalboardCircuitLoweringTest(unittest.TestCase):
    def test_lower_pedalboard_writes_one_circuit_per_pedal(self) -> None:
        fixture = compiled_model_or_skip()
        with tempfile.TemporaryDirectory() as tempdir:
            out_dir = Path(tempdir)
            result = lower_pedalboard(fixture.transpiled_dir, out_dir)
            index = result["index"]

            self.assertEqual("nerve.lowered_pedalboard.v1", index["schema"])
            self.assertEqual(17, index["summary"]["circuit_count"])
            self.assertEqual(
                {
                    "conv": 8,
                    "full_attention": 6,
                    "input_transducer": 1,
                    "output_transducer": 1,
                    "sampler": 1,
                },
                index["summary"]["operator_counts"],
            )
            self.assertEqual("explicit_graph", index["graph"]["wiring"])
            self.assertEqual(17, len(index["graph"]["cables"]))
            self.assertEqual(
                {
                    "id": "cable_0000",
                    "connection": {"kind": "forward"},
                    "source": {
                        "pedal_id": "input_transducer",
                        "port_id": "output_frame",
                    },
                    "destination": {
                        "pedal_id": "layer_00",
                        "port_id": "input_frame",
                    },
                },
                index["graph"]["cables"][0],
            )
            self.assertEqual(
                {
                    "kind": "temporal_feedback",
                    "delay_activations": 1,
                },
                index["graph"]["cables"][-1]["connection"],
            )
            self.assertEqual(
                ["user_input", "random_seed"],
                [
                    port["id"]
                    for port in index["graph"]["boundary"]["external_inputs"]
                ],
            )
            self.assertEqual("nerve.compiled_pedalboard_artifact.v1", index["source"]["format"])
            self.assertEqual(".", index["source"]["artifact_root"])
            self.assertTrue(result["index_path"].exists())

            for circuit_entry in index["graph"]["circuits"]:
                self.assertNotIn("pedal_file", circuit_entry)
                circuit_path = out_dir / circuit_entry["circuit"]
                params_path = out_dir / circuit_entry["params"]
                state_path = out_dir / circuit_entry["state"]
                self.assertTrue(circuit_path.exists(), circuit_path)
                self.assertTrue(params_path.exists(), params_path)
                self.assertTrue(state_path.exists(), state_path)

                circuit = load_circuit(circuit_path)
                report = validate_circuit(circuit)
                self.assertTrue(report.ok, [issue.to_json() for issue in report.errors])
                self.assertEqual(circuit_entry["id"], circuit["source"]["pedal_id"])
                self.assertNotIn("pedal_file", circuit["source"])

                params = read_json(params_path)
                state = read_json(state_path)
                self.assertEqual(circuit["id"], params["circuit"])
                self.assertEqual(circuit["id"], state["circuit"])

    def test_attention_circuit_declares_kv_as_stream_owned_transient_state(self) -> None:
        fixture = compiled_model_or_skip()
        with tempfile.TemporaryDirectory() as tempdir:
            result = lower_pedalboard(fixture.transpiled_dir, Path(tempdir))
            attention = next(circuit for circuit in result["index"]["graph"]["circuits"] if circuit["operator_type"] == "full_attention")
            circuit = load_circuit(Path(tempdir) / attention["circuit"])

            self.assertEqual("source_reference_circuit", circuit["behavioral_role"])
            self.assertEqual("reference_gqa_attention_layer_circuit_v1", circuit["implementation"])
            self.assertEqual("kv_memory", circuit["state_ports"][0]["id"])
            self.assertEqual("append_only_attention_memory", circuit["state_ports"][0]["type"])
            self.assertEqual("stream", circuit["state_ports"][0]["owner"])
            self.assertEqual("append_only_kv", circuit["state_ports"][0]["layout"])
            self.assertIn("kv_memory_append", [node["id"] for node in circuit["nodes"]])
            self.assertIn("attention_read", [node["id"] for node in circuit["nodes"]])


if __name__ == "__main__":
    unittest.main()
