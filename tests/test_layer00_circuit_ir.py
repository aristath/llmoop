from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmoop.circuit_ir import load_circuit, validate_circuit, validate_circuit_against_pedal
from tools.lower_layer00_to_circuit import build_layer00_circuit, lower_layer00


PEDAL_PATH = Path("transpiled/lfm2_5_230m/layers/layer_00.json")


class Layer00CircuitIrTest(unittest.TestCase):
    def test_build_layer00_circuit_validates_against_pedal_contract(self) -> None:
        import json

        pedal = json.loads(PEDAL_PATH.read_text())
        circuit = build_layer00_circuit(pedal, PEDAL_PATH)
        report = validate_circuit_against_pedal(circuit, pedal)

        self.assertTrue(report.ok, [issue.to_json() for issue in report.errors])
        self.assertEqual("llmoop.stream_circuit.v1", circuit["schema"])
        self.assertEqual("layer_00_exact_lfm2_conv_circuit_v1", circuit["id"])
        self.assertEqual("exact_lowering_lfm2_conv_layer_v1", circuit["implementation"])

        self.assertEqual([1024], circuit["boundary"]["inputs"][0]["shape"])
        self.assertEqual([1024], circuit["boundary"]["outputs"][0]["shape"])
        self.assertEqual([3, 1024], circuit["state_ports"][0]["shape"])
        self.assertEqual("stream", circuit["state_ports"][0]["owner"])

        self.assertEqual(16, len(circuit["nodes"]))
        self.assertEqual("operator_norm", circuit["nodes"][0]["id"])
        self.assertEqual("ffn_residual", circuit["nodes"][-1]["id"])

        self.assertEqual(
            {
                "operator_norm",
                "ffn_norm",
                "ffn_gate",
                "ffn_down",
                "ffn_up",
                "conv_in_projection",
                "conv_depthwise_kernel",
                "conv_out_projection",
            },
            set(circuit["parameters"]["refs"]),
        )

    def test_lower_layer00_writes_circuit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = lower_layer00(PEDAL_PATH, Path(tempdir))

            self.assertTrue(result["validation"]["ok"], result["validation"]["issues"])
            self.assertTrue(result["circuit_path"].exists())
            self.assertTrue(result["params_path"].exists())
            self.assertTrue(result["state_path"].exists())

            circuit = load_circuit(result["circuit_path"])
            report = validate_circuit(circuit)
            self.assertTrue(report.ok, [issue.to_json() for issue in report.errors])


if __name__ == "__main__":
    unittest.main()
