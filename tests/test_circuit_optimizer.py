from __future__ import annotations

import unittest

from llmoop.circuit_optimizer import optimize_circuit_for_vulkan


class VulkanCircuitOptimizerTest(unittest.TestCase):
    def test_fuses_discovered_regions_without_layer_or_node_names(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection_a",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight_a"],
                },
                {
                    "id": "skip_a",
                    "op": "residual_add",
                    "inputs": ["input_frame", "projected"],
                    "outputs": ["residual_out"],
                },
                {
                    "id": "activation_a",
                    "op": "silu",
                    "inputs": ["gate"],
                    "outputs": ["activated"],
                },
                {
                    "id": "product_a",
                    "op": "multiply",
                    "inputs": ["up", "activated"],
                    "outputs": ["output_frame"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual(
            ["linear_residual", "silu_multiply"],
            [node["op"] for node in optimized["nodes"]],
        )
        self.assertEqual(
            ["hidden", "input_frame"], optimized["nodes"][0]["inputs"]
        )
        self.assertEqual(["weight_a"], optimized["nodes"][0]["params"])
        self.assertEqual(["gate", "up"], optimized["nodes"][1]["inputs"])
        self.assertEqual("BF16", optimized["nodes"][1]["attrs"]["intermediate_rounding"])
        self.assertEqual(4, len(circuit["nodes"]))

    def test_does_not_fuse_an_intermediate_with_multiple_consumers(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "activation",
                    "op": "silu",
                    "inputs": ["gate"],
                    "outputs": ["activated"],
                },
                {
                    "id": "product",
                    "op": "multiply",
                    "inputs": ["activated", "up"],
                    "outputs": ["product"],
                },
                {
                    "id": "extra_consumer",
                    "op": "multiply",
                    "inputs": ["activated", "other"],
                    "outputs": ["output_frame"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual(circuit, optimized)

    def test_fuses_block_scaled_fp8_linear_with_residual(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight", "weight_scale_inv"],
                },
                {
                    "id": "skip",
                    "op": "residual_add",
                    "inputs": ["residual", "projected"],
                    "outputs": ["output"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual("linear_residual", optimized["nodes"][0]["op"])
        self.assertEqual(
            ["weight", "weight_scale_inv"], optimized["nodes"][0]["params"]
        )


if __name__ == "__main__":
    unittest.main()
