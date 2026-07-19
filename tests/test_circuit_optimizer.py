from __future__ import annotations

import unittest

from llmoop.circuit_optimizer import optimize_circuit_for_vulkan


class VulkanCircuitOptimizerTest(unittest.TestCase):
    def test_fuses_multiply_rolling_state_and_depthwise_convolution(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate",
                    "op": "multiply",
                    "inputs": ["gate_value", "projected"],
                    "outputs": ["gated"],
                },
                {
                    "id": "shift",
                    "op": "rolling_state_update",
                    "inputs": ["gated", "temporal_memory"],
                    "outputs": ["window"],
                    "state_reads": ["temporal_memory"],
                    "state_writes": ["temporal_memory"],
                },
                {
                    "id": "convolve",
                    "op": "depthwise_conv1d",
                    "inputs": ["window"],
                    "outputs": ["convolved"],
                    "params": ["kernel"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_multiply_rolling_depthwise=lambda multiply, rolling, depthwise: (
                multiply["outputs"] == rolling["inputs"][:1]
                and rolling["outputs"] == depthwise["inputs"]
            ),
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("multiply_rolling_depthwise", fused["op"])
        self.assertEqual(
            ["gate_value", "projected", "temporal_memory"], fused["inputs"]
        )
        self.assertEqual(["convolved"], fused["outputs"])
        self.assertEqual(["kernel"], fused["params"])
        self.assertEqual(["temporal_memory"], fused["state_reads"])
        self.assertEqual(["temporal_memory"], fused["state_writes"])
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])

    def test_does_not_fuse_multiply_rolling_state_with_shared_window(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate",
                    "op": "multiply",
                    "inputs": ["gate_value", "projected"],
                    "outputs": ["gated"],
                },
                {
                    "id": "shift",
                    "op": "rolling_state_update",
                    "inputs": ["gated", "temporal_memory"],
                    "outputs": ["window"],
                    "state_reads": ["temporal_memory"],
                    "state_writes": ["temporal_memory"],
                },
                {
                    "id": "convolve",
                    "op": "depthwise_conv1d",
                    "inputs": ["window"],
                    "outputs": ["convolved"],
                    "params": ["kernel"],
                },
                {
                    "id": "extra",
                    "op": "silu",
                    "inputs": ["window"],
                    "outputs": ["extra_out"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_multiply_rolling_depthwise=lambda _multiply, _rolling, _depthwise: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_dual_linear_projection_into_silu_multiply(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate__up",
                    "op": "parallel_linear_2way",
                    "inputs": ["hidden"],
                    "outputs": ["gate", "up"],
                    "params": ["gate_weight", "up_weight"],
                    "attrs": {"branch_count": 2},
                },
                {
                    "id": "activate__multiply",
                    "op": "silu_multiply",
                    "inputs": ["gate", "up"],
                    "outputs": ["ffn_hidden"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_dual_linear_silu_multiply=lambda projection, multiply: (
                projection["outputs"] == multiply["inputs"]
            ),
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("dual_linear_silu_multiply", fused["op"])
        self.assertEqual(["hidden"], fused["inputs"])
        self.assertEqual(["ffn_hidden"], fused["outputs"])
        self.assertEqual(["gate_weight", "up_weight"], fused["params"])
        self.assertEqual(0, fused["attrs"]["activated_input_index"])
        self.assertEqual("BF16", fused["attrs"]["activation_rounding"])

    def test_does_not_fuse_dual_linear_outputs_with_extra_consumer(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate__up",
                    "op": "parallel_linear_2way",
                    "inputs": ["hidden"],
                    "outputs": ["gate", "up"],
                    "params": ["gate_weight", "up_weight"],
                },
                {
                    "id": "activate__multiply",
                    "op": "silu_multiply",
                    "inputs": ["gate", "up"],
                    "outputs": ["ffn_hidden"],
                },
                {
                    "id": "extra",
                    "op": "silu",
                    "inputs": ["gate"],
                    "outputs": ["extra_out"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_dual_linear_silu_multiply=lambda _projection, _multiply: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_parallel_head_norm_rope_branches_across_independent_nodes(
        self,
    ) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "first_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["first_projected"],
                    "outputs": ["first_normed"],
                    "params": ["first_weight"],
                    "attrs": {"head_count": 8},
                },
                {
                    "id": "second_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["second_projected"],
                    "outputs": ["second_normed"],
                    "params": ["second_weight"],
                    "attrs": {"head_count": 2},
                },
                {
                    "id": "first_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["first_normed"],
                    "outputs": ["first_positioned"],
                    "attrs": {"head_count": 8},
                },
                {
                    "id": "second_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["second_normed"],
                    "outputs": ["second_positioned"],
                    "attrs": {"head_count": 2},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_head_norm_rope=lambda branches: len(branches) == 2,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_head_norm_rope_2way", fused["op"])
        self.assertEqual(
            ["first_projected", "second_projected"], fused["inputs"]
        )
        self.assertEqual(
            ["first_positioned", "second_positioned"], fused["outputs"]
        )
        self.assertEqual(["first_weight", "second_weight"], fused["params"])
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])

    def test_does_not_fuse_head_norm_with_multiple_consumers(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "first_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["first_projected"],
                    "outputs": ["first_normed"],
                    "params": ["first_weight"],
                },
                {
                    "id": "second_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["second_projected"],
                    "outputs": ["second_normed"],
                    "params": ["second_weight"],
                },
                {
                    "id": "extra_consumer",
                    "op": "silu",
                    "inputs": ["first_normed"],
                    "outputs": ["extra"],
                },
                {
                    "id": "first_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["first_normed"],
                    "outputs": ["first_positioned"],
                },
                {
                    "id": "second_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["second_normed"],
                    "outputs": ["second_positioned"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_head_norm_rope=lambda _branches: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_two_or_three_independent_linears_with_one_input(self) -> None:
        nodes = [
            {
                "id": branch,
                "op": "linear",
                "inputs": ["hidden"],
                "outputs": [f"{branch}_out"],
                "params": [f"{branch}_weight"],
            }
            for branch in ("a", "b", "c")
        ]

        optimized = optimize_circuit_for_vulkan(
            {"nodes": nodes},
            can_fuse_parallel_linears=lambda group: len(group) in {2, 3},
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_linear_3way", fused["op"])
        self.assertEqual(["hidden"], fused["inputs"])
        self.assertEqual(["a_out", "b_out", "c_out"], fused["outputs"])
        self.assertEqual(
            ["a_weight", "b_weight", "c_weight"], fused["params"]
        )

        pair = optimize_circuit_for_vulkan(
            {"nodes": nodes[:2]},
            can_fuse_parallel_linears=lambda group: len(group) == 2,
        )
        self.assertEqual("parallel_linear_2way", pair["nodes"][0]["op"])

    def test_does_not_fuse_linears_with_different_inputs(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "a",
                    "op": "linear",
                    "inputs": ["first"],
                    "outputs": ["a_out"],
                    "params": ["a_weight"],
                },
                {
                    "id": "b",
                    "op": "linear",
                    "inputs": ["second"],
                    "outputs": ["b_out"],
                    "params": ["b_weight"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_linears=lambda _group: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_compatible_linear_into_contiguous_three_way_split(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight"],
                },
                {
                    "id": "partition",
                    "op": "split",
                    "inputs": ["projected"],
                    "outputs": ["a", "b", "c"],
                    "attrs": {"part_width": 16},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda node: node["params"] == ["weight"],
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("linear_split_3way", fused["op"])
        self.assertEqual(["hidden"], fused["inputs"])
        self.assertEqual(["a", "b", "c"], fused["outputs"])
        self.assertEqual([16, 16, 16], fused["attrs"]["part_widths"])
        self.assertEqual(["projection", "partition"], fused["attrs"]["compiled_from"])

    def test_keeps_linear_split_when_backend_layout_is_not_fusible(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight"],
                },
                {
                    "id": "partition",
                    "op": "split",
                    "inputs": ["projected"],
                    "outputs": ["a", "b", "c"],
                    "attrs": {"part_width": 16},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda _node: False,
        )

        self.assertEqual(circuit, optimized)

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
