from __future__ import annotations

from pathlib import Path

from nerve.circuit_ir import validate_circuit
from nerve.circuit_lowering import build_component_circuit
from nerve.circuit_optimizer import optimize_circuit_for_vulkan
from nerve.model_package_manifest import component_kernel_spec


def _parameter(tensor: str) -> dict[str, str]:
    return {"tensor": tensor}


def _conv_component() -> dict[str, object]:
    parameters = {
        name: _parameter(f"layer.{name}")
        for name in (
            "operator_norm",
            "conv_in_projection",
            "conv_depthwise_kernel",
            "conv_out_projection",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        )
    }
    return {
        "id": "layer_00",
        "source_layer_index": 0,
        "runtime_role": "signal_processor",
        "operator_type": "conv",
        "ports": {
            "inputs": [{"id": "input", "signal": "frame", "shape": [8]}],
            "outputs": [{"id": "output", "signal": "frame", "shape": [8]}],
            "controls": [],
        },
        "state_ports": [
            {
                "id": "temporal_memory",
                "type": "rolling_channel_memory",
                "shape": [3, 8],
                "update": "shift_append",
            }
        ],
        "parameter_block": {
            "layout": "tensor_refs",
            "storage": "safetensors",
            "params": parameters,
        },
        "numerics": {
            "rms_norm_eps": 1e-5,
            "rms_norm_weight_offset": 0.0,
            "residual_scale": 1.0,
            "per_layer_input_width": None,
        },
        "feed_forward": {
            "type": "dense_swiglu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "activation": "silu",
        },
        "transition_contract": {"reference_behavior": "synthetic.shortconv"},
    }


def _assert_exact_coverage(circuit: dict[str, object]) -> None:
    tree = circuit["semantic_module_tree"]
    modules = tree["modules"]
    source_nodes = [
        node_id for module in modules for node_id in module["source_node_ids"]
    ]
    state_ports = [
        state_id for module in modules for state_id in module["owned_state_port_ids"]
    ]
    assert sorted(source_nodes) == sorted(node["id"] for node in circuit["nodes"])
    assert len(source_nodes) == len(set(source_nodes))
    assert sorted(state_ports) == sorted(state["id"] for state in circuit["state_ports"])
    assert len(state_ports) == len(set(state_ports))


def test_shortconv_lowering_builds_exact_recursive_semantic_anatomy() -> None:
    circuit = build_component_circuit(_conv_component(), Path("layer_00.json"))

    assert validate_circuit(circuit).ok
    _assert_exact_coverage(circuit)
    modules = {
        module["id"]: module for module in circuit["semantic_module_tree"]["modules"]
    }
    assert circuit["semantic_module_tree"]["schema"] == "nerve.semantic_module_tree.v1"
    assert modules["layer"]["child_ids"] == [
        "layer.token_mixer",
        "layer.feature_transform",
    ]
    assert modules["layer.token_mixer.temporal_state"]["owned_state_port_ids"] == [
        "temporal_memory"
    ]
    assert modules["layer.token_mixer.temporal_state"]["source_node_ids"] == [
        "temporal_memory_update"
    ]
    assert modules["layer.feature_transform.projections"]["source_node_ids"] == [
        "ffn_gate_projection",
        "ffn_up_projection",
        "ffn_down_projection",
    ]


def test_optimizer_preserves_semantic_tree_while_kernel_provenance_crosses_modules() -> None:
    circuit = build_component_circuit(_conv_component(), Path("layer_00.json"))
    optimized = optimize_circuit_for_vulkan(
        circuit,
        can_fuse_linear_split=lambda _node: True,
        can_fuse_multiply_rolling_depthwise=lambda _multiply, _rolling, _depthwise: True,
        can_fuse_recurrent_output_gate=lambda _recurrent, _gate: True,
        can_fuse_linear_split_recurrent=lambda _projection, _recurrent: True,
    )

    assert optimized["semantic_module_tree"] == circuit["semantic_module_tree"]
    assert optimized["semantic_execution_nodes"] == circuit["nodes"]
    assert validate_circuit(optimized).ok
    fused = next(
        node
        for node in optimized["nodes"]
        if node["op"] == "linear_split_recurrent_depthwise_gate"
    )
    spec = component_kernel_spec(
        execution_index=0,
        node=fused,
        circuit=optimized,
        shader_file="synthetic.comp",
        local_size_x=64,
        workgroup_count_x=1,
    )
    assert spec["source_node_ids"] == [
        "conv_in_projection",
        "split_b_c_x",
        "input_gate",
        "temporal_memory_update",
        "depthwise_temporal_conv",
        "output_gate",
    ]
    assert spec["semantic_module_ids"] == [
        "layer.token_mixer.input_projection",
        "layer.token_mixer.gates",
        "layer.token_mixer.temporal_state",
        "layer.token_mixer.temporal_convolution",
    ]


def test_semantic_validation_rejects_duplicate_node_and_state_ownership() -> None:
    circuit = build_component_circuit(_conv_component(), Path("layer_00.json"))
    modules = circuit["semantic_module_tree"]["modules"]
    modules[0]["source_node_ids"].append("operator_norm")
    modules[0]["owned_state_port_ids"].append("temporal_memory")

    report = validate_circuit(circuit)

    assert not report.ok
    messages = [issue.message for issue in report.errors]
    assert any("belongs to both" in message and "operator_norm" in message for message in messages)
    assert any("owned by both" in message and "temporal_memory" in message for message in messages)
