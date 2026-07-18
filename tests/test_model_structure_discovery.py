from __future__ import annotations

from pathlib import Path

from llmoop.circuit_executors import GQAAttentionCircuitPedal
from llmoop.circuit_lowering import build_pedal_circuit
from llmoop.model_transpiler import discover_model_structure, make_layer


def _tensor(shape: list[int]) -> dict[str, object]:
    return {"dtype": "BF16", "shape": shape}


def test_discovers_attention_without_optional_query_key_norms() -> None:
    tensors = {
        "model.embed_tokens.weight": _tensor([49152, 960]),
        "model.norm.weight": _tensor([960]),
        "model.layers.0.input_layernorm.weight": _tensor([960]),
        "model.layers.0.post_attention_layernorm.weight": _tensor([960]),
        "model.layers.0.self_attn.q_proj.weight": _tensor([960, 960]),
        "model.layers.0.self_attn.k_proj.weight": _tensor([320, 960]),
        "model.layers.0.self_attn.v_proj.weight": _tensor([320, 960]),
        "model.layers.0.self_attn.o_proj.weight": _tensor([960, 960]),
        "model.layers.0.mlp.gate_proj.weight": _tensor([2560, 960]),
        "model.layers.0.mlp.up_proj.weight": _tensor([2560, 960]),
        "model.layers.0.mlp.down_proj.weight": _tensor([960, 2560]),
    }
    config = {
        "model_type": "synthetic_decoder",
        "architectures": ["SyntheticForCausalLM"],
        "hidden_size": 960,
        "intermediate_size": 2560,
        "num_hidden_layers": 1,
        "num_attention_heads": 15,
        "num_key_value_heads": 5,
        "vocab_size": 49152,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000.0,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    layer = structure.layers[0]

    assert layer.operator_type == "full_attention"
    assert "q_norm" not in layer.tensors
    assert "k_norm" not in layer.tensors
    assert structure.norm_eps == 1e-5
    assert structure.rope_theta == 100000.0

    pedal = make_layer(structure, layer)
    circuit = build_pedal_circuit(pedal, Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}

    assert "q_head_norm" not in nodes
    assert "k_head_norm" not in nodes
    assert nodes["q_rope"]["inputs"] == ["q_projected"]
    assert nodes["k_rope"]["inputs"] == ["k_projected"]
    assert nodes["q_rope"]["attrs"]["theta"] == 100000.0
    assert nodes["operator_norm"]["attrs"]["eps"] == 1e-5

    class TensorStore:
        def get(self, name: str) -> str:
            return name

    implementation = GQAAttentionCircuitPedal.from_tensor_store(
        tensor_store=TensorStore(),
        torch=object(),
        circuit=circuit,
    )
    assert "q_norm" not in implementation.weights
    assert "k_norm" not in implementation.weights
    assert implementation.norm_eps == 1e-5
