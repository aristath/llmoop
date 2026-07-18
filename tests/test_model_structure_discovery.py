from __future__ import annotations

from pathlib import Path

from llmoop.circuit_executors import GQAAttentionCircuitPedal
from llmoop.circuit_lowering import build_pedal_circuit
from llmoop.model_transpiler import discover_model_structure, make_layer


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}


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


def test_discovers_nested_hybrid_decoder_by_tensor_structure() -> None:
    root = "model.language_model"
    tensors = {
        f"{root}.embed_tokens.weight": _tensor([248320, 1024]),
        f"{root}.norm.weight": _tensor([1024]),
    }
    for layer_index in range(2):
        prefix = f"{root}.layers.{layer_index}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor([1024]),
                f"{prefix}.post_attention_layernorm.weight": _tensor([1024]),
                f"{prefix}.mlp.gate_proj.weight": _tensor([3584, 1024]),
                f"{prefix}.mlp.up_proj.weight": _tensor([3584, 1024]),
                f"{prefix}.mlp.down_proj.weight": _tensor([1024, 3584]),
            }
        )
    linear = f"{root}.layers.0.linear_attn"
    tensors.update(
        {
            f"{linear}.in_proj_qkv.weight": _tensor([6144, 1024]),
            f"{linear}.in_proj_z.weight": _tensor([2048, 1024]),
            f"{linear}.in_proj_b.weight": _tensor([16, 1024]),
            f"{linear}.in_proj_a.weight": _tensor([16, 1024]),
            f"{linear}.conv1d.weight": _tensor([6144, 1, 4]),
            f"{linear}.A_log": _tensor([16], "F32"),
            f"{linear}.dt_bias": _tensor([16]),
            f"{linear}.norm.weight": _tensor([128], "F32"),
            f"{linear}.out_proj.weight": _tensor([1024, 2048]),
        }
    )
    attention = f"{root}.layers.1.self_attn"
    tensors.update(
        {
            f"{attention}.q_proj.weight": _tensor([4096, 1024]),
            f"{attention}.k_proj.weight": _tensor([512, 1024]),
            f"{attention}.v_proj.weight": _tensor([512, 1024]),
            f"{attention}.o_proj.weight": _tensor([1024, 2048]),
            f"{attention}.q_norm.weight": _tensor([256]),
            f"{attention}.k_norm.weight": _tensor([256]),
        }
    )
    config = {
        "model_type": "synthetic_multimodal_wrapper",
        "architectures": ["SyntheticConditionalGeneration"],
        "text_config": {
            "model_type": "synthetic_hybrid_text",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "attn_output_gate": True,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
            "eos_token_id": 248044,
        },
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    assert structure.model_type == "synthetic_hybrid_text"
    assert structure.head_width == 256
    assert structure.rotary_width == 64
    assert structure.rms_norm_weight_offset == 1.0
    assert structure.token_ids["eos"] == 248044
    assert [layer.operator_type for layer in structure.layers] == [
        "gated_delta",
        "full_attention",
    ]

    gated = make_layer(structure, structure.layers[0])
    assert gated["state_ports"][0]["dtype"] == "BF16"
    assert gated["state_ports"][1]["dtype"] == "F32"
    attention_circuit = build_pedal_circuit(
        make_layer(structure, structure.layers[1]), Path("layer_01.json")
    )
    split = next(node for node in attention_circuit["nodes"] if node["id"] == "q_gate_split")
    assert split["attrs"]["layout"] == "per_head_interleaved"
    assert split["attrs"]["blocks"] == 8
    assert split["attrs"]["block_part_width"] == 256
