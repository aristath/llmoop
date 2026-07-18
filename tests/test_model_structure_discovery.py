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
    split = next(
        node for node in attention_circuit["nodes"] if node["id"] == "q_gate_split"
    )
    assert split["attrs"]["layout"] == "per_head_interleaved"
    assert split["attrs"]["blocks"] == 8
    assert split["attrs"]["block_part_width"] == 256


def test_discovers_sparse_moe_and_model_specific_numerics_by_structure() -> None:
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": _tensor([49155, 1024]),
        "model.norm.weight": _tensor([1024]),
        f"{prefix}.input_layernorm.weight": _tensor([1024]),
        f"{prefix}.post_attention_layernorm.weight": _tensor([1024]),
        f"{prefix}.self_attn.q_proj.weight": _tensor([1024, 1024]),
        f"{prefix}.self_attn.k_proj.weight": _tensor([512, 1024]),
        f"{prefix}.self_attn.v_proj.weight": _tensor([512, 1024]),
        f"{prefix}.self_attn.o_proj.weight": _tensor([1024, 1024]),
        f"{prefix}.block_sparse_moe.input_linear.weight": _tensor([32, 1024, 1024]),
        f"{prefix}.block_sparse_moe.output_linear.weight": _tensor([32, 1024, 512]),
        f"{prefix}.block_sparse_moe.router.layer.weight": _tensor([32, 1024]),
    }
    config = {
        "model_type": "synthetic_sparse_decoder",
        "hidden_size": 1024,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "num_experts_per_tok": 8,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.015625,
        "logits_scaling": 6.0,
        "vocab_size": 49155,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_500_000.0,
        "tie_word_embeddings": True,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    layer = structure.layers[0]
    assert layer.operator_type == "full_attention"
    assert layer.feed_forward_type == "sparse_moe"
    assert structure.num_experts == 32
    assert structure.experts_per_token == 8
    assert structure.embedding_scale == 12.0
    assert structure.residual_scale == 0.22
    assert structure.attention_scale == 0.015625
    assert structure.logits_scale == 6.0

    pedal = make_layer(structure, layer)
    circuit = build_pedal_circuit(pedal, Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["attention_read"]["attrs"]["scale"] == 0.015625
    assert nodes["operator_residual"]["op"] == "scaled_residual_add"
    assert nodes["operator_residual"]["attrs"]["scale"] == 0.22
    assert nodes["moe_topk"]["attrs"] == {
        "num_experts": 32,
        "experts_per_token": 8,
    }
    assert nodes["sparse_moe_experts"]["params"] == ["moe_input", "moe_output"]
    assert nodes["moe_reduce"]["outputs"] == ["ffn_out"]


def test_discovers_recurrent_block_pattern_biases_and_numerics_by_structure() -> None:
    tensors = {
        "model.embed_tokens.weight": _tensor([256, 16]),
        "model.final_norm.weight": _tensor([16]),
    }
    for index in range(3):
        prefix = f"model.layers.{index}"
        tensors.update(
            {
                f"{prefix}.temporal_pre_norm.weight": _tensor([16]),
                f"{prefix}.channel_pre_norm.weight": _tensor([16]),
                f"{prefix}.mlp_block.gate_proj.weight": _tensor([24, 16]),
                f"{prefix}.mlp_block.gate_proj.bias": _tensor([24]),
                f"{prefix}.mlp_block.up_proj.weight": _tensor([24, 16]),
                f"{prefix}.mlp_block.up_proj.bias": _tensor([24]),
                f"{prefix}.mlp_block.down_proj.weight": _tensor([16, 24]),
                f"{prefix}.mlp_block.down_proj.bias": _tensor([16]),
            }
        )
        if index < 2:
            tensors.update(
                {
                    f"{prefix}.temporal_block.linear_x.weight": _tensor([16, 16]),
                    f"{prefix}.temporal_block.linear_x.bias": _tensor([16]),
                    f"{prefix}.temporal_block.linear_y.weight": _tensor([16, 16]),
                    f"{prefix}.temporal_block.linear_y.bias": _tensor([16]),
                    f"{prefix}.temporal_block.linear_out.weight": _tensor([16, 16]),
                    f"{prefix}.temporal_block.linear_out.bias": _tensor([16]),
                    f"{prefix}.temporal_block.conv_1d.weight": _tensor([16, 1, 4]),
                    f"{prefix}.temporal_block.conv_1d.bias": _tensor([16]),
                    f"{prefix}.temporal_block.rg_lru.input_gate_weight": _tensor(
                        [2, 8, 8]
                    ),
                    f"{prefix}.temporal_block.rg_lru.input_gate_bias": _tensor([2, 8]),
                    f"{prefix}.temporal_block.rg_lru.recurrent_gate_weight": _tensor(
                        [2, 8, 8]
                    ),
                    f"{prefix}.temporal_block.rg_lru.recurrent_gate_bias": _tensor(
                        [2, 8]
                    ),
                    f"{prefix}.temporal_block.rg_lru.recurrent_param": _tensor([16]),
                }
            )
        else:
            tensors.update(
                {
                    f"{prefix}.temporal_block.q_proj.weight": _tensor([16, 16]),
                    f"{prefix}.temporal_block.k_proj.weight": _tensor([8, 16]),
                    f"{prefix}.temporal_block.v_proj.weight": _tensor([8, 16]),
                    f"{prefix}.temporal_block.o_proj.weight": _tensor([16, 16]),
                    f"{prefix}.temporal_block.o_proj.bias": _tensor([16]),
                }
            )
    config = {
        "model_type": "synthetic_recurrent_decoder",
        "_block_types": ["recurrent", "recurrent", "attention"],
        "hidden_size": 16,
        "intermediate_size": 48,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "lru_width": 16,
        "conv1d_width": 4,
        "partial_rotary_factor": 0.5,
        "attention_window_size": 8,
        "embeddings_scale_by_sqrt_dim": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "logits_soft_cap": 30.0,
        "vocab_size": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    assert [layer.operator_type for layer in structure.layers] == [
        "rg_lru",
        "rg_lru",
        "full_attention",
    ]
    assert structure.intermediate_size == 24
    assert structure.rotary_width == 4
    assert structure.attention_window_size == 8
    assert structure.embedding_scale == 4.0
    assert structure.rms_norm_weight_offset == 1.0
    assert structure.activation == "gelu_tanh"
    assert structure.logits_soft_cap == 30.0

    recurrent = make_layer(structure, structure.layers[0])
    assert [state["shape"] for state in recurrent["state_ports"]] == [[16, 4], [16]]
    assert [state["dtype"] for state in recurrent["state_ports"]] == ["BF16", "F32"]
    circuit = build_pedal_circuit(recurrent, Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["rg_lru_step"]["attrs"]["block_width"] == 8
    assert nodes["ffn_gate_projection"]["params"] == ["ffn_gate", "ffn_gate_bias"]
    assert nodes["ffn_gate_activation"]["op"] == "gelu_tanh"

    attention = build_pedal_circuit(
        make_layer(structure, structure.layers[2]), Path("layer_02.json")
    )
    attention_nodes = {node["id"]: node for node in attention["nodes"]}
    assert attention_nodes["attention_read"]["attrs"]["window_size"] == 8
    assert attention_nodes["attention_out_projection"]["params"] == [
        "attention_out_projection",
        "attention_out_projection_bias",
    ]
