from model_structure_common import *
from model_structure_common import _tensor

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
    assert [layer.intermediate_size for layer in structure.layers] == [24, 24, 24]
    assert structure.rotary_width == 4
    assert structure.attention_window_size == 8
    assert structure.embedding_scale == 4.0
    assert structure.rms_norm_weight_offset == 1.0
    assert structure.activation == "gelu_tanh"
    assert structure.logits_soft_cap == 30.0

    recurrent = make_layer(structure, structure.layers[0])
    assert [state["shape"] for state in recurrent["state_ports"]] == [[16, 4], [16]]
    assert [state["dtype"] for state in recurrent["state_ports"]] == ["BF16", "F32"]
    circuit = build_component_circuit(recurrent, Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["rg_lru_step"]["attrs"]["block_width"] == 8
    assert nodes["ffn_gate_projection"]["params"] == ["ffn_gate", "ffn_gate_bias"]
    assert nodes["ffn_gate_activation"]["op"] == "gelu_tanh"

    attention = build_component_circuit(
        make_layer(structure, structure.layers[2]), Path("layer_02.json")
    )
    attention_nodes = {node["id"]: node for node in attention["nodes"]}
    assert attention_nodes["attention_read"]["attrs"]["window_size"] == 8
    assert attention_nodes["attention_out_projection"]["params"] == [
        "attention_out_projection",
        "attention_out_projection_bias",
    ]
