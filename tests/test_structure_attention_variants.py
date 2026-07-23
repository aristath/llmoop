from model_structure_common import *
from model_structure_common import _tensor

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
    assert nodes["q_rope"]["attrs"]["head_count"] == 15
    assert nodes["k_rope"]["attrs"]["head_count"] == 5
    assert nodes["q_rope"]["attrs"]["theta"] == 100000.0
    assert nodes["operator_norm"]["attrs"]["eps"] == 1e-5


def test_discovers_source_defined_per_head_attention_gate(tmp_path: Path) -> None:
    (tmp_path / "modeling_custom.py").write_text(
        "gate = F.softplus(self.g_proj(hidden_states).float())\n"
    )
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": _tensor([256, 16]),
        "model.norm.weight": _tensor([16]),
        f"{prefix}.input_layernorm.weight": _tensor([16]),
        f"{prefix}.post_attention_layernorm.weight": _tensor([16]),
        f"{prefix}.self_attn.q_proj.weight": _tensor([16, 16]),
        f"{prefix}.self_attn.k_proj.weight": _tensor([8, 16]),
        f"{prefix}.self_attn.v_proj.weight": _tensor([8, 16]),
        f"{prefix}.self_attn.o_proj.weight": _tensor([16, 16]),
        f"{prefix}.self_attn.g_proj.weight": _tensor([2, 16]),
        f"{prefix}.mlp.gate_proj.weight": _tensor([12, 16]),
        f"{prefix}.mlp.up_proj.weight": _tensor([12, 16]),
        f"{prefix}.mlp.down_proj.weight": _tensor([16, 12]),
    }
    config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
    }

    structure = discover_model_structure(tmp_path, config, tensors)
    layer = structure.layers[0]
    assert layer.attention_gate_activation == "softplus"
    assert layer.attention_gate_per_head is True

    pedal = make_layer(structure, layer)
    circuit = build_pedal_circuit(pedal, Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["attention_gate_projection"]["params"] == ["attention_gate_projection"]
    assert nodes["attention_output_gate"] == {
        "id": "attention_output_gate",
        "op": "softplus_multiply",
        "inputs": ["attention_out", "attention_gate"],
        "outputs": ["attention_gated"],
        "attrs": {"query_heads": 2, "head_width": 8, "per_head": True},
    }
    assert nodes["attention_out_projection"]["inputs"] == ["attention_gated"]


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


def test_discovers_sparse_moe_and_model_specific_numerics_by_structure(
    tmp_path: Path,
) -> None:
    (tmp_path / "modeling_sparse.py").write_text(
        "routing_scores = torch.sigmoid(router_logits)\n"
    )
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
        f"{prefix}.mlp.experts.e_score_correction_bias": _tensor([32]),
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
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
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

    structure = discover_model_structure(tmp_path, config, tensors)
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
        "activation": "sigmoid",
        "normalize_selected": True,
        "logit_softcap": 0.0,
        "selection_bias": True,
    }
    assert nodes["moe_topk"]["params"] == ["moe_router_correction_bias"]
    assert nodes["sparse_moe_gate_up"]["params"] == ["moe_input"]
    assert nodes["sparse_moe_down"]["params"] == ["moe_output"]
    assert nodes["moe_reduce"]["outputs"] == ["ffn_out"]
    assert nodes["moe_reduce"]["attrs"]["routed_scaling_factor"] == 2.5


def test_discovers_mixed_window_attention_sinks_and_shared_sparse_experts() -> None:
    tensors = {
        "model.embed_tokens.weight": _tensor([256, 16]),
        "model.norm.weight": _tensor([16]),
    }
    for index in range(2):
        prefix = f"model.layers.{index}"
        query_heads = 2 if index == 0 else 4
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor([16]),
                f"{prefix}.post_attention_layernorm.weight": _tensor([16]),
                f"{prefix}.self_attn.q_proj.weight": _tensor([query_heads * 8, 16]),
                f"{prefix}.self_attn.k_proj.weight": _tensor([8, 16]),
                f"{prefix}.self_attn.v_proj.weight": _tensor([8, 16]),
                f"{prefix}.self_attn.o_proj.weight": _tensor([16, query_heads * 8]),
                f"{prefix}.self_attn.sinks": _tensor([query_heads]),
                f"{prefix}.block_sparse_moe.experts.gate_up_proj": _tensor([4, 12, 16]),
                f"{prefix}.block_sparse_moe.experts.down_proj": _tensor([4, 16, 6]),
                f"{prefix}.block_sparse_moe.router.weight": _tensor([4, 16]),
                f"{prefix}.shared_mlp.input_linear.weight": _tensor([20, 16]),
                f"{prefix}.shared_mlp.output_linear.weight": _tensor([16, 10]),
            }
        )
    config = {
        "model_type": "synthetic_mixed_window_decoder",
        "hidden_size": 16,
        "intermediate_size": 6,
        "shared_intermediate_size": 10,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_attention_heads_per_layer": [2, 4],
        "num_key_value_heads": 1,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 128,
        "vocab_size": 256,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    assert [layer.num_attention_heads for layer in structure.layers] == [2, 4]
    assert [layer.attention_window_size for layer in structure.layers] == [None, 128]
    assert [layer.shared_intermediate_size for layer in structure.layers] == [10, 10]

    full = make_layer(structure, structure.layers[0])
    sliding = make_layer(structure, structure.layers[1])
    assert full["state_ports"][0]["max_dynamic_activations"] is None
    assert sliding["state_ports"][0]["max_dynamic_activations"] == 128

    circuit = build_pedal_circuit(sliding, Path("layer_01.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["attention_read"]["params"] == ["attention_sinks"]
    assert nodes["attention_read"]["attrs"]["attention_sinks"] is True
    assert nodes["attention_read"]["attrs"]["window_size"] == 128
    assert nodes["moe_reduce"]["outputs"] == ["moe_out"]
    assert nodes["shared_mlp_input_projection"]["params"] == ["shared_mlp_input"]
    assert nodes["shared_mlp_split"]["attrs"] == {"part_width": 10}
    assert nodes["shared_mlp_activation"]["attrs"] == {"element_count": 10}
    assert nodes["shared_mlp_output_projection"]["params"] == ["shared_mlp_output"]
    assert nodes["shared_and_sparse_expert_add"]["outputs"] == ["ffn_out"]
    assert nodes["moe_topk"]["attrs"]["activation"] == "softmax"


def test_discovers_fused_qkv_and_gate_up_projections_by_shape() -> None:
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": _tensor([256, 16]),
        "model.norm.weight": _tensor([16]),
        "lm_head.weight": _tensor([256, 16]),
        f"{prefix}.input_layernorm.weight": _tensor([16]),
        f"{prefix}.post_attention_layernorm.weight": _tensor([16]),
        f"{prefix}.self_attn.qkv_proj.weight": _tensor([32, 16]),
        f"{prefix}.self_attn.o_proj.weight": _tensor([16, 16]),
        f"{prefix}.mlp.gate_up_proj.weight": _tensor([24, 16]),
        f"{prefix}.mlp.down_proj.weight": _tensor([16, 12]),
    }
    config = {
        "model_type": "synthetic_fused_decoder",
        "hidden_size": 16,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 256,
        "max_position_embeddings": 1024,
        "sliding_window": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    layer = structure.layers[0]
    assert layer.intermediate_size == 12
    assert layer.tensors["qkv_projection"].endswith("qkv_proj.weight")
    assert layer.tensors["ffn_gate_up"].endswith("gate_up_proj.weight")

    circuit = build_pedal_circuit(make_layer(structure, layer), Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["qkv_projection"]["params"] == ["qkv_projection"]
    assert nodes["qkv_split"]["attrs"] == {"part_widths": [16, 8, 8]}
    assert nodes["ffn_gate_up_projection"]["params"] == ["ffn_gate_up"]
    assert nodes["ffn_gate_up_split"]["attrs"] == {"part_width": 12}

