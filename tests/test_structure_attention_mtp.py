from model_structure_common import *
from model_structure_common import _tensor

def test_discovers_attention_with_values_derived_from_keys() -> None:
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": _tensor([1024, 512]),
        "model.norm.weight": _tensor([512]),
        f"{prefix}.input_layernorm.weight": _tensor([512]),
        f"{prefix}.post_attention_layernorm.weight": _tensor([512]),
        f"{prefix}.pre_feedforward_layernorm.weight": _tensor([512]),
        f"{prefix}.post_feedforward_layernorm.weight": _tensor([512]),
        f"{prefix}.layer_scalar": _tensor([1]),
        f"{prefix}.self_attn.q_proj.weight": _tensor([1024, 512]),
        f"{prefix}.self_attn.k_proj.weight": _tensor([256, 512]),
        f"{prefix}.self_attn.o_proj.weight": _tensor([512, 1024]),
        f"{prefix}.self_attn.q_norm.weight": _tensor([128]),
        f"{prefix}.self_attn.k_norm.weight": _tensor([128]),
        f"{prefix}.mlp.gate_proj.weight": _tensor([2048, 512]),
        f"{prefix}.mlp.up_proj.weight": _tensor([2048, 512]),
        f"{prefix}.mlp.down_proj.weight": _tensor([512, 2048]),
    }
    config = {
        "model_type": "synthetic_key_value_attention",
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "global_head_dim": 128,
        "layer_types": ["full_attention"],
        "attention_k_eq_v": True,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
            }
        },
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    layer = structure.layers[0]
    circuit = build_component_circuit(make_layer(structure, layer), Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}

    assert layer.attention_key_equals_value
    assert layer.head_width == 128
    assert layer.num_key_value_heads == 2
    assert layer.attention_scale == 1.0
    assert layer.value_head_norm
    assert structure.rms_norm_weight_offset == 0.0
    assert "v_projection" not in nodes
    assert nodes["q_head_norm"]["attrs"]["head_count"] == 8
    assert nodes["k_head_norm"]["attrs"]["head_count"] == 2
    assert nodes["v_head_norm"]["attrs"]["head_count"] == 2
    assert nodes["q_rope"]["attrs"]["head_count"] == 8
    assert nodes["k_rope"]["attrs"]["head_count"] == 2
    assert nodes["v_head_norm"]["inputs"] == ["k_projected"]
    assert nodes["kv_memory_append"]["inputs"][:2] == ["k_positioned", "v_normed"]
    assert nodes["ffn_residual"]["outputs"] == ["ffn_residual_out"]
    assert nodes["layer_scale"]["inputs"] == ["ffn_residual_out"]
    assert nodes["layer_scale"]["outputs"] == ["output_frame"]


def test_discovers_structural_mtp_as_auxiliary_execution_graph() -> None:
    tensors = {
        "model.embed_tokens.weight": _tensor([1024, 512]),
        "model.norm.weight": _tensor([512]),
        "lm_head.weight": _tensor([1024, 512]),
        "mtp.fc.weight": _tensor([512, 1024]),
        "mtp.pre_fc_norm_embedding.weight": _tensor([512]),
        "mtp.pre_fc_norm_hidden.weight": _tensor([512]),
        "mtp.norm.weight": _tensor([512]),
    }
    for prefix in ("model.layers.0", "mtp.layers.0"):
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor([512]),
                f"{prefix}.post_attention_layernorm.weight": _tensor([512]),
                f"{prefix}.self_attn.q_proj.weight": _tensor([512, 512]),
                f"{prefix}.self_attn.k_proj.weight": _tensor([256, 512]),
                f"{prefix}.self_attn.v_proj.weight": _tensor([256, 512]),
                f"{prefix}.self_attn.o_proj.weight": _tensor([512, 512]),
                f"{prefix}.self_attn.q_norm.weight": _tensor([128]),
                f"{prefix}.self_attn.k_norm.weight": _tensor([128]),
                f"{prefix}.mlp.gate_proj.weight": _tensor([2048, 512]),
                f"{prefix}.mlp.up_proj.weight": _tensor([2048, 512]),
                f"{prefix}.mlp.down_proj.weight": _tensor([512, 2048]),
            }
        )
    config = {
        "model_type": "synthetic_decoder",
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    assert len(structure.layers) == 1
    assert len(structure.draft_execution_graphs) == 1
    draft = structure.draft_execution_graphs[0]
    assert draft.id == "draft_00"
    assert draft.prefix == "mtp"
    assert draft.tensors == {
        "embedding_norm": "mtp.pre_fc_norm_embedding.weight",
        "hidden_norm": "mtp.pre_fc_norm_hidden.weight",
        "input_projection": "mtp.fc.weight",
        "output_norm": "mtp.norm.weight",
        "output_projection": "lm_head.weight",
    }
    assert len(draft.layers) == 1
    assert draft.layers[0].prefix == "mtp.layers.0"

    graph = make_model_graph(structure, Path("transpiled"), {"source": {}})
    [draft_graph] = graph["graph"]["draft_execution_graphs"]
    assert draft_graph["type"] == "multi_token_prediction"
    assert draft_graph["input_adapter"]["attrs"]["concatenation_order"] == [
        "token_embedding",
        "target_hidden",
    ]
    assert draft_graph["execution_graph"]["components"][0]["id"] == "draft_00_layer_00"
    assert draft_graph["state_contract"]["draft_updates"] == "tentative"

    draft_layer = make_layer(
        structure,
        draft.layers[0],
        component_id="draft_00_layer_00",
        runtime_role="draft_processor",
    )
    assert draft_layer["runtime_role"] == "draft_processor"
    assert draft_layer["transition_contract"]["reference_behavior"] == (
        "source_checkpoint_entity:mtp.layers.0"
    )

