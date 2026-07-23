from model_structure_common import *
from model_structure_common import _tensor

def test_discovers_multimodal_decoder_with_per_layer_inputs_and_shared_kv() -> None:
    language_root = "model.language_model"
    tensors = {
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight": _tensor(
            [64, 16]
        ),
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_2.linear.weight": _tensor(
            [16, 64]
        ),
        f"{language_root}.embed_tokens.weight": _tensor([256, 16]),
        f"{language_root}.embed_tokens_per_layer.weight": _tensor([256, 8]),
        f"{language_root}.per_layer_model_projection.weight": _tensor([8, 16]),
        f"{language_root}.per_layer_projection_norm.weight": _tensor([2]),
        f"{language_root}.norm.weight": _tensor([16]),
    }
    layer_types = [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    for index, layer_type in enumerate(layer_types):
        prefix = f"{language_root}.layers.{index}"
        head_width = 8 if layer_type == "full_attention" else 4
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor([16]),
                f"{prefix}.post_attention_layernorm.weight": _tensor([16]),
                f"{prefix}.pre_feedforward_layernorm.weight": _tensor([16]),
                f"{prefix}.post_feedforward_layernorm.weight": _tensor([16]),
                f"{prefix}.post_per_layer_input_norm.weight": _tensor([16]),
                f"{prefix}.layer_scalar": _tensor([1]),
                f"{prefix}.per_layer_input_gate.weight": _tensor([2, 16]),
                f"{prefix}.per_layer_projection.weight": _tensor([16, 2]),
                f"{prefix}.self_attn.q_proj.weight": _tensor([2 * head_width, 16]),
                f"{prefix}.self_attn.k_proj.weight": _tensor([head_width, 16]),
                f"{prefix}.self_attn.v_proj.weight": _tensor([head_width, 16]),
                f"{prefix}.self_attn.o_proj.weight": _tensor([16, 2 * head_width]),
                f"{prefix}.self_attn.q_norm.weight": _tensor([head_width]),
                f"{prefix}.self_attn.k_norm.weight": _tensor([head_width]),
                f"{prefix}.mlp.gate_proj.weight": _tensor([32, 16]),
                f"{prefix}.mlp.up_proj.weight": _tensor([32, 16]),
                f"{prefix}.mlp.down_proj.weight": _tensor([16, 32]),
            }
        )
    config = {
        "model_type": "synthetic_multimodal",
        "audio_config": {"num_hidden_layers": 1},
        "text_config": {
            "model_type": "synthetic_ple_decoder",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 8,
            "hidden_size_per_layer_input": 2,
            "num_kv_shared_layers": 2,
            "layer_types": layer_types,
            "sliding_window": 32,
            "vocab_size": 256,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)

    assert structure.model_type == "synthetic_ple_decoder"
    assert structure.embedding_scale == 4.0
    assert structure.logits_soft_cap == 30.0
    assert [layer.head_width for layer in structure.layers] == [4, 8, 4, 8]
    assert [layer.rotary_width for layer in structure.layers] == [4, 2, 4, 2]
    assert [layer.rope_type for layer in structure.layers] == [
        "default",
        "proportional",
        "default",
        "proportional",
    ]
    assert [layer.attention_scale for layer in structure.layers] == [1.0] * 4
    assert [layer.shared_kv_source_layer for layer in structure.layers] == [
        None,
        None,
        0,
        1,
    ]
    assert structure.layers[0].value_head_norm is True
    assert structure.layers[2].value_head_norm is False
    assert "operator_post_norm" in structure.layers[0].tensors
    assert "ffn_post_norm" in structure.layers[0].tensors
    assert structure.layers[0].per_layer_input_width == 2
    assert "k_projection" not in structure.layers[2].tensors

    packed_name = f"{language_root}.embed_tokens_per_layer.weight"
    tensor_index = {
        "tensors": {
            packed_name: {
                "dtype": "BF16",
                "shape": [150_000_000, 8],
                "data_offsets": [64, 2_400_000_064],
                "parameter_count": 1_200_000_000,
                "byte_count": 2_400_000_000,
                "source_file": "/models/source.safetensors",
                "source_header_bytes": 128,
            }
        }
    }
    segment_per_layer_embedding_parameters(structure, tensor_index)
    chunk_names = [
        structure.layers[0].tensors[f"per_layer_embedding_chunk_{index}"]
        for index in range(3)
    ]
    assert all(
        layer.tensors[f"per_layer_embedding_chunk_{index}"] == chunk_names[index]
        for layer in structure.layers
        for index in range(3)
    )
    assert all("per_layer_embedding" not in layer.tensors for layer in structure.layers)
    chunks = [tensor_index["tensors"][name] for name in chunk_names]
    assert [chunk["shape"][0] for chunk in chunks] == [
        67_108_864,
        67_108_864,
        15_782_272,
    ]
    assert chunks[0]["data_offsets"][0] == 64
    assert chunks[-1]["data_offsets"][1] == 2_400_000_064

