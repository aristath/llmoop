from __future__ import annotations

from pathlib import Path
from math import prod

from llmoop.circuit_executors import GQAAttentionCircuitPedal
from llmoop.circuit_lowering import build_pedal_circuit
from llmoop.model_transpiler import (
    attach_block_quantization_scales,
    attach_packed_linear_quantization,
    annotate_packed_linear_tensors,
    discover_model_structure,
    make_layer,
    synthesize_packed_expert_tensors,
)


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}


def test_attaches_block_scale_to_fp8_parameter_by_tensor_structure() -> None:
    tensors = {
        "projection.weight": _tensor([256, 512], "F8_E4M3"),
        "projection.weight_scale_inv": _tensor([2, 4]),
    }
    parameters = {"projection": "projection.weight"}

    attach_block_quantization_scales(tensors, parameters)

    assert parameters == {
        "projection": "projection.weight",
        "projection_scale_inv": "projection.weight_scale_inv",
    }


def test_annotates_auto_gptq_storage_as_logical_packed_linear(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "packing_format": "auto_round:auto_gptq",
            "bits": 4,
            "group_size": 128,
            "sym": true
          }
        }"""
    )
    tensors = {
        "projection.qweight": _tensor([64, 768], "I32"),
        "projection.qzeros": _tensor([4, 96], "I32"),
        "projection.scales": _tensor([4, 768], "F16"),
    }

    annotate_packed_linear_tensors(tmp_path, tensors)
    parameters = {"projection": "projection.qweight"}
    attach_packed_linear_quantization(tensors, parameters)

    assert tensors["projection.qweight"]["logical_shape"] == [768, 512]
    assert tensors["projection.qweight"]["quantization"] == {
        "format": "auto_gptq",
        "bits": 4,
        "group_size": 128,
        "symmetric": True,
        "zero_point_add": 1,
        "qzeros": "projection.qzeros",
        "scales": "projection.scales",
    }
    assert parameters == {
        "projection": "projection.qweight",
        "projection_qzeros": "projection.qzeros",
        "projection_scales": "projection.scales",
    }


def test_synthesizes_separate_experts_as_packed_circuit_parameters() -> None:
    tensors: dict[str, dict[str, object]] = {}

    def source_tensor(name: str, shape: list[int], dtype: str) -> None:
        byte_count = prod(shape) * (1 if dtype == "F8_E4M3" else 2)
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, byte_count],
            "parameter_count": prod(shape),
            "byte_count": byte_count,
            "source_file": "/tmp/source.safetensors",
            "source_header_bytes": 0,
        }

    prefix = "decoder.layers.0"
    for expert in range(2):
        base = f"{prefix}.mlp.experts.{expert}"
        for projection, shape in (
            ("gate_proj", [4, 8]),
            ("up_proj", [4, 8]),
            ("down_proj", [8, 4]),
        ):
            weight = f"{base}.{projection}.weight"
            source_tensor(weight, shape, "F8_E4M3")
            source_tensor(
                f"{weight}_scale_inv",
                [shape[0] // 2, shape[1] // 2],
                "BF16",
            )

    synthesize_packed_expert_tensors(tensors, prefix)

    assert tensors[f"{prefix}.mlp.experts.gate_up_proj"]["shape"] == [2, 8, 8]
    assert tensors[f"{prefix}.mlp.experts.down_proj"]["shape"] == [2, 8, 4]
    assert tensors[f"{prefix}.mlp.experts.gate_up_proj_scale_inv"]["shape"] == [
        2,
        4,
        4,
    ]
    assert len(tensors[f"{prefix}.mlp.experts.gate_up_proj"]["source_parts"]) == 4


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


def test_discovers_mixed_window_attention_sinks_and_shared_sparse_experts() -> None:
    tensors = {
        "model.embed_tokens.weight": _tensor([256, 16]),
        "model.norm.weight": _tensor([16]),
    }
    for index in range(2):
        prefix = f"model.layers.{index}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor([16]),
                f"{prefix}.post_attention_layernorm.weight": _tensor([16]),
                f"{prefix}.self_attn.q_proj.weight": _tensor([16, 16]),
                f"{prefix}.self_attn.k_proj.weight": _tensor([8, 16]),
                f"{prefix}.self_attn.v_proj.weight": _tensor([8, 16]),
                f"{prefix}.self_attn.o_proj.weight": _tensor([16, 16]),
                f"{prefix}.self_attn.sinks": _tensor([2]),
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
    assert [layer.attention_window_size for layer in structure.layers] == [None, 128]
    assert [layer.shared_intermediate_size for layer in structure.layers] == [10, 10]

    full = make_layer(structure, structure.layers[0])
    sliding = make_layer(structure, structure.layers[1])
    assert full["state_ports"][0]["window_size"] is None
    assert sliding["state_ports"][0]["window_size"] == 128

    circuit = build_pedal_circuit(sliding, Path("layer_01.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["attention_read"]["params"] == ["attention_sinks"]
    assert nodes["attention_read"]["attrs"]["attention_sinks"] is True
    assert nodes["attention_read"]["attrs"]["window_size"] == 128
    assert nodes["moe_reduce"]["outputs"] == ["moe_out"]
    assert nodes["shared_mlp_input_projection"]["params"] == ["shared_mlp_input"]
    assert nodes["shared_mlp_split"]["attrs"] == {"part_width": 10}
    assert nodes["shared_mlp_output_projection"]["params"] == ["shared_mlp_output"]
    assert nodes["shared_and_sparse_expert_add"]["outputs"] == ["ffn_out"]


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
    assert structure.intermediate_size == 12
    assert layer.tensors["qkv_projection"].endswith("qkv_proj.weight")
    assert layer.tensors["ffn_gate_up"].endswith("gate_up_proj.weight")

    circuit = build_pedal_circuit(make_layer(structure, layer), Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["qkv_projection"]["params"] == ["qkv_projection"]
    assert nodes["qkv_split"]["attrs"] == {"part_widths": [16, 8, 8]}
    assert nodes["ffn_gate_up_projection"]["params"] == ["ffn_gate_up"]
    assert nodes["ffn_gate_up_split"]["attrs"] == {"part_width": 12}


def test_discovers_multimodal_decoder_with_per_layer_inputs_and_shared_kv() -> None:
    language_root = "model.language_model"
    tensors = {
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight": _tensor([64, 16]),
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_2.linear.weight": _tensor([16, 64]),
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
