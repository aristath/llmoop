from model_structure_common import *
from model_structure_common import _tensor

def test_compiles_exact_yarn_frequency_and_attention_scaling() -> None:
    scaling = compile_rope_scaling(
        {
            "rope_type": "yarn",
            "rope_theta": 500_000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.3465735902799727,
        },
        64,
    )

    assert scaling == {
        "type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": True,
        "attention_factor": 1.3465735902799727,
        "correction_low": 9.0,
        "correction_high": 18.0,
    }


def test_discovers_model_owned_sampling_policy() -> None:
    assert discover_sampling_policy({}) == {
        "method": "greedy",
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert discover_sampling_policy(
        {
            "do_sample": True,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        }
    ) == {
        "method": "temperature_top_k_top_p",
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert discover_sampling_policy(
        {
            "do_sample": True,
            "temperature": 0.1,
            "top_k": 50,
            "min_p": 0.04,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.05,
        }
    ) == {
        "method": "temperature_top_k_top_p",
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": 0.04,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.05,
    }


def test_discovers_dynamic_block_fp8_by_numerical_structure() -> None:
    config = {
        "model_type": "outer_container",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_per_tensor": False,
            "act_per_tensor": False,
            "weight_block_size": [128, 128],
        },
        "text_config": {"model_type": "unrelated_family_name"},
    }

    assert discover_quantization_policy(config) == {
        "weight": {
            "format": "block_scaled_fp8_e4m3",
            "block_shape": [128, 128],
            "per_tensor": False,
        },
        "activation": {
            "format": "dynamic_block_fp8_e4m3",
            "group_size": 128,
            "per_tensor": False,
        },
    }


def test_does_not_invent_dynamic_activation_quantization() -> None:
    assert (
        discover_quantization_policy(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "static",
                    "weight_block_size": [128, 128],
                }
            }
        )
        is None
    )


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


def test_annotates_compressed_tensors_int4_storage_by_structure(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {
              "linear": {
                "format": "pack-quantized",
                "weights": {
                  "type": "int",
                  "num_bits": 4,
                  "group_size": 32,
                  "symmetric": true
                }
              }
            }
          }
        }"""
    )
    tensors = {
        "projection.weight_packed": _tensor([768, 64], "I32"),
        "projection.weight_scale": _tensor([768, 16], "BF16"),
        "projection.weight_shape": _tensor([2], "I64"),
    }

    annotate_packed_linear_tensors(tmp_path, tensors)
    parameters = {"projection": "projection.weight_packed"}
    attach_packed_linear_quantization(tensors, parameters)

    assert tensors["projection.weight_packed"]["logical_shape"] == [768, 512]
    assert tensors["projection.weight_packed"]["quantization"] == {
        "format": "compressed_tensors_pack_quantized",
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "signed_offset": 8,
        "scales": "projection.weight_scale",
    }
    assert parameters == {
        "projection": "projection.weight_packed",
        "projection_scales": "projection.weight_scale",
    }

