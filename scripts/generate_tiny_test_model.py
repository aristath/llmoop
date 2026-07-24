#!/usr/bin/env python3
"""Generate the checked-in tiny compiled package used by Rust Vulkan tests."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "runtime-rs" / "test-fixtures" / "tiny_model"
sys.path.insert(0, str(ROOT))

from nerve.compiler_target import CompilerTarget  # noqa: E402
from nerve.model_compiler import compile_model  # noqa: E402


def tensor(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(offset, offset + count, dtype=torch.float32)
    return ((values.remainder(29) - 14) / 29).reshape(shape).contiguous()


def write_source_model(source: Path) -> None:
    source.mkdir(parents=True)
    config = {
        "architectures": ["TinyTestForCausalLM"],
        "model_type": "tiny_test",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "vocab_size": 32,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "layer_types": ["full_attention"],
    }
    (source / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (source / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": False,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            },
            indent=2,
        )
        + "\n"
    )

    tensors = {
        "model.embed_tokens.weight": tensor((32, 16), 0).to(torch.bfloat16),
        "model.norm.weight": torch.ones(16, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(16, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(
            16, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.q_proj.weight": tensor((16, 16), 512).to(
            torch.bfloat16
        ),
        "model.layers.0.self_attn.k_proj.weight": tensor((8, 16), 768).to(
            torch.bfloat16
        ),
        "model.layers.0.self_attn.v_proj.weight": tensor((8, 16), 896).to(
            torch.bfloat16
        ),
        "model.layers.0.self_attn.o_proj.weight": tensor((16, 16), 1024).to(
            torch.bfloat16
        ),
        "model.layers.0.mlp.gate_proj.weight": tensor((32, 16), 1280).to(
            torch.bfloat16
        ),
        "model.layers.0.mlp.up_proj.weight": tensor((32, 16), 1792).to(
            torch.bfloat16
        ),
        "model.layers.0.mlp.down_proj.weight": tensor((16, 32), 2304).to(
            torch.bfloat16
        ),
    }
    save_file(tensors, source / "model.safetensors")

    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        **{f"token_{index:02d}": index for index in range(4, 32)},
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nerve-tiny-test-source-") as temporary:
        source = Path(temporary) / "source"
        write_source_model(source)
        shutil.rmtree(DESTINATION, ignore_errors=True)
        compile_model(
            source,
            compiled_model_dir=DESTINATION,
            shader_source_dir=ROOT / "runtime-rs" / "shaders",
            target=CompilerTarget.for_features(("shader_bfloat16_type",)),
        )


if __name__ == "__main__":
    main()
