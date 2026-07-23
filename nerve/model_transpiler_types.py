from __future__ import annotations

import json
import math
import re
import shutil
import struct
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from nerve.compilation import check_compile_cancelled, read_json, write_json


Json = dict[str, Any]

# Vulkan storage-buffer ranges are commonly capped below 4 GiB even when the
# backing allocation is larger. Keep compiled shader-visible parameter chunks
# comfortably below that boundary and align chunks to complete tensor rows.
MAX_SHADER_PARAMETER_CHUNK_BYTES = 1 << 30


class ModelTranspileError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayerStructure:
    index: int
    prefix: str
    operator_type: str
    attention_window_size: int | None
    num_attention_heads: int
    num_key_value_heads: int
    head_width: int
    rotary_width: int
    rope_theta: float
    rope_type: str
    rope_scaling: Json | None
    attention_scale: float
    attention_gate_activation: str | None
    attention_gate_per_head: bool
    attention_key_equals_value: bool
    value_head_norm: bool
    shared_kv_source_layer: int | None
    per_layer_input_width: int | None
    feed_forward_type: str
    intermediate_size: int
    shared_intermediate_size: int | None
    tensors: dict[str, str]


@dataclass(frozen=True)
class DraftExecutionGraphStructure:
    id: str
    prefix: str
    tensors: dict[str, str]
    layers: tuple[LayerStructure, ...]


@dataclass(frozen=True)
class ModelStructure:
    model_dir: Path
    model_type: str | None
    architectures: tuple[str, ...]
    dtype: str | None
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_width: int
    rotary_width: int
    attention_window_size: int | None
    attention_output_gate: bool
    conv_l_cache: int
    vocab_size: int
    max_position_embeddings: int | None
    norm_eps: float
    rope_theta: float
    rope_interleaved: bool
    rms_norm_weight_offset: float
    embedding_scale: float
    residual_scale: float
    attention_scale: float
    logits_scale: float
    logits_soft_cap: float | None
    activation: str
    num_experts: int | None
    experts_per_token: int | None
    moe_routing: Json | None
    recurrent_mixer: Json | None
    quantization: Json | None
    sampling: Json
    token_ids: Json
    tensors: dict[str, str]
    layers: tuple[LayerStructure, ...]
    draft_execution_graphs: tuple[DraftExecutionGraphStructure, ...]


TOKEN_EMBEDDING_CANDIDATES = (
    "model.embed_tokens.weight",
    "model.tok_embeddings.weight",
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
)

OUTPUT_NORM_CANDIDATES = (
    "model.embedding_norm.weight",
    "model.norm.weight",
    "model.final_norm.weight",
    "model.final_layernorm.weight",
    "transformer.ln_f.weight",
    "gpt_neox.final_layer_norm.weight",
)

OUTPUT_PROJECTION_CANDIDATES = (
    "lm_head.weight",
    "output.weight",
    "embed_out.weight",
)

OPERATOR_NORM_SUFFIXES = (
    "operator_norm.weight",
    "input_layernorm.weight",
    "self_attn_layer_norm.weight",
    "temporal_pre_norm.weight",
)

FFN_NORM_SUFFIXES = (
    "ffn_norm.weight",
    "post_attention_layernorm.weight",
    "post_attention_layer_norm.weight",
    "channel_pre_norm.weight",
)

OPERATOR_POST_NORM_SUFFIXES = ("post_attention_layernorm.weight",)
FFN_PRE_NORM_SUFFIXES = ("pre_feedforward_layernorm.weight",)
FFN_POST_NORM_SUFFIXES = ("post_feedforward_layernorm.weight",)
PER_LAYER_INPUT_GATE_SUFFIXES = ("per_layer_input_gate.weight",)
PER_LAYER_PROJECTION_SUFFIXES = ("per_layer_projection.weight",)
PER_LAYER_POST_NORM_SUFFIXES = ("post_per_layer_input_norm.weight",)
LAYER_SCALAR_SUFFIXES = ("layer_scalar",)

FFN_GATE_SUFFIXES = (
    "feed_forward.w1.weight",
    "mlp.gate_proj.weight",
    "feed_forward.gate_proj.weight",
    "mlp_block.gate_proj.weight",
)

FFN_DOWN_SUFFIXES = (
    "feed_forward.w2.weight",
    "mlp.down_proj.weight",
    "feed_forward.down_proj.weight",
    "mlp_block.down_proj.weight",
)

FFN_UP_SUFFIXES = (
    "feed_forward.w3.weight",
    "mlp.up_proj.weight",
    "feed_forward.up_proj.weight",
    "mlp_block.up_proj.weight",
)

FFN_FUSED_GATE_UP_SUFFIXES = ("mlp.gate_up_proj.weight",)

MOE_INPUT_SUFFIXES = (
    "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.experts.gate_up_proj",
    "mlp.experts.gate_up_proj",
)
MOE_OUTPUT_SUFFIXES = (
    "block_sparse_moe.output_linear.weight",
    "block_sparse_moe.experts.down_proj",
    "mlp.experts.down_proj",
)
MOE_ROUTER_SUFFIXES = (
    "block_sparse_moe.router.layer.weight",
    "block_sparse_moe.router.weight",
    "block_sparse_moe.gate.weight",
    "mlp.gate.weight",
)
MOE_ROUTER_CORRECTION_BIAS_SUFFIXES = (
    "mlp.experts.e_score_correction_bias",
    "mlp.gate.e_score_correction_bias",
)

SHARED_MLP_INPUT_SUFFIXES = (
    "shared_mlp.input_linear.weight",
    "mlp.shared_expert.gate_up_proj",
)
SHARED_MLP_OUTPUT_SUFFIXES = (
    "shared_mlp.output_linear.weight",
    "mlp.shared_expert.down_proj.weight",
)
SHARED_MLP_GATE_SUFFIXES = ("mlp.shared_expert_gate.weight",)

CONV_IN_PROJECTION_SUFFIXES = ("conv.in_proj.weight",)
CONV_KERNEL_SUFFIXES = ("conv.conv.weight", "conv.depthwise.weight")
CONV_OUT_PROJECTION_SUFFIXES = ("conv.out_proj.weight",)

ATTENTION_Q_PROJECTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "attention.wq.weight",
    "temporal_block.q_proj.weight",
)
ATTENTION_FUSED_QKV_SUFFIXES = ("self_attn.qkv_proj.weight",)
ATTENTION_K_PROJECTION_SUFFIXES = (
    "self_attn.k_proj.weight",
    "attention.wk.weight",
    "temporal_block.k_proj.weight",
)
ATTENTION_V_PROJECTION_SUFFIXES = (
    "self_attn.v_proj.weight",
    "attention.wv.weight",
    "temporal_block.v_proj.weight",
)
ATTENTION_OUT_PROJECTION_SUFFIXES = (
    "self_attn.out_proj.weight",
    "self_attn.o_proj.weight",
    "attention.wo.weight",
    "temporal_block.o_proj.weight",
)
ATTENTION_Q_NORM_SUFFIXES = ("self_attn.q_layernorm.weight", "self_attn.q_norm.weight")
ATTENTION_K_NORM_SUFFIXES = ("self_attn.k_layernorm.weight", "self_attn.k_norm.weight")
ATTENTION_SINK_SUFFIXES = ("self_attn.sinks",)
ATTENTION_GATE_PROJECTION_SUFFIXES = ("self_attn.g_proj.weight",)

GATED_DELTA_QKV_SUFFIXES = ("linear_attn.in_proj_qkv.weight",)
GATED_DELTA_Z_SUFFIXES = ("linear_attn.in_proj_z.weight",)
GATED_DELTA_B_SUFFIXES = ("linear_attn.in_proj_b.weight",)
GATED_DELTA_A_SUFFIXES = ("linear_attn.in_proj_a.weight",)
GATED_DELTA_CONV_SUFFIXES = ("linear_attn.conv1d.weight",)
GATED_DELTA_A_LOG_SUFFIXES = ("linear_attn.A_log",)
GATED_DELTA_DT_BIAS_SUFFIXES = ("linear_attn.dt_bias",)
GATED_DELTA_NORM_SUFFIXES = ("linear_attn.norm.weight",)
GATED_DELTA_OUT_SUFFIXES = ("linear_attn.out_proj.weight",)

RG_LRU_X_SUFFIXES = ("temporal_block.linear_x.weight",)
RG_LRU_Y_SUFFIXES = ("temporal_block.linear_y.weight",)
RG_LRU_OUT_SUFFIXES = ("temporal_block.linear_out.weight",)
RG_LRU_CONV_SUFFIXES = ("temporal_block.conv_1d.weight",)
RG_LRU_INPUT_GATE_WEIGHT_SUFFIXES = ("temporal_block.rg_lru.input_gate_weight",)
RG_LRU_INPUT_GATE_BIAS_SUFFIXES = ("temporal_block.rg_lru.input_gate_bias",)
RG_LRU_RECURRENT_GATE_WEIGHT_SUFFIXES = ("temporal_block.rg_lru.recurrent_gate_weight",)
RG_LRU_RECURRENT_GATE_BIAS_SUFFIXES = ("temporal_block.rg_lru.recurrent_gate_bias",)
RG_LRU_RECURRENT_PARAM_SUFFIXES = ("temporal_block.rg_lru.recurrent_param",)

LAYER_ROOT_PATTERNS = (
    re.compile(r"^(?P<root>.+?\.layers)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<root>transformer\.h)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<root>gpt_neox\.layers)\.(?P<index>\d+)\."),
)

DRAFT_INPUT_PROJECTION_SUFFIXES = (
    "fc.weight",
    "eh_proj.weight",
)
DRAFT_EMBEDDING_NORM_SUFFIXES = (
    "pre_fc_norm_embedding.weight",
    "embedding_norm.weight",
    "enorm.weight",
)
DRAFT_HIDDEN_NORM_SUFFIXES = (
    "pre_fc_norm_hidden.weight",
    "hidden_norm.weight",
    "hnorm.weight",
)
DRAFT_OUTPUT_NORM_SUFFIXES = (
    "norm.weight",
    "shared_head_norm.weight",
)


