from __future__ import annotations

from pathlib import Path
from math import prod

from nerve.circuit_lowering import build_pedal_circuit
from nerve.model_transpiler import (
    attach_block_quantization_scales,
    attach_packed_linear_quantization,
    annotate_packed_linear_tensors,
    compile_rope_scaling,
    discover_model_structure,
    discover_quantization_policy,
    discover_sampling_policy,
    make_layer,
    make_model_graph,
    segment_per_layer_embedding_parameters,
    synthesize_packed_expert_tensors,
)


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}

