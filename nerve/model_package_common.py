from __future__ import annotations

import json
import math
import re
import shutil
import struct
import subprocess
from copy import deepcopy
from hashlib import blake2s, sha256
from pathlib import Path
from typing import Any, Callable

from nerve.behavioral_compiler import (
    build_behavioral_validation,
    validate_behavioral_validation_artifact,
)
from nerve.circuit_ir import validate_circuit
from nerve.circuit_lowering import lower_execution_graph
from nerve.circuit_optimizer import optimize_circuit_for_vulkan
from nerve.compilation import (
    DEFAULT_COMPILED_MODELS_DIR,
    PACKAGE_SCHEMA,
    CompiledModelReport,
    Json,
    ModelCompileError,
    check_compile_cancelled,
    emit_compile_event,
    read_json,
    relative_json_path,
    write_json,
)
from nerve.compiler_fingerprint import (
    COMPILER_FINGERPRINT_SCHEMA,
    package_compiler_fingerprint,
)
from nerve.model_transpiler import read_safetensors_header, transpile_model


TOKENIZER_PACKAGE_DIR = "tokenizer"
WEIGHTS_PACKAGE_DIR = "weights"
PACKAGE_ARTIFACT_INTEGRITY_SCHEMA = "nerve.package_artifact_integrity.v1"
ROW_MAJOR_LAYOUT = "row_major"
CONFIG_PACKAGE_FILE = "config.json"
SCALAR_BATCH_LANE_TILE_WIDTH = 16
EXACT_BATCH_LANE_TILE_WIDTHS = (2, 4, 8, SCALAR_BATCH_LANE_TILE_WIDTH)
CAUSAL_SCAN_LANE_TILE_WIDTH = 64
FP8_SPARSE_GATE_UP_TILE_ROWS = 32
FP8_SPARSE_DOWN_TILE_ROWS = 64
FP8_LINEAR_TILE_ROWS = (16,)
FP8_OUTPUT_PROJECTION_TILE_ROWS = 32
FP8_LINEAR_MIN_WORKGROUPS = 1
FP8_FUSED_FFN_TILE_ROWS = 16
Q8_0_GROUP_SIZE = 32
Q8_0_BLOCK_BYTE_COUNT = 36
INT4_VALUES_PER_PACKED_WORD = 8
INT4_GPTQ_OUTPUT_TILE_ROWS = 64
INT4_CT_OUTPUT_TILE_ROWS = 16
GLSL_VULKAN_DEVICE_EXTENSION_REQUIREMENTS = {
    "GL_EXT_float_e4m3": "VK_EXT_shader_float8",
    "GL_EXT_bfloat16": "VK_KHR_shader_bfloat16",
    "GL_KHR_cooperative_matrix": "VK_KHR_cooperative_matrix",
}
SPIRV_VULKAN_DEVICE_EXTENSION_REQUIREMENTS = {
    "SPV_VALVE_mixed_float_dot_product": "VK_VALVE_shader_mixed_float_dot_product",
}
SPIRV_MAGIC = 0x07230203
SPIRV_OP_CAPABILITY = 17
SPIRV_CAPABILITY_VULKAN_FEATURE_REQUIREMENTS = {
    9: "shader_float16",
    10: "shader_float64",
    11: "shader_int64",
    22: "shader_int16",
    39: "shader_int8",
    4212: "shader_float8",
    4213: "shader_float8_cooperative_matrix",
    4433: "storage_buffer16_bit_access",
    4434: "uniform_and_storage_buffer16_bit_access",
    4435: "storage_push_constant16",
    4436: "storage_input_output16",
    4448: "storage_buffer8_bit_access",
    4449: "uniform_and_storage_buffer8_bit_access",
    4450: "storage_push_constant8",
    5116: "shader_bfloat16_type",
    5117: "shader_bfloat16_dot_product",
    5118: "shader_bfloat16_cooperative_matrix",
    6019: "shader_integer_dot_product",
    6915: "shader_mixed_float_dot_product_float8_acc_float32",
    5345: "vulkan_memory_model",
    5346: "vulkan_memory_model_device_scope",
    6022: "cooperative_matrix",
}
KNOWN_VULKAN_FEATURES = frozenset(SPIRV_CAPABILITY_VULKAN_FEATURE_REQUIREMENTS.values())
SPIRV_CAPABILITY_VULKAN_SUBGROUP_OPERATION_REQUIREMENTS = {
    61: "basic",
    62: "vote",
    63: "arithmetic",
    64: "ballot",
    65: "shuffle",
    66: "shuffle_relative",
    67: "clustered",
    68: "quad",
}
KNOWN_VULKAN_SUBGROUP_OPERATIONS = frozenset(
    SPIRV_CAPABILITY_VULKAN_SUBGROUP_OPERATION_REQUIREMENTS.values()
)
KNOWN_COMPONENT_KERNEL_EXECUTION_DOMAINS = frozenset(
    {"decode", "prefill", "decode_and_prefill"}
)
SUPPORTED_SPIRV_CAPABILITIES = frozenset(
    {0, 1}
    | SPIRV_CAPABILITY_VULKAN_FEATURE_REQUIREMENTS.keys()
    | SPIRV_CAPABILITY_VULKAN_SUBGROUP_OPERATION_REQUIREMENTS.keys()
)
COOPERATIVE_BFLOAT16_SHAPE = [16, 16, 16]
COOPERATIVE_BATCH_LANE_TILE_WIDTH = 64
COOPERATIVE_OUTPUT_TILE_WIDTH = 64
COOPERATIVE_FUSED_OUTPUT_TILE_WIDTH = 64
TOKENIZER_PACKAGE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def package_artifact_path(package_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"compiled package has no {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCompileError(
            f"compiled package {label} path must stay inside the package: {value!r}"
        )
    return package_dir / relative
