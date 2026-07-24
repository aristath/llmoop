from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.model_package import (
    CAUSAL_SCAN_LANE_TILE_WIDTH,
    SCALAR_BATCH_LANE_TILE_WIDTH,
    ROW_MAJOR_LAYOUT,
    attention_workgroup_shape,
    causal_scan_batch_shader_file,
    causal_scan_batch_stages,
    causal_scan_workgroup_count_x,
    cooperative_bfloat16_batch_shader_file,
    cooperative_bfloat16_workgroup_count_x,
    cooperative_float8_e4m3_batch_shader_file,
    cooperative_float8_e4m3_workgroup_count_x,
    copy_shader_templates,
    frame_parallel_batch_shader_file,
    fp8_moe_block_shape_for_stage,
    component_kernel_spec,
    required_vulkan_device_extensions,
    required_vulkan_features,
    required_vulkan_subgroup_operations,
    shader_file_for_node,
    spirv_capabilities,
    workgroup_count_x_for_node,
    weight_shared_batch_shader_file,
    write_compiled_tensor,
)


def write_spirv_module(path: Path, capabilities: list[int]) -> None:
    words = [0x07230203, 0x00010600, 0, 1, 0]
    for capability in capabilities:
        words.extend([(2 << 16) | 17, capability])
    words.extend([(3 << 16) | 14, 0, 3 if 5345 in capabilities else 1])
    path.write_bytes(struct.pack(f"<{len(words)}I", *words))
