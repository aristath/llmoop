from nerve.model_package_common import *

def frame_parallel_batch_shader_file(shader_file: str) -> str | None:
    if re.fullmatch(r"moe_topk_bf16_e\d+_k\d+\.comp", shader_file):
        return shader_file.replace("moe_topk_", "moe_topk_batch1_", 1)
    if re.fullmatch(
        r"moe_topk_(?:sigmoid|softmax)_bf16_e\d+_k\d+_norm[01]_"
        r"cap[0-9eE+.-]+_(?:biasf32|biasbf16|nobias)\.comp",
        shader_file,
    ):
        return shader_file.replace("moe_topk_", "moe_topk_batch1_", 1)
    if re.fullmatch(
        r"sparse_moe_(?:gate_up|down)_(?:bf16|fp8_e4m3_b\d+x\d+|"
        r"int4_ct_s(?:f16|bf16)_g\d+)_"
        r"h\d+_i\d+_e\d+_k\d+\.comp",
        shader_file,
    ):
        return (
            shader_file.replace("_bf16_", "_batch1_bf16_", 1)
            .replace("_fp8_e4m3_", "_batch1_fp8_e4m3_", 1)
            .replace("_int4_ct_", "_batch1_int4_ct_", 1)
        )
    if re.fullmatch(
        r"parallel_linear_[23]way_fp8_e4m3_b\d+x\d+_\d+x\d+_\d+(?:_\d+)?\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "parallel_linear_",
            "parallel_linear_batch1_",
            1,
        )
    if re.fullmatch(
        r"moe_reduce_bf16_h\d+_k\d+_scale[0-9eE+.-]+\.comp", shader_file
    ):
        return shader_file.replace("moe_reduce_", "moe_reduce_batch1_", 1)
    if re.fullmatch(
        r"rms_norm_batch\d+_bf16_h\d+_eps[0-9eE+.-]+_offset[0-9eE+.-]+\.comp",
        shader_file,
    ) or re.fullmatch(
        r"split_batch\d+_bf16_2x\d+x\d+_head_interleaved\.comp",
        shader_file,
    ):
        return re.sub(r"_batch\d+_", "_batch1_", shader_file, count=1)
    return None



def causal_scan_batch_stages(shader_file: str, local_size_x: int) -> list[Json] | None:
    causal_scan_shader = causal_scan_batch_shader_file(shader_file)
    if causal_scan_shader is not None:
        return [
            {
                "shader_path": f"shaders/{causal_scan_shader}",
                "local_size_x": local_size_x,
                "workgroup_count_x": causal_scan_workgroup_count_x(shader_file),
            }
        ]

    attention = re.fullmatch(
        r"append_gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)"
        r"(?:_w(\d+))?(_sinks)?__sc\d+\.comp",
        shader_file,
    )
    if attention is None:
        return None
    query_heads, kv_heads, head_width = map(int, attention.groups()[:3])
    stem = re.sub(r"__sc\d+\.comp$", ".comp", shader_file).replace(
        "append_gqa_attention_bf16_",
        "append_gqa_attention_temporal_read_bf16_",
        1,
    )
    sinks = "_sinks" if attention.group(6) else ""
    attention_window = attention.group(5) or "0"
    return [
        {
            "shader_path": f"shaders/{stem}",
            "local_size_x": local_size_x,
            "workgroup_count_x": query_heads * CAUSAL_SCAN_LANE_TILE_WIDTH,
        },
        {
            "shader_path": (
                "shaders/append_kv_temporal_commit_bf16_"
                f"kv{kv_heads}_d{head_width}_w{attention_window}{sinks}.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": kv_heads,
        },
    ]


def cooperative_bfloat16_batch_shader_file(shader_file: str) -> str | None:
    linear = re.fullmatch(
        r"(linear|linear_residual)_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if linear is not None:
        operation, input_size, output_size = linear.groups()
        return f"{operation}_batch64_cooperative_bf16_{input_size}x{output_size}.comp"
    parallel = re.fullmatch(
        r"parallel_linear_[23]way_bf16_\d+x.+\.comp",
        shader_file,
    )
    if parallel is not None:
        return shader_file.replace(
            "parallel_linear_",
            "parallel_linear_batch64_cooperative_",
            1,
        )
    fused = re.fullmatch(
        r"parallel_linear_silu_multiply_bf16_\d+x\d+\.comp",
        shader_file,
    )
    if fused is not None:
        return shader_file.replace(
            "parallel_linear_silu_multiply_",
            "parallel_linear_silu_multiply_batch64_cooperative_",
            1,
        )
    return None


def cooperative_bfloat16_workgroup_count_x(shader_file: str) -> int:
    linear = re.fullmatch(
        r"(?:linear|linear_residual)_bf16_\d+x(\d+)\.comp",
        shader_file,
    )
    if linear is not None:
        return (
            int(linear.group(1)) + COOPERATIVE_OUTPUT_TILE_WIDTH - 1
        ) // COOPERATIVE_OUTPUT_TILE_WIDTH
    parallel = re.fullmatch(
        r"parallel_linear_[23]way_bf16_\d+x"
        r"(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel is not None:
        return sum(
            (int(width) + COOPERATIVE_OUTPUT_TILE_WIDTH - 1)
            // COOPERATIVE_OUTPUT_TILE_WIDTH
            for width in parallel.groups()
            if width is not None
        )
    fused = re.fullmatch(
        r"parallel_linear_silu_multiply_bf16_"
        r"\d+x(\d+)\.comp",
        shader_file,
    )
    if fused is not None:
        return (
            int(fused.group(1)) + COOPERATIVE_FUSED_OUTPUT_TILE_WIDTH - 1
        ) // COOPERATIVE_FUSED_OUTPUT_TILE_WIDTH
    raise ModelCompileError(
        f"shader {shader_file!r} has no cooperative BF16 batch geometry"
    )


def cooperative_float8_e4m3_batch_shader_file(
    shader_file: str,
    *,
    shape: tuple[int, int, int],
) -> str | None:
    linear = re.fullmatch(
        r"(linear|linear_residual)_prequant_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if linear is None:
        return None
    operation, block_rows, block_columns, input_size, output_size = linear.groups()
    m, n, k = shape
    if (
        min(shape) <= 0
        or int(block_columns) % k
        or int(input_size) % int(block_columns)
    ):
        return None
    batch_tile_width = 4 * n
    return (
        f"{operation}_prequant_batch{batch_tile_width}_cooperative_"
        f"fp8_e4m3_m{m}n{n}k{k}_b{block_rows}x{block_columns}_"
        f"{input_size}x{output_size}.comp"
    )


def cooperative_float8_e4m3_workgroup_count_x(
    shader_file: str,
    *,
    shape: tuple[int, int, int],
) -> int:
    linear = re.fullmatch(
        r"(?:linear|linear_residual)_prequant_fp8_e4m3_"
        r"b\d+x\d+_\d+x(\d+)\.comp",
        shader_file,
    )
    if linear is None:
        raise ModelCompileError(
            f"shader {shader_file!r} has no cooperative FP8 batch geometry"
        )
    output_size = int(linear.group(1))
    output_tile = 4 * shape[0]
    return (output_size + output_tile - 1) // output_tile


def causal_scan_batch_shader_file(shader_file: str) -> str | None:
    if re.fullmatch(r"causal_conv1d_silu_bf16_c\d+_k\d+\.comp", shader_file):
        return shader_file.replace(
            "causal_conv1d_silu_bf16_",
            "causal_conv1d_silu_temporal_bf16_",
            1,
        )
    if re.fullmatch(
        r"gated_delta_step_k\d+x\d+_v\d+x\d+"
        r"_a(?:f32|bf16)_dt(?:f32|bf16)_n(?:f32|bf16)_eps[0-9eE+.-]+"
        r"(?:_qfp8b\d+)?\.comp",
        shader_file,
    ):
        return shader_file.replace("gated_delta_step_", "gated_delta_scan_", 1)
    if re.fullmatch(
        r"parallel_head_norm_rope_2way_bf16_h\d+_\d+_d\d+_r\d+"
        r"_eps[0-9eE+.-]+_offset[0-9eE+.-]+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved|proportional)__sc\d+\.comp",
        shader_file,
    ):
        return re.sub(
            r"__sc\d+\.comp$",
            ".comp",
            shader_file.replace(
                "parallel_head_norm_rope_2way_",
                "parallel_head_norm_rope_2way_temporal_",
                1,
            ),
        )
    return None


def causal_scan_workgroup_count_x(shader_file: str) -> int:
    causal_conv = re.fullmatch(
        r"causal_conv1d_silu_bf16_c(\d+)_k\d+\.comp", shader_file
    )
    if causal_conv is not None:
        channels = int(causal_conv.group(1))
        return (channels + 127) // 128
    gated_delta = re.fullmatch(
        r"gated_delta_step_k\d+x\d+_v(\d+)x\d+"
        r"_a(?:f32|bf16)_dt(?:f32|bf16)_n(?:f32|bf16)_eps[0-9eE+.-]+"
        r"(?:_qfp8b\d+)?\.comp",
        shader_file,
    )
    if gated_delta is not None:
        return int(gated_delta.group(1))
    head_norm_rope = re.fullmatch(
        r"parallel_head_norm_rope_2way_bf16_h(\d+)_(\d+)_d\d+_r\d+"
        r"_eps[0-9eE+.-]+_offset[0-9eE+.-]+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved|proportional)__sc\d+\.comp",
        shader_file,
    )
    if head_norm_rope is not None:
        return int(head_norm_rope.group(1)) + int(head_norm_rope.group(2))
    raise ModelCompileError(f"shader {shader_file!r} is not a causal scan kernel")


def weight_shared_batch_shader_file(
    shader_file: str, *, tile_width: int = SCALAR_BATCH_LANE_TILE_WIDTH
) -> str | None:
    if tile_width <= 0:
        raise ValueError("batch tile width must be positive")
    tile = tile_width
    if re.fullmatch(r"quantize_fp8_e4m3_b128_h\d+\.comp", shader_file):
        return shader_file.replace(
            "quantize_fp8_e4m3_",
            f"quantize_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(
        r"rms_norm_quantize_fp8_e4m3_b128_h\d+_"
        r"eps[0-9eE+.-]+_offset[0-9eE+.-]+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "rms_norm_quantize_fp8_e4m3_",
            f"rms_norm_quantize_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(
        r"sigmoid_multiply_quantize_fp8_e4m3_b128_h\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "sigmoid_multiply_quantize_fp8_e4m3_",
            f"sigmoid_multiply_quantize_batch{tile}_fp8_e4m3_",
            1,
        )
    prequant_fp8 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_prequant_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if prequant_fp8 is not None:
        return shader_file.replace(
            "_prequant_fp8_e4m3_",
            f"_prequant_batch{tile}_fp8_e4m3_",
            1,
        )
    prequant_parallel_fp8 = re.fullmatch(
        r"parallel_linear_[23]way_prequant_fp8_e4m3_"
        r"b\d+x\d+_\d+x\d+_\d+(?:_\d+)?\.comp",
        shader_file,
    )
    if prequant_parallel_fp8 is not None:
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    prequant_fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_prequant_fp8_e4m3_"
        r"b\d+x\d+_\d+x\d+\.comp",
        shader_file,
    )
    if prequant_fused_ffn is not None:
        return shader_file.replace(
            "_prequant_fp8_e4m3_",
            f"_prequant_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(r"split_bf16_2x\d+x\d+_head_interleaved\.comp", shader_file):
        return shader_file.replace("split_bf16_", f"split_batch{tile}_bf16_", 1)
    if shader_file == "sigmoid_multiply_bf16.comp":
        return f"sigmoid_multiply_batch{tile}_bf16.comp"
    attention_gate = re.fullmatch(
        r"softplus_multiply_bf16_q(\d+)_d(\d+)_(per_head|per_element)\.comp",
        shader_file,
    )
    if attention_gate is not None:
        return shader_file.replace(
            "softplus_multiply_bf16_",
            f"softplus_multiply_batch{tile}_bf16_",
            1,
        )
    rms_norm = re.fullmatch(
        r"rms_norm_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if rms_norm is not None and int(rms_norm.group(1)) % 2 == 0:
        return shader_file.replace("rms_norm_bf16_", f"rms_norm_batch{tile}_bf16_", 1)
    fp8 = re.fullmatch(
        r"(linear|linear_residual)_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fp8 is not None:
        operation, block_rows, block_columns, input_size, _ = fp8.groups()
        if (
            int(block_rows) % 2 == 0
            and int(block_columns) % 4 == 0
            and int(input_size) % 4 == 0
        ):
            return shader_file.replace(
                f"{operation}_fp8_e4m3_",
                f"{operation}_batch{tile}_fp8_e4m3_",
                1,
            )
    q8 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if q8 is not None:
        operation, input_size, output_size = q8.groups()
        if int(input_size) % Q8_0_GROUP_SIZE == 0 and int(output_size) % 2 == 0:
            return shader_file.replace(
                f"{operation}_q8_0_",
                f"{operation}_batch{tile}_q8_0_",
                1,
            )
    int4 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_int4_(gptq|ct)_s(?:f16|bf16)_"
        r"g(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if int4 is not None:
        operation, _, group_size, input_size, output_size = int4.groups()
        if (
            int(group_size) % INT4_VALUES_PER_PACKED_WORD == 0
            and int(input_size) % int(group_size) == 0
            and int(output_size) % 2 == 0
        ):
            return shader_file.replace(
                f"{operation}_int4_",
                f"{operation}_batch{tile}_int4_",
                1,
            )
    bf16 = re.fullmatch(
        r"(linear|linear_residual)_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if bf16 is not None:
        operation, input_size, output_size = bf16.groups()
        if int(input_size) % 2 == 0 and int(output_size) % 2 == 0:
            return f"{operation}_batch{tile}_bf16_{input_size}x{output_size}.comp"
    parallel = re.fullmatch(
        r"parallel_linear_[23]way_bf16_(\d+)x.+\.comp",
        shader_file,
    )
    if parallel is not None and int(parallel.group(1)) % 2 == 0:
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_ffn is not None:
        block_rows, block_columns, input_size, _ = map(int, fused_ffn.groups())
        if block_rows % 2 == 0 and block_columns % 4 == 0 and input_size % 4 == 0:
            return shader_file.replace(
                "parallel_linear_silu_multiply_fp8_e4m3_",
                f"parallel_linear_silu_multiply_batch{tile}_fp8_e4m3_",
                1,
            )
    parallel_fp8 = re.fullmatch(
        r"parallel_linear_([23])way_fp8_e4m3_b(\d+)x(\d+)_"
        r"(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel_fp8 is not None:
        branch_count, block_rows, _block_columns, _input_size = map(
            int, parallel_fp8.groups()[:4]
        )
        output_sizes = [
            int(width) for width in parallel_fp8.groups()[4:] if width is not None
        ]
        if len(output_sizes) != branch_count:
            return None
        tile = tile_width or min(
            SCALAR_BATCH_LANE_TILE_WIDTH,
            max(1, FP8_LINEAR_MIN_WORKGROUPS // block_rows),
        )
        if output_sizes:
            tile = min(tile, max(output_sizes))
        if tile <= 1:
            return None
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    parallel_q8 = re.fullmatch(
        r"parallel_linear_([23])way_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if parallel_q8 is not None:
        _branch_count, input_size, output_size = map(int, parallel_q8.groups())
        if input_size % Q8_0_GROUP_SIZE == 0 and output_size % 2 == 0:
            return shader_file.replace(
                "parallel_linear_",
                f"parallel_linear_batch{tile}_",
                1,
            )
    fused_q8_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_q8_ffn is not None:
        input_size, output_size = map(int, fused_q8_ffn.groups())
        if input_size % Q8_0_GROUP_SIZE == 0 and output_size % 2 == 0:
            return shader_file.replace(
                "parallel_linear_silu_multiply_",
                f"parallel_linear_silu_multiply_batch{tile}_",
                1,
            )
    fused_bf16_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_bf16_"
        r"(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_bf16_ffn is not None:
        input_size, output_size = fused_bf16_ffn.groups()
        if int(input_size) % 2 == 0 and int(output_size) % 2 == 0:
            return (
                f"parallel_linear_silu_multiply_batch{tile}_bf16_"
                f"{input_size}x{output_size}.comp"
            )
    return None
