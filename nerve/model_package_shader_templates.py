from nerve.model_package_common import *
from nerve.model_package_shaders import *
from nerve.model_package_tensors import *

def copy_shader_templates(
    source_dir: Path, dest_dir: Path, shader_files: set[str]
) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for shader_file in sorted(shader_files):
        destination = dest_dir / shader_file
        destination.write_text(render_shader_source(source_dir, shader_file))


def render_shader_source(source_dir: Path, shader_file: str) -> str:
    source = source_dir / shader_file
    if source.exists():
        return source.read_text()

    stream_control_variant = re.fullmatch(r"(.+)__sc(\d+)\.comp", shader_file)
    if stream_control_variant is not None:
        source_name, binding = stream_control_variant.groups()
        rendered = render_shader_source(source_dir, f"{source_name}.comp")
        rendered, replacement_count = re.subn(
            r"layout\(set = 0, binding = \d+\) readonly buffer StreamControl",
            f"layout(set = 0, binding = {binding}) readonly buffer StreamControl",
            rendered,
        )
        if replacement_count != 1:
            raise ModelCompileError(
                f"shader {shader_file} has {replacement_count} stream-control bindings; expected one"
            )
        return rendered

    cooperative_parallel_linear = re.fullmatch(
        r"parallel_linear_batch64_cooperative_([23])way_"
        r"bf16_(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if cooperative_parallel_linear is not None:
        branch_count = int(cooperative_parallel_linear.group(1))
        input_size = int(cooperative_parallel_linear.group(2))
        output_widths = [
            int(width)
            for width in cooperative_parallel_linear.groups()[2:]
            if width is not None
        ]
        if (
            len(output_widths) != branch_count
            or input_size <= 0
            or input_size % 2
            or any(width <= 0 or width % 2 for width in output_widths)
        ):
            raise ModelCompileError(
                f"invalid cooperative parallel-linear shader shape {shader_file!r}"
            )
        labels = [chr(ord("A") + index) for index in range(branch_count)]
        output_bindings = "\n\n".join(
            f"layout(set = 0, binding = {index + 1}) buffer Output{label} {{\n"
            "    bfloat16_t values[];\n"
            f"}} output_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_bindings = "\n\n".join(
            f"layout(set = 0, binding = {branch_count + index + 1}) "
            f"readonly buffer Weight{label} {{\n"
            "    bfloat16_t values[];\n"
            f"}} weight_{label.lower()};"
            for index, label in enumerate(labels)
        )
        output_constants = "\n".join(
            f"const uint OUTPUT_{label}_SIZE = {width}u;"
            for label, width in zip(labels, output_widths, strict=True)
        )
        tile_counts = [
            (width + COOPERATIVE_OUTPUT_TILE_WIDTH - 1) // COOPERATIVE_OUTPUT_TILE_WIDTH
            for width in output_widths
        ]
        branch_lines = []
        consumed_tiles = 0
        for index, (label, tile_count) in enumerate(
            zip(labels, tile_counts, strict=True)
        ):
            prefix = "if" if index == 0 else "else if"
            branch_lines.append(
                f"    {prefix} (global_tile < {consumed_tiles + tile_count}u) {{\n"
                f"        branch = {index}u;\n"
                f"        local_tile = global_tile - {consumed_tiles}u;\n"
                "    }"
            )
            consumed_tiles += tile_count
        weight_reads = "\n".join(
            f"    if (branch == {index}u) return weight_{label.lower()}.values[index];"
            for index, label in enumerate(labels[:-1])
        )
        weight_reads += "\n    return weight_" + labels[-1].lower() + ".values[index];"
        output_size_selection = "\n".join(
            f"    if (branch == {index}u) return OUTPUT_{label}_SIZE;"
            for index, label in enumerate(labels[:-1])
        )
        output_size_selection += f"\n    return OUTPUT_{labels[-1]}_SIZE;"
        output_writes = "\n".join(
            f"    if (branch == {index}u) {{ output_{label.lower()}.values["
            f"batch_index * OUTPUT_{label}_SIZE + row] = value; return; }}"
            for index, label in enumerate(labels[:-1])
        )
        output_writes += (
            f"\n    output_{labels[-1].lower()}.values[batch_index * "
            f"OUTPUT_{labels[-1]}_SIZE + row] = value;"
        )
        direct_output_stores = "\n".join(
            (
                f"            {'if' if index == 0 else 'else if'} "
                f"(branch == {index}u) {{\n"
                "                coopMatStore(\n"
                "                    rounded,\n"
                f"                    output_{label.lower()}.values,\n"
                "                    (batch_start + batch_subtile * MATRIX_TILE) "
                f"* OUTPUT_{label}_SIZE + output_start + output_subtile * MATRIX_TILE,\n"
                f"                    OUTPUT_{label}_SIZE,\n"
                "                    gl_CooperativeMatrixLayoutColumnMajor\n"
                "                );\n"
                "            }"
            )
            for index, label in enumerate(labels)
        )
        direct_weight_loads = "\n".join(
            (
                f"            {'if' if index == 0 else 'else if'} "
                f"(branch == {index}u) {{\n"
                "                coopMatLoad(\n"
                "                    a,\n"
                f"                    weight_{label.lower()}.values,\n"
                "                    (output_start + output_subtile * MATRIX_TILE) "
                "* INPUT_SIZE + input_start,\n"
                "                    INPUT_SIZE,\n"
                "                    gl_CooperativeMatrixLayoutRowMajor\n"
                "                );\n"
                "            }"
            )
            for index, label in enumerate(labels)
        )
        return render_shader_template(
            source_dir,
            "parallel_linear_batch_cooperative_bf16.comp.template",
            {
                "OUTPUT_BINDINGS": output_bindings,
                "WEIGHT_BINDINGS": weight_bindings,
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE_CONSTANTS": output_constants,
                "TOTAL_OUTPUT_TILES": str(sum(tile_counts)),
                "BRANCH_SELECTION": "\n".join(branch_lines),
                "WEIGHT_READS": weight_reads,
                "OUTPUT_SIZE_SELECTION": output_size_selection,
                "OUTPUT_WRITES": output_writes,
                "DIRECT_WEIGHT_LOADS": direct_weight_loads,
                "DIRECT_OUTPUT_STORES": direct_output_stores,
            },
        )

    cooperative_bf16_linear = re.fullmatch(
        r"(linear|linear_residual)_batch64_cooperative_"
        r"bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if cooperative_bf16_linear is not None:
        operation = cooperative_bf16_linear.group(1)
        input_size = int(cooperative_bf16_linear.group(2))
        output_size = int(cooperative_bf16_linear.group(3))
        if input_size <= 0 or input_size % 2 or output_size <= 0 or output_size % 2:
            raise ModelCompileError(
                f"invalid cooperative BF16 linear shader shape {shader_file!r}"
            )
        residual = operation == "linear_residual"
        direct_store = (
            "    if (output_start + OUTPUT_TILE <= OUTPUT_SIZE\n"
            "        && batch_start + BATCH_TILE <= batch_control.batch_width) {\n"
            "        for (uint batch_subtile = 0u; batch_subtile < BATCH_SUBTILES; batch_subtile++) {\n"
            "            coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>\n"
            "                residual_values;\n"
            "            coopMatLoad(\n"
            "                residual_values,\n"
            "                residual_frames.values,\n"
            "                (batch_start + batch_subtile * MATRIX_TILE) * OUTPUT_SIZE\n"
            "                    + output_start + output_subtile * MATRIX_TILE,\n"
            "                OUTPUT_SIZE,\n"
            "                gl_CooperativeMatrixLayoutColumnMajor\n"
            "            );\n"
            "            coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>\n"
            "                combined = coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16,\n"
            "                    gl_MatrixUseAccumulator>(bfloat16_t(0.0));\n"
            "            for (uint element = 0u; element < combined.length(); element++) {\n"
            "                bfloat16_t projection = uintBitsToBFloat16EXT(\n"
            "                    uint16_t(f32_to_bf16(sums[batch_subtile][element]))\n"
            "                );\n"
            "                combined[element] = uintBitsToBFloat16EXT(uint16_t(f32_to_bf16(\n"
            "                    float(projection) + float(residual_values[element])\n"
            "                )));\n"
            "            }\n"
            "            coopMatStore(\n"
            "                combined,\n"
            "                output_frames.values,\n"
            "                (batch_start + batch_subtile * MATRIX_TILE) * OUTPUT_SIZE\n"
            "                    + output_start + output_subtile * MATRIX_TILE,\n"
            "                OUTPUT_SIZE,\n"
            "                gl_CooperativeMatrixLayoutColumnMajor\n"
            "            );\n"
            "        }\n"
            "        return;\n"
            "    }"
            if residual
            else "    if (output_start + OUTPUT_TILE <= OUTPUT_SIZE\n"
            "        && batch_start + BATCH_TILE <= batch_control.batch_width) {\n"
            "        for (uint batch_subtile = 0u; batch_subtile < BATCH_SUBTILES; batch_subtile++) {\n"
            "            coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>\n"
            "                rounded = coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16,\n"
            "                    gl_MatrixUseAccumulator>(bfloat16_t(0.0));\n"
            "            for (uint element = 0u; element < rounded.length(); element++) {\n"
            "                rounded[element] = uintBitsToBFloat16EXT(\n"
            "                    uint16_t(f32_to_bf16(sums[batch_subtile][element]))\n"
            "                );\n"
            "            }\n"
            "            coopMatStore(\n"
            "                rounded,\n"
            "                output_frames.values,\n"
            "                (batch_start + batch_subtile * MATRIX_TILE) * OUTPUT_SIZE\n"
            "                    + output_start + output_subtile * MATRIX_TILE,\n"
            "                OUTPUT_SIZE,\n"
            "                gl_CooperativeMatrixLayoutColumnMajor\n"
            "            );\n"
            "        }\n"
            "        return;\n"
            "    }"
        )
        finalize_function = (
            "float rounded_bf16(float value) {\n"
            "    return uintBitsToFloat(f32_to_bf16(value) << 16u);\n"
            "}\n\n"
            "bfloat16_t finalize_result(uint batch_index, uint output_index, float value) {\n"
            "    uint index = batch_index * OUTPUT_SIZE + output_index;\n"
            "    return uintBitsToBFloat16EXT(uint16_t(f32_to_bf16(\n"
            "        rounded_bf16(value) + float(residual_frames.values[index])\n"
            "    )));\n"
            "}"
            if residual
            else "bfloat16_t finalize_result(uint batch_index, uint output_index, float value) {\n"
            "    return uintBitsToBFloat16EXT(uint16_t(f32_to_bf16(value)));\n"
            "}"
        )
        return render_shader_template(
            source_dir,
            "linear_batch_cooperative_bf16.comp.template",
            {
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "RESIDUAL_BINDING": (
                    "layout(set = 0, binding = 1) readonly buffer ResidualFrames {\n"
                    "    bfloat16_t values[];\n"
                    "} residual_frames;"
                    if residual
                    else ""
                ),
                "OUTPUT_BINDING": "2" if residual else "1",
                "OUTPUT_TYPE": "bfloat16_t",
                "WEIGHT_BINDING": "3" if residual else "2",
                "DIRECT_STORE": direct_store,
                "FINALIZE_FUNCTION": finalize_function,
            },
        )

    cooperative_fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_batch64_cooperative_"
        r"bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if cooperative_fused_ffn is not None:
        input_size = int(cooperative_fused_ffn.group(1))
        output_size = int(cooperative_fused_ffn.group(2))
        if input_size <= 0 or input_size % 2 or output_size <= 0 or output_size % 2:
            raise ModelCompileError(
                f"invalid cooperative fused FFN shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            "parallel_linear_silu_multiply_batch_cooperative_bf16.comp.template",
            {
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
            },
        )

    parallel_linear = re.fullmatch(
        r"parallel_linear_(?:batch(\d+)_)?([23])way_bf16_"
        r"(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel_linear is not None:
        batch_tile_width = (
            int(parallel_linear.group(1))
            if parallel_linear.group(1) is not None
            else None
        )
        branch_count = int(parallel_linear.group(2))
        input_size = int(parallel_linear.group(3))
        output_widths = [
            int(width) for width in parallel_linear.groups()[3:] if width is not None
        ]
        if (
            len(output_widths) != branch_count
            or (batch_tile_width is not None and batch_tile_width <= 0)
            or input_size <= 0
            or input_size % 2
            or any(width <= 0 or width % 2 for width in output_widths)
        ):
            raise ModelCompileError(
                f"invalid parallel-linear shader shape {shader_file!r}"
            )
        labels = [chr(ord("A") + index) for index in range(branch_count)]
        output_bindings = "\n\n".join(
            f"layout(set = 0, binding = {index + 1}) buffer Output{label} {{\n"
            "    uint words[];\n"
            f"}} output_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_bindings = "\n\n".join(
            f"layout(set = 0, binding = {branch_count + index + 1}) "
            f"readonly buffer Weight{label} {{\n"
            "    uint words[];\n"
            f"}} weight_{label.lower()};"
            for index, label in enumerate(labels)
        )
        output_constants = "\n".join(
            f"const uint OUTPUT_{label}_WORDS = {width}u / 2u;"
            for label, width in zip(labels, output_widths, strict=True)
        )
        branch_lines = []
        consumed_words = []
        for index, label in enumerate(labels):
            prefix = "if" if index == 0 else "else if"
            threshold = " + ".join([*consumed_words, f"OUTPUT_{label}_WORDS"])
            offset = " + ".join(consumed_words) or "0u"
            branch_lines.append(
                f"    {prefix} (word_index < {threshold}) {{\n"
                f"        branch = {index}u;\n"
                f"        local_word_index = word_index - ({offset});\n"
                "    }"
            )
            consumed_words.append(f"OUTPUT_{label}_WORDS")
        weight_reads = "\n".join(
            f"    if (branch == {index}u) return weight_{label.lower()}.words[weight_index];"
            for index, label in enumerate(labels[:-1])
        )
        weight_reads += (
            "\n    return weight_" + labels[-1].lower() + ".words[weight_index];"
        )
        output_index = (
            "batch_index * OUTPUT_{label}_WORDS + local_word_index"
            if batch_tile_width is not None
            else "local_word_index"
        )
        output_writes = "\n".join(
            f"    if (branch == {index}u) {{ output_{label.lower()}.words["
            + output_index.format(label=label)
            + "] = packed; return; }"
            for index, label in enumerate(labels[:-1])
        )
        output_writes += (
            "\n    output_"
            + labels[-1].lower()
            + ".words["
            + output_index.format(label=labels[-1])
            + "] = packed;"
        )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_batch_bf16.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_bf16.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "OUTPUT_BINDINGS": output_bindings,
                "WEIGHT_BINDINGS": weight_bindings,
                "INPUT_SIZE": str(input_size),
                "OUTPUT_WORD_CONSTANTS": output_constants,
                "TOTAL_OUTPUT_WORDS": " + ".join(
                    f"OUTPUT_{label}_WORDS" for label in labels
                ),
                "BRANCH_SELECTION": "\n".join(branch_lines),
                "WEIGHT_READS": weight_reads,
                "OUTPUT_WRITES": output_writes,
            },
        )

    fp8_quantizer = re.fullmatch(
        r"quantize(?:_batch(\d+))?_fp8_e4m3_b(\d+)_h(\d+)\.comp",
        shader_file,
    )
    if fp8_quantizer is not None:
        batch_tile_width = (
            int(fp8_quantizer.group(1))
            if fp8_quantizer.group(1) is not None
            else None
        )
        block_columns = int(fp8_quantizer.group(2))
        element_count = int(fp8_quantizer.group(3))
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or block_columns != 128
            or element_count <= 0
            or element_count % block_columns
        ):
            raise ModelCompileError(
                f"invalid FP8 activation-quantizer shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "quantize_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "quantize_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BLOCK_COLUMNS": str(block_columns),
                "ELEMENT_COUNT": str(element_count),
            },
        )

    cooperative_prequant_fp8_linear = re.fullmatch(
        r"(linear|linear_residual)_prequant_batch(\d+)_cooperative_"
        r"fp8_e4m3_m(\d+)n(\d+)k(\d+)_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if cooperative_prequant_fp8_linear is not None:
        operation = cooperative_prequant_fp8_linear.group(1)
        (
            batch_tile_width,
            matrix_m,
            matrix_n,
            matrix_k,
            block_rows,
            block_columns,
            input_size,
            output_size,
        ) = map(int, cooperative_prequant_fp8_linear.groups()[1:])
        if (
            min(
                batch_tile_width,
                matrix_m,
                matrix_n,
                matrix_k,
                block_rows,
                block_columns,
                input_size,
                output_size,
            )
            <= 0
            or batch_tile_width != 4 * matrix_n
            or block_columns % matrix_k
            or input_size % block_columns
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid cooperative FP8 linear shader shape {shader_file!r}"
            )
        residual = operation == "linear_residual"
        output_binding = 3 if residual else 2
        weight_binding = output_binding + 1
        finalize_function = (
            "float finalize_result(uint batch_index, uint output_index, float value) {\n"
            "    uint residual_index = batch_index * OUTPUT_SIZE + output_index;\n"
            "    float residual = uintBitsToFloat(\n"
            "        uint(residual_frames.values[residual_index]) << 16u\n"
            "    );\n"
            "    return value + residual;\n"
            "}"
            if residual
            else (
                "float finalize_result(uint batch_index, uint output_index, float value) {\n"
                "    return value;\n"
                "}"
            )
        )
        return render_shader_template(
            source_dir,
            "linear_prequant_batch_cooperative_fp8_e4m3.comp.template",
            {
                "MATRIX_M": str(matrix_m),
                "MATRIX_N": str(matrix_n),
                "MATRIX_K": str(matrix_k),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "RESIDUAL_BINDING": (
                    "layout(set = 0, binding = 2) readonly buffer ResidualFrames {\n"
                    "    uint16_t values[];\n"
                    "} residual_frames;"
                    if residual
                    else ""
                ),
                "OUTPUT_BINDING": str(output_binding),
                "WEIGHT_BINDING": str(weight_binding),
                "WEIGHT_SCALE_BINDING": str(weight_binding + 1),
                "FINALIZE_FUNCTION": finalize_function,
            },
        )

    prequant_fp8_linear = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_prequant"
        r"(?:_batch(\d+))?_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if prequant_fp8_linear is not None:
        operation = prequant_fp8_linear.group(1)
        batch_tile_width = (
            int(prequant_fp8_linear.group(2))
            if prequant_fp8_linear.group(2) is not None
            else None
        )
        block_rows, block_columns, input_size, output_size = map(
            int, prequant_fp8_linear.groups()[2:]
        )
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid prequantized FP8 linear shader shape {shader_file!r}"
            )
        has_residual = operation == "linear_residual"
        has_bias = operation == "linear_bias"
        output_binding = 3 if has_residual else 2
        weight_binding = output_binding + 1
        auxiliary_buffer = (
            "layout(set = 0, binding = 2) readonly buffer ResidualFrame"
            f"{'s' if batch_tile_width is not None else ''} {{\n"
            "    uint words[];\n"
            f"}} residual_frame{'s' if batch_tile_width is not None else ''};"
            if has_residual
            else (
                f"layout(set = 0, binding = {weight_binding + 2}) "
                "readonly buffer Bias {\n"
                "    uint words[];\n"
                "} bias;"
                if has_bias
                else ""
            )
        )
        if batch_tile_width is None:
            finalize_output = (
                "float finalize_output(uint row, float value) {\n"
                "    return read_bf16_word(residual_frame.words[row >> 1u], row)"
                " + value;\n"
                "}"
                if has_residual
                else (
                    "float finalize_output(uint row, float value) {\n"
                    "    return read_bf16_word(bias.words[row >> 1u], row) + value;\n"
                    "}"
                    if has_bias
                    else (
                        "float finalize_output(uint row, float value) {\n"
                        "    return value;\n"
                        "}"
                    )
                )
            )
        else:
            finalize_output = (
                "float finalize_output(uint batch_index, uint row, float value) {\n"
                "    return read_bf16_word(residual_frames.words[\n"
                "        batch_index * (OUTPUT_SIZE / 2u) + (row >> 1u)\n"
                "    ], row) + value;\n"
                "}"
                if has_residual
                else (
                    "float finalize_output(uint batch_index, uint row, float value) {\n"
                    "    return read_bf16_word(bias.words[row >> 1u], row) + value;\n"
                    "}"
                    if has_bias
                    else (
                        "float finalize_output(uint batch_index, uint row, float value) {\n"
                        "    return value;\n"
                        "}"
                    )
                )
            )
        return render_shader_template(
            source_dir,
            (
                "linear_prequant_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "linear_prequant_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(FP8_PREQUANT_TILE_ROWS),
                "AUXILIARY_BUFFER": auxiliary_buffer,
                "OUTPUT_BINDING": str(output_binding),
                "WEIGHT_BINDING": str(weight_binding),
                "WEIGHT_SCALE_BINDING": str(weight_binding + 1),
                "FINALIZE_OUTPUT_FUNCTION": finalize_output,
            },
        )

    prequant_parallel_fp8 = re.fullmatch(
        r"parallel_linear_(?:batch(\d+)_)?([23])way_prequant_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if prequant_parallel_fp8 is not None:
        batch_tile_width = (
            int(prequant_parallel_fp8.group(1))
            if prequant_parallel_fp8.group(1) is not None
            else None
        )
        branch_count = int(prequant_parallel_fp8.group(2))
        block_rows = int(prequant_parallel_fp8.group(3))
        block_columns = int(prequant_parallel_fp8.group(4))
        input_size = int(prequant_parallel_fp8.group(5))
        output_sizes = [
            int(width)
            for width in prequant_parallel_fp8.groups()[5:]
            if width is not None
        ]
        if (
            branch_count not in {2, 3}
            or len(output_sizes) != branch_count
            or (batch_tile_width is not None and batch_tile_width <= 0)
            or block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns
            or any(output_size <= 0 or output_size % 2 for output_size in output_sizes)
        ):
            raise ModelCompileError(
                f"invalid prequantized parallel FP8 shader shape {shader_file!r}"
            )
        labels = [chr(ord("A") + index) for index in range(branch_count)]
        output_constants = "\n".join(
            (
                f"const uint OUTPUT_{label}_SIZE = {output_size}u;\n"
                f"const uint OUTPUT_{label}_WORDS = OUTPUT_{label}_SIZE / 2u;"
            )
            for label, output_size in zip(labels, output_sizes, strict=True)
        )
        output_size_selection = "\n".join(
            f"    if (branch == {index}u) return OUTPUT_{label}_SIZE;"
            for index, label in enumerate(labels[:-1])
        )
        output_size_selection += f"\n    return OUTPUT_{labels[-1]}_SIZE;"
        output_bindings = "\n\n".join(
            f"layout(set = 0, binding = {index + 2}) buffer Output{label} {{\n"
            "    uint words[];\n"
            f"}} output_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_bindings = "\n\n".join(
            "\n\n".join(
                [
                    f"layout(set = 0, binding = {branch_count + index * 2 + 2}) "
                    f"readonly buffer Weight{label} {{\n"
                    "    uint words[];\n"
                    f"}} weight_{label.lower()};",
                    f"layout(set = 0, binding = {branch_count + index * 2 + 3}) "
                    f"readonly buffer WeightScaleInv{label} {{\n"
                    "    uint words[];\n"
                    f"}} weight_scale_inv_{label.lower()};",
                ]
            )
            for index, label in enumerate(labels)
        )
        weight_scale_reads = "\n".join(
            "    if (branch == "
            f"{index}u) return read_bf16_word("
            f"weight_scale_inv_{label.lower()}.words[index >> 1u], index);"
            for index, label in enumerate(labels[:-1])
        )
        weight_scale_reads += (
            "\n    return read_bf16_word(weight_scale_inv_"
            + labels[-1].lower()
            + ".words[index >> 1u], index);"
        )
        weight_reads = "\n".join(
            f"    if (branch == {index}u) "
            f"return read_fp8x4(weight_{label.lower()}.words[index]);"
            for index, label in enumerate(labels[:-1])
        )
        weight_reads += (
            f"\n    return read_fp8x4(weight_{labels[-1].lower()}.words[index]);"
        )
        output_index = (
            "batch_index * OUTPUT_{label}_WORDS + row / 2u"
            if batch_tile_width is not None
            else "row / 2u"
        )
        output_writes = "\n".join(
            f"    if (branch == {index}u) {{ output_{label.lower()}.words["
            + output_index.format(label=label)
            + "] = value; return; }"
            for index, label in enumerate(labels[:-1])
        )
        output_writes += (
            "\n    output_"
            + labels[-1].lower()
            + ".words["
            + output_index.format(label=labels[-1])
            + "] = value;"
        )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_prequant_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_prequant_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BRANCH_COUNT": str(branch_count),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_TILE_ROWS": str(FP8_PREQUANT_TILE_ROWS),
                "OUTPUT_CONSTANTS": output_constants,
                "OUTPUT_SIZE_SELECTION": output_size_selection,
                "OUTPUT_BINDINGS": output_bindings,
                "WEIGHT_BINDINGS": weight_bindings,
                "WEIGHT_SCALE_READS": weight_scale_reads,
                "WEIGHT_READS": weight_reads,
                "OUTPUT_WRITES": output_writes,
            },
        )

    parallel_fp8 = re.fullmatch(
        r"parallel_linear_(?:batch(\d+)_)?([23])way_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel_fp8 is not None:
        batch_tile_width = (
            int(parallel_fp8.group(1))
            if parallel_fp8.group(1) is not None
            else None
        )
        branch_count = int(parallel_fp8.group(2))
        block_rows = int(parallel_fp8.group(3))
        block_columns = int(parallel_fp8.group(4))
        input_size = int(parallel_fp8.group(5))
        output_sizes = [
            int(width)
            for width in parallel_fp8.groups()[5:]
            if width is not None
        ]
        output_tile_rows = fp8_linear_tile_rows(max(output_sizes))
        if (
            branch_count not in {2, 3}
            or len(output_sizes) != branch_count
            or (batch_tile_width is not None and batch_tile_width <= 0)
            or block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns != 0
            or any(output_size <= 0 or output_size % 2 for output_size in output_sizes)
        ):
            raise ModelCompileError(
                f"invalid fused FP8 parallel-linear shader shape {shader_file!r}"
            )
        labels = [chr(ord("A") + index) for index in range(branch_count)]
        output_constants = "\n".join(
            (
                f"const uint OUTPUT_{label}_SIZE = {output_size}u;\n"
                f"const uint OUTPUT_{label}_WORDS = OUTPUT_{label}_SIZE / 2u;"
            )
            for label, output_size in zip(labels, output_sizes, strict=True)
        )
        output_size_selection = "\n".join(
            f"    if (branch == {index}u) return OUTPUT_{label}_SIZE;"
            for index, label in enumerate(labels[:-1])
        )
        output_size_selection += f"\n    return OUTPUT_{labels[-1]}_SIZE;"
        output_bindings = "\n\n".join(
            f"layout(set = 0, binding = {index + 1}) buffer Output{label} {{\n"
            "    uint words[];\n"
            f"}} output_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_bindings = "\n\n".join(
            "\n\n".join(
                [
                    f"layout(set = 0, binding = {branch_count + index * 2 + 1}) "
                    f"readonly buffer Weight{label} {{\n"
                    "    uint words[];\n"
                    f"}} weight_{label.lower()};",
                    f"layout(set = 0, binding = {branch_count + index * 2 + 2}) "
                    f"readonly buffer WeightScaleInv{label} {{\n"
                    "    uint words[];\n"
                    f"}} weight_scale_inv_{label.lower()};",
                ]
            )
            for index, label in enumerate(labels)
        )
        weight_scale_reads = "\n".join(
            "    if (branch == "
            f"{index}u) return read_bf16_word("
            f"weight_scale_inv_{label.lower()}.words[index >> 1u], index);"
            for index, label in enumerate(labels[:-1])
        )
        weight_scale_reads += (
            "\n    return read_bf16_word(weight_scale_inv_"
            + labels[-1].lower()
            + ".words[index >> 1u], index);"
        )
        weight_reads = "\n".join(
            "    if (branch == "
            f"{index}u) {{ uint packed = weight_{label.lower()}.words[index >> 2u]; "
            "return uintBitsToFloate4m3EXT(u8vec4(uint8_t(packed), "
            "uint8_t(packed >> 8u), uint8_t(packed >> 16u), "
            "uint8_t(packed >> 24u))); }"
            for index, label in enumerate(labels[:-1])
        )
        weight_reads += (
            "\n    uint packed = weight_"
            + labels[-1].lower()
            + ".words[index >> 2u]; return uintBitsToFloate4m3EXT(u8vec4("
            "uint8_t(packed), uint8_t(packed >> 8u), uint8_t(packed >> 16u), "
            "uint8_t(packed >> 24u)));"
        )
        if batch_tile_width is None:
            output_writes = "\n".join(
                (
                    f"    if (branch == {index}u) {{ "
                    f"output_{label.lower()}.words[row / 2u] = value; return; }}"
                )
                for index, label in enumerate(labels[:-1])
            )
            output_writes += (
                f"\n    output_{labels[-1].lower()}.words[row / 2u] = value;"
            )
        else:
            output_writes = "\n".join(
                (
                    f"    if (branch == {index}u) {{ output_{label.lower()}.words["
                    f"batch_index * OUTPUT_{label}_WORDS + row / 2u] = value; return; }}"
                )
                for index, label in enumerate(labels[:-1])
            )
            output_writes += (
                f"\n    output_{labels[-1].lower()}.words[batch_index * "
                f"OUTPUT_{labels[-1]}_WORDS + row / 2u] = value;"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BRANCH_COUNT": str(branch_count),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_TILE_ROWS": str(output_tile_rows),
                "OUTPUT_CONSTANTS": output_constants,
                "OUTPUT_SIZE_SELECTION": output_size_selection,
                "OUTPUT_BINDINGS": output_bindings,
                "WEIGHT_BINDINGS": weight_bindings,
                "WEIGHT_SCALE_READS": weight_scale_reads,
                "WEIGHT_READS": weight_reads,
                "OUTPUT_WRITES": output_writes,
            },
        )

    parallel_q8 = re.fullmatch(
        r"parallel_linear_(?:batch(\d+)_)?([23])way_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if parallel_q8 is not None:
        batch_tile_width = (
            int(parallel_q8.group(1)) if parallel_q8.group(1) is not None else None
        )
        branch_count = int(parallel_q8.group(2))
        input_size = int(parallel_q8.group(3))
        output_size = int(parallel_q8.group(4))
        if (
            branch_count not in {2, 3}
            or (batch_tile_width is not None and batch_tile_width <= 0)
            or input_size <= 0
            or input_size % Q8_0_GROUP_SIZE
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid Q8_0 parallel-linear shader shape {shader_file!r}"
            )
        labels = [chr(ord("A") + index) for index in range(branch_count)]
        output_bindings = "\n\n".join(
            f"layout(set = 0, binding = {index + 1}) buffer Output{label} {{\n"
            "    uint words[];\n"
            f"}} output_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_bindings = "\n\n".join(
            f"layout(set = 0, binding = {branch_count + index + 1}) "
            f"readonly buffer Weight{label} {{\n"
            "    uint words[];\n"
            f"}} weight_{label.lower()};"
            for index, label in enumerate(labels)
        )
        weight_reads = "\n".join(
            f"    if (branch == {index}u) return weight_{label.lower()}.words[index];"
            for index, label in enumerate(labels[:-1])
        )
        weight_reads += "\n    return weight_" + labels[-1].lower() + ".words[index];"
        output_writes = "\n".join(
            f"    if (branch == {index}u) {{ output_{label.lower()}.words[__OUTPUT_INDEX__] = value; return; }}"
            for index, label in enumerate(labels[:-1])
        )
        output_writes += (
            "\n    output_" + labels[-1].lower() + ".words[__OUTPUT_INDEX__] = value;"
        )
        output_index = (
            "batch_index * OUTPUT_WORDS + index"
            if batch_tile_width is not None
            else "index"
        )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_batch_q8_0.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_q8_0.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BRANCH_COUNT": str(branch_count),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(Q8_0_OUTPUT_TILE_ROWS),
                "OUTPUT_BINDINGS": output_bindings,
                "WEIGHT_BINDINGS": weight_bindings,
                "WEIGHT_READS": weight_reads,
                "OUTPUT_WRITES": output_writes.replace("__OUTPUT_INDEX__", output_index),
            },
        )

    tied_output_projection_fp8 = re.fullmatch(
        r"tied_output_projection(?:_batch(\d+))?_fp8_e4m3_b(\d+)x(\d+)_"
        r"(\d+)x(\d+)_scale([A-Za-z0-9_]+)_to_f32\.comp",
        shader_file,
    )
    if tied_output_projection_fp8 is not None:
        batch_tile_width = (
            int(tied_output_projection_fp8.group(1))
            if tied_output_projection_fp8.group(1) is not None
            else None
        )
        block_rows = int(tied_output_projection_fp8.group(2))
        block_columns = int(tied_output_projection_fp8.group(3))
        vocab_size = int(tied_output_projection_fp8.group(4))
        input_size = int(tied_output_projection_fp8.group(5))
        output_scale = tied_output_projection_fp8.group(6)
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or
            block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns != 0
            or vocab_size <= 0
        ):
            raise ModelCompileError(
                f"invalid FP8 output projection shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "tied_output_projection_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "tied_output_projection_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "VOCAB_SIZE": str(vocab_size),
                "OUTPUT_TILE_ROWS": str(FP8_OUTPUT_PROJECTION_TILE_ROWS),
                "OUTPUT_SCALE": output_scale,
            },
        )

    batched_bf16_linear = re.fullmatch(
        r"(linear|linear_residual)_batch(\d+)_bf16_"
        r"(\d+)x(\d+)\.comp",
        shader_file,
    )
    if batched_bf16_linear is not None:
        operation = batched_bf16_linear.group(1)
        batch_tile_width = int(batched_bf16_linear.group(2))
        input_size = int(batched_bf16_linear.group(3))
        output_size = int(batched_bf16_linear.group(4))
        if (
            batch_tile_width <= 0
            or input_size <= 0
            or input_size % 2
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid batched BF16 linear shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            f"{operation}_batch_bf16.comp.template",
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
            },
        )

    fused_ffn_projection = re.fullmatch(
        r"parallel_linear_silu_multiply_(?:batch(\d+)_)?"
        r"bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_ffn_projection is not None:
        batch_tile_width = (
            int(fused_ffn_projection.group(1))
            if fused_ffn_projection.group(1) is not None
            else None
        )
        input_size = int(fused_ffn_projection.group(2))
        output_size = int(fused_ffn_projection.group(3))
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or input_size <= 0
            or input_size % 2
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid fused FFN projection shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_silu_multiply_batch_bf16.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_silu_multiply_bf16.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
            },
        )

    concatenate = re.fullmatch(r"concatenate_bf16_(\d+)_(\d+)\.comp", shader_file)
    if concatenate is not None:
        width_a = int(concatenate.group(1))
        width_b = int(concatenate.group(2))
        if width_a <= 0 or width_a % 2 or width_b <= 0 or width_b % 2:
            raise ModelCompileError(f"invalid concatenate shader shape {shader_file!r}")
        return render_shader_template(
            source_dir,
            "concatenate_bf16.comp.template",
            {"WIDTH_A": str(width_a), "WIDTH_B": str(width_b)},
        )

    prequant_fused_fp8_ffn_projection = re.fullmatch(
        r"parallel_linear_silu_multiply_prequant"
        r"(?:_batch(\d+))?_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if prequant_fused_fp8_ffn_projection is not None:
        batch_tile_width = (
            int(prequant_fused_fp8_ffn_projection.group(1))
            if prequant_fused_fp8_ffn_projection.group(1) is not None
            else None
        )
        block_rows = int(prequant_fused_fp8_ffn_projection.group(2))
        block_columns = int(prequant_fused_fp8_ffn_projection.group(3))
        input_size = int(prequant_fused_fp8_ffn_projection.group(4))
        output_size = int(prequant_fused_fp8_ffn_projection.group(5))
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid prequantized FP8 FFN shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_silu_multiply_prequant_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_silu_multiply_prequant_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(FP8_PREQUANT_TILE_ROWS),
            },
        )

    fused_fp8_ffn_projection = re.fullmatch(
        r"parallel_linear_silu_multiply_(?:batch(\d+)_)?fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_fp8_ffn_projection is not None:
        batch_tile_width = (
            int(fused_fp8_ffn_projection.group(1))
            if fused_fp8_ffn_projection.group(1) is not None
            else None
        )
        block_rows = int(fused_fp8_ffn_projection.group(2))
        block_columns = int(fused_fp8_ffn_projection.group(3))
        input_size = int(fused_fp8_ffn_projection.group(4))
        output_size = int(fused_fp8_ffn_projection.group(5))
        if any(
            value <= 0 for value in (block_rows, block_columns, input_size, output_size)
        ) or (batch_tile_width is not None and batch_tile_width <= 0):
            raise ModelCompileError(
                f"invalid fused FP8 FFN projection shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_silu_multiply_batch_fp8_e4m3.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_silu_multiply_fp8_e4m3.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(FP8_FUSED_FFN_TILE_ROWS),
            },
        )

    fused_q8_ffn_projection = re.fullmatch(
        r"parallel_linear_silu_multiply_(?:batch(\d+)_)?q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_q8_ffn_projection is not None:
        batch_tile_width = (
            int(fused_q8_ffn_projection.group(1))
            if fused_q8_ffn_projection.group(1) is not None
            else None
        )
        input_size = int(fused_q8_ffn_projection.group(2))
        output_size = int(fused_q8_ffn_projection.group(3))
        if (
            (batch_tile_width is not None and batch_tile_width <= 0)
            or
            input_size <= 0
            or input_size % Q8_0_GROUP_SIZE
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid fused Q8_0 FFN projection shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "parallel_linear_silu_multiply_batch_q8_0.comp.template"
                if batch_tile_width is not None
                else "parallel_linear_silu_multiply_q8_0.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width or 1),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(Q8_0_OUTPUT_TILE_ROWS),
            },
        )

    native_q8_batch_linear = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_batch(\d+)_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if native_q8_batch_linear is not None:
        operation = native_q8_batch_linear.group(1)
        batch_tile_width = int(native_q8_batch_linear.group(2))
        input_size, output_size = map(int, native_q8_batch_linear.groups()[2:])
        if (
            batch_tile_width <= 0
            or input_size <= 0
            or input_size % Q8_0_GROUP_SIZE
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid native batched Q8_0 linear shader shape {shader_file!r}"
            )
        has_residual = operation == "linear_residual"
        has_bias = operation == "linear_bias"
        output_binding = 2 if has_residual else 1
        weight_binding = 3 if has_residual else 2
        auxiliary_buffer = (
            "layout(set = 0, binding = 3) readonly buffer Bias { "
            "uint words[]; } bias;"
            if has_bias
            else (
                "layout(set = 0, binding = 1) readonly buffer ResidualFrames { "
                "uint words[]; } residual_frames;"
                if has_residual
                else ""
            )
        )
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    uint index = batch_index * OUTPUT_WORDS + (row >> 1u);\n"
            "    return read_bf16_word(residual_frames.words[index], row) + value;\n"
            "}"
            if has_residual
            else (
                "float finalize_output(uint batch_index, uint row, float value) {\n"
                "    return read_bf16_word(bias.words[row >> 1u], row) + value;\n"
                "}"
                if has_bias
                else (
                    "float finalize_output(uint batch_index, uint row, float value) {\n"
                    "    return value;\n"
                    "}"
                )
            )
        )
        return render_shader_template(
            source_dir,
            "linear_batch_q8_0.comp.template",
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width),
                "OUTPUT_BINDING": str(output_binding),
                "WEIGHT_BINDING": str(weight_binding),
                "AUXILIARY_BUFFER": auxiliary_buffer,
                "FINALIZE_OUTPUT_FUNCTION": finalize_output,
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(Q8_0_OUTPUT_TILE_ROWS),
            },
        )

    native_fp8_linear = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if native_fp8_linear is not None:
        operation = native_fp8_linear.group(1)
        block_rows, block_columns, input_size, output_size = map(
            int, native_fp8_linear.groups()[1:]
        )
        if (
            block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns != 0
            or output_size <= 0
            or output_size % 2 != 0
        ):
            raise ModelCompileError(
                f"invalid native FP8 linear shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            f"{operation}_fp8_e4m3.comp.template",
            {
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(fp8_linear_tile_rows(output_size)),
            },
        )

    recurrent_depthwise = re.fullmatch(
        r"multiply_rolling_depthwise(_gate)?_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if recurrent_depthwise is not None:
        has_output_gate = recurrent_depthwise.group(1) is not None
        output_gate_binding = (
            "layout(set = 0, binding = 3) readonly buffer OutputGate {\n"
            "    uint words[];\n"
            "} output_gate;"
            if has_output_gate
            else ""
        )
        finalize_output = (
            "uint finalize_output(uint word_index, uint conv_pair) {\n"
            "    uint gate_pair = output_gate.words[word_index];\n"
            "    uint lo = f32_to_bf16(\n"
            "        bf16_to_f32(conv_pair) * bf16_to_f32(gate_pair)\n"
            "    );\n"
            "    uint hi = f32_to_bf16(\n"
            "        bf16_to_f32(conv_pair >> 16) * bf16_to_f32(gate_pair >> 16)\n"
            "    );\n"
            "    return (hi << 16) | lo;\n"
            "}"
            if has_output_gate
            else (
                "uint finalize_output(uint word_index, uint conv_pair) {\n"
                "    return conv_pair;\n"
                "}"
            )
        )
        binding_offset = 1 if has_output_gate else 0
        return render_shader_template(
            source_dir,
            "multiply_rolling_depthwise_bf16.comp.template",
            {
                "OUTPUT_GATE_BINDING": output_gate_binding,
                "OUTPUT_BINDING": str(3 + binding_offset),
                "KERNEL_BINDING": str(4 + binding_offset),
                "STATE_READ_BINDING": str(5 + binding_offset),
                "STATE_WRITE_BINDING": str(6 + binding_offset),
                "FRAME_COUNT": recurrent_depthwise.group(2),
                "HIDDEN_SIZE": recurrent_depthwise.group(3),
                "FINALIZE_OUTPUT_FUNCTION": finalize_output,
            },
        )

    native_fp8_batch_linear = re.fullmatch(
        r"(linear|linear_residual)_batch(\d+)_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if native_fp8_batch_linear is not None:
        operation = native_fp8_batch_linear.group(1)
        batch_tile_width, block_rows, block_columns, input_size, output_size = map(
            int, native_fp8_batch_linear.groups()[1:]
        )
        if (
            batch_tile_width <= 0
            or block_rows <= 0
            or block_columns != 128
            or input_size <= 0
            or input_size % block_columns != 0
            or output_size <= 0
            or output_size % 2 != 0
        ):
            raise ModelCompileError(
                f"invalid native batched FP8 linear shader shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            f"{operation}_batch_fp8_e4m3.comp.template",
            {
                "BATCH_TILE_WIDTH": str(batch_tile_width),
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(fp8_linear_tile_rows(output_size)),
            },
        )

    native_q8_linear = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if native_q8_linear is not None:
        operation = native_q8_linear.group(1)
        input_size, output_size = map(int, native_q8_linear.groups()[1:])
        if (
            input_size <= 0
            or input_size % Q8_0_GROUP_SIZE
            or output_size <= 0
            or output_size % 2
        ):
            raise ModelCompileError(
                f"invalid native Q8_0 linear shader shape {shader_file!r}"
            )
        has_residual = operation == "linear_residual"
        has_bias = operation == "linear_bias"
        output_binding = 2 if has_residual else 1
        weight_binding = 3 if has_residual else 2
        auxiliary_buffer = (
            "layout(set = 0, binding = 3) readonly buffer Bias { "
            "uint words[]; } bias;"
            if has_bias
            else (
                "layout(set = 0, binding = 1) readonly buffer ResidualFrames { "
                "uint words[]; } residual_frames;"
                if has_residual
                else ""
            )
        )
        finalize_output = (
            "float finalize_output(uint row, float value) {\n"
            "    return read_bf16_word(residual_frames.words[row >> 1u], row) + value;\n"
            "}"
            if has_residual
            else (
                "float finalize_output(uint row, float value) {\n"
                "    return read_bf16_word(bias.words[row >> 1u], row) + value;\n"
                "}"
                if has_bias
                else (
                    "float finalize_output(uint row, float value) {\n"
                    "    return value;\n"
                    "}"
                )
            )
        )
        return render_shader_template(
            source_dir,
            "linear_q8_0.comp.template",
            {
                "OUTPUT_BINDING": str(output_binding),
                "WEIGHT_BINDING": str(weight_binding),
                "AUXILIARY_BUFFER": auxiliary_buffer,
                "FINALIZE_OUTPUT_FUNCTION": finalize_output,
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(Q8_0_OUTPUT_TILE_ROWS),
            },
        )

    native_int4_linear = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_int4_(gptq|ct)_s(f16|bf16)_"
        r"g(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if native_int4_linear is not None:
        operation, quantization_format, scale_dtype = native_int4_linear.groups()[:3]
        group_size, input_size, output_size = map(int, native_int4_linear.groups()[3:])
        output_tile_rows = (
            INT4_GPTQ_OUTPUT_TILE_ROWS
            if quantization_format == "gptq"
            else INT4_CT_OUTPUT_TILE_ROWS
        )
        validate_native_int4_shader_shape(
            shader_file, group_size, input_size, output_size
        )
        return render_shader_template(
            source_dir,
            f"linear_int4_{quantization_format}.comp.template",
            int4_shader_replacements(
                operation=operation,
                quantization_format=quantization_format,
                scale_dtype=scale_dtype,
                batch_tile_width=None,
            )
            | {
                "GROUP_SIZE": str(group_size),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(output_tile_rows),
            },
        )

    native_int4_batch_linear = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_batch(\d+)_int4_(gptq|ct)_"
        r"s(f16|bf16)_"
        r"g(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if native_int4_batch_linear is not None:
        operation = native_int4_batch_linear.group(1)
        batch_tile_width = int(native_int4_batch_linear.group(2))
        quantization_format = native_int4_batch_linear.group(3)
        scale_dtype = native_int4_batch_linear.group(4)
        group_size, input_size, output_size = map(
            int, native_int4_batch_linear.groups()[4:]
        )
        if batch_tile_width <= 0:
            raise ModelCompileError(
                f"invalid native batched INT4 shader shape {shader_file!r}"
            )
        output_tile_rows = (
            INT4_GPTQ_OUTPUT_TILE_ROWS
            if quantization_format == "gptq"
            else INT4_CT_OUTPUT_TILE_ROWS
        )
        validate_native_int4_shader_shape(
            shader_file, group_size, input_size, output_size
        )
        return render_shader_template(
            source_dir,
            f"linear_int4_{quantization_format}.comp.template",
            int4_shader_replacements(
                operation=operation,
                quantization_format=quantization_format,
                scale_dtype=scale_dtype,
                batch_tile_width=batch_tile_width,
            )
            | {
                "GROUP_SIZE": str(group_size),
                "INPUT_SIZE": str(input_size),
                "OUTPUT_SIZE": str(output_size),
                "OUTPUT_TILE_ROWS": str(output_tile_rows),
            },
        )

    attention_gate = re.fullmatch(
        r"softplus_multiply(?:_batch(\d+))?_bf16_q(\d+)_d(\d+)_"
        r"(per_head|per_element)\.comp",
        shader_file,
    )
    if attention_gate is not None:
        batch_tile, query_heads, head_width, mode = attention_gate.groups()
        query_heads, head_width = map(int, (query_heads, head_width))
        if query_heads <= 0 or head_width <= 0 or head_width % 2:
            raise ModelCompileError(
                f"invalid softplus attention gate shape {shader_file!r}"
            )
        return render_shader_template(
            source_dir,
            (
                "softplus_multiply_bf16.comp.template"
                if batch_tile is None
                else "softplus_multiply_batch_bf16.comp.template"
            ),
            {
                "BATCH_TILE_WIDTH": batch_tile or "1",
                "QUERY_HEADS": str(query_heads),
                "HEAD_WIDTH": str(head_width),
                "PER_HEAD": "1" if mode == "per_head" else "0",
            },
        )

    shaped_templates = (
        (
            r"sigmoid_multiply_quantize_fp8_e4m3_b(\d+)_h(\d+)\.comp",
            "sigmoid_multiply_quantize_fp8_e4m3.comp.template",
            ("BLOCK_COLUMNS", "ELEMENT_COUNT"),
        ),
        (
            r"sigmoid_multiply_quantize_batch(\d+)_fp8_e4m3_b(\d+)_h(\d+)\.comp",
            "sigmoid_multiply_quantize_batch_fp8_e4m3.comp.template",
            ("BATCH_TILE_WIDTH", "BLOCK_COLUMNS", "ELEMENT_COUNT"),
        ),
        (
            r"sigmoid_multiply_batch(\d+)_bf16\.comp",
            "sigmoid_multiply_batch_bf16.comp.template",
            ("BATCH_TILE_WIDTH",),
        ),
        (
            r"rms_norm_batch(\d+)_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_batch_bf16.comp.template",
            ("BATCH_TILE_WIDTH", "HIDDEN_SIZE", "NORM_EPS", "WEIGHT_OFFSET"),
        ),
        (
            r"silu_multiply_bf16_(\d+)\.comp",
            "silu_multiply_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"sigmoid_scalar_multiply_bf16_(\d+)\.comp",
            "sigmoid_scalar_multiply_bf16.comp.template",
            ("HIDDEN_SIZE",),
        ),
        (
            r"linear_bf16_(\d+)x(\d+)\.comp",
            "linear_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_split_3way_bf16_(\d+)x(\d+)_(\d+)_(\d+)\.comp",
            "linear_split_3way_bf16.comp.template",
            (
                "INPUT_SIZE",
                "PART_A_WIDTH",
                "PART_B_WIDTH",
                "PART_C_WIDTH",
            ),
        ),
        (
            r"linear_split_recurrent_depthwise_gate_bf16_"
            r"(\d+)x(\d+)_k(\d+)_ig([012])_([012])_og([012])\.comp",
            "linear_split_recurrent_depthwise_gate_bf16.comp.template",
            (
                "INPUT_SIZE",
                "HIDDEN_SIZE",
                "FRAME_COUNT",
                "INPUT_GATE_A_INDEX",
                "INPUT_GATE_B_INDEX",
                "OUTPUT_GATE_INDEX",
            ),
        ),
        (
            r"linear_bias_bf16_(\d+)x(\d+)\.comp",
            "linear_bias_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"linear_residual_bf16_(\d+)x(\d+)\.comp",
            "linear_residual_bf16.comp.template",
            ("INPUT_SIZE", "OUTPUT_SIZE"),
        ),
        (
            r"embedding_lookup_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)\.comp",
            "embedding_lookup_bf16.comp.template",
            ("VOCAB_SIZE", "HIDDEN_SIZE", "EMBEDDING_SCALE"),
        ),
        (
            r"embedding_lookup_batch_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)\.comp",
            "embedding_lookup_batch_bf16.comp.template",
            ("VOCAB_SIZE", "HIDDEN_SIZE", "EMBEDDING_SCALE"),
        ),
        (
            r"tied_output_projection_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_bf16.comp.template",
            ("VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"tied_output_projection_batch(\d+)_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_batch_bf16.comp.template",
            ("BATCH_TILE_WIDTH", "VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"tied_output_projection_dot2_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_dot2_bf16.comp.template",
            ("VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"tied_output_projection_dot2_batch(\d+)_bf16_(\d+)x(\d+)_scale([0-9eE+.-]+)_to_f32\.comp",
            "tied_output_projection_dot2_batch_bf16.comp.template",
            ("BATCH_TILE_WIDTH", "VOCAB_SIZE", "INPUT_SIZE", "OUTPUT_SCALE"),
        ),
        (
            r"rms_norm_quantize_fp8_e4m3_b(\d+)_h(\d+)_"
            r"eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_quantize_fp8_e4m3.comp.template",
            (
                "BLOCK_COLUMNS",
                "HIDDEN_SIZE",
                "NORM_EPS",
                "WEIGHT_OFFSET",
            ),
        ),
        (
            r"rms_norm_quantize_batch(\d+)_fp8_e4m3_b(\d+)_h(\d+)_"
            r"eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_quantize_batch_fp8_e4m3.comp.template",
            (
                "BATCH_TILE_WIDTH",
                "BLOCK_COLUMNS",
                "HIDDEN_SIZE",
                "NORM_EPS",
                "WEIGHT_OFFSET",
            ),
        ),
        (
            r"rms_norm_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_bf16.comp.template",
            ("HIDDEN_SIZE", "NORM_EPS", "WEIGHT_OFFSET"),
        ),
        (
            r"rms_norm_per_head_bf16_(\d+)x(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
            "rms_norm_per_head_bf16.comp.template",
            ("HEAD_COUNT", "HEAD_WIDTH", "NORM_EPS", "WEIGHT_OFFSET"),
        ),
        (
            r"rms_norm_per_head_unscaled_bf16_(\d+)x(\d+)_eps([0-9eE+.-]+)\.comp",
            "rms_norm_per_head_unscaled_bf16.comp.template",
            ("HEAD_COUNT", "HEAD_WIDTH", "NORM_EPS"),
        ),
        (
            r"parallel_head_norm_rope_2way_temporal_bf16_h(\d+)_(\d+)_d(\d+)_r(\d+)"
            r"_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)_theta([0-9eE+.-]+)"
            r"_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)"
            r"_a([0-9eE+.-]+)_(half|interleaved|proportional)\.comp",
            "parallel_head_norm_rope_2way_temporal_bf16.comp.template",
            (
                "BRANCH_A_HEADS",
                "BRANCH_B_HEADS",
                "HEAD_WIDTH",
                "ROTARY_WIDTH",
                "NORM_EPS",
                "WEIGHT_OFFSET",
                "ROPE_THETA",
                "ROPE_FACTOR",
                "ROPE_CORRECTION_LOW",
                "ROPE_CORRECTION_HIGH",
                "ROPE_ATTENTION_FACTOR",
                "ROPE_LAYOUT",
            ),
        ),
        (
            r"parallel_head_norm_rope_2way_temporal_bf16_h(\d+)_(\d+)_d(\d+)_r(\d+)"
            r"_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)_theta([0-9eE+.-]+)"
            r"_(half|interleaved|proportional)\.comp",
            "parallel_head_norm_rope_2way_temporal_bf16.comp.template",
            (
                "BRANCH_A_HEADS",
                "BRANCH_B_HEADS",
                "HEAD_WIDTH",
                "ROTARY_WIDTH",
                "NORM_EPS",
                "WEIGHT_OFFSET",
                "ROPE_THETA",
                "ROPE_LAYOUT",
            ),
        ),
        (
            r"parallel_head_norm_rope_2way_bf16_h(\d+)_(\d+)_d(\d+)_r(\d+)"
            r"_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)_theta([0-9eE+.-]+)"
            r"_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)"
            r"_a([0-9eE+.-]+)_(half|interleaved|proportional)\.comp",
            "parallel_head_norm_rope_2way_bf16.comp.template",
            (
                "BRANCH_A_HEADS",
                "BRANCH_B_HEADS",
                "HEAD_WIDTH",
                "ROTARY_WIDTH",
                "NORM_EPS",
                "WEIGHT_OFFSET",
                "ROPE_THETA",
                "ROPE_FACTOR",
                "ROPE_CORRECTION_LOW",
                "ROPE_CORRECTION_HIGH",
                "ROPE_ATTENTION_FACTOR",
                "ROPE_LAYOUT",
            ),
        ),
        (
            r"parallel_head_norm_rope_2way_bf16_h(\d+)_(\d+)_d(\d+)_r(\d+)"
            r"_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)_theta([0-9eE+.-]+)"
            r"_(half|interleaved|proportional)\.comp",
            "parallel_head_norm_rope_2way_bf16.comp.template",
            (
                "BRANCH_A_HEADS",
                "BRANCH_B_HEADS",
                "HEAD_WIDTH",
                "ROTARY_WIDTH",
                "NORM_EPS",
                "WEIGHT_OFFSET",
                "ROPE_THETA",
                "ROPE_LAYOUT",
            ),
        ),
        (
            r"rotary_bf16_(\d+)x(\d+)_r(\d+)_theta([0-9eE+.-]+)"
            r"_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)"
            r"_a([0-9eE+.-]+)_(half|interleaved|proportional)\.comp",
            "rotary_bf16.comp.template",
            (
                "HEAD_COUNT",
                "HEAD_WIDTH",
                "ROTARY_WIDTH",
                "ROPE_THETA",
                "ROPE_FACTOR",
                "ROPE_CORRECTION_LOW",
                "ROPE_CORRECTION_HIGH",
                "ROPE_ATTENTION_FACTOR",
                "ROPE_LAYOUT",
            ),
        ),
        (
            r"rotary_bf16_(\d+)x(\d+)_r(\d+)_theta([0-9eE+.-]+)_(half|interleaved|proportional)\.comp",
            "rotary_bf16.comp.template",
            ("HEAD_COUNT", "HEAD_WIDTH", "ROTARY_WIDTH", "ROPE_THETA", "ROPE_LAYOUT"),
        ),
        (
            r"append_kv_state_bf16_(\d+)x(\d+)\.comp",
            "append_kv_state_bf16.comp.template",
            ("KV_HEADS", "HEAD_WIDTH"),
        ),
        (
            r"greedy_sampler_f32_(\d+)\.comp",
            "greedy_sampler_f32.comp.template",
            ("VOCAB_SIZE",),
        ),
        (
            r"greedy_sampler_repetition_f32_(\d+)_rp([0-9eE+.-]+)_pp([0-9eE+.-]+)\.comp",
            "greedy_sampler_repetition_f32.comp.template",
            ("VOCAB_SIZE", "REPETITION_PENALTY", "PRESENCE_PENALTY"),
        ),
        (
            r"greedy_sampler_runtime_f32_(\d+)\.comp",
            "greedy_sampler_runtime_f32.comp.template",
            ("VOCAB_SIZE",),
        ),
        (
            r"temperature_top_k_candidates_f32_(\d+)_k(\d+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_candidates_f32.comp.template",
            ("VOCAB_SIZE", "TOP_K", "PARTITION_COUNT", "LOCAL_SIZE_X"),
        ),
        (
            r"temperature_top_k_candidates_repetition_f32_(\d+)_rp([0-9eE+.-]+)_pp([0-9eE+.-]+)_k(\d+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_candidates_repetition_f32.comp.template",
            (
                "VOCAB_SIZE",
                "REPETITION_PENALTY",
                "PRESENCE_PENALTY",
                "TOP_K",
                "PARTITION_COUNT",
                "LOCAL_SIZE_X",
            ),
        ),
        (
            r"temperature_top_k_candidates_runtime_f32_(\d+)_kc(\d+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_candidates_runtime_f32.comp.template",
            ("VOCAB_SIZE", "TOP_K_CAPACITY", "PARTITION_COUNT", "LOCAL_SIZE_X"),
        ),
        (
            r"record_seen_token_(\d+)\.comp",
            "record_seen_token.comp.template",
            ("VOCAB_SIZE",),
        ),
        (
            r"record_seen_tokens_batch64_(\d+)\.comp",
            "record_seen_tokens_batch64.comp.template",
            ("VOCAB_SIZE",),
        ),
        (
            r"temperature_top_k_top_p_sampler_f32_t([0-9eE+.-]+)_k(\d+)_p([0-9eE+.-]+)_m([0-9eE+.-]+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_top_p_sampler_f32.comp.template",
            (
                "TEMPERATURE",
                "TOP_K",
                "TOP_P",
                "MIN_P",
                "PARTITION_COUNT",
                "LOCAL_SIZE_X",
            ),
        ),
        (
            r"temperature_top_k_top_p_sampler_runtime_f32_kc(\d+)_g(\d+)_l(\d+)\.comp",
            "temperature_top_k_top_p_sampler_runtime_f32.comp.template",
            ("TOP_K_CAPACITY", "PARTITION_COUNT", "LOCAL_SIZE_X"),
        ),
        (
            r"split_batch(\d+)_bf16_2x(\d+)x(\d+)_head_interleaved\.comp",
            "split_batch_bf16_2way_head_interleaved.comp.template",
            ("BATCH_TILE_WIDTH", "BLOCKS", "BLOCK_PART_WIDTH"),
        ),
        (
            r"split_bf16_2x(\d+)\.comp",
            "split_bf16_2way.comp.template",
            ("PART_WIDTH",),
        ),
        (
            r"split_bf16_3x(\d+)\.comp",
            "split_bf16_3way.comp.template",
            ("PART_WIDTH",),
        ),
        (
            r"split_bf16_3x(\d+)_(\d+)_(\d+)\.comp",
            "split_bf16_3way_widths.comp.template",
            ("PART_A_WIDTH", "PART_B_WIDTH", "PART_C_WIDTH"),
        ),
        (
            r"split_bf16_2x(\d+)x(\d+)_head_interleaved\.comp",
            "split_bf16_2way_head_interleaved.comp.template",
            ("BLOCKS", "BLOCK_PART_WIDTH"),
        ),
        (
            r"causal_conv1d_silu_temporal_bf16_c(\d+)_k(\d+)\.comp",
            "causal_conv1d_silu_temporal_bf16.comp.template",
            ("CHANNELS", "KERNEL_WIDTH"),
        ),
        (
            r"causal_conv1d_silu_bf16_c(\d+)_k(\d+)\.comp",
            "causal_conv1d_silu_bf16.comp.template",
            ("CHANNELS", "KERNEL_WIDTH"),
        ),
        (
            r"rolling_state_update_bf16_(\d+)x(\d+)\.comp",
            "rolling_state_update_bf16.comp.template",
            ("FRAME_COUNT", "HIDDEN_SIZE"),
        ),
        (
            r"depthwise_conv1d_bf16_(\d+)x(\d+)\.comp",
            "depthwise_conv1d_bf16.comp.template",
            ("FRAME_COUNT", "HIDDEN_SIZE"),
        ),
        (
            r"scaled_add_bf16_(\d+)_scale([0-9eE+.-]+)\.comp",
            "scaled_add_bf16.comp.template",
            ("ELEMENT_COUNT", "RESIDUAL_SCALE"),
        ),
        (
            r"add_bf16_(\d+)\.comp",
            "add_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"multiply_bf16_(\d+)\.comp",
            "multiply_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"scalar_multiply_bf16_(\d+)\.comp",
            "scalar_multiply_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"gelu_tanh_bf16_(\d+)\.comp",
            "gelu_tanh_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
        (
            r"silu_bf16_(\d+)\.comp",
            "silu_bf16.comp.template",
            ("ELEMENT_COUNT",),
        ),
    )
    for pattern, template, names in shaped_templates:
        match = re.fullmatch(pattern, shader_file)
        if match is not None:
            replacements = dict(zip(names, match.groups(), strict=True))
            if template in (
                "causal_conv1d_silu_bf16.comp.template",
                "causal_conv1d_silu_temporal_bf16.comp.template",
            ):
                channels = int(replacements["CHANNELS"])
                kernel_width = int(replacements["KERNEL_WIDTH"])
                if channels % 2 != 0 or kernel_width % 2 != 0:
                    raise ModelCompileError(
                        "packed BF16 causal convolution requires even channel and kernel widths, "
                        f"got {channels} channels and kernel width {kernel_width}"
                    )
            if "ROPE_LAYOUT" in replacements:
                rope_layout = replacements.pop("ROPE_LAYOUT")
                replacements["ROPE_INTERLEAVED"] = (
                    "true" if rope_layout == "interleaved" else "false"
                )
                replacements["ROPE_PROPORTIONAL"] = (
                    "true" if rope_layout == "proportional" else "false"
                )
                replacements["ROPE_YARN"] = (
                    "true" if "ROPE_FACTOR" in replacements else "false"
                )
                replacements.setdefault("ROPE_FACTOR", "1.0")
                replacements.setdefault("ROPE_CORRECTION_LOW", "0.0")
                replacements.setdefault("ROPE_CORRECTION_HIGH", "1.0")
                replacements.setdefault("ROPE_ATTENTION_FACTOR", "1.0")
            return render_shader_template(source_dir, template, replacements)

    per_layer_embedding_shape = re.fullmatch(
        r"per_layer_embedding_bf16_v(\d+)_h(\d+)_p(\d+)_l(\d+)of(\d+)"
        r"_c(\d+)r(\d+)"
        r"_eps([0-9eE+.-]+)_tes([0-9eE+.-]+)_pes([0-9eE+.-]+)"
        r"_mps([0-9eE+.-]+)_cs([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if per_layer_embedding_shape is not None:
        (
            vocab_size,
            hidden_size,
            per_layer_width,
            layer_index,
            layer_count,
            chunk_count,
            chunk_rows,
        ) = map(int, per_layer_embedding_shape.groups()[:7])
        if hidden_size % 2 or per_layer_width % 2:
            raise ModelCompileError(
                "packed BF16 per-layer embeddings require even hidden and per-layer widths"
            )
        if not 0 <= layer_index < layer_count:
            raise ModelCompileError(
                f"per-layer embedding index {layer_index} is outside {layer_count} layers"
            )
        if chunk_count <= 0 or chunk_rows <= 0:
            raise ModelCompileError(
                "per-layer embedding requires positive chunk count and row capacity"
            )
        chunk_bindings = "\n".join(
            f"layout(set = 0, binding = {2 + index}) readonly buffer "
            f"PerLayerEmbeddingChunk{index} {{ uint words[]; }} "
            f"per_layer_embedding_chunk_{index};"
            for index in range(chunk_count)
        )
        chunk_reads = "\n".join(
            f"    if (chunk == {index}u) return "
            f"per_layer_embedding_chunk_{index}.words[row * PACKED_WORDS + word];"
            for index in range(chunk_count)
        )
        return render_shader_template(
            source_dir,
            "per_layer_embedding_bf16.comp.template",
            {
                "VOCAB_SIZE": str(vocab_size),
                "HIDDEN_SIZE": str(hidden_size),
                "PER_LAYER_WIDTH": str(per_layer_width),
                "LAYER_INDEX": str(layer_index),
                "LAYER_COUNT": str(layer_count),
                "EMBEDDING_CHUNK_COUNT": str(chunk_count),
                "EMBEDDING_CHUNK_ROWS": str(chunk_rows),
                "PER_LAYER_EMBEDDING_BINDINGS": chunk_bindings,
                "PER_LAYER_EMBEDDING_READS": chunk_reads,
                "MODEL_PROJECTION_BINDING": str(2 + chunk_count),
                "PROJECTION_NORM_BINDING": str(3 + chunk_count),
                "STREAM_CONTROL_BINDING": str(4 + chunk_count),
                "NORM_EPS": per_layer_embedding_shape.group(8),
                "TOKEN_EMBEDDING_SCALE": per_layer_embedding_shape.group(9),
                "PER_LAYER_EMBEDDING_SCALE": per_layer_embedding_shape.group(10),
                "MODEL_PROJECTION_SCALE": per_layer_embedding_shape.group(11),
                "COMBINATION_SCALE": per_layer_embedding_shape.group(12),
            },
        )

    attention_shape = re.fullmatch(
        r"gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)(?:_w(\d+))?(_sinks)?\.comp",
        shader_file,
    )
    if attention_shape is not None:
        query_heads, kv_heads, head_width = map(int, attention_shape.groups()[:3])
        if query_heads % kv_heads != 0:
            raise ModelCompileError(
                f"query head count {query_heads} is not divisible by KV head count {kv_heads}"
            )
        local_size, tile_tokens = attention_workgroup_shape(head_width)
        if head_width < 2 or head_width % 2 != 0 or tile_tokens == 0:
            raise ModelCompileError(
                f"attention head width {head_width} cannot be tiled into a Vulkan workgroup"
            )
        return render_shader_template(
            source_dir,
            "gqa_attention_bf16.comp.template",
            {
                "QUERY_HEADS": str(query_heads),
                "KV_HEADS": str(kv_heads),
                "QUERY_GROUPS_PER_KV_HEAD": str(query_heads // kv_heads),
                "HEAD_WIDTH": str(head_width),
                "LOCAL_SIZE": str(local_size),
                "TILE_TOKENS": str(tile_tokens),
                "ATTENTION_SCALE": attention_shape.group(4),
                "ATTENTION_WINDOW": attention_shape.group(5) or "0",
                "HAS_SINKS": "1" if attention_shape.group(6) else "0",
            },
        )

    temporal_attention_shape = re.fullmatch(
        r"append_gqa_attention_temporal_read_bf16_q(\d+)_kv(\d+)_d(\d+)"
        r"_scale([0-9eE+.-]+)(?:_w(\d+))?(_sinks)?\.comp",
        shader_file,
    )
    if temporal_attention_shape is not None:
        query_heads, kv_heads, head_width = map(
            int, temporal_attention_shape.groups()[:3]
        )
        if query_heads % kv_heads != 0:
            raise ModelCompileError(
                f"query head count {query_heads} is not divisible by KV head count {kv_heads}"
            )
        local_size, tile_tokens = attention_workgroup_shape(head_width)
        if head_width < 2 or head_width % 2 != 0 or tile_tokens == 0:
            raise ModelCompileError(
                f"attention head width {head_width} cannot be tiled into a Vulkan workgroup"
            )
        has_sinks = temporal_attention_shape.group(6) is not None
        return render_shader_template(
            source_dir,
            "append_gqa_attention_temporal_read_bf16.comp.template",
            {
                "QUERY_HEADS": str(query_heads),
                "KV_HEADS": str(kv_heads),
                "QUERY_GROUPS_PER_KV_HEAD": str(query_heads // kv_heads),
                "HEAD_WIDTH": str(head_width),
                "LOCAL_SIZE": str(local_size),
                "TILE_TOKENS": str(tile_tokens),
                "ATTENTION_SCALE": temporal_attention_shape.group(4),
                "ATTENTION_WINDOW": temporal_attention_shape.group(5) or "0",
                "HAS_SINKS": "1" if has_sinks else "0",
                "ATTENTION_SINK_BINDING": "5",
                "STATE_READ_BINDING": "6" if has_sinks else "5",
            },
        )

    temporal_kv_commit_shape = re.fullmatch(
        r"append_kv_temporal_commit_bf16_kv(\d+)_d(\d+)_w(\d+)(_sinks)?\.comp",
        shader_file,
    )
    if temporal_kv_commit_shape is not None:
        kv_heads, head_width = map(int, temporal_kv_commit_shape.groups()[:2])
        if head_width < 2 or head_width % 2 != 0:
            raise ModelCompileError(
                f"KV head width {head_width} cannot be packed as BF16 pairs"
            )
        return render_shader_template(
            source_dir,
            "append_kv_temporal_commit_bf16.comp.template",
            {
                "KV_HEADS": str(kv_heads),
                "HEAD_WIDTH": str(head_width),
                "ATTENTION_WINDOW": temporal_kv_commit_shape.group(3),
                "STATE_WRITE_BINDING": (
                    "7" if temporal_kv_commit_shape.group(4) else "6"
                ),
            },
        )

    append_attention_shape = re.fullmatch(
        r"append_gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)"
        r"(?:_w(\d+))?(_sinks)?\.comp",
        shader_file,
    )
    if append_attention_shape is not None:
        query_heads, kv_heads, head_width = map(
            int, append_attention_shape.groups()[:3]
        )
        if query_heads % kv_heads != 0:
            raise ModelCompileError(
                f"query head count {query_heads} is not divisible by KV head count {kv_heads}"
            )
        local_size, tile_tokens = attention_workgroup_shape(head_width)
        if head_width < 2 or head_width % 2 != 0 or tile_tokens == 0:
            raise ModelCompileError(
                f"attention head width {head_width} cannot be tiled into a Vulkan workgroup"
            )
        has_sinks = append_attention_shape.group(6) is not None
        return render_shader_template(
            source_dir,
            "append_gqa_attention_bf16.comp.template",
            {
                "QUERY_HEADS": str(query_heads),
                "KV_HEADS": str(kv_heads),
                "QUERY_GROUPS_PER_KV_HEAD": str(query_heads // kv_heads),
                "HEAD_WIDTH": str(head_width),
                "LOCAL_SIZE": str(local_size),
                "TILE_TOKENS": str(tile_tokens),
                "ATTENTION_SCALE": append_attention_shape.group(4),
                "ATTENTION_WINDOW": append_attention_shape.group(5) or "0",
                "HAS_SINKS": "1" if has_sinks else "0",
                "ATTENTION_SINK_BINDING": "5",
                "STATE_READ_BINDING": "6" if has_sinks else "5",
                "STATE_WRITE_BINDING": "7" if has_sinks else "6",
            },
        )

    gated_delta_shape = re.fullmatch(
        r"gated_delta_(step|scan)_k(\d+)x(\d+)_v(\d+)x(\d+)"
        r"_a(f32|bf16)_dt(f32|bf16)_n(f32|bf16)_eps([0-9eE+.-]+)"
        r"(?:_qfp8b(\d+))?\.comp",
        shader_file,
    )
    if gated_delta_shape is not None:
        execution_mode = gated_delta_shape.group(1)
        key_heads, key_width, value_heads, value_width = map(
            int, gated_delta_shape.groups()[1:5]
        )
        if value_heads % key_heads != 0:
            raise ModelCompileError(
                f"gated-delta value head count {value_heads} is not divisible by key head count {key_heads}"
            )
        if value_width > 1024 or value_width < 2 or value_width % 2 != 0:
            raise ModelCompileError(
                f"gated-delta value head width {value_width} is not a supported workgroup width"
            )
        quantized_block_columns = (
            int(gated_delta_shape.group(10))
            if gated_delta_shape.group(10) is not None
            else None
        )
        if quantized_block_columns is not None and (
            quantized_block_columns != value_width or value_width % 4
        ):
            raise ModelCompileError(
                f"gated-delta FP8 block width {quantized_block_columns} does not "
                f"match value head width {value_width}"
            )
        physical_output_bindings = ""
        fp8_extensions = ""
        fp8_helpers = ""
        a_log_binding = 5
        if quantized_block_columns is not None:
            physical_output_bindings = (
                "layout(set = 0, binding = 5) buffer QuantizedOutput "
                "{ uint words[]; } quantized_output;\n"
                "layout(set = 0, binding = 6) buffer OutputScale "
                "{ float values[]; } output_scale;"
            )
            fp8_extensions = (
                "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
                "#extension GL_EXT_float_e4m3 : require"
            )
            fp8_helpers = (
                "uint pack_fp8(fe4m3vec4 value) {\n"
                "    u8vec4 bits = floate4m3BitsToUintEXT(value);\n"
                "    return uint(bits.x)\n"
                "        | (uint(bits.y) << 8u)\n"
                "        | (uint(bits.z) << 16u)\n"
                "        | (uint(bits.w) << 24u);\n"
                "}"
            )
            a_log_binding = 7
        if quantized_block_columns is None:
            output_store = (
                "    if ((value_dim & 1u) == 0u) {\n"
                "        uint output_index = value_head * VALUE_HEAD_WIDTH + value_dim;\n"
                + (
                    "        uint output_word = (position * VALUE_WIDTH + output_index) >> 1u;\n"
                    if execution_mode == "scan"
                    else "        uint output_word = output_index >> 1u;\n"
                )
                + "        output_frame.words[output_word] =\n"
                "            (f32_to_bf16(head_output[value_dim + 1u]) << 16u)\n"
                "            | f32_to_bf16(head_output[value_dim]);\n"
                "    }"
            )
        else:
            output_store = (
                "    uint rounded = f32_to_bf16(head_output[value_dim]);\n"
                "    head_output[value_dim] = bf16_to_f32(rounded);\n"
                "    float subgroup_max = subgroupMax(abs(head_output[value_dim]));\n"
                "    if (gl_SubgroupInvocationID == 0u) {\n"
                "        reduction[gl_SubgroupID] = subgroup_max;\n"
                "    }\n"
                "    barrier();\n"
                "    float block_max = 0.0;\n"
                "    for (uint subgroup = 0u; subgroup < gl_NumSubgroups; subgroup++) {\n"
                "        block_max = max(block_max, reduction[subgroup]);\n"
                "    }\n"
                "    float block_scale = block_max > 0.0 ? block_max / 448.0 : 1.0;\n"
                + (
                    "    uint scale_index = position * VALUE_HEADS + value_head;\n"
                    "    uint output_base = position * VALUE_WIDTH + value_head * VALUE_HEAD_WIDTH;\n"
                    if execution_mode == "scan"
                    else "    uint scale_index = value_head;\n"
                    "    uint output_base = value_head * VALUE_HEAD_WIDTH;\n"
                )
                + "    if (value_dim == 0u) {\n"
                "        output_scale.values[scale_index] = block_scale;\n"
                "    }\n"
                "    if ((value_dim & 1u) == 0u) {\n"
                "        uint output_index = output_base + value_dim;\n"
                "        output_frame.words[output_index >> 1u] =\n"
                "            (f32_to_bf16(head_output[value_dim + 1u]) << 16u)\n"
                "            | f32_to_bf16(head_output[value_dim]);\n"
                "    }\n"
                "    if ((value_dim & 3u) == 0u) {\n"
                "        uint output_index = output_base + value_dim;\n"
                "        quantized_output.words[output_index >> 2u] = pack_fp8(\n"
                "            fe4m3vec4(vec4(\n"
                "                head_output[value_dim],\n"
                "                head_output[value_dim + 1u],\n"
                "                head_output[value_dim + 2u],\n"
                "                head_output[value_dim + 3u]\n"
                "            ) / block_scale)\n"
                "        );\n"
                "    }"
            )
        return render_shader_template(
            source_dir,
            (
                "gated_delta_scan.comp.template"
                if execution_mode == "scan"
                else "gated_delta_step.comp.template"
            ),
            {
                "KEY_HEADS": str(key_heads),
                "KEY_HEAD_WIDTH": str(key_width),
                "VALUE_HEADS": str(value_heads),
                "VALUE_HEAD_WIDTH": str(value_width),
                "KEY_HEAD_REPEAT": str(value_heads // key_heads),
                "READ_A_LOG": scalar_parameter_read_expression(
                    "a_log", gated_delta_shape.group(6)
                ),
                "READ_DT_BIAS": scalar_parameter_read_expression(
                    "dt_bias", gated_delta_shape.group(7)
                ),
                "READ_NORM_WEIGHT": scalar_parameter_read_expression(
                    "norm_weight", gated_delta_shape.group(8)
                ),
                "NORM_EPS": gated_delta_shape.group(9),
                "FP8_EXTENSIONS": fp8_extensions,
                "PHYSICAL_OUTPUT_BINDINGS": physical_output_bindings,
                "A_LOG_BINDING": str(a_log_binding),
                "DT_BIAS_BINDING": str(a_log_binding + 1),
                "NORM_WEIGHT_BINDING": str(a_log_binding + 2),
                "STATE_READ_BINDING": str(a_log_binding + 3),
                "STATE_WRITE_BINDING": str(a_log_binding + 4),
                "FP8_HELPERS": fp8_helpers,
                "OUTPUT_STORE": output_store,
            },
        )

    rg_lru_shape = re.fullmatch(
        r"rg_lru_step_bf16_h(\d+)_b(\d+)x(\d+)_k(\d+)\.comp", shader_file
    )
    if rg_lru_shape is not None:
        width, heads, block_width, kernel_width = map(int, rg_lru_shape.groups())
        if heads * block_width != width:
            raise ModelCompileError(
                f"RG-LRU block shape {heads}x{block_width} does not equal width {width}"
            )
        if block_width > 1024 or block_width % 2:
            raise ModelCompileError(
                f"RG-LRU block width {block_width} is not a supported workgroup width"
            )
        return render_shader_template(
            source_dir,
            "rg_lru_step_bf16.comp.template",
            {
                "WIDTH": str(width),
                "HEADS": str(heads),
                "BLOCK_WIDTH": str(block_width),
                "KERNEL_WIDTH": str(kernel_width),
            },
        )

    custom_moe_topk_shape = re.fullmatch(
        r"moe_topk(?:_batch(\d+))?_(sigmoid|softmax)_bf16_e(\d+)_k(\d+)_"
        r"norm([01])_cap([0-9eE+.-]+)_"
        r"(biasf32|biasbf16|nobias)\.comp",
        shader_file,
    )
    if custom_moe_topk_shape is not None:
        (
            batch_tile,
            activation,
            num_experts,
            experts_per_token,
            normalize_selected,
            logit_softcap,
            bias_mode,
        ) = custom_moe_topk_shape.groups()
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "sparse routing supports only frame-parallel batch tiles"
            )
        num_experts, experts_per_token = map(int, (num_experts, experts_per_token))
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        has_bias = bias_mode != "nobias"
        return render_shader_template(
            source_dir,
            (
                "moe_topk_routed_bf16.comp.template"
                if batch_tile is None
                else "moe_topk_routed_batch1_bf16.comp.template"
            ),
            {
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
                "ROUTER_SIGMOID": "1" if activation == "sigmoid" else "0",
                "NORMALIZE_SELECTED": normalize_selected,
                "LOGIT_SOFTCAP": logit_softcap,
                "BIAS_DECLARATION": (
                    "layout(set = 0, binding = 2) readonly buffer RouterSelectionBias "
                    "{ uint words[]; } router_selection_bias;"
                    if has_bias
                    else ""
                ),
                "READ_SELECTION_BIAS": (
                    "uintBitsToFloat(router_selection_bias.words[expert])"
                    if bias_mode == "biasf32"
                    else "unpack_bf16(router_selection_bias.words[expert >> 1u], expert)"
                    if bias_mode == "biasbf16"
                    else "0.0"
                ),
                # Runtime descriptors are ordered as inputs, outputs, then
                # parameters. A biased router therefore keeps its route output
                # at binding 1 and places the selection-bias parameter at 2.
                "ROUTES_BINDING": "1",
            },
        )

    moe_topk_shape = re.fullmatch(
        r"moe_topk(?:_batch(\d+))?_bf16_e(\d+)_k(\d+)\.comp", shader_file
    )
    if moe_topk_shape is not None:
        batch_tile, num_experts, experts_per_token = moe_topk_shape.groups()
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "sparse routing supports only frame-parallel batch tiles"
            )
        num_experts, experts_per_token = map(int, (num_experts, experts_per_token))
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_shader_template(
            source_dir,
            (
                "moe_topk_bf16.comp.template"
                if batch_tile is None
                else "moe_topk_batch1_bf16.comp.template"
            ),
            {
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
            },
        )

    sparse_moe_int4_shape = re.fullmatch(
        r"sparse_moe_(gate_up|down)(?:_batch(\d+))?_int4_ct_"
        r"s(f16|bf16)_g(\d+)_h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_int4_shape is not None:
        (
            stage,
            batch_tile,
            scale_dtype,
            group_size,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = sparse_moe_int4_shape.groups()
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "INT4 sparse experts support only frame-parallel batch tiles"
            )
        (
            group_size,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = map(
            int,
            (
                group_size,
                hidden_size,
                intermediate_size,
                num_experts,
                experts_per_token,
            ),
        )
        input_width = hidden_size if stage == "gate_up" else intermediate_size
        output_width = intermediate_size if stage == "gate_up" else hidden_size
        if (
            group_size <= 0
            or group_size % INT4_VALUES_PER_PACKED_WORD != 0
            or input_width % group_size != 0
            or input_width % 8 != 0
            or output_width % 2 != 0
            or not 0 < experts_per_token <= num_experts <= 4096
        ):
            raise ModelCompileError(
                f"invalid INT4 sparse expert geometry {shader_file!r}"
            )
        read_scale_body = (
            "    vec2 values = unpackHalf2x16(expert_"
            f"{'input' if stage == 'gate_up' else 'output'}"
            "_scales.words[index >> 1u]);\n"
            "    return (index & 1u) == 0u ? values.x : values.y;"
            if scale_dtype == "f16"
            else (
                "    return read_bf16_word(expert_"
                f"{'input' if stage == 'gate_up' else 'output'}"
                "_scales.words[index >> 1u], index);"
            )
        )
        return render_shader_template(
            source_dir,
            f"sparse_moe_{stage}_int4_ct.comp.template",
            {
                "GROUP_SIZE": str(group_size),
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
                "TILE_ROWS": str(INT4_CT_OUTPUT_TILE_ROWS),
                "BATCH_CONTROL": (
                    "layout(push_constant) uniform DispatchControl { uint "
                    "expert_start; } dispatch_control;"
                    if batch_tile is None
                    else "layout(push_constant) uniform BatchControl { uint "
                    "batch_width; uint expert_start; } batch_control;"
                ),
                "EXPERT_START": (
                    "dispatch_control.expert_start"
                    if batch_tile is None
                    else "batch_control.expert_start"
                ),
                "BATCH_INDEX": ("0u" if batch_tile is None else "gl_WorkGroupID.y"),
                "BATCH_WIDTH": (
                    "1u" if batch_tile is None else "batch_control.batch_width"
                ),
                "READ_SCALE_BODY": read_scale_body,
            },
        )

    sparse_moe_fp8_shape = re.fullmatch(
        r"sparse_moe_(gate_up|down)(?:_batch(\d+))?_fp8_e4m3_"
        r"b(\d+)x(\d+)_h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_fp8_shape is not None:
        (
            stage,
            batch_tile,
            block_rows,
            block_columns,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = sparse_moe_fp8_shape.groups()
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "FP8 sparse experts support only frame-parallel batch tiles"
            )
        (
            block_rows,
            block_columns,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = map(
            int,
            (
                block_rows,
                block_columns,
                hidden_size,
                intermediate_size,
                num_experts,
                experts_per_token,
            ),
        )
        if hidden_size % 2 or intermediate_size % 2:
            raise ModelCompileError(
                "packed BF16 activations for FP8 sparse experts require even dimensions"
            )
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_shader_template(
            source_dir,
            (
                f"sparse_moe_{stage}_fp8_e4m3.comp.template"
                if batch_tile is None
                else f"sparse_moe_{stage}_batch1_fp8_e4m3.comp.template"
            ),
            {
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
                "TILE_ROWS": str(
                    FP8_SPARSE_GATE_UP_TILE_ROWS
                    if stage == "gate_up"
                    else FP8_SPARSE_DOWN_TILE_ROWS
                ),
            },
        )

    sparse_moe_shape = re.fullmatch(
        r"sparse_moe_(gate_up|down)(?:_batch(\d+))?_bf16_"
        r"h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_shape is not None:
        stage = sparse_moe_shape.group(1)
        batch_tile = sparse_moe_shape.group(2)
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "BF16 sparse experts support only frame-parallel batch tiles"
            )
        hidden_size, intermediate_size, num_experts, experts_per_token = map(
            int, sparse_moe_shape.groups()[2:]
        )
        if hidden_size % 2 or intermediate_size % 2:
            raise ModelCompileError(
                "packed BF16 sparse experts require even dimensions"
            )
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_shader_template(
            source_dir,
            (
                f"sparse_moe_{stage}_bf16.comp.template"
                if batch_tile is None
                else f"sparse_moe_{stage}_batch1_bf16.comp.template"
            ),
            {
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
            },
        )

    moe_reduce_shape = re.fullmatch(
        r"moe_reduce(?:_batch(\d+))?_bf16_h(\d+)_k(\d+)_"
        r"scale([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if moe_reduce_shape is not None:
        batch_tile, hidden_size, experts_per_token, routed_scale = (
            moe_reduce_shape.groups()
        )
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "sparse reduction supports only frame-parallel batch tiles"
            )
        hidden_size, experts_per_token = map(int, (hidden_size, experts_per_token))
        return render_shader_template(
            source_dir,
            (
                "moe_reduce_bf16.comp.template"
                if batch_tile is None
                else "moe_reduce_batch1_bf16.comp.template"
            ),
            {
                "HIDDEN_SIZE": str(hidden_size),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
                "ROUTED_SCALE": routed_scale,
            },
        )

    raise ModelCompileError(f"missing shader source or template for {shader_file}")


def render_shader_template(
    source_dir: Path, template_file: str, replacements: dict[str, str]
) -> str:
    template_path = source_dir / template_file
    if not template_path.exists():
        raise ModelCompileError(f"missing shader template {template_path}")
    rendered = template_path.read_text()
    for name, value in replacements.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
    unresolved = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", rendered)))
    if unresolved:
        raise ModelCompileError(
            f"shader template {template_path} has unresolved values: {', '.join(unresolved)}"
        )
    return rendered
