from model_package_layout_common import *

def test_write_compiled_tensor_preserves_canonical_row_major_order(
    tmp_path: Path,
) -> None:
    tensor_name = "matrix.weight"
    values = tuple(range(16))
    source_header = {
        tensor_name: {
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, len(values) * 2],
        }
    }
    source_header_payload = json.dumps(source_header).encode("utf-8")
    source = tmp_path / "source.safetensors"
    source.write_bytes(
        struct.pack("<Q", len(source_header_payload))
        + source_header_payload
        + struct.pack("<16H", *values)
    )
    destination = tmp_path / "compiled.safetensors"

    write_compiled_tensor(
        tensor_name=tensor_name,
        info={
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, len(values) * 2],
            "byte_count": len(values) * 2,
        },
        source=source,
        destination=destination,
        layout=ROW_MAJOR_LAYOUT,
    )

    compiled = destination.read_bytes()
    header_bytes = struct.unpack("<Q", compiled[:8])[0]
    payload = compiled[8 + header_bytes :]
    assert struct.unpack("<16H", payload) == values


def test_compiler_renders_row_major_matrix_and_transducer_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_bf16_768x2048.comp",
        "linear_residual_bf16_2048x768.comp",
        "embedding_lookup_bf16_32000x768_scale12.comp",
        "embedding_lookup_batch_bf16_32000x768_scale12.comp",
        "tied_output_projection_bf16_32000x768_scale0.166666667_to_f32.comp",
        "tied_output_projection_batch4_bf16_32000x768_scale0.166666667_to_f32.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "{{" not in shader
        assert "uint words[]" in shader
    assert (
        "const uint INPUT_SIZE = 768u;"
        in (tmp_path / "linear_bf16_768x2048.comp").read_text()
    )
    assert (
        "const uint VOCAB_SIZE = 32000u;"
        in (tmp_path / "embedding_lookup_bf16_32000x768_scale12.comp").read_text()
    )
    assert (
        "gl_WorkGroupID.y"
        in (tmp_path / "embedding_lookup_batch_bf16_32000x768_scale12.comp").read_text()
    )
    assert (
        "const float EMBEDDING_SCALE = 12;"
        in (tmp_path / "embedding_lookup_bf16_32000x768_scale12.comp").read_text()
    )
    assert (
        "const float OUTPUT_SCALE = 0.166666667;"
        in (
            tmp_path
            / "tied_output_projection_bf16_32000x768_scale0.166666667_to_f32.comp"
        ).read_text()
    )
    batched_projection = (
        tmp_path
        / "tied_output_projection_batch4_bf16_32000x768_scale0.166666667_to_f32.comp"
    ).read_text()
    assert "const uint BATCH_TILE_WIDTH = 4u;" in batched_projection
    assert "layout(push_constant) uniform BatchControl" in batched_projection
    assert "gl_WorkGroupID.y * BATCH_TILE_WIDTH" in batched_projection


def test_compiler_renders_direct_three_way_linear_split_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "linear_split_3way_bf16_1024x1024_1024_1024.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "const uint INPUT_SIZE = 1024u;" in source
    assert "const uint PART_A_WIDTH = 1024u;" in source
    assert "binding = 4) readonly buffer Weight" in source
    assert "output_c.words" in source
    assert "PAIRED_WEIGHT_LAYOUT" not in source
    assert "{{" not in source


def test_compiler_renders_row_major_per_layer_embedding_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "per_layer_embedding_bf16_v32000_h1024_p128_l2of8_c3r12000_"
        "eps1e-06_tes1_pes1_mps1_cs1__sc7.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "readonly buffer TokenEmbedding { uint words[]; }" in source
    assert "binding = 2) readonly buffer PerLayerEmbeddingChunk0" in source
    assert "binding = 4) readonly buffer PerLayerEmbeddingChunk2" in source
    assert "readonly buffer ModelProjection { uint words[]; }" in source
    assert "token_id * INPUT_WORDS + word" in source
    assert "uint chunk = token_id / EMBEDDING_CHUNK_ROWS;" in source
    assert "row * PACKED_WORDS + word" in source
    assert "row * INPUT_WORDS + word" in source
    assert "uvec2" not in source
    assert "layout(set = 0, binding = 7) readonly buffer StreamControl" in source
    assert "round_bf16(lo_projection + lo_identity)" in source
    assert "{{" not in source

