from model_package_layout_common import *

def test_compiler_renders_attention_pedals_from_discovered_dimensions(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rms_norm_bf16_h960_eps1e-05_offset0.comp",
        "rotary_bf16_15x64_r64_theta100000_half__sc2.comp",
        "append_kv_state_bf16_5x64__sc9.comp",
        "gqa_attention_bf16_q15_kv5_d64_scale0.125__sc6.comp",
        "greedy_sampler_f32_49152.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    norm = (tmp_path / "rms_norm_bf16_h960_eps1e-05_offset0.comp").read_text()
    rope = (tmp_path / "rotary_bf16_15x64_r64_theta100000_half__sc2.comp").read_text()
    append = (tmp_path / "append_kv_state_bf16_5x64__sc9.comp").read_text()
    attention = (
        tmp_path / "gqa_attention_bf16_q15_kv5_d64_scale0.125__sc6.comp"
    ).read_text()
    sampler = (tmp_path / "greedy_sampler_f32_49152.comp").read_text()

    assert "const uint HIDDEN_SIZE = 960u;" in norm
    assert "const float NORM_EPS = 1e-05;" in norm
    assert "const float WEIGHT_OFFSET = 0;" in norm
    assert "const uint HEAD_COUNT = 15u;" in rope
    assert "const uint ROTARY_WIDTH = 64u;" in rope
    assert "const float ROPE_THETA = 100000;" in rope
    assert "const bool ROPE_INTERLEAVED = false;" in rope
    assert "binding = 2) readonly buffer StreamControl" in rope
    assert "float sine = round_bf16(sin(angle) * ROPE_ATTENTION_FACTOR);" in rope
    assert "return round_bf16(direct + rotated);" in rope
    assert "const uint KV_HEADS = 5u;" in append
    assert "binding = 9) readonly buffer StreamControl" in append
    assert "const uint QUERY_HEADS = 15u;" in attention
    assert "const uint QUERY_GROUPS_PER_KV_HEAD = 3u;" in attention
    assert "const float ATTENTION_SCALE = 0.125;" in attention
    assert "binding = 6) readonly buffer StreamControl" in attention
    assert "const uint VOCAB_SIZE = 49152u;" in sampler
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )

