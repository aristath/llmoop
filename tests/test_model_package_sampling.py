from model_package_layout_common import *

def test_compiler_renders_model_owned_sampling_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    candidate_shader_file = "temperature_top_k_candidates_f32_248320_k20_g128_l256.comp"
    sampler_shader_file = (
        "temperature_top_k_top_p_sampler_f32_t0.6_k20_p0.95_m0_g128_l256.comp"
    )

    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {candidate_shader_file, sampler_shader_file},
    )

    candidates = (tmp_path / candidate_shader_file).read_text()
    sampler = (tmp_path / sampler_shader_file).read_text()
    assert "const uint VOCAB_SIZE = 248320u;" in candidates
    assert "const uint TOP_K = 20u;" in candidates
    assert "const uint PARTITION_COUNT = 128u;" in candidates
    assert "subgroupMax(local_logit)" in candidates
    assert "const float TEMPERATURE = 0.6;" in sampler
    assert "const float TOP_P = 0.95;" in sampler
    assert "const float MIN_P = 0;" in sampler
    assert "binding = 3) readonly buffer SamplerSeed" in sampler
    assert "partition_cursors" in sampler
    assert "{{" not in candidates
    assert "{{" not in sampler


def test_compiler_renders_repetition_state_as_sampler_component_artifacts(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    tracker_file = "record_seen_token_65536.comp"
    batch_tracker_file = "record_seen_tokens_batch64_65536.comp"
    candidate_file = (
        "temperature_top_k_candidates_repetition_f32_65536_"
        "rp1.05_pp1.5_k50_g128_l256.comp"
    )

    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {tracker_file, batch_tracker_file, candidate_file},
    )

    tracker = (tmp_path / tracker_file).read_text()
    batch_tracker = (tmp_path / batch_tracker_file).read_text()
    candidates = (tmp_path / candidate_file).read_text()
    assert "const uint VOCAB_SIZE = 65536u;" in tracker
    assert "atomicOr(seen_tokens.words" in tracker
    assert "layout(push_constant) uniform PushConstants" in batch_tracker
    assert "const float REPETITION_PENALTY = 1.05;" in candidates
    assert "const float PRESENCE_PENALTY = 1.5;" in candidates
    assert "value < 0.0 ? value * REPETITION_PENALTY" in candidates
    assert "binding = 2) readonly buffer SeenTokens" in candidates
    assert all("{{" not in source for source in (tracker, batch_tracker, candidates))

