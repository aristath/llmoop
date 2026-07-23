from model_package_layout_common import *

def test_compiler_derives_vulkan_features_from_compiled_spirv(tmp_path: Path) -> None:
    shader = tmp_path / "cooperative.spv"
    write_spirv_module(
        shader, [1, 9, 22, 39, 61, 63, 4433, 5116, 5118, 5345, 6019, 6022, 6915]
    )

    assert spirv_capabilities(shader) == {
        1,
        9,
        22,
        39,
        61,
        63,
        4433,
        5116,
        5118,
        5345,
        6019,
        6022,
        6915,
    }
    assert required_vulkan_features(tmp_path, {shader.name}) == [
        "cooperative_matrix",
        "shader_bfloat16_cooperative_matrix",
        "shader_bfloat16_type",
        "shader_float16",
        "shader_int16",
        "shader_int8",
        "shader_integer_dot_product",
        "shader_mixed_float_dot_product_float8_acc_float32",
        "storage_buffer16_bit_access",
        "vulkan_memory_model",
    ]
    assert required_vulkan_subgroup_operations(tmp_path, {shader.name}) == [
        "arithmetic",
        "basic",
    ]


def test_compiler_derives_vendor_device_extension_from_spirv_intrinsic(
    tmp_path: Path,
) -> None:
    shader = tmp_path / "mixed_fp8.comp"
    shader.write_text(
        """#version 460
#extension GL_EXT_spirv_intrinsics : require
spirv_instruction(
    extensions = ["SPV_VALVE_mixed_float_dot_product"],
    capabilities = [6915],
    id = 6918
)
float fp8_dot();
"""
    )

    assert required_vulkan_device_extensions(tmp_path, {shader.name}) == [
        "VK_VALVE_shader_mixed_float_dot_product"
    ]


def test_compiler_rejects_malformed_spirv_during_requirement_derivation(
    tmp_path: Path,
) -> None:
    shader = tmp_path / "malformed.spv"
    shader.write_bytes(struct.pack("<6I", 0x07230203, 0x00010600, 0, 1, 0, 4 << 16))

    with pytest.raises(ModelCompileError, match="malformed SPIR-V instruction"):
        spirv_capabilities(shader)


def test_compiler_fails_closed_for_unmodeled_spirv_capabilities(
    tmp_path: Path,
) -> None:
    shader = tmp_path / "unknown.spv"
    write_spirv_module(shader, [1, 65535])

    with pytest.raises(ModelCompileError, match="without a runtime device contract"):
        spirv_capabilities(shader)

