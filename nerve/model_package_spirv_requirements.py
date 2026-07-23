from nerve.model_package_common import *
from nerve.model_package_shader_selection import *

def required_shader_files(
    pedal_executions: list[Json],
    *,
    embedding_shader_file: str,
    embedding_batch_shader_file: str,
    projection_shader_file: str,
    projection_batch_shader_file: str,
    norm_shader_file: str,
    norm_batch_shader_file: str,
    sampler_shader_files: set[str],
) -> set[str]:
    return {
        norm_shader_file,
        norm_batch_shader_file,
        *sampler_shader_files,
        embedding_shader_file,
        embedding_batch_shader_file,
        projection_shader_file,
        projection_batch_shader_file,
        *(
            kernel["shader_path"].removeprefix("shaders/")
            for pedal in pedal_executions
            for kernel in pedal["kernels"]
        ),
        *(
            stage["shader_path"].removeprefix("shaders/")
            for pedal in pedal_executions
            for kernel in pedal["kernels"]
            for implementation in kernel["batch_implementations"]
            for stage in implementation["stages"]
        ),
    }


def required_vulkan_device_extensions(
    shader_dir: Path, shader_files: set[str]
) -> list[str]:
    required_glsl_extensions = set()
    required_spirv_extensions = set()
    for shader_file in shader_files:
        source = (shader_dir / shader_file).read_text()
        required_glsl_extensions.update(
            re.findall(
                r"^\s*#extension\s+(\S+)\s*:\s*require\s*$", source, re.MULTILINE
            )
        )
        required_spirv_extensions.update(
            re.findall(
                r'"(SPV_[A-Za-z0-9_]+)"',
                source,
            )
        )
    return sorted(
        {
            vulkan_extension
            for glsl_extension, vulkan_extension in (
                GLSL_VULKAN_DEVICE_EXTENSION_REQUIREMENTS.items()
            )
            if glsl_extension in required_glsl_extensions
        }
        | {
            vulkan_extension
            for spirv_extension, vulkan_extension in (
                SPIRV_VULKAN_DEVICE_EXTENSION_REQUIREMENTS.items()
            )
            if spirv_extension in required_spirv_extensions
        }
    )


def required_vulkan_features(shader_dir: Path, shader_files: set[str]) -> list[str]:
    features = set()
    for shader_file in shader_files:
        for capability in spirv_capabilities(shader_dir / shader_file):
            feature = SPIRV_CAPABILITY_VULKAN_FEATURE_REQUIREMENTS.get(capability)
            if feature is not None:
                features.add(feature)
    return sorted(features)


def required_vulkan_subgroup_operations(
    shader_dir: Path, shader_files: set[str]
) -> list[str]:
    operations = set()
    for shader_file in shader_files:
        for capability in spirv_capabilities(shader_dir / shader_file):
            operation = SPIRV_CAPABILITY_VULKAN_SUBGROUP_OPERATION_REQUIREMENTS.get(
                capability
            )
            if operation is not None:
                operations.add(operation)
    return sorted(operations)


def spirv_capabilities(shader_path: Path) -> set[int]:
    payload = shader_path.read_bytes()
    if len(payload) < 20 or len(payload) % 4 != 0:
        raise ModelCompileError(
            f"compiled shader is not a complete SPIR-V module: {shader_path}"
        )
    words = struct.unpack(f"<{len(payload) // 4}I", payload)
    if words[0] != SPIRV_MAGIC:
        raise ModelCompileError(
            f"compiled shader has an invalid SPIR-V magic word: {shader_path}"
        )
    capabilities = set()
    cursor = 5
    while cursor < len(words):
        instruction = words[cursor]
        word_count = instruction >> 16
        opcode = instruction & 0xFFFF
        if word_count == 0 or cursor + word_count > len(words):
            raise ModelCompileError(
                f"compiled shader has a malformed SPIR-V instruction at word {cursor}: {shader_path}"
            )
        if opcode == SPIRV_OP_CAPABILITY:
            if word_count != 2:
                raise ModelCompileError(
                    f"compiled shader has a malformed OpCapability at word {cursor}: {shader_path}"
                )
            capabilities.add(words[cursor + 1])
        cursor += word_count
    unsupported = capabilities - SUPPORTED_SPIRV_CAPABILITIES
    if unsupported:
        raise ModelCompileError(
            "compiled shader declares SPIR-V capabilities without a runtime device "
            f"contract {sorted(unsupported)}: {shader_path}"
        )
    return capabilities


def spirv_vulkan_requirements(
    package_dir: Path, shader_paths: set[str]
) -> tuple[list[str], list[str]]:
    capabilities = set()
    for shader_path in shader_paths:
        capabilities.update(
            spirv_capabilities(
                package_artifact_path(package_dir, shader_path, "shader")
            )
        )
    features = sorted(
        {
            SPIRV_CAPABILITY_VULKAN_FEATURE_REQUIREMENTS[capability]
            for capability in capabilities
            if capability in SPIRV_CAPABILITY_VULKAN_FEATURE_REQUIREMENTS
        }
    )
    subgroup_operations = sorted(
        {
            SPIRV_CAPABILITY_VULKAN_SUBGROUP_OPERATION_REQUIREMENTS[capability]
            for capability in capabilities
            if capability in SPIRV_CAPABILITY_VULKAN_SUBGROUP_OPERATION_REQUIREMENTS
        }
    )
    return features, subgroup_operations


