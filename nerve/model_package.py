from __future__ import annotations

from nerve.model_package_common import *
from nerve.model_package_manifest import *
from nerve.model_package_batching import *
from nerve.model_package_shaders import *
from nerve.model_package_shader_templates import *
from nerve.model_package_shader_compiler import *
from nerve.model_package_assets import *
from nerve.model_package_integrity import *
from nerve.model_package_validation import *
from nerve.model_package_tensors import *
from nerve.model_package_derived_tensors import *

def compile_model_package(
    model_dir: Path,
    *,
    transpiled_dir: Path | None = None,
    lowered_dir: Path | None = None,
    package_dir: Path | None = None,
    shader_source_dir: Path = Path("runtime-rs/shaders"),
    event_sink: Callable[[Json], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> CompiledModelReport:
    slug = compiled_model_slug(model_dir)
    package_dir = package_dir or DEFAULT_COMPILED_MODELS_DIR / slug
    transpiled_dir = transpiled_dir or package_dir / "transpiled"
    lowered_dir = lowered_dir or package_dir / "lowered"

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    structure = transpile_model(
        model_dir,
        transpiled_dir,
        progress=lambda current, total, component_id: emit_compile_event(
            event_sink,
            "ComponentTranspiled",
            current=current,
            total=total,
            component_id=component_id,
        ),
        cancel_requested=cancel_requested,
    )
    check_compile_cancelled(cancel_requested)
    tensor_index = read_json(transpiled_dir / "tensors.json")
    model_graph = read_json(transpiled_dir / "model.json")
    derive_output_projection_tensors(model_graph, tensor_index)
    write_json(transpiled_dir / "tensors.json", tensor_index)
    write_json(transpiled_dir / "model.json", model_graph)
    check_compile_cancelled(cancel_requested)
    if lowered_dir.exists():
        shutil.rmtree(lowered_dir)
    lowered = lower_execution_graph(
        transpiled_dir,
        lowered_dir,
        progress=lambda current, total, component_id: emit_compile_event(
            event_sink,
            "ComponentLoweringStarted",
            current=current,
            total=total,
            component_id=component_id,
        ),
        cancel_requested=cancel_requested,
    )
    check_compile_cancelled(cancel_requested)
    derive_internal_q8_linear_tensors(lowered["index"], lowered_dir, tensor_index)
    write_json(transpiled_dir / "tensors.json", tensor_index)
    tensor_index = referenced_tensor_index(
        tensor_index,
        model_graph=model_graph,
        lowered_index=lowered["index"],
        lowered_dir=lowered_dir,
    )

    emit_compile_event(
        event_sink, "ArtifactWritingStarted", package_dir=str(package_dir)
    )
    write_runtime_config_package(model_graph, package_dir)
    tokenizer_manifest = copy_tokenizer_package(
        model_dir, package_dir / TOKENIZER_PACKAGE_DIR
    )
    packaged_tensor_index = copy_tensor_package(
        tensor_index,
        package_dir,
        progress=lambda current, total, tensor_name: emit_compile_event(
            event_sink,
            "TensorPackagingStarted",
            current=current,
            total=total,
            tensor_name=tensor_name,
        ),
        cancel_requested=cancel_requested,
    )
    write_json(transpiled_dir / "tensors.json", packaged_tensor_index)
    package_manifest = build_vulkan_resident_package_manifest(
        model_graph=model_graph,
        tensor_index=packaged_tensor_index,
        lowered_index=lowered["index"],
        lowered_dir=lowered_dir,
        package_dir=package_dir,
        package_id=f"{slug}_vulkan_resident",
        shader_source_dir=shader_source_dir,
        tokenizer_manifest=tokenizer_manifest,
        event_sink=event_sink,
        cancel_requested=cancel_requested,
    )
    package_manifest["artifact_integrity"] = build_package_artifact_integrity(
        package_dir
    )
    package_manifest_path = package_dir / "vulkan_resident_package.json"
    write_json(package_manifest_path, package_manifest)
    emit_compile_event(
        event_sink, "PackageValidationStarted", package=str(package_manifest_path)
    )
    validate_compiled_package(package_dir, package_manifest)
    check_compile_cancelled(cancel_requested)

    return CompiledModelReport(
        model_dir=model_dir,
        compiled_model_dir=package_dir,
        transpiled_dir=transpiled_dir,
        lowered_dir=lowered_dir,
        package_dir=package_dir,
        package_manifest=package_manifest_path,
        model_type=structure.model_type or "unknown",
        circuit_count=lowered["index"]["summary"]["circuit_count"],
        shader_count=len(list((package_dir / "shaders").glob("*.spv"))),
    )
