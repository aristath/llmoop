from __future__ import annotations

from nerve.model_transpiler_types import *
from nerve.model_transpiler_discovery import *
from nerve.model_transpiler_graph import *
from nerve.model_transpiler_tensor_index import *
from nerve.model_transpiler_quantization import *

def transpile_model(
    model_dir: Path,
    output_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> ModelStructure:
    model_dir = model_dir.expanduser()
    config = read_json(model_dir / "config.json")
    generation_config_path = model_dir / "generation_config.json"
    generation_config = (
        read_json(generation_config_path) if generation_config_path.is_file() else {}
    )
    tensor_index = make_tensor_index(model_dir)
    structure = discover_model_structure(
        model_dir,
        config,
        tensor_index["tensors"],
        generation_config=generation_config,
    )
    segment_per_layer_embedding_parameters(structure, tensor_index)
    check_compile_cancelled(cancel_requested)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "tensors.json", tensor_index)
    write_json(
        output_dir / "model.json", make_model_graph(structure, output_dir, tensor_index)
    )

    emitted_layers = [
        (layer, f"layer_{layer.index:02d}", "layers") for layer in structure.layers
    ]
    emitted_layers.extend(
        (layer, f"{draft.id}_layer_{layer.index:02d}", f"drafts/{draft.id}/layers")
        for draft in structure.draft_pedalboards
        for layer in draft.layers
    )
    total = len(emitted_layers)
    for current, (layer, pedal_id, relative_dir) in enumerate(emitted_layers, start=1):
        check_compile_cancelled(cancel_requested)
        write_json(
            output_dir / relative_dir / f"{pedal_id}.json",
            make_layer(
                structure,
                layer,
                pedal_id=pedal_id,
                runtime_role=(
                    "signal_processor"
                    if relative_dir == "layers"
                    else "draft_processor"
                ),
            ),
        )
        if progress is not None:
            progress(current, total, pedal_id)

    return structure
