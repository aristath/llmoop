from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


Json = dict[str, Any]

PACKAGE_SCHEMA = "nerve.vulkan_resident_model_package.v3"
DEFAULT_COMPILED_MODELS_DIR = Path("compiled_models")


class ModelCompileError(RuntimeError):
    pass


class ModelCompileCancelled(ModelCompileError):
    pass


CompileEventSink = Callable[[Json], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class CompiledModelReport:
    model_dir: Path
    compiled_model_dir: Path
    transpiled_dir: Path
    lowered_dir: Path
    package_dir: Path
    package_manifest: Path
    model_type: str
    circuit_count: int
    shader_count: int

    def to_json(self) -> Json:
        return {
            "ok": True,
            "model_dir": str(self.model_dir),
            "model_type": self.model_type,
            "compiled_model_dir": str(self.compiled_model_dir),
            "transpiled_dir": str(self.transpiled_dir),
            "lowered_dir": str(self.lowered_dir),
            "package_dir": str(self.package_dir),
            "package_manifest": str(self.package_manifest),
            "circuit_count": self.circuit_count,
            "shader_count": self.shader_count,
        }


def emit_compile_event(
    event_sink: CompileEventSink | None, event_type: str, **payload: Any
) -> None:
    if event_sink is not None:
        event_sink({"type": event_type, **payload})


def check_compile_cancelled(cancel_requested: CancelCheck | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ModelCompileCancelled("model compilation cancelled")


def relative_json_path(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target, base_dir)


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
