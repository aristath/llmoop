from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Json = dict[str, Any]

PACKAGE_SCHEMA = "llmoop.vulkan_resident_greedy_model_package.v1"


class ModelCompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompiledModelReport:
    model_dir: Path
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
            "transpiled_dir": str(self.transpiled_dir),
            "lowered_dir": str(self.lowered_dir),
            "package_dir": str(self.package_dir),
            "package_manifest": str(self.package_manifest),
            "circuit_count": self.circuit_count,
            "shader_count": self.shader_count,
        }


def compile_model(
    model_dir: Path,
    *,
    transpiled_dir: Path | None = None,
    lowered_dir: Path | None = None,
    package_dir: Path | None = None,
    clean: bool = True,
    shader_source_dir: Path = Path("runtime-rs/shaders"),
) -> CompiledModelReport:
    model_dir = model_dir.expanduser()
    from llmoop.model_package import compile_model_package

    return compile_model_package(
        model_dir,
        transpiled_dir=transpiled_dir,
        lowered_dir=lowered_dir,
        package_dir=package_dir,
        clean=clean,
        shader_source_dir=shader_source_dir,
    )


def sanitize_slug(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()


def relative_json_path(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target, base_dir)


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
