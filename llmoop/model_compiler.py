from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


Json = dict[str, Any]

PACKAGE_SCHEMA = "llmoop.vulkan_resident_model_package.v1"


class ModelCompileError(RuntimeError):
    pass


class ModelCompileCancelled(ModelCompileError):
    pass


CompileEventSink = Callable[[Json], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class SourceModelDiscovery:
    model_dir: Path
    model_type: str
    architecture: tuple[str, ...]
    config_path: Path
    weight_files: tuple[Path, ...]
    tokenizer_files: tuple[str, ...]
    has_chat_template: bool

    def to_json(self) -> Json:
        return {
            "model_dir": str(self.model_dir),
            "source_format": "safetensors",
            "model_type": self.model_type,
            "architectures": list(self.architecture),
            "config_path": str(self.config_path),
            "weight_files": [str(path) for path in self.weight_files],
            "weight_file_count": len(self.weight_files),
            "tokenizer_files": list(self.tokenizer_files),
            "has_chat_template": self.has_chat_template,
        }


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
    event_sink: CompileEventSink | None = None,
    cancel_requested: CancelCheck | None = None,
) -> CompiledModelReport:
    model_dir = model_dir.expanduser()
    from llmoop.model_package import compiled_model_slug, compile_model_package

    emit_compile_event(event_sink, "DiscoveryStarted", model_dir=str(model_dir))
    try:
        discovery = discover_source_model(model_dir)
        emit_compile_event(event_sink, "SourceDiscovered", source=discovery.to_json())
        check_compile_cancelled(cancel_requested)
        emit_compile_event(event_sink, "ValidationStarted", model_dir=str(model_dir))

        slug = compiled_model_slug(model_dir)
        final_transpiled = transpiled_dir or Path("transpiled") / slug
        final_lowered = lowered_dir or Path("lowered") / slug
        final_package = package_dir or Path("packages") / slug
        token = uuid4().hex
        staged_transpiled = staging_path(final_transpiled, token)
        staged_lowered = staging_path(final_lowered, token)
        staged_package = staging_path(final_package, token)
        staged = (staged_transpiled, staged_lowered, staged_package)
        for path in staged:
            remove_path(path)

        try:
            staged_report = compile_model_package(
                model_dir,
                transpiled_dir=staged_transpiled,
                lowered_dir=staged_lowered,
                package_dir=staged_package,
                clean=True,
                shader_source_dir=shader_source_dir,
                event_sink=event_sink,
                cancel_requested=cancel_requested,
            )
            check_compile_cancelled(cancel_requested)
            publish_staged_directories(
                (
                    (staged_transpiled, final_transpiled),
                    (staged_lowered, final_lowered),
                    (staged_package, final_package),
                ),
                token,
            )
        except BaseException:
            for path in staged:
                remove_path(path)
            raise

        report = CompiledModelReport(
            model_dir=model_dir,
            transpiled_dir=final_transpiled,
            lowered_dir=final_lowered,
            package_dir=final_package,
            package_manifest=final_package / staged_report.package_manifest.name,
            model_type=staged_report.model_type,
            circuit_count=staged_report.circuit_count,
            shader_count=staged_report.shader_count,
        )
        emit_compile_event(event_sink, "Completed", package=report.to_json())
        return report
    except ModelCompileCancelled:
        emit_compile_event(event_sink, "Cancelled", model_dir=str(model_dir))
        raise
    except BaseException as error:
        emit_compile_event(
            event_sink,
            "Failed",
            diagnostics=[{"kind": type(error).__name__, "message": str(error)}],
        )
        raise


def discover_source_model(model_dir: Path) -> SourceModelDiscovery:
    model_dir = model_dir.expanduser()
    if not model_dir.is_dir():
        raise ModelCompileError(f"source model directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ModelCompileError(f"source model does not contain required {config_path}")
    config = read_json(config_path)
    weight_files = tuple(sorted(model_dir.glob("*.safetensors")))
    if not weight_files:
        raise ModelCompileError(f"source model contains no Safetensors weights: {model_dir}")
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise ModelCompileError(
            f"source model does not contain required tokenizer file {tokenizer_path}"
        )
    tokenizer_candidates = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "merges.txt",
    )
    tokenizer_files = tuple(
        name for name in tokenizer_candidates if (model_dir / name).is_file()
    )
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    tokenizer_config = (
        read_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    )
    return SourceModelDiscovery(
        model_dir=model_dir,
        model_type=str(config.get("model_type") or "unknown"),
        architecture=tuple(str(value) for value in (config.get("architectures") or ())),
        config_path=config_path,
        weight_files=weight_files,
        tokenizer_files=tokenizer_files,
        has_chat_template=(model_dir / "chat_template.jinja").is_file()
        or isinstance(tokenizer_config.get("chat_template"), str),
    )


def emit_compile_event(
    event_sink: CompileEventSink | None, event_type: str, **payload: Any
) -> None:
    if event_sink is not None:
        event_sink({"type": event_type, **payload})


def check_compile_cancelled(cancel_requested: CancelCheck | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ModelCompileCancelled("model compilation cancelled")


def staging_path(destination: Path, token: str) -> Path:
    return destination.with_name(f".{destination.name}.llmoop-stage-{token}")


def backup_path(destination: Path, token: str) -> Path:
    return destination.with_name(f".{destination.name}.llmoop-backup-{token}")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish_staged_directories(
    publications: tuple[tuple[Path, Path], ...], token: str
) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _staged, destination in publications:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = backup_path(destination, token)
            remove_path(backup)
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup)
                backups.append((backup, destination))
        for staged, destination in publications:
            os.replace(staged, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            remove_path(destination)
        for backup, destination in reversed(backups):
            if backup.exists() or backup.is_symlink():
                os.replace(backup, destination)
        raise
    for backup, _destination in backups:
        remove_path(backup)


def sanitize_slug(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()


def relative_json_path(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target, base_dir)


def read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
