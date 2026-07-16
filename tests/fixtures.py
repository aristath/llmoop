from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from llmoop.model_compiler import compile_model


TEST_MODEL_ENV = "LLMOOP_TEST_MODEL_DIR"


@dataclass(frozen=True)
class CompiledModelFixture:
    source_model_dir: Path
    transpiled_dir: Path
    lowered_dir: Path
    package_dir: Path
    package_manifest: Path


def source_model_dir_or_skip() -> Path:
    raw = os.environ.get(TEST_MODEL_ENV)
    if not raw:
        raise unittest.SkipTest(f"set {TEST_MODEL_ENV} to run source-model integration tests")
    model_dir = Path(raw).expanduser()
    if not (model_dir / "config.json").exists():
        raise unittest.SkipTest(f"{TEST_MODEL_ENV} does not point to a checkpoint with config.json")
    return model_dir


@lru_cache(maxsize=1)
def compiled_model_or_skip() -> CompiledModelFixture:
    model_dir = source_model_dir_or_skip()
    root = Path(tempfile.mkdtemp(prefix="llmoop_compiled_model_"))
    report = compile_model(
        model_dir,
        transpiled_dir=root / "transpiled",
        lowered_dir=root / "lowered",
        package_dir=root / "package",
        clean=True,
    )
    return CompiledModelFixture(
        source_model_dir=model_dir,
        transpiled_dir=report.transpiled_dir,
        lowered_dir=report.lowered_dir,
        package_dir=report.package_dir,
        package_manifest=report.package_manifest,
    )
