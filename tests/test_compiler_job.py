from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import nerve.model_compiler as compiler_module
from nerve.cli import main
from nerve.compilation import ModelCompileCancelled, ModelCompileError
from nerve.model_compiler import (
    compile_model,
    discover_source_model,
    publish_staged_directories,
)


def write_discoverable_source(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "synthetic_decoder",
                "architectures": ["SyntheticForCausalLM"],
            }
        )
    )
    (root / "model.safetensors").write_bytes(b"not-read-during-discovery")
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"})
    )


def test_discovers_source_artifacts_without_model_family_checks(tmp_path: Path) -> None:
    write_discoverable_source(tmp_path)

    discovery = discover_source_model(tmp_path)

    assert discovery.model_type == "synthetic_decoder"
    assert discovery.architecture == ("SyntheticForCausalLM",)
    assert discovery.weight_files == (tmp_path / "model.safetensors",)
    assert discovery.has_chat_template
    assert discovery.to_json()["source_format"] == "safetensors"


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        ("config.json", "required"),
        ("model.safetensors", "no Safetensors weights"),
        ("tokenizer.json", "required tokenizer"),
    ],
)
def test_discovery_rejects_incomplete_sources_before_compilation(
    tmp_path: Path, remove: str, message: str
) -> None:
    write_discoverable_source(tmp_path)
    (tmp_path / remove).unlink()

    with pytest.raises(ModelCompileError, match=message):
        discover_source_model(tmp_path)


def test_discovery_rejects_a_non_directory_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ModelCompileError, match="does not exist"):
        discover_source_model(missing)


def test_cancelled_job_emits_terminal_event_before_writing_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_discoverable_source(source)
    events: list[dict[str, object]] = []

    with pytest.raises(ModelCompileCancelled):
        compile_model(
            source,
            transpiled_dir=tmp_path / "transpiled",
            lowered_dir=tmp_path / "lowered",
            package_dir=tmp_path / "package",
            event_sink=events.append,
            cancel_requested=lambda: True,
        )

    assert [event["type"] for event in events] == [
        "DiscoveryStarted",
        "SourceDiscovered",
        "Cancelled",
    ]
    assert not (tmp_path / "transpiled").exists()
    assert not (tmp_path / "lowered").exists()
    assert not (tmp_path / "package").exists()


def test_staged_directories_replace_existing_outputs_as_one_publication(
    tmp_path: Path,
) -> None:
    publications = []
    for name in ("transpiled", "lowered", "package"):
        staged = tmp_path / f".{name}.stage"
        destination = tmp_path / name
        staged.mkdir()
        destination.mkdir()
        (staged / "identity").write_text("new")
        (destination / "identity").write_text("old")
        publications.append((staged, destination))

    publish_staged_directories(tuple(publications), "test-token")

    for staged, destination in publications:
        assert not staged.exists()
        assert (destination / "identity").read_text() == "new"
    assert not list(tmp_path.glob(".*.nerve-backup-*"))


def test_publication_failure_restores_every_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publications: list[tuple[Path, Path]] = []
    for name in ("transpiled", "lowered", "package"):
        staged = tmp_path / f".{name}.stage"
        destination = tmp_path / name
        staged.mkdir()
        destination.mkdir()
        (staged / "identity").write_text("new")
        (destination / "identity").write_text("old")
        publications.append((staged, destination))

    real_replace = compiler_module.os.replace
    failed = False

    def fail_during_second_publication(source: Path, destination: Path) -> None:
        nonlocal failed
        if source == publications[1][0] and not failed:
            failed = True
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(compiler_module.os, "replace", fail_during_second_publication)

    with pytest.raises(OSError, match="injected publication failure"):
        publish_staged_directories(tuple(publications), "rollback-token")

    assert failed
    for _staged, destination in publications:
        assert (destination / "identity").read_text() == "old"
    assert not list(tmp_path.glob(".*.nerve-backup-*"))


def test_failed_compile_preserves_public_outputs_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_discoverable_source(source)
    destinations = tuple(tmp_path / name for name in ("transpiled", "lowered", "package"))
    for destination in destinations:
        destination.mkdir()
        (destination / "identity").write_text("published")
    events: list[dict[str, object]] = []

    def fail_after_writing_staged_artifacts(
        _model_dir: Path, **arguments: object
    ) -> None:
        for name in ("transpiled_dir", "lowered_dir", "package_dir"):
            staged = arguments[name]
            assert isinstance(staged, Path)
            staged.mkdir(parents=True)
            (staged / "partial").write_text("incomplete")
        raise RuntimeError("injected compiler failure")

    monkeypatch.setattr(
        compiler_module, "compile_model_package", fail_after_writing_staged_artifacts
    )

    with pytest.raises(RuntimeError, match="injected compiler failure"):
        compile_model(
            source,
            transpiled_dir=destinations[0],
            lowered_dir=destinations[1],
            package_dir=destinations[2],
            event_sink=events.append,
        )

    for destination in destinations:
        assert (destination / "identity").read_text() == "published"
    assert not list(tmp_path.glob(".*.nerve-stage-*"))
    assert events[-1] == {
        "type": "Failed",
        "diagnostics": [
            {"kind": "RuntimeError", "message": "injected compiler failure"}
        ],
    }


def test_cli_discovery_streams_machine_readable_json_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_discoverable_source(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nerve",
            "--discover-model",
            str(tmp_path),
            "--compiler-events-jsonl",
        ],
    )

    main()

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == [
        "DiscoveryStarted",
        "SourceDiscovered",
        "Completed",
    ]
    assert [event["sequence"] for event in events] == [0, 1, 2]
    assert all(event["schema"] == "nerve.compiler_event.v1" for event in events)
