from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llmoop.cli import main
from llmoop.model_compiler import (
    ModelCompileCancelled,
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
    assert not list(tmp_path.glob(".*.llmoop-backup-*"))


def test_cli_discovery_streams_machine_readable_json_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_discoverable_source(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llmoop",
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
    assert all(event["schema"] == "llmoop.compiler_event.v1" for event in events)
