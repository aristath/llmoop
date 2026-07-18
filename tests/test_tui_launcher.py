from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from llmoop import cli


def test_no_argument_cli_launches_tui(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], Path | None]] = []
    monkeypatch.setattr(sys, "argv", ["llmoop"])
    monkeypatch.setenv("LLMOOP_TUI_BIN", "/tmp/llmoop-tui")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, cwd=None: calls.append((command, cwd))
        or SimpleNamespace(returncode=0),
    )

    cli.main()

    assert calls == [
        (["/tmp/llmoop-tui"], Path(cli.__file__).resolve().parent.parent)
    ]


def test_tui_launcher_prefers_explicit_binary(monkeypatch) -> None:
    monkeypatch.setenv("LLMOOP_TUI_BIN", "/opt/llmoop/bin/tui")

    command, workspace = cli.build_tui_command()

    assert command == ["/opt/llmoop/bin/tui"]
    assert workspace == Path(cli.__file__).resolve().parent.parent


def test_tui_launcher_uses_current_source_tree_without_shell(monkeypatch) -> None:
    monkeypatch.delenv("LLMOOP_TUI_BIN", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    workspace = Path(cli.__file__).resolve().parent.parent

    command, command_workspace = cli.build_tui_command()

    assert command == [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(workspace / "runtime-rs/Cargo.toml"),
        "--features",
        "vulkan,tokenizers,tui",
        "--bin",
        "llmoop-tui",
    ]
    assert command_workspace == workspace
