from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from nerve import cli


def test_no_argument_cli_launches_tui(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], Path | None]] = []
    monkeypatch.setattr(sys, "argv", ["nerve"])
    monkeypatch.setenv("NERVE_TUI_BIN", "/tmp/nerve-tui")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, cwd=None: calls.append((command, cwd))
        or SimpleNamespace(returncode=0),
    )

    cli.main()

    assert calls == [
        (["/tmp/nerve-tui"], Path(cli.__file__).resolve().parent.parent)
    ]


def test_tui_launcher_prefers_explicit_binary(monkeypatch) -> None:
    monkeypatch.setenv("NERVE_TUI_BIN", "/opt/nerve/bin/tui")

    command, workspace = cli.build_tui_command()

    assert command == ["/opt/nerve/bin/tui"]
    assert workspace == Path(cli.__file__).resolve().parent.parent


def test_tui_launcher_uses_current_source_tree_without_shell(monkeypatch) -> None:
    monkeypatch.delenv("NERVE_TUI_BIN", raising=False)
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
        "nerve-tui",
    ]
    assert command_workspace == workspace
