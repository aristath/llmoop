from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nerve.cli import main


def discoverable_source(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({"model_type": "synthetic"}))
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer.json").write_text("{}")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--discover-model", "{source}", "--chat"], "--chat is only supported with --run"),
        (
            ["--compile-model", "{source}", "--prompt", "ignored"],
            "--prompt is only supported with --run",
        ),
        (
            [
                "--run",
                "{source}/missing-package",
                "--prompt",
                "hello",
                "--compiled-model-dir",
                "{source}/ignored",
            ],
            "--compiled-model-dir is only supported with --compile-model",
        ),
    ],
)
def test_cli_rejects_options_owned_by_a_different_action_before_running_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    discoverable_source(tmp_path)
    rendered = [argument.replace("{source}", str(tmp_path)) for argument in arguments]
    monkeypatch.setattr(sys, "argv", ["nerve", *rendered])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
