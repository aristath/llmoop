#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Python circuit runtime with the Rust/Vulkan runtime for one compiled package."
    )
    parser.add_argument("--lowered-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-text", "--prompt", dest="prompt_text", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--runtime-bin", type=Path, default=None)
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--keep-special-tokens", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    python_run = run_python_circuit_runtime(args)
    rust_run = run_rust_vulkan_runtime(args)

    report = {
        "prompt_text": args.prompt_text,
        "max_new_tokens": args.max_new_tokens,
        "python_circuit_runtime": python_run,
        "rust_vulkan_runtime": rust_run,
        "comparison": compare_runs(python_run["json"], rust_run["json"]),
    }
    if args.summary:
        report = summary_report(report)

    print(json.dumps(report, indent=2))

    if not all(report["comparison"].values()):
        raise SystemExit("runtime conformance check failed")


def run_python_circuit_runtime(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "llmoop",
        "--run-model",
        str(args.lowered_dir),
        "--model-dir",
        str(args.model_dir),
        "--prompt",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--ignore-eos",
        "--json",
    ]
    append_common_flags(command, args)
    return run_json_command(command)


def run_rust_vulkan_runtime(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "llmoop",
        "--run",
        str(args.lowered_dir),
        "--prompt",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--json",
    ]
    append_common_flags(command, args)
    if args.runtime_bin is not None:
        command.extend(["--runtime-bin", str(args.runtime_bin)])
    return run_json_command(command)


def append_common_flags(command: list[str], args: argparse.Namespace) -> None:
    if args.no_special_tokens:
        command.append("--no-special-tokens")
    if args.keep_special_tokens:
        command.append("--keep-special-tokens")


def run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(
            "command did not produce JSON on stdout:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        ) from error
    return {
        "command": command,
        "json": payload,
        "stderr": completed.stderr,
    }


def compare_runs(python_payload: dict[str, Any], rust_payload: dict[str, Any]) -> dict[str, bool]:
    python_output_ids = python_payload["output_ids"]
    rust_output_ids = rust_payload["prompt_ids"] + rust_payload["generated_ids"]

    return {
        "prompt_ids_match": python_payload["prompt_ids"] == rust_payload["prompt_ids"],
        "generated_ids_match": python_payload["generated_ids"] == rust_payload["generated_ids"],
        "output_ids_match": python_output_ids == rust_output_ids,
        "generated_text_match": python_payload["generated_text"] == rust_payload["generated_text"],
        "output_text_match": python_payload["output_text"] == rust_payload["output_text"],
        "generated_count_match": python_payload["generated_count"]
        == len(rust_payload["generated_ids"]),
    }


def summary_report(report: dict[str, Any]) -> dict[str, Any]:
    python_payload = report["python_circuit_runtime"]["json"]
    rust_payload = report["rust_vulkan_runtime"]["json"]
    return {
        "prompt_text": report["prompt_text"],
        "max_new_tokens": report["max_new_tokens"],
        "comparison": report["comparison"],
        "python_generated_ids": python_payload["generated_ids"],
        "rust_generated_ids": rust_payload["generated_ids"],
        "python_output_text": python_payload["output_text"],
        "rust_output_text": rust_payload["output_text"],
        "rust_engine": {
            "device_name": rust_payload["device_name"],
            "device_id": rust_payload["device_id"],
            "pedal_count": rust_payload["pedal_count"],
            "resident_capacity_activations": rust_payload["resident_capacity_activations"],
            "runtime_cycles": rust_payload["runtime_cycles"],
            "scheduler_turns": rust_payload["scheduler_turns"],
        },
    }


if __name__ == "__main__":
    main()
