from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from llmoop.model_compiler import compile_model


RUNTIME_PACKAGE_MANIFEST = "vulkan_resident_greedy_package.json"


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmoop")
    parser.add_argument(
        "--compile-model",
        type=Path,
        metavar="MODEL_DIR",
        help="compile a source model directory into llmoop engine artifacts",
    )
    parser.add_argument(
        "--run",
        type=Path,
        metavar="LOWERED_DIR_OR_PACKAGE",
        help="run a compiled model package with the Rust/Vulkan runtime engine",
    )
    parser.add_argument(
        "--run-model",
        type=Path,
        metavar="LOWERED_DIR",
        help="run a compiled lowered model package with the Python circuit runtime",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="source model directory for --run-model debug/oracle execution",
    )
    parser.add_argument(
        "--prompt",
        help="prompt text for --run/--run-model",
    )
    parser.add_argument(
        "--runtime-bin",
        type=Path,
        help="path to a built llmoop-runtime binary; defaults to cargo run from a source checkout",
    )
    parser.add_argument(
        "--transpiled-dir",
        type=Path,
        help="directory for model graph/tensor transpilation artifacts",
    )
    parser.add_argument(
        "--lowered-dir",
        type=Path,
        help="directory for lowered circuit/package artifacts",
    )
    parser.add_argument(
        "--shader-source-dir",
        type=Path,
        default=Path("runtime-rs/shaders"),
        help="directory containing backend shader templates",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="resident dynamic-state activation capacity; compile default is 4, runtime default is auto",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="maximum new tokens to generate for --run/--run-model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="use explicit-random temperature sampling for --run-model instead of greedy sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="restrict temperature sampling to the top K logits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for --run-model temperature sampling",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="do not stop --run-model generation when EOS is produced",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="do not add tokenizer special tokens to the runtime prompt",
    )
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="keep special tokens when decoding runtime output",
    )
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="print only newly generated text for --run/--run-model",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="do not delete an existing transpiled model directory before compiling",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable report",
    )
    args = parser.parse_args()

    selected_actions = [
        args.compile_model is not None,
        args.run is not None,
        args.run_model is not None,
    ]
    if sum(selected_actions) > 1:
        parser.error("--compile-model, --run, and --run-model are mutually exclusive")
    if not any(selected_actions):
        parser.print_help()
        raise SystemExit(2)
    if args.run is not None:
        if args.prompt is None:
            parser.error("--prompt is required with --run")
        if args.model_dir is not None:
            parser.error("--model-dir is only supported by --run-model")
        if args.temperature is not None:
            parser.error("--temperature is only supported by --run-model")
        if args.top_k is not None:
            parser.error("--top-k is only supported by --run-model")
        if args.seed != 0:
            parser.error("--seed is only supported by --run-model")
        if args.ignore_eos:
            parser.error("--ignore-eos is only supported by --run-model")
        run_engine(args)
        return
    if args.run_model is not None:
        if args.prompt is None:
            parser.error("--prompt is required with --run-model")
        if args.model_dir is None:
            parser.error("--model-dir is required with --run-model")
        run_model(args)
        return

    report = compile_model(
        args.compile_model,
        transpiled_dir=args.transpiled_dir,
        lowered_dir=args.lowered_dir,
        clean=not args.no_clean,
        shader_source_dir=args.shader_source_dir,
        default_dynamic_state_capacity_activations=args.capacity or 4,
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(f"compiled {report.model_dir}")
        print(f"  model_type: {report.model_type}")
        print(f"  transpiled: {report.transpiled_dir}")
        print(f"  lowered:    {report.lowered_dir}")
        print(f"  package:    {report.package_manifest}")
        print(f"  circuits:   {report.circuit_count}")
        print(f"  shaders:    {report.shader_count}")


def run_engine(args: argparse.Namespace) -> None:
    package_manifest = resolve_runtime_package_manifest(args.run)
    command = build_runtime_command(args, package_manifest)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def resolve_runtime_package_manifest(path: Path) -> Path:
    if path.is_dir():
        package_manifest = path / RUNTIME_PACKAGE_MANIFEST
        if not package_manifest.is_file():
            raise SystemExit(f"{path} does not contain {RUNTIME_PACKAGE_MANIFEST}")
        return package_manifest
    if path.is_file():
        return path
    raise SystemExit(f"compiled model package path does not exist: {path}")


def build_runtime_command(args: argparse.Namespace, package_manifest: Path) -> list[str]:
    runtime_args = [
        "--package",
        str(package_manifest),
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.capacity is not None:
        runtime_args.extend(["--capacity", str(args.capacity)])
    if args.no_special_tokens:
        runtime_args.append("--no-special-tokens")
    if args.keep_special_tokens:
        runtime_args.append("--keep-special-tokens")
    if args.generated_only:
        runtime_args.append("--generated-only")
    if args.json:
        runtime_args.append("--json")

    runtime_bin = args.runtime_bin or runtime_bin_from_env()
    if runtime_bin is not None:
        return [str(runtime_bin), *runtime_args]

    repo_root = Path(__file__).resolve().parents[1]
    cargo_manifest = repo_root / "runtime-rs" / "Cargo.toml"
    if cargo_manifest.is_file():
        return [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(cargo_manifest),
            "--features",
            "vulkan tokenizers",
            "--bin",
            "llmoop-runtime",
            "--",
            *runtime_args,
        ]

    for candidate in (
        repo_root / "runtime-rs" / "target" / "debug" / "llmoop-runtime",
        repo_root / "runtime-rs" / "target" / "release" / "llmoop-runtime",
    ):
        if candidate.is_file():
            return [str(candidate), *runtime_args]

    raise SystemExit(
        "could not find llmoop-runtime; pass --runtime-bin or run from a source checkout with runtime-rs"
    )


def runtime_bin_from_env() -> Path | None:
    raw = os.environ.get("LLMOOP_RUNTIME_BIN")
    return Path(raw).expanduser() if raw else None


def run_model(args: argparse.Namespace) -> None:
    import torch

    from llmoop.circuit_model_runtime import CircuitModelRuntime
    from llmoop.samplers import TemperatureSamplerPedal
    from llmoop.text_generation import generate_text, load_tokenizer

    runtime = CircuitModelRuntime.from_dirs(
        circuit_dir=args.run_model,
        model_dir=args.model_dir,
        torch=torch,
    )
    tokenizer = load_tokenizer(args.model_dir)
    eos_token_id = None if args.ignore_eos else int(runtime.config["eos_token_id"])
    sampler = None
    if args.temperature is not None:
        sampler = TemperatureSamplerPedal(temperature=args.temperature, top_k=args.top_k)

    with torch.no_grad():
        run = generate_text(
            runtime=runtime,
            tokenizer=tokenizer,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            add_special_tokens=not args.no_special_tokens,
            skip_special_tokens=not args.keep_special_tokens,
            sampler=sampler,
            random_seed=args.seed,
        )

    if args.json:
        print(json.dumps(run.to_json(), indent=2))
    elif args.generated_only:
        print(run.generated_text, end="" if run.generated_text.endswith("\n") else "\n")
    else:
        print(run.output_text, end="" if run.output_text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
