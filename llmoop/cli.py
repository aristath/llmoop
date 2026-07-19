from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from llmoop.compilation import ModelCompileCancelled, ModelCompileError
from llmoop.model_compiler import compile_model, discover_source_model


RUNTIME_PACKAGE_MANIFEST = "vulkan_resident_package.json"


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmoop")
    parser.add_argument(
        "--compile-model",
        type=Path,
        metavar="MODEL_DIR",
        help="compile a source model directory into llmoop engine artifacts",
    )
    parser.add_argument(
        "--discover-model",
        type=Path,
        metavar="MODEL_DIR",
        help="discover and validate Safetensors source-model artifacts without compiling",
    )
    parser.add_argument(
        "--run",
        type=Path,
        metavar="PACKAGE_DIR_OR_MANIFEST",
        help="run a compiled model package with the Rust/Vulkan runtime engine",
    )
    parser.add_argument(
        "--prompt",
        help="prompt text for --run",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="start an interactive resident text session for --run",
    )
    parser.add_argument(
        "--inspect-runtime",
        action="store_true",
        help="preview UI-ready package, patch, placement, device, and route facts for --run",
    )
    parser.add_argument(
        "--inspect-device-slice",
        metavar="DEVICE_ID",
        help="mount and summarize only the compiled package pedals assigned to DEVICE_ID for --run",
    )
    parser.add_argument(
        "--inspect-package",
        action="store_true",
        help="summarize the compiled source pedal kit and available runtime devices for --run",
    )
    parser.add_argument(
        "--inspect-patch",
        action="store_true",
        help="preview the effective runtime patch for --run without mounting devices",
    )
    parser.add_argument(
        "--inspect-placement",
        action="store_true",
        help="mount and summarize every logical device slice in a compiled package for --run",
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
        help="directory for lowered circuit artifacts",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="directory for runtime package artifacts",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="default logical device for the runtime pedalboard patch",
    )
    parser.add_argument(
        "--place-pedal",
        action="append",
        default=[],
        metavar="PEDAL=DEVICE",
        help="assign one runtime pedal instance to a logical device in the runtime patch; may be repeated",
    )
    parser.add_argument(
        "--bind-device",
        action="append",
        default=[],
        metavar="DEVICE=TARGET",
        help="bind one logical runtime device to a target such as vulkan:5 or cpu0; may be repeated",
    )
    parser.add_argument(
        "--duplicate-after",
        action="append",
        default=[],
        metavar="AFTER=NEW",
        help="duplicate runtime pedal instance AFTER with id NEW; may be repeated",
    )
    parser.add_argument(
        "--chain",
        default=None,
        metavar="ITEM[,ITEM...]",
        help="runtime source chain for --run; ITEM is SOURCE or INSTANCE=SOURCE",
    )
    parser.add_argument(
        "--shader-source-dir",
        type=Path,
        default=Path("runtime-rs/shaders"),
        help="directory containing backend shader templates",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="runtime transient-state context window; defaults to an automatic size",
    )
    parser.add_argument(
        "--vulkan-device-index",
        type=int,
        default=None,
        help="Vulkan physical device index to use for this --run process",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=65_536,
        help="generation stop condition for --run; independent of context allocation (default: 65536)",
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
        help="print only newly generated text for --run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print human-readable runtime timing and top-pedal summaries for --run",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        metavar="N",
        help="run N fresh prompt trials and report aggregate benchmark stats for --run",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable report",
    )
    parser.add_argument(
        "--compiler-events-jsonl",
        action="store_true",
        help="stream structured compiler job events as JSON Lines",
    )
    args = parser.parse_args()

    selected_actions = [
        args.compile_model is not None,
        args.discover_model is not None,
        args.run is not None,
    ]
    if sum(selected_actions) > 1:
        parser.error("--compile-model, --discover-model, and --run are mutually exclusive")
    if not any(selected_actions):
        if len(sys.argv) == 1:
            run_tui()
            return
        parser.error("choose an action, or run llmoop without arguments to open the TUI")
    if args.context_size is not None and args.context_size < 1:
        parser.error("--context-size must be at least 1")
    if args.compiler_events_jsonl and args.compile_model is None and args.discover_model is None:
        parser.error("--compiler-events-jsonl requires --compile-model or --discover-model")
    if args.compiler_events_jsonl and args.json:
        parser.error("--compiler-events-jsonl and --json are mutually exclusive")
    if args.discover_model is not None:
        reporter = JsonLineCompileReporter() if args.compiler_events_jsonl else None
        if reporter is not None:
            reporter({"type": "DiscoveryStarted", "model_dir": str(args.discover_model)})
        try:
            discovery = discover_source_model(args.discover_model)
        except ModelCompileError as error:
            if reporter is not None:
                reporter(
                    {
                        "type": "Failed",
                        "diagnostics": [
                            {"kind": type(error).__name__, "message": str(error)}
                        ],
                    }
                )
                raise SystemExit(1) from None
            raise SystemExit(str(error)) from None
        if reporter is not None:
            reporter({"type": "SourceDiscovered", "source": discovery.to_json()})
            reporter({"type": "Completed", "discovery": discovery.to_json()})
        elif args.json:
            print(json.dumps({"ok": True, **discovery.to_json()}, indent=2))
        else:
            print(f"discovered {discovery.model_dir}")
            print(f"  model_type: {discovery.model_type}")
            print(f"  weight_files: {len(discovery.weight_files)}")
            print(f"  tokenizer: {', '.join(discovery.tokenizer_files)}")
            print(f"  chat_template: {discovery.has_chat_template}")
        return
    if args.compile_model is None:
        if args.inspect_runtime and args.run is None:
            parser.error("--inspect-runtime is only supported with --run")
        if args.inspect_package and args.run is None:
            parser.error("--inspect-package is only supported with --run")
        if args.inspect_patch and args.run is None:
            parser.error("--inspect-patch is only supported with --run")
        if args.inspect_placement and args.run is None:
            parser.error("--inspect-placement is only supported with --run")
    elif args.inspect_device_slice is not None:
        parser.error("--inspect-device-slice is only supported with --run")
    elif args.inspect_runtime:
        parser.error("--inspect-runtime is only supported with --run")
    elif args.inspect_package:
        parser.error("--inspect-package is only supported with --run")
    elif args.inspect_patch:
        parser.error("--inspect-patch is only supported with --run")
    elif args.inspect_placement:
        parser.error("--inspect-placement is only supported with --run")
    elif args.chat:
        parser.error("--chat is only supported with --run")
    elif args.device is not None:
        parser.error("--device is only supported with --run")
    elif args.place_pedal:
        parser.error("--place-pedal is only supported with --run")
    elif args.bind_device:
        parser.error("--bind-device is only supported with --run")
    elif args.duplicate_after:
        parser.error("--duplicate-after is only supported with --run")
    elif args.chain is not None:
        parser.error("--chain is only supported with --run")
    elif args.vulkan_device_index is not None:
        parser.error("--vulkan-device-index is only supported with --run")
    elif args.context_size is not None:
        parser.error("--context-size is only supported with --run")
    elif args.profile:
        parser.error("--profile is only supported with --run")
    elif args.profile_runs != 1:
        parser.error("--profile-runs is only supported with --run")
    if args.run is not None:
        inspect_mode_count = sum(
            [
                args.inspect_device_slice is not None,
                args.inspect_runtime,
                args.inspect_package,
                args.inspect_patch,
                args.inspect_placement,
            ]
        )
        if inspect_mode_count > 1:
            parser.error(
                "--inspect-runtime, --inspect-package, --inspect-patch, --inspect-device-slice, and --inspect-placement are mutually exclusive"
            )
        if args.chat and inspect_mode_count > 0:
            parser.error("--chat cannot be combined with inspect modes")
        if inspect_mode_count == 0 and args.prompt is None and not args.chat:
            parser.error("--prompt is required with --run")
        if inspect_mode_count > 0 and args.profile_runs != 1:
            parser.error("--profile-runs is only supported for --run prompt execution")
        if args.chat and args.profile:
            parser.error("--profile is not supported with --chat")
        if args.chat and args.profile_runs != 1:
            parser.error("--profile-runs is not supported with --chat")
        if args.chat and args.json:
            parser.error("--json is not supported with --chat yet")
        if args.vulkan_device_index is not None and args.vulkan_device_index < 0:
            parser.error("--vulkan-device-index must be non-negative")
        if args.profile_runs < 1:
            parser.error("--profile-runs must be at least 1")
        run_engine(args)
        return
    reporter = JsonLineCompileReporter() if args.compiler_events_jsonl else None
    cancel_requested = False

    def request_cancel(_signum: int, _frame: object) -> None:
        nonlocal cancel_requested
        cancel_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_cancel)
    signal.signal(signal.SIGTERM, request_cancel)
    try:
        report = compile_model(
            args.compile_model,
            transpiled_dir=args.transpiled_dir,
            lowered_dir=args.lowered_dir,
            package_dir=args.package_dir,
            shader_source_dir=args.shader_source_dir,
            event_sink=reporter,
            cancel_requested=lambda: cancel_requested,
        )
    except ModelCompileCancelled:
        raise SystemExit(130) from None
    except ModelCompileError as error:
        if reporter is not None:
            raise SystemExit(1) from None
        raise SystemExit(str(error)) from None
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    if reporter is not None:
        return
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(f"compiled {report.model_dir}")
        print(f"  model_type: {report.model_type}")
        print(f"  transpiled: {report.transpiled_dir}")
        print(f"  lowered:    {report.lowered_dir}")
        print(f"  package_dir: {report.package_dir}")
        print(f"  package:    {report.package_manifest}")
        print(f"  circuits:   {report.circuit_count}")
        print(f"  shaders:    {report.shader_count}")


class JsonLineCompileReporter:
    def __init__(self) -> None:
        self.sequence = 0

    def __call__(self, event: dict[str, object]) -> None:
        emit_jsonl_event(self.sequence, event)
        self.sequence += 1


def run_tui() -> None:
    command, working_directory = build_tui_command()
    completed = subprocess.run(command, cwd=working_directory)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_tui_command() -> tuple[list[str], Path]:
    workspace = Path(__file__).resolve().parent.parent
    configured = os.environ.get("LLMOOP_TUI_BIN")
    if configured:
        return [configured], workspace
    manifest = workspace / "runtime-rs" / "Cargo.toml"
    if manifest.is_file():
        return [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(manifest),
            "--features",
            "vulkan,tokenizers,tui",
            "--bin",
            "llmoop-tui",
        ], workspace
    installed = shutil.which("llmoop-tui")
    if installed:
        return [installed], workspace
    raise SystemExit(
        "llmoop-tui is not installed; set LLMOOP_TUI_BIN to the executable path"
    )


def emit_jsonl_event(sequence: int, event: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "schema": "llmoop.compiler_event.v1",
                "sequence": sequence,
                **event,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


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
    ]
    if args.inspect_runtime:
        runtime_args.append("--inspect-runtime")
    elif args.inspect_placement:
        runtime_args.append("--inspect-placement")
    elif args.inspect_package:
        runtime_args.append("--inspect-package")
    elif args.inspect_patch:
        runtime_args.append("--inspect-patch")
    elif args.inspect_device_slice is not None:
        runtime_args.extend(["--inspect-device-slice", args.inspect_device_slice])
    else:
        if args.chat:
            runtime_args.append("--chat")
            if args.prompt is not None:
                runtime_args.extend(["--prompt", args.prompt])
        else:
            runtime_args.extend(["--prompt", args.prompt])
        runtime_args.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.device is not None:
        runtime_args.extend(["--device", args.device])
    for raw_placement in args.place_pedal:
        runtime_args.extend(["--place-pedal", raw_placement])
    for raw_binding in args.bind_device:
        runtime_args.extend(["--bind-device", raw_binding])
    for raw_duplicate in args.duplicate_after:
        runtime_args.extend(["--duplicate-after", raw_duplicate])
    if args.chain is not None:
        runtime_args.extend(["--chain", args.chain])
    if args.context_size is not None:
        runtime_args.extend(["--context-size", str(args.context_size)])
    if args.vulkan_device_index is not None:
        runtime_args.extend(["--vulkan-device-index", str(args.vulkan_device_index)])
    if args.no_special_tokens:
        runtime_args.append("--no-special-tokens")
    if args.keep_special_tokens:
        runtime_args.append("--keep-special-tokens")
    if args.generated_only:
        runtime_args.append("--generated-only")
    if args.profile:
        runtime_args.append("--profile")
    if args.profile_runs != 1:
        runtime_args.extend(["--profile-runs", str(args.profile_runs)])
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


if __name__ == "__main__":
    main()
