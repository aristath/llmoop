from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmoop.model_compiler import compile_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmoop")
    parser.add_argument(
        "--compile-model",
        type=Path,
        metavar="MODEL_DIR",
        help="compile a source model directory into llmoop engine artifacts",
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
        default=4,
        help="default resident dynamic-state activation capacity recorded in the package manifest",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="do not delete an existing transpiled model directory before compiling",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable compile report",
    )
    args = parser.parse_args()

    if args.compile_model is None:
        parser.print_help()
        raise SystemExit(2)

    report = compile_model(
        args.compile_model,
        transpiled_dir=args.transpiled_dir,
        lowered_dir=args.lowered_dir,
        clean=not args.no_clean,
        shader_source_dir=args.shader_source_dir,
        default_dynamic_state_capacity_activations=args.capacity,
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


if __name__ == "__main__":
    main()
