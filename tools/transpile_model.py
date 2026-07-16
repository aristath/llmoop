#!/usr/bin/env python3
"""Transpile a local checkpoint into llmoop model-graph artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from llmoop.model_transpiler import transpile_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-clean", action="store_true", help="do not delete an existing output directory first")
    args = parser.parse_args()

    structure = transpile_model(args.model_dir, args.output_dir, clean=not args.no_clean)
    print(
        "transpiled "
        f"{args.model_dir} -> {args.output_dir} "
        f"({structure.num_hidden_layers} layers, "
        f"{sum(1 for layer in structure.layers if layer.operator_type == 'conv')} short-conv, "
        f"{sum(1 for layer in structure.layers if layer.operator_type == 'full_attention')} attention)"
    )


if __name__ == "__main__":
    main()
