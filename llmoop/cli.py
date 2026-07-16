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
        "--run-model",
        type=Path,
        metavar="LOWERED_DIR",
        help="run a compiled lowered model package with the Python circuit runtime",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="source model directory override for --run-model tokenization/tensor loading",
    )
    parser.add_argument(
        "--prompt",
        help="prompt text for --run-model",
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
        "--max-new-tokens",
        type=int,
        default=32,
        help="maximum new tokens to generate for --run-model",
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
        help="do not add tokenizer special tokens to the --run-model prompt",
    )
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="keep special tokens when decoding --run-model output",
    )
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="print only newly generated text for --run-model",
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

    if args.compile_model is not None and args.run_model is not None:
        parser.error("--compile-model and --run-model are mutually exclusive")
    if args.compile_model is None and args.run_model is None:
        parser.print_help()
        raise SystemExit(2)
    if args.run_model is not None:
        if args.prompt is None:
            parser.error("--prompt is required with --run-model")
        run_model(args)
        return

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
    model_dir = args.model_dir or Path(runtime.board.index["source"]["source_model_dir"])
    tokenizer = load_tokenizer(model_dir)
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
