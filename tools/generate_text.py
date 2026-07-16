#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.samplers import TemperatureSamplerPedal
from llmoop.text_generation import generate_text, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the circuit pedalboard runtime.")
    parser.add_argument("prompt", help="Prompt text to feed into the input transducer.")
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None, help="Use stochastic temperature sampling instead of greedy.")
    parser.add_argument("--top-k", type=int, default=None, help="Restrict stochastic sampling to the top K logits.")
    parser.add_argument("--seed", type=int, default=0, help="Explicit random-source seed for stochastic sampling.")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true", help="Do not let the tokenizer add BOS/special tokens.")
    parser.add_argument("--keep-special-tokens", action="store_true", help="Keep special tokens in decoded output.")
    parser.add_argument("--generated-only", action="store_true", help="Print only newly generated text.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report instead of plain text.")
    args = parser.parse_args()

    import torch

    runtime = CircuitModelRuntime.from_dirs(circuit_dir=args.circuit_dir, model_dir=args.model_dir, torch=torch)
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
