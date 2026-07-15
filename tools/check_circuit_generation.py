#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.source_oracle import _oracle_imports
from llmoop.text_generation import generate_text, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check greedy circuit generation against the source model oracle.")
    parser.add_argument("--circuit-dir", type=Path, default=Path("lowered/lfm2_5_230m"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/aristath/models/lfm2.5/230m"))
    parser.add_argument("--prompt-ids", type=str, default=None)
    parser.add_argument("--prompt-text", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--keep-special-tokens", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.prompt_ids is not None and args.prompt_text is not None:
        parser.error("--prompt-ids and --prompt-text are mutually exclusive")

    torch, auto_model, dynamic_cache = _oracle_imports()
    runtime = CircuitModelRuntime.from_dirs(circuit_dir=args.circuit_dir, model_dir=args.model_dir, torch=torch)
    source = auto_model.from_pretrained(args.model_dir, dtype=torch.float32)
    source.eval()

    tokenizer = load_tokenizer(args.model_dir) if args.prompt_text is not None else None
    if args.prompt_text is not None:
        prompt_ids = tuple(
            int(token)
            for token in tokenizer.encode(args.prompt_text, add_special_tokens=not args.no_special_tokens)
        )
    else:
        prompt_ids = _parse_ids(args.prompt_ids or "1")
    eos_token_id = None if args.ignore_eos else int(runtime.config["eos_token_id"])

    with torch.no_grad():
        circuit_run = runtime.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        source_run = _source_greedy_generate(
            torch=torch,
            source=source,
            dynamic_cache=dynamic_cache,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
        )

    text_report = None
    if tokenizer is not None:
        skip_special_tokens = not args.keep_special_tokens
        circuit_text_run = generate_text(
            runtime=runtime,
            tokenizer=tokenizer,
            prompt_text=args.prompt_text,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            add_special_tokens=not args.no_special_tokens,
            skip_special_tokens=skip_special_tokens,
        )
        source_output_text = tokenizer.decode(source_run["output_ids"], skip_special_tokens=skip_special_tokens)
        source_generated_text = tokenizer.decode(source_run["generated_ids"], skip_special_tokens=skip_special_tokens)
        text_report = {
            "prompt_text": args.prompt_text,
            "circuit_generated_text": circuit_text_run.generated_text,
            "source_generated_text": source_generated_text,
            "circuit_output_text": circuit_text_run.output_text,
            "source_output_text": source_output_text,
            "generated_text_match": circuit_text_run.generated_text == source_generated_text,
            "output_text_match": circuit_text_run.output_text == source_output_text,
        }

    report = {
        "prompt_text": args.prompt_text,
        "prompt_ids": list(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": eos_token_id,
        "sampler": circuit_run.sampler,
        "circuit": {
            "generated_ids": list(circuit_run.generated_ids),
            "output_ids": list(circuit_run.output_ids),
            "stop_reason": circuit_run.stop_reason,
            "generated_count": len(circuit_run.generated_ids),
            "last_state": _last_state(circuit_run),
            "implementations": _last_implementations(circuit_run),
        },
        "source": source_run,
        "comparison": {
            "generated_ids_match": list(circuit_run.generated_ids) == source_run["generated_ids"],
            "output_ids_match": list(circuit_run.output_ids) == source_run["output_ids"],
            "stop_reason_match": circuit_run.stop_reason == source_run["stop_reason"],
        },
        "text": text_report,
    }
    if text_report is not None:
        report["comparison"]["generated_text_match"] = bool(text_report["generated_text_match"])
        report["comparison"]["output_text_match"] = bool(text_report["output_text_match"])

    if args.summary:
        report = {
            "prompt_text": report["prompt_text"],
            "prompt_ids": report["prompt_ids"],
            "max_new_tokens": report["max_new_tokens"],
            "eos_token_id": report["eos_token_id"],
            "sampler": report["sampler"],
            "circuit_generated_ids": report["circuit"]["generated_ids"],
            "source_generated_ids": report["source"]["generated_ids"],
            "circuit_output_ids": report["circuit"]["output_ids"],
            "comparison": report["comparison"],
            "text": report["text"],
            "last_state": report["circuit"]["last_state"],
            "implementations": report["circuit"]["implementations"],
        }

    print(json.dumps(report, indent=2))

    if not all(report["comparison"].values()):
        raise SystemExit("circuit generation diverged from source oracle")


def _source_greedy_generate(
    torch: Any,
    source: Any,
    dynamic_cache: Any,
    prompt_ids: tuple[int, ...],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict[str, Any]:
    cache = dynamic_cache(config=source.config)
    last_logits = None
    for token_id in prompt_ids:
        output = source(
            input_ids=torch.tensor([[int(token_id)]], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        last_logits = output.logits

    if last_logits is None:
        raise ValueError("prompt_ids must not be empty")

    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(last_logits[:, -1, :], dim=-1).item())
        generated_ids.append(next_token)
        output = source(
            input_ids=torch.tensor([[next_token]], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        last_logits = output.logits
        if eos_token_id is not None and next_token == int(eos_token_id):
            stop_reason = "eos"
            break

    return {
        "generated_ids": generated_ids,
        "output_ids": list(prompt_ids) + generated_ids,
        "stop_reason": stop_reason,
        "generated_count": len(generated_ids),
    }


def _last_state(run: Any) -> dict[str, Any] | None:
    if run.generated_steps:
        steps = run.generated_steps[-1].tick.output.steps
    elif run.prompt_ticks:
        steps = run.prompt_ticks[-1].output.steps
    else:
        return None
    return next(step.state for step in steps if step.operator_type == "full_attention")


def _last_implementations(run: Any) -> dict[str, str]:
    if run.generated_steps:
        steps = run.generated_steps[-1].tick.output.steps
    elif run.prompt_ticks:
        steps = run.prompt_ticks[-1].output.steps
    else:
        return {}
    return {step.pedal_id: step.implementation for step in steps}


def _parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not ids:
        raise ValueError("at least one prompt token id is required")
    return ids


if __name__ == "__main__":
    main()
