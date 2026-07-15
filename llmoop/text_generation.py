from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.installed_processor import InstalledStreamProcessor
from llmoop.pedalboard import Json


@dataclass(frozen=True)
class TextInputTransducer:
    tokenizer: Any
    add_special_tokens: bool = True

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens))


@dataclass(frozen=True)
class TextOutputTransducer:
    tokenizer: Any
    skip_special_tokens: bool = True

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=self.skip_special_tokens))


@dataclass(frozen=True)
class TextGenerationRun:
    prompt_text: str
    prompt_ids: tuple[int, ...]
    generated_text: str
    output_text: str
    generation: Any

    @property
    def generated_ids(self) -> tuple[int, ...]:
        return self.generation.generated_ids

    @property
    def output_ids(self) -> tuple[int, ...]:
        return self.generation.output_ids

    def to_json(self) -> Json:
        return {
            "prompt_text": self.prompt_text,
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "output_ids": list(self.output_ids),
            "generated_text": self.generated_text,
            "output_text": self.output_text,
            "sampler": self.generation.sampler,
            "stop_reason": self.generation.stop_reason,
            "generated_count": len(self.generated_ids),
        }


def load_tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def generate_text(
    runtime: CircuitModelRuntime,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
    sampler: Any | None = None,
    random_seed: int = 0,
) -> TextGenerationRun:
    input_transducer = TextInputTransducer(tokenizer=tokenizer, add_special_tokens=add_special_tokens)
    output_transducer = TextOutputTransducer(tokenizer=tokenizer, skip_special_tokens=skip_special_tokens)
    encoded_prompt_ids = input_transducer.encode(prompt_text)
    processor = InstalledStreamProcessor.from_runtime(runtime=runtime, sampler=sampler, random_seed=random_seed)
    generation = processor.run_prompt(
        stream_id="stream_0",
        prompt_ids=encoded_prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    return TextGenerationRun(
        prompt_text=prompt_text,
        prompt_ids=generation.prompt_ids,
        generated_text=output_transducer.decode(generation.generated_ids),
        output_text=output_transducer.decode(generation.output_ids),
        generation=generation,
    )
