from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.circuit_executors import GQAAttentionCircuitPedal, ShortConvCircuitPedal
from llmoop.circuit_pedalboard import CircuitPedalboard
from llmoop.pedalboard import Json
from llmoop.samplers import GreedySamplerPedal
from llmoop.stream_state import PedalStreamState
from llmoop.tensor_store import SafetensorsTensorStore


@dataclass(frozen=True)
class CircuitModelStep:
    pedal_id: str
    operator_type: str
    implementation: str
    state: Json

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "operator_type": self.operator_type,
            "implementation": self.implementation,
            "state": self.state,
        }


@dataclass(frozen=True)
class CircuitModelOutput:
    input_ids: tuple[int, ...]
    hidden_states: Any
    logits: Any
    state: PedalStreamState
    steps: tuple[CircuitModelStep, ...]

    def to_json(self) -> Json:
        return {
            "input_ids": list(self.input_ids),
            "hidden_shape": list(self.hidden_states.shape),
            "logits_shape": list(self.logits.shape),
            "steps": [step.to_json() for step in self.steps],
            "state": self.state.to_json(),
        }


@dataclass(frozen=True)
class CircuitModelStreamTick:
    tick: int
    token_id: int
    output: CircuitModelOutput

    def to_json(self) -> Json:
        return {
            "tick": self.tick,
            "token_id": self.token_id,
            "output": self.output.to_json(),
        }


@dataclass(frozen=True)
class CircuitGenerationStep:
    index: int
    token_id: int
    selected_from_tick: int
    sampler: Json
    tick: CircuitModelStreamTick
    stopped: bool
    stop_reason: str | None = None

    def to_json(self) -> Json:
        return {
            "index": self.index,
            "token_id": self.token_id,
            "selected_from_tick": self.selected_from_tick,
            "sampler": self.sampler,
            "tick": self.tick.to_json(),
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class CircuitGenerationRun:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    sampler: str
    stop_reason: str
    prompt_ticks: tuple[CircuitModelStreamTick, ...]
    generated_steps: tuple[CircuitGenerationStep, ...]

    def to_json(self) -> Json:
        return {
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "output_ids": list(self.output_ids),
            "sampler": self.sampler,
            "stop_reason": self.stop_reason,
            "prompt_ticks": [tick.to_json() for tick in self.prompt_ticks],
            "generated_steps": [step.to_json() for step in self.generated_steps],
        }


class CircuitModelRuntime:
    """Source-module-free runtime around the lowered executable pedalboard.

    The source model remains useful as an oracle, but this runtime does not
    instantiate Transformers modules for execution. It loads tensors directly
    from safetensors and runs the model transducers plus executable pedals.
    """

    def __init__(
        self,
        torch: Any,
        config: Json,
        board: CircuitPedalboard,
        tensor_store: SafetensorsTensorStore,
        input_embedding_tensor: str,
        output_norm_tensor: str,
        output_projection_tensor: str,
    ) -> None:
        self.torch = torch
        self.config = config
        self.board = board
        self.tensor_store = tensor_store
        self.embed_tokens_weight = tensor_store.get(input_embedding_tensor)
        self.embedding_norm_weight = tensor_store.get(output_norm_tensor)
        self.output_projection_weight = tensor_store.get(output_projection_tensor)
        self.pedals = tuple(self._build_pedal(pedal) for pedal in board.pedals)
        self.inv_freq, self.attention_scaling = self._build_rope_parameters()

    @classmethod
    def from_dirs(
        cls,
        circuit_dir: Path,
        package_dir: Path,
        torch: Any | None = None,
    ) -> "CircuitModelRuntime":
        if torch is None:
            import torch as torch_module

            torch = torch_module

        circuit_dir = circuit_dir.expanduser()
        package_dir = package_dir.expanduser()
        board = CircuitPedalboard.from_dir(circuit_dir)
        config_path = package_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"runtime package is missing {config_path}")
        config = json.loads(config_path.read_text())

        tensor_index = package_dir / "tensors.json"
        if not tensor_index.is_file():
            raise FileNotFoundError(f"runtime package is missing {tensor_index}")
        tensor_store = SafetensorsTensorStore.from_tensor_index(
            tensor_index=tensor_index,
            torch=torch,
            dtype=torch.float32,
        )
        input_embedding_tensor = board.index["graph"]["input_transducer"]["params"]["weight"]["tensor"]
        output_components = board.index["graph"]["output_transducer"]["components"]
        output_norm_tensor = next(
            component["params"]["weight"]["tensor"]
            for component in output_components
            if component["type"] == "rms_norm"
        )
        output_projection_tensor = next(
            component["params"]["weight"]["tensor"]
            for component in output_components
            if component["type"] == "linear_projection"
        )
        return cls(
            torch=torch,
            config=config,
            board=board,
            tensor_store=tensor_store,
            input_embedding_tensor=input_embedding_tensor,
            output_norm_tensor=output_norm_tensor,
            output_projection_tensor=output_projection_tensor,
        )

    def open_stream(self) -> "CircuitModelStream":
        return CircuitModelStream(runtime=self)

    def generate(
        self,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        sampler: Any | None = None,
    ) -> CircuitGenerationRun:
        return self.open_stream().generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            sampler=sampler,
        )

    def new_state(self) -> PedalStreamState:
        return PedalStreamState.from_pedalboard(self.board, self.torch)  # type: ignore[arg-type]

    def forward_input_ids(
        self,
        input_ids: tuple[int, ...],
        state: PedalStreamState | None = None,
        position_offset: int = 0,
    ) -> CircuitModelOutput:
        if not input_ids:
            raise ValueError("input_ids must not be empty")

        state = state or self.new_state()
        input_tensor = self.torch.tensor([list(input_ids)], dtype=self.torch.long)
        hidden = self.torch.nn.functional.embedding(input_tensor, self.embed_tokens_weight)
        position_ids = (
            self.torch.arange(
                position_offset,
                position_offset + hidden.shape[1],
                device=hidden.device,
                dtype=self.torch.long,
            ).unsqueeze(0)
        )
        position_embeddings = self.rotary_embeddings(hidden, position_ids)

        steps: list[CircuitModelStep] = []
        for pedal, implementation in zip(self.board.pedals, self.pedals):
            hidden = implementation.forward(
                hidden,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=state,
            )
            steps.append(
                CircuitModelStep(
                    pedal_id=pedal.id,
                    operator_type=pedal.operator_type,
                    implementation=implementation.implementation,
                    state=state.summary_for(pedal.circuit["source"]["source_layer_index"], pedal.operator_type),
                )
            )

        hidden = self.rms_norm(hidden, self.embedding_norm_weight)
        logits = self.torch.nn.functional.linear(hidden, self.output_projection_weight)
        return CircuitModelOutput(
            input_ids=tuple(int(token) for token in input_ids),
            hidden_states=hidden,
            logits=logits,
            state=state,
            steps=tuple(steps),
        )

    def rotary_embeddings(self, hidden: Any, position_ids: Any) -> tuple[Any, Any]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(hidden.device)
        position_ids_expanded = position_ids[:, None, :].float()
        with self.torch.no_grad():
            freqs = (inv_freq.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = self.torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=hidden.dtype), sin.to(dtype=hidden.dtype)

    def rms_norm(self, hidden_states: Any, weight: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * self.torch.rsqrt(variance + float(self.config["norm_eps"]))
        return weight * hidden_states.to(input_dtype)

    def _build_pedal(self, pedal: Any) -> Any:
        if pedal.operator_type == "conv":
            return ShortConvCircuitPedal.from_tensor_store(
                tensor_store=self.tensor_store,
                torch=self.torch,
                circuit=pedal.circuit,
                config=self.config,
            )
        if pedal.operator_type == "full_attention":
            return GQAAttentionCircuitPedal.from_tensor_store(
                tensor_store=self.tensor_store,
                torch=self.torch,
                circuit=pedal.circuit,
                config=self.config,
            )
        raise ValueError(f"unsupported circuit pedal operator type {pedal.operator_type!r}")

    def _build_rope_parameters(self) -> tuple[Any, float]:
        base = float(self.config["rope_parameters"]["rope_theta"])
        head_dim = int(self.config.get("head_dim") or self.config["hidden_size"] // self.config["num_attention_heads"])
        inv_freq = 1.0 / (
            base ** (self.torch.arange(0, head_dim, 2, dtype=self.torch.int64).to(dtype=self.torch.float32) / head_dim)
        )
        return inv_freq, 1.0


class CircuitModelStream:
    def __init__(self, runtime: CircuitModelRuntime) -> None:
        self.runtime = runtime
        self.state = runtime.new_state()
        self.position = 0
        self.ticks: list[CircuitModelStreamTick] = []
        self.hidden_frames: list[Any] = []
        self.logit_frames: list[Any] = []

    def tick(self, token_id: int) -> CircuitModelStreamTick:
        output = self.runtime.forward_input_ids(
            (int(token_id),),
            state=self.state,
            position_offset=self.position,
        )
        tick = CircuitModelStreamTick(tick=self.position, token_id=int(token_id), output=output)
        self.position += 1
        self.ticks.append(tick)
        self.hidden_frames.append(output.hidden_states)
        self.logit_frames.append(output.logits)
        return tick

    def run_teacher_forced(self, input_ids: tuple[int, ...]) -> tuple[CircuitModelStreamTick, ...]:
        return tuple(self.tick(token_id) for token_id in input_ids)

    def generate(
        self,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        sampler: Any | None = None,
    ) -> CircuitGenerationRun:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not prompt_ids:
            bos = self.runtime.config.get("bos_token_id")
            if bos is None:
                raise ValueError("prompt_ids is empty and config has no bos_token_id")
            prompt_ids = (int(bos),)

        sampler = sampler or GreedySamplerPedal()
        prompt_ticks = self.run_teacher_forced(tuple(int(token) for token in prompt_ids))
        generated_ids: list[int] = []
        generated_steps: list[CircuitGenerationStep] = []
        stop_reason = "max_new_tokens"

        for index in range(max_new_tokens):
            selected_from_tick = self.ticks[-1].tick
            decision = sampler.sample(self.logit_frames[-1], self.runtime.torch)
            next_token = decision.token_id
            tick = self.tick(next_token)
            generated_ids.append(next_token)
            stopped = eos_token_id is not None and next_token == int(eos_token_id)
            if stopped:
                stop_reason = "eos"
            generated_steps.append(
                CircuitGenerationStep(
                    index=index,
                    token_id=next_token,
                    selected_from_tick=selected_from_tick,
                    sampler=decision.to_json(),
                    tick=tick,
                    stopped=stopped,
                    stop_reason=stop_reason if stopped else None,
                )
            )
            if stopped:
                break

        return CircuitGenerationRun(
            prompt_ids=tuple(int(token) for token in prompt_ids),
            generated_ids=tuple(generated_ids),
            output_ids=tuple(int(token) for token in prompt_ids) + tuple(generated_ids),
            sampler=sampler.id,
            stop_reason=stop_reason,
            prompt_ticks=prompt_ticks,
            generated_steps=tuple(generated_steps),
        )

    def greedy_next_token(self, logits: Any) -> int:
        return GreedySamplerPedal().sample(logits, self.runtime.torch).token_id

    @property
    def hidden_states(self) -> Any:
        return self.runtime.torch.cat(self.hidden_frames, dim=1)

    @property
    def logits(self) -> Any:
        return self.runtime.torch.cat(self.logit_frames, dim=1)
