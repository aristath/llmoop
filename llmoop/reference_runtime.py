from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.pedalboard import Json, PedalInstance, Pedalboard
from llmoop.source_oracle import _oracle_imports
from llmoop.stream_state import PedalStreamState


@dataclass(frozen=True)
class ReferenceFrame:
    id: str
    signal: str
    tensor: Any
    origin: str
    history: tuple[str, ...] = ()

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape[-1:])

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "signal": self.signal,
            "tensor_shape": list(self.tensor.shape),
            "frame_shape": list(self.frame_shape),
            "dtype": str(self.tensor.dtype),
            "origin": self.origin,
            "history": list(self.history),
            "summary": _tensor_summary(self.tensor),
        }


@dataclass(frozen=True)
class ReferencePedalStep:
    pedal_id: str
    operator_type: str
    implementation: str
    input_frame: ReferenceFrame
    output_frame: ReferenceFrame
    state: Json

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "operator_type": self.operator_type,
            "implementation": self.implementation,
            "input_frame": self.input_frame.to_json(),
            "output_frame": self.output_frame.to_json(),
            "state": self.state,
        }


@dataclass(frozen=True)
class ReferenceActivation:
    input_ids: tuple[int, ...]
    input_frame: ReferenceFrame
    pedal_output_frame: ReferenceFrame
    normalized_output_frame: ReferenceFrame
    steps: tuple[ReferencePedalStep, ...]
    comparison: Json

    def to_json(self) -> Json:
        return {
            "input_ids": list(self.input_ids),
            "input_frame": self.input_frame.to_json(),
            "pedal_output_frame": self.pedal_output_frame.to_json(),
            "normalized_output_frame": self.normalized_output_frame.to_json(),
            "steps": [step.to_json() for step in self.steps],
            "comparison": self.comparison,
        }


@dataclass(frozen=True)
class ReferenceStreamTick:
    tick: int
    token_id: int
    activation: ReferenceActivation
    incremental_comparison: Json

    def to_json(self) -> Json:
        return {
            "tick": self.tick,
            "token_id": self.token_id,
            "activation": self.activation.to_json(),
            "incremental_comparison": self.incremental_comparison,
        }


@dataclass(frozen=True)
class ReferenceStreamRun:
    input_ids: tuple[int, ...]
    ticks: tuple[ReferenceStreamTick, ...]
    output_tensor: Any
    comparison: Json

    def to_json(self) -> Json:
        return {
            "input_ids": list(self.input_ids),
            "tick_count": len(self.ticks),
            "ticks": [tick.to_json() for tick in self.ticks],
            "output_tensor_shape": list(self.output_tensor.shape),
            "comparison": self.comparison,
        }


class ReferencePedalExecutor:
    """Executes transpiled pedals by delegating each pedal to the source Transformers layer.

    This is a reference backend, not a new inference implementation. Its purpose
    is to prove that the pedalboard can carry real source behavior behind the
    same layer-pedal boundary used by the symbolic runtime.
    """

    def __init__(
        self,
        pedalboard: Pedalboard,
        model: Any,
        dynamic_cache: Any,
        torch: Any,
        pedal_implementations: dict[str, Any] | None = None,
        use_custom_stream_state: bool = False,
    ) -> None:
        self.pedalboard = pedalboard
        self.model = model
        self.dynamic_cache = dynamic_cache
        self.torch = torch
        self.pedal_implementations = pedal_implementations or {}
        self.use_custom_stream_state = use_custom_stream_state

    @classmethod
    def from_model_dir(cls, pedalboard: Pedalboard, model_dir: Path) -> "ReferencePedalExecutor":
        torch, auto_model, dynamic_cache = _oracle_imports()
        model = auto_model.from_pretrained(model_dir, dtype=torch.float32)
        model.eval()
        return cls(pedalboard=pedalboard, model=model, dynamic_cache=dynamic_cache, torch=torch)

    def install_pedal_implementation(self, pedal_id: str, implementation: Any) -> None:
        self.pedal_implementations[pedal_id] = implementation

    def install_candidate_pedal(self, pedal_id: str, candidate: Any) -> None:
        self.install_pedal_implementation(pedal_id, candidate)

    def use_pedal_stream_state(self, enabled: bool = True) -> None:
        self.use_custom_stream_state = enabled

    def new_execution_state(self) -> Any:
        if self.use_custom_stream_state:
            return PedalStreamState.from_pedalboard(self.pedalboard, self.torch)
        return self.dynamic_cache(config=self.model.config)

    def activate_token(self, token_id: int | None = None) -> ReferenceActivation:
        config = self.model.config
        token_id = token_id if token_id is not None else config.bos_token_id
        if token_id is None:
            token_id = 1
        return self.activate_input_ids((int(token_id),))

    def activate_input_ids(self, input_ids: tuple[int, ...]) -> ReferenceActivation:
        if not input_ids:
            raise ValueError("input_ids must not be empty")

        torch = self.torch
        input_tensor = torch.tensor([list(input_ids)], dtype=torch.long)

        with torch.no_grad():
            source_output = self.model.model(input_ids=input_tensor, use_cache=True).last_hidden_state
            cache = self.new_execution_state()
            activation = self._activate_input_tensor(
                input_tensor=input_tensor,
                position_offset=0,
                cache=cache,
                source_output=source_output,
                frame_id="source_embedding",
            )

        return activation

    def open_stream(self) -> "ReferencePedalStream":
        return ReferencePedalStream(executor=self)

    def _activate_input_tensor(
        self,
        input_tensor: Any,
        position_offset: int,
        cache: Any,
        source_output: Any,
        frame_id: str,
    ) -> ReferenceActivation:
        torch = self.torch
        input_ids = tuple(int(token) for token in input_tensor[0].tolist())

        hidden = self.model.model.embed_tokens(input_tensor)
        position_ids = (
            torch.arange(position_offset, position_offset + hidden.shape[1], device=hidden.device, dtype=torch.long)
            .unsqueeze(0)
        )
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids=position_ids)

        frame = ReferenceFrame(
            id=frame_id,
            signal=self.pedalboard.pedals[0].input_port.signal,
            tensor=hidden,
            origin="source_embedding",
            history=(frame_id,),
        )
        input_frame = frame
        steps: list[ReferencePedalStep] = []

        for layer_index, pedal in enumerate(self.pedalboard.pedals):
            _validate_pedal_matches_source(pedal, layer_index, self.model.config.layer_types[layer_index])
            pedal_implementation = self.pedal_implementations.get(pedal.id)
            if pedal_implementation is not None:
                output = pedal_implementation.forward(
                    frame.tensor,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=cache,
                )
                implementation = pedal_implementation.implementation
            else:
                if isinstance(cache, PedalStreamState):
                    raise ValueError(
                        f"{pedal.id} is still a source layer; custom per-pedal stream state requires an executable pedal implementation"
                    )
                layer = self.model.model.layers[layer_index]
                output = layer(
                    frame.tensor,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=cache,
                )
                implementation = "source_transformers_layer"
            output_frame = ReferenceFrame(
                id=f"{frame.id}.{pedal.id}",
                signal=pedal.output_port.signal,
                tensor=output,
                origin=pedal.id,
                history=frame.history + (pedal.id,),
            )
            steps.append(
                ReferencePedalStep(
                    pedal_id=pedal.id,
                    operator_type=pedal.operator_type,
                    implementation=implementation,
                    input_frame=frame,
                    output_frame=output_frame,
                    state=_state_summary(cache, layer_index, pedal.operator_type),
                )
            )
            frame = output_frame

        normalized = self.model.model.embedding_norm(frame.tensor)
        normalized_frame = ReferenceFrame(
            id=f"{frame.id}.embedding_norm",
            signal="normalized_frame",
            tensor=normalized,
            origin="source_output_transducer.embedding_norm",
            history=frame.history + ("embedding_norm",),
        )

        return ReferenceActivation(
            input_ids=input_ids,
            input_frame=input_frame,
            pedal_output_frame=frame,
            normalized_output_frame=normalized_frame,
            steps=tuple(steps),
            comparison=_compare_tensors(
                torch,
                normalized_frame.tensor,
                source_output,
                reference="AutoModel.model.forward",
                candidate="pedalboard_walk_with_source_layers",
                atol=1e-6,
                rtol=1e-5,
            ),
        )


class ReferencePedalStream:
    """Persistent source-backed stream using one source cache across ticks."""

    def __init__(self, executor: ReferencePedalExecutor) -> None:
        self.executor = executor
        self.cache = executor.new_execution_state()
        self.source_cache = executor.dynamic_cache(config=executor.model.config)
        self.position = 0
        self.ticks: list[ReferenceStreamTick] = []
        self.output_frames: list[ReferenceFrame] = []

    def tick(self, token_id: int) -> ReferenceStreamTick:
        torch = self.executor.torch
        input_tensor = torch.tensor([[int(token_id)]], dtype=torch.long)

        with torch.no_grad():
            source_output = self.executor.model.model(
                input_ids=input_tensor,
                past_key_values=self.source_cache,
                use_cache=True,
            ).last_hidden_state

            activation = self.executor._activate_input_tensor(
                input_tensor=input_tensor,
                position_offset=self.position,
                cache=self.cache,
                source_output=source_output,
                frame_id=f"stream_token_{self.position}",
            )

        comparison = _compare_tensors(
            torch,
            activation.normalized_output_frame.tensor,
            source_output,
            reference="AutoModel.model.forward_incremental_cache",
            candidate="pedalboard_stream_tick_with_source_layers",
            atol=1e-6,
            rtol=1e-5,
        )
        tick = ReferenceStreamTick(
            tick=self.position,
            token_id=int(token_id),
            activation=activation,
            incremental_comparison=comparison,
        )
        self.position += 1
        self.ticks.append(tick)
        self.output_frames.append(activation.normalized_output_frame)
        return tick

    def run_teacher_forced(self, input_ids: tuple[int, ...]) -> ReferenceStreamRun:
        if not input_ids:
            raise ValueError("input_ids must not be empty")
        ticks = tuple(self.tick(token_id) for token_id in input_ids)

        torch = self.executor.torch
        output_tensor = torch.cat([frame.tensor for frame in self.output_frames], dim=1)
        with torch.no_grad():
            source_full_output = self.executor.model.model(
                input_ids=torch.tensor([list(input_ids)], dtype=torch.long),
                use_cache=True,
            ).last_hidden_state

        return ReferenceStreamRun(
            input_ids=tuple(int(token) for token in input_ids),
            ticks=ticks,
            output_tensor=output_tensor,
            comparison=_compare_tensors(
                torch,
                output_tensor,
                source_full_output,
                reference="AutoModel.model.forward_full_sequence",
                candidate="pedalboard_stream_ticks_with_source_layers",
                atol=1e-4,
                rtol=1e-4,
            ),
        )


def _validate_pedal_matches_source(pedal: PedalInstance, layer_index: int, source_layer_type: str) -> None:
    expected_id = f"layer_{layer_index:02d}"
    if pedal.id != expected_id:
        raise ValueError(f"expected {expected_id}, got {pedal.id}")
    if pedal.operator_type != source_layer_type:
        raise ValueError(f"{pedal.id} operator type {pedal.operator_type!r} does not match source {source_layer_type!r}")


def _state_summary(cache: Any, layer_index: int, operator_type: str) -> Json:
    if hasattr(cache, "summary_for"):
        return cache.summary_for(layer_index, operator_type)

    layer_cache = cache.layers[layer_index]
    if operator_type == "conv":
        conv_states = getattr(layer_cache, "conv_states", None)
        return {
            "kind": "rolling_frame_memory",
            "source_layout": "batch_hidden_time",
            "source_shape": list(conv_states.shape) if conv_states is not None else None,
        }

    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)
    return {
        "kind": "append_only_attention_memory",
        "source_layout": "batch_kvheads_seq_headdim",
        "source_key_shape": list(keys.shape) if keys is not None else None,
        "source_value_shape": list(values.shape) if values is not None else None,
    }


def _tensor_summary(tensor: Any) -> Json:
    detached = tensor.detach().float()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def _compare_tensors(
    torch: Any,
    candidate_tensor: Any,
    reference_tensor: Any,
    reference: str,
    candidate: str,
    atol: float,
    rtol: float,
) -> Json:
    diff = (candidate_tensor - reference_tensor).abs()
    return {
        "reference": reference,
        "candidate": candidate,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(torch.allclose(candidate_tensor, reference_tensor, atol=atol, rtol=rtol)),
    }
