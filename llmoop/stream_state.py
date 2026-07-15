from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmoop.pedalboard import Json, Pedalboard


@dataclass
class ConvPedalState:
    pedal_id: str
    layer_index: int
    state_id: str
    tensor: Any | None = None
    updates: int = 0


@dataclass
class AttentionPedalState:
    pedal_id: str
    layer_index: int
    state_id: str
    key: Any | None = None
    value: Any | None = None
    updates: int = 0


@dataclass(frozen=True)
class ConvPedalStateSnapshot:
    pedal_id: str
    layer_index: int
    state_id: str
    tensor: Any | None
    updates: int

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "layer_index": self.layer_index,
            "state_id": self.state_id,
            "source_shape": list(self.tensor.shape) if self.tensor is not None else None,
            "updates": self.updates,
        }


@dataclass(frozen=True)
class AttentionPedalStateSnapshot:
    pedal_id: str
    layer_index: int
    state_id: str
    key: Any | None
    value: Any | None
    updates: int

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "layer_index": self.layer_index,
            "state_id": self.state_id,
            "source_key_shape": list(self.key.shape) if self.key is not None else None,
            "source_value_shape": list(self.value.shape) if self.value is not None else None,
            "updates": self.updates,
        }


@dataclass(frozen=True)
class PedalStreamStateSnapshot:
    kind: str
    conv_states: tuple[ConvPedalStateSnapshot, ...]
    attention_states: tuple[AttentionPedalStateSnapshot, ...]

    def restore(self, torch: Any) -> "PedalStreamState":
        return PedalStreamState(
            torch=torch,
            conv_states={
                state.layer_index: ConvPedalState(
                    pedal_id=state.pedal_id,
                    layer_index=state.layer_index,
                    state_id=state.state_id,
                    tensor=_clone_tensor(state.tensor),
                    updates=state.updates,
                )
                for state in self.conv_states
            },
            attention_states={
                state.layer_index: AttentionPedalState(
                    pedal_id=state.pedal_id,
                    layer_index=state.layer_index,
                    state_id=state.state_id,
                    key=_clone_tensor(state.key),
                    value=_clone_tensor(state.value),
                    updates=state.updates,
                )
                for state in self.attention_states
            },
        )

    def to_json(self) -> Json:
        return {
            "kind": self.kind,
            "conv_states": [state.to_json() for state in self.conv_states],
            "attention_states": [state.to_json() for state in self.attention_states],
        }


class PedalStreamState:
    """Per-pedal transient stream state for executable circuit pedals.

    This intentionally mirrors only the tiny update API currently needed by
    the executable pedals. Logical ownership is per pedal/state port, even
    though a future backend may physically pack these regions into one GPU
    allocation.
    """

    kind = "pedal_stream_state"

    def __init__(
        self,
        torch: Any,
        conv_states: dict[int, ConvPedalState],
        attention_states: dict[int, AttentionPedalState],
    ) -> None:
        self.torch = torch
        self.conv_states = conv_states
        self.attention_states = attention_states

    @classmethod
    def from_pedalboard(cls, pedalboard: Pedalboard, torch: Any) -> "PedalStreamState":
        conv_states: dict[int, ConvPedalState] = {}
        attention_states: dict[int, AttentionPedalState] = {}
        for layer_index, pedal in enumerate(pedalboard.pedals):
            if pedal.operator_type == "conv":
                state_port = pedal.state_ports[0]
                conv_states[layer_index] = ConvPedalState(
                    pedal_id=pedal.id,
                    layer_index=layer_index,
                    state_id=state_port.id,
                )
            elif pedal.operator_type == "full_attention":
                state_port = pedal.state_ports[0]
                attention_states[layer_index] = AttentionPedalState(
                    pedal_id=pedal.id,
                    layer_index=layer_index,
                    state_id=state_port.id,
                )
        return cls(torch=torch, conv_states=conv_states, attention_states=attention_states)

    def has_previous_state(self, layer_index: int) -> bool:
        conv = self.conv_states.get(layer_index)
        if conv is not None:
            return conv.tensor is not None
        attention = self.attention_states.get(layer_index)
        return attention is not None and attention.key is not None and attention.value is not None

    def snapshot(self) -> PedalStreamStateSnapshot:
        return PedalStreamStateSnapshot(
            kind=self.kind,
            conv_states=tuple(
                ConvPedalStateSnapshot(
                    pedal_id=state.pedal_id,
                    layer_index=state.layer_index,
                    state_id=state.state_id,
                    tensor=_clone_tensor(state.tensor),
                    updates=state.updates,
                )
                for _, state in sorted(self.conv_states.items())
            ),
            attention_states=tuple(
                AttentionPedalStateSnapshot(
                    pedal_id=state.pedal_id,
                    layer_index=state.layer_index,
                    state_id=state.state_id,
                    key=_clone_tensor(state.key),
                    value=_clone_tensor(state.value),
                    updates=state.updates,
                )
                for _, state in sorted(self.attention_states.items())
            ),
        )

    def clone(self) -> "PedalStreamState":
        return self.snapshot().restore(self.torch)

    def update_conv_state(self, conv_state: Any, layer_index: int) -> Any:
        state = self.conv_states[layer_index]
        width = _conv_width(conv_state, previous=state.tensor)
        if state.tensor is None:
            state.tensor = conv_state.contiguous()
        else:
            state.tensor = self.torch.cat((state.tensor.to(conv_state.device), conv_state), dim=-1)[..., -width:]
        state.updates += 1
        return state.tensor

    def update(self, key_states: Any, value_states: Any, layer_index: int) -> tuple[Any, Any]:
        state = self.attention_states[layer_index]
        if state.key is None:
            state.key = key_states.contiguous()
            state.value = value_states.contiguous()
        else:
            state.key = self.torch.cat((state.key.to(key_states.device), key_states), dim=2).contiguous()
            state.value = self.torch.cat((state.value.to(value_states.device), value_states), dim=2).contiguous()
        state.updates += 1
        return state.key, state.value

    def summary_for(self, layer_index: int, operator_type: str) -> Json:
        if operator_type == "conv":
            state = self.conv_states[layer_index]
            return {
                "kind": "rolling_frame_memory",
                "owner": "pedal",
                "pedal_id": state.pedal_id,
                "state_id": state.state_id,
                "source_layout": "batch_hidden_time",
                "source_shape": list(state.tensor.shape) if state.tensor is not None else None,
                "updates": state.updates,
            }

        state = self.attention_states[layer_index]
        return {
            "kind": "append_only_attention_memory",
            "owner": "pedal",
            "pedal_id": state.pedal_id,
            "state_id": state.state_id,
            "source_layout": "batch_kvheads_seq_headdim",
            "source_key_shape": list(state.key.shape) if state.key is not None else None,
            "source_value_shape": list(state.value.shape) if state.value is not None else None,
            "updates": state.updates,
        }

    def to_json(self) -> Json:
        return {
            "kind": self.kind,
            "conv_states": [
                self.summary_for(layer_index, "conv")
                for layer_index in sorted(self.conv_states)
            ],
            "attention_states": [
                self.summary_for(layer_index, "full_attention")
                for layer_index in sorted(self.attention_states)
            ],
        }


def _conv_width(conv_state: Any, previous: Any | None) -> int:
    if previous is not None:
        return int(previous.shape[-1])
    return int(conv_state.shape[-1])


def _clone_tensor(tensor: Any | None) -> Any | None:
    if tensor is None:
        return None
    return tensor.detach().clone().contiguous()
