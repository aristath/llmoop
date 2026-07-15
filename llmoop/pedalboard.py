from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Json = dict[str, Any]


@dataclass(frozen=True)
class SignalPort:
    id: str
    signal: str
    shape: tuple[int, ...]

    @classmethod
    def from_json(cls, data: Json) -> "SignalPort":
        return cls(
            id=data["id"],
            signal=data["signal"],
            shape=tuple(data["shape"]),
        )


@dataclass(frozen=True)
class StatePort:
    id: str
    type: str
    spec: Json

    @classmethod
    def from_json(cls, data: Json) -> "StatePort":
        return cls(id=data["id"], type=data["type"], spec=data)

    def static_shape(self) -> tuple[int, ...] | None:
        shape = self.spec.get("shape")
        if shape is None:
            return None
        return tuple(shape)

    def elements_per_token(self) -> int | None:
        key_shape = self.spec.get("key_shape_per_token")
        value_shape = self.spec.get("value_shape_per_token")
        if key_shape is None or value_shape is None:
            return None
        return _product(key_shape) + _product(value_shape)


@dataclass(frozen=True)
class ParameterBlock:
    layout: str
    tensor_refs: tuple[str, ...]

    @classmethod
    def from_json(cls, data: Json) -> "ParameterBlock":
        return cls(
            layout=data["layout"],
            tensor_refs=tuple(data["tensor_refs"]),
        )


@dataclass(frozen=True)
class PedalInstance:
    id: str
    pedal_class: str
    operator_type: str
    input_port: SignalPort
    output_port: SignalPort
    state_ports: tuple[StatePort, ...]
    parameter_block: ParameterBlock
    source_file: Path

    @classmethod
    def from_file(cls, path: Path) -> "PedalInstance":
        data = _read_json(path)
        if data["schema"] != "llmoop.pedal_instance.v1":
            raise ValueError(f"{path} is not a pedal instance file")
        inputs = data["ports"]["inputs"]
        outputs = data["ports"]["outputs"]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{path} must have exactly one input and one output port for series wiring")
        return cls(
            id=data["id"],
            pedal_class=data["pedal_class"],
            operator_type=data["operator_type"],
            input_port=SignalPort.from_json(inputs[0]),
            output_port=SignalPort.from_json(outputs[0]),
            state_ports=tuple(StatePort.from_json(port) for port in data.get("state_ports", [])),
            parameter_block=ParameterBlock.from_json(data["parameter_block"]),
            source_file=path,
        )


@dataclass(frozen=True)
class StateAllocation:
    pedal_id: str
    state_id: str
    state_type: str
    static_shape: tuple[int, ...] | None
    elements_per_token: int | None

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "state_id": self.state_id,
            "state_type": self.state_type,
            "static_shape": list(self.static_shape) if self.static_shape is not None else None,
            "elements_per_token": self.elements_per_token,
        }


@dataclass(frozen=True)
class StreamInstance:
    id: str
    state_allocations: tuple[StateAllocation, ...]

    def allocation_for(self, pedal_id: str, state_id: str) -> StateAllocation:
        for allocation in self.state_allocations:
            if allocation.pedal_id == pedal_id and allocation.state_id == state_id:
                return allocation
        raise KeyError(f"no state allocation for {pedal_id}.{state_id}")

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "state_allocations": [allocation.to_json() for allocation in self.state_allocations],
        }


@dataclass(frozen=True)
class Frame:
    id: str
    signal: str
    shape: tuple[int, ...]
    origin: str
    history: tuple[str, ...] = ()

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "signal": self.signal,
            "shape": list(self.shape),
            "origin": self.origin,
            "history": list(self.history),
        }


@dataclass(frozen=True)
class StateUpdate:
    pedal_id: str
    state_id: str
    update: str

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "state_id": self.state_id,
            "update": self.update,
        }


@dataclass(frozen=True)
class PedalStepResult:
    pedal_id: str
    input_frame: Frame
    output_frame: Frame
    state_updates: tuple[StateUpdate, ...]
    events: tuple[Json, ...] = ()

    def to_json(self) -> Json:
        return {
            "pedal_id": self.pedal_id,
            "input_frame": self.input_frame.to_json(),
            "output_frame": self.output_frame.to_json(),
            "state_updates": [update.to_json() for update in self.state_updates],
            "events": list(self.events),
        }


@dataclass(frozen=True)
class RuntimeActivation:
    stream: StreamInstance
    input_frame: Frame
    output_frame: Frame
    steps: tuple[PedalStepResult, ...]

    def to_json(self) -> Json:
        return {
            "stream": self.stream.to_json(),
            "input_frame": self.input_frame.to_json(),
            "output_frame": self.output_frame.to_json(),
            "steps": [step.to_json() for step in self.steps],
        }


@dataclass(frozen=True)
class StreamTick:
    stream_id: str
    tick: int
    status: str
    activation: RuntimeActivation | None
    state_versions: Json
    events: tuple[Json, ...] = ()

    def to_json(self) -> Json:
        return {
            "stream_id": self.stream_id,
            "tick": self.tick,
            "status": self.status,
            "activation": self.activation.to_json() if self.activation is not None else None,
            "state_versions": self.state_versions,
            "events": list(self.events),
        }


class SymbolicPedalExecutor:
    """A no-math executor that proves pedal activation and state plumbing.

    This intentionally does not evaluate transformer operations. It treats each
    pedal as an opaque transition:

        (output_frame, next_state, events) = pedal(input_frame, state, params, control)
    """

    def step(
        self,
        pedal: PedalInstance,
        input_frame: Frame,
        stream: StreamInstance,
        control: Json | None = None,
    ) -> PedalStepResult:
        if input_frame.signal != pedal.input_port.signal:
            raise ValueError(f"{pedal.id} expected signal {pedal.input_port.signal}, got {input_frame.signal}")
        if input_frame.shape != pedal.input_port.shape:
            raise ValueError(f"{pedal.id} expected shape {pedal.input_port.shape}, got {input_frame.shape}")

        state_updates = tuple(
            StateUpdate(
                pedal_id=pedal.id,
                state_id=stream.allocation_for(pedal.id, state_port.id).state_id,
                update="symbolic_transition",
            )
            for state_port in pedal.state_ports
        )
        output_frame = Frame(
            id=f"{input_frame.id}.{pedal.id}",
            signal=pedal.output_port.signal,
            shape=pedal.output_port.shape,
            origin=pedal.id,
            history=input_frame.history + (pedal.id,),
        )
        events: tuple[Json, ...] = ()
        if control:
            events = ({"type": "control_seen", "keys": sorted(control.keys())},)

        return PedalStepResult(
            pedal_id=pedal.id,
            input_frame=input_frame,
            output_frame=output_frame,
            state_updates=state_updates,
            events=events,
        )


@dataclass(frozen=True)
class PedalboardRuntime:
    pedalboard: "Pedalboard"
    executor: SymbolicPedalExecutor

    @classmethod
    def symbolic(cls, pedalboard: "Pedalboard") -> "PedalboardRuntime":
        return cls(pedalboard=pedalboard, executor=SymbolicPedalExecutor())

    def open_stream(self, stream_id: str = "stream_0") -> "PedalStream":
        return PedalStream(runtime=self, stream=self.pedalboard.instantiate_stream(stream_id=stream_id))

    def activate(
        self,
        input_frame: Frame | None = None,
        stream: StreamInstance | None = None,
        control: Json | None = None,
    ) -> RuntimeActivation:
        stream = stream or self.pedalboard.instantiate_stream()
        frame = input_frame or Frame(
            id="frame_0",
            signal=self.pedalboard.pedals[0].input_port.signal,
            shape=self.pedalboard.pedals[0].input_port.shape,
            origin="external_input",
        )
        initial_frame = frame
        steps = []
        for pedal in self.pedalboard.pedals:
            result = self.executor.step(pedal=pedal, input_frame=frame, stream=stream, control=control)
            steps.append(result)
            frame = result.output_frame
        return RuntimeActivation(
            stream=stream,
            input_frame=initial_frame,
            output_frame=frame,
            steps=tuple(steps),
        )


class PedalStream:
    """A stateful, always-on stream around a pedalboard runtime.

    The stream is deliberately small for now: it owns the transient state
    allocation, accepts frames into an input queue, processes at most one frame
    per tick, and tracks symbolic state versions across ticks.
    """

    def __init__(self, runtime: PedalboardRuntime, stream: StreamInstance) -> None:
        self.runtime = runtime
        self.stream = stream
        self.tick_index = 0
        self.input_queue: list[Frame] = []
        self.output_queue: list[Frame] = []
        self.state_versions = {
            f"{allocation.pedal_id}.{allocation.state_id}": 0 for allocation in stream.state_allocations
        }

    def enqueue(self, frame: Frame | None = None, frame_id: str | None = None) -> Frame:
        frame = frame or Frame(
            id=frame_id or f"{self.stream.id}.in_{len(self.input_queue)}",
            signal=self.runtime.pedalboard.pedals[0].input_port.signal,
            shape=self.runtime.pedalboard.pedals[0].input_port.shape,
            origin="stream_input",
        )
        self.input_queue.append(frame)
        return frame

    def tick(self, control: Json | None = None) -> StreamTick:
        tick = self.tick_index
        self.tick_index += 1

        if not self.input_queue:
            return StreamTick(
                stream_id=self.stream.id,
                tick=tick,
                status="idle",
                activation=None,
                state_versions=dict(self.state_versions),
                events=({"type": "no_input"},),
            )

        frame = self.input_queue.pop(0)
        activation = self.runtime.activate(input_frame=frame, stream=self.stream, control=control)
        for step in activation.steps:
            for update in step.state_updates:
                key = f"{update.pedal_id}.{update.state_id}"
                self.state_versions[key] = self.state_versions.get(key, 0) + 1
        self.output_queue.append(activation.output_frame)

        return StreamTick(
            stream_id=self.stream.id,
            tick=tick,
            status="processed",
            activation=activation,
            state_versions=dict(self.state_versions),
            events=(
                {
                    "type": "frame_processed",
                    "input_frame": frame.id,
                    "output_frame": activation.output_frame.id,
                },
            ),
        )

    def run_until_idle(self, control: Json | None = None) -> tuple[StreamTick, ...]:
        ticks = []
        while self.input_queue:
            ticks.append(self.tick(control=control))
        ticks.append(self.tick(control=control))
        return tuple(ticks)


@dataclass(frozen=True)
class Pedalboard:
    root: Path
    model_graph: Json
    pedals: tuple[PedalInstance, ...]

    @classmethod
    def from_dir(cls, root: Path) -> "Pedalboard":
        root = root.resolve()
        model_graph = _read_json(root / "model.json")
        if model_graph["schema"] != "llmoop.model_graph.v1":
            raise ValueError(f"{root / 'model.json'} is not an llmoop model graph")
        board = model_graph["graph"]["pedalboard"]
        if board["wiring"] != "series":
            raise ValueError(f"unsupported wiring mode: {board['wiring']}")

        pedals = []
        for pedal_ref in board["pedals"]:
            pedal = PedalInstance.from_file(root / pedal_ref["file"])
            if pedal.id != pedal_ref["id"]:
                raise ValueError(f"pedal id mismatch: {pedal.source_file}")
            if pedal.pedal_class != pedal_ref["pedal_class"]:
                raise ValueError(f"pedal class mismatch: {pedal.source_file}")
            pedals.append(pedal)

        instance = cls(root=root, model_graph=model_graph, pedals=tuple(pedals))
        instance.validate_series_wiring()
        return instance

    def validate_series_wiring(self) -> None:
        if not self.pedals:
            raise ValueError("pedalboard has no pedals")
        for left, right in zip(self.pedals, self.pedals[1:]):
            if left.output_port.signal != right.input_port.signal:
                raise ValueError(f"signal mismatch: {left.id} -> {right.id}")
            if left.output_port.shape != right.input_port.shape:
                raise ValueError(f"shape mismatch: {left.id} -> {right.id}")

    def instantiate_stream(self, stream_id: str = "stream_0") -> StreamInstance:
        allocations = []
        for pedal in self.pedals:
            for state_port in pedal.state_ports:
                allocations.append(
                    StateAllocation(
                        pedal_id=pedal.id,
                        state_id=state_port.id,
                        state_type=state_port.type,
                        static_shape=state_port.static_shape(),
                        elements_per_token=state_port.elements_per_token(),
                    )
                )
        return StreamInstance(id=stream_id, state_allocations=tuple(allocations))

    def activation_trace(self) -> list[Json]:
        trace = []
        for index, pedal in enumerate(self.pedals):
            trace.append(
                {
                    "index": index,
                    "pedal_id": pedal.id,
                    "pedal_class": pedal.pedal_class,
                    "operator_type": pedal.operator_type,
                    "input": {
                        "signal": pedal.input_port.signal,
                        "shape": list(pedal.input_port.shape),
                    },
                    "output": {
                        "signal": pedal.output_port.signal,
                        "shape": list(pedal.output_port.shape),
                    },
                    "state_ports": [
                        {
                            "id": port.id,
                            "type": port.type,
                            "static_shape": list(port.static_shape()) if port.static_shape() is not None else None,
                            "elements_per_token": port.elements_per_token(),
                        }
                        for port in pedal.state_ports
                    ],
                    "parameter_layout": pedal.parameter_block.layout,
                    "parameter_tensors": len(pedal.parameter_block.tensor_refs),
                }
            )
        return trace

    def summary(self) -> Json:
        counts: dict[str, int] = {}
        for pedal in self.pedals:
            counts[pedal.operator_type] = counts.get(pedal.operator_type, 0) + 1
        stream = self.instantiate_stream()
        return {
            "root": str(self.root),
            "pedal_count": len(self.pedals),
            "operator_counts": counts,
            "input_shape": list(self.pedals[0].input_port.shape),
            "output_shape": list(self.pedals[-1].output_port.shape),
            "stream_state_count": len(stream.state_allocations),
            "stream_state": stream.to_json(),
        }


def _read_json(path: Path) -> Json:
    return json.loads(path.read_text())


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
