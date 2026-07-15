from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmoop.circuit_ir import load_circuit, validate_circuit
from llmoop.pedalboard import Json, SignalPort, StateAllocation, StatePort, StreamInstance


@dataclass(frozen=True)
class CircuitPedal:
    id: str
    operator_type: str
    implementation: str
    behavioral_role: str
    circuit_path: Path
    params_path: Path
    state_path: Path
    circuit: Json
    input_port: SignalPort
    output_port: SignalPort
    state_ports: tuple[StatePort, ...]

    @classmethod
    def from_index_entry(cls, root: Path, entry: Json) -> "CircuitPedal":
        circuit_path = root / entry["circuit"]
        circuit = load_circuit(circuit_path)
        report = validate_circuit(circuit)
        report.raise_for_errors()
        if circuit["source"]["pedal_id"] != entry["id"]:
            raise ValueError(f"circuit pedal id mismatch in {circuit_path}")

        inputs = circuit["boundary"]["inputs"]
        outputs = circuit["boundary"]["outputs"]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{circuit_path} must have exactly one input and one output for series wiring")

        return cls(
            id=entry["id"],
            operator_type=entry["operator_type"],
            implementation=entry["implementation"],
            behavioral_role=entry["behavioral_role"],
            circuit_path=circuit_path,
            params_path=root / entry["params"],
            state_path=root / entry["state"],
            circuit=circuit,
            input_port=SignalPort.from_json(inputs[0]),
            output_port=SignalPort.from_json(outputs[0]),
            state_ports=tuple(StatePort.from_json(port) for port in circuit.get("state_ports", [])),
        )


@dataclass(frozen=True)
class CircuitPedalboard:
    root: Path
    index: Json
    pedals: tuple[CircuitPedal, ...]

    @classmethod
    def from_dir(cls, root: Path) -> "CircuitPedalboard":
        root = root.resolve()
        index_path = root / "pedalboard.circuits.json"
        index = load_circuit(index_path)
        if index["schema"] != "llmoop.lowered_pedalboard.v1":
            raise ValueError(f"{index_path} is not a lowered pedalboard")
        if index["graph"]["wiring"] != "series":
            raise ValueError(f"unsupported circuit wiring mode: {index['graph']['wiring']}")
        pedals = tuple(CircuitPedal.from_index_entry(root, entry) for entry in index["graph"]["circuits"])
        instance = cls(root=root, index=index, pedals=pedals)
        instance.validate_series_wiring()
        return instance

    def validate_series_wiring(self) -> None:
        if not self.pedals:
            raise ValueError("circuit pedalboard has no pedals")
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
                    "operator_type": pedal.operator_type,
                    "implementation": pedal.implementation,
                    "behavioral_role": pedal.behavioral_role,
                    "circuit": str(pedal.circuit_path.relative_to(self.root)),
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
                }
            )
        return trace

    def summary(self) -> Json:
        counts: dict[str, int] = {}
        roles: dict[str, int] = {}
        for pedal in self.pedals:
            counts[pedal.operator_type] = counts.get(pedal.operator_type, 0) + 1
            roles[pedal.behavioral_role] = roles.get(pedal.behavioral_role, 0) + 1
        stream = self.instantiate_stream()
        return {
            "root": str(self.root),
            "circuit_count": len(self.pedals),
            "operator_counts": counts,
            "behavioral_roles": roles,
            "input_shape": list(self.pedals[0].input_port.shape),
            "output_shape": list(self.pedals[-1].output_port.shape),
            "stream_state_count": len(stream.state_allocations),
            "stream_state": stream.to_json(),
        }
