from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.circuit_model_runtime import CircuitModelRuntime
from llmoop.device_backend import DeviceBackend, PythonDeviceBackend
from llmoop.device_loop import DeviceDispatchRun, DeviceOutputEvent
from llmoop.pedalboard import Json
from llmoop.stream_processor import ControlEvent, RunningStream, StreamProcessor


@dataclass(frozen=True)
class InstalledPromptRun:
    """Host-visible result of one prompt event through an installed processor."""

    stream_id: str
    prompt_ids: tuple[int, ...]
    dispatch: DeviceDispatchRun
    outputs: tuple[DeviceOutputEvent, ...]
    sampler: str
    stop_reason: str

    @property
    def generated_ids(self) -> tuple[int, ...]:
        return tuple(output.output.token_id for output in self.outputs)

    @property
    def output_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.generated_ids

    def to_json(self) -> Json:
        return {
            "stream_id": self.stream_id,
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "output_ids": list(self.output_ids),
            "sampler": self.sampler,
            "stop_reason": self.stop_reason,
            "dispatch": self.dispatch.to_json(),
        }


class InstalledStreamProcessor:
    """Installed stream-circuit façade.

    This is the runtime object the host talks to. It owns the permanent
    processor and a device loop; streams own transient state inside the device.
    The host injects input/control and drains public output, but does not tick
    the private feedback loop directly.
    """

    def __init__(
        self,
        processor: StreamProcessor,
        device: DeviceBackend | None = None,
        install_id: str = "installed_stream_processor",
    ) -> None:
        self.processor = processor
        self.device = device or PythonDeviceBackend(processor=processor)
        self.install_id = install_id

    @classmethod
    def from_runtime(
        cls,
        runtime: CircuitModelRuntime,
        sampler: Any | None = None,
        random_seed: int = 0,
        device_id: str = "device_0",
        install_id: str = "installed_stream_processor",
        backend: str = "python_device_loop",
    ) -> "InstalledStreamProcessor":
        if backend != PythonDeviceBackend.backend_id:
            raise ValueError(f"unsupported Python prototype backend {backend!r}")
        processor = StreamProcessor(runtime=runtime, sampler=sampler, random_seed=random_seed)
        return cls(
            processor=processor,
            device=PythonDeviceBackend(processor=processor, device_id=device_id),
            install_id=install_id,
        )

    @classmethod
    def from_dirs(
        cls,
        circuit_dir: Path,
        model_dir: Path | None = None,
        torch: Any | None = None,
        sampler: Any | None = None,
        random_seed: int = 0,
        device_id: str = "device_0",
        install_id: str = "installed_stream_processor",
        backend: str = "python_device_loop",
    ) -> "InstalledStreamProcessor":
        runtime = CircuitModelRuntime.from_dirs(circuit_dir=circuit_dir, model_dir=model_dir, torch=torch)
        return cls.from_runtime(
            runtime=runtime,
            sampler=sampler,
            random_seed=random_seed,
            device_id=device_id,
            install_id=install_id,
            backend=backend,
        )

    def open_stream(self, stream_id: str, sampler: Any | None = None) -> RunningStream:
        return self.device.create_stream(stream_id=stream_id, sampler=sampler)

    def get_stream(self, stream_id: str) -> RunningStream:
        return self.device.get_stream(stream_id)

    def inject_prompt(
        self,
        stream_id: str,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        origin: str = "external_input",
        create: bool = True,
    ) -> None:
        if create and not self.device.has_stream(stream_id):
            self.open_stream(stream_id)
        self.device.inject_prompt(
            stream_id=stream_id,
            prompt_ids=tuple(int(token) for token in prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            origin=origin,
        )

    def inject_token(
        self,
        stream_id: str,
        token_id: int,
        origin: str = "external_input",
        signal_id: str | None = None,
        create: bool = True,
    ) -> None:
        if create and not self.device.has_stream(stream_id):
            self.open_stream(stream_id)
        self.device.inject_token(
            stream_id=stream_id,
            token_id=token_id,
            origin=origin,
            signal_id=signal_id,
        )

    def continue_stream(
        self,
        stream_id: str,
        additional_public_outputs: int = 0,
        reason: str = "host_continue",
    ) -> ControlEvent:
        return self.device.continue_stream(
            stream_id=stream_id,
            additional_public_outputs=additional_public_outputs,
            reason=reason,
        )

    def interrupt(self, stream_id: str, reason: str = "host_interrupt") -> ControlEvent:
        return self.device.interrupt(stream_id=stream_id, reason=reason)

    def stop_after_current(self, stream_id: str, reason: str = "host_stop_after_current") -> ControlEvent:
        return self.device.stop_after_current(stream_id=stream_id, reason=reason)

    def reset_stream(self, stream_id: str, reason: str = "host_reset_state") -> ControlEvent:
        return self.device.reset_stream(stream_id=stream_id, reason=reason)

    def reseed_stream_random(
        self,
        stream_id: str,
        seed: int,
        reason: str = "host_reseed_random",
        source_id: str | None = None,
    ) -> ControlEvent:
        return self.device.reseed_stream_random(
            stream_id=stream_id,
            seed=seed,
            reason=reason,
            source_id=source_id,
        )

    def fork_stream(
        self,
        parent_stream_id: str,
        child_stream_id: str,
        policy: str = "clone",
        random_policy: str = "clone",
        random_seed: int | None = None,
    ) -> RunningStream:
        return self.device.fork_stream(
            parent_stream_id=parent_stream_id,
            child_stream_id=child_stream_id,
            policy=policy,
            random_policy=random_policy,
            random_seed=random_seed,
        )

    def dispatch(self, max_ticks: int) -> DeviceDispatchRun:
        return self.device.dispatch(max_ticks=max_ticks)

    def run_until_idle(self, max_ticks: int | None = None) -> DeviceDispatchRun:
        return self.device.dispatch_until_idle(max_ticks=max_ticks)

    def drain_outputs(self) -> tuple[DeviceOutputEvent, ...]:
        return self.device.drain_outputs()

    def run_prompt(
        self,
        stream_id: str,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        origin: str = "external_input",
        max_ticks: int | None = None,
    ) -> InstalledPromptRun:
        normalized_prompt_ids = _normalize_prompt_ids(self.processor.runtime, prompt_ids)
        self.inject_prompt(
            stream_id=stream_id,
            prompt_ids=normalized_prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            origin=origin,
            create=True,
        )
        stream = self.get_stream(stream_id)
        dispatch = self.run_until_idle(max_ticks=max_ticks)
        outputs = tuple(
            event
            for event in dispatch.outputs
            if event.stream_id == stream_id
        )
        return InstalledPromptRun(
            stream_id=stream_id,
            prompt_ids=normalized_prompt_ids,
            dispatch=dispatch,
            outputs=outputs,
            sampler=stream.sampler.id,
            stop_reason=stream.last_stop_reason or ("budget_exhausted" if dispatch.status == "budget_exhausted" else "max_new_tokens"),
        )

    def to_json(self) -> Json:
        board = self.processor.runtime.board
        stream_state = board.instantiate_stream(stream_id="stream_template")
        return {
            "install_id": self.install_id,
            "backend": self.device.backend_id,
            "permanent_circuit": {
                "pedal_count": len(board.pedals),
                "input_signal": board.pedals[0].input_port.signal,
                "output_signal": board.pedals[-1].output_port.signal,
                "source_model_dir": str(self.processor.runtime.board.index["source"]["source_model_dir"]),
            },
            "host_ports": {
                "inputs": ["external_input", "control", "random_input"],
                "outputs": ["public_output", "events"],
                "private_feedback": "device_owned_insert_loop",
            },
            "stream_template": stream_state.to_json(),
            "device": self.device.to_json(),
        }


def _normalize_prompt_ids(runtime: CircuitModelRuntime, prompt_ids: tuple[int, ...]) -> tuple[int, ...]:
    if prompt_ids:
        return tuple(int(token) for token in prompt_ids)
    bos = runtime.config.get("bos_token_id")
    if bos is None:
        raise ValueError("prompt_ids is empty and config has no bos_token_id")
    return (int(bos),)
