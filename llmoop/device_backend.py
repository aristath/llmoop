from __future__ import annotations

from typing import Any, Protocol

from llmoop.device_loop import DeviceDispatchRun, DeviceOutputEvent, StreamDevice
from llmoop.pedalboard import Json
from llmoop.stream_processor import ControlEvent, RunningStream, StreamProcessor


class DeviceBackend(Protocol):
    """Host/device contract for an installed stream processor."""

    backend_id: str
    device_id: str
    processor: StreamProcessor

    def has_stream(self, stream_id: str) -> bool: ...

    def list_stream_ids(self) -> tuple[str, ...]: ...

    def create_stream(self, stream_id: str, sampler: Any | None = None) -> RunningStream: ...

    def get_stream(self, stream_id: str) -> RunningStream: ...

    def inject_prompt(
        self,
        stream_id: str,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        origin: str = "external_input",
    ) -> None: ...

    def inject_token(
        self,
        stream_id: str,
        token_id: int,
        origin: str = "external_input",
        signal_id: str | None = None,
    ) -> None: ...

    def continue_stream(
        self,
        stream_id: str,
        additional_public_outputs: int = 0,
        reason: str = "device_continue",
    ) -> ControlEvent: ...

    def interrupt(self, stream_id: str, reason: str = "device_interrupt") -> ControlEvent: ...

    def stop_after_current(self, stream_id: str, reason: str = "device_stop_after_current") -> ControlEvent: ...

    def reset_stream(self, stream_id: str, reason: str = "device_reset_state") -> ControlEvent: ...

    def reseed_stream_random(
        self,
        stream_id: str,
        seed: int,
        reason: str = "device_reseed_random",
        source_id: str | None = None,
    ) -> ControlEvent: ...

    def fork_stream(
        self,
        parent_stream_id: str,
        child_stream_id: str,
        policy: str = "clone",
        random_policy: str = "clone",
        random_seed: int | None = None,
    ) -> RunningStream: ...

    def dispatch(self, max_ticks: int) -> DeviceDispatchRun: ...

    def dispatch_until_idle(self, max_ticks: int | None = None) -> DeviceDispatchRun: ...

    def drain_outputs(self) -> tuple[DeviceOutputEvent, ...]: ...

    def to_json(self) -> Json: ...


class PythonDeviceBackend(StreamDevice):
    """Current executable backend.

    This deliberately keeps the existing Python/PyTorch device loop behind the
    same shape a Rust/Vulkan backend will implement.
    """

    backend_id = "python_device_loop"

    def has_stream(self, stream_id: str) -> bool:
        return stream_id in self.streams

    def list_stream_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.streams))

    def to_json(self) -> Json:
        data = super().to_json()
        data["backend_id"] = self.backend_id
        return data
