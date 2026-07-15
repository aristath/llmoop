from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from llmoop.pedalboard import Json
from llmoop.stream_processor import (
    ControlEvent,
    PublicOutputSignal,
    RunningStream,
    RunningStreamTick,
    StreamProcessor,
)


@dataclass(frozen=True)
class DeviceOutputEvent:
    device_id: str
    stream_id: str
    output: PublicOutputSignal
    dispatch_tick: int

    def to_json(self) -> Json:
        return {
            "device_id": self.device_id,
            "stream_id": self.stream_id,
            "dispatch_tick": self.dispatch_tick,
            "output": self.output.to_json(),
        }


@dataclass(frozen=True)
class DeviceDispatchTick:
    device_id: str
    dispatch_tick: int
    stream_id: str
    stream_tick: RunningStreamTick

    def to_json(self) -> Json:
        return {
            "device_id": self.device_id,
            "dispatch_tick": self.dispatch_tick,
            "stream_id": self.stream_id,
            "stream_tick": self.stream_tick.to_json(),
        }


@dataclass(frozen=True)
class DeviceDispatchRun:
    device_id: str
    ticks: tuple[DeviceDispatchTick, ...]
    outputs: tuple[DeviceOutputEvent, ...]
    status: str
    active_streams: tuple[str, ...]

    def to_json(self) -> Json:
        return {
            "device_id": self.device_id,
            "ticks": [tick.to_json() for tick in self.ticks],
            "outputs": [output.to_json() for output in self.outputs],
            "status": self.status,
            "active_streams": list(self.active_streams),
        }


class StreamDevice:
    """Backend-neutral owner of running stream dispatch.

    This is the Python stand-in for the future GPU/device loop. The host
    injects events and control. The device owns the active queue, advances
    streams while feedback is pending, and emits public output events.
    """

    def __init__(self, processor: StreamProcessor, device_id: str = "device_0") -> None:
        self.processor = processor
        self.device_id = device_id
        self.streams: dict[str, RunningStream] = {}
        self.active_queue: deque[str] = deque()
        self.output_queue: list[DeviceOutputEvent] = []
        self.dispatch_tick = 0
        self.events: list[ControlEvent] = []

    def create_stream(
        self,
        stream_id: str,
        sampler: Any | None = None,
    ) -> RunningStream:
        if stream_id in self.streams:
            raise ValueError(f"stream {stream_id!r} already exists")
        stream = self.processor.open_stream(stream_id=stream_id, sampler=sampler)
        self.streams[stream_id] = stream
        self.events.append(
            ControlEvent(
                "device_stream_created",
                {
                    "stream_id": stream_id,
                    "device_id": self.device_id,
                },
            )
        )
        return stream

    def get_stream(self, stream_id: str) -> RunningStream:
        try:
            return self.streams[stream_id]
        except KeyError as exc:
            raise KeyError(f"unknown stream {stream_id!r}") from exc

    def inject_prompt(
        self,
        stream_id: str,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        origin: str = "external_input",
    ) -> None:
        stream = self.get_stream(stream_id)
        stream.inject_prompt(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            origin=origin,
        )
        self._schedule(stream_id)
        self.events.append(
            ControlEvent(
                "device_prompt_injected",
                {
                    "stream_id": stream_id,
                    "prompt_length": len(prompt_ids),
                    "max_new_tokens": max_new_tokens,
                },
            )
        )

    def inject_token(
        self,
        stream_id: str,
        token_id: int,
        origin: str = "external_input",
        signal_id: str | None = None,
    ) -> None:
        stream = self.get_stream(stream_id)
        signal = stream.inject_token(token_id=token_id, origin=origin, signal_id=signal_id)
        self._schedule(stream_id)
        self.events.append(
            ControlEvent(
                "device_token_injected",
                {
                    "stream_id": stream_id,
                    "signal": signal.id,
                    "token_id": signal.token_id,
                },
            )
        )

    def interrupt(self, stream_id: str, reason: str = "device_interrupt") -> ControlEvent:
        stream = self.get_stream(stream_id)
        event = stream.interrupt(reason=reason)
        self._deschedule_if_idle(stream_id)
        self.events.append(event)
        return event

    def continue_stream(
        self,
        stream_id: str,
        additional_public_outputs: int = 0,
        reason: str = "device_continue",
    ) -> ControlEvent:
        stream = self.get_stream(stream_id)
        event = stream.continue_loop(
            additional_public_outputs=additional_public_outputs,
            reason=reason,
        )
        self._schedule_if_active(stream_id)
        self.events.append(event)
        return event

    def stop_after_current(self, stream_id: str, reason: str = "device_stop_after_current") -> ControlEvent:
        stream = self.get_stream(stream_id)
        event = stream.stop_after_current(reason=reason)
        self._schedule_if_active(stream_id)
        self._deschedule_if_idle(stream_id)
        self.events.append(event)
        return event

    def reset_stream(self, stream_id: str, reason: str = "device_reset_state") -> ControlEvent:
        stream = self.get_stream(stream_id)
        event = stream.reset_state(reason=reason)
        self._deschedule_if_idle(stream_id)
        self.events.append(event)
        return event

    def reseed_stream_random(
        self,
        stream_id: str,
        seed: int,
        reason: str = "device_reseed_random",
        source_id: str | None = None,
    ) -> ControlEvent:
        stream = self.get_stream(stream_id)
        event = stream.reseed_random(seed=seed, reason=reason, source_id=source_id)
        self.events.append(event)
        return event

    def fork_stream(
        self,
        parent_stream_id: str,
        child_stream_id: str,
        policy: str = "clone",
        random_policy: str = "clone",
        random_seed: int | None = None,
    ) -> RunningStream:
        if child_stream_id in self.streams:
            raise ValueError(f"stream {child_stream_id!r} already exists")
        parent = self.get_stream(parent_stream_id)
        child = parent.fork(
            stream_id=child_stream_id,
            policy=policy,
            random_policy=random_policy,
            random_seed=random_seed,
        )
        self.streams[child_stream_id] = child
        self._schedule_if_active(child_stream_id)
        self.events.append(
            ControlEvent(
                "device_stream_forked",
                {
                    "parent_stream_id": parent_stream_id,
                    "child_stream_id": child_stream_id,
                    "state_policy": policy,
                    "random_policy": random_policy,
                },
            )
        )
        return child

    def dispatch(self, max_ticks: int) -> DeviceDispatchRun:
        if max_ticks < 0:
            raise ValueError("max_ticks must be >= 0")
        ticks: list[DeviceDispatchTick] = []
        outputs: list[DeviceOutputEvent] = []

        while self.active_queue and len(ticks) < max_ticks:
            stream_id = self.active_queue.popleft()
            stream = self.get_stream(stream_id)
            if not _stream_has_work(stream):
                continue

            stream_tick = stream.tick()
            dispatch_tick = DeviceDispatchTick(
                device_id=self.device_id,
                dispatch_tick=self.dispatch_tick,
                stream_id=stream_id,
                stream_tick=stream_tick,
            )
            self.dispatch_tick += 1
            ticks.append(dispatch_tick)

            if stream_tick.public_output is not None:
                output = DeviceOutputEvent(
                    device_id=self.device_id,
                    stream_id=stream_id,
                    output=stream_tick.public_output,
                    dispatch_tick=dispatch_tick.dispatch_tick,
                )
                self.output_queue.append(output)
                outputs.append(output)

            self._schedule_if_active(stream_id)

        status = "budget_exhausted" if self.active_queue else "idle"
        return DeviceDispatchRun(
            device_id=self.device_id,
            ticks=tuple(ticks),
            outputs=tuple(outputs),
            status=status,
            active_streams=tuple(self.active_queue),
        )

    def dispatch_until_idle(self, max_ticks: int | None = None) -> DeviceDispatchRun:
        ticks: list[DeviceDispatchTick] = []
        outputs: list[DeviceOutputEvent] = []
        status = "idle"
        while self.active_queue:
            remaining = None if max_ticks is None else max_ticks - len(ticks)
            if remaining is not None and remaining <= 0:
                status = "budget_exhausted"
                break
            run = self.dispatch(max_ticks=remaining if remaining is not None else 1_000_000)
            ticks.extend(run.ticks)
            outputs.extend(run.outputs)
            if run.status == "budget_exhausted":
                status = "budget_exhausted"
                break
        return DeviceDispatchRun(
            device_id=self.device_id,
            ticks=tuple(ticks),
            outputs=tuple(outputs),
            status=status,
            active_streams=tuple(self.active_queue),
        )

    def drain_outputs(self) -> tuple[DeviceOutputEvent, ...]:
        outputs = tuple(self.output_queue)
        self.output_queue.clear()
        return outputs

    def to_json(self) -> Json:
        return {
            "device_id": self.device_id,
            "streams": sorted(self.streams),
            "active_streams": list(self.active_queue),
            "pending_outputs": [output.to_json() for output in self.output_queue],
            "dispatch_tick": self.dispatch_tick,
            "events": [event.to_json() for event in self.events],
        }

    def _schedule(self, stream_id: str) -> None:
        if stream_id not in self.active_queue:
            self.active_queue.append(stream_id)

    def _schedule_if_active(self, stream_id: str) -> None:
        if _stream_has_work(self.get_stream(stream_id)):
            self._schedule(stream_id)

    def _deschedule_if_idle(self, stream_id: str) -> None:
        if _stream_has_work(self.get_stream(stream_id)):
            return
        self.active_queue = deque(item for item in self.active_queue if item != stream_id)


def _stream_has_work(stream: RunningStream) -> bool:
    return bool(stream.external_input_queue or stream.private_feedback_queue)
