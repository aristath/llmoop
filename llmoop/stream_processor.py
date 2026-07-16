from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llmoop.circuit_model_runtime import CircuitModelStreamTick
from llmoop.pedalboard import Json
from llmoop.randomness import RandomSource, RandomSourceSnapshot
from llmoop.samplers import GreedySamplerPedal
from llmoop.stream_state import PedalStreamStateSnapshot


@dataclass(frozen=True)
class ControlEvent:
    type: str
    details: Json

    def to_json(self) -> Json:
        return {
            "type": self.type,
            "details": self.details,
        }


@dataclass(frozen=True)
class ExternalInputSignal:
    id: str
    token_id: int
    origin: str = "external_input"
    route: str = "input"

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "origin": self.origin,
            "route": self.route,
        }


@dataclass(frozen=True)
class PublicOutputSignal:
    id: str
    token_id: int
    source_tick: int
    source_model_tick: int
    sampler: Json
    route: str = "public_output"

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "source_tick": self.source_tick,
            "source_model_tick": self.source_model_tick,
            "sampler": self.sampler,
            "route": self.route,
        }


@dataclass(frozen=True)
class PrivateFeedbackSignal:
    id: str
    token_id: int
    source_public_output_id: str
    feedback_depth: int
    closes_loop_after_processing: bool
    stop_reason: str | None = None
    origin: str = "insert_out"
    route: str = "insert_in"

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "source_public_output_id": self.source_public_output_id,
            "feedback_depth": self.feedback_depth,
            "closes_loop_after_processing": self.closes_loop_after_processing,
            "stop_reason": self.stop_reason,
            "origin": self.origin,
            "route": self.route,
        }


InputSignal = ExternalInputSignal | PrivateFeedbackSignal


@dataclass(frozen=True)
class RunningStreamTick:
    stream_id: str
    tick: int
    status: str
    input_signal: InputSignal | None
    model_tick: CircuitModelStreamTick | None
    public_output: PublicOutputSignal | None
    private_feedback: PrivateFeedbackSignal | None
    events: tuple[ControlEvent, ...]

    def to_json(self) -> Json:
        return {
            "stream_id": self.stream_id,
            "tick": self.tick,
            "status": self.status,
            "input_signal": self.input_signal.to_json() if self.input_signal is not None else None,
            "model_tick": self.model_tick.to_json() if self.model_tick is not None else None,
            "public_output": self.public_output.to_json() if self.public_output is not None else None,
            "private_feedback": self.private_feedback.to_json() if self.private_feedback is not None else None,
            "events": [event.to_json() for event in self.events],
        }


@dataclass(frozen=True)
class RunningStreamRun:
    stream_id: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    sampler: str
    stop_reason: str
    ticks: tuple[RunningStreamTick, ...]
    public_outputs: tuple[PublicOutputSignal, ...]
    private_feedback: tuple[PrivateFeedbackSignal, ...]

    def to_json(self) -> Json:
        return {
            "stream_id": self.stream_id,
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "output_ids": list(self.output_ids),
            "sampler": self.sampler,
            "stop_reason": self.stop_reason,
            "ticks": [tick.to_json() for tick in self.ticks],
            "public_outputs": [signal.to_json() for signal in self.public_outputs],
            "private_feedback": [signal.to_json() for signal in self.private_feedback],
        }


@dataclass(frozen=True)
class RunningStreamSnapshot:
    id: str
    source_stream_id: str
    model_position: int
    transient_state: PedalStreamStateSnapshot
    pending_external_inputs: tuple[ExternalInputSignal, ...]
    pending_private_feedback: tuple[PrivateFeedbackSignal, ...]
    public_outputs: tuple[PublicOutputSignal, ...]
    private_feedback_history: tuple[PrivateFeedbackSignal, ...]
    remaining_public_outputs: int
    eos_token_id: int | None
    loop_open: bool
    last_stop_reason: str | None
    random_source: RandomSourceSnapshot
    tick_index: int
    input_counter: int
    public_counter: int
    feedback_counter: int

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "source_stream_id": self.source_stream_id,
            "model_position": self.model_position,
            "transient_state": self.transient_state.to_json(),
            "pending_external_inputs": [signal.to_json() for signal in self.pending_external_inputs],
            "pending_private_feedback": [signal.to_json() for signal in self.pending_private_feedback],
            "public_outputs": [signal.to_json() for signal in self.public_outputs],
            "private_feedback_history": [signal.to_json() for signal in self.private_feedback_history],
            "remaining_public_outputs": self.remaining_public_outputs,
            "eos_token_id": self.eos_token_id,
            "loop_open": self.loop_open,
            "last_stop_reason": self.last_stop_reason,
            "random_source": self.random_source.to_json(),
            "tick_index": self.tick_index,
            "input_counter": self.input_counter,
            "public_counter": self.public_counter,
            "feedback_counter": self.feedback_counter,
        }


class StreamProcessor:
    """Installed permanent stream processor.

    The processor is the stable, shareable circuit. Each `RunningStream` opened
    from it owns its own mutable transient circuit/state.
    """

    def __init__(self, runtime: Any, sampler: Any | None = None, random_seed: int = 0) -> None:
        self.runtime = runtime
        self.sampler = sampler or GreedySamplerPedal()
        self.random_seed = int(random_seed)

    @classmethod
    def from_dirs(
        cls,
        circuit_dir: Path,
        torch: Any | None = None,
        sampler: Any | None = None,
        random_seed: int = 0,
    ) -> "StreamProcessor":
        from llmoop.circuit_model_runtime import CircuitModelRuntime

        runtime = CircuitModelRuntime.from_dirs(circuit_dir=circuit_dir, torch=torch)
        return cls(runtime=runtime, sampler=sampler, random_seed=random_seed)

    def open_stream(
        self,
        stream_id: str = "stream_0",
        sampler: Any | None = None,
        random_source: RandomSource | None = None,
    ) -> "RunningStream":
        return RunningStream(
            processor=self,
            stream_id=stream_id,
            sampler=sampler or self.sampler,
            random_source=random_source,
        )

    def open_stream_from_snapshot(
        self,
        snapshot: RunningStreamSnapshot,
        stream_id: str,
        policy: str = "clone",
        random_policy: str = "clone",
        random_seed: int | None = None,
        sampler: Any | None = None,
    ) -> "RunningStream":
        _require_fork_policy(policy)
        _require_random_policy(random_policy)
        stream = self.open_stream(stream_id=stream_id, sampler=sampler)
        if policy == "clone":
            stream.restore_snapshot(snapshot, reason="fork_clone", preserve_queues=True)
        stream._apply_random_policy(  # noqa: SLF001 - processor constructs stream from a snapshot.
            parent_snapshot=snapshot,
            policy=random_policy,
            random_seed=random_seed,
            stream_id=stream_id,
        )
        return stream

    def generate(
        self,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        stream_id: str = "stream_0",
    ) -> RunningStreamRun:
        return self.open_stream(stream_id=stream_id).generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )


class RunningStream:
    """Executable four-jack stream around the circuit runtime.

    Level 1 still uses token IDs for both public output and private feedback,
    but they are represented as separate signals:

      external input -> model stream -> public output
                                      -> private feedback -> model stream

    The final feedback token is processed before the loop becomes idle, so the
    retained transient state corresponds to the visible output that was emitted.
    """

    def __init__(
        self,
        processor: StreamProcessor,
        stream_id: str = "stream_0",
        sampler: Any | None = None,
        random_source: RandomSource | None = None,
    ) -> None:
        self.processor = processor
        self.runtime = processor.runtime
        self.stream_id = stream_id
        self.sampler = sampler or processor.sampler
        self.random_source = random_source or RandomSource(
            source_id=f"{stream_id}.random",
            seed=processor.random_seed,
        )
        self.model_stream = self.runtime.open_stream()
        self.tick_index = 0
        self.external_input_queue: list[ExternalInputSignal] = []
        self.private_feedback_queue: list[PrivateFeedbackSignal] = []
        self.public_output_queue: list[PublicOutputSignal] = []
        self.private_feedback_history: list[PrivateFeedbackSignal] = []
        self.ticks: list[RunningStreamTick] = []
        self.input_counter = 0
        self.public_counter = 0
        self.feedback_counter = 0
        self.remaining_public_outputs = 0
        self.eos_token_id: int | None = None
        self.loop_open = False
        self.last_stop_reason: str | None = None
        self.pending_control_events: list[ControlEvent] = []

    def inject_token(self, token_id: int, origin: str = "external_input", signal_id: str | None = None) -> ExternalInputSignal:
        signal = ExternalInputSignal(
            id=signal_id or f"input_{self.input_counter}",
            token_id=int(token_id),
            origin=origin,
        )
        if signal_id is None:
            self.input_counter += 1
        self.external_input_queue.append(signal)
        return signal

    def continue_loop(self, additional_public_outputs: int = 0, reason: str = "continue") -> ControlEvent:
        if additional_public_outputs < 0:
            raise ValueError("additional_public_outputs must be >= 0")
        self.remaining_public_outputs += int(additional_public_outputs)
        if self.remaining_public_outputs > 0:
            self.loop_open = True
            self.last_stop_reason = None
        event = ControlEvent(
            "control_continue",
            {
                "reason": reason,
                "additional_public_outputs": int(additional_public_outputs),
                "remaining_public_outputs": self.remaining_public_outputs,
            },
        )
        self.pending_control_events.append(event)
        return event

    def interrupt(self, reason: str = "interrupt") -> ControlEvent:
        cleared_feedback = tuple(signal.id for signal in self.private_feedback_queue)
        self.private_feedback_queue.clear()
        self.remaining_public_outputs = 0
        self.loop_open = False
        self.last_stop_reason = reason
        event = ControlEvent(
            "control_interrupt",
            {
                "reason": reason,
                "cleared_private_feedback": list(cleared_feedback),
                "state_preserved": True,
            },
        )
        self.pending_control_events.append(event)
        return event

    def stop_after_current(self, reason: str = "stop_after_current") -> ControlEvent:
        cleared_feedback: tuple[str, ...] = ()
        closing_feedback_id = None
        if self.private_feedback_queue:
            current = self.private_feedback_queue[0]
            rest = tuple(signal.id for signal in self.private_feedback_queue[1:])
            self.private_feedback_queue = [
                replace(
                    current,
                    closes_loop_after_processing=True,
                    stop_reason=reason,
                )
            ]
            cleared_feedback = rest
            closing_feedback_id = current.id
            self.loop_open = True
        else:
            self.loop_open = False
        self.remaining_public_outputs = 0
        self.last_stop_reason = reason
        event = ControlEvent(
            "control_stop_after_current",
            {
                "reason": reason,
                "closing_private_feedback": closing_feedback_id,
                "cleared_private_feedback": list(cleared_feedback),
                "state_preserved": True,
            },
        )
        self.pending_control_events.append(event)
        return event

    def reset_state(self, reason: str = "reset_state") -> ControlEvent:
        cleared_feedback = tuple(signal.id for signal in self.private_feedback_queue)
        self.model_stream = self.runtime.open_stream()
        self.private_feedback_queue.clear()
        self.remaining_public_outputs = 0
        self.loop_open = False
        self.last_stop_reason = reason
        event = ControlEvent(
            "control_reset_state",
            {
                "reason": reason,
                "cleared_private_feedback": list(cleared_feedback),
                "state_preserved": False,
            },
        )
        self.pending_control_events.append(event)
        return event

    def reseed_random(self, seed: int, reason: str = "reseed_random", source_id: str | None = None) -> ControlEvent:
        self.random_source.reseed(seed=seed, source_id=source_id or f"{self.stream_id}.random")
        event = ControlEvent(
            "control_reseed_random",
            {
                "reason": reason,
                "random_source": self.random_source.to_json(),
            },
        )
        self.pending_control_events.append(event)
        return event

    def snapshot_state(self, snapshot_id: str | None = None) -> RunningStreamSnapshot:
        return RunningStreamSnapshot(
            id=snapshot_id or f"{self.stream_id}.snapshot_{self.tick_index}",
            source_stream_id=self.stream_id,
            model_position=self.model_stream.position,
            transient_state=self.model_stream.state.snapshot(),
            pending_external_inputs=tuple(self.external_input_queue),
            pending_private_feedback=tuple(self.private_feedback_queue),
            public_outputs=tuple(self.public_output_queue),
            private_feedback_history=tuple(self.private_feedback_history),
            remaining_public_outputs=self.remaining_public_outputs,
            eos_token_id=self.eos_token_id,
            loop_open=self.loop_open,
            last_stop_reason=self.last_stop_reason,
            random_source=self.random_source.snapshot(),
            tick_index=self.tick_index,
            input_counter=self.input_counter,
            public_counter=self.public_counter,
            feedback_counter=self.feedback_counter,
        )

    def restore_snapshot(
        self,
        snapshot: RunningStreamSnapshot,
        reason: str = "restore_snapshot",
        preserve_queues: bool = True,
    ) -> ControlEvent:
        self.model_stream = self.runtime.open_stream()
        self.model_stream.state = snapshot.transient_state.restore(self.runtime.torch)
        self.model_stream.position = snapshot.model_position
        if preserve_queues:
            self.external_input_queue = list(snapshot.pending_external_inputs)
            self.private_feedback_queue = list(snapshot.pending_private_feedback)
        else:
            self.external_input_queue.clear()
            self.private_feedback_queue.clear()
        self.public_output_queue = list(snapshot.public_outputs)
        self.private_feedback_history = list(snapshot.private_feedback_history)
        self.remaining_public_outputs = snapshot.remaining_public_outputs
        self.eos_token_id = snapshot.eos_token_id
        self.loop_open = snapshot.loop_open
        self.last_stop_reason = snapshot.last_stop_reason
        self.random_source = snapshot.random_source.restore()
        self.tick_index = snapshot.tick_index
        self.input_counter = snapshot.input_counter
        self.public_counter = snapshot.public_counter
        self.feedback_counter = snapshot.feedback_counter
        event = ControlEvent(
            "control_restore_snapshot",
            {
                "reason": reason,
                "snapshot": snapshot.id,
                "source_stream_id": snapshot.source_stream_id,
                "preserve_queues": preserve_queues,
                "state_preserved": False,
                "state_restored": True,
            },
        )
        self.pending_control_events.append(event)
        return event

    def fork(
        self,
        stream_id: str,
        policy: str = "clone",
        random_policy: str = "clone",
        random_seed: int | None = None,
    ) -> "RunningStream":
        _require_fork_policy(policy)
        _require_random_policy(random_policy)
        if policy == "fresh":
            child = self.processor.open_stream(stream_id=stream_id, sampler=self.sampler)
            child._apply_random_policy(
                parent_snapshot=self.snapshot_state(snapshot_id=f"{self.stream_id}.fork_{stream_id}"),
                policy=random_policy,
                random_seed=random_seed,
                stream_id=stream_id,
            )
            child.pending_control_events.append(
                ControlEvent(
                    "control_fork_fresh",
                    {
                        "parent_stream_id": self.stream_id,
                        "state_policy": policy,
                    },
                )
            )
            return child
        snapshot = self.snapshot_state(snapshot_id=f"{self.stream_id}.fork_{stream_id}")
        child = self.processor.open_stream_from_snapshot(
            snapshot=snapshot,
            stream_id=stream_id,
            policy=policy,
            random_policy=random_policy,
            random_seed=random_seed,
            sampler=self.sampler,
        )
        child.pending_control_events.append(
            ControlEvent(
                "control_fork_clone",
                {
                    "parent_stream_id": self.stream_id,
                    "snapshot": snapshot.id,
                    "state_policy": policy,
                    "random_policy": random_policy,
                    "random_source": child.random_source.to_json(),
                },
            )
        )
        return child

    def inject_prompt(
        self,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        origin: str = "external_input",
    ) -> tuple[ExternalInputSignal, ...]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not prompt_ids:
            bos = self.runtime.config.get("bos_token_id")
            if bos is None:
                raise ValueError("prompt_ids is empty and config has no bos_token_id")
            prompt_ids = (int(bos),)

        self.remaining_public_outputs += int(max_new_tokens)
        self.eos_token_id = eos_token_id
        self.loop_open = max_new_tokens > 0
        self.last_stop_reason = "max_new_tokens" if max_new_tokens == 0 else None
        return tuple(self.inject_token(token_id=token_id, origin=origin) for token_id in prompt_ids)

    def tick(self) -> RunningStreamTick:
        tick = self.tick_index
        self.tick_index += 1
        events = self._drain_pending_control_events()

        input_signal = self._next_input_signal()
        if input_signal is None:
            events.append(ControlEvent("idle", {"reason": "no_external_input_or_feedback"}))
            stream_tick = RunningStreamTick(
                stream_id=self.stream_id,
                tick=tick,
                status="idle",
                input_signal=None,
                model_tick=None,
                public_output=None,
                private_feedback=None,
                events=tuple(events),
            )
            self.ticks.append(stream_tick)
            return stream_tick

        events.append(
            ControlEvent(
                "input_accepted",
                {
                    "signal": input_signal.id,
                    "route": input_signal.route,
                    "origin": input_signal.origin,
                },
            )
        )
        model_tick = self.model_stream.tick(input_signal.token_id)
        public_output = None
        private_feedback = None

        if self.remaining_public_outputs > 0 and not self.external_input_queue:
            public_output, private_feedback = self._emit_public_and_feedback(
                tick=tick,
                model_tick=model_tick,
                source_input=input_signal,
                events=events,
            )

        if (
            isinstance(input_signal, PrivateFeedbackSignal)
            and input_signal.closes_loop_after_processing
        ):
            self.loop_open = False
            self.last_stop_reason = input_signal.stop_reason or self.last_stop_reason or "max_new_tokens"
            events.append(
                ControlEvent(
                    "loop_closed",
                    {
                        "reason": self.last_stop_reason,
                        "processed_feedback": input_signal.id,
                    },
                )
            )

        stream_tick = RunningStreamTick(
            stream_id=self.stream_id,
            tick=tick,
            status="processed",
            input_signal=input_signal,
            model_tick=model_tick,
            public_output=public_output,
            private_feedback=private_feedback,
            events=tuple(events),
        )
        self.ticks.append(stream_tick)
        return stream_tick

    def run_until_idle(self) -> tuple[RunningStreamTick, ...]:
        start = len(self.ticks)
        while self.external_input_queue or self.private_feedback_queue:
            self.tick()
        self.tick()
        return tuple(self.ticks[start:])

    def generate(
        self,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> RunningStreamRun:
        prompt_ids = self._normalize_prompt_ids(prompt_ids)
        start_public = len(self.public_output_queue)
        start_feedback = len(self.private_feedback_history)

        self.inject_prompt(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        ticks = self.run_until_idle()
        public_outputs = tuple(self.public_output_queue[start_public:])
        private_feedback = tuple(self.private_feedback_history[start_feedback:])
        generated_ids = tuple(signal.token_id for signal in public_outputs)
        stop_reason = self.last_stop_reason or "max_new_tokens"

        return RunningStreamRun(
            stream_id=self.stream_id,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            output_ids=prompt_ids + generated_ids,
            sampler=self.sampler.id,
            stop_reason=stop_reason,
            ticks=ticks,
            public_outputs=public_outputs,
            private_feedback=private_feedback,
        )

    def to_json(self) -> Json:
        return {
            "stream_id": self.stream_id,
            "loop_open": self.loop_open,
            "remaining_public_outputs": self.remaining_public_outputs,
            "pending_external_inputs": [signal.to_json() for signal in self.external_input_queue],
            "pending_private_feedback": [signal.to_json() for signal in self.private_feedback_queue],
            "pending_control_events": [event.to_json() for event in self.pending_control_events],
            "random_source": self.random_source.to_json(),
            "public_outputs": [signal.to_json() for signal in self.public_output_queue],
            "private_feedback_history": [signal.to_json() for signal in self.private_feedback_history],
            "transient_state": self.model_stream.state.to_json(),
        }

    def _drain_pending_control_events(self) -> list[ControlEvent]:
        events = list(self.pending_control_events)
        self.pending_control_events.clear()
        return events

    def _next_input_signal(self) -> InputSignal | None:
        if self.external_input_queue:
            return self.external_input_queue.pop(0)
        if self.private_feedback_queue:
            return self.private_feedback_queue.pop(0)
        return None

    def _emit_public_and_feedback(
        self,
        tick: int,
        model_tick: CircuitModelStreamTick,
        source_input: InputSignal,
        events: list[ControlEvent],
    ) -> tuple[PublicOutputSignal, PrivateFeedbackSignal]:
        random_signal = self._next_random_signal()
        decision = self.sampler.sample(
            model_tick.output.logits,
            self.runtime.torch,
            random_signal=random_signal,
        )
        public_output = PublicOutputSignal(
            id=f"public_{self.public_counter}",
            token_id=decision.token_id,
            source_tick=tick,
            source_model_tick=model_tick.tick,
            sampler=decision.to_json(),
        )
        self.public_counter += 1
        self.public_output_queue.append(public_output)
        self.remaining_public_outputs -= 1
        events.append(
            ControlEvent(
                "public_output_emitted",
                {
                    "output": public_output.id,
                    "token_id": public_output.token_id,
                    "route": public_output.route,
                },
            )
        )

        eos_hit = self.eos_token_id is not None and public_output.token_id == int(self.eos_token_id)
        if eos_hit:
            self.remaining_public_outputs = 0
            stop_reason = "eos"
        elif self.remaining_public_outputs == 0:
            stop_reason = "max_new_tokens"
        else:
            stop_reason = None
        close_after_feedback = stop_reason is not None
        if stop_reason is not None:
            self.last_stop_reason = stop_reason

        feedback_depth = source_input.feedback_depth + 1 if isinstance(source_input, PrivateFeedbackSignal) else 1
        private_feedback = PrivateFeedbackSignal(
            id=f"feedback_{self.feedback_counter}",
            token_id=public_output.token_id,
            source_public_output_id=public_output.id,
            feedback_depth=feedback_depth,
            closes_loop_after_processing=close_after_feedback,
            stop_reason=stop_reason,
        )
        self.feedback_counter += 1
        self.private_feedback_queue.append(private_feedback)
        self.private_feedback_history.append(private_feedback)
        events.append(
            ControlEvent(
                "private_feedback_enqueued",
                {
                    "feedback": private_feedback.id,
                    "token_id": private_feedback.token_id,
                    "route": private_feedback.route,
                    "closes_loop_after_processing": private_feedback.closes_loop_after_processing,
                },
            )
        )
        return public_output, private_feedback

    def _next_random_signal(self) -> Any | None:
        if getattr(self.sampler, "requires_random_signal", False):
            return self.random_source.next_signal()
        return None

    def _apply_random_policy(
        self,
        parent_snapshot: RunningStreamSnapshot,
        policy: str,
        random_seed: int | None,
        stream_id: str,
    ) -> None:
        _require_random_policy(policy)
        if policy == "clone":
            self.random_source = parent_snapshot.random_source.restore()
            return
        seed = self.processor.random_seed if random_seed is None else int(random_seed)
        self.random_source = RandomSource(source_id=f"{stream_id}.random", seed=seed)

    def _normalize_prompt_ids(self, prompt_ids: tuple[int, ...]) -> tuple[int, ...]:
        if prompt_ids:
            return tuple(int(token) for token in prompt_ids)
        bos = self.runtime.config.get("bos_token_id")
        if bos is None:
            raise ValueError("prompt_ids is empty and config has no bos_token_id")
        return (int(bos),)


def _require_fork_policy(policy: str) -> None:
    if policy not in {"clone", "fresh"}:
        raise ValueError(f"unsupported stream fork policy {policy!r}")


def _require_random_policy(policy: str) -> None:
    if policy not in {"clone", "fresh"}:
        raise ValueError(f"unsupported random fork policy {policy!r}")
