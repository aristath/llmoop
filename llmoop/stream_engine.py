from __future__ import annotations

from dataclasses import dataclass

from llmoop.pedalboard import Frame, Json, Pedalboard, PedalboardRuntime, PedalStream, StreamTick


@dataclass(frozen=True)
class TokenPacket:
    id: str
    token_id: int | None
    origin: str
    feedback_depth: int = 0

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "origin": self.origin,
            "feedback_depth": self.feedback_depth,
        }


@dataclass(frozen=True)
class FeedbackPacket:
    id: str
    source_frame_id: str
    signal: str
    shape: tuple[int, ...]
    origin: str
    feedback_depth: int

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "source_frame_id": self.source_frame_id,
            "signal": self.signal,
            "shape": list(self.shape),
            "origin": self.origin,
            "feedback_depth": self.feedback_depth,
        }


@dataclass(frozen=True)
class SymbolicOutputPacket:
    id: str
    source_frame_id: str
    kind: str
    token_id: int | None
    route: str

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "source_frame_id": self.source_frame_id,
            "kind": self.kind,
            "token_id": self.token_id,
            "route": self.route,
        }


@dataclass(frozen=True)
class SymbolicInputTransducer:
    id: str
    kind: str
    tensor_ref: str
    output_signal: str
    output_shape: tuple[int, ...]

    @classmethod
    def from_pedalboard(cls, pedalboard: Pedalboard) -> "SymbolicInputTransducer":
        data = pedalboard.model_graph["graph"]["input_transducer"]
        first_pedal = pedalboard.pedals[0]
        return cls(
            id=data["id"],
            kind=data["type"],
            tensor_ref=data["params"]["weight"]["tensor"],
            output_signal=first_pedal.input_port.signal,
            output_shape=first_pedal.input_port.shape,
        )

    def transduce(self, packet: TokenPacket) -> Frame:
        return Frame(
            id=f"{packet.id}.{self.id}",
            signal=self.output_signal,
            shape=self.output_shape,
            origin=f"input_transducer:{self.id}",
            history=(f"token:{packet.id}", f"input_transducer:{self.id}"),
        )

    def transduce_feedback(self, packet: FeedbackPacket) -> Frame:
        return Frame(
            id=f"{packet.id}.insert_in",
            signal=packet.signal,
            shape=packet.shape,
            origin=f"feedback:{packet.id}",
            history=(f"feedback:{packet.id}", "insert_in"),
        )

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "kind": self.kind,
            "tensor_ref": self.tensor_ref,
            "output_signal": self.output_signal,
            "output_shape": list(self.output_shape),
        }


@dataclass(frozen=True)
class SymbolicOutputTransducer:
    components: tuple[Json, ...]

    @classmethod
    def from_pedalboard(cls, pedalboard: Pedalboard) -> "SymbolicOutputTransducer":
        data = pedalboard.model_graph["graph"]["output_transducer"]
        return cls(components=tuple(data["components"]))

    def transduce(self, frame: Frame, packet_id: str | None = None) -> SymbolicOutputPacket:
        return SymbolicOutputPacket(
            id=packet_id or f"{frame.id}.output",
            source_frame_id=frame.id,
            kind="symbolic_token_distribution",
            token_id=None,
            route="external_output",
        )

    def feedback(self, frame: Frame, packet_id: str, feedback_depth: int) -> FeedbackPacket:
        return FeedbackPacket(
            id=packet_id,
            source_frame_id=frame.id,
            signal=frame.signal,
            shape=frame.shape,
            origin="insert_out",
            feedback_depth=feedback_depth,
        )

    def to_json(self) -> Json:
        return {
            "components": list(self.components),
        }


@dataclass(frozen=True)
class EngineTick:
    tick: int
    status: str
    input_packet: TokenPacket | FeedbackPacket | None
    input_frame: Frame | None
    pedal_tick: StreamTick
    output_packet: SymbolicOutputPacket | None
    feedback_packet: FeedbackPacket | None
    events: tuple[Json, ...] = ()

    def to_json(self) -> Json:
        return {
            "tick": self.tick,
            "status": self.status,
            "input_packet": self.input_packet.to_json() if self.input_packet is not None else None,
            "input_frame": self.input_frame.to_json() if self.input_frame is not None else None,
            "pedal_tick": self.pedal_tick.to_json(),
            "output_packet": self.output_packet.to_json() if self.output_packet is not None else None,
            "feedback_packet": self.feedback_packet.to_json() if self.feedback_packet is not None else None,
            "events": list(self.events),
        }


class SymbolicStreamingEngine:
    """Symbolic full-stream shell: token packet -> frame -> pedals -> output packet.

    This is still intentionally no-math. The transducers name the real model
    boundaries without evaluating embedding lookup, normalization, projection,
    sampling, or token selection.
    """

    def __init__(
        self,
        pedalboard: Pedalboard,
        runtime: PedalboardRuntime,
        pedal_stream: PedalStream,
        input_transducer: SymbolicInputTransducer,
        output_transducer: SymbolicOutputTransducer,
    ) -> None:
        self.pedalboard = pedalboard
        self.runtime = runtime
        self.pedal_stream = pedal_stream
        self.input_transducer = input_transducer
        self.output_transducer = output_transducer
        self.tick_index = 0
        self.input_queue: list[TokenPacket] = []
        self.feedback_queue: list[FeedbackPacket] = []
        self.output_queue: list[SymbolicOutputPacket] = []
        self.token_counter = 0
        self.output_counter = 0
        self.feedback_counter = 0

    @classmethod
    def from_pedalboard(cls, pedalboard: Pedalboard, stream_id: str = "stream_0") -> "SymbolicStreamingEngine":
        runtime = PedalboardRuntime.symbolic(pedalboard)
        return cls(
            pedalboard=pedalboard,
            runtime=runtime,
            pedal_stream=runtime.open_stream(stream_id=stream_id),
            input_transducer=SymbolicInputTransducer.from_pedalboard(pedalboard),
            output_transducer=SymbolicOutputTransducer.from_pedalboard(pedalboard),
        )

    def enqueue_token(
        self,
        token_id: int | None,
        packet_id: str | None = None,
        origin: str = "external_input",
        feedback_depth: int = 0,
    ) -> TokenPacket:
        if packet_id is None:
            packet_id = f"token_{self.token_counter}"
            self.token_counter += 1
        packet = TokenPacket(
            id=packet_id,
            token_id=token_id,
            origin=origin,
            feedback_depth=feedback_depth,
        )
        self.input_queue.append(packet)
        return packet

    def enqueue_feedback(self, packet: FeedbackPacket) -> FeedbackPacket:
        self.feedback_queue.append(packet)
        return packet

    def tick(self, feedback: bool = False, max_feedback_depth: int = 0) -> EngineTick:
        tick = self.tick_index
        self.tick_index += 1

        input_packet: TokenPacket | FeedbackPacket | None = None
        input_frame = None
        events: list[Json] = []

        if self.input_queue:
            input_packet = self.input_queue.pop(0)
            input_frame = self.input_transducer.transduce(input_packet)
            self.pedal_stream.enqueue(input_frame)
            events.append(
                {
                    "type": "external_input_transduced",
                    "token_packet": input_packet.id,
                    "frame": input_frame.id,
                }
            )
        elif self.feedback_queue:
            input_packet = self.feedback_queue.pop(0)
            input_frame = self.input_transducer.transduce_feedback(input_packet)
            self.pedal_stream.enqueue(input_frame)
            events.append(
                {
                    "type": "feedback_input_transduced",
                    "feedback_packet": input_packet.id,
                    "frame": input_frame.id,
                }
            )

        pedal_tick = self.pedal_stream.tick()
        output_packet = None
        feedback_packet = None
        if pedal_tick.activation is not None:
            output_packet = self.output_transducer.transduce(
                pedal_tick.activation.output_frame,
                packet_id=f"public_{self.output_counter}",
            )
            self.output_counter += 1
            self.output_queue.append(output_packet)
            events.append(
                {
                    "type": "public_output_transduced",
                    "frame": output_packet.source_frame_id,
                    "packet": output_packet.id,
                }
            )

            if (
                feedback
                and input_packet is not None
                and input_packet.feedback_depth < max_feedback_depth
            ):
                feedback_packet = self.output_transducer.feedback(
                    pedal_tick.activation.output_frame,
                    packet_id=f"feedback_{self.feedback_counter}",
                    feedback_depth=input_packet.feedback_depth + 1,
                )
                self.feedback_counter += 1
                self.enqueue_feedback(feedback_packet)
                events.append(
                    {
                        "type": "private_feedback_enqueued",
                        "from_insert_out": pedal_tick.activation.output_frame.id,
                        "to": feedback_packet.id,
                    }
                )

        status = "processed" if output_packet is not None else "idle"
        return EngineTick(
            tick=tick,
            status=status,
            input_packet=input_packet,
            input_frame=input_frame,
            pedal_tick=pedal_tick,
            output_packet=output_packet,
            feedback_packet=feedback_packet,
            events=tuple(events),
        )

    def run_until_idle(self, feedback: bool = False, max_feedback_depth: int = 0) -> tuple[EngineTick, ...]:
        ticks = []
        while self.input_queue or self.feedback_queue or self.pedal_stream.input_queue:
            ticks.append(self.tick(feedback=feedback, max_feedback_depth=max_feedback_depth))
        ticks.append(self.tick(feedback=feedback, max_feedback_depth=max_feedback_depth))
        return tuple(ticks)

    def to_json(self) -> Json:
        return {
            "input_transducer": self.input_transducer.to_json(),
            "output_transducer": self.output_transducer.to_json(),
            "pending_inputs": [packet.to_json() for packet in self.input_queue],
            "pending_feedback": [packet.to_json() for packet in self.feedback_queue],
            "outputs": [packet.to_json() for packet in self.output_queue],
        }
