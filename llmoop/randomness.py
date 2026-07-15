from __future__ import annotations

import hashlib
from dataclasses import dataclass

from llmoop.pedalboard import Json


@dataclass(frozen=True)
class RandomSignal:
    id: str
    source_id: str
    seed: int
    counter: int
    value: float
    route: str = "random_input"

    def to_json(self) -> Json:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "seed": self.seed,
            "counter": self.counter,
            "value": self.value,
            "route": self.route,
        }


@dataclass(frozen=True)
class RandomSourceSnapshot:
    source_id: str
    seed: int
    counter: int

    def restore(self) -> "RandomSource":
        return RandomSource(source_id=self.source_id, seed=self.seed, counter=self.counter)

    def to_json(self) -> Json:
        return {
            "source_id": self.source_id,
            "seed": self.seed,
            "counter": self.counter,
        }


class RandomSource:
    """Explicit deterministic random-signal source.

    The source is intentionally not a global PRNG. Its state is snapshottable,
    forkable, and serializable so stochastic sampling can be replayed as part
    of the stream contract.
    """

    def __init__(self, source_id: str = "random_0", seed: int = 0, counter: int = 0) -> None:
        if counter < 0:
            raise ValueError("counter must be >= 0")
        self.source_id = source_id
        self.seed = int(seed)
        self.counter = int(counter)

    def next_signal(self) -> RandomSignal:
        counter = self.counter
        self.counter += 1
        return RandomSignal(
            id=f"{self.source_id}.{counter}",
            source_id=self.source_id,
            seed=self.seed,
            counter=counter,
            value=_unit_interval(self.seed, counter),
        )

    def snapshot(self) -> RandomSourceSnapshot:
        return RandomSourceSnapshot(
            source_id=self.source_id,
            seed=self.seed,
            counter=self.counter,
        )

    def clone(self, source_id: str | None = None) -> "RandomSource":
        snapshot = self.snapshot()
        return RandomSource(
            source_id=source_id or snapshot.source_id,
            seed=snapshot.seed,
            counter=snapshot.counter,
        )

    def reseed(self, seed: int, source_id: str | None = None) -> None:
        if source_id is not None:
            self.source_id = source_id
        self.seed = int(seed)
        self.counter = 0

    def to_json(self) -> Json:
        return self.snapshot().to_json()


def _unit_interval(seed: int, counter: int) -> float:
    payload = f"llmoop-random-v1:{seed}:{counter}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / 2**64
