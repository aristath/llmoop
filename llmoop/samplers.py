from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmoop.pedalboard import Json


@dataclass(frozen=True)
class SamplerDecision:
    token_id: int
    sampler_id: str
    implementation: str
    state: Json

    def to_json(self) -> Json:
        return {
            "token_id": self.token_id,
            "sampler_id": self.sampler_id,
            "implementation": self.implementation,
            "state": self.state,
        }


@dataclass(frozen=True)
class GreedySamplerPedal:
    """Deterministic sampler pedal: logits in, highest-probability token out."""

    id: str = "greedy_sampler"
    implementation: str = "argmax_sampler_v1"
    requires_random_signal: bool = False

    def sample(self, logits: Any, torch: Any, random_signal: Any | None = None) -> SamplerDecision:
        last_logits = logits[:, -1, :]
        token_id = int(torch.argmax(last_logits, dim=-1).item())
        return SamplerDecision(
            token_id=token_id,
            sampler_id=self.id,
            implementation=self.implementation,
            state={
                "input_shape": list(logits.shape),
                "selected_axis": "vocab",
                "selection": "argmax",
                "uses_randomness": False,
                "random_signal": random_signal.to_json() if random_signal is not None else None,
            },
        )


@dataclass(frozen=True)
class TemperatureSamplerPedal:
    """Categorical sampler pedal driven by an explicit random signal."""

    id: str = "temperature_sampler"
    temperature: float = 1.0
    top_k: int | None = None
    implementation: str = "explicit_random_categorical_sampler_v1"
    requires_random_signal: bool = True

    def sample(self, logits: Any, torch: Any, random_signal: Any | None = None) -> SamplerDecision:
        if random_signal is None:
            raise ValueError("TemperatureSamplerPedal requires an explicit random_signal")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

        last_logits = logits[:, -1, :].to(dtype=torch.float32) / float(self.temperature)
        candidate_count = int(last_logits.shape[-1])
        top_k = self.top_k
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be > 0")
            top_k = min(int(top_k), candidate_count)
            values, indices = torch.topk(last_logits, k=top_k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            selected_local = _sample_from_probs(probs, random_signal.value, torch)
            token_id = int(indices[0, selected_local].item())
            candidate_count = top_k
        else:
            probs = torch.softmax(last_logits, dim=-1)
            token_id = _sample_from_probs(probs, random_signal.value, torch)

        return SamplerDecision(
            token_id=token_id,
            sampler_id=self.id,
            implementation=self.implementation,
            state={
                "input_shape": list(logits.shape),
                "selected_axis": "vocab",
                "selection": "categorical",
                "temperature": float(self.temperature),
                "top_k": self.top_k,
                "candidate_count": candidate_count,
                "uses_randomness": True,
                "random_signal": random_signal.to_json(),
            },
        )


def _sample_from_probs(probs: Any, random_value: float, torch: Any) -> int:
    cdf = torch.cumsum(probs[0], dim=-1)
    needle = torch.tensor(float(random_value), dtype=cdf.dtype, device=cdf.device)
    selected = int(torch.searchsorted(cdf, needle, right=False).item())
    return min(selected, int(cdf.shape[0]) - 1)
