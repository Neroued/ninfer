"""Small PyTorch sampler for usable reference-model text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.0
    frequency_penalty: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0,2]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0,1]")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2,2]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2,2]")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True)
class TruncatedDistribution:
    """A normalized target distribution without sampler-state side effects."""

    indices: torch.Tensor
    probabilities: torch.Tensor

    def __post_init__(self) -> None:
        if self.indices.ndim != 1 or self.probabilities.ndim != 1:
            raise ValueError("distribution indices/probabilities must be one-dimensional")
        if self.indices.numel() == 0 or self.indices.shape != self.probabilities.shape:
            raise ValueError("distribution support must be nonempty and aligned")

    def probability(self, token: int) -> float:
        matches = self.indices == token
        if not bool(matches.any().item()):
            return 0.0
        return float(self.probabilities[matches].sum().item())

    def sample(self, uniform: float, *, exclude: int | None = None) -> int:
        """Sample deterministically from ``uniform`` in [0,1), optionally removing one token."""

        if not 0.0 <= uniform < 1.0:
            raise ValueError("uniform sample must be in [0,1)")
        if exclude is None:
            indices = self.indices
            probabilities = self.probabilities
        else:
            keep = self.indices != exclude
            indices = self.indices[keep]
            probabilities = self.probabilities[keep]
            if indices.numel() == 0:
                raise ValueError("residual distribution has empty support")
            total = probabilities.sum()
            if not bool((total > 0).item()):
                raise ValueError("residual distribution has zero probability")
            probabilities = probabilities / total
        threshold = torch.tensor(uniform, device=probabilities.device)
        selected = int(torch.searchsorted(torch.cumsum(probabilities, dim=0), threshold).item())
        selected = min(selected, indices.numel() - 1)
        return int(indices[selected].item())


class Sampler:
    def __init__(self, config: SamplingConfig, vocab: int, device: torch.device):
        config.validate()
        self.config = config
        self.counts = torch.zeros(vocab, dtype=torch.int32, device=device)
        self.generator = torch.Generator(device=device).manual_seed(config.seed)

    def distribution(
        self, logits: torch.Tensor, *, provisional: Iterable[int] = ()
    ) -> TruncatedDistribution:
        """Build one target distribution without committing occurrence counts."""

        logits = logits.flatten()
        if logits.numel() != self.counts.numel():
            raise ValueError("logit vocabulary does not match sampler vocabulary")
        cfg = self.config
        if cfg.temperature <= 0.0:
            token = torch.argmax(logits).reshape(1)
            return TruncatedDistribution(
                token,
                torch.ones(1, device=logits.device, dtype=torch.float32),
            )

        provisional = list(provisional)
        counts = self.counts
        if provisional:
            counts = counts.clone()
            tokens = torch.tensor(provisional, device=counts.device, dtype=torch.long)
            counts.index_add_(
                0,
                tokens,
                torch.ones_like(tokens, dtype=counts.dtype),
            )
        adjusted = logits.float()
        if cfg.presence_penalty != 0.0 or cfg.frequency_penalty != 0.0:
            adjusted = adjusted - (counts > 0) * cfg.presence_penalty
            adjusted = adjusted - counts * cfg.frequency_penalty
        candidate_count = min(logits.numel(), cfg.top_k if 0 < cfg.top_k < 20 else 20)
        values, indices = torch.topk(adjusted, candidate_count, sorted=True)
        probabilities = torch.softmax(values / cfg.temperature, dim=0)
        if cfg.top_p < 1.0:
            support = int(
                torch.searchsorted(
                    torch.cumsum(probabilities, dim=0),
                    torch.tensor(cfg.top_p, device=logits.device),
                ).item()
            ) + 1
            probabilities = probabilities[:support]
            indices = indices[:support]
            probabilities = probabilities / probabilities.sum()
        return TruncatedDistribution(indices, probabilities)

    def _uniform(self) -> float:
        return float(
            torch.rand((), device=self.counts.device, generator=self.generator).item()
        )

    def accept_draft(
        self,
        distribution: TruncatedDistribution,
        token: int,
        *,
        uniform: float | None = None,
    ) -> bool:
        if uniform is None:
            uniform = self._uniform()
        return uniform < distribution.probability(token)

    def sample_distribution(
        self,
        distribution: TruncatedDistribution,
        *,
        exclude: int | None = None,
        uniform: float | None = None,
    ) -> int:
        if uniform is None:
            uniform = self._uniform()
        return distribution.sample(uniform, exclude=exclude)

    def commit(self, tokens: int | Iterable[int]) -> None:
        if isinstance(tokens, int):
            tokens = [tokens]
        tokens = list(tokens)
        if not tokens:
            return
        indices = torch.tensor(tokens, device=self.counts.device, dtype=torch.long)
        self.counts.index_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=self.counts.dtype),
        )

    def __call__(self, logits: torch.Tensor) -> int:
        distribution = self.distribution(logits)
        token = self.sample_distribution(distribution)
        self.commit(token)
        return token

    def summary(self) -> str:
        cfg = self.config
        if cfg.temperature <= 0.0:
            return "greedy"
        return (
            f"temp={cfg.temperature:g} top_p={cfg.top_p:g} top_k={cfg.top_k} "
            f"presence={cfg.presence_penalty:g} frequency={cfg.frequency_penalty:g} "
            f"seed={cfg.seed}"
        )
