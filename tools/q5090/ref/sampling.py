"""Small PyTorch sampler for usable reference-model text generation."""

from __future__ import annotations

from dataclasses import dataclass

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


class Sampler:
    def __init__(self, config: SamplingConfig, vocab: int, device: torch.device):
        config.validate()
        self.config = config
        self.counts = torch.zeros(vocab, dtype=torch.int32, device=device)
        self.generator = torch.Generator(device=device).manual_seed(config.seed)

    def __call__(self, logits: torch.Tensor) -> int:
        cfg = self.config
        if cfg.temperature <= 0.0:
            token = int(torch.argmax(logits).item())
        else:
            adjusted = logits.float()
            if cfg.presence_penalty != 0.0 or cfg.frequency_penalty != 0.0:
                adjusted = adjusted - (self.counts > 0) * cfg.presence_penalty
                adjusted = adjusted - self.counts * cfg.frequency_penalty
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
            selected = torch.multinomial(probabilities, 1, generator=self.generator)
            token = int(indices[selected].item())
        self.counts[token] += 1
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
