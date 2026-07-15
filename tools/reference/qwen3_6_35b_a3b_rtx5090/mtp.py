"""One-layer full-attention sparse-MoE MTP schedule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .config import CFG
from .moe import forward as sparse_moe
from .ops import apply_rope, linear, residual_add, rmsnorm, sigmoid_mul


@dataclass
class MtpStats:
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    fallback_steps: int = 0
    accepted_per_pos: list[int] = field(default_factory=lambda: [0] * 5)

    def record_round(self, drafted: int, accepted: int) -> None:
        if not 0 <= accepted <= drafted <= len(self.accepted_per_pos):
            raise ValueError("invalid MTP round counts")
        self.rounds += 1
        self.draft_tokens += drafted
        self.accepted_tokens += accepted
        for position in range(accepted):
            self.accepted_per_pos[position] += 1

    def record_fallback(self) -> None:
        self.fallback_steps += 1


def truncate_at_stop(
    tokens: Iterable[int],
    stop_token_ids: set[int] | None,
) -> list[int]:
    output: list[int] = []
    for token in tokens:
        output.append(token)
        if stop_token_ids and token in stop_token_ids:
            break
    return output


def forward(
    model,
    ids: Iterable[int],
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    start: int,
    small_t: bool,
    sample: bool = True,
    input_embeddings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int | None]:
    if not model.mtp_enabled:
        raise RuntimeError("MTP is not enabled")
    ids = list(ids)
    mtp_weights = model.binding.mtp
    layer_weights = mtp_weights.layer
    attention_weights = layer_weights.attention

    embeddings = model.embed(ids) if input_embeddings is None else input_embeddings
    embedding_input = rmsnorm(
        embeddings,
        model.weight(mtp_weights.embedding_norm),
    )
    hidden_input = rmsnorm(hidden, model.weight(mtp_weights.hidden_norm))
    x = linear(
        torch.cat((embedding_input, hidden_input), dim=-1),
        model.weight(mtp_weights.input_projection, small_t=small_t),
        small_t=small_t,
    )

    h = rmsnorm(x, model.weight(layer_weights.input_norm))
    projected = linear(
        h,
        model.block_weight(
            attention_weights.query_key_gate_value,
            small_t=small_t,
        ),
        small_t=small_t,
    )
    q1 = CFG.q_size
    k1 = q1 + CFG.kv_size
    gate1 = k1 + CFG.q_size
    q = projected[:, :q1].reshape(-1, CFG.q_heads, CFG.head_dim)
    k = projected[:, q1:k1].reshape(-1, CFG.kv_heads, CFG.head_dim)
    gate = projected[:, k1:gate1].reshape(-1, CFG.q_heads, CFG.head_dim)
    value = projected[:, gate1:].reshape(-1, CFG.kv_heads, CFG.head_dim)
    q = apply_rope(
        rmsnorm(q, model.weight(attention_weights.query_norm)),
        positions,
    )
    k = apply_rope(
        rmsnorm(k, model.weight(attention_weights.key_norm)),
        positions,
    )
    attended = model._gqa(q, k, value, 0, start, mtp=True)
    x = residual_add(
        x,
        linear(
            sigmoid_mul(gate, attended).reshape(-1, CFG.q_size),
            model.weight(attention_weights.output, small_t=small_t),
            small_t=small_t,
        ),
    )

    h = rmsnorm(x, model.weight(layer_weights.post_attention_norm))
    x = residual_add(
        x,
        sparse_moe(model, layer_weights.moe, h, small_t=small_t).output,
    )
    output = rmsnorm(x, model.weight(mtp_weights.final_norm))
    token = (
        int(
            torch.argmax(
                model.logits_last(output, draft=model.draft_head)[: CFG.token_domain]
            ).item()
        )
        if sample
        else None
    )
    _, state = model._ready()
    state.mtp_kv.length = start + len(ids)
    return output, token


__all__ = ["MtpStats", "forward", "truncate_at_stop"]
