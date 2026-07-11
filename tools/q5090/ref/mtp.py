"""Explicit one-layer Qwen3.6 MTP schedule."""

from __future__ import annotations

from typing import Iterable

import torch

from .config import CFG
from .ops import apply_rope, linear, residual_add, rmsnorm, sigmoid_mul, silu_mul


def forward(
    model,
    ids: Iterable[int],
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    start: int,
    sample: bool = True,
) -> tuple[torch.Tensor, int | None]:
    if not model.mtp_enabled:
        raise RuntimeError("MTP is not enabled")
    ids = list(ids)
    emb = rmsnorm(model.embed(ids), model.weight("mtp.pre_fc_norm_embedding.weight"))
    previous = rmsnorm(hidden, model.weight("mtp.pre_fc_norm_hidden.weight"))
    x = linear(torch.cat((emb, previous), dim=-1), model.weight("mtp.fc.weight"))
    h = rmsnorm(x, model.weight("mtp.layers.0.input_layernorm.weight"))
    qk_gatev = linear(h, model.block_weight("mtp.layers.0.attn_in.w8"))
    q = qk_gatev[:, :CFG.q_size].reshape(-1, CFG.q_heads, CFG.head_dim)
    k0 = CFG.q_size
    k1 = k0 + CFG.kv_size
    g1 = k1 + CFG.q_size
    k = qk_gatev[:, k0:k1].reshape(-1, CFG.kv_heads, CFG.head_dim)
    gate = qk_gatev[:, k1:g1].reshape(-1, CFG.q_heads, CFG.head_dim)
    v = qk_gatev[:, g1:].reshape(-1, CFG.kv_heads, CFG.head_dim)
    q = apply_rope(
        rmsnorm(q, model.weight("mtp.layers.0.self_attn.q_norm.weight")), positions
    )
    k = apply_rope(
        rmsnorm(k, model.weight("mtp.layers.0.self_attn.k_norm.weight")), positions
    )
    attended = model._gqa(q, k, v, 0, start, mtp=True)
    x = residual_add(
        x,
        linear(
            sigmoid_mul(gate, attended).reshape(-1, CFG.q_size),
            model.weight("mtp.layers.0.self_attn.o_proj.weight"),
        ),
    )
    h = rmsnorm(x, model.weight("mtp.layers.0.post_attention_layernorm.weight"))
    gate_up = linear(h, model.block_weight("mtp.layers.0.mlp.gateup.w8"))
    gate, up = gate_up.split(CFG.intermediate, dim=-1)
    x = residual_add(
        x,
        linear(silu_mul(gate, up), model.weight("mtp.layers.0.mlp.down_proj.weight")),
    )
    out = rmsnorm(x, model.weight("mtp.norm.weight"))
    token = (
        int(torch.argmax(model.logits_last(out, draft=model.draft_head)).item())
        if sample
        else None
    )
    _, state = model._ready()
    state.mtp_kv.length = start + len(ids)
    return out, token
