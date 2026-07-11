"""Explicit Qwen3.6 text-core layer schedule."""

from __future__ import annotations

import torch

from .config import CFG
from .ops import (
    apply_rope,
    causal_conv1d,
    gated_delta_rule,
    gdn_gating,
    l2norm,
    linear,
    residual_add,
    rmsnorm,
    sigmoid_mul,
    silu_mul,
)
from .tap import NullTap


def attention_mixer(model, layer, x, positions, start, tap, context) -> torch.Tensor:
    prefix = f"layers.{layer}."
    h = rmsnorm(x, model.weight(prefix + "input_layernorm.weight"))
    qk = linear(h, model.block_weight(prefix + "attn_in.q4"))
    gatev = linear(h, model.block_weight(prefix + "attn_in.q5"))
    q = qk[:, :CFG.q_size].reshape(-1, CFG.q_heads, CFG.head_dim)
    k = qk[:, CFG.q_size:].reshape(-1, CFG.kv_heads, CFG.head_dim)
    gate = gatev[:, :CFG.q_size].reshape(-1, CFG.q_heads, CFG.head_dim)
    v = gatev[:, CFG.q_size:].reshape(-1, CFG.kv_heads, CFG.head_dim)
    q = apply_rope(
        rmsnorm(q, model.weight(prefix + "self_attn.q_norm.weight")), positions
    )
    k = apply_rope(
        rmsnorm(k, model.weight(prefix + "self_attn.k_norm.weight")), positions
    )
    if tap.level == "op":
        for name, value in (("q", q), ("k", k), ("v", v), ("gate", gate)):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", value, **context)
    attended = model._gqa(q, k, v, CFG.full_index(layer), start)
    out = linear(
        sigmoid_mul(gate, attended).reshape(-1, CFG.q_size),
        model.weight(prefix + "self_attn.o_proj.weight"),
    )
    return residual_add(x, out)


def gdn_mixer(model, layer, x, tap, context) -> torch.Tensor:
    _, state = model._ready()
    prefix = f"layers.{layer}."
    index = CFG.gdn_index(layer)
    h = rmsnorm(x, model.weight(prefix + "input_layernorm.weight"))
    qk = linear(h, model.block_weight(prefix + "gdn_in.q4"))
    value = linear(h, model.block_weight(prefix + "gdn_in.q5"))
    qkv = torch.cat((qk[:, :CFG.key_dim], qk[:, CFG.key_dim:], value), dim=-1)
    a = linear(h, model.weight(prefix + "linear_attn.in_proj_a.weight"))
    b = linear(h, model.weight(prefix + "linear_attn.in_proj_b.weight"))
    qkv, state.conv[index] = causal_conv1d(
        qkv,
        model.weight(prefix + "linear_attn.conv1d.weight"),
        state.conv[index],
    )
    g, beta = gdn_gating(
        a,
        b,
        model.weight(prefix + "linear_attn.A_log"),
        model.weight(prefix + "linear_attn.dt_bias"),
    )
    q = l2norm(qkv[:, :CFG.key_dim].reshape(-1, CFG.gdn_k_heads, CFG.gdn_k_dim))
    k = l2norm(
        qkv[:, CFG.key_dim:2 * CFG.key_dim].reshape(
            -1, CFG.gdn_k_heads, CFG.gdn_k_dim
        )
    )
    value = qkv[:, 2 * CFG.key_dim:].reshape(-1, CFG.gdn_v_heads, CFG.gdn_v_dim)
    out, state.ssm[index] = gated_delta_rule(q, k, value, g, beta, state.ssm[index])
    z = linear(h, model.weight(prefix + "linear_attn.in_proj_z.weight")).reshape(
        -1, CFG.gdn_v_heads, CFG.gdn_v_dim
    )
    if tap.level == "op":
        for name, tensor in (("conv", qkv), ("g", g), ("beta", beta), ("gdn", out)):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", tensor, **context)
    out = rmsnorm(
        out,
        model.weight(prefix + "linear_attn.norm.weight"),
        unit_offset=False,
        z=z,
    )
    return residual_add(
        x,
        linear(
            out.reshape(-1, CFG.value_dim),
            model.weight(prefix + "linear_attn.out_proj.weight"),
        ),
    )


def mlp(model, layer, x) -> torch.Tensor:
    prefix = f"layers.{layer}."
    h = rmsnorm(x, model.weight(prefix + "post_attention_layernorm.weight"))
    gate_up = linear(h, model.block_weight(prefix + "mlp.gateup"))
    gate, up = gate_up.split(CFG.intermediate, dim=-1)
    return residual_add(
        x,
        linear(silu_mul(gate, up), model.weight(prefix + "mlp.down_proj.weight")),
    )


def run(
    model,
    x: torch.Tensor,
    positions: torch.Tensor,
    start: int,
    *,
    phase: str,
    step: int,
    chunk: int,
    tap=None,
) -> torch.Tensor:
    tap = tap or NullTap()
    context = dict(phase=phase, step=step, chunk=chunk, position=start)
    for layer in range(CFG.layers):
        x = (
            attention_mixer(model, layer, x, positions, start, tap, context)
            if CFG.is_full(layer)
            else gdn_mixer(model, layer, x, tap, context)
        )
        model._tap(tap, f"layer_{layer:02d}/mixer", x, **context)
        x = mlp(model, layer, x)
        model._tap(tap, f"layer_{layer:02d}/mlp", x, **context)
    return x
