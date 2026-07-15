"""Tensor-only mathematical operators for the 35B-A3B reference target."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .config import ATTN_SCALE, CFG, GDN_SCALE


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    small_t: bool = False,
) -> torch.Tensor:
    """Projection with the target's prefill or Small-T weight boundary."""

    activation = x.to(torch.bfloat16)
    if small_t:
        return bf16(activation.float() @ weight.float().t())
    return bf16(activation @ weight.to(torch.bfloat16).t())


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    unit_offset: bool = True,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Qwen offset RMSNorm, plus the plain gated GDN variant."""

    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + CFG.rms_eps)
    normalized = xf * inv
    if z is None:
        learned = normalized * (weight.float() + (1.0 if unit_offset else 0.0))
        return bf16(learned)

    # The source GDN oracle rounds the normalized activation before applying
    # its learned plain-RMSNorm weight, then combines the result with FP32
    # SiLU(z) before returning to the activation dtype.
    learned = bf16(normalized) * weight.to(torch.bfloat16)
    return bf16(learned.float() * F.silu(z.float()))


def l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return bf16(
        xf
        * torch.rsqrt(
            torch.sum(xf * xf, dim=-1, keepdim=True) + CFG.rms_eps
        )
    )


def residual_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return bf16(x.float() + y.float())


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return bf16(F.silu(gate.float()) * up.float())


def sigmoid_mul(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return bf16(torch.sigmoid(gate.float()) * x.float())


@lru_cache(maxsize=16)
def _rope_frequency(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    pair = torch.arange(CFG.rotary_dim // 2, device=device, dtype=torch.float32)
    return torch.pow(
        torch.tensor(CFG.rope_theta, device=device, dtype=torch.float32),
        -2.0 * pair / CFG.rotary_dim,
    )


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    half = CFG.rotary_dim // 2
    freq = _rope_frequency(x.device.type, x.device.index)
    positions = positions.to(device=x.device, dtype=torch.float32)
    if positions.ndim == 1:
        angle = positions[:, None] * freq[None, :]
    elif positions.ndim == 2 and positions.shape[0] == 3:
        if positions.shape[1] != x.shape[0]:
            raise ValueError("MRoPE position length must match token count")
        axes = positions[:, :, None] * freq[None, None, :]
        angle = axes[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            end = CFG.mrope_section[axis] * 3
            angle[:, offset:end:3] = axes[axis, :, offset:end:3]
    else:
        raise ValueError("RoPE positions must have shape [T] or [3,T]")
    cos = torch.cos(angle)[:, None, :]
    sin = torch.sin(angle)[:, None, :]
    out = x.clone()
    x1 = x[:, :, :half].float()
    x2 = x[:, :, half : CFG.rotary_dim].float()
    out[:, :, :half] = bf16(x1 * cos - x2 * sin)
    out[:, :, half : CFG.rotary_dim] = bf16(x2 * cos + x1 * sin)
    return out


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Grouped-query attention over tensors in ``[T,H,D]`` layout."""

    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    mask = None
    if causal and q.shape[0] != k.shape[0]:
        from torch.nn.attention.bias import causal_lower_right

        mask = causal_lower_right(q.shape[0], k.shape[0])
        causal = False
    out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=ATTN_SCALE,
        enable_gqa=True,
    )
    return bf16(out.squeeze(0).transpose(0, 1))


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise convolution and SiLU with BF16 history state."""

    w = weight.contiguous().float()
    width = w.shape[1]
    sequence = torch.cat((state.float().t(), x.float()), dim=0)
    out = F.conv1d(
        sequence.t().unsqueeze(0),
        w.unsqueeze(1),
        groups=x.shape[1],
    ).squeeze(0).t()
    next_state = sequence[-(width - 1) :].t().contiguous().to(torch.bfloat16)
    return bf16(F.silu(out)), next_state


def gdn_gating(
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce FP32 decay and update controls after BF16 A/B projections."""

    u = a.float() + dt_bias.float()
    softplus = torch.where(u > 20.0, u, F.softplus(u))
    return -torch.exp(a_log.float()) * softplus, torch.sigmoid(b.float())


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA GDN over ``[T,H,K/V]`` inputs and an FP32 recurrent state."""

    if q.shape[0] == 1:
        head_map = torch.arange(CFG.gdn_v_heads, device=q.device) // (
            CFG.gdn_v_heads // CFG.gdn_k_heads
        )
        kt = k[0].float().index_select(0, head_map)
        qt = q[0].float().index_select(0, head_map)
        next_state = state.float() * torch.exp(g[0].float()).view(
            1, CFG.gdn_v_heads, 1, 1
        )
        prediction = torch.einsum("bhkv,hk->bhv", next_state, kt)
        delta = beta[0].float().view(1, CFG.gdn_v_heads, 1) * (
            v[0].float() - prediction
        )
        next_state = next_state + kt.view(
            1, CFG.gdn_v_heads, CFG.gdn_k_dim, 1
        ) * delta.unsqueeze(-2)
        out = torch.einsum("bhkv,hk->bhv", next_state, qt) * GDN_SCALE
        return bf16(out.squeeze(0).unsqueeze(0)), next_state

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3.6-35B-A3B reference requires flash-linear-attention>=0.5.1 "
            "for prefill GDN"
        ) from exc
    q, k, v, g, beta = (tensor.clone() for tensor in (q, k, v, g, beta))
    out, final = chunk_gated_delta_rule(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        scale=GDN_SCALE,
        initial_state=state,
        output_final_state=True,
    )
    return bf16(out.squeeze(0)), final.float()


def _naive_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential small-shape oracle; never used by the model schedule."""

    s = state[0].float().clone()
    out = torch.empty_like(v, dtype=torch.float32)
    head_map = torch.arange(CFG.gdn_v_heads, device=q.device) // (
        CFG.gdn_v_heads // CFG.gdn_k_heads
    )
    for token in range(q.shape[0]):
        kt = k[token].float().index_select(0, head_map)
        qt = q[token].float().index_select(0, head_map)
        s.mul_(torch.exp(g[token].float()).view(CFG.gdn_v_heads, 1, 1))
        prediction = torch.einsum("hkv,hk->hv", s, kt)
        delta = beta[token].float().unsqueeze(-1) * (
            v[token].float() - prediction
        )
        s.add_(kt.unsqueeze(-1) * delta.unsqueeze(-2))
        out[token] = torch.einsum("hkv,hk->hv", s, qt) * GDN_SCALE
    return bf16(out), s.unsqueeze(0)


__all__ = [
    "_naive_gated_delta_rule",
    "apply_rope",
    "attention",
    "bf16",
    "causal_conv1d",
    "gated_delta_rule",
    "gdn_gating",
    "l2norm",
    "linear",
    "residual_add",
    "rmsnorm",
    "sigmoid_mul",
    "silu_mul",
]
