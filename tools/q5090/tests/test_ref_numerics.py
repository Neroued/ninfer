"""Numerical contracts for the library-backed reference operators."""

from __future__ import annotations

import numpy as np
import torch

from tools.q5090.ref.config import ATTN_SCALE, CFG
from tools.q5090.ref.ops import (
    _naive_gated_delta_rule,
    attention,
    causal_conv1d,
    gated_delta_rule,
)
from tools.q5090.ref.state import KVCache


def test_causal_conv_matches_direct_real_shape() -> None:
    generator = torch.Generator().manual_seed(7)
    tokens = 7
    x = torch.randn(tokens, CFG.conv_dim, generator=generator, dtype=torch.bfloat16)
    state = torch.randn(
        CFG.conv_dim, CFG.conv_width - 1, generator=generator, dtype=torch.float32
    )
    tap_major = torch.randn(
        CFG.conv_width, CFG.conv_dim, generator=generator, dtype=torch.bfloat16
    )
    runtime_weight = tap_major.reshape(CFG.conv_dim, CFG.conv_width, 1)
    actual, next_state = causal_conv1d(x, runtime_weight, state)
    sequence = torch.cat((state.t(), x.float()), dim=0)
    expected = []
    for token in range(tokens):
        window = sequence[token:token + CFG.conv_width]
        expected.append(torch.nn.functional.silu((window * tap_major.float()).sum(dim=0)))
    expected = torch.stack(expected).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), atol=6.25e-2, rtol=1e-2)
    assert torch.equal(next_state, sequence[-(CFG.conv_width - 1):].t())


def test_int8_kv_codec_matches_fp16_scale_contract() -> None:
    generator = torch.Generator().manual_seed(11)
    value = torch.randn(
        3, CFG.kv_heads, CFG.head_dim, generator=generator, dtype=torch.bfloat16
    )
    code, scale = KVCache._quantize(value)
    source = value.float().numpy().reshape(3, CFG.kv_heads, CFG.head_dim // 64, 64)
    expected_scale = (np.max(np.abs(source), axis=-1) / np.float32(127.0)).astype(np.float16)
    safe = np.where(expected_scale == 0, np.float16(1), expected_scale).astype(np.float32)
    expected_code = np.rint(source / safe[..., None]).clip(-127, 127).astype(np.int8)
    expected_code[expected_scale == 0] = 0
    assert np.array_equal(scale.numpy().view(np.uint16), expected_scale.view(np.uint16))
    assert np.array_equal(code.numpy(), expected_code.reshape(code.shape))


def test_t1_gdn_matches_sequential_oracle_real_shape() -> None:
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(1, CFG.gdn_k_heads, CFG.gdn_k_dim, generator=generator).to(torch.bfloat16)
    k = torch.randn(1, CFG.gdn_k_heads, CFG.gdn_k_dim, generator=generator).to(torch.bfloat16)
    v = torch.randn(1, CFG.gdn_v_heads, CFG.gdn_v_dim, generator=generator).to(torch.bfloat16)
    g = -torch.rand(1, CFG.gdn_v_heads, generator=generator) * 3
    beta = torch.rand(1, CFG.gdn_v_heads, generator=generator)
    state = torch.randn(
        1, CFG.gdn_v_heads, CFG.gdn_k_dim, CFG.gdn_v_dim, generator=generator
    ) * 0.01
    actual, actual_state = gated_delta_rule(q, k, v, g, beta, state)
    expected, expected_state = _naive_gated_delta_rule(q, k, v, g, beta, state)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_state, expected_state)


def test_sdpa_gqa_matches_explicit_math_real_heads() -> None:
    generator = torch.Generator().manual_seed(23)
    tq, tk = 3, 11
    q = torch.randn(tq, CFG.q_heads, CFG.head_dim, generator=generator).to(torch.bfloat16)
    k = torch.randn(tk, CFG.kv_heads, CFG.head_dim, generator=generator).to(torch.bfloat16)
    v = torch.randn(tk, CFG.kv_heads, CFG.head_dim, generator=generator).to(torch.bfloat16)
    actual = attention(q, k, v, causal=True).float()
    repeated_k = k.repeat_interleave(CFG.q_heads // CFG.kv_heads, dim=1).float()
    repeated_v = v.repeat_interleave(CFG.q_heads // CFG.kv_heads, dim=1).float()
    scores = torch.einsum("thd,shd->ths", q.float(), repeated_k) * ATTN_SCALE
    query = torch.arange(tq)
    key = torch.arange(tk)
    scores.masked_fill_(key[None, None, :] > query[:, None, None] + tk - tq, -torch.inf)
    expected = torch.einsum("ths,shd->thd", torch.softmax(scores, dim=-1), repeated_v)
    expected = expected.to(torch.bfloat16).float()
    torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
