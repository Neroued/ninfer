"""Numerical contracts for multimodal positions and packed vision attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.q5090.ref.config import CFG, VISION_CFG
from tools.q5090.ref.multimodal import build_mrope_positions
from tools.q5090.ref.ops import apply_rope
from tools.q5090.ref.vision_ops import vision_attention


def test_image_mrope_positions_match_qwen_grid_progression() -> None:
    types = torch.tensor([0, 0, 0] + [1] * 6 + [0, 0])
    positions, delta = build_mrope_positions(types, torch.tensor([[1, 4, 6]]), None)
    expected = torch.tensor(
        [
            [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7],
            [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7],
            [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7],
        ]
    )
    assert torch.equal(positions, expected)
    assert delta == -3


def test_video_grid_splits_one_position_grid_per_temporal_patch() -> None:
    # Two temporal patches, each represented by a separate video placeholder run
    # because the processor inserts a textual timestamp between them.
    types = torch.tensor([0] + [2] * 4 + [0, 0] + [2] * 4 + [0])
    positions, delta = build_mrope_positions(types, None, torch.tensor([[2, 4, 4]]))
    assert positions.shape == (3, len(types))
    assert torch.equal(
        positions[:, 1:5],
        torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]),
    )
    assert torch.equal(
        positions[:, 7:11],
        torch.tensor([[5, 5, 5, 5], [5, 5, 6, 6], [5, 6, 5, 6]]),
    )
    assert delta == -4


def test_interleaved_mrope_reduces_to_text_rope_when_axes_match() -> None:
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(5, CFG.kv_heads, CFG.head_dim, generator=generator).to(torch.bfloat16)
    positions = torch.arange(5, dtype=torch.int32)
    ordinary = apply_rope(x, positions)
    multimodal = apply_rope(x, positions.unsqueeze(0).expand(3, -1))
    assert torch.equal(ordinary, multimodal)


def test_packed_vision_attention_matches_independent_sdpa_segments() -> None:
    generator = torch.Generator().manual_seed(43)
    q = torch.randn(
        12, VISION_CFG.heads, VISION_CFG.head_dim, generator=generator
    ).to(torch.bfloat16)
    k = torch.randn(
        12, VISION_CFG.heads, VISION_CFG.head_dim, generator=generator
    ).to(torch.bfloat16)
    v = torch.randn(
        12, VISION_CFG.heads, VISION_CFG.head_dim, generator=generator
    ).to(torch.bfloat16)
    cu = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    actual = vision_attention(q, k, v, cu)
    expected = []
    for begin, end in ((0, 4), (4, 8), (8, 12)):
        result = F.scaled_dot_product_attention(
            q[begin:end].transpose(0, 1).unsqueeze(0),
            k[begin:end].transpose(0, 1).unsqueeze(0),
            v[begin:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
        )
        expected.append(result.squeeze(0).transpose(0, 1))
    assert torch.equal(actual, torch.cat(expected).to(torch.bfloat16))
