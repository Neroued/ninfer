"""User-visible chat rendering and sampling contracts."""

from __future__ import annotations

import torch

from tools.q5090.ref.chat import render_prompt
from tools.q5090.ref.sampling import Sampler, SamplingConfig


def test_single_turn_chat_rendering_matches_qwen_modes() -> None:
    prefix = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert render_prompt("你好", thinking=True) == prefix
    assert render_prompt("你好", thinking=False) == prefix + "\n</think>\n\n"


def test_sampler_greedy_and_top_k_contracts() -> None:
    logits = torch.tensor([-3.0, 1.0, 4.0, 2.0])
    greedy = Sampler(SamplingConfig(temperature=0.0), logits.numel(), logits.device)
    assert greedy(logits) == 2

    sampled = Sampler(
        SamplingConfig(
            temperature=1.0,
            top_p=1.0,
            top_k=2,
            presence_penalty=0.0,
            seed=17,
        ),
        logits.numel(),
        logits.device,
    )
    assert {sampled(logits) for _ in range(32)} <= {2, 3}
