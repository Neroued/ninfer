from __future__ import annotations

import torch

from tools.reference.qwen3_6_27b_rtx5090.sampling import (
    Sampler,
    SamplingConfig,
    TruncatedDistribution,
)


def test_provisional_penalty_does_not_commit_counts() -> None:
    sampler = Sampler(
        SamplingConfig(
            temperature=1.0,
            top_p=1.0,
            top_k=3,
            presence_penalty=2.0,
            frequency_penalty=0.0,
        ),
        token_domain=3,
        device=torch.device("cpu"),
    )
    logits = torch.tensor([2.0, 1.0, 0.0])
    base = sampler.distribution(logits)
    provisional = sampler.distribution(logits, provisional=[0])

    assert provisional.probability(0) < base.probability(0)
    assert sampler.counts.tolist() == [0, 0, 0]

    sampler.commit([2, 2])
    assert sampler.counts.tolist() == [0, 0, 2]


def test_greedy_call_is_argmax_and_commits_only_selected_token() -> None:
    sampler = Sampler(
        SamplingConfig(temperature=0.0, presence_penalty=2.0),
        token_domain=4,
        device=torch.device("cpu"),
    )
    sampler.commit([3, 3])
    token = sampler(torch.tensor([0.0, 4.0, 2.0, 5.0]))

    assert token == 3
    assert sampler.counts.tolist() == [0, 0, 0, 3]


def test_one_hot_draft_acceptance_and_rejection_residual() -> None:
    distribution = TruncatedDistribution(
        indices=torch.tensor([10, 11, 12]),
        probabilities=torch.tensor([0.25, 0.5, 0.25]),
    )
    sampler = Sampler(
        SamplingConfig(temperature=1.0),
        token_domain=16,
        device=torch.device("cpu"),
    )

    assert sampler.accept_draft(distribution, 10, uniform=0.249)
    assert not sampler.accept_draft(distribution, 10, uniform=0.25)
    assert not sampler.accept_draft(distribution, 15, uniform=0.0)
    assert sampler.sample_distribution(distribution, exclude=11, uniform=0.49) == 10
    assert sampler.sample_distribution(distribution, exclude=11, uniform=0.51) == 12
    assert sampler.counts.tolist() == [0] * 16


def test_sampler_ignores_physical_padding_rows() -> None:
    sampler = Sampler(
        SamplingConfig(temperature=0.0),
        token_domain=3,
        device=torch.device("cpu"),
    )
    token = sampler(torch.tensor([1.0, 5.0, 2.0, 100.0, 200.0]))

    assert token == 1
    assert sampler.counts.tolist() == [0, 1, 0]
