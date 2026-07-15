"""Sparse-MoE oracle over the selected rows of the 35B expert banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from .bindings import ExpertBank, MoeBinding
from .config import CFG
from .ops import bf16, linear, silu_mul


@dataclass(frozen=True, slots=True)
class MoeResult:
    output: torch.Tensor
    router_logits: torch.Tensor
    route_weights: torch.Tensor
    expert_ids: torch.Tensor


def route(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 softmax/top-8/renormalization with a BF16 route boundary."""

    probabilities = torch.softmax(router_logits.float(), dim=-1)
    values, expert_ids = torch.topk(
        probabilities,
        CFG.experts_per_token,
        dim=-1,
    )
    values = values / values.sum(dim=-1, keepdim=True)
    return values.to(torch.bfloat16), expert_ids


def _expert_rows(
    model: Any,
    bank: ExpertBank,
    expert: int,
    device: torch.device,
    *,
    small_t: bool,
) -> torch.Tensor:
    begin = expert * bank.rows_per_expert
    rows = torch.arange(
        begin,
        begin + bank.rows_per_expert,
        device=device,
        dtype=torch.long,
    )
    return model.rows(bank.block, rows, small_t=small_t)


def forward(
    model: Any,
    weights: MoeBinding,
    x: torch.Tensor,
    *,
    small_t: bool,
) -> MoeResult:
    """Evaluate routed and gated-shared experts without decoding unused banks."""

    router_shared = linear(
        x,
        model.block_weight(weights.router_shared_gate, small_t=small_t),
        small_t=small_t,
    )
    router_logits = router_shared[:, : CFG.experts]
    shared_scale = torch.sigmoid(router_shared[:, CFG.experts :].float())
    route_weights, expert_ids = route(router_logits)

    shared_gate_up = linear(
        x,
        model.block_weight(weights.shared_gate_up, small_t=small_t),
        small_t=small_t,
    )
    shared_gate, shared_up = shared_gate_up.split(
        CFG.shared_intermediate,
        dim=-1,
    )
    shared_hidden = silu_mul(shared_gate, shared_up)
    shared_down = linear(
        shared_hidden,
        model.block_weight(weights.shared_down, small_t=small_t),
        small_t=small_t,
    )

    routed_accum = torch.zeros(
        (x.shape[0], CFG.hidden),
        device=x.device,
        dtype=torch.float32,
    )
    for expert in torch.unique(expert_ids).tolist():
        selected = (expert_ids == expert).nonzero(as_tuple=False)
        token_rows = selected[:, 0]
        route_slots = selected[:, 1]
        expert_input = x.index_select(0, token_rows)

        gate_up_weight = _expert_rows(
            model,
            weights.routed_gate_up,
            expert,
            x.device,
            small_t=small_t,
        )
        split = cast(int, weights.routed_gate_up.split_rows)
        gate_up = linear(expert_input, gate_up_weight, small_t=small_t)
        gate, up = gate_up.split(split, dim=-1)
        expert_hidden = silu_mul(gate, up)

        down_weight = _expert_rows(
            model,
            weights.routed_down,
            expert,
            x.device,
            small_t=small_t,
        )
        expert_down = linear(expert_hidden, down_weight, small_t=small_t)
        selected_weights = route_weights[token_rows, route_slots]
        routed_accum.index_add_(
            0,
            token_rows,
            expert_down.float() * selected_weights.float().unsqueeze(-1),
        )

    routed = bf16(routed_accum)
    output = bf16(routed.float() + shared_scale * shared_down.float())
    return MoeResult(output, router_logits, route_weights, expert_ids)


__all__ = ["MoeResult", "forward", "route"]
