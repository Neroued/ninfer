import torch
import torch.nn.functional as F

from tools.reference.qwen3_6_35b_a3b_rtx5090.bindings import (
    ExpertBank,
    MoeBinding,
)
from tools.reference.qwen3_6_35b_a3b_rtx5090.config import CFG
from tools.reference.qwen3_6_35b_a3b_rtx5090.moe import forward


def test_selected_expert_rows_and_moe_precision_boundaries() -> None:
    selected = (255, 0, 17, 31, 63, 127, 191, 223)
    x = torch.zeros((1, CFG.hidden), dtype=torch.bfloat16)
    x[0, 0] = 1

    router_shared = torch.full(
        (CFG.experts + 1, CFG.hidden),
        -20.0,
        dtype=torch.bfloat16,
    )
    for rank, expert in enumerate(selected):
        router_shared[expert, 0] = 8.0 - rank * 0.5
    router_shared[CFG.experts, 0] = 0.0
    shared_gate_weight = 0.2513
    shared_up_weight = 3.0061
    shared_down_weight = 2.0097
    expert_gate_weight = 0.5031
    expert_up_weight = 2.0063
    shared_gate_up = torch.zeros((1024, CFG.hidden), dtype=torch.float32)
    shared_gate_up[0, 0] = shared_gate_weight
    shared_gate_up[512, 0] = shared_up_weight
    shared_down = torch.zeros(
        (CFG.hidden, CFG.shared_intermediate),
        dtype=torch.float32,
    )
    shared_down[0, 0] = shared_down_weight

    class FakeModel:
        def __init__(self) -> None:
            self.blocks = {
                "router": router_shared,
                "shared_gate_up": shared_gate_up,
                "shared_down": shared_down,
            }
            self.reads: list[tuple[str, int, int]] = []

        def block_weight(self, block: str, *, small_t: bool) -> torch.Tensor:
            assert small_t
            return self.blocks[block]

        def rows(
            self,
            block: str,
            rows: torch.Tensor,
            *,
            small_t: bool,
        ) -> torch.Tensor:
            assert small_t
            indices = rows.cpu()
            begin, end = int(indices[0]), int(indices[-1])
            self.reads.append((block, begin, end))
            if block == "routed_gate_up":
                assert begin // 1024 == end // 1024
                assert end - begin + 1 == 1024
                weight = torch.zeros((1024, CFG.hidden), dtype=torch.float32)
                weight[0, 0] = expert_gate_weight
                weight[512, 0] = expert_up_weight
                return weight
            assert begin // 2048 == end // 2048
            assert end - begin + 1 == 2048
            expert = begin // 2048
            weight = torch.zeros(
                (CFG.hidden, CFG.expert_intermediate),
                dtype=torch.float32,
            )
            weight[0, 0] = expert + 1.0031
            return weight

    binding = MoeBinding(
        router_shared_gate="router",
        router=None,
        shared_gate=None,
        routed_gate_up=ExpertBank("routed_gate_up", 256, 1024, 512),
        routed_down=ExpertBank("routed_down", 256, 2048, None),
        shared_gate_up="shared_gate_up",
        shared_expert_gate=None,
        shared_up=None,
        shared_down="shared_down",
    )
    model = FakeModel()
    actual = forward(model, binding, x, small_t=True)

    router_logits = router_shared[: CFG.experts, 0].unsqueeze(0)
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    expected_values, expected_ids = torch.topk(
        probabilities,
        CFG.experts_per_token,
        dim=-1,
    )
    expected_weights = (
        expected_values / expected_values.sum(dim=-1, keepdim=True)
    ).to(torch.bfloat16)
    assert torch.equal(actual.expert_ids, expected_ids)
    assert torch.equal(actual.route_weights, expected_weights)
    assert set(actual.expert_ids[0].tolist()) == set(selected)

    read_experts = {
        begin // (1024 if block == "routed_gate_up" else 2048)
        for block, begin, _ in model.reads
    }
    assert read_experts == set(selected)
    assert len(model.reads) == 2 * CFG.experts_per_token

    gate = torch.tensor(expert_gate_weight).to(torch.bfloat16)
    up = torch.tensor(expert_up_weight).to(torch.bfloat16)
    expert_hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    routed_terms = []
    for expert_id, route_weight in zip(
        expected_ids[0].tolist(),
        expected_weights[0],
        strict=True,
    ):
        expert_down = (
            expert_hidden.float() * float(expert_id + 1.0031)
        ).to(torch.bfloat16)
        routed_terms.append(route_weight.float() * expert_down.float())
    routed = torch.stack(routed_terms).sum().to(torch.bfloat16)

    shared_hidden = (
        F.silu(torch.tensor(shared_gate_weight).to(torch.bfloat16).float())
        * torch.tensor(shared_up_weight).to(torch.bfloat16).float()
    ).to(torch.bfloat16)
    shared = (
        shared_hidden.float() * shared_down_weight
    ).to(torch.bfloat16)
    shared_scale = torch.sigmoid(torch.tensor(0.0))
    expected = (
        routed.float() + shared_scale * shared.float()
    ).to(torch.bfloat16)
    assert torch.equal(actual.output[0, 0], expected)
    assert torch.count_nonzero(actual.output[0, 1:]) == 0
