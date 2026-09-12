from __future__ import annotations

import torch

from tools.convert.qwen3_8_27b import dflash2_recipe as recipe


class TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_fused_qkv_and_gate_up_materialize_runtime_row_order() -> None:
    prefix = "layers.0."
    query = torch.full((4096, 1), 1, dtype=torch.uint8).expand(-1, 5120)
    key = torch.full((1024, 1), 2, dtype=torch.uint8).expand(-1, 5120)
    value = torch.full((1024, 1), 3, dtype=torch.uint8).expand(-1, 5120)
    qkv = recipe.materialize_tensor(
        "dflash2/layers/0/attention/query_key_value",
        TensorReader(
            {
                prefix + "self_attn.q_proj.weight": query,
                prefix + "self_attn.k_proj.weight": key,
                prefix + "self_attn.v_proj.weight": value,
            }
        ),
    )
    assert qkv.shape == (6144, 5120)
    assert torch.all(qkv[:4096, 0] == 1)
    assert torch.all(qkv[4096:5120, 0] == 2)
    assert torch.all(qkv[5120:, 0] == 3)
    del qkv

    gate = torch.full((17408, 1), 4, dtype=torch.uint8).expand(-1, 5120)
    up = torch.full((17408, 1), 5, dtype=torch.uint8).expand(-1, 5120)
    gate_up = recipe.materialize_tensor(
        "dflash2/layers/0/mlp/gate_up",
        TensorReader(
            {
                prefix + "mlp.gate_proj.weight": gate,
                prefix + "mlp.up_proj.weight": up,
            }
        ),
    )
    assert gate_up.shape == (34816, 5120)
    assert torch.all(gate_up[:17408, 0] == 4)
    assert torch.all(gate_up[17408:, 0] == 5)
