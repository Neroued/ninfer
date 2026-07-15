import torch

from tools.convert.qwen3_6_35b_a3b_rtx5090 import inventory, recipe


class TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_recipe_exactly_covers_inventory_and_checkpoint_sources() -> None:
    assert len(recipe.RECIPE_SPECS) == 883
    assert len(recipe.RECIPES_BY_NAME) == 883
    assert tuple(item.object_name for item in recipe.RECIPE_SPECS) == tuple(
        item.name for item in inventory.TENSOR_SPECS
    )
    assert all(
        recipe.expression_shape(item.expression) == inventory.TENSOR_SPECS[index].shape
        for index, item in enumerate(recipe.RECIPE_SPECS)
    )

    requirements = recipe.source_requirements()
    assert len(requirements) == 1045
    assert {item.dtype for item in requirements.values()} == {"BF16"}
    assert requirements["model.language_model.embed_tokens.weight"].shape == (
        248320,
        2048,
    )
    assert requirements["model.language_model.layers.0.mlp.experts.gate_up_proj"].shape == (
        256,
        1024,
        2048,
    )
    assert requirements["model.language_model.layers.0.mlp.experts.down_proj"].shape == (
        256,
        2048,
        512,
    )
    assert requirements["mtp.layers.0.self_attn.q_proj.weight"].shape == (8192, 2048)
    assert requirements["model.visual.merger.linear_fc2.weight"].shape == (2048, 4608)


def test_attention_recipe_materializes_q_k_gate_v_row_order() -> None:
    prefix = "model.language_model.layers.3.self_attn."
    q_rows = torch.arange(8192, dtype=torch.int32).view(-1, 1).expand(-1, 2048)
    key = torch.full((512, 1), -1, dtype=torch.int32).expand(-1, 2048)
    value = torch.full((512, 1), -2, dtype=torch.int32).expand(-1, 2048)
    fused = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/3/attention/query_key_gate_value"],
        TensorReader(
            {
                prefix + "q_proj.weight": q_rows,
                prefix + "k_proj.weight": key,
                prefix + "v_proj.weight": value,
            }
        ),
    )

    query_rows = torch.cat(
        [torch.arange(head * 512, head * 512 + 256) for head in range(16)]
    ).to(torch.int32)
    gate_rows = torch.cat(
        [torch.arange(head * 512 + 256, head * 512 + 512) for head in range(16)]
    ).to(torch.int32)
    assert fused.shape == (9216, 2048)
    assert torch.equal(fused[:4096, 0], query_rows)
    assert torch.all(fused[4096:4608, 0] == -1)
    assert torch.equal(fused[4608:8704, 0], gate_rows)
    assert torch.all(fused[8704:, 0] == -2)


def test_moe_recipe_preserves_expert_major_half_split_rows() -> None:
    prefix = "model.language_model.layers.0.mlp."
    gate_up_rows = (
        torch.arange(256 * 1024, dtype=torch.int32)
        .reshape(256, 1024, 1)
        .expand(-1, -1, 2048)
    )
    routed_gate_up = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/moe/routed_gate_up"],
        TensorReader({prefix + "experts.gate_up_proj": gate_up_rows}),
    )
    for expert, projection, row in ((0, 0, 0), (0, 1, 0), (7, 1, 511), (255, 1, 511)):
        stored = inventory.routed_gate_up_row(expert, projection, row)
        expected = expert * 1024 + projection * 512 + row
        assert int(routed_gate_up[stored, 0]) == expected
        assert int(routed_gate_up[stored, -1]) == expected

    down_rows = (
        torch.arange(256 * 2048, dtype=torch.int32)
        .reshape(256, 2048, 1)
        .expand(-1, -1, 512)
    )
    routed_down = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/moe/routed_down"],
        TensorReader({prefix + "experts.down_proj": down_rows}),
    )
    for expert, row in ((0, 0), (9, 123), (255, 2047)):
        stored = inventory.routed_down_row(expert, row)
        assert int(routed_down[stored, 0]) == expert * 2048 + row


def test_gdn_recipe_materializes_half_split_ab_and_qkvz() -> None:
    prefix = "model.language_model.layers.0.linear_attn."
    a = torch.full((32, 1), 1, dtype=torch.uint8).expand(-1, 2048)
    b = torch.full((32, 1), 2, dtype=torch.uint8).expand(-1, 2048)
    ab = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/a_b_projection"],
        TensorReader(
            {
                prefix + "in_proj_a.weight": a,
                prefix + "in_proj_b.weight": b,
            }
        ),
    )
    assert torch.all(ab[:32] == 1)
    assert torch.all(ab[32:] == 2)

    qkv = torch.arange(8192, dtype=torch.int32).view(-1, 1).expand(-1, 2048)
    z = torch.full((4096, 1), -1, dtype=torch.int32).expand(-1, 2048)
    qkvz = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/query_key_value_z"],
        TensorReader(
            {
                prefix + "in_proj_qkv.weight": qkv,
                prefix + "in_proj_z.weight": z,
            }
        ),
    )
    assert torch.equal(qkvz[:8192, 0], torch.arange(8192, dtype=torch.int32))
    assert torch.all(qkvz[8192:, 0] == -1)
