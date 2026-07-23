import torch
from safetensors.torch import save_file

from tools.convert.qwen3_6_35b_a3b import recipe


class TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


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
    for expert, projection, row in (
        (0, 0, 0),
        (0, 1, 0),
        (7, 1, 511),
        (255, 1, 511),
    ):
        physical_row = expert * 1024 + projection * 512 + row
        assert int(routed_gate_up[physical_row, 0]) == physical_row
        assert int(routed_gate_up[physical_row, -1]) == physical_row

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
        physical_row = expert * 2048 + row
        assert int(routed_down[physical_row, 0]) == physical_row


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


def test_dflash_recipe_materializes_q_k_v_and_gate_up_row_order() -> None:
    prefix = "layers.0."
    query = torch.full((4096, 1), 1, dtype=torch.uint8).expand(-1, 2048)
    key = torch.full((1024, 1), 2, dtype=torch.uint8).expand(-1, 2048)
    value = torch.full((1024, 1), 3, dtype=torch.uint8).expand(-1, 2048)
    qkv = recipe.materialize_recipe(
        recipe.DFLASH_RECIPES_BY_NAME[
            "dflash/layers/0/attention/query_key_value"
        ],
        TensorReader(
            {
                prefix + "self_attn.q_proj.weight": query,
                prefix + "self_attn.k_proj.weight": key,
                prefix + "self_attn.v_proj.weight": value,
            }
        ),
    )
    assert qkv.shape == (6144, 2048)
    assert torch.all(qkv[:4096] == 1)
    assert torch.all(qkv[4096:5120] == 2)
    assert torch.all(qkv[5120:] == 3)

    gate = torch.full((6144, 1), 4, dtype=torch.uint8).expand(-1, 2048)
    up = torch.full((6144, 1), 5, dtype=torch.uint8).expand(-1, 2048)
    gate_up = recipe.materialize_recipe(
        recipe.DFLASH_RECIPES_BY_NAME["dflash/layers/0/mlp/gate_up"],
        TensorReader(
            {
                prefix + "mlp.gate_proj.weight": gate,
                prefix + "mlp.up_proj.weight": up,
            }
        ),
    )
    assert gate_up.shape == (12288, 2048)
    assert torch.all(gate_up[:6144] == 4)
    assert torch.all(gate_up[6144:] == 5)


def test_single_file_reader_is_explicit_and_lazy(tmp_path) -> None:
    path = tmp_path / "dflash.safetensors"
    save_file({"weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)}, path)
    with recipe.ShardReader.from_file(path) as reader:
        assert reader.names == ("weight",)
        metadata = reader.metadata(("weight",))["weight"]
        assert metadata.shape == (2, 4)
        assert metadata.dtype == "BF16"
        assert metadata.shard == path.name
        assert torch.equal(
            reader.get("weight"),
            torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        )
