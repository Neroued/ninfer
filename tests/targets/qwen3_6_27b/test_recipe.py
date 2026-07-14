import torch

from tools.convert.qwen3_6_27b_rtx5090 import inventory, recipe


class TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_recipe_exactly_covers_inventory() -> None:
    assert len(recipe.RECIPE_SPECS) == 1166
    assert len(recipe.RECIPES_BY_NAME) == 1166
    assert tuple(item.object_name for item in recipe.RECIPE_SPECS) == tuple(
        item.name for item in inventory.TENSOR_SPECS
    )
    assert all(
        recipe.expression_shape(item.expression)
        == inventory.TENSOR_SPECS[index].shape
        for index, item in enumerate(recipe.RECIPE_SPECS)
    )

    requirements = recipe.source_requirements()
    assert len(requirements) == 1199
    assert {requirement.dtype for requirement in requirements.values()} == {"BF16"}
    assert requirements["model.language_model.embed_tokens.weight"].shape == (248320, 5120)
    assert requirements["mtp.layers.0.self_attn.q_proj.weight"].shape == (12288, 5120)
    assert requirements["model.visual.patch_embed.proj.weight"].shape == (
        1152,
        3,
        2,
        16,
        16,
    )


def test_representative_recipe_structure() -> None:
    query_key = recipe.RECIPES_BY_NAME["text/layers/3/attention/query_key"].expression
    assert isinstance(query_key, recipe.Concat)
    assert recipe.expression_shape(query_key) == (7168, 5120)
    assert isinstance(query_key.sources[0], recipe.Reshape)
    query_slice = query_key.sources[0].source
    assert isinstance(query_slice, recipe.Slice)
    assert (query_slice.axis, query_slice.begin, query_slice.end) == (1, 0, 256)

    gate_value = recipe.RECIPES_BY_NAME["text/layers/3/attention/gate_value"].expression
    assert isinstance(gate_value, recipe.Concat)
    gate_slice = gate_value.sources[0].source
    assert isinstance(gate_slice, recipe.Slice)
    assert (gate_slice.axis, gate_slice.begin, gate_slice.end) == (1, 256, 512)

    mtp = recipe.RECIPES_BY_NAME[
        "mtp/layer/attention/query_key_gate_value"
    ].expression
    assert isinstance(mtp, recipe.Concat)
    assert tuple(recipe.expression_shape(part)[0] for part in mtp.sources) == (
        6144,
        1024,
        6144,
        1024,
    )

    draft_ids = recipe.RECIPES_BY_NAME["text/draft_head_token_ids"].expression
    assert isinstance(draft_ids, recipe.DraftHeadTokenIds)
    assert draft_ids.ranking_path.endswith("ranking.train.counts.i64")
    assert draft_ids.tokenizer_resource == "frontend/tokenizer_config.json"
    assert (draft_ids.vocab_rows, draft_ids.tokenizer_id_count, draft_ids.rows) == (
        248320,
        248077,
        131072,
    )
    draft_rows = recipe.RECIPES_BY_NAME["text/draft_head"].expression
    assert isinstance(draft_rows, recipe.GatherRows)
    assert draft_rows.token_ids_object == "text/draft_head_token_ids"


def test_representative_transforms_materialize_without_full_weights() -> None:
    convolution_source = torch.arange(10240 * 4, dtype=torch.float32).to(torch.bfloat16)
    convolution_source = convolution_source.reshape(10240, 1, 4)
    convolution_name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    convolution = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/convolution"],
        TensorReader({convolution_name: convolution_source}),
    )
    assert convolution.shape == (4, 10240)
    assert torch.equal(convolution, convolution_source[:, 0, :].transpose(0, 1).contiguous())

    a_log_name = "model.language_model.layers.0.linear_attn.A_log"
    a_log_source = torch.linspace(-2, 2, 48, dtype=torch.bfloat16)
    a_log = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/a_log"],
        TensorReader({a_log_name: a_log_source}),
    )
    assert a_log.dtype == torch.float32
    assert torch.equal(a_log, a_log_source.float())

    small_source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expression = recipe.Concat(
        (
            recipe.Reshape(recipe.Slice(recipe.SourceTensor("small", (2, 3, 4)), 1, 0, 2), (4, 4)),
            recipe.Transpose(
                recipe.Reshape(
                    recipe.Slice(recipe.SourceTensor("small", (2, 3, 4)), 1, 2, 3),
                    (2, 4),
                ),
                (0, 1),
            ),
        ),
        0,
    )
    materialized = recipe.materialize_expression(expression, TensorReader({"small": small_source}))
    assert materialized.shape == (6, 4)
    assert torch.equal(materialized[:4], small_source[:, :2, :].reshape(4, 4))

    token_ids = torch.arange(131072, dtype=torch.int64)
    materialized_ids = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/draft_head_token_ids"],
        TensorReader({}),
        {"text/draft_head_token_ids": token_ids},
    )
    assert materialized_ids.dtype == torch.int32
    assert materialized_ids.shape == (131072,)
