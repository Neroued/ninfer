"""Hugging Face source recipe for the complete Qwen3.6-27B inventory."""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    SOURCE_DTYPE,
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    ShardReader,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    attention_qproj_part,
    build_vision_recipes,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_sources as _preflight_recipe_sources,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory


DRAFT_ROWS = 131072


_sources = expression_sources
_source = source


def _attention_qproj_part(source_name: str, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=24,
        hidden_size=5120,
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            _source("model.language_model.embed_tokens.weight", (248320, 5120)),
        )
    ]

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (5120,)),
            )
        )

        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            query = _attention_qproj_part(q_proj, gate=False)
            gate = _attention_qproj_part(q_proj, gate=True)
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key",
                        Concat(
                            (
                                query,
                                _source(source_prefix + "self_attn.k_proj.weight", (1024, 5120)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/gate_value",
                        Concat(
                            (
                                gate,
                                _source(source_prefix + "self_attn.v_proj.weight", (1024, 5120)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        _source(source_prefix + "self_attn.q_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        _source(source_prefix + "self_attn.k_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        _source(source_prefix + "self_attn.o_proj.weight", (5120, 6144)),
                    ),
                )
            )
        else:
            qkv_source = _source(
                source_prefix + "linear_attn.in_proj_qkv.weight",
                (10240, 5120),
            )
            convolution = _source(
                source_prefix + "linear_attn.conv1d.weight",
                (10240, 1, 4),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(_source(source_prefix + "linear_attn.A_log", (48,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(_source(source_prefix + "linear_attn.dt_bias", (48,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (10240, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight", (48, 5120)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight", (48, 5120)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key",
                        Slice(qkv_source, 0, 0, 4096),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/value",
                        Slice(qkv_source, 0, 4096, 10240),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        _source(source_prefix + "linear_attn.norm.weight", (128,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/z",
                        _source(source_prefix + "linear_attn.in_proj_z.weight", (6144, 5120)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight", (5120, 6144)),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight", (5120,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight", (17408, 5120)),
                            _source(source_prefix + "mlp.up_proj.weight", (17408, 5120)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight", (5120, 17408)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source("model.language_model.norm.weight", (5120,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source("lm_head.weight", (248320, 5120)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                _source("lm_head.weight", (248320, 5120)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path="tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64",
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=248320,
                tokenizer_id_count=248077,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "mtp.layers.0."
    q_proj = source_prefix + "self_attn.q_proj.weight"
    return (
        TensorRecipe("mtp/input_projection", _source("mtp.fc.weight", (5120, 10240))),
        TensorRecipe(
            "mtp/embedding_norm",
            _source("mtp.pre_fc_norm_embedding.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            _source("mtp.pre_fc_norm_hidden.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/layer/input_norm",
            _source(source_prefix + "input_layernorm.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_key_gate_value",
            Concat(
                (
                    _attention_qproj_part(q_proj, gate=False),
                    _source(source_prefix + "self_attn.k_proj.weight", (1024, 5120)),
                    _attention_qproj_part(q_proj, gate=True),
                    _source(source_prefix + "self_attn.v_proj.weight", (1024, 5120)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_norm",
            _source(source_prefix + "self_attn.q_norm.weight", (256,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/key_norm",
            _source(source_prefix + "self_attn.k_norm.weight", (256,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/output",
            _source(source_prefix + "self_attn.o_proj.weight", (5120, 6144)),
        ),
        TensorRecipe(
            "mtp/layer/post_attention_norm",
            _source(source_prefix + "post_attention_layernorm.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/layer/mlp/gate_up",
            Concat(
                (
                    _source(source_prefix + "mlp.gate_proj.weight", (17408, 5120)),
                    _source(source_prefix + "mlp.up_proj.weight", (17408, 5120)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/mlp/down",
            _source(source_prefix + "mlp.down_proj.weight", (5120, 17408)),
        ),
        TensorRecipe("mtp/final_norm", _source("mtp.norm.weight", (5120,))),
    )


def _build_vision_recipes() -> tuple[TensorRecipe, ...]:
    return build_vision_recipes(5120)


RECIPE_SPECS = (
    _build_text_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + _build_vision_recipes()
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    return _preflight_recipe_sources(model_dir, RECIPE_SPECS)


validate_recipe_coverage()
