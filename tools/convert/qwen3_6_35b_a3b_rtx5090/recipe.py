"""Hugging Face source recipe for Qwen3.6-35B-A3B on RTX 5090."""

from __future__ import annotations

from pathlib import Path

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common.recipe import (
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    SOURCE_DTYPE,
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
    preflight_sources as _common_preflight_sources,
    source,
    source_requirements as _common_source_requirements,
    validate_recipe_coverage as _common_validate_recipe_coverage,
)

from . import inventory


DRAFT_ROWS = 131072
DRAFT_RANKING_PATH = (
    "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
)


def _attention_part(source_name: str, *, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=16,
        hidden_size=2048,
    )


def _moe_recipes(source_prefix: str, object_prefix: str) -> tuple[TensorRecipe, ...]:
    """Build one expert-major sparse-MoE source transform.

    The routed gate/up source is already expert-major and half-split inside
    each expert.  A contiguous reshape therefore implements exactly
    ``stored_row(e,p,r) = e*1024 + p*512 + r`` without a permutation.
    """

    return (
        TensorRecipe(
            object_prefix + "router_shared_gate",
            Concat(
                (
                    source(source_prefix + "gate.weight", (256, 2048)),
                    source(source_prefix + "shared_expert_gate.weight", (1, 2048)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "routed_gate_up",
            Reshape(
                source(source_prefix + "experts.gate_up_proj", (256, 1024, 2048)),
                (262144, 2048),
            ),
        ),
        TensorRecipe(
            object_prefix + "routed_down",
            Reshape(
                source(source_prefix + "experts.down_proj", (256, 2048, 512)),
                (524288, 512),
            ),
        ),
        TensorRecipe(
            object_prefix + "shared_gate_up",
            Concat(
                (
                    source(
                        source_prefix + "shared_expert.gate_proj.weight",
                        (512, 2048),
                    ),
                    source(
                        source_prefix + "shared_expert.up_proj.weight",
                        (512, 2048),
                    ),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "shared_down",
            source(
                source_prefix + "shared_expert.down_proj.weight",
                (2048, 512),
            ),
        ),
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            source("model.language_model.embed_tokens.weight", (248320, 2048)),
        )
    ]

    for layer in inventory.TEXT_LAYERS:
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                source(source_prefix + "input_layernorm.weight", (2048,)),
            )
        )

        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        Concat(
                            (
                                _attention_part(q_proj, gate=False),
                                source(
                                    source_prefix + "self_attn.k_proj.weight",
                                    (512, 2048),
                                ),
                                _attention_part(q_proj, gate=True),
                                source(
                                    source_prefix + "self_attn.v_proj.weight",
                                    (512, 2048),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        source(source_prefix + "self_attn.q_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        source(source_prefix + "self_attn.k_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        source(
                            source_prefix + "self_attn.o_proj.weight",
                            (2048, 4096),
                        ),
                    ),
                )
            )
        else:
            convolution = source(
                source_prefix + "linear_attn.conv1d.weight",
                (8192, 1, 4),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(
                            source(source_prefix + "linear_attn.A_log", (32,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(
                            source(source_prefix + "linear_attn.dt_bias", (32,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (8192, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        Concat(
                            (
                                source(
                                    source_prefix + "linear_attn.in_proj_a.weight",
                                    (32, 2048),
                                ),
                                source(
                                    source_prefix + "linear_attn.in_proj_b.weight",
                                    (32, 2048),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        Concat(
                            (
                                source(
                                    source_prefix + "linear_attn.in_proj_qkv.weight",
                                    (8192, 2048),
                                ),
                                source(
                                    source_prefix + "linear_attn.in_proj_z.weight",
                                    (4096, 2048),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        source(source_prefix + "linear_attn.norm.weight", (128,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        source(
                            source_prefix + "linear_attn.out_proj.weight",
                            (2048, 4096),
                        ),
                    ),
                )
            )

        recipes.append(
            TensorRecipe(
                object_prefix + "post_attention_norm",
                source(source_prefix + "post_attention_layernorm.weight", (2048,)),
            )
        )
        recipes.extend(
            _moe_recipes(
                source_prefix + "mlp.",
                object_prefix + "moe/",
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                source("model.language_model.norm.weight", (2048,)),
            ),
            TensorRecipe(
                "text/output_head",
                source("lm_head.weight", (248320, 2048)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                source("lm_head.weight", (248320, 2048)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path=DRAFT_RANKING_PATH,
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=248320,
                tokenizer_id_count=248077,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "mtp.layers.0."
    object_prefix = "mtp/layer/"
    q_proj = source_prefix + "self_attn.q_proj.weight"
    recipes = [
        TensorRecipe("mtp/input_projection", source("mtp.fc.weight", (2048, 4096))),
        TensorRecipe(
            "mtp/embedding_norm",
            source("mtp.pre_fc_norm_embedding.weight", (2048,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            source("mtp.pre_fc_norm_hidden.weight", (2048,)),
        ),
        TensorRecipe(
            object_prefix + "input_norm",
            source(source_prefix + "input_layernorm.weight", (2048,)),
        ),
        TensorRecipe(
            object_prefix + "attention/query_key_gate_value",
            Concat(
                (
                    _attention_part(q_proj, gate=False),
                    source(source_prefix + "self_attn.k_proj.weight", (512, 2048)),
                    _attention_part(q_proj, gate=True),
                    source(source_prefix + "self_attn.v_proj.weight", (512, 2048)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "attention/query_norm",
            source(source_prefix + "self_attn.q_norm.weight", (256,)),
        ),
        TensorRecipe(
            object_prefix + "attention/key_norm",
            source(source_prefix + "self_attn.k_norm.weight", (256,)),
        ),
        TensorRecipe(
            object_prefix + "attention/output",
            source(source_prefix + "self_attn.o_proj.weight", (2048, 4096)),
        ),
        TensorRecipe(
            object_prefix + "post_attention_norm",
            source(source_prefix + "post_attention_layernorm.weight", (2048,)),
        ),
    ]
    recipes.extend(_moe_recipes(source_prefix + "mlp.", object_prefix + "moe/"))
    recipes.append(
        TensorRecipe("mtp/final_norm", source("mtp.norm.weight", (2048,)))
    )
    return tuple(recipes)


RECIPE_SPECS = (
    _build_text_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + build_vision_recipes(2048)
)
RECIPES_BY_NAME = {item.object_name: item for item in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    """Validate exact object pairing and complete 1045-source coverage."""

    _common_validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)
    if len(RECIPE_SPECS) != 883 or len(RECIPES_BY_NAME) != 883:
        raise ValueError("35B recipe does not contain exactly 883 tensor transforms")
    requirements = source_requirements()
    if len(requirements) != 1045:
        raise ValueError(
            f"35B recipe covers {len(requirements)} unique sources, expected 1045"
        )
    if {item.dtype for item in requirements.values()} != {SOURCE_DTYPE}:
        raise ValueError("35B source recipe must contain only BF16 tensors")


def source_requirements() -> dict[str, SourceTensor]:
    return _common_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    """Require the selected checkpoint's exact 1045-tensor source inventory."""

    requirements = source_requirements()
    with ShardReader(model_dir) as reader:
        actual_names = set(reader.names)
    required_names = set(requirements)
    if actual_names != required_names:
        missing = sorted(required_names - actual_names)
        extra = sorted(actual_names - required_names)
        details = []
        if missing:
            details.append(f"missing={missing[:8]!r}")
        if extra:
            details.append(f"extra={extra[:8]!r}")
        raise ValueError(
            "35B checkpoint source inventory differs from the exact 1045 tensors"
            + (": " + ", ".join(details) if details else "")
        )
    result = _common_preflight_sources(model_dir, RECIPE_SPECS)
    if (
        result.source_tensor_count != 1045
        or result.source_shard_count != 26
        or result.source_dtype_counts != {SOURCE_DTYPE: 1045}
    ):
        raise ValueError(f"35B checkpoint source preflight drifted: {result}")
    return result


validate_recipe_coverage()


__all__ = [
    "Cast",
    "Concat",
    "DRAFT_RANKING_PATH",
    "DRAFT_ROWS",
    "DraftHeadTokenIds",
    "Expression",
    "GatherRows",
    "RECIPE_SPECS",
    "RECIPES_BY_NAME",
    "Reshape",
    "SOURCE_DTYPE",
    "ShardReader",
    "Slice",
    "SourcePreflight",
    "SourceTensor",
    "TensorRecipe",
    "Transpose",
    "expression_shape",
    "expression_sources",
    "materialize_expression",
    "materialize_recipe",
    "preflight_sources",
    "source_requirements",
    "validate_recipe_coverage",
]
