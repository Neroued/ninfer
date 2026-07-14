"""Hugging Face source recipe for the complete Qwen3.6-27B inventory."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Mapping

import torch

from tools.convert.common.safetensors import ShardReader

from . import inventory


SOURCE_DTYPE = "BF16"
DRAFT_ROWS = 131072


@dataclass(frozen=True, slots=True)
class SourceTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = SOURCE_DTYPE


@dataclass(frozen=True, slots=True)
class Slice:
    source: Expression
    axis: int
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class Reshape:
    source: Expression
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Transpose:
    source: Expression
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Concat:
    sources: tuple[Expression, ...]
    axis: int


@dataclass(frozen=True, slots=True)
class Cast:
    source: Expression
    dtype: str


@dataclass(frozen=True, slots=True)
class DraftHeadTokenIds:
    ranking_path: str
    tokenizer_resource: str
    vocab_rows: int
    tokenizer_id_count: int
    rows: int


@dataclass(frozen=True, slots=True)
class GatherRows:
    source: SourceTensor
    token_ids_object: str
    rows: int


Expression = (
    SourceTensor
    | Slice
    | Reshape
    | Transpose
    | Concat
    | Cast
    | DraftHeadTokenIds
    | GatherRows
)


@dataclass(frozen=True, slots=True)
class TensorRecipe:
    object_name: str
    expression: Expression


@dataclass(frozen=True, slots=True)
class SourcePreflight:
    recipe_count: int
    source_tensor_count: int
    source_shard_count: int
    source_dtype_counts: dict[str, int]


def expression_shape(expression: Expression) -> tuple[int, ...]:
    if isinstance(expression, SourceTensor):
        return expression.shape

    if isinstance(expression, Slice):
        shape = list(expression_shape(expression.source))
        if not 0 <= expression.axis < len(shape):
            raise ValueError(f"slice axis {expression.axis} is outside rank {len(shape)}")
        if not 0 <= expression.begin < expression.end <= shape[expression.axis]:
            raise ValueError(
                f"invalid slice [{expression.begin},{expression.end}) for axis size "
                f"{shape[expression.axis]}"
            )
        shape[expression.axis] = expression.end - expression.begin
        return tuple(shape)

    if isinstance(expression, Reshape):
        source_shape = expression_shape(expression.source)
        if prod(source_shape) != prod(expression.shape):
            raise ValueError(f"cannot reshape {source_shape} to {expression.shape}")
        return expression.shape

    if isinstance(expression, Transpose):
        source_shape = expression_shape(expression.source)
        if tuple(sorted(expression.axes)) != tuple(range(len(source_shape))):
            raise ValueError(f"invalid transpose axes {expression.axes} for {source_shape}")
        return tuple(source_shape[axis] for axis in expression.axes)

    if isinstance(expression, Concat):
        if not expression.sources:
            raise ValueError("concat requires at least one source")
        shapes = [expression_shape(source) for source in expression.sources]
        rank = len(shapes[0])
        if not 0 <= expression.axis < rank:
            raise ValueError(f"concat axis {expression.axis} is outside rank {rank}")
        output = list(shapes[0])
        output[expression.axis] = 0
        for shape in shapes:
            if len(shape) != rank:
                raise ValueError("concat sources have different ranks")
            for axis, (got, expected) in enumerate(zip(shape, shapes[0])):
                if axis != expression.axis and got != expected:
                    raise ValueError("concat sources have incompatible shapes")
            output[expression.axis] += shape[expression.axis]
        return tuple(output)

    if isinstance(expression, Cast):
        return expression_shape(expression.source)

    if isinstance(expression, DraftHeadTokenIds):
        return (expression.rows,)

    if isinstance(expression, GatherRows):
        return (expression.rows, *expression.source.shape[1:])

    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def _sources(expression: Expression) -> tuple[SourceTensor, ...]:
    if isinstance(expression, SourceTensor):
        return (expression,)
    if isinstance(expression, (Slice, Reshape, Transpose, Cast)):
        return _sources(expression.source)
    if isinstance(expression, Concat):
        return tuple(source for part in expression.sources for source in _sources(part))
    if isinstance(expression, GatherRows):
        return (expression.source,)
    if isinstance(expression, DraftHeadTokenIds):
        return ()
    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def _source(name: str, shape: tuple[int, ...]) -> SourceTensor:
    return SourceTensor(name=name, shape=shape)


def _attention_qproj_part(source_name: str, gate: bool) -> Expression:
    source = _source(source_name, (12288, 5120))
    per_head = Reshape(source, (24, 512, 5120))
    begin = 256 if gate else 0
    return Reshape(Slice(per_head, 1, begin, begin + 256), (6144, 5120))


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
    source_prefix = "model.visual."
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "vision/patch_embedding",
            Reshape(
                _source(source_prefix + "patch_embed.proj.weight", (1152, 3, 2, 16, 16)),
                (1152, 1536),
            ),
        ),
        TensorRecipe(
            "vision/patch_embedding_bias",
            _source(source_prefix + "patch_embed.proj.bias", (1152,)),
        ),
        TensorRecipe(
            "vision/position_embedding",
            _source(source_prefix + "pos_embed.weight", (2304, 1152)),
        ),
    ]

    for layer in inventory.VISION_LAYERS:
        source_layer = source_prefix + f"blocks.{layer}."
        object_layer = f"vision/layers/{layer}/"
        for object_suffix, source_suffix, shape in (
            ("attention/qkv", "attn.qkv.weight", (3456, 1152)),
            ("attention/qkv_bias", "attn.qkv.bias", (3456,)),
            ("attention/output", "attn.proj.weight", (1152, 1152)),
            ("attention/output_bias", "attn.proj.bias", (1152,)),
            ("mlp/fc1", "mlp.linear_fc1.weight", (4304, 1152)),
            ("mlp/fc1_bias", "mlp.linear_fc1.bias", (4304,)),
            ("mlp/fc2", "mlp.linear_fc2.weight", (1152, 4304)),
            ("mlp/fc2_bias", "mlp.linear_fc2.bias", (1152,)),
            ("norm1/weight", "norm1.weight", (1152,)),
            ("norm1/bias", "norm1.bias", (1152,)),
            ("norm2/weight", "norm2.weight", (1152,)),
            ("norm2/bias", "norm2.bias", (1152,)),
        ):
            recipes.append(
                TensorRecipe(
                    object_layer + object_suffix,
                    _source(source_layer + source_suffix, shape),
                )
            )

    for object_suffix, source_suffix, shape in (
        ("fc1", "linear_fc1.weight", (4608, 4608)),
        ("fc1_bias", "linear_fc1.bias", (4608,)),
        ("fc2", "linear_fc2.weight", (5120, 4608)),
        ("fc2_bias", "linear_fc2.bias", (5120,)),
        ("norm/weight", "norm.weight", (1152,)),
        ("norm/bias", "norm.bias", (1152,)),
    ):
        recipes.append(
            TensorRecipe(
                "vision/merger/" + object_suffix,
                _source(source_prefix + "merger." + source_suffix, shape),
            )
        )
    return tuple(recipes)


RECIPE_SPECS = (
    _build_text_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + _build_vision_recipes()
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    inventory_names = tuple(spec.name for spec in inventory.TENSOR_SPECS)
    recipe_names = tuple(recipe.object_name for recipe in RECIPE_SPECS)
    if recipe_names != inventory_names:
        raise ValueError("recipe order or coverage does not match the tensor inventory")
    if len(RECIPES_BY_NAME) != len(RECIPE_SPECS):
        raise ValueError("more than one recipe targets the same artifact object")
    inventory_by_name = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    for recipe in RECIPE_SPECS:
        expected = inventory_by_name[recipe.object_name].shape
        actual = expression_shape(recipe.expression)
        if actual != expected:
            raise ValueError(f"{recipe.object_name}: recipe shape {actual} != inventory {expected}")


def source_requirements() -> dict[str, SourceTensor]:
    requirements: dict[str, SourceTensor] = {}
    for recipe in RECIPE_SPECS:
        for source in _sources(recipe.expression):
            previous = requirements.setdefault(source.name, source)
            if previous != source:
                raise ValueError(f"inconsistent source declaration for {source.name}")
    return requirements


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    requirements = source_requirements()
    with ShardReader(model_dir) as reader:
        metadata = reader.metadata(requirements)

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, requirement in requirements.items():
        actual = metadata[name]
        if actual.shape != requirement.shape:
            raise ValueError(f"{name}: source shape {actual.shape} != required {requirement.shape}")
        if actual.dtype != requirement.dtype:
            raise ValueError(f"{name}: source dtype {actual.dtype} != required {requirement.dtype}")
        dtype_counts[actual.dtype] = dtype_counts.get(actual.dtype, 0) + 1
        shards.add(actual.shard)

    return SourcePreflight(
        recipe_count=len(RECIPE_SPECS),
        source_tensor_count=len(requirements),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def materialize_expression(
    expression: Expression,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if isinstance(expression, SourceTensor):
        tensor = reader.get(expression.name)
        if tuple(tensor.shape) != expression.shape:
            raise ValueError(
                f"{expression.name}: source shape {tuple(tensor.shape)} != {expression.shape}"
            )
        return tensor

    if isinstance(expression, Slice):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.narrow(expression.axis, expression.begin, expression.end - expression.begin)

    if isinstance(expression, Reshape):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.reshape(expression.shape)

    if isinstance(expression, Transpose):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.permute(expression.axes).contiguous()

    if isinstance(expression, Concat):
        tensors = [
            materialize_expression(source, reader, derived_tensors)
            for source in expression.sources
        ]
        return torch.cat(tensors, dim=expression.axis)

    if isinstance(expression, Cast):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        if expression.dtype != inventory.FP32:
            raise ValueError(f"unsupported direct cast target {expression.dtype}")
        return tensor.to(torch.float32)

    if isinstance(expression, DraftHeadTokenIds):
        if derived_tensors is None or "text/draft_head_token_ids" not in derived_tensors:
            raise ValueError("draft-head token IDs have not been derived")
        return derived_tensors["text/draft_head_token_ids"].to(torch.int32)

    if isinstance(expression, GatherRows):
        if derived_tensors is None or expression.token_ids_object not in derived_tensors:
            raise ValueError("draft-head token IDs are required to gather head rows")
        source = materialize_expression(expression.source, reader, derived_tensors)
        token_ids = derived_tensors[expression.token_ids_object].to(torch.int64)
        return source.index_select(0, token_ids)

    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def materialize_recipe(
    recipe: TensorRecipe,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    tensor = materialize_expression(recipe.expression, reader, derived_tensors)
    expected = expression_shape(recipe.expression)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{recipe.object_name}: materialized shape {tuple(tensor.shape)} != {expected}"
        )
    return tensor


validate_recipe_coverage()
