"""Closed dual-source recipe for the Qwen3.8-27B NVFP4 artifact."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import recipe as official_recipe

from . import inventory_nvfp4 as inventory


BASE_REPOSITORY = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QUANTIZED_REPOSITORY = "unsloth/Qwen3.8-27B-NVFP4"
QUANTIZED_REVISION = "60e813d4dbbdc5d64cf3f5a8caf2897bedf03679"


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class MatrixSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class MatrixPart:
    source: MatrixSource
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Fp8WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]


@dataclass(frozen=True, slots=True)
class Nvfp4WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    divisor_sources: tuple[MatrixSource, ...]


@dataclass(frozen=True, slots=True)
class InputDivisorRecipe:
    object_name: str
    sources: tuple[MatrixSource, ...]
    weight_names: tuple[str, ...]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    begin = 256 if gate else 0
    return MatrixPart(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def _build_quantized_matrix_recipes() -> tuple[
    tuple[Fp8WeightRecipe, ...],
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
    tuple[tuple[MatrixSource, ...], ...],
]:
    fp8_weights: list[Fp8WeightRecipe] = []
    nvfp4_weights: list[Nvfp4WeightRecipe] = []
    input_divisors: list[InputDivisorRecipe] = []
    divisor_groups: list[tuple[MatrixSource, ...]] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        (
                            _q_part(query, False),
                            _all(key),
                            _q_part(query, True),
                            _all(value),
                        ),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "attention/output",
                        output.shape,
                        (_all(output),),
                    ),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(
                source_prefix + "linear_attn.out_proj", 5120, 6144
            )
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        (16384, 5120),
                        (_all(query_key_value), _all(z)),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "gdn/output",
                        output.shape,
                        (_all(output),),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        if layer in inventory.NVFP4_MLP_LAYERS:
            gate_up_sources = (gate, up)
            divisor_groups.append(gate_up_sources)
            nvfp4_weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "mlp/gate_up",
                        (34816, 5120),
                        (_all(gate), _all(up)),
                        gate_up_sources,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "mlp/down",
                        down.shape,
                        (_all(down),),
                        (down,),
                    ),
                )
            )
            input_divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix
                        + "mlp/gate_up_projection/input_scale_divisor",
                        gate_up_sources,
                        (object_prefix + "mlp/gate_up",),
                    ),
                    InputDivisorRecipe(
                        object_prefix + "mlp/down_projection/input_scale_divisor",
                        (down,),
                        (object_prefix + "mlp/down",),
                    ),
                )
            )
        else:
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "mlp/gate_up",
                        (34816, 5120),
                        (_all(gate), _all(up)),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "mlp/down",
                        down.shape,
                        (_all(down),),
                    ),
                )
            )

    output_head = _source("lm_head", 248320, 5120)
    fp8_weights.append(
        Fp8WeightRecipe(
            "text/output_head", output_head.shape, (_all(output_head),)
        )
    )
    return (
        tuple(fp8_weights),
        tuple(nvfp4_weights),
        tuple(input_divisors),
        tuple(divisor_groups),
    )


def _build_quantized_direct_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = []
    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "input_norm",
                family_recipe.source(
                    source_prefix + "input_layernorm.weight", (5120,)
                ),
            )
        )
        if layer in inventory.FULL_ATTENTION_LAYERS:
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/query_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.q_norm.weight", (256,)
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/key_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.k_norm.weight", (256,)
                        ),
                    ),
                )
            )
        else:
            convolution = family_recipe.source(
                source_prefix + "linear_attn.conv1d.weight", (10240, 1, 4)
            )
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_log",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.A_log", (48,)
                            ),
                            inventory.FP32,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.dt_bias", (48,)
                            ),
                            inventory.FP32,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/convolution",
                        family_recipe.Transpose(
                            family_recipe.Reshape(
                                family_recipe.Slice(convolution, 1, 0, 1),
                                (10240, 4),
                            ),
                            (1, 0),
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        family_recipe.Concat(
                            (
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_a.weight",
                                    (48, 5120),
                                ),
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_b.weight",
                                    (48, 5120),
                                ),
                            ),
                            0,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/norm",
                        family_recipe.source(
                            source_prefix + "linear_attn.norm.weight", (128,)
                        ),
                    ),
                )
            )
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "post_attention_norm",
                family_recipe.source(
                    source_prefix + "post_attention_layernorm.weight", (5120,)
                ),
            )
        )
    recipes.append(
        family_recipe.TensorRecipe(
            "text/final_norm",
            family_recipe.source("model.language_model.norm.weight", (5120,)),
        )
    )
    return tuple(recipes)


(
    FP8_WEIGHT_RECIPES,
    NVFP4_WEIGHT_RECIPES,
    INPUT_DIVISOR_RECIPES,
    WEIGHT_DIVISOR_GROUPS,
) = _build_quantized_matrix_recipes()
FP8_WEIGHTS_BY_NAME = {item.object_name: item for item in FP8_WEIGHT_RECIPES}
NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in NVFP4_WEIGHT_RECIPES
}
INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in INPUT_DIVISOR_RECIPES
}

FP8_SOURCES = tuple(
    dict.fromkeys(
        part.source for recipe in FP8_WEIGHT_RECIPES for part in recipe.parts
    )
)
NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source
        for recipe in NVFP4_WEIGHT_RECIPES
        for part in recipe.parts
    )
)

QUANTIZED_DIRECT_RECIPES = _build_quantized_direct_recipes()
QUANTIZED_DIRECT_BY_NAME = {
    item.object_name: item for item in QUANTIZED_DIRECT_RECIPES
}
QUANTIZED_DIRECT_SPECS = tuple(
    spec
    for spec in inventory.TEXT_CORE_TENSOR_SPECS
    if spec.name in QUANTIZED_DIRECT_BY_NAME
)

OFFICIAL_TENSOR_SPECS = tuple(
    spec
    for spec in inventory.TENSOR_SPECS
    if spec.name == "text/token_embedding"
    or spec.name.startswith("text/draft_head")
    or spec.name.startswith("mtp/")
    or spec.name.startswith("vision/")
)
OFFICIAL_RECIPES = tuple(
    official_recipe.RECIPES_BY_NAME[spec.name]
    for spec in OFFICIAL_TENSOR_SPECS
)
OFFICIAL_RECIPES_BY_NAME = {
    item.object_name: item for item in OFFICIAL_RECIPES
}
_embedding_expression = OFFICIAL_RECIPES_BY_NAME[
    "text/token_embedding"
].expression
if not isinstance(_embedding_expression, family_recipe.SourceTensor):
    raise ValueError("token embedding must map to one direct official source")
OFFICIAL_EMBEDDING_SOURCE = _embedding_expression


def _validate_matrix_recipe(
    object_name: str,
    shape: tuple[int, int],
    parts: tuple[MatrixPart, ...],
) -> None:
    rows = sum(part.output_rows for part in parts)
    if not parts or (rows, parts[0].source.shape[1]) != shape:
        raise ValueError(f"{object_name}: invalid fused row geometry")
    if any(part.source.shape[1] != shape[1] for part in parts):
        raise ValueError(f"{object_name}: incompatible source K")


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(
        QUANTIZED_DIRECT_RECIPES, QUANTIZED_DIRECT_SPECS
    )
    family_recipe.validate_recipe_coverage(
        OFFICIAL_RECIPES, OFFICIAL_TENSOR_SPECS
    )
    if (
        len(FP8_WEIGHT_RECIPES),
        len(NVFP4_WEIGHT_RECIPES),
        len(INPUT_DIVISOR_RECIPES),
        len(WEIGHT_DIVISOR_GROUPS),
        len(FP8_SOURCES),
        len(NVFP4_SOURCES),
        len(QUANTIZED_DIRECT_RECIPES),
        len(OFFICIAL_RECIPES),
    ) != (145, 112, 112, 56, 233, 168, 401, 348):
        raise ValueError("Qwen3.8 NVFP4 source recipe is incomplete")
    ownership = (
        {"text/token_embedding"},
        set(FP8_WEIGHTS_BY_NAME),
        set(NVFP4_WEIGHTS_BY_NAME),
        set(INPUT_DIVISORS_BY_NAME),
        set(QUANTIZED_DIRECT_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME).difference({"text/token_embedding"}),
    )
    all_names: set[str] = set()
    for names in ownership:
        if all_names.intersection(names):
            raise ValueError("more than one source route owns an artifact tensor")
        all_names.update(names)
    if all_names != {spec.name for spec in inventory.TENSOR_SPECS}:
        raise ValueError("source routes do not cover the complete tensor inventory")
    if tuple(FP8_WEIGHTS_BY_NAME) != tuple(
        spec.name
        for spec in inventory.FP8_TENSOR_SPECS
        if spec.name != "text/token_embedding"
    ):
        raise ValueError("FP8 recipe order does not match inventory")
    if tuple(NVFP4_WEIGHTS_BY_NAME) != tuple(
        spec.name for spec in inventory.NVFP4_TENSOR_SPECS
    ):
        raise ValueError("NVFP4 recipe order does not match inventory")
    if tuple(INPUT_DIVISORS_BY_NAME) != tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    ):
        raise ValueError("input-divisor recipe order does not match inventory")
    for recipe in FP8_WEIGHT_RECIPES:
        _validate_matrix_recipe(recipe.object_name, recipe.shape, recipe.parts)
    for recipe in NVFP4_WEIGHT_RECIPES:
        _validate_matrix_recipe(recipe.object_name, recipe.shape, recipe.parts)
    bound_weights = tuple(
        name for site in INPUT_DIVISOR_RECIPES for name in site.weight_names
    )
    if (
        len(bound_weights) != 112
        or len(set(bound_weights)) != 112
        or set(bound_weights) != set(NVFP4_WEIGHTS_BY_NAME)
    ):
        raise ValueError("input-divisor sites do not cover NVFP4 parents once")


def _merge_requirement(
    result: dict[str, tuple[tuple[int, ...], str]],
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    signature = (shape, dtype)
    previous = result.setdefault(name, signature)
    if previous != signature:
        raise ValueError(f"inconsistent source declaration for {name}")


def _source_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in FP8_SOURCES:
        n, k = source.shape
        _merge_requirement(result, source.field("weight"), (n, k), "F8_E4M3")
        _merge_requirement(
            result, source.field("weight_scale"), (n, 1), "BF16"
        )
    for source in NVFP4_SOURCES:
        n, k = source.shape
        _merge_requirement(
            result, source.field("weight_packed"), (n, k // 2), "U8"
        )
        _merge_requirement(
            result, source.field("weight_scale"), (n, k // 16), "F8_E4M3"
        )
        _merge_requirement(
            result, source.field("weight_global_scale"), (1,), "F32"
        )
        _merge_requirement(
            result, source.field("input_global_scale"), (1,), "F32"
        )
    for source in family_recipe.source_requirements(
        QUANTIZED_DIRECT_RECIPES
    ).values():
        _merge_requirement(result, source.name, source.shape, source.dtype)
    return result


SOURCE_REQUIREMENTS = _source_requirements()
EXPECTED_QUANTIZED_FIELDS = frozenset(
    name
    for name, (_, dtype) in SOURCE_REQUIREMENTS.items()
    if dtype in ("F8_E4M3", "F32", "U8")
)


def preflight_quantized_metadata(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    missing = set(SOURCE_REQUIREMENTS).difference(reader.names)
    if missing:
        raise ValueError(f"quantized source is missing {sorted(missing)[0]}")

    for source in FP8_SOURCES:
        for suffix in (
            "weight_packed",
            "weight_global_scale",
            "input_global_scale",
        ):
            name = source.field(suffix)
            if reader.has(name):
                raise ValueError(f"FP8 source has forbidden field {name}")
    for source in NVFP4_SOURCES:
        if reader.has(source.field("weight")):
            raise ValueError(
                f"NVFP4 source has forbidden field {source.field('weight')}"
            )

    metadata = reader.metadata(reader.names)
    actual_quantized_fields = frozenset(
        name
        for name, item in metadata.items()
        if item.dtype in ("F8_E4M3", "F32", "U8")
    )
    if actual_quantized_fields != EXPECTED_QUANTIZED_FIELDS:
        unexpected = actual_quantized_fields.difference(EXPECTED_QUANTIZED_FIELDS)
        missing_fields = EXPECTED_QUANTIZED_FIELDS.difference(
            actual_quantized_fields
        )
        detail = (
            sorted(unexpected)[0]
            if unexpected
            else sorted(missing_fields)[0]
        )
        raise ValueError(f"quantized source allocation is not closed: {detail}")

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shards.add(actual.shard)
    return family_recipe.SourcePreflight(
        recipe_count=(
            len(FP8_WEIGHT_RECIPES)
            + len(NVFP4_WEIGHT_RECIPES)
            + len(INPUT_DIVISOR_RECIPES)
            + len(QUANTIZED_DIRECT_RECIPES)
        ),
        source_tensor_count=len(SOURCE_REQUIREMENTS),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def preflight_official_sources(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    return family_recipe.preflight_source_reader(reader, OFFICIAL_RECIPES)


def _word(tensor: torch.Tensor, name: str) -> int:
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError(f"{name}: divisor must be FP32[1]")
    word = int(tensor.detach().contiguous().cpu().view(torch.int32).item())
    word &= 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return word


def _same_divisor(
    reader: ShardReader,
    sources: Iterable[MatrixSource],
    suffix: str,
) -> int:
    items = tuple(sources)
    words = tuple(
        _word(reader.get(source.field(suffix)), source.field(suffix))
        for source in items
    )
    if len(set(words)) != 1:
        raise ValueError(f"{items[0].name}: fused {suffix} words differ")
    return words[0]


def _select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows)
        for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


def materialize_fp8_weight(
    recipe: Fp8WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor]:
    code_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe.parts:
        words = source_words.get(part.source)
        if words is None:
            source_codes = reader.get(part.source.field("weight"))
            source_scales = reader.get(
                part.source.field("weight_scale")
            ).reshape(-1)
            if (
                source_codes.dtype != torch.float8_e4m3fn
                or tuple(source_codes.shape) != part.source.shape
                or source_scales.dtype != torch.bfloat16
                or tuple(source_scales.shape) != (part.source.shape[0],)
            ):
                raise ValueError(
                    f"{part.source.name}: materialized FP8 source signature mismatch"
                )
            words = (source_codes.view(torch.uint8), source_scales)
            source_words[part.source] = words
        code_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    codes = (
        code_parts[0].contiguous()
        if len(code_parts) == 1
        else torch.cat(code_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    if tuple(codes.shape) != recipe.shape or tuple(scales.shape) != (
        recipe.shape[0],
    ):
        raise ValueError(f"{recipe.object_name}: materialized FP8 shape mismatch")
    return codes, scales


def materialize_nvfp4_weight(
    recipe: Nvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe.parts:
        words = source_words.get(part.source)
        if words is None:
            n, k = part.source.shape
            source_packed = reader.get(part.source.field("weight_packed"))
            source_scales = reader.get(part.source.field("weight_scale"))
            if (
                source_packed.dtype != torch.uint8
                or tuple(source_packed.shape) != (n, k // 2)
                or source_scales.dtype != torch.float8_e4m3fn
                or tuple(source_scales.shape) != (n, k // 16)
            ):
                raise ValueError(
                    f"{part.source.name}: materialized NVFP4 source signature mismatch"
                )
            words = (source_packed, source_scales.view(torch.uint8))
            source_words[part.source] = words
        packed_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    divisor = _same_divisor(
        reader, recipe.divisor_sources, "weight_global_scale"
    )
    if tuple(packed.shape) != (recipe.shape[0], recipe.shape[1] // 2) or tuple(
        scales.shape
    ) != (recipe.shape[0], recipe.shape[1] // 16):
        raise ValueError(
            f"{recipe.object_name}: materialized NVFP4 shape mismatch"
        )
    return packed, scales, struct.pack("<I", divisor)


def materialize_input_divisor(
    recipe: InputDivisorRecipe,
    reader: ShardReader,
) -> torch.Tensor:
    word = _same_divisor(reader, recipe.sources, "input_global_scale")
    return torch.frombuffer(
        bytearray(struct.pack("<I", word)), dtype=torch.float32
    ).reshape(())


def materialize_quantized_direct(
    object_name: str,
    reader: ShardReader,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        QUANTIZED_DIRECT_BY_NAME[object_name], reader
    )


def materialize_official(
    object_name: str,
    reader: ShardReader,
    derived_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        OFFICIAL_RECIPES_BY_NAME[object_name], reader, derived_tensors
    )


validate_recipe()


__all__ = [
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "EXPECTED_QUANTIZED_FIELDS",
    "FP8_SOURCES",
    "FP8_WEIGHT_RECIPES",
    "FP8_WEIGHTS_BY_NAME",
    "INPUT_DIVISOR_RECIPES",
    "INPUT_DIVISORS_BY_NAME",
    "NVFP4_SOURCES",
    "NVFP4_WEIGHT_RECIPES",
    "NVFP4_WEIGHTS_BY_NAME",
    "OFFICIAL_EMBEDDING_SOURCE",
    "OFFICIAL_RECIPES",
    "OFFICIAL_RECIPES_BY_NAME",
    "OFFICIAL_TENSOR_SPECS",
    "QUANTIZED_DIRECT_BY_NAME",
    "QUANTIZED_DIRECT_RECIPES",
    "QUANTIZED_DIRECT_SPECS",
    "QUANTIZED_REPOSITORY",
    "QUANTIZED_REVISION",
    "SOURCE_REQUIREMENTS",
    "WEIGHT_DIVISOR_GROUPS",
    "materialize_fp8_weight",
    "materialize_input_divisor",
    "materialize_nvfp4_weight",
    "materialize_official",
    "materialize_quantized_direct",
    "preflight_official_sources",
    "preflight_quantized_metadata",
    "validate_recipe",
]
