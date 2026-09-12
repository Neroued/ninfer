"""Source contract and recipes for the Qwen3.8-27B DFlash2 suffix."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe

from .dflash2_inventory import DFLASH2_LAYERS, DFLASH2_TENSOR_SPECS


REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"
REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"

_ROOT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "attention_bias": False,
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 5,
    "use_sliding_window": True,
    "max_window_layers": 5,
    "sliding_window": 2048,
    "vocab_size": 248320,
    "num_target_layers": 64,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "rope_type": "default",
}
_DRAFT_CONFIG = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61],
}


def _effective_optional(
    config: Mapping[str, object],
    draft: Mapping[str, object],
    name: str,
    default: object,
) -> object:
    if name in draft:
        return draft[name]
    return config.get(name, default)


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every checkpoint field consumed by the fixed DFlash2 runtime."""

    family_conversion.check_members("DFlash2 config", config, _ROOT_CONFIG)
    rope = config.get("rope_parameters")
    draft = config.get("dflash_config")
    if not isinstance(rope, Mapping) or not isinstance(draft, Mapping):
        raise ValueError(
            "DFlash2 config.json must contain rope_parameters and dflash_config"
        )
    family_conversion.check_members(
        "DFlash2 config.rope_parameters", rope, _ROPE_CONFIG
    )
    family_conversion.check_members(
        "DFlash2 config.dflash_config", draft, _DRAFT_CONFIG
    )

    sample_from_anchor = _effective_optional(
        config, draft, "sample_from_anchor", False
    )
    if sample_from_anchor is not False:
        raise ValueError("DFlash2 sample_from_anchor must resolve to false")

    input_embedding_scale = float(
        _effective_optional(config, draft, "input_embedding_scale", 1.0)
    )
    if input_embedding_scale != 1.0:
        raise ValueError("DFlash2 input_embedding_scale must resolve to 1.0")

    output_multiplier = float(
        _effective_optional(config, draft, "output_multiplier", 1.0)
    )
    if output_multiplier != 1.0:
        raise ValueError("DFlash2 output_multiplier must resolve to 1.0")

    softcap = _effective_optional(
        config, draft, "final_logit_softcapping", None
    )
    if softcap is not None and float(softcap) != 0.0:
        raise ValueError("DFlash2 final_logit_softcapping must be disabled")

    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "dtype": config["dtype"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
        "layer_types": list(config["layer_types"]),
        "is_causal": config["is_causal"],
        "sliding_window": config["sliding_window"],
        "vocab_size": config["vocab_size"],
        "num_target_layers": config["num_target_layers"],
        "max_position_embeddings": config["max_position_embeddings"],
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_parameters": {name: rope[name] for name in _ROPE_CONFIG},
        "dflash_config": {
            **{name: draft[name] for name in _DRAFT_CONFIG},
            "sample_from_anchor": sample_from_anchor,
            "input_embedding_scale": input_embedding_scale,
            "output_multiplier": output_multiplier,
            "final_logit_softcapping": None,
        },
    }


def validate_base_compatibility(
    base_summary: Mapping[str, object],
    dflash2_summary: Mapping[str, object],
) -> None:
    """Validate the shared geometry between target and DFlash2 checkpoints."""

    text = base_summary.get("text")
    rope = base_summary.get("rope")
    draft_rope = dflash2_summary.get("rope_parameters")
    draft_config = dflash2_summary.get("dflash_config")
    if not isinstance(text, Mapping) or not isinstance(rope, Mapping):
        raise ValueError("base config summary is missing text or rope facts")
    if not isinstance(draft_rope, Mapping) or not isinstance(
        draft_config, Mapping
    ):
        raise ValueError("DFlash2 config summary is incomplete")

    pairs = (
        ("hidden_size", text.get("hidden_size"), dflash2_summary.get("hidden_size")),
        ("vocab_size", text.get("vocab_size"), dflash2_summary.get("vocab_size")),
        (
            "num_target_layers",
            text.get("num_hidden_layers"),
            dflash2_summary.get("num_target_layers"),
        ),
        (
            "max_position_embeddings",
            text.get("max_position_embeddings"),
            dflash2_summary.get("max_position_embeddings"),
        ),
        ("rope_theta", rope.get("rope_theta"), draft_rope.get("rope_theta")),
    )
    for name, base_value, draft_value in pairs:
        if base_value != draft_value:
            raise ValueError(
                f"base/DFlash2 {name} mismatch: {base_value!r} != "
                f"{draft_value!r}"
            )

    target_layer_ids = draft_config.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or target_layer_ids != sorted(
        set(target_layer_ids)
    ):
        raise ValueError(
            "DFlash2 target_layer_ids must be unique and strictly increasing"
        )
    target_layers = int(dflash2_summary["num_target_layers"])
    if not target_layer_ids or not all(
        isinstance(layer, int) and 0 <= layer < target_layers
        for layer in target_layer_ids
    ):
        raise ValueError("DFlash2 target_layer_ids are outside the target")
    if not 0 <= int(draft_config["mask_token_id"]) < 248077:
        raise ValueError("DFlash2 mask token is not tokenizer-addressable")
    if len(target_layer_ids) * int(dflash2_summary["hidden_size"]) != 25600:
        raise ValueError("DFlash2 target-feature width does not match fc.weight")


def _build_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = [
        family_recipe.TensorRecipe(
            "dflash2/feature_projection",
            family_recipe.source("fc.weight", (5120, 25600)),
        ),
        family_recipe.TensorRecipe(
            "dflash2/context_norm",
            family_recipe.source("hidden_norm.weight", (5120,)),
        ),
    ]
    for layer in DFLASH2_LAYERS:
        source_prefix = f"layers.{layer}."
        object_prefix = f"dflash2/layers/{layer}/"
        recipes.extend(
            (
                family_recipe.TensorRecipe(
                    object_prefix + "input_norm",
                    family_recipe.source(
                        source_prefix + "input_layernorm.weight", (5120,)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention_conv/base_kernel",
                    family_recipe.source(
                        source_prefix + "attention_conv.base_kernel",
                        (2, 2, 5120),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention_conv/kernel_projection",
                    family_recipe.source(
                        source_prefix
                        + "attention_conv.kernel_projection.weight",
                        (1280, 5120),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/query_key_value",
                    family_recipe.Concat(
                        (
                            family_recipe.source(
                                source_prefix + "self_attn.q_proj.weight",
                                (4096, 5120),
                            ),
                            family_recipe.source(
                                source_prefix + "self_attn.k_proj.weight",
                                (1024, 5120),
                            ),
                            family_recipe.source(
                                source_prefix + "self_attn.v_proj.weight",
                                (1024, 5120),
                            ),
                        ),
                        0,
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/query_norm",
                    family_recipe.source(
                        source_prefix + "self_attn.q_norm.weight", (128,)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/key_norm",
                    family_recipe.source(
                        source_prefix + "self_attn.k_norm.weight", (128,)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/output",
                    family_recipe.source(
                        source_prefix + "self_attn.o_proj.weight",
                        (5120, 4096),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "post_attention_norm",
                    family_recipe.source(
                        source_prefix + "post_attention_layernorm.weight",
                        (5120,),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "mlp_conv/base_kernel",
                    family_recipe.source(
                        source_prefix + "mlp_conv.base_kernel", (2, 2, 5120)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "mlp_conv/kernel_projection",
                    family_recipe.source(
                        source_prefix + "mlp_conv.kernel_projection.weight",
                        (1280, 5120),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    family_recipe.Concat(
                        (
                            family_recipe.source(
                                source_prefix + "mlp.gate_proj.weight",
                                (17408, 5120),
                            ),
                            family_recipe.source(
                                source_prefix + "mlp.up_proj.weight",
                                (17408, 5120),
                            ),
                        ),
                        0,
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "mlp/down",
                    family_recipe.source(
                        source_prefix + "mlp.down_proj.weight",
                        (5120, 17408),
                    ),
                ),
            )
        )
    recipes.extend(
        (
            family_recipe.TensorRecipe(
                "dflash2/final_norm",
                family_recipe.source("norm.weight", (5120,)),
            ),
            family_recipe.TensorRecipe(
                "dflash2/candidate_selector/hidden_projection",
                family_recipe.source(
                    "candidate_selector.hidden_projection.weight",
                    (256, 5120),
                ),
            ),
            family_recipe.TensorRecipe(
                "dflash2/candidate_selector/predecessor_codebook",
                family_recipe.source(
                    "candidate_selector.predecessor_codebook", (248320, 256)
                ),
            ),
            family_recipe.TensorRecipe(
                "dflash2/candidate_selector/successor_codebook",
                family_recipe.source(
                    "candidate_selector.successor_codebook", (248320, 256)
                ),
            ),
        )
    )
    return tuple(recipes)


DFLASH2_RECIPE_SPECS = _build_recipes()
DFLASH2_RECIPES_BY_NAME = {
    item.object_name: item for item in DFLASH2_RECIPE_SPECS
}


def validate_recipe_coverage() -> None:
    family_recipe.validate_recipe_coverage(
        DFLASH2_RECIPE_SPECS, DFLASH2_TENSOR_SPECS
    )


def source_requirements() -> dict[str, family_recipe.SourceTensor]:
    return family_recipe.source_requirements(DFLASH2_RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> family_recipe.SourcePreflight:
    model = Path(model_dir)
    requirements = source_requirements()
    with ShardReader.from_file(model / "model.safetensors") as reader:
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
                "DFlash2 source inventory differs from its exact tensor contract"
                + (": " + ", ".join(details) if details else "")
            )
        return family_recipe.preflight_source_reader(
            reader, DFLASH2_RECIPE_SPECS
        )


def materialize_tensor(object_name: str, reader: ShardReader) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        DFLASH2_RECIPES_BY_NAME[object_name], reader
    )


validate_recipe_coverage()


__all__ = [
    "DFLASH2_RECIPES_BY_NAME",
    "DFLASH2_RECIPE_SPECS",
    "REPOSITORY",
    "REVISION",
    "materialize_tensor",
    "preflight_sources",
    "source_requirements",
    "validate_base_compatibility",
    "validate_config",
    "validate_recipe_coverage",
]
