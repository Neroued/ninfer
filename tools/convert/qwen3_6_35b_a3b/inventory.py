"""Persistent-object contract for the Qwen3.6-35B-A3B target.

This target module owns the complete ordered inventory. Source-checkpoint
names and transformations live in :mod:`recipe`.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    build_vision_specs,
    tensor_spec,
)


MODEL_ID = "qwen3.6-35b-a3b"
TARGET_KEY = "qwen3_6_35b_a3b"

TEXT_LAYERS = tuple(range(40))
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
GDN_LAYERS = tuple(layer for layer in TEXT_LAYERS if layer not in FULL_ATTENTION_LAYERS)
Q6_ROUTED_DOWN_LAYERS = (34, 38, 39)
DFLASH_LAYERS = tuple(range(6))


def _routed_down_format(layer: int) -> str:
    return Q6 if layer in Q6_ROUTED_DOWN_LAYERS else Q5


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (248320, 2048), W8),
    ]

    for layer in TEXT_LAYERS:
        prefix = f"text/layers/{layer}/"
        specs.append(tensor_spec(prefix + "input_norm", (2048,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "attention/query_key_gate_value",
                        (9216, 2048),
                        W8,
                    ),
                    tensor_spec(prefix + "attention/query_norm", (256,), BF16),
                    tensor_spec(prefix + "attention/key_norm", (256,), BF16),
                    tensor_spec(prefix + "attention/output", (2048, 4096), W8),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "gdn/a_log", (32,), FP32),
                    tensor_spec(prefix + "gdn/dt_bias", (32,), FP32),
                    tensor_spec(prefix + "gdn/convolution", (4, 8192), BF16),
                    tensor_spec(prefix + "gdn/a_b_projection", (64, 2048), BF16),
                    tensor_spec(prefix + "gdn/query_key_value_z", (12288, 2048), W8),
                    tensor_spec(prefix + "gdn/norm", (128,), BF16),
                    tensor_spec(prefix + "gdn/output", (2048, 4096), W8),
                )
            )

        specs.extend(
            (
                tensor_spec(prefix + "post_attention_norm", (2048,), BF16),
                tensor_spec(prefix + "moe/router_shared_gate", (257, 2048), BF16),
                tensor_spec(prefix + "moe/routed_gate_up", (262144, 2048), Q4),
                tensor_spec(
                    prefix + "moe/routed_down",
                    (524288, 512),
                    _routed_down_format(layer),
                ),
                tensor_spec(prefix + "moe/shared_gate_up", (1024, 2048), W8),
                tensor_spec(prefix + "moe/shared_down", (2048, 512), W8),
            )
        )

    specs.extend(
        (
            tensor_spec("text/final_norm", (2048,), BF16),
            tensor_spec("text/output_head", (248320, 2048), Q6),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        tensor_spec("text/draft_head", (131072, 2048), Q4),
        tensor_spec("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        tensor_spec("mtp/input_projection", (2048, 4096), W8),
        tensor_spec("mtp/embedding_norm", (2048,), BF16),
        tensor_spec("mtp/hidden_norm", (2048,), BF16),
        tensor_spec("mtp/layer/input_norm", (2048,), BF16),
        tensor_spec(
            "mtp/layer/attention/query_key_gate_value", (9216, 2048), W8
        ),
        tensor_spec("mtp/layer/attention/query_norm", (256,), BF16),
        tensor_spec("mtp/layer/attention/key_norm", (256,), BF16),
        tensor_spec("mtp/layer/attention/output", (2048, 4096), W8),
        tensor_spec("mtp/layer/post_attention_norm", (2048,), BF16),
        tensor_spec("mtp/layer/moe/router_shared_gate", (257, 2048), BF16),
        tensor_spec("mtp/layer/moe/routed_gate_up", (262144, 2048), W8),
        tensor_spec("mtp/layer/moe/routed_down", (524288, 512), W8),
        tensor_spec("mtp/layer/moe/shared_gate_up", (1024, 2048), W8),
        tensor_spec("mtp/layer/moe/shared_down", (2048, 512), W8),
        tensor_spec("mtp/final_norm", (2048,), BF16),
    )


def _build_dflash_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("dflash/feature_projection", (2048, 16384), W8),
        tensor_spec("dflash/context_norm", (2048,), BF16),
    ]
    for layer in DFLASH_LAYERS:
        prefix = f"dflash/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "input_norm", (2048,), BF16),
                tensor_spec(
                    prefix + "attention/query_key_value",
                    (6144, 2048),
                    W8,
                ),
                tensor_spec(prefix + "attention/query_norm", (128,), BF16),
                tensor_spec(prefix + "attention/key_norm", (128,), BF16),
                tensor_spec(prefix + "attention/output", (2048, 4096), W8),
                tensor_spec(prefix + "post_attention_norm", (2048,), BF16),
                tensor_spec(prefix + "mlp/gate_up", (12288, 2048), W8),
                tensor_spec(prefix + "mlp/down", (2048, 6144), W8),
            )
        )
    specs.append(tensor_spec("dflash/final_norm", (2048,), BF16))
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(2048)
DFLASH_TENSOR_SPECS = _build_dflash_specs()

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
    + DFLASH_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


__all__ = [
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DFLASH_LAYERS",
    "DFLASH_TENSOR_SPECS",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "Q6_ROUTED_DOWN_LAYERS",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TEXT_LAYERS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
]
