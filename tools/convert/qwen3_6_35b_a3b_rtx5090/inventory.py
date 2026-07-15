"""Persistent-object contract for Qwen3.6-35B-A3B on RTX 5090.

This target module owns the complete ordered inventory and the compiled
logical views that give its fused matrices meaning.  Source-checkpoint names
and transformations live in :mod:`recipe`.
"""

from __future__ import annotations

from collections import Counter

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    VISION_LAYERS,
    W8,
    build_vision_specs,
    tensor_spec,
)


MODEL_ID = "qwen3.6-35b-a3b"
TARGET_KEY = "qwen3_6_35b_a3b_rtx5090"

TEXT_LAYERS = tuple(range(40))
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
GDN_LAYERS = tuple(layer for layer in TEXT_LAYERS if layer not in FULL_ATTENTION_LAYERS)
Q6_ROUTED_DOWN_LAYERS = (34, 38, 39)

ROUTED_EXPERTS = 256
ROUTED_INTERMEDIATE = 512
ROUTED_GATE_UP_ROWS_PER_EXPERT = 2 * ROUTED_INTERMEDIATE
ROUTED_DOWN_ROWS_PER_EXPERT = 2048


def routed_gate_up_row(expert: int, projection: int, row: int) -> int:
    """Return the physical row for expert-local half-split gate/up storage."""

    if not 0 <= expert < ROUTED_EXPERTS:
        raise IndexError("routed expert id is outside 0..255")
    if projection not in (0, 1):
        raise IndexError("routed gate/up projection must be 0 or 1")
    if not 0 <= row < ROUTED_INTERMEDIATE:
        raise IndexError("routed intermediate row is outside 0..511")
    return (
        expert * ROUTED_GATE_UP_ROWS_PER_EXPERT
        + projection * ROUTED_INTERMEDIATE
        + row
    )


def routed_down_row(expert: int, hidden_row: int) -> int:
    """Return the physical row for an expert-major routed-down bank."""

    if not 0 <= expert < ROUTED_EXPERTS:
        raise IndexError("routed expert id is outside 0..255")
    if not 0 <= hidden_row < ROUTED_DOWN_ROWS_PER_EXPERT:
        raise IndexError("routed-down hidden row is outside 0..2047")
    return expert * ROUTED_DOWN_ROWS_PER_EXPERT + hidden_row


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


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(2048)

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
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


LOGICAL_ROW_VIEW_SPECS = (
    # Text full-attention [query,key,output-gate,value].
    LogicalRowViewSpec(
        "text/layers/{l}/attention/query",
        "text/layers/{l}/attention/query_key_gate_value",
        0,
        4096,
        (4096, 2048),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key_gate_value",
        4096,
        4608,
        (512, 2048),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/query_key_gate_value",
        4608,
        8704,
        (4096, 2048),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/query_key_gate_value",
        8704,
        9216,
        (512, 2048),
        FULL_ATTENTION_LAYERS,
    ),
    # GDN [query,key,value,z] and half-split A/B projections.
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key_value_z",
        0,
        2048,
        (2048, 2048),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key_value_z",
        2048,
        4096,
        (2048, 2048),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/query_key_value_z",
        4096,
        8192,
        (4096, 2048),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/query_key_value_z",
        8192,
        12288,
        (4096, 2048),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/a_projection",
        "text/layers/{l}/gdn/a_b_projection",
        0,
        32,
        (32, 2048),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/b_projection",
        "text/layers/{l}/gdn/a_b_projection",
        32,
        64,
        (32, 2048),
        GDN_LAYERS,
    ),
    # Text router/shared-gate and shared-expert half-split gate/up.
    LogicalRowViewSpec(
        "text/layers/{l}/moe/router",
        "text/layers/{l}/moe/router_shared_gate",
        0,
        256,
        (256, 2048),
        TEXT_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/moe/shared_expert_gate",
        "text/layers/{l}/moe/router_shared_gate",
        256,
        257,
        (1, 2048),
        TEXT_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/moe/shared_expert/gate",
        "text/layers/{l}/moe/shared_gate_up",
        0,
        512,
        (512, 2048),
        TEXT_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/moe/shared_expert/up",
        "text/layers/{l}/moe/shared_gate_up",
        512,
        1024,
        (512, 2048),
        TEXT_LAYERS,
    ),
    # MTP uses the same fused row order but has no layer-number placeholder.
    LogicalRowViewSpec(
        "mtp/layer/attention/query",
        "mtp/layer/attention/query_key_gate_value",
        0,
        4096,
        (4096, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/key",
        "mtp/layer/attention/query_key_gate_value",
        4096,
        4608,
        (512, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/output_gate",
        "mtp/layer/attention/query_key_gate_value",
        4608,
        8704,
        (4096, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/value",
        "mtp/layer/attention/query_key_gate_value",
        8704,
        9216,
        (512, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/moe/router",
        "mtp/layer/moe/router_shared_gate",
        0,
        256,
        (256, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/moe/shared_expert_gate",
        "mtp/layer/moe/router_shared_gate",
        256,
        257,
        (1, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/moe/shared_expert/gate",
        "mtp/layer/moe/shared_gate_up",
        0,
        512,
        (512, 2048),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/moe/shared_expert/up",
        "mtp/layer/moe/shared_gate_up",
        512,
        1024,
        (512, 2048),
        None,
    ),
)


ALIAS_SPECS = (
    LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
    LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
    LogicalAliasSpec(
        "mtp/optimized_proposal_head",
        ("text/draft_head", "text/draft_head_token_ids"),
    ),
    LogicalAliasSpec(
        "text/layers/{l}/gdn/channel_major_convolution",
        ("text/layers/{l}/gdn/convolution",),
        layers=GDN_LAYERS,
        axis_order=(1, 0),
    ),
)


def validate_inventory() -> None:
    """Reject drift from the accepted complete target contract at import time."""

    component_counts = (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(RESOURCE_SPECS),
        len(OBJECT_SPECS),
    )
    if component_counts != (533, 2, 15, 333, 883, 6, 889):
        raise ValueError(f"incomplete 35B inventory: {component_counts}")

    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("35B inventory contains duplicate object names")
    if FORMAT_COUNTS != {
        BF16: 461,
        FP32: 60,
        I32: 1,
        Q4: 95,
        Q5: 91,
        Q6: 5,
        W8: 170,
    }:
        raise ValueError(f"35B numeric-format counts drifted: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {CONTIGUOUS_LAYOUT: 522, ROW_SPLIT_LAYOUT: 361}:
        raise ValueError(f"35B storage-layout counts drifted: {LAYOUT_COUNTS}")
    if Counter(spec.format for spec in TENSOR_SPECS) != Counter(FORMAT_COUNTS):
        raise ValueError("35B tensor format counts are inconsistent")

    tensor_by_name = {spec.name: spec for spec in TENSOR_SPECS}
    for view in LOGICAL_ROW_VIEW_SPECS:
        layers = (None,) if view.layers is None else view.layers
        for layer in layers:
            parent_name = (
                view.parent_pattern
                if layer is None
                else view.parent_pattern.format(l=layer)
            )
            parent = tensor_by_name.get(parent_name)
            if parent is None or len(parent.shape) != 2:
                raise ValueError(f"invalid logical row-view parent: {parent_name}")
            if view.row_end > parent.shape[0] or view.shape != (
                view.row_count,
                parent.shape[1],
            ):
                raise ValueError(f"invalid logical row-view geometry: {view.name_pattern}")

    if routed_gate_up_row(255, 1, 511) != 262143:
        raise ValueError("routed gate/up row mapping does not cover the parent")
    if routed_down_row(255, 2047) != 524287:
        raise ValueError("routed-down row mapping does not cover the parent")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "LogicalAliasSpec",
    "LogicalRowViewSpec",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "Q6_ROUTED_DOWN_LAYERS",
    "RESOURCE_ENCODING",
    "RESOURCE_SPECS",
    "ROUTED_DOWN_ROWS_PER_EXPERT",
    "ROUTED_EXPERTS",
    "ROUTED_GATE_UP_ROWS_PER_EXPERT",
    "ROUTED_INTERMEDIATE",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TEXT_LAYERS",
    "TensorSpec",
    "VISION_LAYERS",
    "VISION_TENSOR_SPECS",
    "W8",
    "routed_down_row",
    "routed_gate_up_row",
    "validate_inventory",
]
