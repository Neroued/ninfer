"""Declarative q5090 v3 tensor plan.

TEXT_CORE, MTP_DRAFT, and VISION_ENCODER are expressed as stored blocks,
logical segments, and fusion groups. Each segment carries the source TensorSpec
needed to materialize its rows before block-level concatenation and
quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from . import qtypes as qt


TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE = "gdn_conv1d_runtime_native"
TRANSFORM_ATTN_QPROJ_QUERY = "attn_qproj_query"
TRANSFORM_ATTN_QPROJ_GATE = "attn_qproj_gate"

# Full-attention head geometry (q_proj packs query+gate interleaved per head).
_ATTN_HEADS = 24
_ATTN_HEAD_DIM = 256

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
VOCAB_SIZE = 248320
TEXT_LAYER_COUNT = 64
FULL_ATTENTION_INTERVAL = 4

ATTN_Q_ROWS = _ATTN_HEADS * _ATTN_HEAD_DIM
ATTN_KV_ROWS = 4 * _ATTN_HEAD_DIM
ATTN_IN_BLOCK_ROWS = ATTN_Q_ROWS + ATTN_KV_ROWS
MTP_ATTN_IN_ROWS = (2 * ATTN_Q_ROWS) + (2 * ATTN_KV_ROWS)
MTP_MLP_GATEUP_ROWS = 2 * INTERMEDIATE_SIZE

GDN_KEY_ROWS = 16 * 128
GDN_VALUE_ROWS = 48 * 128
GDN_CONV_ROWS = 2 * GDN_KEY_ROWS + GDN_VALUE_ROWS
GDN_STATE_ROWS = 48
GDN_NORM_ROWS = 128
GDN_CONV_WIDTH = 4


@dataclass
class TensorSpec:
    name: str                      # canonical output name
    qtype: int
    layout: int
    module_kind: int
    src_name: str                  # safetensors source key
    source_kind: int
    source_layer: int = qt.NO_LAYER
    row_slice: Optional[Tuple[int, int]] = None   # take src.weight[start:end] rows
    reshape: Optional[Tuple[int, ...]] = None      # reshape source before encoding
    transform: Optional[str] = None


@dataclass(frozen=True)
class TensorSegmentSpec:
    block_index: int
    segment_index: int
    name: str
    source_kind: int
    source_layer: int
    row_begin: int
    row_count: int
    source: TensorSpec


@dataclass(frozen=True)
class TensorBlockSpec:
    block_index: int
    name: str
    qtype: int
    layout: int
    module_kind: int
    shape: Tuple[int, ...]
    source_layer: int
    source_kind: int
    segment_begin: int
    segments: Tuple[TensorSegmentSpec, ...]
    fusion_group_id: int = qt.FUSION_NONE
    fusion_index: int = 0

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class TensorFusionGroupSpec:
    group_id: int
    source_layer: int
    block_indices: Tuple[int, ...]
    shared_input_kind: int
    total_n: int
    shared_k: int

    @property
    def first_block_index(self) -> int:
        return self.block_indices[0]

    @property
    def block_count(self) -> int:
        return len(self.block_indices)


@dataclass(frozen=True)
class ExpectedManifest:
    blocks: Tuple[TensorBlockSpec, ...]
    segments: Tuple[TensorSegmentSpec, ...]
    fusion_groups: Tuple[TensorFusionGroupSpec, ...]

    def to_dict(self) -> dict:
        return {
            "block_count": len(self.blocks),
            "segment_count": len(self.segments),
            "fusion_group_count": len(self.fusion_groups),
            "blocks": [
                {
                    "block_index": b.block_index,
                    "name": b.name,
                    "qtype": b.qtype,
                    "layout": b.layout,
                    "module_kind": b.module_kind,
                    "shape": list(b.shape),
                    "source_layer": b.source_layer,
                    "source_kind": b.source_kind,
                    "segment_begin": b.segment_begin,
                    "segment_count": b.segment_count,
                    "fusion_group_id": b.fusion_group_id,
                    "fusion_index": b.fusion_index,
                }
                for b in self.blocks
            ],
            "segments": [
                {
                    "segment_index": s.segment_index,
                    "block_index": s.block_index,
                    "name": s.name,
                    "source_kind": s.source_kind,
                    "source_layer": s.source_layer,
                    "row_begin": s.row_begin,
                    "row_count": s.row_count,
                    "source": {
                        "src_name": s.source.src_name,
                        "row_slice": list(s.source.row_slice) if s.source.row_slice is not None else None,
                        "reshape": list(s.source.reshape) if s.source.reshape is not None else None,
                        "transform": s.source.transform,
                    },
                }
                for s in self.segments
            ],
            "fusion_groups": [
                {
                    "group_id": g.group_id,
                    "source_layer": g.source_layer,
                    "block_indices": list(g.block_indices),
                    "block_count": g.block_count,
                    "first_block_index": g.first_block_index,
                    "shared_input_kind": g.shared_input_kind,
                    "total_n": g.total_n,
                    "shared_k": g.shared_k,
                }
                for g in self.fusion_groups
            ],
        }


# GDN fused in_proj_qkv row splits: q=16*128, k=16*128, v=48*128 -> [10240, 5120]
# These ARE contiguous in the reference (torch.split on [q_all, k_all, v_all]).
_GDN_Q = (0, 2048)
_GDN_K = (2048, 4096)
_GDN_V = (4096, 10240)


def attn_qproj_split(t: "torch.Tensor", take_gate: bool):
    """Extract query or gate from the fused attention q_proj weight.

    The reference (modeling_qwen3_5.Qwen3_5Attention) computes
        q_proj(x).view(-1, head_dim*2) -> chunk(2, dim=-1) -> (query, gate)
    i.e. each head owns a contiguous [query(head_dim) | gate(head_dim)] block, so
    query/gate are interleaved per head across the output rows. A naive
    [0:n/2] / [n/2:] row split scrambles heads (cos ~0.04 vs the true query),
    which corrupts every full-attention layer. Recover the correct per-head
    halves here so `.q` holds the query of all heads and `.gate` the gate.
    """
    rows, hidden = t.shape
    head = _ATTN_HEAD_DIM
    if rows != _ATTN_HEADS * head * 2:
        raise ValueError(
            f"attn q_proj expected [{_ATTN_HEADS * head * 2}, H], got {tuple(t.shape)}"
        )
    t = t.reshape(_ATTN_HEADS, 2 * head, hidden)
    part = t[:, head:, :] if take_gate else t[:, :head, :]
    return part.reshape(_ATTN_HEADS * head, hidden).contiguous()


def runtime_native_gdn_conv1d(t: "torch.Tensor"):
    if t.dim() != 3 or t.shape[1] != 1:
        raise ValueError(f"gdn conv1d expected [C,1,K], got {tuple(t.shape)}")
    c = int(t.shape[0])
    k = int(t.shape[2])
    if k != 4:
        raise ValueError(f"gdn conv1d expected kernel width 4, got {k}")

    # q5090 Tensor uses dim0-fastest runtime order. The returned tensor deliberately
    # has metadata shape [C,K,1] while its flat bytes are [K,C].
    return t[:, 0, :].transpose(0, 1).contiguous().reshape(c, k, 1)


def _bf16(name, src, sk, layer=qt.NO_LAYER, module=qt.MODULE_TEXT) -> TensorSpec:
    return TensorSpec(name, qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, module, src, sk, layer)


def _w8(name, src, sk, layer, module) -> TensorSpec:
    return TensorSpec(name, qt.QT_W8G128, qt.LAYOUT_ROW_SPLIT, module, src, sk, layer)


def _text_src(layer: int, suffix: str) -> str:
    return f"model.language_model.layers.{layer}.{suffix}"


def _text_name(layer: int, suffix: str) -> str:
    return f"layers.{layer}.{suffix}"


def _source(
    name: str,
    qtype: int,
    layout: int,
    src_name: str,
    source_kind: int,
    source_layer: int,
    row_slice: Optional[Tuple[int, int]] = None,
    transform: Optional[str] = None,
    module_kind: int = qt.MODULE_TEXT,
) -> TensorSpec:
    return TensorSpec(
        name,
        qtype,
        layout,
        module_kind,
        src_name,
        source_kind,
        source_layer,
        row_slice=row_slice,
        transform=transform,
    )


def _append_block(
    blocks: List[TensorBlockSpec],
    flat_segments: List[TensorSegmentSpec],
    name: str,
    qtype: int,
    layout: int,
    shape: Tuple[int, ...],
    source_layer: int,
    segment_defs: List[Tuple[str, int, int, int, TensorSpec]],
    fusion_group_id: int = qt.FUSION_NONE,
    fusion_index: int = 0,
    module_kind: int = qt.MODULE_TEXT,
) -> TensorBlockSpec:
    block_index = len(blocks)
    segment_begin = len(flat_segments)
    segments: List[TensorSegmentSpec] = []
    row = 0
    for seg_name, source_kind, row_begin, row_count, source in segment_defs:
        if row_begin != row:
            raise ValueError(f"{name}: non-contiguous segment {seg_name} at row {row_begin}, expected {row}")
        segments.append(
            TensorSegmentSpec(
                block_index=block_index,
                segment_index=segment_begin + len(segments),
                name=seg_name,
                source_kind=source_kind,
                source_layer=source.source_layer,
                row_begin=row_begin,
                row_count=row_count,
                source=source,
            )
        )
        row += row_count
    if row != shape[0]:
        raise ValueError(f"{name}: segments cover {row} rows, expected {shape[0]}")

    source_kind = segments[0].source_kind if len(segments) == 1 else qt.SK_OTHER
    block = TensorBlockSpec(
        block_index=block_index,
        name=name,
        qtype=qtype,
        layout=layout,
        module_kind=module_kind,
        shape=shape,
        source_layer=source_layer,
        source_kind=source_kind,
        segment_begin=segment_begin,
        segments=tuple(segments),
        fusion_group_id=fusion_group_id,
        fusion_index=fusion_index,
    )
    blocks.append(block)
    flat_segments.extend(segments)
    return block


def _append_standalone_block(
    blocks: List[TensorBlockSpec],
    flat_segments: List[TensorSegmentSpec],
    name: str,
    qtype: int,
    layout: int,
    shape: Tuple[int, ...],
    src_name: str,
    source_kind: int,
    source_layer: int = qt.NO_LAYER,
    row_slice: Optional[Tuple[int, int]] = None,
    transform: Optional[str] = None,
    module_kind: int = qt.MODULE_TEXT,
) -> TensorBlockSpec:
    source = _source(
        name,
        qtype,
        layout,
        src_name,
        source_kind,
        source_layer,
        row_slice,
        transform,
        module_kind,
    )
    return _append_block(
        blocks,
        flat_segments,
        name,
        qtype,
        layout,
        shape,
        source_layer,
        [(name, source_kind, 0, int(shape[0]), source)],
        module_kind=module_kind,
    )


def _append_fusion_group(
    groups: List[TensorFusionGroupSpec],
    blocks: List[TensorBlockSpec],
    group_id: int,
    source_layer: int,
    first_block_index: int,
    block_count: int,
    total_n: int,
    shared_k: int,
) -> None:
    block_indices = tuple(range(first_block_index, first_block_index + block_count))
    for fusion_index, block_index in enumerate(block_indices):
        block = blocks[block_index]
        if block.fusion_group_id != group_id or block.fusion_index != fusion_index:
            raise ValueError(f"{block.name}: inconsistent fusion metadata")
        if block.source_layer != source_layer:
            raise ValueError(f"{block.name}: fusion layer mismatch")
        if block.shape[1] != shared_k:
            raise ValueError(f"{block.name}: fusion K mismatch")
    groups.append(
        TensorFusionGroupSpec(
            group_id=group_id,
            source_layer=source_layer,
            block_indices=block_indices,
            shared_input_kind=qt.SK_OTHER,
            total_n=total_n,
            shared_k=shared_k,
        )
    )


def build_text_manifest(layer_types: List[str]) -> ExpectedManifest:
    blocks: List[TensorBlockSpec] = []
    segments: List[TensorSegmentSpec] = []
    fusion_groups: List[TensorFusionGroupSpec] = []

    _append_standalone_block(
        blocks,
        segments,
        "model.language_model.embed_tokens.weight",
        qt.QT_Q6G64,
        qt.LAYOUT_ROW_SPLIT,
        (VOCAB_SIZE, HIDDEN_SIZE),
        "model.language_model.embed_tokens.weight",
        qt.SK_EMBED,
    )

    for li, lt in enumerate(layer_types):
        _append_standalone_block(
            blocks,
            segments,
            _text_name(li, "input_layernorm.weight"),
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            (HIDDEN_SIZE,),
            _text_src(li, "input_layernorm.weight"),
            qt.SK_INPUT_LAYERNORM,
            li,
        )

        if lt == "full_attention":
            qp = _text_src(li, "self_attn.q_proj.weight")
            attn_first = len(blocks)
            _append_block(
                blocks,
                segments,
                _text_name(li, "attn_in.q4"),
                qt.QT_Q4G64,
                qt.LAYOUT_ROW_SPLIT,
                (ATTN_IN_BLOCK_ROWS, HIDDEN_SIZE),
                li,
                [
                    (
                        _text_name(li, "self_attn.q_proj.q"),
                        qt.SK_ATTN_Q,
                        0,
                        ATTN_Q_ROWS,
                        _source(
                            _text_name(li, "self_attn.q_proj.q"),
                            qt.QT_Q4G64,
                            qt.LAYOUT_ROW_SPLIT,
                            qp,
                            qt.SK_ATTN_Q,
                            li,
                            transform=TRANSFORM_ATTN_QPROJ_QUERY,
                        ),
                    ),
                    (
                        _text_name(li, "self_attn.k_proj.weight"),
                        qt.SK_ATTN_K,
                        ATTN_Q_ROWS,
                        ATTN_KV_ROWS,
                        _source(
                            _text_name(li, "self_attn.k_proj.weight"),
                            qt.QT_Q4G64,
                            qt.LAYOUT_ROW_SPLIT,
                            _text_src(li, "self_attn.k_proj.weight"),
                            qt.SK_ATTN_K,
                            li,
                        ),
                    ),
                ],
                qt.FUSION_ATTN_IN,
                0,
            )
            _append_block(
                blocks,
                segments,
                _text_name(li, "attn_in.q5"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                (ATTN_IN_BLOCK_ROWS, HIDDEN_SIZE),
                li,
                [
                    (
                        _text_name(li, "self_attn.q_proj.gate"),
                        qt.SK_ATTN_GATE,
                        0,
                        ATTN_Q_ROWS,
                        _source(
                            _text_name(li, "self_attn.q_proj.gate"),
                            qt.QT_Q5G64,
                            qt.LAYOUT_ROW_SPLIT,
                            qp,
                            qt.SK_ATTN_GATE,
                            li,
                            transform=TRANSFORM_ATTN_QPROJ_GATE,
                        ),
                    ),
                    (
                        _text_name(li, "self_attn.v_proj.weight"),
                        qt.SK_ATTN_V,
                        ATTN_Q_ROWS,
                        ATTN_KV_ROWS,
                        _source(
                            _text_name(li, "self_attn.v_proj.weight"),
                            qt.QT_Q5G64,
                            qt.LAYOUT_ROW_SPLIT,
                            _text_src(li, "self_attn.v_proj.weight"),
                            qt.SK_ATTN_V,
                            li,
                        ),
                    ),
                ],
                qt.FUSION_ATTN_IN,
                1,
            )
            _append_fusion_group(
                fusion_groups,
                blocks,
                qt.FUSION_ATTN_IN,
                li,
                attn_first,
                2,
                2 * ATTN_IN_BLOCK_ROWS,
                HIDDEN_SIZE,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "self_attn.q_norm.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                (_ATTN_HEAD_DIM,),
                _text_src(li, "self_attn.q_norm.weight"),
                qt.SK_ATTN_Q_NORM,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "self_attn.k_norm.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                (_ATTN_HEAD_DIM,),
                _text_src(li, "self_attn.k_norm.weight"),
                qt.SK_ATTN_K_NORM,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "self_attn.o_proj.weight"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                (HIDDEN_SIZE, ATTN_Q_ROWS),
                _text_src(li, "self_attn.o_proj.weight"),
                qt.SK_ATTN_O,
                li,
            )
        elif lt == "linear_attention":
            qkv = _text_src(li, "linear_attn.in_proj_qkv.weight")
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.A_log"),
                qt.QT_FP32,
                qt.LAYOUT_CONTIGUOUS,
                (GDN_STATE_ROWS,),
                _text_src(li, "linear_attn.A_log"),
                qt.SK_GDN_A_LOG,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.dt_bias"),
                qt.QT_FP32,
                qt.LAYOUT_CONTIGUOUS,
                (GDN_STATE_ROWS,),
                _text_src(li, "linear_attn.dt_bias"),
                qt.SK_GDN_DT_BIAS,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.conv1d.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                (GDN_CONV_ROWS, GDN_CONV_WIDTH, 1),
                _text_src(li, "linear_attn.conv1d.weight"),
                qt.SK_GDN_CONV1D,
                li,
                transform=TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.in_proj_a.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                (GDN_STATE_ROWS, HIDDEN_SIZE),
                _text_src(li, "linear_attn.in_proj_a.weight"),
                qt.SK_GDN_IN_PROJ_A,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.in_proj_b.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                (GDN_STATE_ROWS, HIDDEN_SIZE),
                _text_src(li, "linear_attn.in_proj_b.weight"),
                qt.SK_GDN_IN_PROJ_B,
                li,
            )
            gdn_first = len(blocks)
            _append_block(
                blocks,
                segments,
                _text_name(li, "gdn_in.q4"),
                qt.QT_Q4G64,
                qt.LAYOUT_ROW_SPLIT,
                (2 * GDN_KEY_ROWS, HIDDEN_SIZE),
                li,
                [
                    (
                        _text_name(li, "linear_attn.in_proj_qkv.q"),
                        qt.SK_GDN_IN_PROJ_Q,
                        0,
                        GDN_KEY_ROWS,
                        _source(
                            _text_name(li, "linear_attn.in_proj_qkv.q"),
                            qt.QT_Q4G64,
                            qt.LAYOUT_ROW_SPLIT,
                            qkv,
                            qt.SK_GDN_IN_PROJ_Q,
                            li,
                            row_slice=_GDN_Q,
                        ),
                    ),
                    (
                        _text_name(li, "linear_attn.in_proj_qkv.k"),
                        qt.SK_GDN_IN_PROJ_K,
                        GDN_KEY_ROWS,
                        GDN_KEY_ROWS,
                        _source(
                            _text_name(li, "linear_attn.in_proj_qkv.k"),
                            qt.QT_Q4G64,
                            qt.LAYOUT_ROW_SPLIT,
                            qkv,
                            qt.SK_GDN_IN_PROJ_K,
                            li,
                            row_slice=_GDN_K,
                        ),
                    ),
                ],
                qt.FUSION_GDN_IN,
                0,
            )
            _append_block(
                blocks,
                segments,
                _text_name(li, "gdn_in.q5"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                (GDN_VALUE_ROWS, HIDDEN_SIZE),
                li,
                [
                    (
                        _text_name(li, "linear_attn.in_proj_qkv.v"),
                        qt.SK_GDN_IN_PROJ_V,
                        0,
                        GDN_VALUE_ROWS,
                        _source(
                            _text_name(li, "linear_attn.in_proj_qkv.v"),
                            qt.QT_Q5G64,
                            qt.LAYOUT_ROW_SPLIT,
                            qkv,
                            qt.SK_GDN_IN_PROJ_V,
                            li,
                            row_slice=_GDN_V,
                        ),
                    )
                ],
                qt.FUSION_GDN_IN,
                1,
            )
            _append_fusion_group(
                fusion_groups,
                blocks,
                qt.FUSION_GDN_IN,
                li,
                gdn_first,
                2,
                GDN_CONV_ROWS,
                HIDDEN_SIZE,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.norm.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                (GDN_NORM_ROWS,),
                _text_src(li, "linear_attn.norm.weight"),
                qt.SK_GDN_NORM,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.in_proj_z.weight"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                (GDN_VALUE_ROWS, HIDDEN_SIZE),
                _text_src(li, "linear_attn.in_proj_z.weight"),
                qt.SK_GDN_IN_PROJ_Z,
                li,
            )
            _append_standalone_block(
                blocks,
                segments,
                _text_name(li, "linear_attn.out_proj.weight"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                (HIDDEN_SIZE, GDN_VALUE_ROWS),
                _text_src(li, "linear_attn.out_proj.weight"),
                qt.SK_GDN_OUT_PROJ,
                li,
            )
        else:
            raise ValueError(f"layer {li}: unknown layer_type {lt!r}")

        _append_standalone_block(
            blocks,
            segments,
            _text_name(li, "post_attention_layernorm.weight"),
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            (HIDDEN_SIZE,),
            _text_src(li, "post_attention_layernorm.weight"),
            qt.SK_POST_ATTN_LAYERNORM,
            li,
        )
        mlp_first = len(blocks)
        _append_block(
            blocks,
            segments,
            _text_name(li, "mlp.gateup"),
            qt.QT_Q4G64,
            qt.LAYOUT_ROW_SPLIT,
            (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            li,
            [
                (
                    _text_name(li, "mlp.gate_proj.weight"),
                    qt.SK_MLP_GATE,
                    0,
                    INTERMEDIATE_SIZE,
                    _source(
                        _text_name(li, "mlp.gate_proj.weight"),
                        qt.QT_Q4G64,
                        qt.LAYOUT_ROW_SPLIT,
                        _text_src(li, "mlp.gate_proj.weight"),
                        qt.SK_MLP_GATE,
                        li,
                    ),
                ),
                (
                    _text_name(li, "mlp.up_proj.weight"),
                    qt.SK_MLP_UP,
                    INTERMEDIATE_SIZE,
                    INTERMEDIATE_SIZE,
                    _source(
                        _text_name(li, "mlp.up_proj.weight"),
                        qt.QT_Q4G64,
                        qt.LAYOUT_ROW_SPLIT,
                        _text_src(li, "mlp.up_proj.weight"),
                        qt.SK_MLP_UP,
                        li,
                    ),
                ),
            ],
            qt.FUSION_MLP_GATEUP,
            0,
        )
        _append_fusion_group(
            fusion_groups,
            blocks,
            qt.FUSION_MLP_GATEUP,
            li,
            mlp_first,
            1,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )
        _append_standalone_block(
            blocks,
            segments,
            _text_name(li, "mlp.down_proj.weight"),
            qt.QT_Q5G64,
            qt.LAYOUT_ROW_SPLIT,
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            _text_src(li, "mlp.down_proj.weight"),
            qt.SK_MLP_DOWN,
            li,
        )

    _append_standalone_block(
        blocks,
        segments,
        "model.language_model.norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (HIDDEN_SIZE,),
        "model.language_model.norm.weight",
        qt.SK_FINAL_NORM,
    )
    _append_standalone_block(
        blocks,
        segments,
        "lm_head.weight",
        qt.QT_Q6G64,
        qt.LAYOUT_ROW_SPLIT,
        (VOCAB_SIZE, HIDDEN_SIZE),
        "lm_head.weight",
        qt.SK_LM_HEAD,
    )

    return ExpectedManifest(tuple(blocks), tuple(segments), tuple(fusion_groups))


def build_text_specs(layer_types: List[str]) -> List[TensorBlockSpec]:
    return list(build_text_manifest(layer_types).blocks)


def _mtp_source(
    name: str,
    src_name: str,
    source_kind: int,
    source_layer: int,
    *,
    qtype: int = qt.QT_W8G128,
    layout: int = qt.LAYOUT_ROW_SPLIT,
    transform: Optional[str] = None,
) -> TensorSpec:
    return TensorSpec(
        name,
        qtype,
        layout,
        qt.MODULE_MTP,
        src_name,
        source_kind,
        source_layer,
        transform=transform,
    )


def build_mtp_manifest() -> ExpectedManifest:
    blocks: List[TensorBlockSpec] = []
    segments: List[TensorSegmentSpec] = []
    fusion_groups: List[TensorFusionGroupSpec] = []
    p = "mtp.layers.0."
    q_proj = p + "self_attn.q_proj.weight"

    _append_standalone_block(
        blocks,
        segments,
        "mtp.fc.weight",
        qt.QT_W8G128,
        qt.LAYOUT_ROW_SPLIT,
        (HIDDEN_SIZE, 2 * HIDDEN_SIZE),
        "mtp.fc.weight",
        qt.SK_MTP_FC,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        "mtp.pre_fc_norm_embedding.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (HIDDEN_SIZE,),
        "mtp.pre_fc_norm_embedding.weight",
        qt.SK_MTP_PRE_FC_NORM_EMB,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        "mtp.pre_fc_norm_hidden.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (HIDDEN_SIZE,),
        "mtp.pre_fc_norm_hidden.weight",
        qt.SK_MTP_PRE_FC_NORM_HID,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        p + "input_layernorm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (HIDDEN_SIZE,),
        p + "input_layernorm.weight",
        qt.SK_INPUT_LAYERNORM,
        0,
        module_kind=qt.MODULE_MTP,
    )

    attn_first = len(blocks)
    _append_block(
        blocks,
        segments,
        p + "attn_in.w8",
        qt.QT_W8G128,
        qt.LAYOUT_ROW_SPLIT,
        (MTP_ATTN_IN_ROWS, HIDDEN_SIZE),
        0,
        [
            (
                p + "self_attn.q_proj.q",
                qt.SK_ATTN_Q,
                0,
                ATTN_Q_ROWS,
                _mtp_source(
                    p + "self_attn.q_proj.q",
                    q_proj,
                    qt.SK_ATTN_Q,
                    0,
                    transform=TRANSFORM_ATTN_QPROJ_QUERY,
                ),
            ),
            (
                p + "self_attn.k_proj.weight",
                qt.SK_ATTN_K,
                ATTN_Q_ROWS,
                ATTN_KV_ROWS,
                _mtp_source(
                    p + "self_attn.k_proj.weight",
                    p + "self_attn.k_proj.weight",
                    qt.SK_ATTN_K,
                    0,
                ),
            ),
            (
                p + "self_attn.q_proj.gate",
                qt.SK_ATTN_GATE,
                ATTN_Q_ROWS + ATTN_KV_ROWS,
                ATTN_Q_ROWS,
                _mtp_source(
                    p + "self_attn.q_proj.gate",
                    q_proj,
                    qt.SK_ATTN_GATE,
                    0,
                    transform=TRANSFORM_ATTN_QPROJ_GATE,
                ),
            ),
            (
                p + "self_attn.v_proj.weight",
                qt.SK_ATTN_V,
                (2 * ATTN_Q_ROWS) + ATTN_KV_ROWS,
                ATTN_KV_ROWS,
                _mtp_source(
                    p + "self_attn.v_proj.weight",
                    p + "self_attn.v_proj.weight",
                    qt.SK_ATTN_V,
                    0,
                ),
            ),
        ],
        qt.FUSION_ATTN_IN,
        0,
        module_kind=qt.MODULE_MTP,
    )
    _append_fusion_group(
        fusion_groups,
        blocks,
        qt.FUSION_ATTN_IN,
        0,
        attn_first,
        1,
        MTP_ATTN_IN_ROWS,
        HIDDEN_SIZE,
    )

    _append_standalone_block(
        blocks,
        segments,
        p + "self_attn.q_norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (_ATTN_HEAD_DIM,),
        p + "self_attn.q_norm.weight",
        qt.SK_ATTN_Q_NORM,
        0,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        p + "self_attn.k_norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (_ATTN_HEAD_DIM,),
        p + "self_attn.k_norm.weight",
        qt.SK_ATTN_K_NORM,
        0,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        p + "self_attn.o_proj.weight",
        qt.QT_W8G128,
        qt.LAYOUT_ROW_SPLIT,
        (HIDDEN_SIZE, ATTN_Q_ROWS),
        p + "self_attn.o_proj.weight",
        qt.SK_ATTN_O,
        0,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        p + "post_attention_layernorm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (HIDDEN_SIZE,),
        p + "post_attention_layernorm.weight",
        qt.SK_POST_ATTN_LAYERNORM,
        0,
        module_kind=qt.MODULE_MTP,
    )

    mlp_first = len(blocks)
    _append_block(
        blocks,
        segments,
        p + "mlp.gateup.w8",
        qt.QT_W8G128,
        qt.LAYOUT_ROW_SPLIT,
        (MTP_MLP_GATEUP_ROWS, HIDDEN_SIZE),
        0,
        [
            (
                p + "mlp.gate_proj.weight",
                qt.SK_MLP_GATE,
                0,
                INTERMEDIATE_SIZE,
                _mtp_source(
                    p + "mlp.gate_proj.weight",
                    p + "mlp.gate_proj.weight",
                    qt.SK_MLP_GATE,
                    0,
                ),
            ),
            (
                p + "mlp.up_proj.weight",
                qt.SK_MLP_UP,
                INTERMEDIATE_SIZE,
                INTERMEDIATE_SIZE,
                _mtp_source(
                    p + "mlp.up_proj.weight",
                    p + "mlp.up_proj.weight",
                    qt.SK_MLP_UP,
                    0,
                ),
            ),
        ],
        qt.FUSION_MLP_GATEUP,
        0,
        module_kind=qt.MODULE_MTP,
    )
    _append_fusion_group(
        fusion_groups,
        blocks,
        qt.FUSION_MLP_GATEUP,
        0,
        mlp_first,
        1,
        MTP_MLP_GATEUP_ROWS,
        HIDDEN_SIZE,
    )
    _append_standalone_block(
        blocks,
        segments,
        p + "mlp.down_proj.weight",
        qt.QT_W8G128,
        qt.LAYOUT_ROW_SPLIT,
        (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        p + "mlp.down_proj.weight",
        qt.SK_MLP_DOWN,
        0,
        module_kind=qt.MODULE_MTP,
    )
    _append_standalone_block(
        blocks,
        segments,
        "mtp.norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        (HIDDEN_SIZE,),
        "mtp.norm.weight",
        qt.SK_MTP_NORM,
        module_kind=qt.MODULE_MTP,
    )

    return ExpectedManifest(tuple(blocks), tuple(segments), tuple(fusion_groups))


def build_vision_specs(depth: int) -> List[TensorSpec]:
    m = qt.MODULE_VISION

    def vrow_split(name, qtype, sk, layer):
        return TensorSpec("model.visual." + name, qtype, qt.LAYOUT_ROW_SPLIT, m,
                          "model.visual." + name, sk, layer)

    def vbf16(name, sk, layer=qt.NO_LAYER):
        return _bf16("model.visual." + name, "model.visual." + name, sk, layer, m)

    s: List[TensorSpec] = []
    # patch embed: Conv3d [1152,3,2,16,16] -> [1152, 1536]
    s.append(TensorSpec("model.visual.patch_embed.proj.weight", qt.QT_Q5G64, qt.LAYOUT_ROW_SPLIT, m,
                        "model.visual.patch_embed.proj.weight", qt.SK_VIS_PATCH_EMBED, qt.NO_LAYER,
                        reshape=(1152, 1536)))
    s.append(vbf16("patch_embed.proj.bias", qt.SK_VIS_PATCH_EMBED_BIAS))
    s.append(vbf16("pos_embed.weight", qt.SK_VIS_POS_EMBED))
    for b in range(depth):
        p = f"blocks.{b}."
        s += [
            vrow_split(p + "attn.qkv.weight", qt.QT_Q4G64, qt.SK_VIS_BLOCK_QKV, b),
            vbf16(p + "attn.qkv.bias", qt.SK_VIS_BLOCK_QKV_BIAS, b),
            vrow_split(p + "attn.proj.weight", qt.QT_Q5G64, qt.SK_VIS_BLOCK_PROJ, b),
            vbf16(p + "attn.proj.bias", qt.SK_VIS_BLOCK_PROJ_BIAS, b),
            vrow_split(p + "mlp.linear_fc1.weight", qt.QT_Q4G64, qt.SK_VIS_BLOCK_FC1, b),
            vbf16(p + "mlp.linear_fc1.bias", qt.SK_VIS_BLOCK_FC1_BIAS, b),
            vrow_split(p + "mlp.linear_fc2.weight", qt.QT_Q5G64, qt.SK_VIS_BLOCK_FC2, b),
            vbf16(p + "mlp.linear_fc2.bias", qt.SK_VIS_BLOCK_FC2_BIAS, b),
            vbf16(p + "norm1.weight", qt.SK_VIS_BLOCK_NORM1_W, b),
            vbf16(p + "norm1.bias", qt.SK_VIS_BLOCK_NORM1_B, b),
            vbf16(p + "norm2.weight", qt.SK_VIS_BLOCK_NORM2_W, b),
            vbf16(p + "norm2.bias", qt.SK_VIS_BLOCK_NORM2_B, b),
        ]
    s += [
        _w8("model.visual.merger.linear_fc1.weight", "model.visual.merger.linear_fc1.weight", qt.SK_VIS_MERGER_FC1, qt.NO_LAYER, m),
        vbf16("merger.linear_fc1.bias", qt.SK_VIS_MERGER_FC1_BIAS),
        _w8("model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.weight", qt.SK_VIS_MERGER_FC2, qt.NO_LAYER, m),
        vbf16("merger.linear_fc2.bias", qt.SK_VIS_MERGER_FC2_BIAS),
        vbf16("merger.norm.weight", qt.SK_VIS_MERGER_NORM_W),
        vbf16("merger.norm.bias", qt.SK_VIS_MERGER_NORM_B),
    ]
    return s
