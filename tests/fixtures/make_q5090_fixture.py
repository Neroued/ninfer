#!/usr/bin/env python3
"""Emit compact q5090 v3 fixtures through the real converter format helpers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.layouts import encode_tensor
from tools.q5090_convert.tensor_plan import runtime_native_gdn_conv1d


HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
N_LAYERS = 64
FULL_INTERVAL = 4
HEAD_DIM = 256
N_Q_HEADS = 24
N_KV_HEADS = 4
GDN_K_HEADS = 16
GDN_V_HEADS = 48
GDN_K_DIM = 128
GDN_V_DIM = 128
GDN_CONV_WIDTH = 4

ATTN_Q_ROWS = N_Q_HEADS * HEAD_DIM
ATTN_KV_ROWS = N_KV_HEADS * HEAD_DIM
GDN_K_ROWS = GDN_K_HEADS * GDN_K_DIM
GDN_V_ROWS = GDN_V_HEADS * GDN_V_DIM
GDN_CONV_ROWS = 2 * GDN_K_ROWS + GDN_V_ROWS


@dataclass
class SegmentSpec:
    name: str
    source_kind: int
    source_layer: int
    shape: tuple[int, ...]
    values: torch.Tensor | None = None
    random_seed: int | None = None


@dataclass
class BlockSpec:
    name: str
    qtype: int
    layout: int
    module_kind: int
    source_layer: int
    segments: list[SegmentSpec]
    fusion_group_id: int = qt.FUSION_NONE
    fusion_index: int = 0

    @property
    def source_kind(self) -> int:
        return self.segments[0].source_kind if len(self.segments) == 1 else qt.SK_OTHER

    @property
    def shape(self) -> tuple[int, ...]:
        if self.layout == qt.LAYOUT_CONTIGUOUS:
            return self.segments[0].shape
        rows = sum(s.shape[0] for s in self.segments)
        k = self.segments[0].shape[1]
        return (rows, k)


@dataclass(frozen=True)
class FusionSpec:
    group_id: int
    source_layer: int
    first_block_index: int
    block_count: int
    total_n: int
    shared_k: int
    shared_input_kind: int = qt.SK_OTHER


def align_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def make_values(shape: tuple[int, ...], offset: float) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    base = torch.arange(count, dtype=torch.float32).reshape(shape)
    return (base + offset) / 17.0


def make_constant_values(shape: tuple[int, ...], value: float) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def make_runtime_native_conv1d_values(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.01, 0.01, size=(GDN_CONV_ROWS, 1, GDN_CONV_WIDTH)).astype(np.float32)
    return runtime_native_gdn_conv1d(torch.from_numpy(values).to(torch.bfloat16))


def row_split_sizes(n: int, k: int, qtype: int) -> tuple[int, int, int, int, int, int, int, int]:
    spec = qt.QUANT_SPECS[qtype]
    padded_k = align_up(k, 128)
    groups = padded_k // spec.group_size
    nibble_plane_bytes = n * groups * spec.nibble_bytes_per_group
    high_plane_offset = align_up(nibble_plane_bytes, fmt.PAYLOAD_ALIGN)
    high_plane_bytes = n * groups * spec.high_bytes_per_group
    scale_plane_offset = high_plane_offset + align_up(high_plane_bytes, fmt.PAYLOAD_ALIGN)
    scale_plane_bytes = n * groups * 2
    payload_bytes = scale_plane_offset + scale_plane_bytes
    return (
        padded_k,
        groups,
        nibble_plane_bytes,
        high_plane_offset,
        high_plane_bytes,
        scale_plane_offset,
        scale_plane_bytes,
        payload_bytes,
    )


def encode_zero_tensor(shape: tuple[int, ...], qtype: int, layout: int):
    logical = list(shape)
    if layout == qt.LAYOUT_CONTIGUOUS:
        if qtype == qt.QT_BF16:
            elem_size = 2
        elif qtype == qt.QT_FP32:
            elem_size = 4
        else:
            raise ValueError(f"zero CONTIGUOUS qtype must be BF16/FP32, got {qtype}")
        count = 1
        for dim in shape:
            count *= dim
        raw = bytes(count * elem_size)
        return raw, logical, logical, 0, qt.SCALE_NONE, len(raw), 0, 0

    if layout != qt.LAYOUT_ROW_SPLIT:
        raise ValueError(f"unknown layout {layout}")
    n, k = shape
    spec = qt.QUANT_SPECS[qtype]
    padded_k, _, nibble_bytes, _, high_bytes, _, scale_bytes, payload_bytes = row_split_sizes(
        n, k, qtype
    )
    return (
        bytes(payload_bytes),
        logical,
        [n, padded_k],
        spec.group_size,
        qt.SCALE_FP16,
        nibble_bytes,
        high_bytes,
        scale_bytes,
    )


def encode_random_tensor(shape: tuple[int, ...], qtype: int, layout: int, seed: int):
    rng = np.random.default_rng(seed)
    logical = list(shape)
    if layout == qt.LAYOUT_CONTIGUOUS:
        if qtype == qt.QT_BF16:
            values = rng.uniform(-0.01, 0.01, size=shape).astype(np.float32)
            raw = (
                torch.from_numpy(values)
                .to(torch.bfloat16)
                .contiguous()
                .view(torch.int16)
                .numpy()
                .astype("<i2")
                .tobytes()
            )
            return raw, logical, logical, 0, qt.SCALE_NONE, len(raw), 0, 0
        if qtype == qt.QT_FP32:
            values = rng.uniform(-0.01, 0.01, size=shape).astype("<f4")
            raw = values.tobytes()
            return raw, logical, logical, 0, qt.SCALE_NONE, len(raw), 0, 0
        raise ValueError(f"random CONTIGUOUS qtype must be BF16/FP32, got {qtype}")

    if layout != qt.LAYOUT_ROW_SPLIT:
        raise ValueError(f"unknown layout {layout}")
    n, k = shape
    spec = qt.QUANT_SPECS[qtype]
    (
        padded_k,
        groups,
        nibble_bytes,
        high_off,
        high_bytes,
        scale_off,
        scale_bytes,
        payload_bytes,
    ) = row_split_sizes(n, k, qtype)
    nibble = rng.integers(0, 256, size=nibble_bytes, dtype=np.uint8).tobytes()
    high = rng.integers(0, 256, size=high_bytes, dtype=np.uint8).tobytes()
    scales = rng.uniform(0.0002, 0.002, size=n * groups).astype(np.float16).tobytes()
    payload = bytearray(payload_bytes)
    payload[:nibble_bytes] = nibble
    payload[high_off: high_off + high_bytes] = high
    payload[scale_off: scale_off + scale_bytes] = scales
    return (
        bytes(payload),
        logical,
        [n, padded_k],
        spec.group_size,
        qt.SCALE_FP16,
        nibble_bytes,
        high_bytes,
        scale_bytes,
    )


def encode_block(block: BlockSpec):
    if block.layout == qt.LAYOUT_CONTIGUOUS:
        segment = block.segments[0]
        if segment.values is not None:
            return encode_tensor(segment.values, block.qtype, block.layout, torch.device("cpu"))
        if segment.random_seed is not None:
            return encode_random_tensor(segment.shape, block.qtype, block.layout, segment.random_seed)
        return encode_zero_tensor(segment.shape, block.qtype, block.layout)

    if all(s.values is not None for s in block.segments):
        values = torch.cat([s.values for s in block.segments], dim=0)
        return encode_tensor(values, block.qtype, block.layout, torch.device("cpu"))

    random_seeds = [s.random_seed for s in block.segments if s.random_seed is not None]
    if random_seeds:
        return encode_random_tensor(block.shape, block.qtype, block.layout, random_seeds[0])
    return encode_zero_tensor(block.shape, block.qtype, block.layout)


def block(
    blocks: list[BlockSpec],
    name: str,
    qtype: int,
    layout: int,
    module_kind: int,
    source_layer: int,
    segments: list[SegmentSpec],
    fusion_group_id: int = qt.FUSION_NONE,
    fusion_index: int = 0,
) -> BlockSpec:
    if layout == qt.LAYOUT_CONTIGUOUS and len(segments) != 1:
        raise ValueError(f"{name}: CONTIGUOUS blocks must have one segment")
    if layout == qt.LAYOUT_ROW_SPLIT:
        k = segments[0].shape[1]
        for segment in segments:
            if len(segment.shape) != 2 or segment.shape[1] != k:
                raise ValueError(f"{name}: ROW_SPLIT segments must be 2D with a shared K")
    out = BlockSpec(name, qtype, layout, module_kind, source_layer, segments, fusion_group_id, fusion_index)
    blocks.append(out)
    return out


def seg(
    name: str,
    source_kind: int,
    source_layer: int,
    shape: tuple[int, ...],
    values: torch.Tensor | None = None,
    random_seed: int | None = None,
) -> SegmentSpec:
    return SegmentSpec(name, source_kind, source_layer, shape, values, random_seed)


def lname(layer: int, suffix: str) -> str:
    return f"layers.{layer}.{suffix}"


def small_quant(shape: tuple[int, int], source_kind: int, source_layer: int, name: str) -> SegmentSpec:
    return seg(name, source_kind, source_layer, shape)


def sampled_quant(
    shape: tuple[int, int],
    source_kind: int,
    source_layer: int,
    name: str,
    random_sample: bool,
    seed: int,
) -> SegmentSpec:
    return seg(name, source_kind, source_layer, shape, random_seed=seed if random_sample else None)


def sampled_dense(
    shape: tuple[int, ...],
    source_kind: int,
    source_layer: int,
    name: str,
    random_sample: bool,
    seed: int,
) -> SegmentSpec:
    return seg(name, source_kind, source_layer, shape, random_seed=seed if random_sample else None)


def add_compact_mtp(blocks: list[BlockSpec], fusions: list[FusionSpec]) -> None:
    no_layer = qt.NO_LAYER
    hidden = 8
    fc_in = 16
    q_rows = 8
    kv_rows = 4
    intermediate = 10
    head_dim = 4
    p = "mtp.layers.0."

    block(
        blocks,
        "mtp.fc.weight",
        qt.QT_W8G32,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_MTP,
        no_layer,
        [seg("mtp.fc.weight", qt.SK_MTP_FC, no_layer, (hidden, fc_in))],
    )
    block(
        blocks,
        "mtp.pre_fc_norm_embedding.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        no_layer,
        [seg("mtp.pre_fc_norm_embedding.weight", qt.SK_MTP_PRE_FC_NORM_EMB, no_layer, (hidden,))],
    )
    block(
        blocks,
        "mtp.pre_fc_norm_hidden.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        no_layer,
        [seg("mtp.pre_fc_norm_hidden.weight", qt.SK_MTP_PRE_FC_NORM_HID, no_layer, (hidden,))],
    )
    block(
        blocks,
        p + "input_layernorm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        0,
        [seg(p + "input_layernorm.weight", qt.SK_INPUT_LAYERNORM, 0, (hidden,))],
    )

    attn_first = len(blocks)
    block(
        blocks,
        p + "attn_in.w8",
        qt.QT_W8G32,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_MTP,
        0,
        [
            seg(p + "self_attn.q_proj.q", qt.SK_ATTN_Q, 0, (q_rows, hidden)),
            seg(p + "self_attn.k_proj.weight", qt.SK_ATTN_K, 0, (kv_rows, hidden)),
            seg(p + "self_attn.q_proj.gate", qt.SK_ATTN_GATE, 0, (q_rows, hidden)),
            seg(p + "self_attn.v_proj.weight", qt.SK_ATTN_V, 0, (kv_rows, hidden)),
        ],
        qt.FUSION_ATTN_IN,
        0,
    )
    fusions.append(FusionSpec(qt.FUSION_ATTN_IN, 0, attn_first, 1, 2 * q_rows + 2 * kv_rows, hidden))
    block(
        blocks,
        p + "self_attn.q_norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        0,
        [seg(p + "self_attn.q_norm.weight", qt.SK_ATTN_Q_NORM, 0, (head_dim,))],
    )
    block(
        blocks,
        p + "self_attn.k_norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        0,
        [seg(p + "self_attn.k_norm.weight", qt.SK_ATTN_K_NORM, 0, (head_dim,))],
    )
    block(
        blocks,
        p + "self_attn.o_proj.weight",
        qt.QT_W8G32,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_MTP,
        0,
        [seg(p + "self_attn.o_proj.weight", qt.SK_ATTN_O, 0, (hidden, q_rows))],
    )
    block(
        blocks,
        p + "post_attention_layernorm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        0,
        [seg(p + "post_attention_layernorm.weight", qt.SK_POST_ATTN_LAYERNORM, 0, (hidden,))],
    )

    mlp_first = len(blocks)
    block(
        blocks,
        p + "mlp.gateup.w8",
        qt.QT_W8G32,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_MTP,
        0,
        [
            seg(p + "mlp.gate_proj.weight", qt.SK_MLP_GATE, 0, (intermediate, hidden)),
            seg(p + "mlp.up_proj.weight", qt.SK_MLP_UP, 0, (intermediate, hidden)),
        ],
        qt.FUSION_MLP_GATEUP,
        0,
    )
    fusions.append(FusionSpec(qt.FUSION_MLP_GATEUP, 0, mlp_first, 1, 2 * intermediate, hidden))
    block(
        blocks,
        p + "mlp.down_proj.weight",
        qt.QT_W8G32,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_MTP,
        0,
        [seg(p + "mlp.down_proj.weight", qt.SK_MLP_DOWN, 0, (hidden, intermediate))],
    )
    block(
        blocks,
        "mtp.norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_MTP,
        no_layer,
        [seg("mtp.norm.weight", qt.SK_MTP_NORM, no_layer, (hidden,))],
    )


def build_default() -> tuple[list[BlockSpec], list[FusionSpec]]:
    blocks: list[BlockSpec] = []
    no_layer = qt.NO_LAYER
    block(
        blocks,
        "model.language_model.embed_tokens.weight",
        qt.QT_Q6G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_TEXT,
        no_layer,
        [seg("model.language_model.embed_tokens.weight", qt.SK_EMBED, no_layer, (3, 5), make_values((3, 5), 1.0))],
    )
    mlp_first = len(blocks)
    block(
        blocks,
        lname(0, "mlp.gateup"),
        qt.QT_Q4G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_TEXT,
        0,
        [
            seg(lname(0, "mlp.gate_proj.weight"), qt.SK_MLP_GATE, 0, (5, 7), make_values((5, 7), 2.0)),
            seg(lname(0, "mlp.up_proj.weight"), qt.SK_MLP_UP, 0, (4, 7), make_values((4, 7), 3.0)),
        ],
        qt.FUSION_MLP_GATEUP,
        0,
    )
    block(
        blocks,
        lname(0, "mlp.down_proj.weight"),
        qt.QT_Q5G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_TEXT,
        0,
        [seg(lname(0, "mlp.down_proj.weight"), qt.SK_MLP_DOWN, 0, (6, 9), make_values((6, 9), 4.0))],
    )
    block(
        blocks,
        "lm_head.weight",
        qt.QT_Q6G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_TEXT,
        no_layer,
        [seg("lm_head.weight", qt.SK_LM_HEAD, no_layer, (4, 8), make_values((4, 8), 5.0))],
    )
    block(
        blocks,
        lname(0, "input_layernorm.weight"),
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_TEXT,
        0,
        [seg(lname(0, "input_layernorm.weight"), qt.SK_INPUT_LAYERNORM, 0, (4,), make_values((4,), 6.0))],
    )
    block(
        blocks,
        lname(0, "linear_attn.A_log"),
        qt.QT_FP32,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_TEXT,
        0,
        [seg(lname(0, "linear_attn.A_log"), qt.SK_GDN_A_LOG, 0, (3,), make_values((3,), 7.0))],
    )
    fusions = [FusionSpec(qt.FUSION_MLP_GATEUP, 0, mlp_first, 1, 9, 7)]
    add_compact_mtp(blocks, fusions)
    block(
        blocks,
        "model.visual.patch_embed.proj.weight",
        qt.QT_Q5G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_VISION,
        no_layer,
        [seg("model.visual.patch_embed.proj.weight", qt.SK_VIS_PATCH_EMBED, no_layer, (7, 11), make_values((7, 11), 10.0))],
    )
    block(
        blocks,
        "model.visual.patch_embed.proj.bias",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_VISION,
        no_layer,
        [seg("model.visual.patch_embed.proj.bias", qt.SK_VIS_PATCH_EMBED_BIAS, no_layer, (7,), make_values((7,), 11.0))],
    )
    return blocks, fusions


def build_model_bind(real_sample_blocks: bool, random_sample_blocks: bool) -> tuple[list[BlockSpec], list[FusionSpec]]:
    blocks: list[BlockSpec] = []
    fusions: list[FusionSpec] = []
    no_layer = qt.NO_LAYER

    block(
        blocks,
        "model.language_model.embed_tokens.weight",
        qt.QT_Q6G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_TEXT,
        no_layer,
        [seg("model.language_model.embed_tokens.weight", qt.SK_EMBED, no_layer, (16, 8))],
    )

    for layer in range(N_LAYERS):
        sample_full = real_sample_blocks and (layer == 3)
        sample_gdn = real_sample_blocks and (layer == 0)
        sample_mlp = sample_full or sample_gdn
        random_sample = random_sample_blocks and (sample_full or sample_gdn)

        block(
            blocks,
            lname(layer, "input_layernorm.weight"),
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            qt.MODULE_TEXT,
            layer,
            [seg(lname(layer, "input_layernorm.weight"), qt.SK_INPUT_LAYERNORM, layer, (HIDDEN,), make_values((HIDDEN,), 100.0 + layer))],
        )

        if (layer + 1) % FULL_INTERVAL == 0:
            k = HIDDEN if sample_full else 8
            q_rows = ATTN_Q_ROWS if sample_full else 8
            kv_rows = ATTN_KV_ROWS if sample_full else 8
            attn_first = len(blocks)
            block(
                blocks,
                lname(layer, "attn_in.q4"),
                qt.QT_Q4G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [
                    sampled_quant((q_rows, k), qt.SK_ATTN_Q, layer, lname(layer, "self_attn.q_proj.q"), random_sample, 10000 + layer * 101 + qt.SK_ATTN_Q),
                    sampled_quant((kv_rows, k), qt.SK_ATTN_K, layer, lname(layer, "self_attn.k_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_ATTN_K),
                ],
                qt.FUSION_ATTN_IN,
                0,
            )
            block(
                blocks,
                lname(layer, "attn_in.q5"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [
                    sampled_quant((q_rows, k), qt.SK_ATTN_GATE, layer, lname(layer, "self_attn.q_proj.gate"), random_sample, 10000 + layer * 101 + qt.SK_ATTN_GATE),
                    sampled_quant((kv_rows, k), qt.SK_ATTN_V, layer, lname(layer, "self_attn.v_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_ATTN_V),
                ],
                qt.FUSION_ATTN_IN,
                1,
            )
            fusions.append(FusionSpec(qt.FUSION_ATTN_IN, layer, attn_first, 2, 2 * (q_rows + kv_rows), k))
            block(
                blocks,
                lname(layer, "self_attn.q_norm.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [seg(lname(layer, "self_attn.q_norm.weight"), qt.SK_ATTN_Q_NORM, layer, (HEAD_DIM,), make_values((HEAD_DIM,), 600.0 + layer))],
            )
            block(
                blocks,
                lname(layer, "self_attn.k_norm.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [seg(lname(layer, "self_attn.k_norm.weight"), qt.SK_ATTN_K_NORM, layer, (HEAD_DIM,), make_values((HEAD_DIM,), 700.0 + layer))],
            )
            o_shape = (HIDDEN, ATTN_Q_ROWS) if sample_full else (8, 8)
            block(
                blocks,
                lname(layer, "self_attn.o_proj.weight"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [sampled_quant(o_shape, qt.SK_ATTN_O, layer, lname(layer, "self_attn.o_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_ATTN_O)],
            )
        else:
            block(
                blocks,
                lname(layer, "linear_attn.A_log"),
                qt.QT_FP32,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [seg(lname(layer, "linear_attn.A_log"), qt.SK_GDN_A_LOG, layer, (48,), make_values((48,), 900.0 + layer))],
            )
            block(
                blocks,
                lname(layer, "linear_attn.dt_bias"),
                qt.QT_FP32,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [seg(lname(layer, "linear_attn.dt_bias"), qt.SK_GDN_DT_BIAS, layer, (48,), make_values((48,), 1000.0 + layer))],
            )
            block(
                blocks,
                lname(layer, "linear_attn.conv1d.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [seg(lname(layer, "linear_attn.conv1d.weight"), qt.SK_GDN_CONV1D, layer, (GDN_CONV_ROWS, GDN_CONV_WIDTH, 1), make_runtime_native_conv1d_values(10000 + layer * 101 + qt.SK_GDN_CONV1D))],
            )
            dense_k = HIDDEN if sample_gdn else 8
            block(
                blocks,
                lname(layer, "linear_attn.in_proj_a.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [sampled_dense((48, dense_k), qt.SK_GDN_IN_PROJ_A, layer, lname(layer, "linear_attn.in_proj_a.weight"), random_sample, 10000 + layer * 101 + qt.SK_GDN_IN_PROJ_A)],
            )
            block(
                blocks,
                lname(layer, "linear_attn.in_proj_b.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [sampled_dense((48, dense_k), qt.SK_GDN_IN_PROJ_B, layer, lname(layer, "linear_attn.in_proj_b.weight"), random_sample, 10000 + layer * 101 + qt.SK_GDN_IN_PROJ_B)],
            )
            k = HIDDEN if sample_gdn else 8
            q_rows = GDN_K_ROWS if sample_gdn else 8
            v_rows = GDN_V_ROWS if sample_gdn else 8
            gdn_first = len(blocks)
            block(
                blocks,
                lname(layer, "gdn_in.q4"),
                qt.QT_Q4G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [
                    sampled_quant((q_rows, k), qt.SK_GDN_IN_PROJ_Q, layer, lname(layer, "linear_attn.in_proj_qkv.q"), random_sample, 10000 + layer * 101 + qt.SK_GDN_IN_PROJ_Q),
                    sampled_quant((q_rows, k), qt.SK_GDN_IN_PROJ_K, layer, lname(layer, "linear_attn.in_proj_qkv.k"), random_sample, 10000 + layer * 101 + qt.SK_GDN_IN_PROJ_K),
                ],
                qt.FUSION_GDN_IN,
                0,
            )
            block(
                blocks,
                lname(layer, "gdn_in.q5"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [sampled_quant((v_rows, k), qt.SK_GDN_IN_PROJ_V, layer, lname(layer, "linear_attn.in_proj_qkv.v"), random_sample, 10000 + layer * 101 + qt.SK_GDN_IN_PROJ_V)],
                qt.FUSION_GDN_IN,
                1,
            )
            fusions.append(FusionSpec(qt.FUSION_GDN_IN, layer, gdn_first, 2, 2 * q_rows + v_rows, k))
            norm_values = make_constant_values((128,), 1.0) if random_sample else make_values((128,), 1800.0 + layer)
            block(
                blocks,
                lname(layer, "linear_attn.norm.weight"),
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                layer,
                [seg(lname(layer, "linear_attn.norm.weight"), qt.SK_GDN_NORM, layer, (128,), norm_values)],
            )
            z_shape = (GDN_V_ROWS, HIDDEN) if sample_gdn else (8, 8)
            block(
                blocks,
                lname(layer, "linear_attn.in_proj_z.weight"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [sampled_quant(z_shape, qt.SK_GDN_IN_PROJ_Z, layer, lname(layer, "linear_attn.in_proj_z.weight"), random_sample, 10000 + layer * 101 + qt.SK_GDN_IN_PROJ_Z)],
            )
            out_shape = (HIDDEN, GDN_V_ROWS) if sample_gdn else (8, 8)
            block(
                blocks,
                lname(layer, "linear_attn.out_proj.weight"),
                qt.QT_Q5G64,
                qt.LAYOUT_ROW_SPLIT,
                qt.MODULE_TEXT,
                layer,
                [sampled_quant(out_shape, qt.SK_GDN_OUT_PROJ, layer, lname(layer, "linear_attn.out_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_GDN_OUT_PROJ)],
            )

        block(
            blocks,
            lname(layer, "post_attention_layernorm.weight"),
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            qt.MODULE_TEXT,
            layer,
            [seg(lname(layer, "post_attention_layernorm.weight"), qt.SK_POST_ATTN_LAYERNORM, layer, (HIDDEN,), make_values((HIDDEN,), 2000.0 + layer))],
        )
        mlp_k = HIDDEN if sample_mlp else 8
        mlp_n = INTERMEDIATE if sample_mlp else 8
        mlp_first = len(blocks)
        block(
            blocks,
            lname(layer, "mlp.gateup"),
            qt.QT_Q4G64,
            qt.LAYOUT_ROW_SPLIT,
            qt.MODULE_TEXT,
            layer,
            [
                sampled_quant((mlp_n, mlp_k), qt.SK_MLP_GATE, layer, lname(layer, "mlp.gate_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_MLP_GATE),
                sampled_quant((mlp_n, mlp_k), qt.SK_MLP_UP, layer, lname(layer, "mlp.up_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_MLP_UP),
            ],
            qt.FUSION_MLP_GATEUP,
            0,
        )
        fusions.append(FusionSpec(qt.FUSION_MLP_GATEUP, layer, mlp_first, 1, 2 * mlp_n, mlp_k))
        down_shape = (HIDDEN, INTERMEDIATE) if sample_mlp else (8, 8)
        block(
            blocks,
            lname(layer, "mlp.down_proj.weight"),
            qt.QT_Q5G64,
            qt.LAYOUT_ROW_SPLIT,
            qt.MODULE_TEXT,
            layer,
            [sampled_quant(down_shape, qt.SK_MLP_DOWN, layer, lname(layer, "mlp.down_proj.weight"), random_sample, 10000 + layer * 101 + qt.SK_MLP_DOWN)],
        )

    block(
        blocks,
        "model.language_model.norm.weight",
        qt.QT_BF16,
        qt.LAYOUT_CONTIGUOUS,
        qt.MODULE_TEXT,
        no_layer,
        [seg("model.language_model.norm.weight", qt.SK_FINAL_NORM, no_layer, (HIDDEN,), make_values((HIDDEN,), 2400.0))],
    )
    block(
        blocks,
        "lm_head.weight",
        qt.QT_Q6G64,
        qt.LAYOUT_ROW_SPLIT,
        qt.MODULE_TEXT,
        no_layer,
        [seg("lm_head.weight", qt.SK_LM_HEAD, no_layer, (16, 8))],
    )
    add_compact_mtp(blocks, fusions)
    return blocks, fusions


def module_counts(blocks: Iterable[BlockSpec]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for kind, policy in (
        (qt.MODULE_TEXT, qt.LOAD_RESIDENT),
        (qt.MODULE_MTP, qt.LOAD_RESIDENT),
        (qt.MODULE_VISION, qt.LOAD_LAZY_GPU),
    ):
        count = sum(1 for b in blocks if b.module_kind == kind)
        if count:
            result.append((kind, count, policy))
    return result


def build_file(out_path: Path, profile: str) -> None:
    if profile == "model-bind":
        blocks, fusion_specs = build_model_bind(False, False)
    elif profile == "model-blocks":
        blocks, fusion_specs = build_model_bind(True, False)
    elif profile == "model-blocks-random":
        blocks, fusion_specs = build_model_bind(True, True)
    else:
        blocks, fusion_specs = build_default()

    entries: list[fmt.TensorEntry] = []
    segment_records: list[fmt.SegmentRecord] = []
    for block_index, spec in enumerate(blocks):
        segment_begin = len(segment_records)
        entries.append(
            fmt.TensorEntry(
                name=spec.name,
                qtype=spec.qtype,
                layout=spec.layout,
                module_kind=spec.module_kind,
                shape=[],
                padded_shape=[],
                group_size=0,
                scale_dtype=qt.SCALE_NONE,
                segment_count=len(spec.segments),
                source_layer=spec.source_layer,
                source_kind=spec.source_kind,
                segment_begin=segment_begin,
                fusion_group_id=spec.fusion_group_id,
                fusion_index=spec.fusion_index,
            )
        )
        row = 0
        for segment in spec.segments:
            row_count = segment.shape[0]
            segment_records.append(
                fmt.SegmentRecord(
                    name=segment.name,
                    source_kind=segment.source_kind,
                    source_layer=segment.source_layer,
                    row_begin=row,
                    row_count=row_count,
                )
            )
            row += row_count
        if len(entries) != block_index + 1:
            raise AssertionError("non-canonical block index")

    string_table = fmt.build_string_table(entries, segment_records)
    modules = module_counts(blocks)
    module_index_offset = fmt.HEADER_SIZE
    module_index_bytes = len(modules) * fmt.MODULE_RECORD_SIZE
    tensor_index_offset = module_index_offset + module_index_bytes
    tensor_index_bytes = len(entries) * fmt.TENSOR_ENTRY_SIZE
    segment_index_offset = tensor_index_offset + tensor_index_bytes
    segment_index_bytes = len(segment_records) * fmt.SEGMENT_RECORD_SIZE
    fusion_group_index_offset = segment_index_offset + segment_index_bytes
    fusion_group_index_bytes = len(fusion_specs) * fmt.FUSION_GROUP_RECORD_SIZE
    string_table_offset = fusion_group_index_offset + fusion_group_index_bytes
    string_table_bytes = len(string_table)
    payload_offset = fmt.align_up(string_table_offset + string_table_bytes, fmt.REGION_ALIGN)

    file_bytes = bytearray(payload_offset)
    pos = payload_offset
    module_span: dict[int, list[int]] = {}
    for spec, entry in zip(blocks, entries):
        aligned = fmt.align_up(pos, fmt.PAYLOAD_ALIGN)
        if aligned > len(file_bytes):
            file_bytes.extend(b"\x00" * (aligned - len(file_bytes)))
        pos = aligned
        (
            payload,
            logical,
            padded,
            group,
            scale_dtype,
            nibble_plane_bytes,
            high_plane_bytes,
            scale_plane_bytes,
        ) = encode_block(spec)
        entry.shape = logical
        entry.padded_shape = padded
        entry.group_size = group
        entry.scale_dtype = scale_dtype
        entry.payload_offset = pos
        entry.payload_bytes = len(payload)
        entry.crc32 = fmt.crc32(payload)
        entry.nibble_plane_bytes = nibble_plane_bytes
        entry.high_plane_bytes = high_plane_bytes
        entry.scale_plane_bytes = scale_plane_bytes
        file_bytes.extend(payload)
        end = pos + len(payload)
        span = module_span.setdefault(spec.module_kind, [pos, end])
        span[0] = min(span[0], pos)
        span[1] = max(span[1], end)
        pos = end

    records = []
    begin = 0
    for kind, count, policy in modules:
        span = module_span[kind]
        records.append(
            fmt.ModuleRecord(
                module_kind=kind,
                module_version=fmt.VERSION,
                tensor_index_begin=begin,
                tensor_index_count=count,
                payload_offset=span[0],
                payload_bytes=span[1] - span[0],
                load_policy=policy,
            )
        )
        begin += count

    fusion_records: list[fmt.FusionGroupRecord] = []
    for fusion in fusion_specs:
        first = entries[fusion.first_block_index]
        last = entries[fusion.first_block_index + fusion.block_count - 1]
        fusion_records.append(
            fmt.FusionGroupRecord(
                group_id=fusion.group_id,
                source_layer=fusion.source_layer,
                block_count=fusion.block_count,
                shared_input_kind=fusion.shared_input_kind,
                first_block_tensor_index=fusion.first_block_index,
                payload_offset=first.payload_offset,
                payload_bytes=last.payload_offset + last.payload_bytes - first.payload_offset,
                total_n=fusion.total_n,
                shared_k=fusion.shared_k,
            )
        )

    header_flags = 0
    for kind, _, _ in modules:
        header_flags |= 1 << kind

    header = fmt.FileHeaderFields(
        tensor_count=len(entries),
        module_count=len(records),
        layer_count=N_LAYERS,
        flags=header_flags,
        segment_count=len(segment_records),
        module_index_offset=module_index_offset,
        module_index_bytes=module_index_bytes,
        tensor_index_offset=tensor_index_offset,
        tensor_index_bytes=tensor_index_bytes,
        segment_index_offset=segment_index_offset,
        segment_index_bytes=segment_index_bytes,
        fusion_group_index_offset=fusion_group_index_offset,
        fusion_group_index_bytes=fusion_group_index_bytes,
        string_table_offset=string_table_offset,
        string_table_bytes=string_table_bytes,
        payload_offset=payload_offset,
        payload_bytes=len(file_bytes) - payload_offset,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        num_attention_heads=N_Q_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        gdn_key_heads=GDN_K_HEADS,
        gdn_value_heads=GDN_V_HEADS,
        gdn_key_head_dim=GDN_K_DIM,
        gdn_value_head_dim=GDN_V_DIM,
        gdn_conv_width=GDN_CONV_WIDTH,
        full_attention_interval=FULL_INTERVAL,
        max_position_embeddings=262144,
        fusion_group_count=len(fusion_records),
    )

    meta = bytearray(fmt.pack_header(header))
    for record in records:
        meta.extend(fmt.pack_module_record(record))
    for entry in entries:
        meta.extend(fmt.pack_tensor_entry(entry))
    for segment_record in segment_records:
        meta.extend(fmt.pack_segment_record(segment_record))
    for fusion_record in fusion_records:
        meta.extend(fmt.pack_fusion_group_record(fusion_record))
    meta.extend(string_table)
    if len(meta) > payload_offset:
        raise RuntimeError(f"metadata {len(meta)} exceeds payload offset {payload_offset}")
    meta.extend(b"\x00" * (payload_offset - len(meta)))
    file_bytes[:payload_offset] = meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(file_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--profile",
        choices=("default", "model-bind", "model-blocks", "model-blocks-random"),
        default="default",
    )
    args = parser.parse_args()
    build_file(Path(args.out), args.profile)


if __name__ == "__main__":
    main()
