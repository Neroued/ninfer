#!/usr/bin/env python3
"""Emit a tiny q5090 fixture through the real converter layout code."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.layouts import encode_tensor
from tools.q5090_convert.packing import pack_lowbit_groups


@dataclass
class FixtureTensor:
    name: str
    qtype: int
    layout: int
    module_kind: int
    source_kind: int
    source_layer: int
    values: torch.Tensor | None = None
    zero_shape: tuple[int, ...] | None = None
    random_shape: tuple[int, ...] | None = None
    random_seed: int = 0


def make_values(shape: tuple[int, ...], offset: float) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    base = torch.arange(count, dtype=torch.float32).reshape(shape)
    return (base + offset) / 17.0


def make_zero_values(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def make_constant_values(shape: tuple[int, ...], value: float) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def make_zero_spec(
    name: str,
    qtype: int,
    layout: int,
    module_kind: int,
    source_kind: int,
    source_layer: int,
    shape: tuple[int, ...],
) -> FixtureTensor:
    return FixtureTensor(name, qtype, layout, module_kind, source_kind, source_layer, None, shape)


def make_random_spec(
    name: str,
    qtype: int,
    layout: int,
    module_kind: int,
    source_kind: int,
    source_layer: int,
    shape: tuple[int, ...],
    seed: int,
) -> FixtureTensor:
    return FixtureTensor(
        name,
        qtype,
        layout,
        module_kind,
        source_kind,
        source_layer,
        None,
        None,
        shape,
        seed,
    )


def align_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def encode_zero_tensor(
    shape: tuple[int, ...], qtype: int, layout: int
) -> tuple[bytes, List[int], List[int], int, int]:
    logical = list(shape)
    if layout == qt.LAYOUT_CONTIGUOUS:
        if qtype == qt.QT_BF16:
            elem_size = 2
        elif qtype == qt.QT_FP32:
            elem_size = 4
        else:
            raise ValueError(f"zero contiguous qtype must be BF16/FP32, got {qtype}")
        count = 1
        for dim in shape:
            count *= dim
        return bytes(count * elem_size), logical, logical, 0, qt.SCALE_NONE

    if layout == qt.LAYOUT_TILE_N64_K64:
        if qtype not in (qt.QT_Q4G64, qt.QT_Q5G64, qt.QT_Q6G64):
            raise ValueError(f"zero tile64 qtype is unsupported: {qtype}")
        n, k = shape
        spec = qt.QUANT_SPECS[qtype]
        np_ = align_up(n, 64)
        kp = align_up(k, spec.group_size)
        nt = np_ // 64
        kg = kp // spec.group_size
        tilew = 64 * 2 + 64 * spec.bytes_per_group
        return bytes(nt * kg * tilew), logical, [np_, kp], spec.group_size, qt.SCALE_FP16

    if layout == qt.LAYOUT_TILE_N64_K128:
        if qtype != qt.QT_W8G128:
            raise ValueError(f"zero tile128 qtype is unsupported: {qtype}")
        n, k = shape
        spec = qt.QUANT_SPECS[qtype]
        np_ = align_up(n, 64)
        kp = align_up(k, spec.group_size)
        nt = np_ // 64
        kg = kp // spec.group_size
        tilew = 64 * 2 + 64 * spec.group_size
        return bytes(nt * kg * tilew), logical, [np_, kp], spec.group_size, qt.SCALE_FP16

    if layout == qt.LAYOUT_ROW_GROUPED_G64:
        if qtype not in (qt.QT_Q4G64, qt.QT_Q5G64, qt.QT_Q6G64):
            raise ValueError(f"zero row-grouped qtype is unsupported: {qtype}")
        n, k = shape
        spec = qt.QUANT_SPECS[qtype]
        kp = align_up(k, spec.group_size)
        kg = kp // spec.group_size
        roww = 2 + spec.bytes_per_group
        return bytes(n * kg * roww), logical, [n, kp], spec.group_size, qt.SCALE_FP16

    raise ValueError(f"unknown layout {layout}")


def encode_random_tensor(
    shape: tuple[int, ...], qtype: int, layout: int, seed: int
) -> tuple[bytes, List[int], List[int], int, int]:
    logical = list(shape)
    rng = np.random.default_rng(seed)
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
            return raw, logical, logical, 0, qt.SCALE_NONE
        if qtype == qt.QT_FP32:
            values = rng.uniform(-0.01, 0.01, size=shape).astype("<f4")
            return values.tobytes(), logical, logical, 0, qt.SCALE_NONE
        raise ValueError(f"random contiguous qtype must be BF16/FP32, got {qtype}")

    if layout != qt.LAYOUT_TILE_N64_K64:
        raise ValueError(f"random fixture only supports tile64/contiguous, got layout {layout}")
    if qtype not in (qt.QT_Q4G64, qt.QT_Q5G64, qt.QT_Q6G64):
        raise ValueError(f"random tile64 qtype is unsupported: {qtype}")

    n, k = shape
    spec = qt.QUANT_SPECS[qtype]
    np_ = align_up(n, 64)
    kp = align_up(k, spec.group_size)
    nt = np_ // 64
    kg = kp // spec.group_size
    bpr = spec.bytes_per_group
    tilew = 64 * 2 + 64 * bpr

    scales = rng.uniform(0.0002, 0.002, size=(np_, kg)).astype(np.float16)
    codes = rng.integers(
        spec.qmin, spec.qmax + 1, size=(np_, kg, spec.group_size), dtype=np.int16
    ).astype(np.int8)
    packed = pack_lowbit_groups(codes.reshape(np_ * kg, spec.group_size), spec.bits).reshape(
        np_, kg, bpr
    )
    scale_bytes = (
        scales.reshape(nt, 64, kg)
        .transpose(0, 2, 1)
        .copy()
        .view(np.uint8)
        .reshape(nt, kg, 64 * 2)
    )
    data_bytes = (
        packed.reshape(nt, 64, kg, bpr)
        .transpose(0, 2, 1, 3)
        .copy()
        .reshape(nt, kg, 64 * bpr)
    )
    payload = np.concatenate([scale_bytes, data_bytes], axis=2).reshape(nt * kg * tilew)
    return payload.tobytes(), logical, [np_, kp], spec.group_size, qt.SCALE_FP16


def encode_fixture_tensor(spec: FixtureTensor) -> tuple[bytes, List[int], List[int], int, int]:
    if spec.values is not None:
        return encode_tensor(spec.values, spec.qtype, spec.layout, torch.device("cpu"))
    if spec.zero_shape is not None:
        return encode_zero_tensor(spec.zero_shape, spec.qtype, spec.layout)
    if spec.random_shape is not None:
        return encode_random_tensor(spec.random_shape, spec.qtype, spec.layout, spec.random_seed)
    raise ValueError(f"fixture tensor has neither values nor zero shape: {spec.name}")


def build_specs() -> List[FixtureTensor]:
    no_layer = qt.NO_LAYER
    return [
        FixtureTensor(
            "model.language_model.embed_tokens.weight",
            qt.QT_Q6G64,
            qt.LAYOUT_ROW_GROUPED_G64,
            qt.MODULE_TEXT,
            qt.SK_EMBED,
            no_layer,
            make_values((3, 5), 1.0),
        ),
        FixtureTensor(
            "model.language_model.layers.0.mlp.gate_proj.weight",
            qt.QT_Q4G64,
            qt.LAYOUT_TILE_N64_K64,
            qt.MODULE_TEXT,
            qt.SK_MLP_GATE,
            0,
            make_values((5, 7), 2.0),
        ),
        FixtureTensor(
            "model.language_model.layers.0.mlp.down_proj.weight",
            qt.QT_Q5G64,
            qt.LAYOUT_TILE_N64_K64,
            qt.MODULE_TEXT,
            qt.SK_MLP_DOWN,
            0,
            make_values((6, 9), 3.0),
        ),
        FixtureTensor(
            "lm_head.weight",
            qt.QT_Q6G64,
            qt.LAYOUT_TILE_N64_K64,
            qt.MODULE_TEXT,
            qt.SK_LM_HEAD,
            no_layer,
            make_values((4, 8), 4.0),
        ),
        FixtureTensor(
            "model.language_model.layers.0.input_layernorm.weight",
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            qt.MODULE_TEXT,
            qt.SK_INPUT_LAYERNORM,
            0,
            make_values((4,), 5.0),
        ),
        FixtureTensor(
            "model.language_model.layers.0.linear_attn.A_log",
            qt.QT_FP32,
            qt.LAYOUT_CONTIGUOUS,
            qt.MODULE_TEXT,
            qt.SK_GDN_A_LOG,
            0,
            make_values((3,), 6.0),
        ),
        FixtureTensor(
            "mtp.fc.weight",
            qt.QT_W8G128,
            qt.LAYOUT_TILE_N64_K128,
            qt.MODULE_MTP,
            qt.SK_MTP_FC,
            no_layer,
            make_values((5, 9), 7.0),
        ),
        FixtureTensor(
            "mtp.norm.weight",
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            qt.MODULE_MTP,
            qt.SK_MTP_NORM,
            no_layer,
            make_values((4,), 8.0),
        ),
        FixtureTensor(
            "model.visual.patch_embed.proj.weight",
            qt.QT_Q5G64,
            qt.LAYOUT_TILE_N64_K64,
            qt.MODULE_VISION,
            qt.SK_VIS_PATCH_EMBED,
            no_layer,
            make_values((7, 11), 9.0),
        ),
        FixtureTensor(
            "model.visual.patch_embed.proj.bias",
            qt.QT_BF16,
            qt.LAYOUT_CONTIGUOUS,
            qt.MODULE_VISION,
            qt.SK_VIS_PATCH_EMBED_BIAS,
            no_layer,
            make_values((7,), 10.0),
        ),
    ]


def build_model_bind_specs(
    real_sample_blocks: bool = False, random_sample_blocks: bool = False
) -> List[FixtureTensor]:
    no_layer = qt.NO_LAYER
    specs = [
        FixtureTensor(
            "model.language_model.embed_tokens.weight",
            qt.QT_Q6G64,
            qt.LAYOUT_ROW_GROUPED_G64,
            qt.MODULE_TEXT,
            qt.SK_EMBED,
            no_layer,
            make_values((16, 8), 1.0),
        )
    ]

    for layer in range(64):
        p = f"model.language_model.layers.{layer}."
        sample_full = real_sample_blocks and layer == 3
        sample_gdn = real_sample_blocks and layer == 0
        sample_mlp = sample_full or sample_gdn
        sample_random = random_sample_blocks and (sample_full or sample_gdn)

        def control_values(shape, offset):
            if sample_random:
                return make_zero_values(shape)
            return make_values(shape, offset)

        def sampled_spec(name, qtype, layout, module, source, source_layer, shape, seed):
            if sample_random:
                return make_random_spec(name, qtype, layout, module, source, source_layer, shape, seed)
            return make_zero_spec(name, qtype, layout, module, source, source_layer, shape)

        specs.append(
            FixtureTensor(
                p + "input_layernorm.weight",
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                qt.SK_INPUT_LAYERNORM,
                layer,
                control_values((5120,), 100.0 + layer),
            )
        )
        if (layer + 1) % 4 == 0:
            specs.extend(
                [
                    sampled_spec(
                        p + "self_attn.q_proj.q",
                        qt.QT_Q4G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_Q,
                        layer,
                        (6144, 5120) if sample_full else (8, 8),
                        10000 + layer * 101 + qt.SK_ATTN_Q,
                    ),
                    sampled_spec(
                        p + "self_attn.q_proj.gate",
                        qt.QT_Q5G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_GATE,
                        layer,
                        (6144, 5120) if sample_full else (8, 8),
                        10000 + layer * 101 + qt.SK_ATTN_GATE,
                    ),
                    sampled_spec(
                        p + "self_attn.k_proj.weight",
                        qt.QT_Q4G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_K,
                        layer,
                        (1024, 5120) if sample_full else (8, 8),
                        10000 + layer * 101 + qt.SK_ATTN_K,
                    ),
                    sampled_spec(
                        p + "self_attn.v_proj.weight",
                        qt.QT_Q5G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_V,
                        layer,
                        (1024, 5120) if sample_full else (8, 8),
                        10000 + layer * 101 + qt.SK_ATTN_V,
                    ),
                    FixtureTensor(
                        p + "self_attn.q_norm.weight",
                        qt.QT_BF16,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_Q_NORM,
                        layer,
                        control_values((256,), 600.0 + layer),
                    ),
                    FixtureTensor(
                        p + "self_attn.k_norm.weight",
                        qt.QT_BF16,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_K_NORM,
                        layer,
                        control_values((256,), 700.0 + layer),
                    ),
                    sampled_spec(
                        p + "self_attn.o_proj.weight",
                        qt.QT_Q5G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_ATTN_O,
                        layer,
                        (5120, 6144) if sample_full else (8, 8),
                        10000 + layer * 101 + qt.SK_ATTN_O,
                    ),
                ]
            )
        else:
            specs.extend(
                [
                    FixtureTensor(
                        p + "linear_attn.A_log",
                        qt.QT_FP32,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_A_LOG,
                        layer,
                        control_values((48,), 900.0 + layer),
                    ),
                    FixtureTensor(
                        p + "linear_attn.dt_bias",
                        qt.QT_FP32,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_DT_BIAS,
                        layer,
                        control_values((48,), 1000.0 + layer),
                    ),
                    sampled_spec(
                        p + "linear_attn.conv1d.weight",
                        qt.QT_BF16,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_CONV1D,
                        layer,
                        (10240, 1, 4),
                        10000 + layer * 101 + qt.SK_GDN_CONV1D,
                    ),
                    sampled_spec(
                        p + "linear_attn.in_proj_a.weight",
                        qt.QT_BF16,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_IN_PROJ_A,
                        layer,
                        (48, 5120) if sample_gdn else (48, 8),
                        10000 + layer * 101 + qt.SK_GDN_IN_PROJ_A,
                    ),
                    sampled_spec(
                        p + "linear_attn.in_proj_b.weight",
                        qt.QT_BF16,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_IN_PROJ_B,
                        layer,
                        (48, 5120) if sample_gdn else (48, 8),
                        10000 + layer * 101 + qt.SK_GDN_IN_PROJ_B,
                    ),
                    sampled_spec(
                        p + "linear_attn.in_proj_qkv.q",
                        qt.QT_Q4G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_IN_PROJ_Q,
                        layer,
                        (2048, 5120) if sample_gdn else (8, 8),
                        10000 + layer * 101 + qt.SK_GDN_IN_PROJ_Q,
                    ),
                    sampled_spec(
                        p + "linear_attn.in_proj_qkv.k",
                        qt.QT_Q4G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_IN_PROJ_K,
                        layer,
                        (2048, 5120) if sample_gdn else (8, 8),
                        10000 + layer * 101 + qt.SK_GDN_IN_PROJ_K,
                    ),
                    sampled_spec(
                        p + "linear_attn.in_proj_qkv.v",
                        qt.QT_Q5G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_IN_PROJ_V,
                        layer,
                        (6144, 5120) if sample_gdn else (8, 8),
                        10000 + layer * 101 + qt.SK_GDN_IN_PROJ_V,
                    ),
                    sampled_spec(
                        p + "linear_attn.in_proj_z.weight",
                        qt.QT_Q5G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_IN_PROJ_Z,
                        layer,
                        (6144, 5120) if sample_gdn else (8, 8),
                        10000 + layer * 101 + qt.SK_GDN_IN_PROJ_Z,
                    ),
                    FixtureTensor(
                        p + "linear_attn.norm.weight",
                        qt.QT_BF16,
                        qt.LAYOUT_CONTIGUOUS,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_NORM,
                        layer,
                        make_constant_values((128,), 1.0)
                        if sample_random
                        else make_values((128,), 1800.0 + layer),
                    ),
                    sampled_spec(
                        p + "linear_attn.out_proj.weight",
                        qt.QT_Q5G64,
                        qt.LAYOUT_TILE_N64_K64,
                        qt.MODULE_TEXT,
                        qt.SK_GDN_OUT_PROJ,
                        layer,
                        (5120, 6144) if sample_gdn else (8, 8),
                        10000 + layer * 101 + qt.SK_GDN_OUT_PROJ,
                    ),
                ]
            )
        specs.extend(
            [
                FixtureTensor(
                    p + "post_attention_layernorm.weight",
                    qt.QT_BF16,
                    qt.LAYOUT_CONTIGUOUS,
                    qt.MODULE_TEXT,
                    qt.SK_POST_ATTN_LAYERNORM,
                    layer,
                    control_values((5120,), 2000.0 + layer),
                ),
                sampled_spec(
                    p + "mlp.gate_proj.weight",
                    qt.QT_Q4G64,
                    qt.LAYOUT_TILE_N64_K64,
                    qt.MODULE_TEXT,
                    qt.SK_MLP_GATE,
                    layer,
                    (17408, 5120) if sample_mlp else (8, 8),
                    10000 + layer * 101 + qt.SK_MLP_GATE,
                ),
                sampled_spec(
                    p + "mlp.up_proj.weight",
                    qt.QT_Q4G64,
                    qt.LAYOUT_TILE_N64_K64,
                    qt.MODULE_TEXT,
                    qt.SK_MLP_UP,
                    layer,
                    (17408, 5120) if sample_mlp else (8, 8),
                    10000 + layer * 101 + qt.SK_MLP_UP,
                ),
                sampled_spec(
                    p + "mlp.down_proj.weight",
                    qt.QT_Q5G64,
                    qt.LAYOUT_TILE_N64_K64,
                    qt.MODULE_TEXT,
                    qt.SK_MLP_DOWN,
                    layer,
                    (5120, 17408) if sample_mlp else (8, 8),
                    10000 + layer * 101 + qt.SK_MLP_DOWN,
                ),
            ]
        )

    specs.extend(
        [
            FixtureTensor(
                "model.language_model.norm.weight",
                qt.QT_BF16,
                qt.LAYOUT_CONTIGUOUS,
                qt.MODULE_TEXT,
                qt.SK_FINAL_NORM,
                no_layer,
                make_values((5120,), 2400.0),
            ),
            FixtureTensor(
                "lm_head.weight",
                qt.QT_Q6G64,
                qt.LAYOUT_TILE_N64_K64,
                qt.MODULE_TEXT,
                qt.SK_LM_HEAD,
                no_layer,
                make_values((16, 8), 2500.0),
            ),
        ]
    )
    return specs


def module_counts(specs: List[FixtureTensor]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for kind, policy in (
        (qt.MODULE_TEXT, qt.LOAD_RESIDENT),
        (qt.MODULE_MTP, qt.LOAD_RESIDENT),
        (qt.MODULE_VISION, qt.LOAD_LAZY_GPU),
    ):
        count = sum(1 for spec in specs if spec.module_kind == kind)
        if count:
            result.append((kind, count, policy))
    return result


def build_file(out_path: Path, profile: str) -> None:
    if profile == "model-bind":
        specs = build_model_bind_specs()
    elif profile == "model-blocks":
        specs = build_model_bind_specs(real_sample_blocks=True)
    elif profile == "model-blocks-random":
        specs = build_model_bind_specs(real_sample_blocks=True, random_sample_blocks=True)
    else:
        specs = build_specs()
    entries = [
        fmt.TensorEntry(
            name=spec.name,
            qtype=spec.qtype,
            layout=spec.layout,
            module_kind=spec.module_kind,
            shape=[],
            padded_shape=[],
            group_size=0,
            scale_dtype=qt.SCALE_NONE,
            source_layer=spec.source_layer,
            source_kind=spec.source_kind,
        )
        for spec in specs
    ]

    string_table = fmt.build_string_table(entries)
    modules = module_counts(specs)
    module_index_offset = fmt.HEADER_SIZE
    module_index_bytes = len(modules) * fmt.MODULE_RECORD_SIZE
    tensor_index_offset = module_index_offset + module_index_bytes
    tensor_index_bytes = len(entries) * fmt.TENSOR_ENTRY_SIZE
    string_table_offset = tensor_index_offset + tensor_index_bytes
    string_table_bytes = len(string_table)
    payload_offset = fmt.align_up(string_table_offset + string_table_bytes, fmt.REGION_ALIGN)

    payloads: list[bytes] = []
    for spec, entry in zip(specs, entries):
        payload, logical, padded, group, scale_dtype = encode_fixture_tensor(spec)
        entry.shape = logical
        entry.padded_shape = padded
        entry.group_size = group
        entry.scale_dtype = scale_dtype
        payloads.append(payload)

    file_bytes = bytearray(payload_offset)
    pos = payload_offset
    module_span: dict[int, list[int]] = {}
    for spec, entry, payload in zip(specs, entries, payloads):
        aligned = fmt.align_up(pos, fmt.PAYLOAD_ALIGN)
        if aligned > len(file_bytes):
            file_bytes.extend(b"\x00" * (aligned - len(file_bytes)))
        pos = aligned
        entry.payload_offset = pos
        entry.payload_bytes = len(payload)
        entry.crc32 = fmt.crc32(payload)
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
                module_version=1,
                tensor_index_begin=begin,
                tensor_index_count=count,
                payload_offset=span[0],
                payload_bytes=span[1] - span[0],
                load_policy=policy,
            )
        )
        begin += count

    header_flags = 0
    for kind, _, _ in modules:
        header_flags |= 1 << kind

    header = fmt.FileHeaderFields(
        tensor_count=len(entries),
        module_count=len(records),
        layer_count=64,
        flags=header_flags,
        module_index_offset=module_index_offset,
        module_index_bytes=module_index_bytes,
        tensor_index_offset=tensor_index_offset,
        tensor_index_bytes=tensor_index_bytes,
        string_table_offset=string_table_offset,
        string_table_bytes=string_table_bytes,
        payload_offset=payload_offset,
        payload_bytes=len(file_bytes) - payload_offset,
        hidden_size=5120,
        intermediate_size=17408,
        vocab_size=248320,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        gdn_key_heads=16,
        gdn_value_heads=48,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_width=4,
        full_attention_interval=4,
        max_position_embeddings=262144,
    )

    meta = bytearray(fmt.pack_header(header))
    for record in records:
        meta.extend(fmt.pack_module_record(record))
    for entry in entries:
        meta.extend(fmt.pack_tensor_entry(entry))
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
