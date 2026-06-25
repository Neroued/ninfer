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

from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.layouts import encode_tensor


@dataclass
class FixtureTensor:
    name: str
    qtype: int
    layout: int
    module_kind: int
    source_kind: int
    source_layer: int
    values: torch.Tensor


def make_values(shape: tuple[int, ...], offset: float) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    base = torch.arange(count, dtype=torch.float32).reshape(shape)
    return (base + offset) / 17.0


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


def build_file(out_path: Path) -> None:
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
        payload, logical, padded, group, scale_dtype = encode_tensor(
            spec.values, spec.qtype, spec.layout, torch.device("cpu")
        )
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

    header = fmt.FileHeaderFields(
        tensor_count=len(entries),
        module_count=len(records),
        layer_count=64,
        flags=1 | 2 | 4,
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
    args = parser.parse_args()
    build_file(Path(args.out))


if __name__ == "__main__":
    main()
