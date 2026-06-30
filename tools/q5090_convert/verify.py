"""Verify a q5090_w4g64_mixed_v3 file with L0 structure checks and L1 value checks.

Usage:
  python -m tools.q5090_convert.verify out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus

L0 validates the binary ABI and plan conformance. L1 recovers ROW_SPLIT scales/codes from the
file and compares them bit-identically to tools.q5090_convert.quantize over the same block rows.
The verifier writes a deterministic structural dump to out/conv_dump.v3.json by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import format as fmt
from . import qtypes as qt
from .layouts import decode_row_split_quantized, decode_tensor, encode_tensor
from .packing import row_split_plane_sizes
from .quantize import pick_device, quantize_core
from .convert import ShardReader, assert_config, build_conversion_plan, load_config, materialize_block

DEFAULT_MODEL = "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16"


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _all_zero(buf: bytes) -> bool:
    return not buf or buf == b"\x00" * len(buf)


def _check_zero_range(f, begin: int, end: int, label: str, problems: List[str]) -> None:
    if end <= begin:
        return
    f.seek(begin)
    if not _all_zero(f.read(end - begin)):
        problems.append(f"{label}: nonzero reserved/padding bytes")


_TEXT_SOURCE_KINDS = {
    qt.SK_OTHER,
    qt.SK_EMBED,
    qt.SK_LM_HEAD,
    qt.SK_FINAL_NORM,
    qt.SK_INPUT_LAYERNORM,
    qt.SK_POST_ATTN_LAYERNORM,
    qt.SK_GDN_A_LOG,
    qt.SK_GDN_DT_BIAS,
    qt.SK_GDN_CONV1D,
    qt.SK_GDN_IN_PROJ_A,
    qt.SK_GDN_IN_PROJ_B,
    qt.SK_GDN_IN_PROJ_Q,
    qt.SK_GDN_IN_PROJ_K,
    qt.SK_GDN_IN_PROJ_V,
    qt.SK_GDN_IN_PROJ_Z,
    qt.SK_GDN_NORM,
    qt.SK_GDN_OUT_PROJ,
    qt.SK_ATTN_Q,
    qt.SK_ATTN_GATE,
    qt.SK_ATTN_K,
    qt.SK_ATTN_V,
    qt.SK_ATTN_Q_NORM,
    qt.SK_ATTN_K_NORM,
    qt.SK_ATTN_O,
    qt.SK_MLP_GATE,
    qt.SK_MLP_UP,
    qt.SK_MLP_DOWN,
}
_MTP_SOURCE_KINDS = {
    qt.SK_OTHER,
    qt.SK_INPUT_LAYERNORM,
    qt.SK_POST_ATTN_LAYERNORM,
    qt.SK_ATTN_Q,
    qt.SK_ATTN_GATE,
    qt.SK_ATTN_K,
    qt.SK_ATTN_V,
    qt.SK_ATTN_Q_NORM,
    qt.SK_ATTN_K_NORM,
    qt.SK_ATTN_O,
    qt.SK_MLP_GATE,
    qt.SK_MLP_UP,
    qt.SK_MLP_DOWN,
    qt.SK_MTP_FC,
    qt.SK_MTP_PRE_FC_NORM_EMB,
    qt.SK_MTP_PRE_FC_NORM_HID,
    qt.SK_MTP_NORM,
}
_VISION_SOURCE_KINDS = {
    qt.SK_OTHER,
    qt.SK_VIS_PATCH_EMBED,
    qt.SK_VIS_PATCH_EMBED_BIAS,
    qt.SK_VIS_POS_EMBED,
    qt.SK_VIS_BLOCK_QKV,
    qt.SK_VIS_BLOCK_QKV_BIAS,
    qt.SK_VIS_BLOCK_PROJ,
    qt.SK_VIS_BLOCK_PROJ_BIAS,
    qt.SK_VIS_BLOCK_FC1,
    qt.SK_VIS_BLOCK_FC1_BIAS,
    qt.SK_VIS_BLOCK_FC2,
    qt.SK_VIS_BLOCK_FC2_BIAS,
    qt.SK_VIS_BLOCK_NORM1_W,
    qt.SK_VIS_BLOCK_NORM1_B,
    qt.SK_VIS_BLOCK_NORM2_W,
    qt.SK_VIS_BLOCK_NORM2_B,
    qt.SK_VIS_MERGER_FC1,
    qt.SK_VIS_MERGER_FC1_BIAS,
    qt.SK_VIS_MERGER_FC2,
    qt.SK_VIS_MERGER_FC2_BIAS,
    qt.SK_VIS_MERGER_NORM_W,
    qt.SK_VIS_MERGER_NORM_B,
}
_SOURCE_KINDS_BY_MODULE = {
    qt.MODULE_TEXT: _TEXT_SOURCE_KINDS,
    qt.MODULE_MTP: _MTP_SOURCE_KINDS,
    qt.MODULE_VISION: _VISION_SOURCE_KINDS,
}


def _source_kind_defined(module_kind: int, source_kind: int) -> bool:
    return source_kind in _SOURCE_KINDS_BY_MODULE.get(module_kind, set())


def _read_name(table: bytes, offset: int, length: int, label: str, problems: List[str]) -> str:
    end = offset + length
    if offset < 0 or length < 0 or end > len(table):
        problems.append(f"{label}: string range [{offset},{end}) outside string table")
        return f"<bad:{label}>"
    if end >= len(table) or table[end] != 0:
        problems.append(f"{label}: string is not NUL-terminated")
    raw = table[offset:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"{label}: invalid utf-8 name: {exc}")
        return f"<bad:{label}>"


def _read_file(path: str):
    problems: List[str] = []
    with open(path, "rb") as f:
        hdr = fmt.unpack_header(f.read(fmt.HEADER_SIZE))
        f.seek(hdr["module_index_offset"])
        modules = [
            fmt.unpack_module_record(f.read(fmt.MODULE_RECORD_SIZE))
            for _ in range(hdr["module_count"])
        ]
        f.seek(hdr["tensor_index_offset"])
        entries = [
            fmt.unpack_tensor_entry(f.read(fmt.TENSOR_ENTRY_SIZE))
            for _ in range(hdr["tensor_count"])
        ]
        f.seek(hdr["segment_index_offset"])
        segments = [
            fmt.unpack_segment_record(f.read(fmt.SEGMENT_RECORD_SIZE))
            for _ in range(hdr["segment_count"])
        ]
        f.seek(hdr["fusion_group_index_offset"])
        fusions = [
            fmt.unpack_fusion_group_record(f.read(fmt.FUSION_GROUP_RECORD_SIZE))
            for _ in range(hdr["fusion_group_count"])
        ]
        f.seek(hdr["string_table_offset"])
        table = f.read(hdr["string_table_bytes"])

    for i, e in enumerate(entries):
        e["name"] = _read_name(table, e["name_offset"], e["name_len"], f"block[{i}]", problems)
        if fmt.fnv1a_64(e["name"]) != e["name_hash"]:
            problems.append(f"{e['name']}: name_hash mismatch")
    for i, s in enumerate(segments):
        s["name"] = _read_name(table, s["name_offset"], s["name_len"], f"segment[{i}]", problems)
        if fmt.fnv1a_64(s["name"]) != s["name_hash"]:
            problems.append(f"{s['name']}: segment name_hash mismatch")
    return hdr, modules, entries, segments, fusions, problems


def _header_dump(hdr: dict) -> dict:
    out = {}
    for k, v in hdr.items():
        if isinstance(v, bytes):
            out[k] = v.hex()
        else:
            out[k] = v
    return out


def _make_dump(path: str, hdr, modules, entries, segments, fusions) -> dict:
    blocks = []
    for i, e in enumerate(entries):
        begin = e["segment_begin"]
        count = e["segment_count"]
        block_segments = []
        if begin + count <= len(segments):
            for j, s in enumerate(segments[begin: begin + count]):
                block_segments.append(
                    {
                        "segment_index": begin + j,
                        "name": s["name"],
                        "source_kind": s["source_kind"],
                        "source_layer": s["source_layer"],
                        "row_begin": s["row_begin"],
                        "row_count": s["row_count"],
                    }
                )
        blocks.append(
            {
                "block_index": i,
                "name": e["name"],
                "source_kind": e["source_kind"],
                "source_layer": e["source_layer"],
                "qtype": qt.QTYPE_NAME.get(e["qtype"], str(e["qtype"])),
                "layout": qt.LAYOUT_NAME.get(e["layout"], str(e["layout"])),
                "shape": e["shape"],
                "padded_shape": e["padded_shape"],
                "payload_offset": e["payload_offset"],
                "payload_bytes": e["payload_bytes"],
                "nibble_plane_bytes": e["nibble_plane_bytes"],
                "high_plane_bytes": e["high_plane_bytes"],
                "scale_plane_bytes": e["scale_plane_bytes"],
                "crc32": e["crc32"],
                "fusion_group_id": e["fusion_group_id"],
                "fusion_index": e["fusion_index"],
                "segments": block_segments,
                "dequant_probes": [],
            }
        )

    fusion_dump = []
    for g in fusions:
        first = g["first_block_tensor_index"]
        count = g["block_count"]
        member_names = [
            entries[i]["name"]
            for i in range(first, min(first + count, len(entries)))
        ]
        fusion_dump.append(
            {
                "group_id": qt.FUSION_GROUP_NAME.get(g["group_id"], str(g["group_id"])),
                "source_layer": g["source_layer"],
                "block_count": g["block_count"],
                "shared_input_kind": g["shared_input_kind"],
                "first_block_tensor_index": first,
                "payload_offset": g["payload_offset"],
                "payload_bytes": g["payload_bytes"],
                "total_n": g["total_n"],
                "shared_k": g["shared_k"],
                "members": member_names,
            }
        )

    return {
        "format": "q5090_w4g64_mixed_v3",
        "file": path,
        "header": _header_dump(hdr),
        "modules": modules,
        "blocks": blocks,
        "fusion_groups": fusion_dump,
    }


def _check_entry_planes(e: dict) -> List[str]:
    problems: List[str] = []
    name = e["name"]
    if e["layout"] == qt.LAYOUT_ROW_SPLIT:
        if not qt.is_quant(e["qtype"]):
            return [f"{name}: ROW_SPLIT with non-quant qtype {e['qtype']}"]
        if len(e["shape"]) != 2 or len(e["padded_shape"]) != 2:
            return [f"{name}: ROW_SPLIT shape must be rank-2, got {e['shape']} / {e['padded_shape']}"]
        n, k = e["shape"]
        pn, pk = e["padded_shape"]
        spec = qt.QUANT_SPECS[e["qtype"]]
        if pn != n:
            problems.append(f"{name}: padded N {pn} != N {n}")
        if pk < k:
            problems.append(f"{name}: padded K {pk} < K {k}")
        if pk % spec.group_size != 0:
            problems.append(f"{name}: padded K {pk} not multiple of group {spec.group_size}")
        if e["group_size"] != spec.group_size:
            problems.append(f"{name}: group_size {e['group_size']} != {spec.group_size}")
        if e["scale_dtype"] != qt.SCALE_FP16:
            problems.append(f"{name}: scale_dtype {e['scale_dtype']} != FP16")
        want_pk = fmt.align_up(k, 128)
        if pk != want_pk:
            problems.append(f"{name}: padded K {pk} != align_up(K,128) {want_pk}")
        groups = pk // spec.group_size if spec.group_size else 0
        nib = spec.nibble_bytes_per_group
        high = spec.high_bytes_per_group
        nibble_bytes, _, high_bytes, _, scale, payload = row_split_plane_sizes(n, groups, nib, high)
        if e["nibble_plane_bytes"] != nibble_bytes:
            problems.append(f"{name}: nibble_plane_bytes {e['nibble_plane_bytes']} != {nibble_bytes}")
        if e["high_plane_bytes"] != high_bytes:
            problems.append(f"{name}: high_plane_bytes {e['high_plane_bytes']} != {high_bytes}")
        if e["scale_plane_bytes"] != scale:
            problems.append(f"{name}: scale_plane_bytes {e['scale_plane_bytes']} != {scale}")
        if e["payload_bytes"] != payload:
            problems.append(f"{name}: payload_bytes {e['payload_bytes']} != ROW_SPLIT size {payload}")
    elif e["layout"] == qt.LAYOUT_CONTIGUOUS:
        if e["qtype"] not in (qt.QT_BF16, qt.QT_FP32):
            problems.append(f"{name}: CONTIGUOUS with non-control qtype {e['qtype']}")
        if e["padded_shape"] != e["shape"]:
            problems.append(f"{name}: CONTIGUOUS padded_shape {e['padded_shape']} != shape {e['shape']}")
        element_bytes = 2 if e["qtype"] == qt.QT_BF16 else 4
        raw_bytes = _prod(e["shape"]) * element_bytes
        if e["group_size"] != 0:
            problems.append(f"{name}: CONTIGUOUS group_size {e['group_size']} != 0")
        if e["scale_dtype"] != qt.SCALE_NONE:
            problems.append(f"{name}: CONTIGUOUS scale_dtype {e['scale_dtype']} != none")
        if e["high_plane_bytes"] != 0:
            problems.append(f"{name}: CONTIGUOUS high_plane_bytes {e['high_plane_bytes']} != 0")
        if e["scale_plane_bytes"] != 0:
            problems.append(f"{name}: CONTIGUOUS scale_plane_bytes {e['scale_plane_bytes']} != 0")
        if e["nibble_plane_bytes"] != raw_bytes:
            problems.append(f"{name}: nibble_plane_bytes {e['nibble_plane_bytes']} != raw bytes {raw_bytes}")
        if e["payload_bytes"] != raw_bytes:
            problems.append(f"{name}: payload_bytes {e['payload_bytes']} != raw bytes {raw_bytes}")
    else:
        problems.append(f"{name}: unknown layout {e['layout']}")
    return problems


def _l0_checks(path: str, hdr, modules, entries, segments, fusions, plan) -> List[str]:
    problems: List[str] = []
    file_size = os.path.getsize(path)
    if hdr["magic"] != fmt.MAGIC:
        problems.append(f"bad magic {hdr['magic']!r}")
    if hdr["version"] != fmt.VERSION:
        problems.append(f"bad version {hdr['version']}")
    if hdr["endian"] != fmt.ENDIAN_TAG:
        problems.append(f"bad endian {hdr['endian']:#x}")
    if hdr["header_size"] != fmt.HEADER_SIZE:
        problems.append(f"bad header_size {hdr['header_size']}")
    if hdr["module_count"] != 3:
        problems.append(f"module_count {hdr['module_count']} != 3")
    if hdr["layer_count"] != 64:
        problems.append(f"layer_count {hdr['layer_count']} != 64")
    if hdr["format_minor"] != fmt.FORMAT_MINOR:
        problems.append(f"format_minor {hdr['format_minor']} != {fmt.FORMAT_MINOR}")
    if hdr["flags"] & fmt.FLAG_RESERVED_MASK:
        problems.append(f"reserved header flags set: {hdr['flags']:#x}")
    if hdr["flags"] != fmt.FLAG_MODULE_PRESENT_MASK:
        problems.append(f"flags {hdr['flags']:#x} != full module flags {fmt.FLAG_MODULE_PRESENT_MASK:#x}")
    if hdr["tensor_count"] != 1167:
        problems.append(f"tensor_count {hdr['tensor_count']} != full artifact count 1167")
    if hdr["segment_count"] != 1311:
        problems.append(f"segment_count {hdr['segment_count']} != full artifact count 1311")
    if hdr["fusion_group_count"] != 128:
        problems.append(f"fusion_group_count {hdr['fusion_group_count']} != full artifact count 128")
    if hdr["tensor_count"] != len(entries):
        problems.append(f"tensor_count {hdr['tensor_count']} != table entries {len(entries)}")
    if hdr["segment_count"] != len(segments):
        problems.append(f"segment_count {hdr['segment_count']} != table entries {len(segments)}")
    if hdr["fusion_group_count"] != len(fusions):
        problems.append(f"fusion_group_count {hdr['fusion_group_count']} != table entries {len(fusions)}")
    present_mask = 0
    for m in modules:
        if m["module_kind"] in (qt.MODULE_TEXT, qt.MODULE_MTP, qt.MODULE_VISION):
            present_mask |= 1 << m["module_kind"]
        else:
            problems.append(f"unknown module_kind {m['module_kind']}")
    if (hdr["flags"] & fmt.FLAG_MODULE_PRESENT_MASK) != present_mask:
        problems.append(
            f"module present flags {hdr['flags'] & fmt.FLAG_MODULE_PRESENT_MASK:#x} != module table {present_mask:#x}"
        )
    if not (hdr["flags"] & fmt.FLAG_TEXT_PRESENT):
        problems.append("TEXT_PRESENT flag is not set")

    expected_offsets = [
        ("module_index_offset", fmt.HEADER_SIZE),
        ("module_index_bytes", hdr["module_count"] * fmt.MODULE_RECORD_SIZE),
        ("tensor_index_offset", hdr["module_index_offset"] + hdr["module_index_bytes"]),
        ("tensor_index_bytes", hdr["tensor_count"] * fmt.TENSOR_ENTRY_SIZE),
        ("segment_index_offset", hdr["tensor_index_offset"] + hdr["tensor_index_bytes"]),
        ("segment_index_bytes", hdr["segment_count"] * fmt.SEGMENT_RECORD_SIZE),
        ("fusion_group_index_offset", hdr["segment_index_offset"] + hdr["segment_index_bytes"]),
        ("fusion_group_index_bytes", hdr["fusion_group_count"] * fmt.FUSION_GROUP_RECORD_SIZE),
        ("string_table_offset", hdr["fusion_group_index_offset"] + hdr["fusion_group_index_bytes"]),
    ]
    for key, want in expected_offsets:
        if hdr[key] != want:
            problems.append(f"{key} {hdr[key]} != {want}")
    string_end = hdr["string_table_offset"] + hdr["string_table_bytes"]
    want_payload = fmt.align_up(string_end, fmt.REGION_ALIGN)
    if hdr["payload_offset"] != want_payload:
        problems.append(f"payload_offset {hdr['payload_offset']} != align_up(string table end) {want_payload}")
    if hdr["payload_offset"] % fmt.REGION_ALIGN != 0:
        problems.append(f"payload_offset {hdr['payload_offset']} not {fmt.REGION_ALIGN}-aligned")
    if hdr["payload_bytes"] != file_size - hdr["payload_offset"]:
        problems.append(f"payload_bytes {hdr['payload_bytes']} != file payload {file_size - hdr['payload_offset']}")
    if hdr["payload_offset"] > file_size:
        problems.append(f"payload_offset {hdr['payload_offset']} > file size {file_size}")

    module_kinds = [m["module_kind"] for m in modules]
    if module_kinds != [qt.MODULE_TEXT, qt.MODULE_MTP, qt.MODULE_VISION]:
        problems.append(f"module kinds {module_kinds} != full artifact modules [TEXT, MTP, VISION]")
    if module_kinds != sorted(module_kinds) or len(set(module_kinds)) != len(module_kinds):
        problems.append(f"module kinds are not distinct TEXT->MTP->VISION ordered: {module_kinds}")
    next_tensor = 0
    for i, got in enumerate(modules):
        if got["flags"] != 0:
            problems.append(f"module[{i}]: flags {got['flags']} != 0")
        if got["tensor_index_begin"] != next_tensor:
            problems.append(f"module[{i}]: tensor range begins at {got['tensor_index_begin']}, expected {next_tensor}")
        next_tensor = got["tensor_index_begin"] + got["tensor_index_count"]
    if next_tensor != len(entries):
        problems.append(f"module tensor ranges end at {next_tensor}, expected {len(entries)}")

    if len(modules) != len(plan.modules):
        problems.append(f"module_count {len(modules)} != expected {len(plan.modules)}")
    for i, expected in enumerate(plan.modules[: len(modules)]):
        got = modules[i]
        if got["module_kind"] != expected.module_kind:
            problems.append(f"module[{i}]: kind {got['module_kind']} != {expected.module_kind}")
        if got["module_version"] != fmt.VERSION:
            problems.append(f"module[{i}]: version {got['module_version']} != {fmt.VERSION}")
        if got["tensor_index_begin"] != expected.tensor_index_begin:
            problems.append(
                f"module[{i}]: tensor_index_begin {got['tensor_index_begin']} != {expected.tensor_index_begin}"
            )
        if got["tensor_index_count"] != expected.tensor_index_count:
            problems.append(
                f"module[{i}]: tensor_index_count {got['tensor_index_count']} != {expected.tensor_index_count}"
            )
        if got["load_policy"] != expected.load_policy:
            problems.append(f"module[{i}]: load_policy {got['load_policy']} != {expected.load_policy}")
        begin = got["tensor_index_begin"]
        end = begin + got["tensor_index_count"]
        if end <= len(entries) and begin < end:
            first = entries[begin]
            last = entries[end - 1]
            if got["payload_offset"] != first["payload_offset"]:
                problems.append(f"module[{i}]: payload_offset does not match first block")
            if got["payload_bytes"] != last["payload_offset"] + last["payload_bytes"] - first["payload_offset"]:
                problems.append(f"module[{i}]: payload_bytes does not span module blocks")

    if len(entries) != len(plan.blocks):
        problems.append(f"block_count {len(entries)} != expected plan {len(plan.blocks)}")
    for i, expected in enumerate(plan.blocks[: len(entries)]):
        got = entries[i]
        for key, want in (
            ("name", expected.name),
            ("qtype", expected.qtype),
            ("layout", expected.layout),
            ("module_kind", expected.module_kind),
            ("source_layer", expected.source_layer),
            ("source_kind", expected.source_kind),
            ("segment_begin", expected.segment_begin),
            ("segment_count", expected.segment_count),
            ("fusion_group_id", expected.fusion_group_id),
            ("fusion_index", expected.fusion_index),
        ):
            if got[key] != want:
                problems.append(f"block[{i}] {got['name']}: {key} {got[key]!r} != expected {want!r}")
        if not _source_kind_defined(got["module_kind"], got["source_kind"]):
            problems.append(
                f"block[{i}] {got['name']}: source_kind {got['source_kind']} not defined for module {got['module_kind']}"
            )
        if expected.shape is not None and got["shape"] != list(expected.shape):
            problems.append(f"block[{i}] {got['name']}: shape {got['shape']} != expected {list(expected.shape)}")

    if len(segments) != len(plan.segments):
        problems.append(f"segment_count {len(segments)} != expected plan {len(plan.segments)}")
    for i, expected in enumerate(plan.segments[: len(segments)]):
        got = segments[i]
        for key, want in (
            ("name", expected.name),
            ("source_kind", expected.source_kind),
            ("source_layer", expected.source_layer),
            ("row_begin", expected.row_begin),
        ):
            if got[key] != want:
                problems.append(f"segment[{i}] {got['name']}: {key} {got[key]!r} != expected {want!r}")
        if expected.row_count is not None and got["row_count"] != expected.row_count:
            problems.append(
                f"segment[{i}] {got['name']}: row_count {got['row_count']} != expected {expected.row_count}"
            )

    if len(fusions) != len(plan.fusion_groups):
        problems.append(f"fusion_group_count {len(fusions)} != expected plan {len(plan.fusion_groups)}")
    for i, expected in enumerate(plan.fusion_groups[: len(fusions)]):
        got = fusions[i]
        for key, want in (
            ("group_id", expected.group_id),
            ("source_layer", expected.source_layer),
            ("block_count", expected.block_count),
            ("shared_input_kind", expected.shared_input_kind),
            ("first_block_tensor_index", expected.first_block_index),
            ("total_n", expected.total_n),
            ("shared_k", expected.shared_k),
        ):
            if got[key] != want:
                problems.append(f"fusion[{i}]: {key} {got[key]!r} != expected {want!r}")

    prev_end = None
    with open(path, "rb") as f:
        _check_zero_range(f, fmt._HEADER_STRUCT.size, fmt.HEADER_SIZE, "FileHeader reserved", problems)
        for i in range(len(modules)):
            begin = hdr["module_index_offset"] + i * fmt.MODULE_RECORD_SIZE
            _check_zero_range(
                f,
                begin + fmt._MODULE_STRUCT.size,
                begin + fmt.MODULE_RECORD_SIZE,
                f"ModuleRecord[{i}] reserved",
                problems,
            )
        for i in range(len(entries)):
            begin = hdr["tensor_index_offset"] + i * fmt.TENSOR_ENTRY_SIZE
            _check_zero_range(
                f,
                begin + fmt._ENTRY_STRUCT.size,
                begin + fmt.TENSOR_ENTRY_SIZE,
                f"TensorEntry[{i}] reserved",
                problems,
            )
        for i in range(len(fusions)):
            begin = hdr["fusion_group_index_offset"] + i * fmt.FUSION_GROUP_RECORD_SIZE
            _check_zero_range(
                f,
                begin + fmt._FUSION_GROUP_STRUCT.size,
                begin + fmt.FUSION_GROUP_RECORD_SIZE,
                f"FusionGroupRecord[{i}] reserved",
                problems,
            )
        _check_zero_range(f, string_end, hdr["payload_offset"], "string-to-payload padding", problems)
        for i, e in enumerate(entries):
            problems += _check_entry_planes(e)
            off, nb = e["payload_offset"], e["payload_bytes"]
            if off % fmt.PAYLOAD_ALIGN != 0:
                problems.append(f"{e['name']}: payload not {fmt.PAYLOAD_ALIGN}-aligned ({off})")
            if off < hdr["payload_offset"] or off + nb > file_size:
                problems.append(f"{e['name']}: payload out of range")
                continue
            if prev_end is not None:
                expected_off = fmt.align_up(prev_end, fmt.PAYLOAD_ALIGN)
                if off != expected_off:
                    problems.append(f"{e['name']}: payload_offset {off} != next aligned offset {expected_off}")
                _check_zero_range(f, prev_end, off, f"{e['name']}: inter-block padding", problems)
            f.seek(off)
            payload = f.read(nb)
            if fmt.crc32(payload) != e["crc32"]:
                problems.append(f"{e['name']}: crc32 mismatch")
            if (
                e["layout"] == qt.LAYOUT_ROW_SPLIT
                and qt.is_quant(e["qtype"])
                and len(e["shape"]) == 2
                and len(e["padded_shape"]) == 2
            ):
                spec = qt.QUANT_SPECS[e["qtype"]]
                n, _ = e["shape"]
                _, pk = e["padded_shape"]
                groups = pk // spec.group_size
                nibble_bytes, high_off, high_bytes, scale_off, _, _ = row_split_plane_sizes(
                    n,
                    groups,
                    spec.nibble_bytes_per_group,
                    spec.high_bytes_per_group,
                )
                _check_zero_range(
                    f,
                    off + nibble_bytes,
                    off + high_off,
                    f"{e['name']}: nibble-to-high padding",
                    problems,
                )
                _check_zero_range(
                    f,
                    off + high_off + high_bytes,
                    off + scale_off,
                    f"{e['name']}: high-to-scale padding",
                    problems,
                )
            prev_end = off + nb
        if prev_end is not None and prev_end != file_size:
            problems.append(f"last block ends at {prev_end}, file_size is {file_size}")

    for i, e in enumerate(entries):
        begin, count = e["segment_begin"], e["segment_count"]
        if count < 1:
            problems.append(f"{e['name']}: segment_count {count} < 1")
            continue
        if begin + count > len(segments):
            problems.append(f"{e['name']}: segment range out of table")
            continue
        row = 0
        block_segments = segments[begin: begin + count]
        for s in block_segments:
            if s["row_begin"] != row:
                problems.append(f"{e['name']}: segment {s['name']} row_begin {s['row_begin']} != {row}")
            if s["row_count"] <= 0:
                problems.append(f"{e['name']}: segment {s['name']} row_count {s['row_count']} <= 0")
            if not _source_kind_defined(e["module_kind"], s["source_kind"]):
                problems.append(
                    f"{e['name']}: segment {s['name']} source_kind {s['source_kind']} "
                    f"not defined for module {e['module_kind']}"
                )
            row = s["row_begin"] + s["row_count"]
        if e["shape"] and row != e["shape"][0]:
            problems.append(f"{e['name']}: segments cover {row} rows, expected {e['shape'][0]}")
        if e["layout"] == qt.LAYOUT_CONTIGUOUS and count != 1:
            problems.append(f"{e['name']}: CONTIGUOUS block has {count} segments")
        if count == 1:
            s = block_segments[0]
            if e["source_kind"] != s["source_kind"] or e["source_layer"] != s["source_layer"]:
                problems.append(f"{e['name']}: single-segment source identity does not mirror segment")
            if e["fusion_group_id"] == qt.FUSION_NONE and s["name"] != e["name"]:
                problems.append(f"{e['name']}: standalone segment name {s['name']} != block name")
        elif e["source_kind"] != qt.SK_OTHER:
            problems.append(f"{e['name']}: multi-segment block source_kind {e['source_kind']} != OTHER")

    for i, g in enumerate(fusions):
        first, count = g["first_block_tensor_index"], g["block_count"]
        if count < 1 or first + count > len(entries):
            problems.append(f"fusion[{i}]: member range out of tensor table")
            continue
        members = entries[first: first + count]
        start = members[0]["payload_offset"]
        end = members[-1]["payload_offset"] + members[-1]["payload_bytes"]
        if g["payload_offset"] != start:
            problems.append(f"fusion[{i}]: payload_offset {g['payload_offset']} != first member {start}")
        if g["payload_bytes"] != end - start:
            problems.append(f"fusion[{i}]: payload_bytes {g['payload_bytes']} != member span {end - start}")
        total_n = 0
        shared_k = members[0]["shape"][1] if len(members[0]["shape"]) > 1 else 0
        prev = None
        for fusion_index, e in enumerate(members):
            if e["fusion_group_id"] != g["group_id"]:
                problems.append(f"fusion[{i}] {e['name']}: fusion_group_id mismatch")
            if e["fusion_index"] != fusion_index:
                problems.append(f"fusion[{i}] {e['name']}: fusion_index {e['fusion_index']} != {fusion_index}")
            if e["source_layer"] != g["source_layer"]:
                problems.append(f"fusion[{i}] {e['name']}: source_layer mismatch")
            if len(e["shape"]) < 2 or e["shape"][1] != shared_k:
                problems.append(f"fusion[{i}] {e['name']}: shared K mismatch")
            if prev is not None and e["payload_offset"] != fmt.align_up(prev, fmt.PAYLOAD_ALIGN):
                problems.append(f"fusion[{i}] {e['name']}: payload not consecutive within group")
            total_n += e["shape"][0]
            prev = e["payload_offset"] + e["payload_bytes"]
        if g["total_n"] != total_n:
            problems.append(f"fusion[{i}]: total_n {g['total_n']} != {total_n}")
        if g["shared_k"] != shared_k:
            problems.append(f"fusion[{i}]: shared_k {g['shared_k']} != {shared_k}")
    return problems


def _manifest_checks(path: str, hdr, modules, entries, segments, fusions) -> List[str]:
    problems: List[str] = []
    manifest_path = path + fmt.MANIFEST_SUFFIX
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return [f"missing manifest sidecar {manifest_path}"]
    except json.JSONDecodeError as exc:
        return [f"manifest {manifest_path}: invalid JSON: {exc}"]

    def expect(key: str, want) -> None:
        got = manifest.get(key)
        if got != want:
            problems.append(f"manifest {key} {got!r} != {want!r}")

    module_names = [qt.MODULE_NAME.get(m["module_kind"], str(m["module_kind"])) for m in modules]
    present_qtypes = [
        name
        for qtype, name in qt.QTYPE_NAME.items()
        if any(e["qtype"] == qtype for e in entries)
    ]
    present_layouts = [
        name
        for layout, name in qt.LAYOUT_NAME.items()
        if any(e["layout"] == layout for e in entries)
    ]

    expect("format", "q5090_w4g64_mixed_v3")
    expect("format_version", fmt.VERSION)
    expect("format_minor", fmt.FORMAT_MINOR)
    expect("binary_spec", "docs/q5090_packed_file_format_v3.md")
    expect("tensor_plan", "docs/qwen3_6_27b_q5090_v2_tensor_plan.md")
    expect("weights_file", os.path.basename(path))
    expect("file_bytes", os.path.getsize(path))
    expect("sha256_safetensors_index", hdr["sha256_safetensors_index"].hex())
    expect("calibrated", bool(hdr["flags"] & fmt.FLAG_CALIBRATED))
    expect("layouts", present_layouts)
    expect("code_planes", ["nibble", "high", "scale"])
    expect("qtypes", present_qtypes)
    expect("modules", module_names)
    expect("absent_modules", [])
    expect("module_count", hdr["module_count"])
    expect("tensor_count", hdr["tensor_count"])
    expect("segment_count", hdr["segment_count"])
    expect("fusion_group_count", hdr["fusion_group_count"])

    alignment = manifest.get("alignment")
    want_alignment = {
        "header": fmt.HEADER_SIZE,
        "payload": fmt.REGION_ALIGN,
        "block": fmt.PAYLOAD_ALIGN,
        "k_pad": 128,
        "group_size": 64,
    }
    if alignment != want_alignment:
        problems.append(f"manifest alignment {alignment!r} != {want_alignment!r}")

    return problems


def _expected_quantized(reader, block, entry, device):
    w = materialize_block(reader, block).to(device=device, dtype=torch.float32)
    n, k = w.shape
    pn, pk = entry["padded_shape"]
    if int(n) != entry["shape"][0] or int(k) != entry["shape"][1]:
        raise ValueError(f"{block.name}: source shape {tuple(w.shape)} != file shape {entry['shape']}")
    if pk != k:
        w = F.pad(w, (0, pk - k, 0, 0), value=0.0)
    spec = qt.QUANT_SPECS[entry["qtype"]]
    return quantize_core(w, spec.group_size, spec.qmax, spec.qmin)


def _row_split_probes(scale16: torch.Tensor, codes: torch.Tensor, entry: dict) -> List[dict]:
    n, k = entry["shape"]
    spec = qt.QUANT_SPECS[entry["qtype"]]
    coords = [(0, 0), (n // 2, k // 2), (n - 1, k - 1)]
    out = []
    for row, col in coords:
        group = col // spec.group_size
        lane = col % spec.group_size
        q = int(codes[row, group, lane].item())
        scale = float(scale16[row, group].float().item())
        out.append({"row": int(row), "col": int(col), "scale": scale, "q": q, "value": scale * q})
    return out


def _contiguous_probes(payload: bytes, entry: dict, device) -> List[dict]:
    t = decode_tensor(payload, entry["qtype"], entry["layout"], entry["shape"], entry["padded_shape"], device)
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return []
    idxs = [0, int(flat.numel() // 2), int(flat.numel() - 1)]
    out = []
    shape = entry["shape"]
    for idx in idxs:
        if len(shape) == 1:
            row, col = idx, 0
        else:
            cols = _prod(shape[1:])
            row, col = idx // cols, idx % cols
        out.append({"row": int(row), "col": int(col), "value": float(flat[idx].item())})
    return out


def _l1_checks(path: str, entries, plan, reader, device, dump: dict) -> Tuple[List[str], Dict[int, dict]]:
    fails: List[str] = []
    agg = defaultdict(lambda: {"n": 0, "scale_bad": 0, "code_bad": 0, "control_bad": 0})
    block_dumps = dump["blocks"]
    t0 = time.time()
    with open(path, "rb") as f:
        for i, (entry, block) in enumerate(zip(entries, plan.blocks)):
            f.seek(entry["payload_offset"])
            payload = f.read(entry["payload_bytes"])
            qagg = agg[entry["qtype"]]
            qagg["n"] += 1
            try:
                if qt.is_quant(entry["qtype"]):
                    got_scale, got_codes = decode_row_split_quantized(
                        payload, entry["padded_shape"], entry["qtype"], device
                    )
                    exp_scale, exp_codes = _expected_quantized(reader, block, entry, device)
                    if not torch.equal(got_scale.contiguous().view(torch.int16), exp_scale.contiguous().view(torch.int16)):
                        bad = int(
                            (
                                got_scale.contiguous().view(torch.int16)
                                != exp_scale.contiguous().view(torch.int16)
                            )
                            .sum()
                            .item()
                        )
                        qagg["scale_bad"] += bad
                        fails.append(f"{entry['name']}: scale16 mismatch ({bad} elements)")
                    if not torch.equal(got_codes, exp_codes):
                        bad = int((got_codes != exp_codes).sum().item())
                        qagg["code_bad"] += bad
                        fails.append(f"{entry['name']}: quant code mismatch ({bad} elements)")
                    block_dumps[i]["dequant_probes"] = _row_split_probes(got_scale, got_codes, entry)
                    del got_scale, got_codes, exp_scale, exp_codes
                else:
                    src = materialize_block(reader, block)
                    exp_payload, logical, padded, _, _, _, _, _ = encode_tensor(
                        src, entry["qtype"], entry["layout"], device
                    )
                    if logical != entry["shape"] or padded != entry["padded_shape"]:
                        fails.append(
                            f"{entry['name']}: control source shape {logical}/{padded} != file "
                            f"{entry['shape']}/{entry['padded_shape']}"
                        )
                    if exp_payload != payload:
                        qagg["control_bad"] += 1
                        fails.append(f"{entry['name']}: control payload mismatch")
                    block_dumps[i]["dequant_probes"] = _contiguous_probes(payload, entry, device)
                    del src
            except Exception as exc:
                fails.append(f"{entry['name']}: L1 exception: {exc}")
            if i % 50 == 0 or i == len(entries) - 1:
                print(f"  L1 [{i + 1}/{len(entries)}] {entry['name']} ({time.time() - t0:.0f}s)", flush=True)
            if device.type == "cuda" and (i % 25 == 0):
                torch.cuda.empty_cache()
    return fails, dict(agg)


def _write_dump(path: str, dump: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dump, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="verify q5090_w4g64_mixed_v3 packed file")
    ap.add_argument("file", help="v3 .qus file")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF bf16 source model for L1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="L0 + CRC only; skip L1")
    ap.add_argument("--dump", default=os.path.join("out", "conv_dump.v3.json"))
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.model)
    assert_config(cfg, force=False)

    hdr, modules, entries, segments, fusions, read_problems = _read_file(args.file)
    plan = build_conversion_plan(cfg)
    dump = _make_dump(args.file, hdr, modules, entries, segments, fusions)

    print(f"file: {args.file}")
    print(
        f"blocks={hdr['tensor_count']} segments={hdr['segment_count']} "
        f"fusion_groups={hdr['fusion_group_count']} modules={hdr['module_count']} "
        f"file_bytes={os.path.getsize(args.file)}",
        flush=True,
    )
    for m in modules:
        print(
            f"  module {qt.MODULE_NAME.get(m['module_kind'], m['module_kind'])}: "
            f"blocks={m['tensor_index_count']} payload={m['payload_bytes']}",
            flush=True,
        )

    l0 = (
        read_problems
        + _l0_checks(args.file, hdr, modules, entries, segments, fusions, plan)
        + _manifest_checks(args.file, hdr, modules, entries, segments, fusions)
    )
    print(f"L0 structural checks done; problems={len(l0)}", flush=True)

    l1: List[str] = []
    agg: Dict[int, dict] = {}
    if not args.quick and not l0:
        device = pick_device(args.device)
        print(f"device: {device}", flush=True)
        with open(os.path.join(args.model, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        reader = ShardReader(args.model, weight_map)
        missing = sorted({s.source.src_name for s in plan.segments if not reader.has(s.source.src_name)})
        if missing:
            l1.append("missing source tensors:\n  " + "\n  ".join(missing))
        else:
            l1, agg = _l1_checks(args.file, entries, plan, reader, device, dump)
            print("L1 value checks done; problems=" + str(len(l1)), flush=True)
    elif args.quick:
        print("L1 skipped (--quick)", flush=True)
    else:
        print("L1 skipped because L0 failed", flush=True)

    _write_dump(args.dump, dump)
    print(f"wrote dump: {args.dump}", flush=True)

    if agg:
        print("\nL1 summary:")
        for qtype, a in sorted(agg.items()):
            print(
                f"  {qt.QTYPE_NAME[qtype]:12s} blocks={a['n']:4d} "
                f"scale_bad={a['scale_bad']} code_bad={a['code_bad']} control_bad={a['control_bad']}"
            )

    fails = l0 + l1
    print()
    if fails:
        print(f"FAILED ({len(fails)} problems):")
        for p in fails[:80]:
            print("  -", p)
        raise SystemExit(1)
    print(f"OK: L0{' + L1' if not args.quick else ''} passed in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
