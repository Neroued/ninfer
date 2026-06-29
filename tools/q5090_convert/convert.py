"""CLI: Qwen3.6-27B bf16 safetensors -> q5090_w4g64_mixed_v2 packed file.

Usage:
  python -m tools.q5090_convert.convert --model /path/to/Qwen3.6-27B --out out/file.qus

Default output includes all modules present in the HF source: TEXT_CORE plus MTP_DRAFT and
VISION_ENCODER when their weights exist. Binary format:
../../docs/q5090_packed_file_format_v2.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open

from . import format as fmt
from . import qtypes as qt
from . import tensor_plan as tp
from .layouts import encode_tensor
from .quantize import pick_device

# Expected config (policy lock); see policy doc section 0.
EXPECTED = dict(
    num_hidden_layers=64,
    full_attention_interval=4,
    hidden_size=5120,
    intermediate_size=17408,
    vocab_size=248320,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    max_position_embeddings=262144,
    vision_depth=27,
    vision_hidden=1152,
    vision_intermediate=4304,
    vision_out_hidden=5120,
)
OUTPUT_FORMAT_MARKER = "mixed_v2"

@dataclass(frozen=True)
class SegmentPlan:
    block_index: int
    segment_index: int
    name: str
    source_kind: int
    source_layer: int
    row_begin: int
    row_count: Optional[int]
    source: tp.TensorSpec


@dataclass(frozen=True)
class BlockPlan:
    block_index: int
    name: str
    qtype: int
    layout: int
    module_kind: int
    shape: Optional[Tuple[int, ...]]
    source_layer: int
    source_kind: int
    segment_begin: int
    segments: Tuple[SegmentPlan, ...]
    fusion_group_id: int = qt.FUSION_NONE
    fusion_index: int = 0

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class FusionPlan:
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
class ModulePlan:
    module_kind: int
    tensor_index_begin: int
    tensor_index_count: int
    load_policy: int


@dataclass(frozen=True)
class ConversionPlan:
    modules: Tuple[ModulePlan, ...]
    blocks: Tuple[BlockPlan, ...]
    segments: Tuple[SegmentPlan, ...]
    fusion_groups: Tuple[FusionPlan, ...]


class ShardReader:
    """Lazy per-shard safetensors reader with a 1-entry source cache."""

    def __init__(self, model_dir: str, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._handles: Dict[str, object] = {}
        self._cache_name: Optional[str] = None
        self._cache_tensor: Optional[torch.Tensor] = None

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.model_dir, shard), framework="pt", device="cpu")
            self._handles[shard] = h
        return h

    def get(self, name: str) -> torch.Tensor:
        if name == self._cache_name:
            assert self._cache_tensor is not None
            return self._cache_tensor
        shard = self.weight_map[name]
        t = self._handle(shard).get_tensor(name)
        self._cache_name = name
        self._cache_tensor = t
        return t

    def has(self, name: str) -> bool:
        return name in self.weight_map


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        return json.load(f)


def assert_config(cfg: dict, force: bool) -> None:
    tc = cfg.get("text_config", cfg)
    vc = cfg.get("vision_config", {})
    got = dict(
        num_hidden_layers=tc.get("num_hidden_layers"),
        full_attention_interval=tc.get("full_attention_interval"),
        hidden_size=tc.get("hidden_size"),
        intermediate_size=tc.get("intermediate_size"),
        vocab_size=tc.get("vocab_size"),
        num_attention_heads=tc.get("num_attention_heads"),
        num_key_value_heads=tc.get("num_key_value_heads"),
        head_dim=tc.get("head_dim"),
        linear_num_key_heads=tc.get("linear_num_key_heads"),
        linear_num_value_heads=tc.get("linear_num_value_heads"),
        linear_key_head_dim=tc.get("linear_key_head_dim"),
        linear_value_head_dim=tc.get("linear_value_head_dim"),
        linear_conv_kernel_dim=tc.get("linear_conv_kernel_dim"),
        max_position_embeddings=tc.get("max_position_embeddings"),
        vision_depth=vc.get("depth"),
        vision_hidden=vc.get("hidden_size"),
        vision_intermediate=vc.get("intermediate_size"),
        vision_out_hidden=vc.get("out_hidden_size"),
    )
    mism = [(k, EXPECTED[k], got[k]) for k in EXPECTED if got[k] != EXPECTED[k]]
    if mism:
        lines = "\n".join(f"  {k}: expected {e}, got {g}" for k, e, g in mism)
        msg = f"config.json does not match policy lock:\n{lines}"
        if force:
            print("WARNING:", msg)
        else:
            raise SystemExit(msg + "\n(use --force to override)")


def _layer_types(cfg: dict) -> List[str]:
    tc = cfg.get("text_config", cfg)
    lts = tc.get("layer_types")
    if lts is None:
        n = tc["num_hidden_layers"]
        interval = tc["full_attention_interval"]
        lts = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]
    return list(lts)


def _prepare_source(reader: ShardReader, spec: tp.TensorSpec) -> torch.Tensor:
    t = reader.get(spec.src_name)
    if spec.row_slice is not None:
        a, b = spec.row_slice
        if t.shape[0] < b:
            raise ValueError(f"{spec.src_name}: rows {b} > tensor rows {t.shape[0]}")
        t = t[a:b]
    if spec.reshape is not None:
        numel = 1
        for x in spec.reshape:
            numel *= int(x)
        if t.numel() != numel:
            raise ValueError(f"{spec.src_name}: cannot reshape {tuple(t.shape)} -> {spec.reshape}")
        t = t.reshape(spec.reshape)
    if spec.transform is not None:
        if spec.transform == tp.TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE:
            t = tp.runtime_native_gdn_conv1d(t)
        elif spec.transform == tp.TRANSFORM_ATTN_QPROJ_QUERY:
            t = tp.attn_qproj_split(t, take_gate=False)
        elif spec.transform == tp.TRANSFORM_ATTN_QPROJ_GATE:
            t = tp.attn_qproj_split(t, take_gate=True)
        else:
            raise ValueError(f"{spec.name}: unknown tensor transform {spec.transform!r}")
    if spec.layout != qt.LAYOUT_CONTIGUOUS and t.dim() != 2:
        raise ValueError(f"{spec.name}: quant layout needs 2D, got {tuple(t.shape)}")
    return t.contiguous()


def _text_plan(layer_types: Sequence[str]) -> Tuple[List[BlockPlan], List[SegmentPlan], List[FusionPlan]]:
    manifest = tp.build_text_manifest(list(layer_types))
    segments: List[SegmentPlan] = []
    blocks: List[BlockPlan] = []
    for b in manifest.blocks:
        bsegs = []
        for s in b.segments:
            seg = SegmentPlan(
                block_index=s.block_index,
                segment_index=s.segment_index,
                name=s.name,
                source_kind=s.source_kind,
                source_layer=s.source_layer,
                row_begin=s.row_begin,
                row_count=s.row_count,
                source=s.source,
            )
            bsegs.append(seg)
            segments.append(seg)
        blocks.append(
            BlockPlan(
                block_index=b.block_index,
                name=b.name,
                qtype=b.qtype,
                layout=b.layout,
                module_kind=b.module_kind,
                shape=b.shape,
                source_layer=b.source_layer,
                source_kind=b.source_kind,
                segment_begin=b.segment_begin,
                segments=tuple(bsegs),
                fusion_group_id=b.fusion_group_id,
                fusion_index=b.fusion_index,
            )
        )
    groups = [
        FusionPlan(
            group_id=g.group_id,
            source_layer=g.source_layer,
            block_indices=g.block_indices,
            shared_input_kind=g.shared_input_kind,
            total_n=g.total_n,
            shared_k=g.shared_k,
        )
        for g in manifest.fusion_groups
    ]
    return blocks, segments, groups


def _append_standalone_blocks(
    blocks: List[BlockPlan],
    segments: List[SegmentPlan],
    specs: Iterable[tp.TensorSpec],
) -> None:
    for spec in specs:
        block_index = len(blocks)
        segment_index = len(segments)
        seg = SegmentPlan(
            block_index=block_index,
            segment_index=segment_index,
            name=spec.name,
            source_kind=spec.source_kind,
            source_layer=spec.source_layer,
            row_begin=0,
            row_count=None,
            source=spec,
        )
        blocks.append(
            BlockPlan(
                block_index=block_index,
                name=spec.name,
                qtype=spec.qtype,
                layout=spec.layout,
                module_kind=spec.module_kind,
                shape=None,
                source_layer=spec.source_layer,
                source_kind=spec.source_kind,
                segment_begin=segment_index,
                segments=(seg,),
            )
        )
        segments.append(seg)


def build_conversion_plan(
    cfg: dict,
    include_mtp: bool,
    include_vision: bool,
    limit_text_layers: int = -1,
    vision_blocks: int = -1,
) -> ConversionPlan:
    layer_types = _layer_types(cfg)
    if limit_text_layers >= 0:
        layer_types = layer_types[:limit_text_layers]

    blocks, segments, fusion_groups = _text_plan(layer_types)
    modules: List[ModulePlan] = [
        ModulePlan(qt.MODULE_TEXT, 0, len(blocks), qt.LOAD_RESIDENT),
    ]

    if include_mtp:
        begin = len(blocks)
        _append_standalone_blocks(blocks, segments, tp.build_mtp_specs())
        modules.append(ModulePlan(qt.MODULE_MTP, begin, len(blocks) - begin, qt.LOAD_RESIDENT))

    if include_vision:
        begin = len(blocks)
        depth = EXPECTED["vision_depth"] if vision_blocks < 0 else vision_blocks
        _append_standalone_blocks(blocks, segments, tp.build_vision_specs(depth))
        modules.append(ModulePlan(qt.MODULE_VISION, begin, len(blocks) - begin, qt.LOAD_LAZY_GPU))

    return ConversionPlan(tuple(modules), tuple(blocks), tuple(segments), tuple(fusion_groups))


def plan_source_names(plan: ConversionPlan) -> List[str]:
    return sorted({s.source.src_name for s in plan.segments})


def materialize_block(reader: ShardReader, block: BlockPlan) -> torch.Tensor:
    parts = []
    for seg in block.segments:
        part = _prepare_source(reader, seg.source)
        if seg.row_count is not None and int(part.shape[0]) != seg.row_count:
            raise ValueError(
                f"{seg.name}: source rows {int(part.shape[0])} != planned rows {seg.row_count}"
            )
        parts.append(part)

    if block.layout == qt.LAYOUT_CONTIGUOUS:
        if len(parts) != 1:
            raise ValueError(f"{block.name}: CONTIGUOUS block must have one segment")
        w = parts[0]
    else:
        for part in parts:
            if part.dim() != 2:
                raise ValueError(f"{block.name}: ROW_SPLIT segment needs 2D, got {tuple(part.shape)}")
        k0 = int(parts[0].shape[1])
        for part in parts[1:]:
            if int(part.shape[1]) != k0:
                raise ValueError(f"{block.name}: segment K mismatch {int(part.shape[1])} != {k0}")
        w = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    if block.shape is not None and tuple(int(x) for x in w.shape) != tuple(block.shape):
        raise ValueError(f"{block.name}: materialized shape {tuple(w.shape)} != plan {block.shape}")
    return w.contiguous()


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _require_v2_output_path(out_path: str) -> None:
    basename = os.path.basename(os.path.normpath(out_path))
    if OUTPUT_FORMAT_MARKER not in basename or not basename.endswith(".qus"):
        raise SystemExit(
            f"q5090 v2 converter output path must be a {OUTPUT_FORMAT_MARKER} .qus file: "
            f"{out_path}"
        )


def _fill_segment_row_counts(
    block: BlockPlan,
    logical_shape: Sequence[int],
    segment_records: List[fmt.SegmentRecord],
) -> None:
    leading = int(logical_shape[0])
    row = 0
    for seg in block.segments:
        rec = segment_records[seg.segment_index]
        row_count = seg.row_count if seg.row_count is not None else leading
        if rec.row_begin != row:
            raise ValueError(f"{block.name}: segment {seg.name} row_begin {rec.row_begin} != {row}")
        rec.row_count = int(row_count)
        row += int(row_count)
    if row != leading:
        raise ValueError(f"{block.name}: segments cover {row} rows, expected {leading}")


def _write_manifest(
    out_path: str,
    cfg: dict,
    plan: ConversionPlan,
    module_records: Sequence[fmt.ModuleRecord],
    entries: Sequence[fmt.TensorEntry],
    segment_records: Sequence[fmt.SegmentRecord],
    fusion_records: Sequence[fmt.FusionGroupRecord],
    disabled: Sequence[str],
    file_size: int,
    source_index_sha256: bytes,
) -> None:
    tc = cfg.get("text_config", cfg)
    modules = []
    for m in module_records:
        module_entries = entries[m.tensor_index_begin: m.tensor_index_begin + m.tensor_index_count]
        modules.append(
            {
                "kind": qt.MODULE_NAME[m.module_kind],
                "policy": qt.MODULE_POLICY[m.module_kind],
                "block_count": int(m.tensor_index_count),
                "segment_count": int(sum(e.segment_count for e in module_entries)),
                "fusion_group_count": int(
                    sum(
                        1
                        for g in fusion_records
                        if m.tensor_index_begin
                        <= g.first_block_tensor_index
                        < m.tensor_index_begin + m.tensor_index_count
                    )
                ),
                "payload_offset": int(m.payload_offset),
                "payload_bytes": int(m.payload_bytes),
                "payload_gib": round(m.payload_bytes / (1024**3), 4),
                "load_policy": int(m.load_policy),
            }
        )

    text_bytes = sum(e.payload_bytes for e in entries if e.module_kind == qt.MODULE_TEXT)
    text_elems = sum(_prod(e.shape) for e in entries if e.module_kind == qt.MODULE_TEXT)
    group_summary = []
    for gid in (qt.FUSION_ATTN_IN, qt.FUSION_GDN_IN, qt.FUSION_MLP_GATEUP):
        records = [g for g in fusion_records if g.group_id == gid]
        if records:
            group_summary.append(
                {
                    "group_id": qt.FUSION_GROUP_NAME[gid],
                    "groups": len(records),
                    "blocks_per_group": records[0].block_count,
                    "total_n": records[0].total_n,
                    "shared_k": records[0].shared_k,
                }
            )

    manifest = {
        "format": "q5090_w4g64_mixed_v2",
        "format_version": 2,
        "model": cfg.get("_name_or_path", "Qwen/Qwen3.6-27B"),
        "binary_spec": "docs/q5090_packed_file_format_v2.md",
        "tensor_plan": "docs/qwen3_6_27b_q5090_v2_tensor_plan.md",
        "output": os.path.basename(out_path),
        "file_bytes": int(file_size),
        "source_index_sha256": source_index_sha256.hex(),
        "block_count": len(entries),
        "segment_count": len(segment_records),
        "fusion_group_count": len(fusion_records),
        "modules": modules,
        "disabled_modules": list(disabled),
        "layouts": list(qt.LAYOUT_NAME.values()),
        "qtypes": list(qt.QTYPE_NAME.values()),
        "fusion_groups": group_summary,
        "value_source": "qwen3_6_27b_q5090_final_quant_format_v1 (policy)",
        "effective_text_bpw": round(text_bytes * 8 / max(1, text_elems), 4),
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "zero_point": False,
        "alignment": {
            "file_header": fmt.HEADER_SIZE,
            "tensor_payload": fmt.PAYLOAD_ALIGN,
            "payload_region": fmt.REGION_ALIGN,
        },
    }
    mpath = os.path.join(os.path.dirname(out_path), fmt.MANIFEST_V2_NAME)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {mpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description="q5090_w4g64_mixed_v2 quantizing converter")
    ap.add_argument("--model", required=True, help="path to original Qwen3.6-27B safetensors dir")
    ap.add_argument("--out", default=None, help="output .qus file path")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="warn instead of abort on config mismatch")
    ap.add_argument("--limit-text-layers", type=int, default=-1, help="debug: only first N text layers")
    ap.add_argument("--vision-blocks", type=int, default=-1, help="debug: only first M vision blocks")
    args = ap.parse_args()

    model_dir = args.model.rstrip("/")
    out_path = args.out or os.path.join(
        os.getcwd(), "out", "qwen3_6_27b.q5090_w4g64_mixed_v2.qus"
    )
    _require_v2_output_path(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(model_dir)
    assert_config(cfg, args.force)
    tc = cfg.get("text_config", cfg)
    device = pick_device(args.device)
    print(f"device: {device}")

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "rb") as f:
        index_raw = f.read()
    sha = hashlib.sha256(index_raw).digest()
    weight_map = json.loads(index_raw)["weight_map"]
    reader = ShardReader(model_dir, weight_map)

    want_mtp = (not args.no_mtp) and reader.has("mtp.fc.weight")
    want_vision = (not args.no_vision) and reader.has("model.visual.merger.linear_fc1.weight")
    disabled: List[str] = []
    if not args.no_mtp and not want_mtp:
        disabled.append("MTP_DRAFT")
    if not args.no_vision and not want_vision:
        disabled.append("VISION_ENCODER")

    plan = build_conversion_plan(
        cfg,
        include_mtp=want_mtp,
        include_vision=want_vision,
        limit_text_layers=args.limit_text_layers,
        vision_blocks=args.vision_blocks,
    )
    missing = [name for name in plan_source_names(plan) if not reader.has(name)]
    if missing:
        raise SystemExit("missing source tensors:\n  " + "\n  ".join(missing))

    entries: List[fmt.TensorEntry] = []
    segment_records: List[fmt.SegmentRecord] = []
    for block in plan.blocks:
        entries.append(
            fmt.TensorEntry(
                name=block.name,
                qtype=block.qtype,
                layout=block.layout,
                module_kind=block.module_kind,
                shape=list(block.shape or []),
                padded_shape=list(block.shape or []),
                group_size=0,
                scale_dtype=qt.SCALE_NONE,
                segment_count=block.segment_count,
                source_layer=block.source_layer,
                source_kind=block.source_kind,
                segment_begin=block.segment_begin,
                fusion_group_id=block.fusion_group_id,
                fusion_index=block.fusion_index,
            )
        )
        for seg in block.segments:
            if seg.segment_index != len(segment_records):
                raise ValueError(f"{seg.name}: non-canonical segment index {seg.segment_index}")
            segment_records.append(
                fmt.SegmentRecord(
                    name=seg.name,
                    source_kind=seg.source_kind,
                    source_layer=seg.source_layer,
                    row_begin=seg.row_begin,
                    row_count=int(seg.row_count or 0),
                )
            )

    string_table = fmt.build_string_table(entries, segment_records)
    tensor_count = len(entries)
    module_count = len(plan.modules)
    segment_count = len(segment_records)
    fusion_group_count = len(plan.fusion_groups)
    module_index_offset = fmt.HEADER_SIZE
    module_index_bytes = module_count * fmt.MODULE_RECORD_SIZE
    tensor_index_offset = module_index_offset + module_index_bytes
    tensor_index_bytes = tensor_count * fmt.TENSOR_ENTRY_SIZE
    segment_index_offset = tensor_index_offset + tensor_index_bytes
    segment_index_bytes = segment_count * fmt.SEGMENT_RECORD_SIZE
    fusion_group_index_offset = segment_index_offset + segment_index_bytes
    fusion_group_index_bytes = fusion_group_count * fmt.FUSION_GROUP_RECORD_SIZE
    string_table_offset = fusion_group_index_offset + fusion_group_index_bytes
    string_table_bytes = len(string_table)
    payload_region_start = fmt.align_up(string_table_offset + string_table_bytes, fmt.REGION_ALIGN)

    print(
        f"blocks: {tensor_count}  segments: {segment_count}  fusion_groups: {fusion_group_count}  "
        f"modules: {module_count}  payload starts @ {payload_region_start}"
    )

    t0 = time.time()
    with open(out_path, "wb") as f:
        f.seek(payload_region_start)
        for i, (block, entry) in enumerate(zip(plan.blocks, entries)):
            pos = f.tell()
            pad = fmt.align_up(pos, fmt.PAYLOAD_ALIGN) - pos
            if pad:
                f.write(b"\x00" * pad)
                pos += pad

            w = materialize_block(reader, block)
            (
                payload,
                logical,
                padded,
                group,
                scale_dtype,
                code_plane_bytes,
                scale_plane_bytes,
            ) = encode_tensor(w, block.qtype, block.layout, device)
            if block.shape is not None and list(block.shape) != logical:
                raise ValueError(f"{block.name}: encoded logical shape {logical} != plan {block.shape}")
            _fill_segment_row_counts(block, logical, segment_records)

            f.write(payload)
            entry.shape = logical
            entry.padded_shape = padded
            entry.group_size = group
            entry.scale_dtype = scale_dtype
            entry.payload_offset = pos
            entry.payload_bytes = len(payload)
            entry.crc32 = fmt.crc32(payload)
            entry.code_plane_bytes = code_plane_bytes
            entry.scale_plane_bytes = scale_plane_bytes

            if len(payload) >= (1 << 20) or i % 50 == 0 or i == tensor_count - 1:
                elapsed = time.time() - t0
                print(
                    f"  [{i + 1}/{tensor_count}] {block.name}  {qt.QTYPE_NAME[block.qtype]}  "
                    f"{logical} -> {_fmt_bytes(len(payload))}  ({elapsed:.0f}s)",
                    flush=True,
                )
            del w
        file_size = f.tell()

        module_records: List[fmt.ModuleRecord] = []
        for m in plan.modules:
            if m.tensor_index_count <= 0:
                sp = [payload_region_start, payload_region_start]
            else:
                first = entries[m.tensor_index_begin]
                last = entries[m.tensor_index_begin + m.tensor_index_count - 1]
                sp = [first.payload_offset, last.payload_offset + last.payload_bytes]
            module_records.append(
                fmt.ModuleRecord(
                    module_kind=m.module_kind,
                    module_version=fmt.VERSION,
                    tensor_index_begin=m.tensor_index_begin,
                    tensor_index_count=m.tensor_index_count,
                    payload_offset=sp[0],
                    payload_bytes=sp[1] - sp[0],
                    load_policy=m.load_policy,
                )
            )

        fusion_records: List[fmt.FusionGroupRecord] = []
        for g in plan.fusion_groups:
            first = entries[g.first_block_index]
            last = entries[g.first_block_index + g.block_count - 1]
            fusion_records.append(
                fmt.FusionGroupRecord(
                    group_id=g.group_id,
                    source_layer=g.source_layer,
                    block_count=g.block_count,
                    shared_input_kind=g.shared_input_kind,
                    first_block_tensor_index=g.first_block_index,
                    payload_offset=first.payload_offset,
                    payload_bytes=last.payload_offset + last.payload_bytes - first.payload_offset,
                    total_n=g.total_n,
                    shared_k=g.shared_k,
                )
            )

        flags = 1
        if want_mtp:
            flags |= 2
        if want_vision:
            flags |= 4
        header = fmt.FileHeaderFields(
            tensor_count=tensor_count,
            module_count=module_count,
            layer_count=EXPECTED["num_hidden_layers"],
            flags=flags,
            segment_count=segment_count,
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
            payload_offset=payload_region_start,
            payload_bytes=file_size - payload_region_start,
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            gdn_key_heads=tc["linear_num_key_heads"],
            gdn_value_heads=tc["linear_num_value_heads"],
            gdn_key_head_dim=tc["linear_key_head_dim"],
            gdn_value_head_dim=tc["linear_value_head_dim"],
            gdn_conv_width=tc["linear_conv_kernel_dim"],
            full_attention_interval=tc["full_attention_interval"],
            max_position_embeddings=tc["max_position_embeddings"],
            fusion_group_count=fusion_group_count,
            sha256_safetensors_index=sha,
        )
        meta = bytearray()
        meta += fmt.pack_header(header)
        for m in module_records:
            meta += fmt.pack_module_record(m)
        for e in entries:
            meta += fmt.pack_tensor_entry(e)
        for s in segment_records:
            meta += fmt.pack_segment_record(s)
        for g in fusion_records:
            meta += fmt.pack_fusion_group_record(g)
        meta += string_table
        assert len(meta) <= payload_region_start, (len(meta), payload_region_start)
        meta = meta.ljust(payload_region_start, b"\x00")
        f.seek(0)
        f.write(meta)

    _write_manifest(
        out_path,
        cfg,
        plan,
        module_records,
        entries,
        segment_records,
        fusion_records,
        disabled,
        file_size,
        sha,
    )

    print(f"\nwrote {out_path}  ({_fmt_bytes(file_size)})  in {time.time() - t0:.0f}s")
    for m in module_records:
        print(
            f"  {qt.MODULE_NAME[m.module_kind]:14s} blocks={m.tensor_index_count:4d}  "
            f"payload={_fmt_bytes(m.payload_bytes)}  policy={qt.MODULE_POLICY[m.module_kind]}"
        )
    if disabled:
        print("  disabled_modules:", list(disabled))


if __name__ == "__main__":
    main()
