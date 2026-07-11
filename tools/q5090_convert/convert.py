"""CLI: Qwen3.6-27B bf16 safetensors -> q5090_w4g64_mixed_v4_2 artifact.

Usage:
  python -m tools.q5090_convert.convert --model /path/to/Qwen3.6-27B --out out/file.qus

Default output includes TEXT_CORE, the independent LM_HEAD_DRAFT module unless disabled, MTP_DRAFT,
VISION_ENCODER, and the three mandatory tokenizer assets. Binary format:
../../docs/q5090_packed_file_format_v4.md
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
from .tokenizer_contract import validate_tokenizer_assets
from . import draft_head as dh
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
    vision_num_heads=16,
    vision_in_channels=3,
    vision_patch_size=16,
    vision_temporal_patch_size=2,
    vision_spatial_merge_size=2,
    vision_num_position_embeddings=2304,
)
OUTPUT_FORMAT_MARKER = "mixed_v4_2"


@dataclass(frozen=True)
class TokenizerAsset:
    kind: int
    source_name: str
    data: bytes
    crc32: int
    sha256: bytes

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


def assert_config(cfg: dict) -> None:
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
        vision_num_heads=vc.get("num_heads"),
        vision_in_channels=vc.get("in_channels"),
        vision_patch_size=vc.get("patch_size"),
        vision_temporal_patch_size=vc.get("temporal_patch_size"),
        vision_spatial_merge_size=vc.get("spatial_merge_size"),
        vision_num_position_embeddings=vc.get("num_position_embeddings"),
    )
    mism = [(k, EXPECTED[k], got[k]) for k in EXPECTED if got[k] != EXPECTED[k]]
    if mism:
        lines = "\n".join(f"  {k}: expected {e}, got {g}" for k, e, g in mism)
        msg = f"config.json does not match policy lock:\n{lines}"
        raise SystemExit(msg)


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
    if spec.expected_shape is not None and tuple(t.shape) != spec.expected_shape:
        raise ValueError(
            f"{spec.name}: source shape {tuple(t.shape)} != canonical {spec.expected_shape}"
        )
    return t.contiguous()


def _text_plan(
    layer_types: Sequence[str],
) -> Tuple[List[BlockPlan], List[SegmentPlan], List[FusionPlan]]:
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
                shape=spec.expected_shape,
                source_layer=spec.source_layer,
                source_kind=spec.source_kind,
                segment_begin=segment_index,
                segments=(seg,),
            )
        )
        segments.append(seg)


def _append_manifest(
    blocks: List[BlockPlan],
    segments: List[SegmentPlan],
    fusion_groups: List[FusionPlan],
    manifest: tp.ExpectedManifest,
) -> None:
    block_offset = len(blocks)
    segment_offset = len(segments)
    for b in manifest.blocks:
        bsegs = []
        for s in b.segments:
            seg = SegmentPlan(
                block_index=block_offset + s.block_index,
                segment_index=segment_offset + s.segment_index,
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
                block_index=block_offset + b.block_index,
                name=b.name,
                qtype=b.qtype,
                layout=b.layout,
                module_kind=b.module_kind,
                shape=b.shape,
                source_layer=b.source_layer,
                source_kind=b.source_kind,
                segment_begin=segment_offset + b.segment_begin,
                segments=tuple(bsegs),
                fusion_group_id=b.fusion_group_id,
                fusion_index=b.fusion_index,
            )
        )
    for g in manifest.fusion_groups:
        fusion_groups.append(
            FusionPlan(
                group_id=g.group_id,
                source_layer=g.source_layer,
                block_indices=tuple(block_offset + i for i in g.block_indices),
                shared_input_kind=g.shared_input_kind,
                total_n=g.total_n,
                shared_k=g.shared_k,
            )
        )


def build_conversion_plan(
    cfg: dict,
    include_draft_head: bool = False,
    include_mtp: bool = True,
    include_vision: bool = True,
) -> ConversionPlan:
    layer_types = _layer_types(cfg)

    blocks, segments, fusion_groups = _text_plan(layer_types)
    modules: List[ModulePlan] = [
        ModulePlan(qt.MODULE_TEXT, 0, len(blocks)),
    ]

    if include_draft_head:
        begin = len(blocks)
        _append_manifest(
            blocks,
            segments,
            fusion_groups,
            tp.build_lm_head_draft_manifest(),
        )
        modules.append(ModulePlan(qt.MODULE_LM_HEAD_DRAFT, begin, len(blocks) - begin))

    if include_mtp:
        begin = len(blocks)
        _append_manifest(blocks, segments, fusion_groups, tp.build_mtp_manifest())
        modules.append(ModulePlan(qt.MODULE_MTP, begin, len(blocks) - begin))

    if include_vision:
        begin = len(blocks)
        _append_standalone_blocks(blocks, segments, tp.build_vision_specs(EXPECTED["vision_depth"]))
        modules.append(ModulePlan(qt.MODULE_VISION, begin, len(blocks) - begin))

    return ConversionPlan(tuple(modules), tuple(blocks), tuple(segments), tuple(fusion_groups))


def plan_source_names(plan: ConversionPlan) -> List[str]:
    return sorted({s.source.src_name for s in plan.segments})


def _materialize_draft_block(
    reader: ShardReader,
    block: BlockPlan,
    kind: str,
    draft: Optional["dh.DraftHeadContext"],
) -> torch.Tensor:
    if draft is None:
        raise ValueError(f"{block.name}: draft-head block requires a DraftHeadContext")
    if int(block.shape[0]) != draft.n:
        raise ValueError(f"{block.name}: block N {block.shape[0]} != shortlist N {draft.n}")
    sel = torch.from_numpy(draft.selected)
    if kind == "draft_idmap":
        return sel.to(torch.int32).contiguous()
    if kind == "draft_weights":
        full = reader.get("lm_head.weight")
        if int(full.shape[0]) <= int(sel.max().item()):
            raise ValueError(f"{block.name}: shortlist id exceeds lm_head rows {full.shape[0]}")
        return full.index_select(0, sel).contiguous()
    raise ValueError(f"{block.name}: unknown synthetic kind {kind!r}")


def materialize_block(
    reader: ShardReader,
    block: BlockPlan,
    draft: Optional["dh.DraftHeadContext"] = None,
) -> torch.Tensor:
    syn = block.segments[0].source.synthetic
    if syn is not None:
        w = _materialize_draft_block(reader, block, syn, draft)
        if block.shape is not None and tuple(int(x) for x in w.shape) != tuple(block.shape):
            raise ValueError(f"{block.name}: materialized shape {tuple(w.shape)} != plan {block.shape}")
        return w

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


def load_tokenizer_assets(tokenizer_dir: str, vocab_size: int) -> Tuple[TokenizerAsset, ...]:
    assets = []
    raw_assets = {}
    for kind in fmt.TOKENIZER_KINDS:
        source_name = fmt.TOKENIZER_SOURCE_NAME[kind]
        path = os.path.join(tokenizer_dir, source_name)
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            raise ValueError(f"missing tokenizer asset {path}: {exc}") from exc
        if size <= 0:
            raise ValueError(f"tokenizer asset is empty: {path}")
        if size > fmt.TOKENIZER_MAX_BYTES[kind]:
            raise ValueError(
                f"tokenizer asset {path} is {size} bytes, max {fmt.TOKENIZER_MAX_BYTES[kind]}"
            )
        try:
            with open(path, "rb") as f:
                data = f.read(size + 1)
        except OSError as exc:
            raise ValueError(f"failed to read tokenizer asset {path}: {exc}") from exc
        if len(data) != size:
            raise ValueError(f"tokenizer asset changed while reading: {path}")
        raw_assets[kind] = data
        assets.append(
            TokenizerAsset(
                kind=kind,
                source_name=source_name,
                data=data,
                crc32=fmt.crc32(data),
                sha256=hashlib.sha256(data).digest(),
            )
        )
    validate_tokenizer_assets(raw_assets, vocab_size)
    return tuple(assets)


def layout_tokenizer_assets(
    assets: Sequence[TokenizerAsset], data_offset: int
) -> Tuple[List[fmt.TokenizerRecord], bytes]:
    if tuple(asset.kind for asset in assets) != fmt.TOKENIZER_KINDS:
        raise ValueError("tokenizer assets are not in canonical kind order")
    records: List[fmt.TokenizerRecord] = []
    data_region = bytearray()
    for asset in assets:
        absolute = fmt.align_up(data_offset + len(data_region), fmt.TOKENIZER_ALIGN)
        data_region.extend(b"\x00" * (absolute - (data_offset + len(data_region))))
        records.append(
            fmt.TokenizerRecord(
                kind=asset.kind,
                encoding=fmt.TOKENIZER_RAW_UTF8,
                data_offset=absolute,
                data_bytes=len(asset.data),
                crc32=asset.crc32,
                sha256=asset.sha256,
            )
        )
        data_region.extend(asset.data)
    return records, bytes(data_region)


def _require_v4_2_output_path(out_path: str) -> None:
    basename = os.path.basename(os.path.normpath(out_path))
    if not basename.endswith(f"{OUTPUT_FORMAT_MARKER}.qus"):
        raise SystemExit(
            f"q5090 v4.2 converter output path must be a {OUTPUT_FORMAT_MARKER} .qus file: "
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
    tokenizer_assets: Sequence[TokenizerAsset],
    tokenizer_records: Sequence[fmt.TokenizerRecord],
    file_size: int,
    source_index_sha256: bytes,
) -> None:
    tc = cfg.get("text_config", cfg)
    text_bytes = sum(e.payload_bytes for e in entries if e.module_kind == qt.MODULE_TEXT)
    text_elems = sum(_prod(e.shape) for e in entries if e.module_kind == qt.MODULE_TEXT)
    group_summary = []
    fusion_summary_keys = []
    for g in fusion_records:
        first = entries[g.first_block_tensor_index]
        key = (
            first.module_kind,
            g.group_id,
            g.block_count,
            g.total_n,
            g.shared_k,
        )
        if key not in fusion_summary_keys:
            fusion_summary_keys.append(key)
    for module_kind, gid, block_count, total_n, shared_k in fusion_summary_keys:
        records = [
            g
            for g in fusion_records
            if entries[g.first_block_tensor_index].module_kind == module_kind
            and g.group_id == gid
            and g.block_count == block_count
            and g.total_n == total_n
            and g.shared_k == shared_k
        ]
        if records:
            group_summary.append(
                {
                    "module": qt.MODULE_NAME[module_kind],
                    "group_id": qt.FUSION_GROUP_NAME[gid],
                    "group_count": len(records),
                    "blocks_per_group": block_count,
                    "total_n": total_n,
                    "shared_k": shared_k,
                }
            )
    present_modules = [qt.MODULE_NAME[m.module_kind] for m in module_records]
    absent_modules = [
        qt.MODULE_NAME[k]
        for k in qt.MODULE_CANONICAL_ORDER
        if qt.MODULE_NAME[k] not in present_modules
    ]
    present_qtypes = [
        name
        for qtype, name in qt.QTYPE_NAME.items()
        if any(e.qtype == qtype for e in entries)
    ]
    present_layouts = [
        name
        for layout, name in qt.LAYOUT_NAME.items()
        if any(e.layout == layout for e in entries)
    ]

    draft_w = next(
        (
            e
            for e in entries
            if e.module_kind == qt.MODULE_LM_HEAD_DRAFT
            and e.source_kind == qt.SK_LM_HEAD_DRAFT
        ),
        None,
    )
    draft_id = next(
        (
            e
            for e in entries
            if e.module_kind == qt.MODULE_LM_HEAD_DRAFT
            and e.source_kind == qt.SK_LM_HEAD_DRAFT_IDMAP
        ),
        None,
    )
    draft_present = draft_w is not None

    manifest = {
        "format": "q5090_w4g64_mixed_v4_2",
        "format_version": fmt.VERSION,
        "format_minor": fmt.FORMAT_MINOR,
        "binary_spec": "docs/q5090_packed_file_format_v4.md",
        "tensor_plan": "docs/q5090_packed_file_format_v4.md",
        "value_source": (
            "per-group symmetric weight-only quant (§8); MTP dense/fused linears W8G32; "
            "optional draft lm_head = Q4G64 of original bf16 lm_head shortlist rows"
        ),
        "weights_file": os.path.basename(out_path),
        "file_bytes": int(file_size),
        "sha256_safetensors_index": source_index_sha256.hex(),
        "calibrated": False,
        "lm_head_draft_present": draft_present,
        "alignment": {
            "header": fmt.HEADER_SIZE,
            "tokenizer": fmt.TOKENIZER_ALIGN,
            "payload": fmt.REGION_ALIGN,
            "block": fmt.PAYLOAD_ALIGN,
            "k_pad": 128,
            "group_sizes": [32, 64],
        },
        "layouts": present_layouts,
        "code_planes": ["nibble", "high", "scale"],
        "qtypes": present_qtypes,
        "modules": present_modules,
        "absent_modules": absent_modules,
        "module_count": len(module_records),
        "tensor_count": len(entries),
        "segment_count": len(segment_records),
        "fusion_group_count": len(fusion_records),
        "tokenizer": {
            "record_count": len(tokenizer_records),
            "assets": [
                {
                    "kind": fmt.TOKENIZER_KIND_NAME[asset.kind],
                    "bytes": len(asset.data),
                    "crc32": f"{record.crc32:08x}",
                    "sha256": record.sha256.hex(),
                }
                for asset, record in zip(tokenizer_assets, tokenizer_records)
            ],
        },
        "fusion_groups": group_summary,
        "effective_text_bpw": round(text_bytes * 8 / max(1, text_elems), 4),
    }
    if draft_present:
        manifest["lm_head_draft"] = {
            "present": True,
            "module": "LM_HEAD_DRAFT",
            "n": int(draft_w.shape[0]),
            "k": int(draft_w.shape[1]),
            "weights_qtype": qt.QTYPE_NAME[draft_w.qtype],
            "weights_source_kind": "LM_HEAD_DRAFT",
            "weights_payload_bytes": int(draft_w.payload_bytes),
            "idmap_qtype": qt.QTYPE_NAME[draft_id.qtype],
            "idmap_source_kind": "LM_HEAD_DRAFT_IDMAP",
            "idmap_payload_bytes": int(draft_id.payload_bytes),
            "selection":
                "docs/archive/optimization-era/2026-07-06-lm-head-draft-q4-decision.md",
        }
    mpath = out_path + fmt.MANIFEST_SUFFIX
    mpath_tmp = mpath + ".tmp"
    with open(mpath_tmp, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(mpath_tmp, mpath)
    print(f"wrote {mpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description="q5090_w4g64_mixed_v4_2 artifact converter")
    ap.add_argument("--model", required=True, help="path to original Qwen3.6-27B safetensors dir")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer directory (default: --model)",
    )
    ap.add_argument("--out", default=None, help="output .qus file path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--ranking",
        default=dh.DEFAULT_RANKING,
        help="frequency ranking (.counts.i64) driving the draft-head shortlist",
    )
    ap.add_argument(
        "--no-draft-head",
        action="store_true",
        help="emit a v4.2 artifact without the LM_HEAD_DRAFT module",
    )
    args = ap.parse_args()

    model_dir = args.model.rstrip("/")
    out_path = args.out or os.path.join(
        os.getcwd(), "out", "qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus"
    )
    _require_v4_2_output_path(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(model_dir)
    assert_config(cfg)
    tc = cfg.get("text_config", cfg)
    device = pick_device(args.device)
    print(f"device: {device}")

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "rb") as f:
        index_raw = f.read()
    sha = hashlib.sha256(index_raw).digest()
    weight_map = json.loads(index_raw)["weight_map"]
    reader = ShardReader(model_dir, weight_map)

    tokenizer_dir = (args.tokenizer or model_dir).rstrip("/")
    tokenizer_assets = load_tokenizer_assets(tokenizer_dir, tc["vocab_size"])
    print(
        "tokenizer assets: "
        + ", ".join(f"{asset.source_name}={_fmt_bytes(len(asset.data))}" for asset in tokenizer_assets)
    )

    draft: Optional[dh.DraftHeadContext] = None
    if not args.no_draft_head:
        draft = dh.compute_shortlist(
            args.ranking, tokenizer_dir, dh.DRAFT_HEAD_N, tc["vocab_size"]
        )
        if draft.n != dh.DRAFT_HEAD_N:
            raise ValueError(
                f"draft shortlist size {draft.n} != fixed v4.2 size {dh.DRAFT_HEAD_N}"
            )
        print(
            f"draft head: N={draft.n} from {args.ranking}  "
            f"force-include specials={len(draft.force_include)}"
        )

    plan = build_conversion_plan(cfg, include_draft_head=draft is not None)
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
    tokenizer_index_offset = fmt.align_up(
        string_table_offset + string_table_bytes, fmt.TOKENIZER_ALIGN
    )
    tokenizer_index_bytes = fmt.TOKENIZER_RECORD_COUNT * fmt.TOKENIZER_RECORD_SIZE
    tokenizer_data_offset = fmt.align_up(
        tokenizer_index_offset + tokenizer_index_bytes, fmt.TOKENIZER_ALIGN
    )
    tokenizer_records, tokenizer_data = layout_tokenizer_assets(
        tokenizer_assets, tokenizer_data_offset
    )
    tokenizer_data_bytes = len(tokenizer_data)
    payload_region_start = fmt.align_up(
        tokenizer_data_offset + tokenizer_data_bytes, fmt.REGION_ALIGN
    )

    print(
        f"blocks: {tensor_count}  segments: {segment_count}  fusion_groups: {fusion_group_count}  "
        f"modules: {module_count}  tokenizer: {_fmt_bytes(tokenizer_data_bytes)}  "
        f"payload starts @ {payload_region_start}"
    )

    t0 = time.time()
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    with open(tmp_path, "wb") as f:
        f.seek(payload_region_start)
        for i, (block, entry) in enumerate(zip(plan.blocks, entries)):
            pos = f.tell()
            pad = fmt.align_up(pos, fmt.PAYLOAD_ALIGN) - pos
            if pad:
                f.write(b"\x00" * pad)
                pos += pad

            w = materialize_block(reader, block, draft)
            (
                payload,
                logical,
                padded,
                group,
                scale_dtype,
                nibble_plane_bytes,
                high_plane_bytes,
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
            entry.nibble_plane_bytes = nibble_plane_bytes
            entry.high_plane_bytes = high_plane_bytes
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

        module_flags = {
            qt.MODULE_TEXT: fmt.FLAG_TEXT_PRESENT,
            qt.MODULE_MTP: fmt.FLAG_MTP_PRESENT,
            qt.MODULE_VISION: fmt.FLAG_VISION_PRESENT,
            qt.MODULE_LM_HEAD_DRAFT: fmt.FLAG_LM_HEAD_DRAFT_PRESENT,
        }
        flags = 0
        for module in plan.modules:
            flags |= module_flags[module.module_kind]
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
            tokenizer_index_offset=tokenizer_index_offset,
            tokenizer_index_bytes=tokenizer_index_bytes,
            tokenizer_data_offset=tokenizer_data_offset,
            tokenizer_data_bytes=tokenizer_data_bytes,
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
        meta = meta.ljust(tokenizer_index_offset, b"\x00")
        for record in tokenizer_records:
            meta += fmt.pack_tokenizer_record(record)
        assert len(meta) == tokenizer_index_offset + tokenizer_index_bytes
        meta = meta.ljust(tokenizer_data_offset, b"\x00")
        meta += tokenizer_data
        assert len(meta) == tokenizer_data_offset + tokenizer_data_bytes
        assert len(meta) <= payload_region_start, (len(meta), payload_region_start)
        meta = meta.ljust(payload_region_start, b"\x00")
        f.seek(0)
        f.write(meta)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, out_path)

    _write_manifest(
        out_path,
        cfg,
        plan,
        module_records,
        entries,
        segment_records,
        fusion_records,
        tokenizer_assets,
        tokenizer_records,
        file_size,
        sha,
    )

    print(f"\nwrote {out_path}  ({_fmt_bytes(file_size)})  in {time.time() - t0:.0f}s")
    for m in module_records:
        print(
            f"  {qt.MODULE_NAME[m.module_kind]:14s} blocks={m.tensor_index_count:4d}  "
            f"payload={_fmt_bytes(m.payload_bytes)}  format={qt.MODULE_FORMAT[m.module_kind]}"
        )


if __name__ == "__main__":
    main()
