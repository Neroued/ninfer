"""CLI: Qwen3.6-27B bf16 safetensors -> q5090_w4g64_mixed_v1 packed file.

Usage:
  python -m tools.q5090_convert.convert --model /path/to/Qwen3.6-27B [--out FILE]
                                        [--no-mtp] [--no-vision] [--device cuda]
                                        [--limit-text-layers N] [--vision-blocks M]

Default output includes all three segments (TEXT_CORE + MTP_DRAFT + VISION_ENCODER).
Binary format: ../../docs/q5090_packed_file_format_v1.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

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
    return lts


def _prepare_source(reader: ShardReader, spec: tp.TensorSpec) -> torch.Tensor:
    t = reader.get(spec.src_name)
    if spec.row_slice is not None:
        a, b = spec.row_slice
        if t.shape[0] < b:
            raise ValueError(f"{spec.src_name}: rows {b} > tensor rows {t.shape[0]}")
        t = t[a:b]
    if spec.reshape is not None:
        if t.numel() != int(torch.tensor(spec.reshape).prod()):
            raise ValueError(f"{spec.src_name}: cannot reshape {tuple(t.shape)} -> {spec.reshape}")
        t = t.reshape(spec.reshape)
    if spec.layout != qt.LAYOUT_CONTIGUOUS and t.dim() != 2:
        raise ValueError(f"{spec.name}: quant layout needs 2D, got {tuple(t.shape)}")
    return t.contiguous()


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main() -> None:
    ap = argparse.ArgumentParser(description="q5090_w4g64_mixed_v1 quantizing converter")
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
        os.getcwd(), "out", "qwen3_6_27b.q5090_w4g64_mixed_v1.qus"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

    # --- build specs (text always; mtp/vision if present and enabled) ---
    layer_types = _layer_types(cfg)
    if args.limit_text_layers >= 0:
        layer_types = layer_types[: args.limit_text_layers]
    specs: List[tp.TensorSpec] = tp.build_text_specs(layer_types)
    module_specs = [(qt.MODULE_TEXT, len(specs), qt.LOAD_RESIDENT)]
    disabled: List[str] = []

    want_mtp = (not args.no_mtp) and reader.has("mtp.fc.weight")
    if not args.no_mtp and not want_mtp:
        disabled.append("MTP_DRAFT")
    if want_mtp:
        mtp = tp.build_mtp_specs()
        specs += mtp
        module_specs.append((qt.MODULE_MTP, len(mtp), qt.LOAD_RESIDENT))

    want_vision = (not args.no_vision) and reader.has("model.visual.merger.linear_fc1.weight")
    if not args.no_vision and not want_vision:
        disabled.append("VISION_ENCODER")
    if want_vision:
        depth = EXPECTED["vision_depth"] if args.vision_blocks < 0 else args.vision_blocks
        vis = tp.build_vision_specs(depth)
        specs += vis
        module_specs.append((qt.MODULE_VISION, len(vis), qt.LOAD_LAZY_GPU))

    # validate sources exist
    missing = sorted({s.src_name for s in specs if not reader.has(s.src_name)})
    if missing:
        raise SystemExit("missing source tensors:\n  " + "\n  ".join(missing))

    entries = [
        fmt.TensorEntry(
            name=s.name, qtype=s.qtype, layout=s.layout, module_kind=s.module_kind,
            shape=[], padded_shape=[], group_size=0, scale_dtype=qt.SCALE_NONE,
            source_layer=s.source_layer, source_kind=s.source_kind,
        )
        for s in specs
    ]

    # --- metadata layout (sizes known up front) ---
    string_table = fmt.build_string_table(entries)  # also assigns name_offset
    tensor_count = len(entries)
    module_count = len(module_specs)
    module_index_offset = fmt.HEADER_SIZE
    module_index_bytes = module_count * fmt.MODULE_RECORD_SIZE
    tensor_index_offset = module_index_offset + module_index_bytes
    tensor_index_bytes = tensor_count * fmt.TENSOR_ENTRY_SIZE
    string_table_offset = tensor_index_offset + tensor_index_bytes
    string_table_bytes = len(string_table)
    payload_region_start = fmt.align_up(string_table_offset + string_table_bytes, fmt.REGION_ALIGN)

    print(f"tensors: {tensor_count}  modules: {module_count}  payload starts @ {payload_region_start}")

    # --- stream payloads ---
    t0 = time.time()
    module_span: Dict[int, List[int]] = {}
    with open(out_path, "wb") as f:
        f.seek(payload_region_start)
        for i, (spec, entry) in enumerate(zip(specs, entries)):
            pos = f.tell()
            pad = fmt.align_up(pos, fmt.PAYLOAD_ALIGN) - pos
            if pad:
                f.write(b"\x00" * pad)
                pos += pad
            w = _prepare_source(reader, spec)
            payload, logical, padded, group, scale_dtype = encode_tensor(w, spec.qtype, spec.layout, device)
            f.write(payload)
            entry.shape = logical
            entry.padded_shape = padded
            entry.group_size = group
            entry.scale_dtype = scale_dtype
            entry.payload_offset = pos
            entry.payload_bytes = len(payload)
            entry.crc32 = fmt.crc32(payload)
            span = module_span.setdefault(spec.module_kind, [pos, pos + len(payload)])
            span[0] = min(span[0], pos)
            span[1] = max(span[1], pos + len(payload))
            if len(payload) >= (1 << 20) or i % 50 == 0 or i == tensor_count - 1:
                el = time.time() - t0
                print(f"  [{i + 1}/{tensor_count}] {spec.name}  {qt.QTYPE_NAME[spec.qtype]}  "
                      f"{logical} -> {_fmt_bytes(len(payload))}  ({el:.0f}s)", flush=True)
        file_size = f.tell()

        # --- build + write metadata block at offset 0 ---
        module_records = []
        begin = 0
        for kind, count, policy in module_specs:
            sp = module_span.get(kind, [payload_region_start, payload_region_start])
            module_records.append(fmt.ModuleRecord(
                module_kind=kind, module_version=1,
                tensor_index_begin=begin, tensor_index_count=count,
                payload_offset=sp[0], payload_bytes=sp[1] - sp[0], load_policy=policy,
            ))
            begin += count

        flags = 1
        if want_mtp:
            flags |= 2
        if want_vision:
            flags |= 4
        header = fmt.FileHeaderFields(
            tensor_count=tensor_count, module_count=module_count, layer_count=EXPECTED["num_hidden_layers"],
            flags=flags,
            module_index_offset=module_index_offset, module_index_bytes=module_index_bytes,
            tensor_index_offset=tensor_index_offset, tensor_index_bytes=tensor_index_bytes,
            string_table_offset=string_table_offset, string_table_bytes=string_table_bytes,
            payload_offset=payload_region_start, payload_bytes=file_size - payload_region_start,
            hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"], num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
            gdn_key_heads=tc["linear_num_key_heads"], gdn_value_heads=tc["linear_num_value_heads"],
            gdn_key_head_dim=tc["linear_key_head_dim"], gdn_value_head_dim=tc["linear_value_head_dim"],
            gdn_conv_width=tc["linear_conv_kernel_dim"], full_attention_interval=tc["full_attention_interval"],
            max_position_embeddings=tc["max_position_embeddings"], sha256_index=sha,
        )
        meta = bytearray()
        meta += fmt.pack_header(header)
        for m in module_records:
            meta += fmt.pack_module_record(m)
        for e in entries:
            meta += fmt.pack_tensor_entry(e)
        meta += string_table
        assert len(meta) <= payload_region_start, (len(meta), payload_region_start)
        meta = meta.ljust(payload_region_start, b"\x00")
        f.seek(0)
        f.write(meta)

    # --- manifest ---
    write_manifest(out_path, cfg, module_records, entries, disabled, file_size)

    # --- summary ---
    print(f"\nwrote {out_path}  ({_fmt_bytes(file_size)})  in {time.time() - t0:.0f}s")
    for m in module_records:
        print(f"  {qt.MODULE_NAME[m.module_kind]:14s} tensors={m.tensor_index_count:4d}  "
              f"payload={_fmt_bytes(m.payload_bytes)}  policy={qt.MODULE_POLICY[m.module_kind]}")
    if disabled:
        print("  disabled_segments:", disabled)


def write_manifest(out_path, cfg, module_records, entries, disabled, file_size) -> None:
    tc = cfg.get("text_config", cfg)
    by_mod = {}
    for e in entries:
        d = by_mod.setdefault(e.module_kind, [0, 0])
        d[0] += 1
        d[1] += e.payload_bytes
    # effective text bpw = text payload bits / text logical elements
    text_bytes = by_mod.get(qt.MODULE_TEXT, [0, 0])[1]
    text_elems = sum(int(_prod(e.shape)) for e in entries if e.module_kind == qt.MODULE_TEXT)
    segments = []
    for m in module_records:
        seg = {
            "kind": qt.MODULE_NAME[m.module_kind],
            "policy": qt.MODULE_POLICY[m.module_kind],
            "tensor_count": int(m.tensor_index_count),
            "payload_bytes": int(m.payload_bytes),
            "payload_gib": round(m.payload_bytes / (1024 ** 3), 4),
            "load_policy": int(m.load_policy),
        }
        if m.module_kind == qt.MODULE_VISION:
            seg["fallback"] = "vision_merger_bf16_strict"
        segments.append(seg)
    manifest = {
        "format": "q5090_w4g64_mixed_v1_final",
        "format_version": 1,
        "model": cfg.get("_name_or_path", "Qwen/Qwen3.6-27B"),
        "binary_spec": "docs/q5090_packed_file_format_v1.md",
        "file_bytes": int(file_size),
        "segments": segments,
        "disabled_segments": disabled,
        "effective_text_bpw": round(text_bytes * 8 / max(1, text_elems), 4),
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "qtypes": list(qt.QTYPE_NAME.values()),
        "layouts": list(qt.LAYOUT_NAME.values()),
        "zero_point": False,
        "alignment": {"file_header": fmt.HEADER_SIZE, "tensor_payload": fmt.PAYLOAD_ALIGN,
                      "payload_region": fmt.REGION_ALIGN},
    }
    mpath = os.path.join(os.path.dirname(out_path), "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {mpath}")


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


if __name__ == "__main__":
    main()
