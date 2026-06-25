"""Verify a packed q5090 file: structure + round-trip dequant error vs the source (GPU).

Usage:
  python -m tools.q5090_convert.verify --model /path/to/Qwen3.6-27B --file OUT.qus
                                       [--device cuda] [--quick]

Structural: magic/version, alignment, crc32, no payload overlap, name hashes.
Numerical: decode each tensor on GPU and compare to the original bf16 weights on GPU;
hard-fail on NaN/Inf, shape/crc mismatch, non-exact control tensors, or cosine < 0.95.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import torch

from . import format as fmt
from . import qtypes as qt
from . import tensor_plan as tp
from .layouts import decode_tensor
from .quantize import pick_device

COSINE_FAIL = 0.95
COSINE_WARN = 0.999


def _read_entries(path: str):
    with open(path, "rb") as f:
        hdr = fmt.unpack_header(f.read(fmt.HEADER_SIZE))
        problems = []
        if hdr["magic"] != fmt.MAGIC:
            problems.append(f"bad magic {hdr['magic']!r}")
        if hdr["version"] != fmt.VERSION:
            problems.append(f"bad version {hdr['version']}")
        if hdr["endian"] != fmt.ENDIAN_TAG:
            problems.append(f"bad endian {hdr['endian']:#x}")
        if hdr["header_size"] != fmt.HEADER_SIZE:
            problems.append(f"bad header_size {hdr['header_size']}")
        f.seek(hdr["module_index_offset"])
        modules = [fmt.unpack_module_record(f.read(fmt.MODULE_RECORD_SIZE)) for _ in range(hdr["module_count"])]
        f.seek(hdr["tensor_index_offset"])
        entries = [fmt.unpack_tensor_entry(f.read(fmt.TENSOR_ENTRY_SIZE)) for _ in range(hdr["tensor_count"])]
        f.seek(hdr["string_table_offset"])
        table = f.read(hdr["string_table_bytes"])
    for e in entries:
        e["name"] = table[e["name_offset"]: e["name_offset"] + e["name_len"]].decode("utf-8")
        if fmt.fnv1a_64(e["name"]) != e["name_hash"]:
            problems.append(f"{e['name']}: name_hash mismatch")
    return hdr, modules, entries, problems


def _structural_checks(path: str, hdr, entries) -> List[str]:
    problems = []
    file_size = os.path.getsize(path)
    ranges = []
    for e in entries:
        off, nb = e["payload_offset"], e["payload_bytes"]
        if off % fmt.PAYLOAD_ALIGN != 0:
            problems.append(f"{e['name']}: payload not {fmt.PAYLOAD_ALIGN}-aligned ({off})")
        if off < hdr["payload_offset"] or off + nb > file_size:
            problems.append(f"{e['name']}: payload out of range")
        ranges.append((off, off + nb, e["name"]))
    ranges.sort()
    for i in range(1, len(ranges)):
        if ranges[i][0] < ranges[i - 1][1]:
            problems.append(f"payload overlap: {ranges[i-1][2]} & {ranges[i][2]}")
    return problems


def _metrics(got: torch.Tensor, ref: torch.Tensor):
    g = got.reshape(-1)
    r = ref.reshape(-1)
    diff = g - r
    maxabs = float(diff.abs().max())
    meanabs = float(diff.abs().mean())
    dn2 = torch.sum(diff * diff, dtype=torch.float64)
    rn2 = torch.sum(r * r, dtype=torch.float64)
    gn2 = torch.sum(g * g, dtype=torch.float64)
    dot = torch.sum(g * r, dtype=torch.float64)
    rn = float(rn2.sqrt())
    dn = float(dn2.sqrt())
    gn = float(gn2.sqrt())
    rel = dn / rn if rn > 0 else dn
    if rn == 0 and gn == 0:
        cos = 1.0
    elif rn == 0 or gn == 0:
        cos = 0.0
    else:
        cos = float(dot / (rn2.sqrt() * gn2.sqrt()))
    nan = not bool(torch.isfinite(g).all())
    return maxabs, meanabs, rel, cos, nan


def main() -> None:
    ap = argparse.ArgumentParser(description="verify q5090 packed file")
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="structure + crc only, skip dequant")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    hdr, modules, entries, problems = _read_entries(args.file)
    print(f"file: {args.file}")
    print(f"tensors={hdr['tensor_count']} modules={hdr['module_count']} "
          f"flags={hdr['flags']:#x} file_bytes={os.path.getsize(args.file)}", flush=True)
    for m in modules:
        print(f"  module {qt.MODULE_NAME.get(m['module_kind'], m['module_kind'])}: "
              f"count={m['tensor_index_count']} payload={m['payload_bytes']}")
    problems += _structural_checks(args.file, hdr, entries)
    print(f"structural checks done; problems={len(problems)}", flush=True)

    fails = list(problems)
    spec_map = {}
    reader = None
    if not args.quick:
        from .convert import ShardReader, _layer_types, _prepare_source, load_config
        cfg = load_config(args.model)
        with open(os.path.join(args.model, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        reader = ShardReader(args.model, weight_map)
        for s in (tp.build_text_specs(_layer_types(cfg))
                  + tp.build_mtp_specs()
                  + tp.build_vision_specs(cfg.get("vision_config", {}).get("depth", 27))):
            spec_map[s.name] = s

    agg = defaultdict(lambda: {"n": 0, "rel": 0.0, "worst_rel": 0.0, "worst_cos": 1.0, "worst_max": 0.0})
    t0 = time.time()
    with open(args.file, "rb") as f:
        for i, e in enumerate(entries):
            f.seek(e["payload_offset"])
            payload = f.read(e["payload_bytes"])
            if fmt.crc32(payload) != e["crc32"]:
                fails.append(f"{e['name']}: crc32 mismatch")
            if args.quick:
                continue
            spec = spec_map.get(e["name"])
            if spec is None:
                fails.append(f"{e['name']}: no source spec")
                continue
            got = decode_tensor(payload, e["qtype"], e["layout"], e["shape"], e["padded_shape"], device).float()
            from .convert import _prepare_source  # local import keeps safetensors optional
            src = _prepare_source(reader, spec).to(device=device, dtype=torch.float32)
            if list(got.shape) != list(src.shape):
                fails.append(f"{e['name']}: shape {tuple(got.shape)} != source {tuple(src.shape)}")
                continue
            maxabs, meanabs, rel, cos, nan = _metrics(got, src)
            a = agg[e["qtype"]]
            a["n"] += 1
            a["rel"] += rel
            a["worst_rel"] = max(a["worst_rel"], rel)
            a["worst_cos"] = min(a["worst_cos"], cos)
            a["worst_max"] = max(a["worst_max"], maxabs)
            if nan:
                fails.append(f"{e['name']}: NaN/Inf in dequant")
            if not qt.is_quant(e["qtype"]) and maxabs > 0:
                fails.append(f"{e['name']}: control tensor not exact (maxabs={maxabs:.3e})")
            if qt.is_quant(e["qtype"]) and cos < COSINE_FAIL:
                fails.append(f"{e['name']}: cosine {cos:.4f} < {COSINE_FAIL}")
            elif qt.is_quant(e["qtype"]) and cos < COSINE_WARN:
                print(f"  WARN {e['name']}: cosine={cos:.5f} rel={rel:.4f}", flush=True)
            del got, src
            if i % 100 == 0 or i == len(entries) - 1:
                print(f"  [{i + 1}/{len(entries)}] {e['name']} ({time.time() - t0:.0f}s)", flush=True)

    if not args.quick:
        print("\nper-qtype round-trip summary:")
        for qtype, a in sorted(agg.items()):
            print(f"  {qt.QTYPE_NAME[qtype]:12s} n={a['n']:4d}  mean_rel={a['rel']/max(1,a['n']):.4f}  "
                  f"worst_rel={a['worst_rel']:.4f}  worst_cos={a['worst_cos']:.5f}  "
                  f"worst_maxabs={a['worst_max']:.3e}")

    print()
    if fails:
        print(f"FAILED ({len(fails)} problems):")
        for p in fails[:50]:
            print("  -", p)
        raise SystemExit(1)
    print(f"OK: all checks passed in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
