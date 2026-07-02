#!/usr/bin/env python3
"""Self-contained PyTorch oracle for the qwen3.6-ultraspeed M2 schedule.

The oracle reads our q5090 file directly and dequantizes weights through
tools.q5090_convert.layouts.decode_tensor. It is deliberately correctness-first:
weights stream layer-by-layer, activations are rounded to bf16 after each
L1-equivalent op, and greedy decode is implemented without tokenizer dependencies.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.layouts import decode_row_split_quantized, decode_tensor
from tools.q5090_convert.packing import row_split_plane_sizes

torch.set_grad_enabled(False)

DEFAULT_MODEL = "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16"
DEFAULT_PROMPT = "请用三条短句说明，为什么每天适量喝水很重要？每条不超过 18 个字。"
DEFAULT_STOP_TOKEN_IDS = "248046,248044"

D = 5120
L = 64
I = 17408
V = 248320
H_Q = 24
H_KV = 4
DH = 256
Q_SIZE = H_Q * DH
KV_SIZE = H_KV * DH
GDN_HK = 16
GDN_HV = 48
GDN_DK = 128
GDN_DV = 128
GDN_KEY = GDN_HK * GDN_DK
GDN_VALUE = GDN_HV * GDN_DV
GDN_CONV = 2 * GDN_KEY + GDN_VALUE
GDN_GROUP = GDN_HV // GDN_HK
EPS = 1.0e-6
ROPE_THETA = 1.0e7
ROTARY_DIM = 64
ATTN_SCALE = 1.0 / (DH**0.5)
GDN_SCALE = 1.0 / (GDN_DK**0.5)


def is_full(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def full_idx(layer: int) -> int:
    return (layer + 1) // 4 - 1


def gdn_idx(layer: int) -> int:
    return layer - full_idx(layer) - 1 if is_full(layer) else layer - ((layer + 1) // 4)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def prod(xs: Iterable[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return bf16(x.to(torch.bfloat16) @ w.to(torch.bfloat16).t())


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    unit_offset: bool = True,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + EPS)
    wf = weight.float()
    if unit_offset:
        wf = wf + 1.0
    out = xf * inv * wf
    if z is not None:
        out = out * F.silu(z.float())
    return bf16(out)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.sum(xf * xf, dim=-1, keepdim=True) + EPS)
    return bf16(xf * inv)


def residual_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return bf16(x.float() + y.float())


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return bf16(F.silu(gate.float()) * up.float())


def sigmoid_gate_mul(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return bf16(torch.sigmoid(gate.float()) * x.float())


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q [T,24,256], k [T,4,256], partial NeoX split over dims [0:32] and [32:64].
    device = q.device
    half = ROTARY_DIM // 2
    pair = torch.arange(half, device=device, dtype=torch.float32)
    freq = torch.pow(torch.tensor(ROPE_THETA, device=device, dtype=torch.float32), -2.0 * pair / ROTARY_DIM)
    angle = positions.to(device=device, dtype=torch.float32)[:, None] * freq[None, :]
    cos = torch.cos(angle)[:, None, :]
    sin = torch.sin(angle)[:, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        x1 = x[:, :, :half].float()
        x2 = x[:, :, half:ROTARY_DIM].float()
        y[:, :, :half] = bf16(x1 * cos - x2 * sin)
        y[:, :, half:ROTARY_DIM] = bf16(x2 * cos + x1 * sin)
        return y

    return rotate(q), rotate(k)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    # x [T,C], state [C,3] oldest -> newest.
    # The q5090 conv1d weight is stored in the engine's runtime-native layout:
    # metadata shape [C,K,1] but the flat bytes are [K,C] (dim0-fastest). A naive
    # `reshape(C,K)` would scramble taps across channels, so recover [C,K] as
    # w[c,k] = flat[k*C + c].
    T, C = x.shape
    K = weight.numel() // C
    w = weight.reshape(-1).reshape(K, C).transpose(0, 1).contiguous().float()
    old = state.clone()
    outs = []
    for t in range(T):
        x0 = x[t - 3] if t >= 3 else old[:, t]
        x1 = x[t - 2] if t >= 2 else old[:, t + 1]
        x2 = x[t - 1] if t >= 1 else old[:, t + 2]
        x3 = x[t]
        acc = w[:, 0] * x0.float() + w[:, 1] * x1.float() + w[:, 2] * x2.float() + w[:, 3] * x3.float()
        outs.append(bf16(F.silu(acc)))
    for s in range(3):
        seq_pos = T + s
        if seq_pos == 0:
            state[:, s] = old[:, 0]
        elif seq_pos == 1:
            state[:, s] = old[:, 1]
        elif seq_pos == 2:
            state[:, s] = old[:, 2]
        else:
            state[:, s] = x[seq_pos - 3]
    return torch.stack(outs, dim=0)


def gdn_gating(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sp = torch.where(a.float() + dt_bias.float() > 20.0, a.float() + dt_bias.float(), F.softplus(a.float() + dt_bias.float()))
    g = -torch.exp(a_log.float()) * sp
    beta = torch.sigmoid(b.float())
    return g.float(), beta.float()


@dataclass
class Entry:
    block_index: int
    name: str
    qtype: int
    layout: int
    module_kind: int
    shape: List[int]
    padded_shape: List[int]
    payload_offset: int
    payload_bytes: int
    nibble_plane_bytes: int
    high_plane_bytes: int
    scale_plane_bytes: int
    source_layer: int
    source_kind: int
    crc32: int
    segment_begin: int
    segment_count: int
    fusion_group_id: int
    fusion_index: int


@dataclass
class WeightView:
    segment_index: int
    name: str
    block: Entry
    source_kind: int
    source_layer: int
    row_begin: int
    row_count: int


@dataclass
class SpeculativeStats:
    draft_tokens: int = 0
    accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.draft_tokens


class Q5090File:
    def __init__(self, path: str | Path, resident_device: torch.device | str | None = None):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        # When set, each tensor payload is uploaded to this device once and
        # decoded in place on every later forward. This turns the oracle from
        # "re-read + re-upload 15 GiB per forward" into a one-time upload, which
        # is the dominant cost of decode-step throughput.
        self._resident_device = (
            torch.device(resident_device) if resident_device is not None else None
        )
        self._gpu: Dict[str, torch.Tensor] = {}
        hdr = fmt.unpack_header(self._mm[: fmt.HEADER_SIZE])
        if hdr["magic"] != fmt.MAGIC or hdr["version"] != fmt.VERSION:
            raise ValueError(f"bad q5090 v3 header in {self.path}")
        self.header = hdr
        self.modules = []
        off = hdr["module_index_offset"]
        for _ in range(hdr["module_count"]):
            self.modules.append(fmt.unpack_module_record(self._mm[off : off + fmt.MODULE_RECORD_SIZE]))
            off += fmt.MODULE_RECORD_SIZE
        raw_entries = []
        off = hdr["tensor_index_offset"]
        for _ in range(hdr["tensor_count"]):
            raw_entries.append(fmt.unpack_tensor_entry(self._mm[off : off + fmt.TENSOR_ENTRY_SIZE]))
            off += fmt.TENSOR_ENTRY_SIZE
        raw_segments = []
        off = hdr["segment_index_offset"]
        for _ in range(hdr["segment_count"]):
            raw_segments.append(fmt.unpack_segment_record(self._mm[off : off + fmt.SEGMENT_RECORD_SIZE]))
            off += fmt.SEGMENT_RECORD_SIZE
        self.fusions = []
        off = hdr["fusion_group_index_offset"]
        for _ in range(hdr["fusion_group_count"]):
            self.fusions.append(fmt.unpack_fusion_group_record(self._mm[off : off + fmt.FUSION_GROUP_RECORD_SIZE]))
            off += fmt.FUSION_GROUP_RECORD_SIZE
        table = self._mm[hdr["string_table_offset"] : hdr["string_table_offset"] + hdr["string_table_bytes"]]

        def read_name(record: dict, label: str) -> str:
            begin = record["name_offset"]
            end = begin + record["name_len"]
            if begin < 0 or end > len(table) or end >= len(table) or table[end] != 0:
                raise ValueError(f"{label}: invalid string table range")
            name = table[begin:end].decode("utf-8")
            if fmt.fnv1a_64(name) != record["name_hash"]:
                raise ValueError(f"{label}: name_hash mismatch")
            return name

        for i, e in enumerate(raw_entries):
            e["name"] = read_name(e, f"block[{i}]")
        for i, s in enumerate(raw_segments):
            s["name"] = read_name(s, f"segment[{i}]")

        self.segments = raw_segments
        self.blocks: List[Entry] = []
        self.entries: Dict[str, Entry] = {}
        for i, e in enumerate(raw_entries):
            entry = Entry(
                block_index=i,
                name=e["name"],
                qtype=e["qtype"],
                layout=e["layout"],
                module_kind=e["module_kind"],
                shape=e["shape"],
                padded_shape=e["padded_shape"],
                payload_offset=e["payload_offset"],
                payload_bytes=e["payload_bytes"],
                nibble_plane_bytes=e["nibble_plane_bytes"],
                high_plane_bytes=e["high_plane_bytes"],
                scale_plane_bytes=e["scale_plane_bytes"],
                source_layer=e["source_layer"],
                source_kind=e["source_kind"],
                crc32=e["crc32"],
                segment_begin=e["segment_begin"],
                segment_count=e["segment_count"],
                fusion_group_id=e["fusion_group_id"],
                fusion_index=e["fusion_index"],
            )
            self.blocks.append(entry)
            self.entries[entry.name] = entry
        self.views: Dict[str, WeightView] = {}
        for block in self.blocks:
            begin = block.segment_begin
            end = begin + block.segment_count
            if end > len(self.segments):
                raise ValueError(f"{block.name}: segment range outside segment table")
            for segment_index, segment in enumerate(self.segments[begin:end], begin):
                view = WeightView(
                    segment_index=segment_index,
                    name=segment["name"],
                    block=block,
                    source_kind=segment["source_kind"],
                    source_layer=segment["source_layer"],
                    row_begin=segment["row_begin"],
                    row_count=segment["row_count"],
                )
                if view.name in self.views:
                    raise ValueError(f"duplicate segment name: {view.name}")
                self.views[view.name] = view

    def _resident_payload(self, block: Entry) -> Optional[torch.Tensor]:
        """Return the full payload of `block` as a resident uint8 tensor, or None.

        Populated lazily on first use. If the upload runs the device out of
        memory we transparently disable residency and stream from mmap instead,
        so the oracle still completes (just slower) on memory-tight devices.
        """
        if self._resident_device is None:
            return None
        # Only the large low-bit quant tensors are worth keeping resident and
        # decode cleanly from a raw uint8 blob. The small CONTIGUOUS control
        # tensors need uint16/f4 reinterpretation, so they keep streaming from
        # mmap (negligible cost).
        if block.layout == qt.LAYOUT_CONTIGUOUS:
            return None
        cached = self._gpu.get(block.name)
        if cached is not None:
            return cached
        host = np.frombuffer(
            self._mm[block.payload_offset : block.payload_offset + block.payload_bytes], dtype=np.uint8
        )
        try:
            tensor = torch.from_numpy(host.copy()).to(self._resident_device)
        except torch.cuda.OutOfMemoryError:
            self._resident_device = None
            self._gpu.clear()
            torch.cuda.empty_cache()
            return None
        self._gpu[block.name] = tensor
        return tensor

    def tensor(
        self,
        name: str,
        device: torch.device | str,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        view = self.views[name]
        return self._decode_view(view, device, output_dtype=output_dtype)

    def block_tensor(
        self,
        name: str,
        device: torch.device | str,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        e = self.entries[name]
        resident = self._resident_payload(e)
        payload = (
            resident
            if resident is not None
            else self._mm[e.payload_offset : e.payload_offset + e.payload_bytes]
        )
        return decode_tensor(
            payload,
            e.qtype,
            e.layout,
            e.shape,
            e.padded_shape,
            device=device,
            output_dtype=output_dtype,
        )

    def _decode_view(
        self,
        view: WeightView,
        device: torch.device | str,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        e = view.block
        resident = self._resident_payload(e)
        payload = (
            resident
            if resident is not None
            else self._mm[e.payload_offset : e.payload_offset + e.payload_bytes]
        )
        if e.layout == qt.LAYOUT_ROW_SPLIT:
            _, k = e.shape
            _, kp = e.padded_shape
            if view.row_begin == 0 and view.row_count == e.shape[0]:
                payload = (
                    resident
                    if resident is not None
                    else self._mm[e.payload_offset : e.payload_offset + e.payload_bytes]
                )
            else:
                payload = self._row_split_slice_payload(e, resident, view.row_begin, view.row_count)
            return decode_tensor(
                payload,
                e.qtype,
                e.layout,
                [view.row_count, k],
                [view.row_count, kp],
                device=device,
                output_dtype=output_dtype,
            )
        if view.row_begin != 0 or view.row_count != e.shape[0]:
            raise ValueError(f"{view.name}: cannot slice CONTIGUOUS segment from {e.name}")
        return decode_tensor(
            payload,
            e.qtype,
            e.layout,
            e.shape,
            e.padded_shape,
            device=device,
            output_dtype=output_dtype,
        )

    def _row_split_geometry(self, e: Entry):
        if e.layout != qt.LAYOUT_ROW_SPLIT:
            raise ValueError(f"{e.name} is not ROW_SPLIT")
        if e.qtype not in qt.QUANT_SPECS:
            raise ValueError(f"{e.name} is not a quantized ROW_SPLIT tensor")
        spec = qt.QUANT_SPECS[e.qtype]
        n, k = e.shape
        _, kp = e.padded_shape
        groups = kp // spec.group_size
        nibble_bytes, high_off, high_bytes, scale_off, scale_bytes, _ = row_split_plane_sizes(
            n,
            groups,
            spec.nibble_bytes_per_group,
            spec.high_bytes_per_group,
        )
        if (
            e.nibble_plane_bytes != nibble_bytes
            or e.high_plane_bytes != high_bytes
            or e.scale_plane_bytes != scale_bytes
        ):
            raise ValueError(
                f"{e.name} ROW_SPLIT plane size mismatch: "
                f"entry=({e.nibble_plane_bytes},{e.high_plane_bytes},{e.scale_plane_bytes}) "
                f"computed=({nibble_bytes},{high_bytes},{scale_bytes})"
            )
        return (
            n,
            k,
            kp,
            groups * spec.nibble_bytes_per_group,
            groups * spec.high_bytes_per_group,
            groups * 2,
            high_off,
            scale_off,
        )

    def _payload_slice(self, e: Entry, resident: Optional[torch.Tensor], rel: int, nbytes: int):
        if resident is not None:
            return resident[rel : rel + nbytes]
        start = e.payload_offset + rel
        return self._mm[start : start + nbytes]

    def _row_split_slice_payload(self, e: Entry, resident: Optional[torch.Tensor], row0: int, row_count: int):
        _, _, _, row_nibble_bytes, row_high_bytes, row_scale_bytes, high_off, scale_off = self._row_split_geometry(e)
        nibble_nbytes = row_count * row_nibble_bytes
        high_nbytes = row_count * row_high_bytes
        scale_nbytes = row_count * row_scale_bytes
        nibble = self._payload_slice(e, resident, row0 * row_nibble_bytes, nibble_nbytes)
        high = self._payload_slice(e, resident, high_off + row0 * row_high_bytes, high_nbytes)
        scale = self._payload_slice(e, resident, scale_off + row0 * row_scale_bytes, scale_nbytes)
        nibble_pad_bytes = fmt.align_up(nibble_nbytes, fmt.PAYLOAD_ALIGN) - nibble_nbytes
        high_pad_bytes = fmt.align_up(high_nbytes, fmt.PAYLOAD_ALIGN) - high_nbytes if high_nbytes else 0
        if resident is not None:
            parts = [nibble]
            if nibble_pad_bytes:
                parts.append(torch.zeros((nibble_pad_bytes,), dtype=torch.uint8, device=resident.device))
            if high_nbytes:
                parts.append(high)
                if high_pad_bytes:
                    parts.append(torch.zeros((high_pad_bytes,), dtype=torch.uint8, device=resident.device))
            parts.append(scale)
            return torch.cat(parts, dim=0)
        return (
            nibble
            + (b"\x00" * nibble_pad_bytes)
            + high
            + (b"\x00" * high_pad_bytes)
            + scale
        )

    def row_split_rows(
        self,
        name: str,
        rows: torch.Tensor,
        device: torch.device | str,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        view = self.views[name]
        e = view.block
        _, k, kp, *_ = self._row_split_geometry(e)
        resident = self._resident_payload(e)
        out = []
        for row in rows.detach().cpu().tolist():
            if row < 0 or row >= view.row_count:
                raise IndexError(f"{name} row out of range: {row}")
            payload = self._row_split_slice_payload(e, resident, view.row_begin + row, 1)
            out.append(
                decode_tensor(
                    payload,
                    e.qtype,
                    e.layout,
                    [1, k],
                    [1, kp],
                    device=device,
                    output_dtype=output_dtype,
                )
            )
        return torch.cat(out, dim=0)

    def row_split_row_chunks(
        self,
        name: str,
        device: torch.device | str,
        rows_per_chunk: int = 8192,
        *,
        output_dtype: torch.dtype = torch.float32,
    ):
        view = self.views[name]
        e = view.block
        _, k, kp, *_ = self._row_split_geometry(e)
        resident = self._resident_payload(e)
        for row0 in range(0, view.row_count, rows_per_chunk):
            row1 = min(view.row_count, row0 + rows_per_chunk)
            row_count = row1 - row0
            payload = self._row_split_slice_payload(e, resident, view.row_begin + row0, row_count)
            chunk = decode_tensor(
                payload,
                e.qtype,
                e.layout,
                [row_count, k],
                [row_count, kp],
                device=device,
                output_dtype=output_dtype,
            )
            yield row0, row1, chunk

    def _raw_payload(self, e: Entry):
        return self._mm[e.payload_offset : e.payload_offset + e.payload_bytes]

    def _row_split_probes(self, e: Entry, device: torch.device | str = "cpu") -> List[dict]:
        n, k = e.shape
        _, kp = e.padded_shape
        spec = qt.QUANT_SPECS[e.qtype]
        probes = []
        for row, col in [(0, 0), (n // 2, k // 2), (n - 1, k - 1)]:
            payload = self._row_split_slice_payload(e, None, row, 1)
            scale16, codes = decode_row_split_quantized(payload, [1, kp], e.qtype, device)
            group = col // spec.group_size
            lane = col % spec.group_size
            q = int(codes[0, group, lane].item())
            scale = float(scale16[0, group].float().item())
            probes.append({"row": int(row), "col": int(col), "scale": scale, "q": q, "value": scale * q})
        return probes

    def _contiguous_probes(self, e: Entry, device: torch.device | str = "cpu") -> List[dict]:
        t = decode_tensor(self._raw_payload(e), e.qtype, e.layout, e.shape, e.padded_shape, device)
        flat = t.reshape(-1)
        if flat.numel() == 0:
            return []
        out = []
        for idx in [0, int(flat.numel() // 2), int(flat.numel() - 1)]:
            if len(e.shape) == 1:
                row, col = idx, 0
            else:
                cols = prod(e.shape[1:])
                row, col = idx // cols, idx % cols
            out.append({"row": int(row), "col": int(col), "value": float(flat[idx].item())})
        return out

    def dump(self, path: str | Path, probe_device: torch.device | str = "cpu") -> None:
        blocks = []
        for e in self.blocks:
            block_segments = []
            for j, s in enumerate(self.segments[e.segment_begin : e.segment_begin + e.segment_count]):
                block_segments.append(
                    {
                        "segment_index": e.segment_begin + j,
                        "name": s["name"],
                        "source_kind": s["source_kind"],
                        "source_layer": s["source_layer"],
                        "row_begin": s["row_begin"],
                        "row_count": s["row_count"],
                    }
                )
            if qt.is_quant(e.qtype):
                dequant_probes = self._row_split_probes(e, probe_device)
            else:
                dequant_probes = self._contiguous_probes(e, probe_device)
            blocks.append(
                {
                    "block_index": e.block_index,
                    "name": e.name,
                    "source_kind": e.source_kind,
                    "source_layer": e.source_layer,
                    "qtype": qt.QTYPE_NAME.get(e.qtype, str(e.qtype)),
                    "layout": qt.LAYOUT_NAME.get(e.layout, str(e.layout)),
                    "shape": e.shape,
                    "padded_shape": e.padded_shape,
                    "payload_offset": e.payload_offset,
                    "payload_bytes": e.payload_bytes,
                    "nibble_plane_bytes": e.nibble_plane_bytes,
                    "high_plane_bytes": e.high_plane_bytes,
                    "scale_plane_bytes": e.scale_plane_bytes,
                    "crc32": e.crc32,
                    "fusion_group_id": e.fusion_group_id,
                    "fusion_index": e.fusion_index,
                    "segments": block_segments,
                    "dequant_probes": dequant_probes,
                }
            )

        fusion_dump = []
        for g in self.fusions:
            first = g["first_block_tensor_index"]
            count = g["block_count"]
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
                    "members": [self.blocks[i].name for i in range(first, min(first + count, len(self.blocks)))],
                }
            )

        header = {
            k: (v.hex() if isinstance(v, bytes) else v)
            for k, v in self.header.items()
        }
        out = {
            "format": "q5090_w4g64_mixed_v3",
            "file": str(self.path),
            "header": header,
            "modules": self.modules,
            "blocks": blocks,
            "fusion_groups": fusion_dump,
        }
        path = Path(path)
        os.makedirs(path.parent or ".", exist_ok=True)
        with path.open("w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")

    def close(self) -> None:
        self._mm.close()
        self._fh.close()


class RefModel:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        resident: str = "auto",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python has CPU-only PyTorch. "
                "Run with a CUDA torch environment, for example "
                "/home/neroued/miniconda3/envs/vllm-bench/bin/python, "
                "or pass --device cpu explicitly."
            )
        self.device = torch.device(device)
        # "auto": keep payloads resident on the compute device when it is CUDA
        # (one-time upload instead of per-forward streaming). "gpu"/"stream"
        # force the behaviour. CPU stays on the mmap-streaming path.
        if resident not in {"auto", "gpu", "stream"}:
            raise ValueError(f"resident must be auto/gpu/stream, got {resident!r}")
        if resident == "gpu" or (resident == "auto" and self.device.type == "cuda"):
            resident_device: torch.device | None = self.device
        else:
            resident_device = None
        self.q5090 = Q5090File(weights, resident_device=resident_device)
        self.reset_state()

    def weight(self, name: str) -> torch.Tensor:
        return self.q5090.tensor(name, self.device, output_dtype=torch.bfloat16)

    def block_weight(self, name: str) -> torch.Tensor:
        return self.q5090.block_tensor(name, self.device, output_dtype=torch.bfloat16)

    def reset_state(self) -> None:
        self.kv: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.mtp_kv: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.conv: Dict[int, torch.Tensor] = {
            i: torch.zeros(GDN_CONV, 3, device=self.device, dtype=torch.float32) for i in range(48)
        }
        self.ssm: Dict[int, torch.Tensor] = {
            i: torch.zeros(GDN_HV, GDN_DV, GDN_DK, device=self.device, dtype=torch.float32) for i in range(48)
        }
        self.pos = 0

    def mtp_kv_len(self, layer: int = 0) -> int:
        kv = self.mtp_kv.get(layer)
        if kv is None:
            return 0
        return int(kv[0].shape[0])

    def truncate_mtp_kv(self, length: int, layer: int = 0) -> None:
        if length < 0:
            raise ValueError("MTP KV length must be nonnegative")
        kv = self.mtp_kv.get(layer)
        if kv is None:
            if length != 0:
                raise ValueError(f"cannot truncate empty MTP KV to {length}")
            return
        k, v = kv
        if length > k.shape[0]:
            raise ValueError(f"cannot extend MTP KV from {k.shape[0]} to {length}")
        self.mtp_kv[layer] = (k[:length].clone(), v[:length].clone())

    def embed(self, ids: Iterable[int]) -> torch.Tensor:
        idx = torch.tensor(list(ids), device=self.device, dtype=torch.long)
        return bf16(
            self.q5090.row_split_rows(
                "model.language_model.embed_tokens.weight",
                idx,
                self.device,
                output_dtype=torch.bfloat16,
            )
        )

    def gqa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        fidx: int,
        phase: str,
    ) -> torch.Tensor:
        T = q.shape[0]
        if phase == "prefill":
            self.kv[fidx] = (k.clone(), v.clone())
            k_all, v_all = self.kv[fidx]
            k_rep = k_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            v_rep = v_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            scores = torch.einsum("thd,shd->ths", q.float(), k_rep) * ATTN_SCALE
            t_idx = torch.arange(T, device=self.device)
            s_idx = torch.arange(k_all.shape[0], device=self.device)
            scores = scores.masked_fill(s_idx[None, None, :] > t_idx[:, None, None], -torch.inf)
            probs = torch.softmax(scores, dim=-1)
            return bf16(torch.einsum("ths,shd->thd", probs, v_rep))
        else:
            old_k, old_v = self.kv.get(
                fidx,
                (
                    torch.empty(0, H_KV, DH, device=self.device),
                    torch.empty(0, H_KV, DH, device=self.device),
                ),
            )
            self.kv[fidx] = (torch.cat([old_k, k], dim=0), torch.cat([old_v, v], dim=0))
            k_all, v_all = self.kv[fidx]
            k_rep = k_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            v_rep = v_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            scores = torch.einsum("thd,shd->ths", q.float(), k_rep) * ATTN_SCALE
            probs = torch.softmax(scores, dim=-1)
            return bf16(torch.einsum("ths,shd->thd", probs, v_rep))

    def mtp_gqa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        phase: str,
    ) -> torch.Tensor:
        T = q.shape[0]
        if phase == "prefill":
            self.mtp_kv[layer] = (k.clone(), v.clone())
            k_all, v_all = self.mtp_kv[layer]
            k_rep = k_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            v_rep = v_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            scores = torch.einsum("thd,shd->ths", q.float(), k_rep) * ATTN_SCALE
            t_idx = torch.arange(T, device=self.device)
            s_idx = torch.arange(k_all.shape[0], device=self.device)
            scores = scores.masked_fill(s_idx[None, None, :] > t_idx[:, None, None], -torch.inf)
            probs = torch.softmax(scores, dim=-1)
            return bf16(torch.einsum("ths,shd->thd", probs, v_rep))
        if phase != "decode":
            raise ValueError(f"unknown MTP attention phase {phase!r}")
        old_k, old_v = self.mtp_kv.get(
            layer,
            (
                torch.empty(0, H_KV, DH, device=self.device),
                torch.empty(0, H_KV, DH, device=self.device),
            ),
        )
        self.mtp_kv[layer] = (torch.cat([old_k, k], dim=0), torch.cat([old_v, v], dim=0))
        k_all, v_all = self.mtp_kv[layer]
        k_rep = k_all.repeat_interleave(H_Q // H_KV, dim=1).float()
        v_rep = v_all.repeat_interleave(H_Q // H_KV, dim=1).float()
        scores = torch.einsum("thd,shd->ths", q.float(), k_rep) * ATTN_SCALE
        probs = torch.softmax(scores, dim=-1)
        return bf16(torch.einsum("ths,shd->thd", probs, v_rep))

    def gdn_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        gidx: int,
    ) -> torch.Tensor:
        T = q.shape[0]
        state = self.ssm[gidx]
        out = torch.empty(T, GDN_HV, GDN_DV, device=self.device, dtype=torch.float32)
        hv_to_hk = torch.arange(GDN_HV, device=self.device, dtype=torch.long) // GDN_GROUP
        for t in range(T):
            kt = k[t].float().index_select(0, hv_to_hk)
            qt = q[t].float().index_select(0, hv_to_hk)
            state.mul_(torch.exp(g[t].float()).view(GDN_HV, 1, 1))
            sk = torch.bmm(state, kt.unsqueeze(-1)).squeeze(-1)
            delta = beta[t].float().unsqueeze(-1) * (v[t].float() - sk)
            state.add_(delta.unsqueeze(-1) * kt.unsqueeze(1))
            out[t] = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1) * GDN_SCALE
        return bf16(out)

    def attn_mix(self, layer: int, x: torch.Tensor, phase: str, positions: torch.Tensor) -> torch.Tensor:
        p = f"layers.{layer}."
        h = rmsnorm(x, self.weight(p + "input_layernorm.weight"), unit_offset=True)
        qk = linear(h, self.block_weight(p + "attn_in.q4"))
        gatev = linear(h, self.block_weight(p + "attn_in.q5"))
        q = qk[:, :Q_SIZE].reshape(-1, H_Q, DH)
        k = qk[:, Q_SIZE:].reshape(-1, H_KV, DH)
        gate = gatev[:, :Q_SIZE].reshape(-1, H_Q, DH)
        v = gatev[:, Q_SIZE:].reshape(-1, H_KV, DH)
        qn = rmsnorm(q, self.weight(p + "self_attn.q_norm.weight"), unit_offset=True)
        kn = rmsnorm(k, self.weight(p + "self_attn.k_norm.weight"), unit_offset=True)
        qn, kn = apply_rope(qn, kn, positions)
        a = self.gqa_attention(qn, kn, v, full_idx(layer), phase)
        a = sigmoid_gate_mul(gate, a).reshape(-1, Q_SIZE)
        o = linear(a, self.weight(p + "self_attn.o_proj.weight"))
        return residual_add(x, o)

    def gdn_mix(self, layer: int, x: torch.Tensor, phase: str) -> torch.Tensor:
        p = f"layers.{layer}."
        gidx = gdn_idx(layer)
        h = rmsnorm(x, self.weight(p + "input_layernorm.weight"), unit_offset=True)
        qk = linear(h, self.block_weight(p + "gdn_in.q4"))
        q = qk[:, :GDN_KEY]
        k = qk[:, GDN_KEY:]
        v = linear(h, self.block_weight(p + "gdn_in.q5"))
        qkv = torch.cat([q, k, v], dim=-1)
        a = linear(h, self.weight(p + "linear_attn.in_proj_a.weight"))
        b = linear(h, self.weight(p + "linear_attn.in_proj_b.weight"))
        qkv_c = causal_conv1d(qkv, self.weight(p + "linear_attn.conv1d.weight"), self.conv[gidx])
        g, beta = gdn_gating(a, b, self.weight(p + "linear_attn.A_log"), self.weight(p + "linear_attn.dt_bias"))
        qc = qkv_c[:, :GDN_KEY].reshape(-1, GDN_HK, GDN_DK)
        kc = qkv_c[:, GDN_KEY : 2 * GDN_KEY].reshape(-1, GDN_HK, GDN_DK)
        vc = qkv_c[:, 2 * GDN_KEY :].reshape(-1, GDN_HV, GDN_DV)
        qn = l2norm(qc)
        kn = l2norm(kc)
        o = self.gdn_recurrent(qn, kn, vc, g, beta, gidx)
        z = linear(h, self.weight(p + "linear_attn.in_proj_z.weight")).reshape(-1, GDN_HV, GDN_DV)
        on = rmsnorm(o, self.weight(p + "linear_attn.norm.weight"), unit_offset=False, z=z)
        out = linear(on.reshape(-1, GDN_VALUE), self.weight(p + "linear_attn.out_proj.weight"))
        return residual_add(x, out)

    def mlp_tail(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{layer}."
        h = rmsnorm(x, self.weight(p + "post_attention_layernorm.weight"), unit_offset=True)
        gate_up = linear(h, self.block_weight(p + "mlp.gateup"))
        gate, up = gate_up.split(I, dim=-1)
        a = silu_and_mul(gate, up)
        d = linear(a, self.weight(p + "mlp.down_proj.weight"))
        return residual_add(x, d)

    def block(self, layer: int, x: torch.Tensor, phase: str, positions: torch.Tensor) -> torch.Tensor:
        if is_full(layer):
            x = self.attn_mix(layer, x, phase, positions)
        else:
            x = self.gdn_mix(layer, x, phase)
        return self.mlp_tail(layer, x)

    def run_layers(
        self,
        x: torch.Tensor,
        phase: str,
        positions: torch.Tensor,
        dumps: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        for layer in range(L):
            x = self.block(layer, x, phase, positions)
            if dumps is not None:
                dumps[f"layer_{layer}"] = x.detach().float().cpu()
        return x

    def final_norm_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight("model.language_model.norm.weight"), unit_offset=True)

    def logits_from_hidden_last(self, hidden: torch.Tensor) -> torch.Tensor:
        xlast = hidden[-1:, :].to(torch.bfloat16)
        logits = torch.empty(V, device=self.device, dtype=torch.bfloat16)
        for row0, row1, weight in self.q5090.row_split_row_chunks(
            "lm_head.weight",
            self.device,
            output_dtype=torch.bfloat16,
        ):
            logits[row0:row1] = (xlast @ weight.to(torch.bfloat16).t())[0]
            del weight
        return logits

    def logits_last(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_hidden_last(self.final_norm_hidden(x))

    def mtp_forward(
        self,
        input_ids: Iterable[int],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        phase: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        ids = list(input_ids)
        if not ids:
            raise ValueError("MTP input_ids must not be empty")
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"unknown MTP phase {phase!r}")
        if hidden_states.shape != (len(ids), D):
            raise ValueError(f"MTP hidden_states shape {tuple(hidden_states.shape)} != {(len(ids), D)}")
        positions = positions.to(device=self.device, dtype=torch.int32)
        if positions.shape != (len(ids),):
            raise ValueError(f"MTP positions shape {tuple(positions.shape)} != {(len(ids),)}")

        p = "mtp.layers.0."
        emb = self.embed(ids)
        e = rmsnorm(emb, self.weight("mtp.pre_fc_norm_embedding.weight"), unit_offset=True)
        h = rmsnorm(hidden_states, self.weight("mtp.pre_fc_norm_hidden.weight"), unit_offset=True)
        x = linear(torch.cat([e, h], dim=-1), self.weight("mtp.fc.weight"))

        ah = rmsnorm(x, self.weight(p + "input_layernorm.weight"), unit_offset=True)
        attn_in = linear(ah, self.block_weight(p + "attn_in.w8"))
        q = attn_in[:, :Q_SIZE].reshape(-1, H_Q, DH)
        k = attn_in[:, Q_SIZE : Q_SIZE + KV_SIZE].reshape(-1, H_KV, DH)
        gate = attn_in[:, Q_SIZE + KV_SIZE : Q_SIZE + KV_SIZE + Q_SIZE].reshape(-1, H_Q, DH)
        v = attn_in[:, Q_SIZE + KV_SIZE + Q_SIZE :].reshape(-1, H_KV, DH)

        qn = rmsnorm(q, self.weight(p + "self_attn.q_norm.weight"), unit_offset=True)
        kn = rmsnorm(k, self.weight(p + "self_attn.k_norm.weight"), unit_offset=True)
        qn, kn = apply_rope(qn, kn, positions)
        a = self.mtp_gqa_attention(qn, kn, v, 0, phase)
        a = sigmoid_gate_mul(gate, a).reshape(-1, Q_SIZE)
        o = linear(a, self.weight(p + "self_attn.o_proj.weight"))
        x = residual_add(x, o)

        mh = rmsnorm(x, self.weight(p + "post_attention_layernorm.weight"), unit_offset=True)
        gate_up = linear(mh, self.block_weight(p + "mlp.gateup.w8"))
        gate_mlp, up = gate_up.split(I, dim=-1)
        d = linear(silu_and_mul(gate_mlp, up), self.weight(p + "mlp.down_proj.weight"))
        x = residual_add(x, d)

        mtp_hidden = rmsnorm(x, self.weight("mtp.norm.weight"), unit_offset=True)
        logits = self.logits_from_hidden_last(mtp_hidden)
        draft = int(torch.argmax(logits).item())
        return mtp_hidden, logits, draft

    def _target_decode_one(self, token: int, position: int) -> tuple[int, torch.Tensor]:
        x = self.embed([token])
        pos = torch.tensor([position], device=self.device, dtype=torch.int32)
        x = self.run_layers(x, "decode", pos)
        hidden = self.final_norm_hidden(x)
        next_token = int(torch.argmax(self.logits_from_hidden_last(hidden)).item())
        return next_token, hidden

    def _mtp_make_drafts(
        self,
        last_hidden: torch.Tensor,
        last_logits: torch.Tensor,
        last_position: int,
        draft_count: int,
        committed_kv_len: int,
    ) -> List[int]:
        if draft_count < 1 or draft_count > 5:
            raise ValueError("draft_count must be in [1,5]")
        self.truncate_mtp_kv(committed_kv_len)
        drafts = [int(torch.argmax(last_logits).item())]
        hidden = last_hidden[-1:, :]
        token = drafts[0]
        position = last_position + 1
        while len(drafts) < draft_count:
            hidden, logits, token = self.mtp_forward(
                [token],
                hidden,
                torch.tensor([position], device=self.device, dtype=torch.int32),
                "decode",
            )
            drafts.append(token)
            position += 1
        return drafts

    def forward_mtp_verified(
        self,
        prompt: Iterable[int],
        n_decode: int,
        *,
        draft_count: int,
        stop_token_ids: Optional[set[int]] = None,
    ) -> tuple[List[int], SpeculativeStats]:
        if draft_count < 1 or draft_count > 5:
            raise ValueError("draft_count must be in [1,5]")
        prompt_ids = list(prompt)
        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        if n_decode < 0:
            raise ValueError("n_decode must be nonnegative")
        self.reset_state()
        stats = SpeculativeStats()
        if n_decode == 0:
            return [], stats

        x = self.embed(prompt_ids)
        positions = torch.arange(len(prompt_ids), device=self.device, dtype=torch.int32)
        x = self.run_layers(x, "prefill", positions)
        target_hidden = self.final_norm_hidden(x)
        token = int(torch.argmax(self.logits_from_hidden_last(target_hidden)).item())
        out = [token]
        self.pos = len(prompt_ids)
        if len(out) >= n_decode or (stop_token_ids and token in stop_token_ids):
            return out, stats

        mtp_ids = prompt_ids[1:] + [token]
        mtp_hidden, mtp_logits, _ = self.mtp_forward(mtp_ids, target_hidden, positions, "prefill")
        last_mtp_hidden = mtp_hidden[-1:, :]
        last_mtp_logits = mtp_logits
        last_mtp_position = len(prompt_ids) - 1
        committed_mtp_len = self.mtp_kv_len()
        drafts = self._mtp_make_drafts(
            last_mtp_hidden,
            last_mtp_logits,
            last_mtp_position,
            draft_count,
            committed_mtp_len,
        )
        draft_index = 0

        while len(out) < n_decode:
            if draft_index >= len(drafts):
                drafts = self._mtp_make_drafts(
                    last_mtp_hidden,
                    last_mtp_logits,
                    last_mtp_position,
                    draft_count,
                    committed_mtp_len,
                )
                draft_index = 0
            candidate = drafts[draft_index]

            target_position = self.pos
            target_token, target_hidden = self._target_decode_one(token, target_position)
            self.pos += 1
            stats.draft_tokens += 1
            accepted = target_token == candidate
            if accepted:
                stats.accepted_tokens += 1
                token = candidate
                draft_index += 1
            else:
                token = target_token
                drafts = []
                draft_index = 0

            out.append(token)
            if stop_token_ids and token in stop_token_ids:
                break

            self.truncate_mtp_kv(committed_mtp_len)
            last_mtp_hidden, last_mtp_logits, _ = self.mtp_forward(
                [token],
                target_hidden,
                torch.tensor([target_position], device=self.device, dtype=torch.int32),
                "decode",
            )
            last_mtp_hidden = last_mtp_hidden[-1:, :]
            last_mtp_position = target_position
            committed_mtp_len = self.mtp_kv_len()

            if not accepted or draft_index >= len(drafts):
                drafts = self._mtp_make_drafts(
                    last_mtp_hidden,
                    last_mtp_logits,
                    last_mtp_position,
                    draft_count,
                    committed_mtp_len,
                )
                draft_index = 0

        return out, stats

    def forward(
        self,
        prompt: Iterable[int],
        n_decode: int,
        *,
        dumps: Optional[Dict[str, torch.Tensor]] = None,
        stop_token_ids: Optional[set[int]] = None,
    ) -> List[int]:
        prompt_ids = list(prompt)
        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        if n_decode < 0:
            raise ValueError("n_decode must be nonnegative")
        self.reset_state()
        if n_decode == 0:
            return []

        x = self.embed(prompt_ids)
        if dumps is not None:
            dumps["embed"] = x.detach().float().cpu()
        pos = torch.arange(len(prompt_ids), device=self.device, dtype=torch.int32)
        x = self.run_layers(x, "prefill", pos, dumps)
        lg = self.logits_last(x)
        if dumps is not None:
            xf = rmsnorm(x, self.weight("model.language_model.norm.weight"), unit_offset=True)
            dumps["final_norm"] = xf.detach().float().cpu()
            dumps["logits_last"] = lg.detach().float().cpu()
        token = int(torch.argmax(lg).item())
        out = [token]
        self.pos = len(prompt_ids)
        if stop_token_ids and token in stop_token_ids:
            return out

        while len(out) < n_decode:
            x = self.embed([token])
            pos = torch.tensor([self.pos], device=self.device, dtype=torch.int32)
            x = self.run_layers(x, "decode", pos, dumps)
            token = int(torch.argmax(self.logits_last(x)).item())
            out.append(token)
            self.pos += 1
            if stop_token_ids and token in stop_token_ids:
                break
        return out


def parse_token_ids(text: str) -> List[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def parse_stop_token_ids(text: str) -> set[int]:
    return {int(part) for part in text.replace(",", " ").split() if part.strip()}


def load_tokenizer(model_dir: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for chat-template prompts; run with the vllm-bench environment"
        ) from exc
    return AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def load_messages(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a non-empty JSON message list")
    messages = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{i}]: expected object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{path}[{i}]: expected string role/content")
        messages.append({"role": role, "content": content})
    return messages


def chat_prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[List[int], str]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not ids:
        raise ValueError("chat template produced an empty prompt")
    return [int(x) for x in ids], str(rendered)


def main() -> None:
    ap = argparse.ArgumentParser(description="q5090 PyTorch oracle")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model/tokenizer directory")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="user prompt text rendered through the Qwen chat template")
    ap.add_argument("--messages-json", default=None, help="JSON chat messages rendered through the Qwen chat template")
    ap.add_argument("--token-ids", default=None, help="comma or space separated token ids; bypasses tokenizer/chat template")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--mtp-draft-tokens", type=int, default=0, help="0 disables MTP; 1..5 enables verified greedy MTP")
    ap.add_argument("--stop-token-ids", default=DEFAULT_STOP_TOKEN_IDS)
    ap.add_argument("--show-rendered-prompt", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None, help="write v3 structural dump JSON")
    ap.add_argument(
        "--resident",
        default="auto",
        choices=["auto", "gpu", "stream"],
        help="keep quantized payloads resident on device (auto/gpu) or stream from mmap",
    )
    args = ap.parse_args()
    if args.mtp_draft_tokens < 0 or args.mtp_draft_tokens > 5:
        raise SystemExit("--mtp-draft-tokens must be 0 or an integer in [1,5]")

    model = RefModel(
        args.weights,
        device=args.device,
        resident=args.resident,
    )
    if args.dump:
        model.q5090.dump(args.dump)
    tokenizer = None
    rendered_prompt = None
    if args.token_ids is not None:
        prompt_ids = parse_token_ids(args.token_ids)
    else:
        tokenizer = load_tokenizer(args.model)
        messages = load_messages(args.messages_json) if args.messages_json else [
            {"role": "user", "content": args.prompt}
        ]
        prompt_ids, rendered_prompt = chat_prompt_ids(tokenizer, messages)
    stop_token_ids = parse_stop_token_ids(args.stop_token_ids)
    with torch.inference_mode():
        if args.mtp_draft_tokens:
            tokens, spec_stats = model.forward_mtp_verified(
                prompt_ids,
                args.decode,
                draft_count=args.mtp_draft_tokens,
                stop_token_ids=stop_token_ids,
            )
        else:
            tokens = model.forward(prompt_ids, args.decode, stop_token_ids=stop_token_ids)
            spec_stats = None
    print(f"PROMPT_TOKENS: {len(prompt_ids)}")
    if rendered_prompt is not None and args.show_rendered_prompt:
        print(f"RENDERED_PROMPT: {rendered_prompt!r}")
    print(f"GENERATED_TOKEN_IDS: {tokens}")
    if spec_stats is not None:
        print(f"MTP_DRAFT_TOKENS: {spec_stats.draft_tokens}")
        print(f"MTP_ACCEPTED_TOKENS: {spec_stats.accepted_tokens}")
        print(f"MTP_ACCEPTANCE_RATE: {spec_stats.acceptance_rate:.6f}")
    if tokenizer is not None:
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"GENERATED_TEXT: {text!r}")
    else:
        print(" ".join(str(t) for t in tokens))


if __name__ == "__main__":
    main()
