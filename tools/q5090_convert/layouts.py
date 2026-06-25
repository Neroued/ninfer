"""On-disk payload encoders/decoders for each layout (Torch / GPU).

Ties quantize + packing into the exact byte layouts of
../../docs/q5090_packed_file_format_v1.md section 7. Everything runs on the chosen
device (CUDA) so the host only sees the final byte blob.

Encoders return (payload_bytes, logical_shape, padded_shape, group_size, scale_dtype).
Decoders return a float32 torch.Tensor on `device` (used by verify.py).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import qtypes as qt
from .packing import bytes_per_group, pack_lowbit_torch, unpack_lowbit_torch
from .quantize import quantize_core

EncodeResult = Tuple[bytes, List[int], List[int], int, int]


def _pad_2d(w: torch.Tensor, n_mult: int, k_mult: int):
    n, k = w.shape
    np_ = (n + n_mult - 1) // n_mult * n_mult
    kp = (k + k_mult - 1) // k_mult * k_mult
    if np_ != n or kp != k:
        w = F.pad(w, (0, kp - k, 0, np_ - n), value=0.0)
    return w, n, k, np_, kp


def _to_bytes(t: torch.Tensor) -> bytes:
    return t.reshape(-1).contiguous().cpu().numpy().tobytes()


def _from_bytes(payload: bytes, device, dtype=np.uint8) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(payload, dtype).copy()).to(device)


# ----------------------------- encoders -----------------------------

def encode_tile_lowbit(w: torch.Tensor, qtype: int, device) -> EncodeResult:
    spec = qt.QUANT_SPECS[qtype]
    gs, bits = spec.group_size, spec.bits
    bpr = bytes_per_group(gs, bits)
    wf = w.to(device=device, dtype=torch.float32)
    wf, n, k, np_, kp = _pad_2d(wf, 64, gs)
    scale16, codes = quantize_core(wf, gs, spec.qmax, spec.qmin)
    del wf
    kg = kp // gs
    nt = np_ // 64
    packed = pack_lowbit_torch(codes.reshape(np_ * kg, gs), bits).reshape(np_, kg, bpr)
    del codes
    sb = scale16.reshape(nt, 64, kg).permute(0, 2, 1).contiguous().view(torch.uint8)   # [nt,kg,128]
    data = packed.reshape(nt, 64, kg, bpr).permute(0, 2, 1, 3).contiguous().reshape(nt, kg, 64 * bpr)
    tile = torch.cat([sb, data], dim=2)
    return _to_bytes(tile), [n, k], [np_, kp], gs, qt.SCALE_FP16


def encode_tile_w8(w: torch.Tensor, device) -> EncodeResult:
    spec = qt.QUANT_SPECS[qt.QT_W8G128]
    gs = spec.group_size
    wf = w.to(device=device, dtype=torch.float32)
    wf, n, k, np_, kp = _pad_2d(wf, 64, gs)
    scale16, codes = quantize_core(wf, gs, spec.qmax, spec.qmin)
    del wf
    kg = kp // gs
    nt = np_ // 64
    sb = scale16.reshape(nt, 64, kg).permute(0, 2, 1).contiguous().view(torch.uint8)
    data = codes.reshape(nt, 64, kg, gs).permute(0, 2, 1, 3).contiguous().view(torch.uint8).reshape(nt, kg, 64 * gs)
    tile = torch.cat([sb, data], dim=2)
    return _to_bytes(tile), [n, k], [np_, kp], gs, qt.SCALE_FP16


def encode_row_grouped(w: torch.Tensor, qtype: int, device) -> EncodeResult:
    spec = qt.QUANT_SPECS[qtype]
    gs, bits = spec.group_size, spec.bits
    bpr = bytes_per_group(gs, bits)
    wf = w.to(device=device, dtype=torch.float32)
    wf, n, k, np_, kp = _pad_2d(wf, 1, gs)
    scale16, codes = quantize_core(wf, gs, spec.qmax, spec.qmin)
    del wf
    kg = kp // gs
    packed = pack_lowbit_torch(codes.reshape(n * kg, gs), bits).reshape(n, kg, bpr)
    sb = scale16.contiguous().view(torch.uint8).reshape(n, kg, 2)
    row = torch.cat([sb, packed], dim=2)
    return _to_bytes(row), [n, k], [np_, kp], gs, qt.SCALE_FP16


def encode_contiguous(w: torch.Tensor, qtype: int) -> EncodeResult:
    shape = list(w.shape)
    w = w.contiguous()
    if qtype == qt.QT_BF16:
        raw = w.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().astype("<i2").tobytes()
    elif qtype == qt.QT_FP32:
        raw = w.to(torch.float32).contiguous().cpu().numpy().astype("<f4").tobytes()
    else:
        raise ValueError(f"contiguous qtype must be BF16/FP32, got {qtype}")
    return raw, shape, list(shape), 0, qt.SCALE_NONE


def encode_tensor(w: torch.Tensor, qtype: int, layout: int, device) -> EncodeResult:
    if layout == qt.LAYOUT_CONTIGUOUS:
        return encode_contiguous(w, qtype)
    if layout == qt.LAYOUT_TILE_N64_K64:
        return encode_tile_lowbit(w, qtype, device)
    if layout == qt.LAYOUT_TILE_N64_K128:
        return encode_tile_w8(w, device)
    if layout == qt.LAYOUT_ROW_GROUPED_G64:
        return encode_row_grouped(w, qtype, device)
    raise ValueError(f"unknown layout {layout}")


# ----------------------------- decoders (verify) -----------------------------

def decode_tile_lowbit(payload, padded, logical, qtype, device) -> torch.Tensor:
    spec = qt.QUANT_SPECS[qtype]
    gs, bits = spec.group_size, spec.bits
    bpr = bytes_per_group(gs, bits)
    np_, kp = padded
    n, k = logical
    nt, kg = np_ // 64, kp // gs
    tilew = 64 * 2 + 64 * bpr
    arr = _from_bytes(payload, device).reshape(nt, kg, tilew)
    scale = arr[:, :, : 64 * 2].contiguous().view(torch.float16).to(torch.float32)        # [nt,kg,64]
    data = arr[:, :, 64 * 2 :].contiguous().reshape(nt * kg * 64, bpr)
    codes = unpack_lowbit_torch(data, bits, gs).reshape(nt, kg, 64, gs).to(torch.float32)
    deq = (codes * scale.unsqueeze(-1)).permute(0, 2, 1, 3).reshape(np_, kp)
    return deq[:n, :k]


def decode_tile_w8(payload, padded, logical, device) -> torch.Tensor:
    gs = 128
    np_, kp = padded
    n, k = logical
    nt, kg = np_ // 64, kp // gs
    tilew = 64 * 2 + 64 * gs
    arr = _from_bytes(payload, device).reshape(nt, kg, tilew)
    scale = arr[:, :, : 64 * 2].contiguous().view(torch.float16).to(torch.float32)
    data = arr[:, :, 64 * 2 :].contiguous().reshape(nt, kg, 64, gs).view(torch.int8).to(torch.float32)
    deq = (data * scale.unsqueeze(-1)).permute(0, 2, 1, 3).reshape(np_, kp)
    return deq[:n, :k]


def decode_row_grouped(payload, padded, logical, qtype, device) -> torch.Tensor:
    spec = qt.QUANT_SPECS[qtype]
    gs, bits = spec.group_size, spec.bits
    bpr = bytes_per_group(gs, bits)
    n, kp = padded
    _, k = logical
    kg = kp // gs
    roww = 2 + bpr
    arr = _from_bytes(payload, device).reshape(n, kg, roww)
    scale = arr[:, :, :2].contiguous().view(torch.float16).reshape(n, kg).to(torch.float32)
    data = arr[:, :, 2:].contiguous().reshape(n * kg, bpr)
    codes = unpack_lowbit_torch(data, bits, gs).reshape(n, kg, gs).to(torch.float32)
    deq = (codes * scale.unsqueeze(-1)).reshape(n, kp)
    return deq[:, :k]


def decode_contiguous(payload, shape, qtype, device) -> torch.Tensor:
    if qtype == qt.QT_BF16:
        u16 = _from_bytes(payload, device, np.uint16)
        f32 = (u16.to(torch.int32) << 16).view(torch.float32)
    elif qtype == qt.QT_FP32:
        f32 = _from_bytes(payload, device, "<f4")
    else:
        raise ValueError(f"contiguous qtype must be BF16/FP32, got {qtype}")
    return f32.reshape(shape)


def decode_tensor(payload, qtype, layout, logical, padded, device="cpu") -> torch.Tensor:
    if layout == qt.LAYOUT_CONTIGUOUS:
        return decode_contiguous(payload, logical, qtype, device)
    if layout == qt.LAYOUT_TILE_N64_K64:
        return decode_tile_lowbit(payload, padded, logical, qtype, device)
    if layout == qt.LAYOUT_TILE_N64_K128:
        return decode_tile_w8(payload, padded, logical, device)
    if layout == qt.LAYOUT_ROW_GROUPED_G64:
        return decode_row_grouped(payload, padded, logical, qtype, device)
    raise ValueError(f"unknown layout {layout}")
