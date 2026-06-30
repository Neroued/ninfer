"""On-disk payload encoders/decoders for q5090 v3 layouts.

Encoders return:
  (payload_bytes, logical_shape, padded_shape, group_size, scale_dtype,
   nibble_plane_bytes, high_plane_bytes, scale_plane_bytes)

Decoders return a float32 torch.Tensor on `device` unless explicitly decoding the
ROW_SPLIT quantized planes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import qtypes as qt
from .packing import (
    assemble_row_split_payload,
    row_split_plane_sizes,
    split_row_split_payload,
    pack_plane_split_torch,
    unpack_plane_split_torch,
)
from .quantize import quantize_core

EncodeResult = Tuple[bytes, List[int], List[int], int, int, int, int, int]


def _pad_row_split_2d(w: torch.Tensor, k_mult: int):
    n, k = w.shape
    kp = (k + k_mult - 1) // k_mult * k_mult
    if kp != k:
        w = F.pad(w, (0, kp - k, 0, 0), value=0.0)
    return w, n, k, kp


def _to_bytes(t: torch.Tensor) -> bytes:
    return t.reshape(-1).contiguous().cpu().numpy().tobytes()


def _from_bytes(payload, device, dtype=np.uint8) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        if dtype is not np.uint8:
            raise ValueError("resident tensor payloads must be decoded as uint8")
        return payload
    return torch.from_numpy(np.frombuffer(payload, dtype).copy()).to(device)


# ----------------------------- encoders -----------------------------

def encode_row_split(w: torch.Tensor, qtype: int, device) -> EncodeResult:
    if qtype not in qt.QUANT_SPECS:
        raise ValueError(f"ROW_SPLIT qtype must be quantized, got {qtype}")

    spec = qt.QUANT_SPECS[qtype]
    gs, bits = spec.group_size, spec.bits
    nib = spec.nibble_bytes_per_group
    high = spec.high_bytes_per_group
    wf = w.to(device=device, dtype=torch.float32)
    if wf.dim() != 2:
        raise ValueError(f"ROW_SPLIT needs 2D [N,K], got {tuple(wf.shape)}")
    wf, n, k, kp = _pad_row_split_2d(wf, 128)
    scale16, codes = quantize_core(wf, gs, spec.qmax, spec.qmin)
    del wf

    groups = kp // gs
    if qtype == qt.QT_W8G128:
        nibble_plane = codes.contiguous().view(torch.uint8).reshape(n, groups, nib)
        high_plane = torch.empty((n, groups, 0), dtype=torch.uint8, device=device)
    else:
        nibble_flat, high_flat = pack_plane_split_torch(codes.reshape(n * groups, gs), bits)
        nibble_plane = nibble_flat.reshape(n, groups, nib)
        high_plane = high_flat.reshape(n, groups, high)
    del codes

    payload, nibble_plane_bytes, high_plane_bytes, scale_plane_bytes = assemble_row_split_payload(
        nibble_plane, high_plane, scale16
    )
    exp_nibble, _, exp_high, _, exp_scale, _ = row_split_plane_sizes(n, groups, nib, high)
    assert nibble_plane_bytes == exp_nibble
    assert high_plane_bytes == exp_high
    assert scale_plane_bytes == exp_scale
    return (
        _to_bytes(payload),
        [n, k],
        [n, kp],
        gs,
        qt.SCALE_FP16,
        nibble_plane_bytes,
        high_plane_bytes,
        scale_plane_bytes,
    )


def encode_contiguous(w: torch.Tensor, qtype: int) -> EncodeResult:
    shape = list(w.shape)
    w = w.contiguous()
    if qtype == qt.QT_BF16:
        raw = w.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().astype("<i2").tobytes()
    elif qtype == qt.QT_FP32:
        raw = w.to(torch.float32).contiguous().cpu().numpy().astype("<f4").tobytes()
    else:
        raise ValueError(f"CONTIGUOUS qtype must be BF16/FP32, got {qtype}")
    return raw, shape, list(shape), 0, qt.SCALE_NONE, len(raw), 0, 0


def encode_tensor(w: torch.Tensor, qtype: int, layout: int, device) -> EncodeResult:
    if layout == qt.LAYOUT_ROW_SPLIT:
        return encode_row_split(w, qtype, device)
    if layout == qt.LAYOUT_CONTIGUOUS:
        return encode_contiguous(w, qtype)
    raise ValueError(f"unknown layout {layout}")


# ----------------------------- decoders -----------------------------

def decode_row_split_quantized(payload, padded, qtype, device="cpu"):
    spec = qt.QUANT_SPECS[qtype]
    gs, bits = spec.group_size, spec.bits
    nib = spec.nibble_bytes_per_group
    high = spec.high_bytes_per_group
    n, kp = padded
    groups = kp // gs
    nibble_bytes, _, high_bytes, _, scale_bytes, _ = row_split_plane_sizes(n, groups, nib, high)
    arr = _from_bytes(payload, device)
    nibble_plane, high_plane, scale_plane = split_row_split_payload(
        arr, nibble_bytes, high_bytes, scale_bytes
    )

    if qtype == qt.QT_W8G128:
        codes = nibble_plane.reshape(n, groups, gs).view(torch.int8)
    else:
        nibble_packed = nibble_plane.reshape(n * groups, nib)
        high_packed = high_plane.reshape(n * groups, high)
        codes = unpack_plane_split_torch(nibble_packed, high_packed, bits, gs).reshape(n, groups, gs)
    scale16 = scale_plane.reshape(n * groups * 2).view(torch.float16).reshape(n, groups)
    return scale16, codes


def decode_row_split(
    payload,
    padded,
    logical,
    qtype,
    device,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    scale16, codes = decode_row_split_quantized(payload, padded, qtype, device)
    n, k = logical
    _, kp = padded
    if output_dtype == torch.float32:
        deq = (codes.to(torch.float32) * scale16.to(torch.float32).unsqueeze(-1)).reshape(n, kp)
        return deq[:, :k]
    if output_dtype == torch.bfloat16:
        deq = torch.empty(codes.shape, device=codes.device, dtype=torch.bfloat16)
        torch.mul(
            codes.to(torch.float32),
            scale16.to(torch.float32).unsqueeze(-1),
            out=deq,
        )
        return deq.reshape(n, kp)[:, :k]
    deq = (codes.to(torch.float32) * scale16.to(torch.float32).unsqueeze(-1)).reshape(n, kp)
    return deq[:, :k].to(output_dtype)


def decode_contiguous(payload, shape, qtype, device) -> torch.Tensor:
    if qtype == qt.QT_BF16:
        u16 = _from_bytes(payload, device, np.uint16)
        f32 = (u16.to(torch.int32) << 16).view(torch.float32)
    elif qtype == qt.QT_FP32:
        f32 = _from_bytes(payload, device, "<f4")
    else:
        raise ValueError(f"CONTIGUOUS qtype must be BF16/FP32, got {qtype}")
    return f32.reshape(shape)


def decode_tensor(
    payload,
    qtype,
    layout,
    logical,
    padded,
    device="cpu",
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if layout == qt.LAYOUT_ROW_SPLIT:
        return decode_row_split(
            payload,
            padded,
            logical,
            qtype,
            device,
            output_dtype=output_dtype,
        )
    if layout == qt.LAYOUT_CONTIGUOUS:
        out = decode_contiguous(payload, logical, qtype, device)
        if qtype == qt.QT_BF16:
            return out.to(output_dtype)
        return out
    raise ValueError(f"unknown layout {layout}")
