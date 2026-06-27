"""Low-bit code packing/unpacking (LSB-first two's complement).

Layout: within each group of `gs` signed codes, each code is stored as a `bits`-bit
two's-complement value; codes are concatenated LSB-first and packed into bytes with
numpy bitorder="little". See ../../docs/q5090_packed_file_format_v1.md section 7.1.

Two implementations:
  * numpy `*_groups`     -- reference, used by the unit tests
  * torch `*_torch`      -- GPU fast path used by the converter / verifier
They are bit-for-bit equivalent (cross-checked in tests).
"""

from __future__ import annotations

import numpy as np
import torch

_CHUNK_BITS_BUDGET = 256 * 1024 * 1024       # numpy temp cap
_TORCH_CHUNK_BYTES = 4 * 1024 * 1024 * 1024  # gpu temp cap for the expanded-bit array
_UNPACK_G64_INDEX_CACHE = {}


def bytes_per_group(gs: int, bits: int) -> int:
    assert (gs * bits) % 8 == 0, "group_size * bits must be byte-aligned"
    return gs * bits // 8


# ----------------------------- numpy reference -----------------------------

def pack_lowbit_groups(codes_int8: np.ndarray, bits: int) -> np.ndarray:
    """[num_groups, gs] signed int8 codes -> [num_groups, bytes_per_group] uint8."""
    assert codes_int8.ndim == 2
    num_groups, gs = codes_int8.shape
    bpr = bytes_per_group(gs, bits)
    mask = (1 << bits) - 1
    shifts = np.arange(bits, dtype=np.uint16)
    out = np.empty((num_groups, bpr), dtype=np.uint8)
    chunk = max(1, _CHUNK_BITS_BUDGET // (gs * bits))
    for g0 in range(0, num_groups, chunk):
        g1 = min(num_groups, g0 + chunk)
        u = codes_int8[g0:g1].astype(np.uint16) & mask
        bits_arr = ((u[:, :, None] >> shifts[None, None, :]) & 1).astype(np.uint8)
        bits_arr = bits_arr.reshape(g1 - g0, gs * bits)
        out[g0:g1] = np.packbits(bits_arr, axis=1, bitorder="little")
    return out


def unpack_lowbit_groups(packed: np.ndarray, bits: int, gs: int) -> np.ndarray:
    """[num_groups, bytes_per_group] uint8 -> [num_groups, gs] sign-extended int8."""
    assert packed.ndim == 2
    num_groups = packed.shape[0]
    shifts = np.arange(bits, dtype=np.uint16)
    sign_bit = 1 << (bits - 1)
    span = 1 << bits
    out = np.empty((num_groups, gs), dtype=np.int8)
    chunk = max(1, _CHUNK_BITS_BUDGET // (gs * bits))
    for g0 in range(0, num_groups, chunk):
        g1 = min(num_groups, g0 + chunk)
        bits_arr = np.unpackbits(packed[g0:g1], axis=1, bitorder="little")
        bits_arr = bits_arr[:, : gs * bits].reshape(g1 - g0, gs, bits).astype(np.uint16)
        u = (bits_arr << shifts[None, None, :]).sum(axis=2)
        signed = np.where(u & sign_bit, u.astype(np.int32) - span, u.astype(np.int32))
        out[g0:g1] = signed.astype(np.int8)
    return out


def pack_w8_groups(codes_int8: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(codes_int8, dtype=np.int8).view(np.uint8)


def unpack_w8_groups(packed_uint8: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(packed_uint8, dtype=np.uint8).view(np.int8)


# ----------------------------- torch GPU path -----------------------------

def _sign_extend_torch(u: torch.Tensor, bits: int) -> torch.Tensor:
    sign_bit = 1 << (bits - 1)
    span = 1 << bits
    return torch.where((u & sign_bit) != 0, u - span, u).to(torch.int8)


def _unpack_g64_indices(device: torch.device, bits: int, bpr: int):
    key = (device.type, device.index if device.index is not None else -1, bits, bpr)
    cached = _UNPACK_G64_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    lanes = torch.arange(64, device=device, dtype=torch.int64)
    bitpos = lanes * bits
    byte_idx = torch.div(bitpos, 8, rounding_mode="floor")
    hi_idx = torch.clamp(byte_idx + 1, max=bpr - 1)
    shift = bitpos - byte_idx * 8
    cached = (byte_idx, hi_idx, shift)
    _UNPACK_G64_INDEX_CACHE[key] = cached
    return cached


def _unpack_g64_fast_torch(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Fast q5090 G64 unpack without materializing every bit separately."""
    g, bpr = packed.shape
    dev = packed.device

    if bits == 4:
        nibbles = packed.to(torch.int16)
        out = torch.empty((g, 64), dtype=torch.int8, device=dev)
        out[:, 0::2] = _sign_extend_torch(nibbles & 0x0f, 4)
        out[:, 1::2] = _sign_extend_torch((nibbles >> 4) & 0x0f, 4)
        return out

    byte_idx, hi_idx, shift = _unpack_g64_indices(dev, bits, bpr)
    lo = packed.index_select(1, byte_idx).to(torch.int16)
    hi = packed.index_select(1, hi_idx).to(torch.int16)
    u = ((lo >> shift) | (hi << (8 - shift))) & ((1 << bits) - 1)
    return _sign_extend_torch(u, bits)


def pack_lowbit_torch(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[G, gs] int codes (device) -> [G, bytes_per_group] uint8 (device)."""
    g, gs = codes.shape
    bpr = bytes_per_group(gs, bits)
    dev = codes.device
    mask = (1 << bits) - 1
    shifts = torch.arange(bits, device=dev, dtype=torch.int32)
    bweight = (1 << torch.arange(8, device=dev, dtype=torch.int32))
    out = torch.empty((g, bpr), dtype=torch.uint8, device=dev)
    per = max(1, _TORCH_CHUNK_BYTES // max(1, gs * bits * 4))
    for a in range(0, g, per):
        b = min(g, a + per)
        u = codes[a:b].to(torch.int32) & mask                       # [n, gs]
        bit = ((u.unsqueeze(-1) >> shifts) & 1).reshape(b - a, bpr, 8)
        out[a:b] = (bit * bweight).sum(-1).to(torch.uint8)
    return out


def unpack_lowbit_torch(packed: torch.Tensor, bits: int, gs: int) -> torch.Tensor:
    """[G, bytes_per_group] uint8 (device) -> [G, gs] sign-extended int8 (device)."""
    g, bpr = packed.shape
    if gs == 64 and bits in (4, 5, 6):
        return _unpack_g64_fast_torch(packed, bits)

    dev = packed.device
    shifts8 = torch.arange(8, device=dev, dtype=torch.int32)
    wbits = (1 << torch.arange(bits, device=dev, dtype=torch.int32))
    sign_bit = 1 << (bits - 1)
    span = 1 << bits
    out = torch.empty((g, gs), dtype=torch.int8, device=dev)
    per = max(1, _TORCH_CHUNK_BYTES // max(1, bpr * 8 * 4))
    for a in range(0, g, per):
        b = min(g, a + per)
        bit = (packed[a:b].to(torch.int32).unsqueeze(-1) >> shifts8) & 1   # [n,bpr,8]
        bit = bit.reshape(b - a, bpr * 8)[:, : gs * bits].reshape(b - a, gs, bits)
        u = (bit * wbits).sum(-1)
        out[a:b] = torch.where((u & sign_bit) != 0, u - span, u).to(torch.int8)
    return out
