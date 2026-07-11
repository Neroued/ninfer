"""Low-bit code packing/unpacking and ROW_SPLIT plane assembly.

Layout: within each group of `gs` signed codes, each code is stored as a `bits`-bit
two's-complement value. Q4/Q5/Q6 store the low nibble in a Q4-compatible base
plane, and Q5/Q6 store the remaining high bit(s) in a separate high plane. W8
stores one signed int8 byte per code in the base plane. See
../../docs/q5090_packed_file_format_v4.md section 9.1.

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
_UNPACK_HIGH_G64_INDEX_CACHE = {}
ROW_SPLIT_PLANE_ALIGN = 256


def bytes_per_group(gs: int, bits: int) -> int:
    assert (gs * bits) % 8 == 0, "group_size * bits must be byte-aligned"
    return gs * bits // 8


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def nibble_bytes_per_group(gs: int, bits: int) -> int:
    if bits == 8:
        return gs
    if bits not in (4, 5, 6):
        raise ValueError(f"unsupported low-bit width for nibble plane: {bits}")
    if gs % 2 != 0:
        raise ValueError(f"group_size must be even for nibble packing, got {gs}")
    return gs // 2


def high_bytes_per_group(gs: int, bits: int) -> int:
    if bits in (4, 8):
        return 0
    if bits not in (5, 6):
        raise ValueError(f"unsupported low-bit width for high plane: {bits}")
    high_bits = bits - 4
    assert (gs * high_bits) % 8 == 0, "high plane must be byte-aligned"
    return gs * high_bits // 8


def row_split_plane_sizes(
    n: int,
    groups: int,
    nibble_bytes_per_group_: int,
    high_bytes_per_group_: int,
):
    """Return v3 ROW_SPLIT plane sizes and relative offsets.

    Tuple layout:
      (nibble bytes, high offset, high bytes, scale offset, scale bytes, payload bytes)
    """
    nibble_plane_bytes = n * groups * nibble_bytes_per_group_
    high_plane_bytes = n * groups * high_bytes_per_group_
    high_plane_off = align_up(nibble_plane_bytes, ROW_SPLIT_PLANE_ALIGN)
    scale_plane_off = high_plane_off + align_up(high_plane_bytes, ROW_SPLIT_PLANE_ALIGN)
    scale_plane_bytes = n * groups * 2
    payload_bytes = scale_plane_off + scale_plane_bytes
    return (
        nibble_plane_bytes,
        high_plane_off,
        high_plane_bytes,
        scale_plane_off,
        scale_plane_bytes,
        payload_bytes,
    )


def assemble_row_split_payload(
    nibble_plane: torch.Tensor,
    high_plane: torch.Tensor,
    scale_plane: torch.Tensor,
):
    """Flatten nibble/high/fp16 scale planes into the v3 ROW_SPLIT payload tensor."""
    nibble = nibble_plane.reshape(-1).contiguous().view(torch.uint8)
    high = high_plane.reshape(-1).contiguous().view(torch.uint8)
    scale = scale_plane.reshape(-1).contiguous().view(torch.uint8)
    nibble_pad_bytes = align_up(nibble.numel(), ROW_SPLIT_PLANE_ALIGN) - nibble.numel()
    high_pad_bytes = align_up(high.numel(), ROW_SPLIT_PLANE_ALIGN) - high.numel() if high.numel() else 0
    parts = [nibble]
    if nibble_pad_bytes:
        parts.append(torch.zeros((nibble_pad_bytes,), dtype=torch.uint8, device=nibble.device))
    if high.numel():
        parts.append(high)
        if high_pad_bytes:
            parts.append(torch.zeros((high_pad_bytes,), dtype=torch.uint8, device=nibble.device))
    parts.append(scale)
    return torch.cat(parts, dim=0), nibble.numel(), high.numel(), scale.numel()


def split_row_split_payload(
    payload: torch.Tensor,
    nibble_plane_bytes: int,
    high_plane_bytes: int,
    scale_plane_bytes: int,
):
    """Return flattened nibble, high, and scale byte views from a ROW_SPLIT payload tensor."""
    high_plane_off = align_up(nibble_plane_bytes, ROW_SPLIT_PLANE_ALIGN)
    scale_plane_off = high_plane_off + align_up(high_plane_bytes, ROW_SPLIT_PLANE_ALIGN)
    scale_end = scale_plane_off + scale_plane_bytes
    if payload.numel() < scale_end:
        raise ValueError(
            f"ROW_SPLIT payload too short: need {scale_end} bytes, got {payload.numel()}"
        )
    return (
        payload[:nibble_plane_bytes],
        payload[high_plane_off: high_plane_off + high_plane_bytes],
        payload[scale_plane_off:scale_end],
    )


# ----------------------------- numpy reference -----------------------------

def pack_low_nibble_groups(codes_int8: np.ndarray) -> np.ndarray:
    """[num_groups, gs] signed int8 codes -> low-nibble plane bytes."""
    assert codes_int8.ndim == 2
    num_groups, gs = codes_int8.shape
    if gs % 2 != 0:
        raise ValueError(f"group_size must be even for nibble packing, got {gs}")
    out = np.empty((num_groups, gs // 2), dtype=np.uint8)
    chunk = max(1, _CHUNK_BITS_BUDGET // max(1, gs * 4))
    for g0 in range(0, num_groups, chunk):
        g1 = min(num_groups, g0 + chunk)
        u = codes_int8[g0:g1].astype(np.uint16) & 0x0F
        out[g0:g1] = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)
    return out


def pack_highbit_groups(codes_int8: np.ndarray, bits: int) -> np.ndarray:
    """[num_groups, gs] signed int8 codes -> v3 high-bit plane bytes."""
    assert codes_int8.ndim == 2
    if bits == 4:
        return np.empty((codes_int8.shape[0], 0), dtype=np.uint8)
    if bits not in (5, 6):
        raise ValueError(f"high-bit plane supports Q5/Q6, got bits={bits}")
    num_groups, gs = codes_int8.shape
    high_bits = bits - 4
    bpr = high_bytes_per_group(gs, bits)
    mask = (1 << bits) - 1
    shifts = np.arange(high_bits, dtype=np.uint16)
    out = np.empty((num_groups, bpr), dtype=np.uint8)
    chunk = max(1, _CHUNK_BITS_BUDGET // max(1, gs * high_bits))
    for g0 in range(0, num_groups, chunk):
        g1 = min(num_groups, g0 + chunk)
        high = ((codes_int8[g0:g1].astype(np.uint16) & mask) >> 4).astype(np.uint16)
        bits_arr = ((high[:, :, None] >> shifts[None, None, :]) & 1).astype(np.uint8)
        bits_arr = bits_arr.reshape(g1 - g0, gs * high_bits)
        out[g0:g1] = np.packbits(bits_arr, axis=1, bitorder="little")
    return out


def pack_plane_split_groups(codes_int8: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """[num_groups, gs] signed int8 codes -> (nibble plane, high plane)."""
    if bits == 8:
        return pack_w8_groups(codes_int8), np.empty((codes_int8.shape[0], 0), dtype=np.uint8)
    return pack_low_nibble_groups(codes_int8), pack_highbit_groups(codes_int8, bits)


def _unpack_low_nibbles_np(nibble_packed: np.ndarray, gs: int) -> np.ndarray:
    assert nibble_packed.ndim == 2
    if nibble_packed.shape[1] != gs // 2:
        raise ValueError(f"nibble plane row bytes {nibble_packed.shape[1]} != {gs // 2}")
    u8 = np.ascontiguousarray(nibble_packed, dtype=np.uint8)
    out = np.empty((u8.shape[0], gs), dtype=np.uint16)
    out[:, 0::2] = u8 & 0x0F
    out[:, 1::2] = (u8 >> 4) & 0x0F
    return out


def _unpack_highbits_np(high_packed: np.ndarray, bits: int, gs: int) -> np.ndarray:
    assert high_packed.ndim == 2
    if bits == 4:
        if high_packed.shape[1] != 0:
            raise ValueError("Q4 high plane must be empty")
        return np.zeros((high_packed.shape[0], gs), dtype=np.uint16)
    if bits not in (5, 6):
        raise ValueError(f"high-bit plane supports Q5/Q6, got bits={bits}")
    high_bits = bits - 4
    bpr = high_bytes_per_group(gs, bits)
    if high_packed.shape[1] != bpr:
        raise ValueError(f"high plane row bytes {high_packed.shape[1]} != {bpr}")
    shifts = np.arange(high_bits, dtype=np.uint16)
    out = np.empty((high_packed.shape[0], gs), dtype=np.uint16)
    chunk = max(1, _CHUNK_BITS_BUDGET // max(1, gs * high_bits))
    for g0 in range(0, high_packed.shape[0], chunk):
        g1 = min(high_packed.shape[0], g0 + chunk)
        bits_arr = np.unpackbits(high_packed[g0:g1], axis=1, bitorder="little")
        bits_arr = bits_arr[:, : gs * high_bits].reshape(g1 - g0, gs, high_bits).astype(np.uint16)
        out[g0:g1] = (bits_arr << shifts[None, None, :]).sum(axis=2)
    return out


def unpack_plane_split_groups(
    nibble_packed: np.ndarray,
    high_packed: np.ndarray,
    bits: int,
    gs: int,
) -> np.ndarray:
    """v3 nibble/high plane bytes -> [num_groups, gs] sign-extended int8."""
    if bits == 8:
        if high_packed.shape[1] != 0:
            raise ValueError("W8 high plane must be empty")
        return unpack_w8_groups(nibble_packed).reshape(nibble_packed.shape[0], gs)
    low = _unpack_low_nibbles_np(nibble_packed, gs)
    high = _unpack_highbits_np(high_packed, bits, gs)
    u = low | (high << 4)
    sign_bit = 1 << (bits - 1)
    span = 1 << bits
    signed = np.where(u & sign_bit, u.astype(np.int32) - span, u.astype(np.int32))
    return signed.astype(np.int8)


def pack_w8_groups(codes_int8: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(codes_int8, dtype=np.int8).view(np.uint8)


def unpack_w8_groups(packed_uint8: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(packed_uint8, dtype=np.uint8).view(np.int8)


# ----------------------------- torch GPU path -----------------------------

def _sign_extend_torch(u: torch.Tensor, bits: int) -> torch.Tensor:
    sign_bit = 1 << (bits - 1)
    span = 1 << bits
    return torch.where((u & sign_bit) != 0, u - span, u).to(torch.int8)


def _unpack_high_g64_indices(device: torch.device, high_bits: int, bpr: int):
    key = (device.type, device.index if device.index is not None else -1, high_bits, bpr)
    cached = _UNPACK_HIGH_G64_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    lanes = torch.arange(64, device=device, dtype=torch.int64)
    bitpos = lanes * high_bits
    byte_idx = torch.div(bitpos, 8, rounding_mode="floor")
    shift = bitpos - byte_idx * 8
    cached = (byte_idx, shift)
    _UNPACK_HIGH_G64_INDEX_CACHE[key] = cached
    return cached


def _unpack_high_g64_fast_torch(high_packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Fast G64 high-plane unpack for Q5/Q6."""
    high_bits = bits - 4
    if high_bits <= 0:
        return torch.zeros((high_packed.shape[0], 64), dtype=torch.int16, device=high_packed.device)
    _, bpr = high_packed.shape
    byte_idx, shift = _unpack_high_g64_indices(high_packed.device, high_bits, bpr)
    raw = high_packed.index_select(1, byte_idx).to(torch.int16)
    return (raw >> shift) & ((1 << high_bits) - 1)


def pack_low_nibble_torch(codes: torch.Tensor) -> torch.Tensor:
    """[G, gs] int codes (device) -> low-nibble plane bytes."""
    g, gs = codes.shape
    if gs % 2 != 0:
        raise ValueError(f"group_size must be even for nibble packing, got {gs}")
    out = torch.empty((g, gs // 2), dtype=torch.uint8, device=codes.device)
    per = max(1, _TORCH_CHUNK_BYTES // max(1, gs * 4))
    for a in range(0, g, per):
        b = min(g, a + per)
        u = codes[a:b].to(torch.int16) & 0x0F
        out[a:b] = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)
    return out


def pack_highbit_torch(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[G, gs] int codes (device) -> v3 high-bit plane bytes."""
    g, gs = codes.shape
    if bits == 4:
        return torch.empty((g, 0), dtype=torch.uint8, device=codes.device)
    if bits not in (5, 6):
        raise ValueError(f"high-bit plane supports Q5/Q6, got bits={bits}")
    high_bits = bits - 4
    bpr = high_bytes_per_group(gs, bits)
    dev = codes.device
    mask = (1 << bits) - 1
    shifts = torch.arange(high_bits, device=dev, dtype=torch.int32)
    bweight = (1 << torch.arange(8, device=dev, dtype=torch.int32))
    out = torch.empty((g, bpr), dtype=torch.uint8, device=dev)
    per = max(1, _TORCH_CHUNK_BYTES // max(1, gs * high_bits * 4))
    for a in range(0, g, per):
        b = min(g, a + per)
        high = ((codes[a:b].to(torch.int32) & mask) >> 4) & ((1 << high_bits) - 1)
        bit = ((high.unsqueeze(-1) >> shifts) & 1).reshape(b - a, bpr, 8)
        out[a:b] = (bit * bweight).sum(-1).to(torch.uint8)
    return out


def pack_plane_split_torch(codes: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[G, gs] int codes (device) -> (nibble plane, high plane)."""
    if bits == 8:
        return codes.contiguous().view(torch.uint8), torch.empty(
            (codes.shape[0], 0), dtype=torch.uint8, device=codes.device
        )
    return pack_low_nibble_torch(codes), pack_highbit_torch(codes, bits)


def _unpack_low_nibbles_torch(nibble_packed: torch.Tensor, gs: int) -> torch.Tensor:
    g, bpr = nibble_packed.shape
    if bpr != gs // 2:
        raise ValueError(f"nibble plane row bytes {bpr} != {gs // 2}")
    nibbles = nibble_packed.to(torch.int16)
    out = torch.empty((g, gs), dtype=torch.int16, device=nibble_packed.device)
    out[:, 0::2] = nibbles & 0x0F
    out[:, 1::2] = (nibbles >> 4) & 0x0F
    return out


def _unpack_highbits_torch(high_packed: torch.Tensor, bits: int, gs: int) -> torch.Tensor:
    g, bpr = high_packed.shape
    if bits == 4:
        if bpr != 0:
            raise ValueError("Q4 high plane must be empty")
        return torch.zeros((g, gs), dtype=torch.int16, device=high_packed.device)
    if bits not in (5, 6):
        raise ValueError(f"high-bit plane supports Q5/Q6, got bits={bits}")
    exp_bpr = high_bytes_per_group(gs, bits)
    if bpr != exp_bpr:
        raise ValueError(f"high plane row bytes {bpr} != {exp_bpr}")
    if gs == 64:
        return _unpack_high_g64_fast_torch(high_packed, bits)
    high_bits = bits - 4
    dev = high_packed.device
    shifts8 = torch.arange(8, device=dev, dtype=torch.int32)
    wbits = (1 << torch.arange(high_bits, device=dev, dtype=torch.int32))
    out = torch.empty((g, gs), dtype=torch.int16, device=dev)
    per = max(1, _TORCH_CHUNK_BYTES // max(1, bpr * 8 * 4))
    for a in range(0, g, per):
        b = min(g, a + per)
        bit = (high_packed[a:b].to(torch.int32).unsqueeze(-1) >> shifts8) & 1
        bit = bit.reshape(b - a, bpr * 8)[:, : gs * high_bits].reshape(b - a, gs, high_bits)
        out[a:b] = (bit * wbits).sum(-1).to(torch.int16)
    return out


def unpack_plane_split_torch(
    nibble_packed: torch.Tensor,
    high_packed: torch.Tensor,
    bits: int,
    gs: int,
) -> torch.Tensor:
    """v3 nibble/high plane bytes -> [G, gs] sign-extended int8 (device)."""
    if bits == 8:
        if high_packed.shape[1] != 0:
            raise ValueError("W8 high plane must be empty")
        return nibble_packed.reshape(nibble_packed.shape[0], gs).view(torch.int8)
    low = _unpack_low_nibbles_torch(nibble_packed, gs)
    high = _unpack_highbits_torch(high_packed, bits, gs)
    u = low | (high << 4)
    return _sign_extend_torch(u, bits)
