"""NF4 words in complete 4 KiB pages of compact encoded rows."""

from __future__ import annotations

import struct
from typing import Sequence

import numpy as np
import torch

from ..formats import valid_positive_fp32_word
from ..layouts import row_paged_geometry
from ._tensor_bytes import _exact_uint8_matrix


def divisor_word(value: bytes | torch.Tensor) -> bytes:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise TypeError("NF4 divisor must be one FP32 word")
        value = (
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    value = bytes(value)
    if len(value) != 4 or not valid_positive_fp32_word(struct.unpack("<I", value)[0]):
        raise ValueError("NF4 divisor must be positive finite FP32")
    return value


def nf4_rows(codes: torch.Tensor, scales: torch.Tensor, k: int) -> np.ndarray:
    rows = codes.shape[0]
    codes = _exact_uint8_matrix(codes, (rows, k // 2), "NF4 packed codes")
    scales = _exact_uint8_matrix(scales, (rows, k // 16), "NF4 scales")
    if bool((scales > 126).any()):
        raise ValueError("NF4 scales must be nonnegative finite E4M3FN")
    return np.concatenate((codes.numpy(), scales.numpy()), axis=1)


def encode_nf4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    divisor: bytes | torch.Tensor,
    shape: Sequence[int],
) -> bytes:
    g = row_paged_geometry("nf4_g16_e4m3fn", shape)
    if codes.shape[0] != g.n:
        raise ValueError("NF4 codes differ from logical rows")
    rows = nf4_rows(codes, scales, g.k)
    pages = np.zeros((g.page_count, 4096), dtype=np.uint8)
    full, tail = divmod(g.n, g.rows_per_page)
    used = g.rows_per_page * g.row_bytes
    pages[:full, :used] = rows[: full * g.rows_per_page].reshape(full, used)
    if tail:
        pages[full, : tail * g.row_bytes] = rows[full * g.rows_per_page :].reshape(-1)
    return pages.tobytes() + divisor_word(divisor)


def decode_nf4_words(payload: bytes | bytearray | memoryview, shape: Sequence[int]):
    g = row_paged_geometry("nf4_g16_e4m3fn", shape)
    raw = memoryview(payload).cast("B")
    if len(raw) != g.payload_bytes:
        raise ValueError("NF4 payload size differs from geometry")
    divisor = divisor_word(bytes(raw[g.weight_divisor_offset :]))
    pages = np.frombuffer(raw[: g.weight_divisor_offset], dtype=np.uint8).reshape(
        -1, 4096
    )
    used = g.rows_per_page * g.row_bytes
    if np.any(pages[:, used:]):
        raise ValueError("nonzero NF4 page padding")
    rows = pages[:, :used].reshape(-1, g.row_bytes)
    if np.any(rows[g.n :]):
        raise ValueError("nonzero NF4 final-page padding")
    rows = rows[: g.n]
    codes = torch.from_numpy(rows[:, : g.k // 2].copy())
    scales = torch.from_numpy(rows[:, g.k // 2 :].copy())
    if bool((scales > 126).any()):
        raise ValueError("NF4 scales must be nonnegative finite E4M3FN")
    return (
        codes,
        scales,
        torch.frombuffer(bytearray(divisor), dtype=torch.float32).reshape(()),
    )
