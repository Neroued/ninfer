"""Grouped 1D discrete FP16 MSE quantization used by NInfer artifact converters.

Each group independently selects the finite signed binary16 scale and signed
two's-complement codes that minimize FP64 squared reconstruction error across
all 63,486 nonzero finite binary16 magnitudes/signs.  Tie-breaking prefers
positive scale over negative scale, then smaller FP16 magnitude word; midpoint
ties round to even integer codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import QuantFormat, get_format
from tools.artifact.layouts import row_split_geometry

_MAX_FP16_MAG_WORD = 0x7BFF


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and signed binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


@lru_cache(maxsize=1)
def _cuda_module():
    from torch.utils.cpp_extension import load

    return load(
        name="ninfer_groupwise_mse",
        sources=[
            str(Path(__file__).with_name("groupwise" + suffix))
            for suffix in (".cpp", ".cu")
        ],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )


@lru_cache(maxsize=1)
def _fp16_mag_table() -> np.ndarray:
    words = np.arange(_MAX_FP16_MAG_WORD + 1, dtype=np.uint16)
    return words.view(np.float16).astype(np.float64)


def _pairwise_sum(values: np.ndarray) -> np.ndarray:
    """Exact binary-tree reduction matching the CUDA warp butterfly order."""
    s = values
    while s.shape[-1] > 1:
        s = s[..., 0::2] + s[..., 1::2]
    return s[..., 0]


def _nearest_codes_fp64(
    y: np.ndarray, a: np.ndarray, qmin: int, qmax: int
) -> np.ndarray:
    """Exact nearest integer codes in [qmin, qmax] with ties-to-even."""
    u = y / a
    q = np.clip(np.rint(u), qmin, qmax).astype(np.int32)
    up = (q < qmax) & (
        (y > (q + 0.5) * a) | ((y == (q + 0.5) * a) & ((q & 1) != 0))
    )
    q = np.where(up, q + 1, q)
    down = (
        ~up
        & (q > qmin)
        & ((y < (q - 0.5) * a) | ((y == (q - 0.5) * a) & ((q & 1) != 0)))
    )
    return np.where(down, q - 1, q)


def _fp16_ceil_word(table: np.ndarray, target: float) -> int:
    if target <= table[1]:
        return 1
    if target >= 65504.0:
        return _MAX_FP16_MAG_WORD
    return int(np.searchsorted(table, target, side="left"))


def _fp16_nearest_word(table: np.ndarray, target: float) -> int:
    if target <= table[1]:
        return 1
    if target >= 65504.0:
        return _MAX_FP16_MAG_WORD
    hi = int(np.searchsorted(table, target, side="left"))
    if hi <= 1:
        return 1
    return hi - 1 if (target - table[hi - 1]) <= (table[hi] - target) else hi


def _cpu_quantize_groups(
    groups_f32: np.ndarray, qmin: int, qmax: int
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(groups_f32).all():
        raise ValueError("grouped quantization source contains NaN or infinity")

    table = _fp16_mag_table()
    num_groups, group_size = groups_f32.shape
    out_codes = np.zeros((num_groups, group_size), dtype=np.int8)
    out_scales = np.zeros((num_groups,), dtype=np.uint16)

    q_neg_abs = float(-qmin)
    q_pos_abs = float(qmax)
    chunk = 64

    for g in range(num_groups):
        x = groups_f32[g].astype(np.float64)
        xmax = float(np.maximum(0.0, x.max()))
        xmin = float(np.maximum(0.0, (-x).max()))
        maxd = max(xmax, xmin)
        if maxd == 0.0:
            continue

        abs_x = np.abs(x)
        x_sq = x * x
        w_pos0 = _fp16_ceil_word(table, max(xmax / q_pos_abs, xmin / q_neg_abs))
        w_neg0 = _fp16_ceil_word(table, max(xmax / q_neg_abs, xmin / q_pos_abs))
        w_max = _fp16_ceil_word(table, maxd)
        w_hi_max = _fp16_ceil_word(table, 2.0 * maxd)

        best_err = float("inf")
        best_sign = 1
        best_word = w_pos0

        def eval_one(sign: int, word: int) -> None:
            nonlocal best_err, best_sign, best_word
            a = table[word]
            y = x if sign > 0 else -x
            q = _nearest_codes_fp64(y, a, qmin, qmax)
            d = y - q * a
            e = float(_pairwise_sum(d * d))
            if (
                e < best_err
                or (
                    e == best_err
                    and (sign > best_sign or (sign == best_sign and word < best_word))
                )
            ):
                best_err = e
                best_sign = sign
                best_word = word

        def eval_batch(sign: int, words: np.ndarray) -> None:
            nonlocal best_err, best_sign, best_word
            if words.size == 0:
                return
            a = table[words][:, None]
            y = (x if sign > 0 else -x)[None, :]
            q = _nearest_codes_fp64(y, a, qmin, qmax)
            d = y - q * a
            errs = _pairwise_sum(d * d)
            idx = int(np.argmin(errs))
            e = float(errs[idx])
            if e <= best_err:
                tied = np.flatnonzero(errs == e)
                word = int(words[tied].min())
                if (
                    e < best_err
                    or sign > best_sign
                    or (sign == best_sign and word < best_word)
                ):
                    best_err = e
                    best_sign = sign
                    best_word = word

        # Tight initial anchors + 3 AM steps
        eval_one(1, w_pos0)
        if w_pos0 > 1:
            eval_one(1, w_pos0 - 1)
        eval_one(-1, w_neg0)
        if w_neg0 > 1:
            eval_one(-1, w_neg0 - 1)
        eval_one(1, w_max)

        for sign, w_init in ((1, w_pos0), (-1, w_neg0)):
            w_cur = w_init
            y = x if sign > 0 else -x
            for _ in range(3):
                a = table[w_cur]
                q = _nearest_codes_fp64(y, a, qmin, qmax)
                qq = float(_pairwise_sum((q * q).astype(np.float64)))
                xq = float(_pairwise_sum(y * q))
                if qq <= 0.0 or xq <= 0.0:
                    break
                w_next = _fp16_nearest_word(table, xq / qq)
                eval_one(sign, w_next)
                if w_next == w_cur:
                    break
                w_cur = w_next

        # 1 & 2. Exact downward sweeps for + and - with monotone clipping bound
        for sign, w0, qp, qn in (
            (1, w_pos0, q_pos_abs, q_neg_abs),
            (-1, w_neg0, q_neg_abs, q_pos_abs),
        ):
            hi = w0 - 1
            while hi >= 1:
                lo = max(1, hi - chunk + 1)
                cand = np.arange(hi, lo - 1, -1, dtype=np.int32)
                a = table[cand]
                cp = np.maximum(0.0, xmax - qp * a)
                cn = np.maximum(0.0, xmin - qn * a)
                clip_lb = cp * cp + cn * cn
                exceed = np.flatnonzero(clip_lb > best_err)
                if exceed.size > 0:
                    eval_batch(sign, cand[: int(exceed[0])])
                    break
                eval_batch(sign, cand)
                hi = lo - 1

        # 3. Exact upward sweep stopping when monotone zero-rounding energy > best_err
        lo = min(w_pos0, w_neg0)
        while lo <= w_hi_max:
            hi = min(w_hi_max, lo + chunk - 1)
            cand = np.arange(lo, hi + 1, dtype=np.int32)
            a = table[cand]
            zero_mask = abs_x[None, :] <= (0.5 * a[:, None])
            z_energy = _pairwise_sum(np.where(zero_mask, x_sq[None, :], 0.0))
            exceed = np.flatnonzero(z_energy > best_err)
            if exceed.size > 0:
                cand = cand[: int(exceed[0])]
                if cand.size > 0:
                    eval_batch(1, cand[cand >= w_pos0])
                    eval_batch(-1, cand[cand >= w_neg0])
                break
            eval_batch(1, cand[cand >= w_pos0])
            eval_batch(-1, cand[cand >= w_neg0])
            lo = hi + 1

        y_best = x if best_sign > 0 else -x
        out_codes[g] = _nearest_codes_fp64(y_best, table[best_word], qmin, qmax).astype(
            np.int8
        )
        out_scales[g] = np.uint16((0x8000 if best_sign < 0 else 0) | best_word)

    return out_codes, out_scales.view(np.float16)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values to exact minimum-MSE codes and FP16 scales.

    Padding lanes beyond logical ``K`` are zero and always map to code 0 with zero
    error contribution, leaving partially populated tail groups unaffected.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target).float()
    if not bool(torch.isfinite(logical).all()):
        raise ValueError("grouped quantization source contains NaN or infinity")
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(-1, spec.group_size).contiguous()
    if target.type == "cuda":
        codes, scales = _cuda_module().quantize(grouped, spec.bits)
    elif target.type == "cpu":
        codes_np, scales_np = _cpu_quantize_groups(
            grouped.numpy(), spec.qmin, spec.qmax
        )
        codes = torch.from_numpy(codes_np)
        scales = torch.from_numpy(scales_np)
    else:
        raise ValueError("grouped quantization supports CPU or CUDA")
    return QuantizedMatrix(
        codes=codes.reshape(geometry.n, geometry.groups_per_row, spec.group_size),
        scales=scales.reshape(geometry.n, geometry.groups_per_row),
    )
