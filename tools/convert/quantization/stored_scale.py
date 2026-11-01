"""G16 MSE search over every positive finite stored E4M3FN scale.

The object divisor is already rounded to FP32. Compare scaled reconstruction
errors in FP64, so the common 1/divisor**2 factor need not be evaluated.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import struct

import numpy as np
import torch

from tools.artifact.formats import NF4_CODEBOOK, valid_positive_fp32_word
from .groupwise import QuantizedMatrix


@lru_cache(maxsize=1)
def _cuda_module():
    from torch.utils.cpp_extension import load

    return load(
        name="ninfer_stored_scale",
        sources=[
            str(Path(__file__).with_name("stored_scale" + suffix))
            for suffix in (".cpp", ".cu")
        ],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _cpu_words(values: np.ndarray, divisor: float, nf4: bool):
    groups = values.reshape(-1, 16).astype(np.float64) * divisor
    codes = np.empty_like(groups, dtype=np.uint8)
    scales = np.empty(len(groups), dtype=np.uint8)
    # This ordering implements the E2M1 magnitude tie rule with argmin's first hit.
    order = np.arange(16) if nf4 else np.array([0, 2, 4, 6, 1, 3, 5, 7])
    table = np.array(NF4_CODEBOOK if nf4 else [0, 0.5, 1, 1.5, 2, 3, 4, 6])
    candidates = (
        torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).double().numpy()
    )
    for begin in range(0, len(groups), 1024):
        x = groups[begin : begin + 1024]
        target = x if nf4 else np.abs(x)
        best = np.full(len(x), np.inf)
        best_codes = np.zeros(x.shape, dtype=np.uint8)
        best_scales = np.ones(len(x), dtype=np.uint8)
        for word in range(1, 127):
            s = candidates[word]
            index = order[np.abs(target[..., None] - table[order] * s).argmin(-1)]
            error = np.square(target - table[index] * s).sum(-1)
            better = error < best
            best[better] = error[better]
            best_codes[better] = index[better]
            best_scales[better] = word
        if not nf4:
            best_codes |= np.signbit(x).astype(np.uint8) * 8
        zero = (x == 0).all(-1)
        best_codes[zero] = 7 if nf4 else 0
        best_scales[zero] = 0
        codes[begin : begin + len(x)] = best_codes
        scales[begin : begin + len(x)] = best_scales
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return packed, scales


def quantize_stored_scale(
    weight: torch.Tensor, format: str, divisor: bytes, *, device="cuda"
) -> QuantizedMatrix:
    if format not in ("nf4_g16_e4m3fn", "nvfp4"):
        raise ValueError("stored-scale MSE supports NF4 and NVFP4")
    if weight.ndim != 2 or weight.shape[1] % 16 or not weight.dtype.is_floating_point:
        raise ValueError(
            "stored-scale MSE requires a floating [N,K] matrix with K divisible by 16"
        )
    if len(divisor) != 4 or not valid_positive_fp32_word(
        struct.unpack("<I", divisor)[0]
    ):
        raise ValueError("object divisor must be positive finite FP32")
    d = struct.unpack("<f", divisor)[0]
    n, k = weight.shape
    target = torch.device(device)
    values = weight.detach().to(device=target).float().contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("stored-scale source contains NaN or infinity")
    if target.type == "cuda":
        codes, scales = _cuda_module().quantize(values, d, format == "nf4_g16_e4m3fn")
    elif target.type == "cpu":
        codes, scales = _cpu_words(values.numpy(), d, format == "nf4_g16_e4m3fn")
        codes, scales = torch.from_numpy(codes), torch.from_numpy(scales)
    else:
        raise ValueError("stored-scale MSE supports CPU or CUDA")
    return QuantizedMatrix(codes.reshape(n, k // 2), scales.reshape(n, k // 16))
