"""GPU weight-only symmetric quantization (per row, per K-group).

For a logical [N, K] matrix this produces fp16 group scales [N, K/group] and signed
codes [N, K/group, group]. Codes are quantized with the *fp16-rounded* scale so the
on-disk value is optimal for what the runtime reads back. See
../../docs/q5090_packed_file_format_v2.md sections 8-9.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

# Smallest positive fp16 (subnormal, 2**-24): rescues groups whose true scale would
# underflow fp16 to zero while the data is not all-zero.
_FP16_MIN_SUBNORMAL = 2.0 ** -24


def pick_device(prefer: str = "cuda") -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


def quantize_core(w: torch.Tensor, group_size: int, qmax: int, qmin: int
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D [N, K] tensor on its current device.

    Returns (scale_fp16 [N, K/group] fp16, codes [N, K/group, group] int8), both on
    w's device. K must be a multiple of group_size.
    """
    assert w.dim() == 2, f"expected 2D, got {tuple(w.shape)}"
    n, k = w.shape
    assert k % group_size == 0, f"K={k} not multiple of group {group_size}"
    kg = k // group_size
    wg = w.reshape(n, kg, group_size)
    if wg.dtype != torch.float32:
        wg = wg.float()
    maxabs = wg.abs().amax(dim=2)                         # [n, kg]
    scale16 = (maxabs / float(qmax)).to(torch.float16)
    underflow = (scale16 == 0) & (maxabs > 0)
    if bool(underflow.any()):
        tiny = torch.tensor(_FP16_MIN_SUBNORMAL, device=w.device, dtype=torch.float16)
        scale16 = torch.where(underflow, tiny, scale16)
    inv = scale16.to(torch.float32)
    inv = torch.where(inv > 0, 1.0 / inv, torch.zeros_like(inv))
    q = torch.clamp(torch.round(wg * inv.unsqueeze(-1)), float(qmin), float(qmax))
    return scale16, q.to(torch.int8)


def quantize_rows(w: torch.Tensor, group_size: int, qmax: int, qmin: int,
                  device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy wrapper around quantize_core (used by the reference tests)."""
    if device is None:
        device = pick_device()
    scale16, codes = quantize_core(w.to(device), group_size, qmax, qmin)
    return scale16.cpu().numpy(), codes.cpu().numpy()


def dequantize_rows(scale_fp16: np.ndarray, codes_int8: np.ndarray) -> np.ndarray:
    """Inverse of quantize_rows: returns float32 [N, K] = scale * code."""
    scale = scale_fp16.astype(np.float32)[:, :, None]
    return (codes_int8.astype(np.float32) * scale).reshape(scale_fp16.shape[0], -1)
