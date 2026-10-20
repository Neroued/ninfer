"""ModelOpt NVFP4 import: preserved nibbles/scales and multiplier-to-divisor conversion."""

from __future__ import annotations

from functools import lru_cache
import struct

import torch

from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource


class ModelOptNvfp4:
    def __init__(self, store: SafetensorsSource):
        self.store = store

    @lru_cache(maxsize=None)
    def scales(self, prefix: str) -> tuple[bytes, bytes, bytes, bytes]:
        result = []
        for suffix in ("weight_scale_2", "input_scale"):
            name = f"{prefix}.{suffix}"
            info = self.store.describe(name)
            if info.dtype != "F32" or info.shape != ():
                raise ValueError(f"{name}: expected an FP32 scalar multiplier")
            raw = self.store.read_flat(name).view(torch.uint8).numpy().tobytes()
            if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
                raise ValueError(f"{name}: multiplier must be positive finite")
            divisor = struct.pack("<f", 1.0 / struct.unpack("<f", raw)[0])
            if not valid_positive_fp32_word(struct.unpack("<I", divisor)[0]):
                raise ValueError(f"{name}: reciprocal is not positive finite FP32")
            result.extend((raw, divisor))
        return tuple(result)

    def matrix(self, prefix: str, shape: tuple[int, int]) -> LogicalSource:
        n, k = shape
        for suffix, dims, dtype in (
            ("weight", (n, k // 2), "U8"),
            ("weight_scale", (n, k // 16), "F8_E4M3"),
        ):
            info = self.store.describe(prefix + "." + suffix)
            if info.shape != dims or info.dtype != dtype:
                raise ValueError(f"{prefix}.{suffix}: unexpected encoded geometry/type")
        if k % 16:
            raise ValueError("ModelOpt NVFP4 requires K divisible by 16")
        _, dw, _, da = self.scales(prefix)

        def encoded(first, last):
            if not 0 <= first < last <= n:
                raise ValueError("encoded source row range exceeds matrix")
            codes = self.store.read_flat(
                prefix + ".weight", first * (k // 2), last * (k // 2)
            ).reshape(-1, k // 2)
            scales = (
                self.store.read_flat(
                    prefix + ".weight_scale", first * (k // 16), last * (k // 16)
                )
                .view(torch.uint8)
                .reshape(-1, k // 16)
            )
            if bool((scales > 126).any()):
                raise ValueError(f"{prefix}: invalid E4M3FN scale")
            return EncodedRows("nvfp4", codes, scales, dw)

        def values(begin, end):
            if begin == end:
                return torch.empty(0, dtype=torch.float64)
            first, last = begin // k, (end + k - 1) // k
            words = encoded(first, last)
            codes = (
                torch.stack((words.codes & 15, words.codes >> 4), -1)
                .reshape(-1, k)
                .long()
            )
            table = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=torch.float64,
            )
            decoded = table[codes] * words.scales.view(
                torch.float8_e4m3fn
            ).double().repeat_interleave(16, -1)
            decoded /= struct.unpack("<f", dw)[0]
            return decoded.reshape(-1)[begin - first * k : end - first * k]

        return LogicalSource(
            shape,
            f"{self.store.path}:{prefix} (modelopt nvfp4)",
            values,
            encoded,
            lambda: dw,
            lambda: da,
            encoded_format="nvfp4",
        )
