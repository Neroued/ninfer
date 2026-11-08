"""Native ModelOpt weight representations, independent of architecture and recipe."""

from __future__ import annotations

from functools import lru_cache
from math import prod
import struct

import torch

from tools.artifact.formats import valid_positive_fp32_word
from tools.artifact.codecs.csf32 import decode_csf32_scales
from .logical import EncodedPlanes, EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

_CSF_FIELDS = {
    "bases": "csf32_bases",
    "nibbles": "csf32_codes",
    "offsets": "csf32_offsets",
    "page_bases": "csf32_page_bases",
    "exceptions": "csf32_exceptions",
}


class ModelOptSource:
    """Read exact codes/scales; only explicit value consumers decode weights."""

    def __init__(self, store: SafetensorsSource):
        self.store = store

    def names(self, prefix: str) -> tuple[str, ...]:
        names = [prefix + ".weight"]
        if self.store.has(prefix + ".weight_scale.csf32_bases"):
            names.extend(
                prefix + ".weight_scale." + field for field in _CSF_FIELDS.values()
            )
        elif self.store.has(prefix + ".weight_scale"):
            names.append(prefix + ".weight_scale")
        for field in ("weight_scale_2", "input_scale"):
            if self.store.has(prefix + "." + field):
                names.append(prefix + "." + field)
        return tuple(names)

    @lru_cache(maxsize=None)
    def scalar(self, name: str) -> bytes:
        info = self.store.describe(name)
        if info.dtype != "F32" or info.shape != ():
            raise ValueError(f"{name}: expected an FP32 scalar multiplier")
        raw = self.store.read_flat(name).view(torch.uint8).numpy().tobytes()
        if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError(f"{name}: multiplier must be positive finite")
        return raw

    def matrix(self, prefix: str, shape: tuple[int, int]) -> LogicalSource:
        n, k = shape
        info = self.store.describe(prefix + ".weight")
        if info.dtype == "U8":
            if info.shape != (n, k // 2) or k % 16:
                raise ValueError(f"{prefix}: NVFP4 geometry differs from {shape}")
            return self._nvfp4(prefix, shape)
        if info.dtype != "F8_E4M3" or info.shape != shape:
            raise ValueError(f"{prefix}: unsupported ModelOpt encoded representation")
        scale = self.store.describe(prefix + ".weight_scale")
        if scale.dtype == "U8" and scale.shape == (n, k // 32) and k % 32 == 0:
            return self._fp8(prefix, shape, "mxfp8", 1, 32)
        if scale.dtype == "F32" and scale.shape == (n // 128, 1, k // 128, 1):
            if n % 128 or k % 128:
                raise ValueError(
                    f"{prefix}: block FP8 requires complete 128x128 blocks"
                )
            return self._fp8(prefix, shape, "fp8_e4m3fn_block128x128_fp32", 128, 128)
        raise ValueError(f"{prefix}: unsupported ModelOpt scale representation")

    def _fp8(self, prefix, shape, format, block_rows, block_columns):
        n, k = shape

        def encoded(first, last):
            if not 0 <= first < last <= n or first % block_rows or last % block_rows:
                raise ValueError(
                    f"{prefix}: encoded rows must contain complete scale blocks"
                )
            codes = self.store.read_flat(prefix + ".weight", first * k, last * k)
            scales = self.store.read_flat(
                prefix + ".weight_scale",
                first // block_rows * (k // block_columns),
                last // block_rows * (k // block_columns),
            )
            return EncodedRows(
                format,
                codes.view(torch.uint8).reshape(last - first, k),
                scales.reshape((last - first) // block_rows, k // block_columns),
            )

        def values(begin, end):
            if begin == end:
                return torch.empty(0, dtype=torch.float32)
            width = block_rows * k
            first, last = (
                begin // width * block_rows,
                (end + width - 1) // width * block_rows,
            )
            words = encoded(first, last)
            scales = words.scales
            if format == "mxfp8":
                scales = torch.ldexp(
                    torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
                )
            scales = scales.repeat_interleave(block_rows, 0).repeat_interleave(
                block_columns, 1
            )
            decoded = words.codes.view(torch.float8_e4m3fn).float() * scales
            return decoded.flatten()[begin - first * k : end - first * k]

        return LogicalSource(
            shape,
            f"{self.store.path}:{prefix} ({format})",
            values,
            encoded,
            encoded_format=format,
        )

    def _nvfp4(self, prefix, shape):
        n, k = shape
        multiplier = lambda: self.scalar(prefix + ".weight_scale_2")
        activation = (
            (lambda: self.scalar(prefix + ".input_scale"))
            if self.store.has(prefix + ".input_scale")
            else None
        )
        native = None
        if self.store.has(prefix + ".weight_scale.csf32_bases"):
            fields = {
                "codes": prefix + ".weight",
                "matrix_scale": prefix + ".weight_scale_2",
            }
            fields.update(
                {
                    name: prefix + ".weight_scale." + suffix
                    for name, suffix in _CSF_FIELDS.items()
                }
            )
            info = {name: self.store.describe(field) for name, field in fields.items()}
            dtypes = {
                "codes": "U8",
                "matrix_scale": "F32",
                "bases": "U8",
                "nibbles": "U8",
                "offsets": "U16",
                "page_bases": "U32",
                "exceptions": "U16",
            }
            for name, value in info.items():
                if value.dtype != dtypes[name]:
                    raise ValueError(f"{fields[name]}: expected {dtypes[name]}")

            def plane(name, begin, end):
                item = info[name]
                word = {"U8": 1, "U16": 2, "U32": 4, "F32": 4}[item.dtype]
                if not 0 <= begin <= end <= item.bytes or begin % word or end % word:
                    raise ValueError(f"{fields[name]}: invalid native byte range")
                return self.store.read_flat(
                    fields[name], begin // word, end // word
                ).view(torch.uint8)

            native = EncodedPlanes(
                "csf_32x2_v1",
                {name: value.bytes for name, value in info.items()},
                plane,
                prod(info["exceptions"].shape),
            )
            encoded = None
        else:
            info = self.store.describe(prefix + ".weight_scale")
            if info.shape != (n, k // 16) or info.dtype != "F8_E4M3":
                raise ValueError(f"{prefix}: unexpected NVFP4 scale geometry/type")

            def encoded(first, last):
                if not 0 <= first < last <= n:
                    raise ValueError(f"{prefix}: encoded row range exceeds matrix")
                codes = self.store.read_flat(
                    prefix + ".weight", first * k // 2, last * k // 2
                )
                scales = self.store.read_flat(
                    prefix + ".weight_scale", first * k // 16, last * k // 16
                )
                return EncodedRows(
                    "nvfp4_mul",
                    codes.reshape(-1, k // 2),
                    scales.view(torch.uint8).reshape(-1, k // 16),
                    multiplier(),
                )

        def values(begin, end):
            if begin == end:
                return torch.empty(0, dtype=torch.float64)
            first, last = begin // k, (end + k - 1) // k
            if encoded is None:
                planes = {
                    name: native.read(name, 0, size).numpy().tobytes()
                    for name, size in native.sizes.items()
                    if name not in ("codes", "matrix_scale")
                }
                scales = decode_csf32_scales(planes, shape, native.exception_count)[
                    first:last
                ]
                codes = self.store.read_flat(
                    prefix + ".weight", first * k // 2, last * k // 2
                )
                words = EncodedRows(
                    "nvfp4_mul", codes.reshape(-1, k // 2), scales, multiplier()
                )
            else:
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
            decoded *= struct.unpack("<f", multiplier())[0]
            return decoded.flatten()[begin - first * k : end - first * k]

        return LogicalSource(
            shape,
            f"{self.store.path}:{prefix} (native NVFP4)",
            values,
            encoded,
            multiplier,
            activation,
            "nvfp4_mul",
            native,
        )
