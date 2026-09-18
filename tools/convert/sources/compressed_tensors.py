"""Interpret the current compressed-tensors FP8/NVFP4 fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource, tensor_source


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret the current compressed-tensors NVFP4 or per-row FP8 representation."""
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported encoded source format {format}")
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def divisor(suffix: str) -> bytes:
        return _divisor_word(store, f"{prefix}.{suffix}")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4":
            packed, scale = f"{prefix}.weight_packed", f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, divisor("weight_global_scale"))
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        if info.dtype != "BF16" or prod(info.shape) != n:
            raise ValueError(f"{scale}: expected one BF16 scale per row")
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        scales = store.read_flat(scale, begin, end)
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        if format == "fp8_e4m3fn_row_bf16":
            values = (
                words.codes.view(torch.float8_e4m3fn).float()
                * words.scales.float()[:, None]
            )
        else:
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
                last - first, k
            )
            values_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            values = values_table[codes.long()]
            scales = (
                words.scales.view(torch.float8_e4m3fn)
                .float()
                .repeat_interleave(16, dim=1)
            )
            values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        (lambda: divisor("weight_global_scale")) if format == "nvfp4" else None,
        (lambda: divisor("input_global_scale")) if format == "nvfp4" else None,
    )


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None and store.has(prefix + ".weight_packed"):
                actual = "nvfp4"
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                actual = "fp8_e4m3fn_row_bf16"
            resolved = (
                tensor_source(store, name, shape)
                if actual is None
                else compressed_matrix_source(store, prefix, shape, actual)
            )
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )
