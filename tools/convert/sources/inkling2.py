"""Native Inkling-2 MoE expert bank sources (NVFP4_MUL, MXFP8, and BF16).

Inkling-2 checkpoints store routed experts as rank-3 tensors `[E, rows, cols]` where
`w13_weight` (and its scale tensor) interleaves `(gate_0, up_0, gate_1, up_1, ...)`.
These sources de-interleave even/odd rows into contiguous `[gate(0..I-1)]` and
`[up(0..I-1)]` logical views per expert with zero re-quantization.
"""

from __future__ import annotations

from functools import lru_cache
import struct

import numpy as np
import torch

from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def _deinterleave_slice(
    store: SafetensorsSource,
    name: str,
    expert: int,
    total_rows: int,
    row_width: int,
    first: int,
    last: int,
    half: int | None,
) -> torch.Tensor:
    base = expert * total_rows * row_width
    if half is None:
        raw = store.read_flat(name, base + first * row_width, base + last * row_width)
        return raw.reshape(last - first, row_width)
    raw = store.read_flat(
        name, base + (2 * first) * row_width, base + (2 * last) * row_width
    )
    return raw.reshape(2 * (last - first), row_width)[half::2].contiguous()


class InklingExpertBankSource:
    """Inspect and read one Inkling-2 MoE layer's native expert tensors."""

    def __init__(
        self,
        store: SafetensorsSource,
        prefix: str,
        experts: int,
        hidden: int,
        intermediate: int,
    ) -> None:
        self.store = store
        self.prefix = prefix
        self.experts = experts
        self.hidden = hidden
        self.intermediate = intermediate

        w13_info = store.describe(prefix + "w13_weight")
        w2_info = store.describe(prefix + "w2_weight")
        if w13_info.dtype != w2_info.dtype:
            raise ValueError(
                f"{prefix}: w13_weight ({w13_info.dtype}) and w2_weight ({w2_info.dtype}) dtypes differ"
            )
        if w13_info.dtype == "U8":
            self.format = "nvfp4_mul"
            self._validate_nvfp4()
        elif w13_info.dtype == "F8_E4M3":
            self.format = "mxfp8"
            self._validate_mxfp8()
        elif w13_info.dtype == "BF16":
            self.format = "bf16"
            self._validate_bf16()
        else:
            raise ValueError(
                f"{prefix}: unsupported Inkling2 expert weight dtype {w13_info.dtype!r}"
            )

    def _validate_nvfp4(self) -> None:
        e, h, i = self.experts, self.hidden, self.intermediate
        if h % 16 or i % 16:
            raise ValueError(f"{self.prefix}: NVFP4 requires K divisible by 16")
        checks = (
            ("w13_weight", (e, 2 * i, h // 2), "U8"),
            ("w13_weight_scale", (e, 2 * i, h // 16), "F8_E4M3"),
            ("w13_weight_amax", (e,), "F32"),
            ("w13_weight_outer_scale_denominator", (1,), "F32"),
            ("w2_weight", (e, h, i // 2), "U8"),
            ("w2_weight_scale", (e, h, i // 16), "F8_E4M3"),
            ("w2_weight_amax", (e,), "F32"),
            ("w2_weight_outer_scale_denominator", (1,), "F32"),
        )
        for suffix, dims, dtype in checks:
            info = self.store.describe(self.prefix + suffix)
            if info.shape != dims or info.dtype != dtype:
                raise ValueError(
                    f"{self.prefix}{suffix}: expected {dtype} {dims}, got {info.dtype} {info.shape}"
                )

    def _validate_mxfp8(self) -> None:
        e, h, i = self.experts, self.hidden, self.intermediate
        if h % 32 or i % 32:
            raise ValueError(f"{self.prefix}: MXFP8 requires K divisible by 32")
        checks = (
            ("w13_weight", (e, 2 * i, h), "F8_E4M3"),
            ("w13_weight_scale", (e, 2 * i, h // 32), "U8"),
            ("w2_weight", (e, h, i), "F8_E4M3"),
            ("w2_weight_scale", (e, h, i // 32), "U8"),
        )
        for suffix, dims, dtype in checks:
            info = self.store.describe(self.prefix + suffix)
            if info.shape != dims or info.dtype != dtype:
                raise ValueError(
                    f"{self.prefix}{suffix}: expected {dtype} {dims}, got {info.dtype} {info.shape}"
                )
        for suffix in (
            "w13_weight_amax",
            "w13_weight_outer_scale_denominator",
            "w2_weight_amax",
            "w2_weight_outer_scale_denominator",
        ):
            if self.store.has(self.prefix + suffix):
                raise ValueError(
                    f"{self.prefix}{suffix}: MXFP8 experts must not carry NVFP4 outer scales"
                )

    def _validate_bf16(self) -> None:
        e, h, i = self.experts, self.hidden, self.intermediate
        checks = (
            ("w13_weight", (e, 2 * i, h), "BF16"),
            ("w2_weight", (e, h, i), "BF16"),
        )
        for suffix, dims, dtype in checks:
            info = self.store.describe(self.prefix + suffix)
            if info.shape != dims or info.dtype != dtype:
                raise ValueError(
                    f"{self.prefix}{suffix}: expected {dtype} {dims}, got {info.dtype} {info.shape}"
                )

    def names(self) -> tuple[str, ...]:
        if self.format == "nvfp4_mul":
            return tuple(
                self.prefix + suffix
                for suffix in (
                    "w13_weight",
                    "w13_weight_scale",
                    "w13_weight_amax",
                    "w13_weight_outer_scale_denominator",
                    "w2_weight",
                    "w2_weight_scale",
                    "w2_weight_amax",
                    "w2_weight_outer_scale_denominator",
                )
            )
        if self.format == "mxfp8":
            return tuple(
                self.prefix + suffix
                for suffix in (
                    "w13_weight",
                    "w13_weight_scale",
                    "w2_weight",
                    "w2_weight_scale",
                )
            )
        return (self.prefix + "w13_weight", self.prefix + "w2_weight")

    @lru_cache(maxsize=2)
    def _nvfp4_multipliers(self, tensor_kind: str) -> tuple[bytes, ...]:
        amax_name = self.prefix + tensor_kind + "_weight_amax"
        denom_name = self.prefix + tensor_kind + "_weight_outer_scale_denominator"
        amax = self.store.read_flat(amax_name).numpy().astype(np.float32, copy=False)
        denom = float(self.store.read_flat(denom_name).item())
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError(f"{denom_name}: expected positive finite denominator")
        denom_f32 = np.float32(denom)
        words = []
        for expert in range(self.experts):
            val = np.float32(amax[expert]) / denom_f32
            raw = struct.pack("<f", float(val))
            if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
                raise ValueError(
                    f"{amax_name}[{expert}]: derived NVFP4 multiplier {float(val)!r} is invalid"
                )
            words.append(raw)
        return tuple(words)

    def expert_matrix(self, role: str, expert: int) -> LogicalSource:
        if role not in ("gate", "up", "down"):
            raise ValueError(f"unknown MoE expert role {role!r}")
        if not 0 <= expert < self.experts:
            raise ValueError(f"expert index {expert} out of range")
        if role == "down":
            tensor_kind = "w2"
            half = None
            n, k = self.hidden, self.intermediate
            total_rows = self.hidden
        else:
            tensor_kind = "w13"
            half = 0 if role == "gate" else 1
            n, k = self.intermediate, self.hidden
            total_rows = 2 * self.intermediate

        weight_name = self.prefix + tensor_kind + "_weight"
        scale_name = self.prefix + tensor_kind + "_weight_scale"
        label = f"{self.store.path}:{weight_name}[e={expert},{role}] ({self.format})"

        if self.format == "bf16":
            def read_bf16(begin: int, end: int) -> torch.Tensor:
                if begin == end:
                    return torch.empty(0, dtype=torch.bfloat16)
                first, last = begin // k, (end + k - 1) // k
                rows = _deinterleave_slice(
                    self.store, weight_name, expert, total_rows, k, first, last, half
                )
                return rows.reshape(-1)[begin - first * k : end - first * k]

            return LogicalSource((n, k), label, read_bf16)

        if self.format == "nvfp4_mul":
            def multiplier() -> bytes:
                return self._nvfp4_multipliers(tensor_kind)[expert]

            def encoded_nvfp4(first: int, last: int) -> EncodedRows:
                if not 0 <= first < last <= n:
                    raise ValueError(f"{label}: encoded row range [{first},{last}) exceeds {n}")
                codes = _deinterleave_slice(
                    self.store, weight_name, expert, total_rows, k // 2, first, last, half
                )
                scales = _deinterleave_slice(
                    self.store, scale_name, expert, total_rows, k // 16, first, last, half
                )
                return EncodedRows(
                    "nvfp4_mul",
                    codes.view(torch.uint8),
                    scales.view(torch.uint8),
                    multiplier(),
                )

            def values_nvfp4(begin: int, end: int) -> torch.Tensor:
                if begin == end:
                    return torch.empty(0, dtype=torch.float64)
                first, last = begin // k, (end + k - 1) // k
                words = encoded_nvfp4(first, last)
                nibbles = (
                    torch.stack((words.codes & 15, words.codes >> 4), dim=-1)
                    .reshape(-1, k)
                    .long()
                )
                block_scales = (
                    words.scales.view(torch.float8_e4m3fn)
                    .double()
                    .repeat_interleave(16, dim=-1)
                )
                decoded = _E2M1_TABLE[nibbles] * block_scales
                decoded *= struct.unpack("<f", multiplier())[0]
                return decoded.reshape(-1)[begin - first * k : end - first * k]

            return LogicalSource(
                (n, k),
                label,
                values_nvfp4,
                encoded_nvfp4,
                multiplier,
                None,
                "nvfp4_mul",
            )

        # MXFP8
        def encoded_mxfp8(first: int, last: int) -> EncodedRows:
            if not 0 <= first < last <= n:
                raise ValueError(f"{label}: encoded row range [{first},{last}) exceeds {n}")
            codes = _deinterleave_slice(
                self.store, weight_name, expert, total_rows, k, first, last, half
            )
            scales = _deinterleave_slice(
                self.store, scale_name, expert, total_rows, k // 32, first, last, half
            )
            return EncodedRows("mxfp8", codes.view(torch.uint8), scales.view(torch.uint8))

        def values_mxfp8(begin: int, end: int) -> torch.Tensor:
            if begin == end:
                return torch.empty(0, dtype=torch.float32)
            first, last = begin // k, (end + k - 1) // k
            words = encoded_mxfp8(first, last)
            scales = torch.ldexp(
                torch.ones_like(words.scales, dtype=torch.float32),
                words.scales.int() - 127,
            ).repeat_interleave(32, dim=-1)
            decoded = words.codes.view(torch.float8_e4m3fn).float() * scales
            return decoded.reshape(-1)[begin - first * k : end - first * k]

        return LogicalSource(
            (n, k),
            label,
            values_mxfp8,
            encoded_mxfp8,
            None,
            None,
            "mxfp8",
        )

    def role_bank(self, role: str) -> LogicalSource:
        matrices = tuple(self.expert_matrix(role, e) for e in range(self.experts))
        n, k = matrices[0].shape
        shape = (self.experts, n, k)
        width = n * k

        def read_values(begin: int, end: int) -> torch.Tensor:
            parts = []
            while begin < end:
                expert, local = divmod(begin, width)
                stop = min(end, (expert + 1) * width)
                parts.append(matrices[expert].values(local, local + stop - begin))
                begin = stop
            if not parts:
                return matrices[0].values(0, 0)
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        return LogicalSource(
            shape,
            f"inkling2_bank({self.prefix}{role}, {self.experts} experts, {self.format})",
            read_values,
            encoded_format=None if self.format == "bf16" else self.format,
            matrix_sources=matrices,
        )
