"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Callable, Mapping

import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

from .quantization.fp8_row import quantize_bf16_rows
from .quantization.groupwise import quantize_matrix
from .quantization.nvfp4 import E2M1_MAX, quantize_nvfp4, weight_divisor
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class MethodInput:
    parameter: str
    source: LogicalSource
    uses: tuple[UseKey, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryValue:
    format: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def activation_divisor(cls, value: bytes | float) -> AuxiliaryValue:
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input divisor must be positive finite FP32")
        return cls("fp32", (), raw)


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    produce: Callable[[TensorOutput], None]
    auxiliaries: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    target: TensorSpec
    inputs: tuple[MethodInput, ...]
    policies: Mapping[UseKey, str]
    parameters: Mapping[str, object]
    device: str = "cuda"
    rows_per_chunk: int = 512
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)
    source_offsets: tuple[int, ...] = field(init=False)
    # Parent row where each input starts, when every input is a complete row block of the parent;
    # None otherwise, and encoded_rows then names the first input that is not.
    row_offsets: tuple[int, ...] | None = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))
        rows = [0]
        for item in self.inputs:
            shape = item.source.shape
            if len(self.target.shape) != 2 or len(shape) != 2 or shape[1] != self.target.shape[1]:
                rows = None
                break
            rows.append(rows[-1] + shape[0])
        object.__setattr__(self, "row_offsets", None if rows is None else tuple(rows))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def require_input_shape(self, shape: tuple[int, ...]) -> None:
        if self.source.shape != tuple(shape):
            raise ValueError(f"source shape {self.source.shape} differs from {shape}")

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def values(self, begin: int, end: int) -> torch.Tensor:
        """Read the ordered logical inputs as one C-order parent element sequence."""
        pieces = []
        if not 0 <= begin <= end <= self.source_offsets[-1]:
            raise ValueError(f"{self.target.id}: requested values exceed method inputs")
        index = bisect_right(self.source_offsets, begin) - 1
        while begin < end:
            high = min(end, self.source_offsets[index + 1])
            pieces.append(
                self.inputs[index].source.values(
                    begin - self.source_offsets[index],
                    high - self.source_offsets[index],
                )
            )
            begin = high
            index += 1
        if not pieces:
            return torch.empty(0)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def encoded_rows(self, begin: int, end: int) -> EncodedRows:
        offsets = self.row_offsets
        if offsets is None:
            for item in self.inputs:
                shape = item.source.shape
                if len(shape) != 2 or shape[1] != self.target.shape[1]:
                    raise ValueError(f"{item.parameter}: encoded grouping requires complete rows")
        if not 0 <= begin < end <= offsets[-1]:
            raise ValueError("encoded method input range is invalid")
        pieces = []
        # A stacked bank has one input per expert; find the first overlapping one directly
        # instead of walking every input for every chunk.
        index = bisect_right(offsets, begin) - 1
        while offsets[index] < end:
            item = self.inputs[index]
            cursor = offsets[index]
            low, high = max(begin, cursor), min(end, offsets[index + 1])
            if low < high:
                if item.source.read_encoded is None:
                    raise ValueError(f"{item.parameter}: encoded rows are unavailable")
                pieces.append(item.source.read_encoded(low - cursor, high - cursor))
            index += 1
        first = pieces[0]
        if any((p.format, p.weight_divisor) != (first.format, first.weight_divisor) for p in pieces):
            raise ValueError("encoded inputs cannot share one parent format/divisor")
        if len(pieces) == 1:
            return first
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
        )


Method = Callable[[PrepareRequest], PreparedMethod]
_DIRECT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}


def _preflight(request: PrepareRequest, *, values: bool = True) -> None:
    if request.parameters:
        raise ValueError(f"{request.target.id}: this method accepts no numerical parameters")
    if prod(request.target.shape) != sum(prod(item.source.shape) for item in request.inputs):
        raise ValueError(f"{request.target.id}: parent and logical source counts differ")
    if type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    chunk = request.rows_per_chunk * (request.target.shape[-1] if len(request.target.shape) > 1 else 1)
    elements = prod(request.target.shape)

    def produce(output):
        for begin in range(0, elements, chunk):
            values = request.values(begin, min(elements, begin + chunk))
            if dtype == torch.int32 and (bool((values < -(1 << 31)).any()) or bool((values > (1 << 31) - 1).any())):
                raise ValueError("int32 conversion source is outside the representable range")
            values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    if not isinstance(get_format(request.target.format), QuantFormat) or len(request.target.shape) != 2:
        raise ValueError("grouped_absmax requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError("grouped_absmax source must provide floating-point values")
            encoded = quantize_matrix(values, request.target.format, device=request.device)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k).to(torch.bfloat16)
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


IMPORT_CHUNK_BYTES = 64 * 1024 * 1024


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve the current FP8/NVFP4/grouped-integer source codes, scales and weight divisors.

    A parent with one divisor requires its sources to agree on it. A stacked NVFP4 parent
    (``target.divisors`` > 1, one equal row block per source) keeps every source's divisor, and
    every source's activation divisor stays on that source's own Uses.
    """
    grouped = isinstance(get_format(request.target.format), QuantFormat)
    if (request.target.format not in ("nvfp4", "fp8_e4m3fn_row_bf16") and not grouped) or len(
        request.target.shape
    ) != 2:
        raise ValueError("import_encoded requires a known encoded matrix target")
    _preflight(request, values=False)
    stacked = request.target.divisors > 1
    if stacked and request.target.divisors != len(request.inputs):
        raise ValueError("a stacked NVFP4 parent stores one divisor per source")
    auxiliaries = {}
    weight_divisor = None
    for item in request.inputs:
        source = item.source
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = source.read_encoded(0, 1)
        if first.format != request.target.format:
            raise ValueError(f"{item.parameter}: source {first.format} differs from target {request.target.format}")
        if first.format == "nvfp4":
            if stacked:
                pass
            elif weight_divisor is None:
                weight_divisor = first.weight_divisor
            elif first.weight_divisor != weight_divisor:
                raise ValueError("NVFP4 weight divisors differ; choose separate parents or a conversion method")
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    auxiliaries[key] = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(f"{parameter}: supply an activation divisor for AllowA4")
                    auxiliaries[key] = AuxiliaryValue.activation_divisor(source.input_divisor())
    n, k = request.target.shape
    # Imported words are copied, not computed, so the chunk is sized by bytes: rows_per_chunk
    # alone gives the 160-byte rows of an n-gram table tiny chunks and millions of reads. Chunk
    # boundaries do not change the stored bytes.
    if grouped:  # signed codes + binary16 multipliers (q3_g128_fp16 / q4_g64_fp16 sources)
        spec = get_format(request.target.format)
        row_bytes = k * spec.bits // 8 + 2 * (k // spec.group_size)
    else:
        row_bytes = k // 2 + k // 16 if request.target.format == "nvfp4" else k + 2
    chunk = max(request.rows_per_chunk, IMPORT_CHUNK_BYTES // row_bytes)
    if request.target.format == "nvfp4":
        chunk = max(128, chunk // 128 * 128)

    def produce(output):
        # A stacked plane starts a new run of chunks at every source boundary: a row block
        # carries one divisor, so a chunk may not straddle two sources.
        bounds = list(request.row_offsets[1:]) if stacked else [n]
        cursor = 0
        for edge in bounds:
            for begin in range(cursor, edge, chunk):
                words = request.encoded_rows(begin, min(edge, begin + chunk))
                output.write_codes(begin, words.codes, words.scales, words.weight_divisor)
            cursor = edge

    return request.job(produce=produce, auxiliaries=auxiliaries)


NVFP4_MSE_TARGETS = (E2M1_MAX, 5.5, 5.0, 4.5, 4.0)


def _nvfp4_quantized(request: PrepareRequest, targets: tuple[float, ...], *, stacked: bool = False) -> PreparedMethod:
    if request.target.format != "nvfp4" or len(request.target.shape) != 2:
        raise ValueError("NVFP4 quantization requires an nvfp4 matrix target")
    if stacked:
        if request.target.divisors != len(request.inputs) or request.row_offsets is None:
            raise ValueError("stacked NVFP4 quantization stores one divisor per complete source matrix")
    elif request.target.divisors != 1:
        raise ValueError("NVFP4 quantization writes one divisor per parent")
    _preflight(request)
    n, k = request.target.shape
    auxiliaries = {}
    for item in request.inputs:
        for parameter, input_name in item.uses:
            key = (parameter, input_name, "activation_input_divisor")
            if key in request.auxiliary_overrides:
                auxiliaries[key] = request.auxiliary_overrides[key]
            elif request.policies[(parameter, input_name)] == "AllowA4":
                raise ValueError(
                    f"{parameter}: AllowA4 needs an activation divisor; supply one with "
                    "recipe.use(..., auxiliaries=...) or use A16Only"
                )
    rows = max(128, request.rows_per_chunk // 128 * 128)
    # Row blocks quantised against one divisor: the whole parent, or each source of a stack.
    blocks = list(zip(request.row_offsets[:-1], request.row_offsets[1:])) if stacked else [(0, n)]

    def produce(output):
        for first, last in blocks:
            amax = 0.0
            for begin in range(first, last, rows):
                end = min(last, begin + rows)
                values = request.values(begin * k, end * k)
                if not values.dtype.is_floating_point:
                    raise TypeError("NVFP4 quantization source must be floating point")
                amax = max(amax, float(values.abs().max()))
            divisor = weight_divisor(amax, targets)
            for begin in range(first, last, rows):
                end = min(last, begin + rows)
                values = request.values(begin * k, end * k).reshape(end - begin, k)
                words = quantize_nvfp4(values.to(request.device), divisor, targets)
                output.write_codes(begin, words.codes, words.scales, words.weight_divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def nvfp4_absmax(request: PrepareRequest) -> PreparedMethod:
    """NVFP4 with every 16-value group's largest magnitude at the largest E2M1 code."""
    return _nvfp4_quantized(request, (E2M1_MAX,))


def nvfp4_mse(request: PrepareRequest) -> PreparedMethod:
    """NVFP4 with each group's scale chosen among five targets by least squared error."""
    return _nvfp4_quantized(request, NVFP4_MSE_TARGETS)


def nvfp4_mse_stacked(request: PrepareRequest) -> PreparedMethod:
    """nvfp4_mse with one divisor per source matrix of a stacked parent (a routed-expert bank).

    Each source is quantised alone, exactly as nvfp4_mse would quantise it as its own parent, and
    keeps its divisor word; the recipe planner gives such a parent one divisor per source.
    """
    return _nvfp4_quantized(request, NVFP4_MSE_TARGETS, stacked=True)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
    "nvfp4_absmax": nvfp4_absmax,
    "nvfp4_mse": nvfp4_mse,
    "nvfp4_mse_stacked": nvfp4_mse_stacked,
}
