"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import Executor
from dataclasses import dataclass, field
from math import isfinite, prod
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

from .quantization.fp8_block import quantize_blocks
from .quantization.fp8_row import quantize_bf16_rows
from .quantization.groupwise import quantize_matrix
from .quantization.stored_scale import quantize_stored_scale
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]
CHUNK_BYTES = 16 * 1024 * 1024
SOURCE_CACHE_BYTES = 2 * 1024 * 1024 * 1024


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
    rows_per_chunk: int | None = None
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_cache_bytes: int = SOURCE_CACHE_BYTES
    reader: Executor | None = None
    source_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def chunk_rows(self, row_bytes: int, *, alignment: int = 1) -> int:
        rows = self.rows_per_chunk or max(1, CHUNK_BYTES // row_bytes)
        return max(alignment, rows // alignment * alignment)

    def read_chunks(self, read: Callable, count: int, chunk: int):
        """Keep one ordered source read ahead of computation, bounded by one chunk."""
        pending = None
        for begin in range(0, count, chunk):
            end = min(count, begin + chunk)
            if self.reader is None:
                yield begin, end, read(begin, end)
                continue
            if pending is None:
                pending = self.reader.submit(read, begin, end)
            value = pending.result()
            pending = (
                self.reader.submit(read, end, min(count, end + chunk))
                if end < count
                else None
            )
            yield begin, end, value

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
        pieces = []
        k = self.target.shape[-1]
        if not 0 <= begin < end <= self.source_offsets[-1] // k:
            raise ValueError("encoded method input range is invalid")
        index = bisect_right(self.source_offsets, begin * k) - 1
        while begin < end:
            item = self.inputs[index]
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != k:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            cursor = self.source_offsets[index] // k
            high = min(end, cursor + source.shape[0])
            if source.read_encoded is None:
                raise ValueError(f"{item.parameter}: encoded rows are unavailable")
            pieces.append(source.read_encoded(begin - cursor, high - cursor))
            if pieces[-1].format != self.target.format:
                raise ValueError(
                    f"{item.parameter}: encoded rows differ from target format"
                )
            begin = high
            index += 1
        first = pieces[0]
        if any(
            (p.format, p.weight_divisor) != (first.format, first.weight_divisor)
            for p in pieces
        ):
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
        raise ValueError(
            f"{request.target.id}: this method accepts no numerical parameters"
        )
    if prod(request.target.shape) != sum(
        prod(item.source.shape) for item in request.inputs
    ):
        raise ValueError(
            f"{request.target.id}: parent and logical source counts differ"
        )
    if request.rows_per_chunk is not None and (
        type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0
    ):
        raise ValueError("rows_per_chunk must be positive")
    if type(request.source_cache_bytes) is not int or request.source_cache_bytes < 0:
        raise ValueError("source_cache_bytes must be nonnegative")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    k = request.target.shape[-1] if len(request.target.shape) > 1 else 1
    chunk = request.chunk_rows(k * dtype.itemsize) * k
    elements = prod(request.target.shape)

    def produce(output):
        for begin, _, values in request.read_chunks(request.values, elements, chunk):
            if dtype == torch.int32 and (
                bool((values < -(1 << 31)).any())
                or bool((values > (1 << 31) - 1).any())
            ):
                raise ValueError(
                    "int32 conversion source is outside the representable range"
                )
            values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError("grouped_absmax requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape
    chunk = request.chunk_rows(k * 4)

    def produce(output):
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = values.reshape((end - begin) // k, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin // k, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape
    chunk = request.chunk_rows(k * 4)

    def produce(output):
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = (
                values.reshape((end - begin) // k, k)
                .to(device=request.device)
                .to(torch.bfloat16)
            )
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin // k, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_block_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Encode independent 128-by-128 blocks with BF16 scales and E4M3FN codes."""
    if request.target.format != "fp8_e4m3fn_block128x128_bf16" or len(
        request.target.shape
    ) not in (2, 3):
        raise ValueError("fp8_block_maxabs requires a block-scaled FP8 matrix or bank")
    _preflight(request)
    n, k = request.target.shape[-2:]
    matrices = request.target.shape[0] if len(request.target.shape) == 3 else 1
    if matrices <= 0 or n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("block FP8 target matrices require complete 128-by-128 tiles")
    for item in request.inputs:
        shape = item.source.shape
        if (
            len(shape) not in (2, 3)
            or shape[-1] != k
            or shape[-2] % 128
            or any(extent <= 0 for extent in shape)
        ):
            raise ValueError(
                f"{item.parameter}: block FP8 sources require whole 128-row matrices"
            )
    chunk = request.chunk_rows(k * 4, alignment=128)

    def produce(output):
        for matrix in range(matrices):
            offset = matrix * n * k
            read = lambda begin, end: request.values(offset + begin, offset + end)
            for begin, end, values in request.read_chunks(read, n * k, chunk * k):
                encoded = quantize_blocks(
                    values.reshape((end - begin) // k, k).to(device=request.device)
                )
                output.write_codes(
                    begin // k, encoded.codes, encoded.scales, matrix=matrix
                )

    return request.job(produce=produce)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve FP8/NVFP4 words, with independent divisors for bank matrices."""
    block_fp8 = request.target.format == "fp8_e4m3fn_block128x128_bf16"
    if (
        request.target.format
        not in ("nvfp4", "fp8_e4m3fn_row_bf16", "fp8_e4m3fn_block128x128_bf16")
        or len(request.target.shape) not in (2, 3)
        or (
            len(request.target.shape) == 3
            and request.target.format == "fp8_e4m3fn_row_bf16"
        )
    ):
        raise ValueError("import_encoded requires an encoded matrix or bank target")
    _preflight(request, values=False)
    n, k = request.target.shape[-2:]
    matrices = request.target.shape[0] if len(request.target.shape) == 3 else 1
    auxiliaries = {}
    divisors = {}
    cursor = 0
    for item in request.inputs:
        source = item.source
        if len(source.shape) != 2 or source.shape[1] != k:
            raise ValueError(
                f"{item.parameter}: encoded grouping requires complete rows"
            )
        if block_fp8 and source.shape[0] % 128:
            raise ValueError(
                f"{item.parameter}: FP8 source must contain whole 128-row blocks"
            )
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        encoding = source.encoding()
        if encoding != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {encoding} differs from target {request.target.format}"
            )
        if encoding == "nvfp4":
            if source.weight_divisor is None:
                raise ValueError(
                    f"{item.parameter}: NVFP4 source needs a weight divisor"
                )
            divisor = source.weight_divisor()
            for matrix in range(cursor // n, (cursor + source.shape[0] - 1) // n + 1):
                if matrix in divisors and divisors[matrix] != divisor:
                    raise ValueError("NVFP4 weight divisors differ within one matrix")
                divisors[matrix] = divisor
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    value = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    value = AuxiliaryValue.activation_divisor(source.input_divisor())
                else:
                    continue
                if key in auxiliaries and auxiliaries[key] != value:
                    raise ValueError(
                        f"{parameter}: activation divisors differ within one Use"
                    )
                auxiliaries[key] = value
        cursor += source.shape[0]
    chunk = request.chunk_rows(
        k, alignment=128 if request.target.format == "nvfp4" or block_fp8 else 1
    )

    def produce(output):
        for matrix in range(matrices):
            read = lambda begin, end: request.encoded_rows(
                matrix * n + begin, matrix * n + end
            )
            for begin, _, words in request.read_chunks(read, n, chunk):
                output.write_codes(
                    begin,
                    words.codes,
                    words.scales,
                    words.weight_divisor,
                    matrix=matrix,
                )

    return request.job(produce=produce, auxiliaries=auxiliaries)


def stored_scale_mse(request: PrepareRequest) -> PreparedMethod:
    """Two-pass object amax and exhaustive G16 stored-scale MSE encoding."""
    if (
        request.target.format not in ("nf4_g16_e4m3fn", "nvfp4")
        or len(request.target.shape) != 2
    ):
        raise ValueError("stored_scale_mse requires an NF4 or NVFP4 matrix")
    _preflight(request)
    n, k = request.target.shape
    chunk = request.chunk_rows(
        k * 4, alignment=128 if request.target.format == "nvfp4" else 1
    )

    def produce(output):
        maximum = torch.zeros((), dtype=torch.float32, device=request.device)
        cached = [] if n * k * 4 <= request.source_cache_bytes else None
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = values.to(request.device).float()
            maximum = torch.maximum(maximum, values.abs().amax())
            if cached is not None:
                cached.append((begin, end, values))
        maximum = maximum.item()
        if not isfinite(maximum):
            raise ValueError("stored-scale source contains NaN or infinity")
        limit = 448.0 if request.target.format == "nf4_g16_e4m3fn" else 2688.0
        divisor = struct.pack("<f", limit / maximum if maximum else 1.0)
        if not valid_positive_fp32_word(struct.unpack("<I", divisor)[0]):
            raise ValueError("amax produces an invalid FP32 object divisor")
        blocks = (
            cached
            if cached is not None
            else request.read_chunks(request.values, n * k, chunk * k)
        )
        for begin, end, values in blocks:
            values = values.reshape((end - begin) // k, k)
            encoded = quantize_stored_scale(
                values, request.target.format, divisor, device=request.device
            )
            output.write_codes(begin // k, encoded.codes, encoded.scales, divisor)

    return request.job(produce=produce)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "fp8_row_maxabs": fp8_row_maxabs,
    "fp8_block_maxabs": fp8_block_maxabs,
    "import_encoded": import_encoded,
    "stored_scale_mse": stored_scale_mse,
}
