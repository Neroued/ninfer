from __future__ import annotations

import math
import struct
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pytest
import torch

from tools.artifact.codecs.fp8_block import decode_fp8_block_scaled_words
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import MethodInput, PrepareRequest, fp8_block_maxabs
from tools.convert.model import Model, Parameter
from tools.convert.quantization.fp8_block import quantize_blocks
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _positive_fp8(word: int) -> float:
    exponent, fraction = divmod(word, 8)
    return (
        fraction * 2.0**-9
        if exponent == 0
        else (1.0 + fraction / 8.0) * 2.0 ** (exponent - 7)
    )


_FP8_VALUES = tuple(_positive_fp8(word) for word in range(127))


def _reference(block: torch.Tensor) -> tuple[torch.Tensor, int]:
    values = block.float().flatten().tolist()
    maximum = max(map(abs, values))
    if maximum == 0:
        return torch.zeros(block.shape, dtype=torch.uint8), 0
    raw_scale = _f32(maximum / 448.0)
    lower = struct.unpack("<I", struct.pack("<f", raw_scale))[0] >> 16
    scale_word = min(
        (lower, lower + 1),
        key=lambda word: (
            abs(struct.unpack("<f", struct.pack("<I", word << 16))[0] - raw_scale),
            word & 1,
        ),
    )
    scale_word = max(scale_word, 1)
    scale = struct.unpack("<f", struct.pack("<I", scale_word << 16))[0]
    inverse = _f32(1.0 / scale)

    @lru_cache(None)
    def encode(raw: bytes) -> int:
        value = struct.unpack("<f", raw)[0]
        normalized = _f32(value * inverse)
        magnitude = min(abs(normalized), 448.0)
        word = min(
            range(127), key=lambda code: (abs(_FP8_VALUES[code] - magnitude), code & 1)
        )
        return word | (0x80 if math.copysign(1.0, normalized) < 0 else 0)

    codes = [encode(struct.pack("<f", value)) for value in values]
    return torch.tensor(codes, dtype=torch.uint8).reshape(block.shape), scale_word


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_block_fp8_matches_exact_codec_reference(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = torch.zeros((256, 256), dtype=torch.float32)
    midpoints = [(_FP8_VALUES[i] + _FP8_VALUES[i + 1]) / 2 for i in range(126)]
    patterns = (
        [448.0, 0.0, -0.0, 1.0624, 1.0626, *midpoints, *(-x for x in midpoints)],
        [449.75, 1.0625, -1.1875, 0.0, -0.0],
        [0.0, -0.0],
        [364.0, 0.01031494140625, -0.01031494140625, 0.0, -0.0],
    )
    expected_codes = torch.empty_like(source, dtype=torch.uint8)
    expected_scales = torch.empty((2, 2), dtype=torch.int16)
    for index, pattern in enumerate(patterns):
        row, column = divmod(index, 2)
        tile = torch.tensor(pattern * (16384 // len(pattern) + 1))[:16384].reshape(
            128, 128
        )
        tile = tile.to(dtype)
        source[row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128] = tile
        codes, scale = _reference(tile)
        expected_codes[
            row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128
        ] = codes
        expected_scales[row, column] = scale
    actual = quantize_blocks(source.to(device=device, dtype=dtype))
    assert torch.equal(actual.codes.cpu(), expected_codes)
    assert torch.equal(actual.scales.cpu().view(torch.int16), expected_scales)
    # Reciprocal-then-multiply rounds above this E4M3 midpoint; direct division
    # lands exactly on the tie and would incorrectly encode 0x06 instead.
    assert actual.codes[128, 129].item() == 0x07


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), 2.0**-133]
)
def test_block_fp8_rejects_nonfinite_encoding(device, value):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = torch.zeros((128, 128), dtype=torch.bfloat16, device=device)
    source[0, 0] = value
    with pytest.raises(ValueError, match="finite|infinity"):
        quantize_blocks(source)


def test_block_fp8_streaming_preserves_expert_and_matrix_boundaries(tmp_path):
    model = Model({"text": {"config": {}}})
    banks = {}
    for role, factor in (("gate", 1.0), ("up", 7.0)):
        bank = torch.empty((2, 256, 256), dtype=torch.bfloat16)
        for expert in range(2):
            for row in range(2):
                for column in range(2):
                    bank[
                        expert,
                        row * 128 : (row + 1) * 128,
                        column * 128 : (column + 1) * 128,
                    ] = (
                        factor * (expert + 1) * (row + 1) / (column + 1)
                    )
        banks[role] = bank
        model.add(Parameter(role, tuple(bank.shape), array_source(bank, role)))
    recipe = Recipe(model)
    parts = []
    for expert in range(2):
        for role in banks:
            parts.append(
                (
                    role,
                    expert * 256 * 256,
                    (expert + 1) * 256 * 256,
                    array_source(banks[role][expert], role),
                )
            )
    recipe.parent(
        parts,
        shape=(2, 512, 256),
        format="fp8_e4m3fn_block128x128_bf16",
        method="fp8_block_maxabs",
    )
    payloads = []
    with ThreadPoolExecutor(max_workers=1) as reader:
        for chunk in (1, 129, 387, 1024):
            prepared = recipe.prepare(device="cpu", rows_per_chunk=chunk, reader=reader)
            path = tmp_path / f"block-{chunk}.ninfer"
            with ArtifactWriter(
                path,
                [w.spec for w in prepared.weights],
                components=model.components,
                bindings=prepared.bindings,
                uses=prepared.uses,
            ) as writer:
                for job in prepared.weights:
                    job.prepared.produce(TensorOutput(writer, job.spec.id))
            with Artifact(path) as artifact:
                payloads.append(artifact.read_object(prepared.weights[0].spec.id))
    assert all(payload == payloads[0] for payload in payloads)
    codes, scales = decode_fp8_block_scaled_words(payloads[0], (2, 512, 256))
    for expert in range(2):
        for position, role in enumerate(banks):
            for row in range(2):
                for column in range(2):
                    expected, scale = _reference(
                        banks[role][
                            expert,
                            row * 128 : (row + 1) * 128,
                            column * 128 : (column + 1) * 128,
                        ]
                    )
                    start = position * 256 + row * 128
                    assert torch.equal(
                        codes[
                            expert,
                            start : start + 128,
                            column * 128 : (column + 1) * 128,
                        ],
                        expected,
                    )
                    assert (
                        scales.view(torch.int16)[
                            expert, position * 2 + row, column
                        ].item()
                        == scale
                    )


def test_block_fp8_rejects_incomplete_source_matrix():
    sources = [array_source(torch.ones((64, 128)), name) for name in ("gate", "up")]
    request = PrepareRequest(
        target=TensorSpec(
            "gu", (128, 128), "fp8_e4m3fn_block128x128_bf16", "block_scale_m128_k128_v1"
        ),
        inputs=tuple(MethodInput(source.label, source, ()) for source in sources),
        policies={},
        parameters={},
        device="cpu",
    )
    with pytest.raises(ValueError, match="128-row"):
        fp8_block_maxabs(request)
