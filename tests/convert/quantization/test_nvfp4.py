from __future__ import annotations

import struct

import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.quantization import nvfp4
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _decode(words: nvfp4.Nvfp4Words, divisor: bytes) -> torch.Tensor:
    low, high = words.codes & 15, words.codes >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(words.codes.shape[0], -1)
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1))
    values = table[nibbles.long()]
    scales = words.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return values * scales / struct.unpack("<f", divisor)[0]


def test_divisor_uses_the_full_scale_range() -> None:
    assert struct.unpack("<f", nvfp4.weight_divisor_for(0.5))[0] == 5376.0
    assert struct.unpack("<f", nvfp4.weight_divisor_for(0.0))[0] == 1.0


def test_e2m1_ties_round_to_even_code_word() -> None:
    magnitude = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0])
    assert nvfp4._e2m1_magnitude_index(magnitude).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 7]


def test_blocks_round_trip_within_one_scale_unit() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(4, 64, generator=generator) * 0.05
    divisor = nvfp4.weight_divisor_for(float(values.abs().max()))
    words = nvfp4.quantize_rows(values, divisor)
    decoded = _decode(words, divisor)
    block_max = values.reshape(4, 4, 16).abs().amax(dim=2).repeat_interleave(16, 1)
    # The widest E2M1 step (4 to 6) bounds the error at one scale unit; E4M3 adds ~6%.
    assert bool(((decoded - values).abs() <= block_max / 6.0 * 1.07 + 1e-6).all())
    assert int(words.scales.max()) <= 0x7E


def test_zero_blocks_and_signed_values() -> None:
    values = torch.zeros(1, 32)
    values[0, 16:] = torch.tensor([-1.0, 1.0] * 8)
    divisor = nvfp4.weight_divisor_for(1.0)
    words = nvfp4.quantize_rows(values, divisor)
    assert words.scales[0, 0].item() == 0
    assert words.codes[0, :8].tolist() == [0] * 8
    decoded = _decode(words, divisor)
    assert torch.allclose(decoded[0, 16:], values[0, 16:], atol=1e-3)
    assert bool((decoded[0, :16] == 0).all())


def test_streamed_conversion_matches_direct_quantization(tmp_path) -> None:
    generator = torch.Generator().manual_seed(11)
    source = (torch.randn(256, 128, generator=generator) * 0.02).to(torch.bfloat16)
    model = Model({"text": {"config": {}}})
    model.add(
        Parameter(
            "text/w", (256, 128), array_source(source, "w"), None, ("text/x",)
        )
    )
    recipe = Recipe(model)
    recipe.assign("text/w", format="nvfp4", method="nvfp4_absmax")
    convert(model, recipe, tmp_path / "m.ninfer", device="cpu", rows_per_chunk=128)
    with Artifact(tmp_path / "m.ninfer") as artifact:
        object_id = artifact.directory.bindings["text/w"]["object"]
        codes, scales, divisor = decode_nvfp4_words(
            artifact.read_object(object_id), (256, 128)
        )
    expected_divisor = nvfp4.weight_divisor_for(float(source.float().abs().max()))
    expected = nvfp4.quantize_rows(source, expected_divisor)
    assert struct.pack("<f", float(divisor)) == expected_divisor
    assert torch.equal(codes, expected.codes)
    assert torch.equal(scales, expected.scales)
