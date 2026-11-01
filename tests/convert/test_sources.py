from __future__ import annotations

import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.logical import select_rows
from tools.convert.sources.modelopt_nvfp4 import ModelOptNvfp4


def test_source_failure_does_not_publish_conversion(tmp_path):
    from tools.convert.model import Model, Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.logical import LogicalSource

    def read(begin, end):
        if begin >= 32:
            raise OSError("source read failed")
        return torch.ones(end - begin, dtype=torch.bfloat16)

    model = Model({"text": {"config": {}}})
    model.add(Parameter("w", (8, 16), LogicalSource((8, 16), "fault", read)))
    with pytest.raises(ValueError, match="source read failed"):
        convert(
            model,
            Recipe(model),
            tmp_path / "failed.ninfer",
            device="cpu",
            rows_per_chunk=2,
        )
    assert not list(tmp_path.iterdir())


def test_nvfp4_source_preserves_words_and_decodes_independently(tmp_path):
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight_packed": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
            "proj.input_global_scale": torch.tensor([1.5], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        expected = torch.tensor(
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
        expected = torch.stack((expected / 2, expected))
        assert torch.equal(source.values().reshape(2, 16), expected)
        assert source.input_divisor() == struct.pack("<f", 1.5)
        assert source.values(16, 16).numel() == 0


def test_row_fp8_source_and_reordered_encoded_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": scales,
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_bf16")
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        reordered = select_rows(source, ((1, 2), (0, 1)))
        words = reordered.read_encoded(0, 2)
        assert torch.equal(words.codes, codes.flip(0))
        assert torch.equal(words.scales, scales.flatten().flip(0))


def test_modelopt_preserves_words_and_converts_multiplier_to_divisor(tmp_path):
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    scales = torch.tensor([[0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_scale_2": torch.tensor(0.375),
            "proj.input_scale": torch.tensor(0.125),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = ModelOptNvfp4(store).matrix("proj", (1, 16))
        words = source.read_encoded(0, 1)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 1 / 0.375)
        assert source.input_divisor() == struct.pack("<f", 8.0)
        expected = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float64,
        )
        expected *= 2.0 / struct.unpack("<f", words.weight_divisor)[0]
        assert torch.equal(source.values(), expected)
