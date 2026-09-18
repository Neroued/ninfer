from __future__ import annotations

import json
import struct

from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import (
    decode_fp8_row_scaled_words,
    encode_fp8_row_scaled,
)
from tools.artifact.reader import Artifact
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.quantization import fp8_row
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource, tensor_source


def test_fp8_quantization_zero_rows_ties_and_signed_zero() -> None:
    source = torch.tensor(
        (
            (0.0, -0.0, 0.0, -0.0, 0.0),
            (448.0, 1.0625, 1.1875, -1.0625, -1.1875),
            (1.0, -0.0, 0.5, -1.0, -0.5),
        ),
        dtype=torch.bfloat16,
    )
    quantized = fp8_row.quantize_bf16_rows(source)
    assert quantized.scales.view(torch.int16).tolist() == [0x0000, 0x3F80, 0x3B12]
    assert quantized.codes[0].tolist() == [0, 0, 0, 0, 0]
    assert quantized.codes[1].tolist() == [0x7E, 0x38, 0x3A, 0xB8, 0xBA]
    assert quantized.codes[2, 1].item() == 0x80

    payload = encode_fp8_row_scaled(quantized.codes, quantized.scales, source.shape)
    codes, scales = decode_fp8_row_scaled_words(payload, source.shape)
    assert torch.equal(codes, quantized.codes)
    assert torch.equal(scales.view(torch.int16), quantized.scales.view(torch.int16))


def test_streamed_fp8_conversion_matches_known_words(tmp_path) -> None:
    name = "model.language_model.embed_tokens.weight"
    source = torch.tensor(
        [
            [448.0, 1.0625, -1.1875, 0.0],
            [0.0, -0.0, 0.0, -0.0],
            [2.0, -2.0, 1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    shard = "model.safetensors"
    save_file({name: source}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard}}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text("{}")
    model = Model({"text": {"config": {}}})
    with SafetensorsSource(tmp_path) as reader:
        model.add(
            Parameter(
                "embedding",
                tuple(source.shape),
                tensor_source(reader, name, tuple(source.shape)),
            )
        )
        recipe = Recipe(model)
        recipe.assign(
            "embedding", format="fp8_e4m3fn_row_bf16", method="fp8_row_maxabs"
        )
        path = tmp_path / "embedding.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=2)
    with Artifact(path) as artifact:
        object_id = artifact.directory.bindings["embedding"]["object"]
        codes = bytes([0x7E, 0x38, 0xBA, 0, 0, 0, 0, 0, 0x7E, 0xFE, 0x76, 0xF6])
        expected = codes + bytes(244) + struct.pack("<3H", 0x3F80, 0, 0x3B92)
        assert artifact.read_object(object_id) == expected
