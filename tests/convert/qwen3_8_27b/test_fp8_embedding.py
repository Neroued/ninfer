from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.layouts import decode_fp8_row_scaled_words
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_8_27b import fp8_embedding


def test_embedding_profile_zero_rows_ties_and_signed_zero() -> None:
    source = torch.tensor(
        (
            (0.0, -0.0, 0.0, -0.0, 0.0),
            (448.0, 1.0625, 1.1875, -1.0625, -1.1875),
            (1.0, -0.0, 0.5, -1.0, -0.5),
        ),
        dtype=torch.bfloat16,
    )
    quantized = fp8_embedding.quantize_bf16_rows(source)
    assert quantized.scales.view(torch.int16).tolist() == [0x0000, 0x3F80, 0x3B12]
    assert quantized.codes[0].tolist() == [0, 0, 0, 0, 0]
    assert quantized.codes[1].tolist() == [0x7E, 0x38, 0x3A, 0xB8, 0xBA]
    assert quantized.codes[2, 1].item() == 0x80

    payload = fp8_embedding.encode_bf16_rows(source)
    codes, scales = decode_fp8_row_scaled_words(payload, source.shape)
    assert torch.equal(codes, quantized.codes)
    assert torch.equal(
        scales.view(torch.int16), quantized.scales.view(torch.int16)
    )


def test_embedding_profile_rejects_wrong_signature_and_nonfinite_source() -> None:
    with pytest.raises(TypeError, match="rank-two BF16"):
        fp8_embedding.quantize_bf16_rows(torch.ones(2, 3))
    with pytest.raises(TypeError, match="rank-two BF16"):
        fp8_embedding.quantize_bf16_rows(
            torch.ones(3, dtype=torch.bfloat16)
        )
    source = torch.tensor([[1.0, float("inf")]], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="NaN or infinity"):
        fp8_embedding.quantize_bf16_rows(source)


def test_streamed_reader_payload_matches_in_memory_encoder(tmp_path) -> None:
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
    with ShardReader(tmp_path) as reader:
        streamed = b"".join(
            fp8_embedding.iter_reader_payload(
                reader, name, source.shape, rows_per_chunk=2
            )
        )
    assert streamed == fp8_embedding.encode_bf16_rows(source)
