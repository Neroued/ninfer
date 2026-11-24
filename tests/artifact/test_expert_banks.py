import random
import struct

import pytest
import torch

from tools.artifact.codecs.fp8_block import (
    decode_fp8_block_scaled_words,
    dequantize_fp8_block_scaled,
    encode_fp8_block_scaled,
)
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.layouts import encoded_size


def test_nvfp4_bank_keeps_independent_scales_and_divisors():
    shape = (3, 256, 128)
    rng = random.Random(20261012)
    codes = torch.tensor(
        [rng.randrange(256) for _ in range(49152)], dtype=torch.uint8
    ).reshape(3, 256, 64)
    scales = torch.tensor(
        [rng.randrange(127) for _ in range(6144)], dtype=torch.uint8
    ).reshape(3, 256, 8)
    divisors = struct.pack("<fff", 2.0, 3.0, 5.0)
    encoded = encode_nvfp4(codes, scales, divisors, shape)
    expected = bytearray(codes.numpy().tobytes() + bytes(6144) + divisors)
    for expert in range(3):
        for row in range(256):
            for group in range(8):
                tile = (row // 128 * 2 + group // 4) * 512
                within = row % 32 * 16 + row % 128 // 32 * 4 + group % 4
                expected[49152 + expert * 2048 + tile + within] = int(
                    scales[expert, row, group]
                )
    assert encoded == expected
    q, s, d = decode_nvfp4_words(expected, shape)
    assert torch.equal(q, codes)
    assert torch.equal(s, scales)
    assert d.tolist() == [2.0, 3.0, 5.0]
    assert encoded_size("batched_block_scale_k16_m128x4_v1", "nvfp4", shape) == 55308
    with pytest.raises(ValueError, match="rank 2"):
        encoded_size("block_scale_k16_m128x4_v1", "nvfp4", shape)
    with pytest.raises(TypeError, match="per matrix"):
        encode_nvfp4(codes, scales, struct.pack("<f", 2), shape)
    with pytest.raises(ValueError, match="finite and positive"):
        encode_nvfp4(codes, scales, struct.pack("<fff", 2, 0, 5), shape)


@pytest.mark.parametrize("bank", [False, True])
def test_fp8_blocks_have_exact_words_and_two_dimensional_scale_ownership(bank):
    shape = (2, 256, 384) if bank else (256, 384)
    codes = torch.full(shape, 0x38, dtype=torch.uint8)
    codes[..., 127, 128] = 0xB8
    codes[..., 128, 127] = 0x01
    codes[..., 255, 383] = 0x7E
    scales = torch.tensor([1, 2, 3, 4, 5, 6] * (2 if bank else 1), dtype=torch.bfloat16)
    if bank:
        scales[6:] *= 2
    scales = scales.reshape((*shape[:-2], 2, 3))
    scale_words = scales.view(torch.int16).flatten().tolist()
    expected_bytes = codes.numpy().tobytes() + struct.pack(
        "<" + "H" * len(scale_words), *scale_words
    )
    assert encode_fp8_block_scaled(codes, scales, shape) == expected_bytes
    q, s = decode_fp8_block_scaled_words(expected_bytes, shape)
    assert torch.equal(q, codes)
    assert torch.equal(s.view(torch.int16), scales.view(torch.int16))
    expected = torch.ones(shape)
    expected[..., 127, 128] = -1
    expected[..., 128, 127] = 2**-9
    expected[..., 255, 383] = 448
    for row in range(256):
        for column in range(384):
            expected[..., row, column] *= scales[..., row // 128, column // 128].float()
    assert torch.equal(dequantize_fp8_block_scaled(expected_bytes, shape), expected)


def test_fp8_block_word_validation():
    codes = torch.zeros((128, 128), dtype=torch.uint8)
    scales = torch.zeros((1, 1), dtype=torch.bfloat16)
    encode_fp8_block_scaled(codes, scales, (128, 128))
    codes[0, 0] = 0x38
    with pytest.raises(ValueError, match="zero block scale"):
        encode_fp8_block_scaled(codes, scales, (128, 128))
    scales.fill_(1)
    codes[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite E4M3FN"):
        encode_fp8_block_scaled(codes, scales, (128, 128))
    codes[0, 0] = 0x38
    scales.fill_(-1)
    with pytest.raises(ValueError, match="nonnegative finite"):
        encode_fp8_block_scaled(codes, scales, (128, 128))
