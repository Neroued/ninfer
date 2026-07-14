from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    RowPlanes,
    assemble_row_planes,
    decode_direct,
    decode_row_split_codes,
    dequantize_row_split,
    encode_direct,
    encode_row_split,
    encoded_size,
    gather_row_planes,
    row_split_geometry,
    split_row_planes,
)


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "BF16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "FP32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "I32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "BF16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


def test_row_split_geometry_and_encoded_size_are_derived_from_format_and_shape():
    geometry = row_split_geometry("Q5G64_F16S", (2, 130))
    assert (
        geometry.k_pad,
        geometry.groups_per_row,
        geometry.base_bytes,
        geometry.high_offset,
        geometry.high_bytes,
        geometry.scale_offset,
        geometry.scale_bytes,
        geometry.payload_bytes,
    ) == (256, 4, 256, 256, 64, 512, 16, 528)
    assert encoded_size("row-split-k128-v1", "Q5G64_F16S", (2, 130)) == 528

    q4 = row_split_geometry("Q4G64_F16S", (1, 4304))
    w8 = row_split_geometry("W8G32_F16S", (1, 4304))
    assert (q4.k_pad, q4.groups_per_row, q4.base_row_bytes, q4.high_row_bytes) == (
        4352,
        68,
        2176,
        0,
    )
    assert (w8.k_pad, w8.groups_per_row, w8.base_row_bytes, w8.high_row_bytes) == (
        4352,
        136,
        4352,
        0,
    )


@pytest.mark.parametrize(
    ("format_name", "k", "prefix", "base_prefix", "high_prefix"),
    [
        ("Q4G64_F16S", 65, (-8, -7, -1, 0, 1, 7), b"\x98\x0f\x71", b""),
        ("Q5G64_F16S", 65, (-16, -15, -1, 0, 1, 15), b"\x10\x0f\xf1", b"\x07"),
        (
            "Q6G64_F16S",
            65,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
        ),
        ("W8G32_F16S", 33, (-127, -1, 0, 1, 127), b"\x81\xff\x00\x01\x7f", b""),
    ],
)
def test_row_split_plane_bit_order_and_round_trip(
    format_name, k, prefix, base_prefix, high_prefix
):
    geometry = row_split_geometry(format_name, (1, k))
    group_size = geometry.k_pad // geometry.groups_per_row
    codes = torch.zeros((1, geometry.groups_per_row, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, geometry.groups_per_row), dtype=torch.float16)
    scales[0, 0] = 1.5
    scales[0, 1] = 0.25

    payload = encode_row_split(codes, scales, format_name, (1, k))
    assert len(payload) == geometry.payload_bytes
    planes = split_row_planes(payload, geometry)
    assert bytes(planes.base[: len(base_prefix)]) == base_prefix
    assert bytes(planes.high[: len(high_prefix)]) == high_prefix
    assert payload[geometry.base_bytes : geometry.high_offset] == bytes(
        geometry.high_offset - geometry.base_bytes
    )
    assert payload[
        geometry.high_offset + geometry.high_bytes : geometry.scale_offset
    ] == bytes(geometry.scale_offset - geometry.high_offset - geometry.high_bytes)

    decoded_scales, decoded_codes = decode_row_split_codes(
        payload, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)


def test_consecutive_views_arbitrary_gathers_and_standalone_assembly():
    format_name = "Q5G64_F16S"
    shape = (4, 130)
    geometry = row_split_geometry(format_name, shape)
    codes = (
        torch.arange(geometry.n * geometry.groups_per_row * 64, dtype=torch.int32)
        .remainder(32)
        .sub(16)
        .to(torch.int8)
        .reshape(geometry.n, geometry.groups_per_row, 64)
    )
    codes.reshape(geometry.n, geometry.k_pad)[:, geometry.k :] = 0
    scales = torch.tensor(
        [
            [0.25, 0.5, 1.0, 0.0],
            [0.5, 1.0, 1.5, 0.0],
            [1.0, 1.5, 2.0, 0.0],
            [1.5, 2.0, 2.5, 0.0],
        ],
        dtype=torch.float16,
    )
    payload = encode_row_split(codes, scales, format_name, shape)

    consecutive = split_row_planes(payload, geometry, 1, 2)
    assert isinstance(consecutive.base, memoryview)
    assert consecutive.base.obj is payload
    standalone = assemble_row_planes(consecutive, format_name, shape[1])
    consecutive_scales, consecutive_codes = decode_row_split_codes(
        standalone, format_name, (2, shape[1])
    )
    assert torch.equal(consecutive_scales, scales[1:3])
    assert torch.equal(consecutive_codes, codes[1:3])

    gathered = gather_row_planes(payload, geometry, [3, 1])
    assert isinstance(gathered, RowPlanes)
    gathered_payload = assemble_row_planes(gathered, format_name, shape[1])
    gathered_scales, gathered_codes = decode_row_split_codes(
        gathered_payload, format_name, (2, shape[1])
    )
    assert torch.equal(gathered_scales, scales[[3, 1]])
    assert torch.equal(gathered_codes, codes[[3, 1]])
    expected = (
        codes[[3, 1]].float() * scales[[3, 1]].float().unsqueeze(-1)
    ).reshape(2, geometry.k_pad)[:, : shape[1]]
    actual = dequantize_row_split(
        gathered, format_name, (2, shape[1]), dtype=torch.float32
    )
    assert torch.equal(actual, expected)

    resident = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    tensor_gather = gather_row_planes(
        resident, geometry, torch.tensor([2, 0], dtype=torch.long)
    )
    assert all(
        isinstance(plane, torch.Tensor)
        for plane in (tensor_gather.base, tensor_gather.high, tensor_gather.scale)
    )
    tensor_payload = assemble_row_planes(tensor_gather, format_name, shape[1])
    tensor_scales, tensor_codes = decode_row_split_codes(
        tensor_payload, format_name, (2, shape[1])
    )
    assert torch.equal(tensor_scales, scales[[2, 0]])
    assert torch.equal(tensor_codes, codes[[2, 0]])
