import struct

import numpy as np
import pytest
import torch

from tools.artifact.codecs.csf32 import decode_csf32_scales
from tools.artifact.codecs.fp8_block import (
    decode_fp8_block_scaled_words,
    encode_fp8_block_scaled,
)
from tools.artifact.codecs.mxfp8 import (
    decode_mxfp8_words,
    dequantize_mxfp8,
    encode_mxfp8,
)
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.layouts import csf32_geometry, encoded_size
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter


def test_native_scales_preserve_original_words():
    multiplier = struct.pack("<I", 0x3DCCCCCD)
    q4 = torch.full((128, 32), 0x21, dtype=torch.uint8)
    s4 = torch.full((128, 4), 0x38, dtype=torch.uint8)
    packed = encode_nvfp4(q4, s4, multiplier, (128, 64), "nvfp4_mul")
    q, s, m = decode_nvfp4_words(packed, (128, 64), "nvfp4_mul")
    assert torch.equal(q, q4) and torch.equal(s, s4)
    assert m.view(torch.int32).item() == 0x3DCCCCCD

    q8 = torch.full((128, 128), 0x38, dtype=torch.uint8)
    scale = torch.tensor([[0x3F800001]], dtype=torch.int32).view(torch.float32)
    fmt = "fp8_e4m3fn_block128x128_fp32"
    packed = encode_fp8_block_scaled(q8, scale, (128, 128), fmt)
    q, s = decode_fp8_block_scaled_words(packed, (128, 128), fmt)
    assert torch.equal(q, q8)
    assert torch.equal(s.view(torch.int32), scale.view(torch.int32))
    assert packed[-4:] == struct.pack("<I", 0x3F800001)


def test_mxfp8_group_scales_and_matrix_axes():
    shape = (2, 3, 64)
    codes = torch.full(shape, 0x38, dtype=torch.uint8)
    codes[1, 2, 63] = 0xB8
    scales = torch.tensor(
        [[[126, 127], [128, 129], [130, 131]], [[127, 126], [125, 124], [123, 122]]],
        dtype=torch.uint8,
    )
    packed = encode_mxfp8(codes, scales, shape)
    q, s = decode_mxfp8_words(packed, shape)
    assert torch.equal(q, codes) and torch.equal(s, scales)
    values = dequantize_mxfp8(packed, shape)
    assert values[0, 0, 0] == 0.5
    assert values[0, 0, 32] == 1
    assert values[1, 2, 63] == -(2**-5)
    scales[0, 0, 0] = 255
    with pytest.raises(ValueError, match="UE8M0"):
        encode_mxfp8(codes, scales, shape)


@pytest.mark.parametrize(
    "format", ["nvfp4_mul", "mxfp8", "fp8_e4m3fn_block128x128_fp32"]
)
def test_native_tensor_output_preserves_banked_planes(tmp_path, format):
    scalar = None
    if format == "nvfp4_mul":
        shape, layout = (2, 128, 64), "batched_block_scale_k16_m128x4_v1"
        codes = torch.full((2, 128, 32), 0x21, dtype=torch.uint8)
        scales = torch.full((2, 128, 4), 0x38, dtype=torch.uint8)
        scalar = struct.pack("<II", 0x3DCCCCCD, 0x3F800001)
        expected = encode_nvfp4(codes, scales, scalar, shape, format)
    elif format == "mxfp8":
        shape, layout = (2, 3, 64), "row_group_scale_v1"
        codes = torch.full(shape, 0x38, dtype=torch.uint8)
        scales = torch.arange(120, 132, dtype=torch.uint8).reshape(2, 3, 2)
        expected = encode_mxfp8(codes, scales, shape)
    else:
        shape, layout = (2, 128, 128), "block_scale_m128_k128_v1"
        codes = torch.full(shape, 0x38, dtype=torch.uint8)
        scales = (
            torch.tensor([0x3F800001, 0x3DCCCCCD], dtype=torch.int32)
            .view(torch.float32)
            .reshape(2, 1, 1)
        )
        expected = encode_fp8_block_scaled(codes, scales, shape, format)
    path = tmp_path / "native.ninfer"
    spec = TensorSpec("bank", shape, format, layout)
    with ArtifactWriter(
        path, [spec], components={"text": {"config": {}}}, bindings={}, uses=[]
    ) as writer:
        output = TensorOutput(writer, "bank")
        for matrix in range(2):
            word = None if scalar is None else scalar[4 * matrix : 4 * (matrix + 1)]
            output.write_codes(0, codes[matrix], scales[matrix], word, matrix=matrix)
    with Artifact(path) as artifact:
        assert artifact.read_object("bank") == expected


def test_csf_bank_global_exceptions_and_plane_output(tmp_path):
    shape = (2, 32, 32)
    bases = np.tile(np.arange(40, 72, dtype=np.uint8), 2)
    planes = {
        "codes": bytes([0x12]) * (2 * 32 * 16),
        "matrix_scale": struct.pack("<II", 0x3DCCCCCD, 0x3F800001),
        "bases": bases.tobytes(),
        "nibbles": bytes([0x32]) * 64,
        "offsets": struct.pack("<HHHH", 0, 2, 0, 0),
        "page_bases": struct.pack("<II", 0, 4),
        "exceptions": struct.pack("<HH", 0x0100, 0x7E3F),
    }
    expected = np.stack((bases.reshape(2, 32) + 2, bases.reshape(2, 32) + 3), axis=-1)
    expected[0, 0, 0] = 1
    expected[0, 31, 1] = 126
    assert np.array_equal(decode_csf32_scales(planes, shape, 2).numpy(), expected)

    spec = TensorSpec("bank", shape, "nvfp4_mul", "csf_32x2_v1", {"exception_count": 2})
    path = tmp_path / "bank.ninfer"
    with ArtifactWriter(
        path, [spec], components={"text": {"config": {}}}, bindings={}, uses=[]
    ) as writer:
        output = TensorOutput(writer, "bank")
        for name, data in planes.items():
            output.write_csf_plane(name, 0, data)
    with Artifact(path) as artifact:
        obj = artifact.by_id["bank"]
        assert obj.layout_parameters == {"exception_count": 2}
        geometry = csf32_geometry(shape, 2)
        payload = artifact.read_object("bank")
        for name, offset, size in geometry.planes:
            assert payload[offset : offset + size] == planes[name]
        assert len(payload) == encoded_size(
            spec.layout, spec.format, shape, spec.layout_parameters
        )


def test_csf_partial_tile_odd_column_and_page_boundary():
    # 160x133 scales: pages contain 256, 72 and 7 blocks. The last block is 32x1.
    shape = (160, 2128)
    geometry = csf32_geometry(shape, 1)
    assert (
        geometry.blocks_per_matrix,
        geometry.pages_per_matrix,
        geometry.offsets_per_matrix,
    ) == (335, 3, 338)
    offsets = np.zeros(338, dtype="<u2")
    offsets[-1] = 1
    planes = {
        "bases": bytes([40]) * 160,
        "nibbles": bytes(160 * 133 // 2),
        "offsets": offsets,
        "page_bases": np.zeros(3, dtype="<u4"),
        "exceptions": np.array([(99 << 8) | 62], dtype="<u2"),
    }
    actual = decode_csf32_scales(planes, shape, 1)
    expected = torch.full((160, 133), 40, dtype=torch.uint8)
    expected[159, 132] = 99
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="exception_count"):
        encoded_size("csf_32x2_v1", "nvfp4_mul", shape)
    with pytest.raises(ValueError, match="layout parameters"):
        encoded_size("row_group_scale_v1", "mxfp8", (32, 64), {"exception_count": 0})
