import struct

import pytest
import torch

from tools.artifact.codecs.nf4 import decode_nf4_words, encode_nf4
from tools.artifact.formats import NF4_CODEBOOK
from tools.artifact.layouts import row_paged_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter


@pytest.mark.parametrize("rows", [1, 44, 45, 46, 90, 91])
def test_nf4_pages_and_streaming_are_chunk_independent(tmp_path, rows):
    codes = (torch.arange(rows * 80) % 256).to(torch.uint8).reshape(rows, 80)
    scales = (torch.arange(rows * 10) % 127).to(torch.uint8).reshape(rows, 10)
    divisor = struct.pack("<f", 3.25)
    expected = bytearray(((rows + 44) // 45) * 4096)
    for row in range(rows):
        offset = row // 45 * 4096 + row % 45 * 90
        expected[offset : offset + 80] = bytes(codes[row].tolist())
        expected[offset + 80 : offset + 90] = bytes(scales[row].tolist())
    expected.extend(divisor)
    assert encode_nf4(codes, scales, divisor, (rows, 160)) == expected
    for chunk in (1, 17, 45, 64):
        path = tmp_path / f"{chunk}.ninfer"
        with ArtifactWriter(
            path,
            [TensorSpec("table", (rows, 160), "nf4_g16_e4m3fn", "row_paged_g16_4k_v1")],
            components={"text": {"config": {}}},
            bindings={"table": {"object": "table"}},
            max_file_bytes=12288,
        ) as writer:
            out = TensorOutput(writer, "table")
            for first in range(0, rows, chunk):
                out.write_codes(
                    first,
                    codes[first : first + chunk],
                    scales[first : first + chunk],
                    divisor,
                )
        with Artifact(path) as artifact:
            raw = artifact.read_object("table")
            assert raw == expected
            c, s, d = decode_nf4_words(raw, (rows, 160))
            assert torch.equal(c, codes) and torch.equal(s, scales) and d.item() == 3.25


def test_nf4_representation_and_invalid_words():
    # Low nibble is the first value; the two endpoint codes represent -1 and +1.
    payload = encode_nf4(
        torch.full((1, 8), 0xF0, dtype=torch.uint8),
        torch.full((1, 1), 0x38, dtype=torch.uint8),
        struct.pack("<f", 2.0),
        (1, 16),
    )
    codes, scales, divisor = decode_nf4_words(payload, (1, 16))
    indices = torch.stack((codes & 15, codes >> 4), -1).flatten()
    values = (
        torch.tensor(NF4_CODEBOOK)[indices.long()]
        * scales.view(torch.float8_e4m3fn).float().item()
        / divisor
    )
    assert values.tolist() == [-0.5, 0.5] * 8
    for position, word in ((8, 0x7F), (8, 0x80), (9, 1), (4099, 0xFF)):
        bad = bytearray(payload)
        bad[position] = word
        with pytest.raises(ValueError):
            decode_nf4_words(bad, (1, 16))
    g = row_paged_geometry("nf4_g16_e4m3fn", (2500012, 160))
    assert (g.rows_per_page, g.page_count, g.payload_bytes) == (45, 55556, 227557380)
    assert g.row_offset(45) == 4096 and g.row_offset(44) == 3960
    for shape in ((2, 17), (2, 8192)):
        with pytest.raises(ValueError):
            row_paged_geometry("nf4_g16_e4m3fn", shape)
