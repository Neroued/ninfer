"""Binary-contract tests for the embedded v4.1 tokenizer region."""

from __future__ import annotations

import hashlib

import pytest

from .. import format as fmt
from ..convert import layout_tokenizer_assets, load_tokenizer_assets


def _write_valid_assets(path) -> dict[str, bytes]:
    data = {
        "tokenizer.json": b'{"model":{"type":"BPE"},"added_tokens":[]}\n',
        "merges.txt": b"#version: 0.2\na b\n",
        "generation_config.json": b'{"eos_token_id":[248046,248044]}\n',
    }
    for name, raw in data.items():
        (path / name).write_bytes(raw)
    return data


def test_embedded_tokenizer_records_preserve_source_bytes(tmp_path):
    source = _write_valid_assets(tmp_path)
    assets = load_tokenizer_assets(str(tmp_path))
    assert tuple(asset.kind for asset in assets) == fmt.TOKENIZER_KINDS

    data_offset = 64 * 101
    records, region = layout_tokenizer_assets(assets, data_offset)
    assert len(records) == fmt.TOKENIZER_RECORD_COUNT
    assert records[0].data_offset == data_offset

    previous_end = data_offset
    for asset, record in zip(assets, records):
        assert record.data_offset == fmt.align_up(previous_end, fmt.TOKENIZER_ALIGN)
        assert record.data_bytes == len(source[asset.source_name])
        assert record.crc32 == fmt.crc32(source[asset.source_name])
        assert record.sha256 == hashlib.sha256(source[asset.source_name]).digest()
        relative = record.data_offset - data_offset
        assert region[relative : relative + record.data_bytes] == source[asset.source_name]
        assert region[previous_end - data_offset : relative] == b"\x00" * (
            record.data_offset - previous_end
        )
        packed = fmt.pack_tokenizer_record(record)
        unpacked = fmt.unpack_tokenizer_record(packed)
        assert unpacked["kind"] == asset.kind
        assert unpacked["reserved"] == 0
        assert unpacked["sha256"] == record.sha256
        previous_end = record.data_offset + record.data_bytes

    assert len(region) == previous_end - data_offset


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("tokenizer.json", b"[]"),
        ("merges.txt", b"a\x00b"),
        ("generation_config.json", b"{}"),
    ],
)
def test_invalid_runtime_tokenizer_asset_is_rejected(tmp_path, name, replacement):
    _write_valid_assets(tmp_path)
    (tmp_path / name).write_bytes(replacement)
    with pytest.raises(ValueError):
        load_tokenizer_assets(str(tmp_path))


def test_missing_runtime_tokenizer_asset_is_rejected(tmp_path):
    _write_valid_assets(tmp_path)
    (tmp_path / "merges.txt").unlink()
    with pytest.raises(ValueError):
        load_tokenizer_assets(str(tmp_path))
