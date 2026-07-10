"""Binary serialization for the q5090_w4g64_mixed_v4_1 artifact.

Exact byte layout: ../../docs/q5090_packed_file_format_v4.md sections 2-9.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

MAGIC = b"Q5090MIXEDV4\x00\x00\x00\x00"
assert len(MAGIC) == 16

VERSION = 4
FORMAT_MINOR = 1
ENDIAN_TAG = 0x01020304
HEADER_SIZE = 4096
MANIFEST_SUFFIX = ".manifest.json"
MODULE_RECORD_SIZE = 64
TENSOR_ENTRY_SIZE = 128
SEGMENT_RECORD_SIZE = 32
FUSION_GROUP_RECORD_SIZE = 64
TOKENIZER_RECORD_SIZE = 64
TOKENIZER_RECORD_COUNT = 3

FLAG_TEXT_PRESENT = 1 << 0
FLAG_MTP_PRESENT = 1 << 1
FLAG_VISION_PRESENT = 1 << 2
FLAG_CALIBRATED = 1 << 3
FLAG_LM_HEAD_DRAFT_PRESENT = 1 << 4
FLAG_MODULE_PRESENT_MASK = (
    FLAG_TEXT_PRESENT | FLAG_MTP_PRESENT | FLAG_VISION_PRESENT | FLAG_LM_HEAD_DRAFT_PRESENT
)
FLAG_RESERVED_MASK = 0xFFFFFFE0

PAYLOAD_ALIGN = 256
REGION_ALIGN = 4096
TOKENIZER_ALIGN = 64

TOKENIZER_RAW_UTF8 = 0
TOKENIZER_JSON = 1
TOKENIZER_MERGES = 2
TOKENIZER_GENERATION_CONFIG = 3
TOKENIZER_KINDS = (
    TOKENIZER_JSON,
    TOKENIZER_MERGES,
    TOKENIZER_GENERATION_CONFIG,
)
TOKENIZER_KIND_NAME = {
    TOKENIZER_JSON: "TOKENIZER_JSON",
    TOKENIZER_MERGES: "MERGES_TXT",
    TOKENIZER_GENERATION_CONFIG: "GENERATION_CONFIG_JSON",
}
TOKENIZER_SOURCE_NAME = {
    TOKENIZER_JSON: "tokenizer.json",
    TOKENIZER_MERGES: "merges.txt",
    TOKENIZER_GENERATION_CONFIG: "generation_config.json",
}
TOKENIZER_MAX_BYTES = {
    TOKENIZER_JSON: 256 << 20,
    TOKENIZER_MERGES: 64 << 20,
    TOKENIZER_GENERATION_CONFIG: 1 << 20,
}

_HEADER_STRUCT = struct.Struct("<16s8I8Q14I32s4QI3I4Q")
_MODULE_STRUCT = struct.Struct("<IIQQQQII")
_TOKENIZER_STRUCT = struct.Struct("<IIQQII32s")
_ENTRY_STRUCT = struct.Struct("<IIQHHHH4I4IIHHQQIIIIHHQQQ")
_SEGMENT_STRUCT = struct.Struct("<IIIIIIQ")
_FUSION_GROUP_STRUCT = struct.Struct("<IIIIQQQII")

assert _HEADER_STRUCT.size == 280, _HEADER_STRUCT.size
assert _MODULE_STRUCT.size == 48, _MODULE_STRUCT.size
assert _TOKENIZER_STRUCT.size == TOKENIZER_RECORD_SIZE, _TOKENIZER_STRUCT.size
assert _ENTRY_STRUCT.size == 124, _ENTRY_STRUCT.size
assert _SEGMENT_STRUCT.size == SEGMENT_RECORD_SIZE, _SEGMENT_STRUCT.size
assert _FUSION_GROUP_STRUCT.size == 48, _FUSION_GROUP_STRUCT.size

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_U64 = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(name) -> int:
    """FNV-1a 64-bit hash of a UTF-8 name."""
    if isinstance(name, str):
        name = name.encode("utf-8")
    h = _FNV64_OFFSET
    for b in name:
        h = ((h ^ b) * _FNV64_PRIME) & _U64
    return h


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _pad4(values: Sequence[int]) -> List[int]:
    out = list(values[:4]) + [1, 1, 1, 1]
    return out[:4]


@dataclass
class FileHeaderFields:
    tensor_count: int
    module_count: int
    layer_count: int
    flags: int
    segment_count: int
    module_index_offset: int
    module_index_bytes: int
    tensor_index_offset: int
    tensor_index_bytes: int
    string_table_offset: int
    string_table_bytes: int
    payload_offset: int
    payload_bytes: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_key_head_dim: int
    gdn_value_head_dim: int
    gdn_conv_width: int
    full_attention_interval: int
    max_position_embeddings: int
    fusion_group_count: int
    segment_index_offset: int
    segment_index_bytes: int
    fusion_group_index_offset: int
    fusion_group_index_bytes: int
    sha256_safetensors_index: bytes = b"\x00" * 32
    format_minor: int = FORMAT_MINOR
    tokenizer_record_count: int = TOKENIZER_RECORD_COUNT
    tokenizer_record_size: int = TOKENIZER_RECORD_SIZE
    tokenizer_flags: int = 0
    tokenizer_index_offset: int = 0
    tokenizer_index_bytes: int = TOKENIZER_RECORD_COUNT * TOKENIZER_RECORD_SIZE
    tokenizer_data_offset: int = 0
    tokenizer_data_bytes: int = 0


def pack_header(h: FileHeaderFields) -> bytes:
    body = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        ENDIAN_TAG,
        HEADER_SIZE,
        h.tensor_count,
        h.module_count,
        h.layer_count,
        h.flags,
        h.segment_count,
        h.module_index_offset,
        h.module_index_bytes,
        h.tensor_index_offset,
        h.tensor_index_bytes,
        h.string_table_offset,
        h.string_table_bytes,
        h.payload_offset,
        h.payload_bytes,
        h.hidden_size,
        h.intermediate_size,
        h.vocab_size,
        h.num_attention_heads,
        h.num_key_value_heads,
        h.head_dim,
        h.gdn_key_heads,
        h.gdn_value_heads,
        h.gdn_key_head_dim,
        h.gdn_value_head_dim,
        h.gdn_conv_width,
        h.full_attention_interval,
        h.max_position_embeddings,
        h.fusion_group_count,
        h.sha256_safetensors_index[:32].ljust(32, b"\x00"),
        h.segment_index_offset,
        h.segment_index_bytes,
        h.fusion_group_index_offset,
        h.fusion_group_index_bytes,
        h.format_minor,
        h.tokenizer_record_count,
        h.tokenizer_record_size,
        h.tokenizer_flags,
        h.tokenizer_index_offset,
        h.tokenizer_index_bytes,
        h.tokenizer_data_offset,
        h.tokenizer_data_bytes,
    )
    assert len(body) == _HEADER_STRUCT.size
    return body.ljust(HEADER_SIZE, b"\x00")


def unpack_header(buf: bytes) -> dict:
    vals = _HEADER_STRUCT.unpack(buf[: _HEADER_STRUCT.size])
    keys = [
        "magic",
        "version",
        "endian",
        "header_size",
        "tensor_count",
        "module_count",
        "layer_count",
        "flags",
        "segment_count",
        "module_index_offset",
        "module_index_bytes",
        "tensor_index_offset",
        "tensor_index_bytes",
        "string_table_offset",
        "string_table_bytes",
        "payload_offset",
        "payload_bytes",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "gdn_key_heads",
        "gdn_value_heads",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
        "gdn_conv_width",
        "full_attention_interval",
        "max_position_embeddings",
        "fusion_group_count",
        "sha256_safetensors_index",
        "segment_index_offset",
        "segment_index_bytes",
        "fusion_group_index_offset",
        "fusion_group_index_bytes",
        "format_minor",
        "tokenizer_record_count",
        "tokenizer_record_size",
        "tokenizer_flags",
        "tokenizer_index_offset",
        "tokenizer_index_bytes",
        "tokenizer_data_offset",
        "tokenizer_data_bytes",
    ]
    return dict(zip(keys, vals))


@dataclass
class ModuleRecord:
    module_kind: int
    module_version: int
    tensor_index_begin: int
    tensor_index_count: int
    payload_offset: int
    payload_bytes: int


def pack_module_record(m: ModuleRecord) -> bytes:
    body = _MODULE_STRUCT.pack(
        m.module_kind,
        m.module_version,
        m.tensor_index_begin,
        m.tensor_index_count,
        m.payload_offset,
        m.payload_bytes,
        0,
        0,
    )
    return body.ljust(MODULE_RECORD_SIZE, b"\x00")


def unpack_module_record(buf: bytes) -> dict:
    vals = _MODULE_STRUCT.unpack(buf[: _MODULE_STRUCT.size])
    keys = [
        "module_kind",
        "module_version",
        "tensor_index_begin",
        "tensor_index_count",
        "payload_offset",
        "payload_bytes",
        "reserved0",
        "reserved1",
    ]
    return dict(zip(keys, vals))


@dataclass(frozen=True)
class TokenizerRecord:
    kind: int
    encoding: int
    data_offset: int
    data_bytes: int
    crc32: int
    sha256: bytes


def pack_tokenizer_record(record: TokenizerRecord) -> bytes:
    return _TOKENIZER_STRUCT.pack(
        record.kind,
        record.encoding,
        record.data_offset,
        record.data_bytes,
        record.crc32,
        0,
        record.sha256[:32].ljust(32, b"\x00"),
    )


def unpack_tokenizer_record(buf: bytes) -> dict:
    vals = _TOKENIZER_STRUCT.unpack(buf[:TOKENIZER_RECORD_SIZE])
    keys = [
        "kind",
        "encoding",
        "data_offset",
        "data_bytes",
        "crc32",
        "reserved",
        "sha256",
    ]
    return dict(zip(keys, vals))


@dataclass
class TensorEntry:
    name: str
    qtype: int
    layout: int
    module_kind: int
    shape: List[int]
    padded_shape: List[int]
    group_size: int
    scale_dtype: int
    segment_count: int
    source_layer: int
    source_kind: int
    name_offset: int = 0
    payload_offset: int = 0
    payload_bytes: int = 0
    crc32: int = 0
    segment_begin: int = 0
    fusion_group_id: int = 0
    fusion_index: int = 0
    nibble_plane_bytes: int = 0
    high_plane_bytes: int = 0
    scale_plane_bytes: int = 0

    @property
    def ndim(self) -> int:
        return len(self.shape)


def pack_tensor_entry(e: TensorEntry) -> bytes:
    name_bytes = e.name.encode("utf-8")
    shape = _pad4(e.shape)
    padded = _pad4(e.padded_shape)
    body = _ENTRY_STRUCT.pack(
        e.name_offset,
        len(name_bytes),
        fnv1a_64(name_bytes),
        e.qtype,
        e.layout,
        e.module_kind,
        e.ndim,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        padded[0],
        padded[1],
        padded[2],
        padded[3],
        e.group_size,
        e.scale_dtype,
        e.segment_count,
        e.payload_offset,
        e.payload_bytes,
        e.source_layer,
        e.source_kind,
        e.crc32,
        e.segment_begin,
        e.fusion_group_id,
        e.fusion_index,
        e.nibble_plane_bytes,
        e.high_plane_bytes,
        e.scale_plane_bytes,
    )
    return body.ljust(TENSOR_ENTRY_SIZE, b"\x00")


def unpack_tensor_entry(buf: bytes) -> dict:
    vals = _ENTRY_STRUCT.unpack(buf[: _ENTRY_STRUCT.size])
    keys = [
        "name_offset",
        "name_len",
        "name_hash",
        "qtype",
        "layout",
        "module_kind",
        "ndim",
        "shape0",
        "shape1",
        "shape2",
        "shape3",
        "padded0",
        "padded1",
        "padded2",
        "padded3",
        "group_size",
        "scale_dtype",
        "segment_count",
        "payload_offset",
        "payload_bytes",
        "source_layer",
        "source_kind",
        "crc32",
        "segment_begin",
        "fusion_group_id",
        "fusion_index",
        "nibble_plane_bytes",
        "high_plane_bytes",
        "scale_plane_bytes",
    ]
    d = dict(zip(keys, vals))
    d["shape"] = [d["shape0"], d["shape1"], d["shape2"], d["shape3"]][: d["ndim"]]
    d["padded_shape"] = [d["padded0"], d["padded1"], d["padded2"], d["padded3"]][: d["ndim"]]
    return d


@dataclass
class SegmentRecord:
    name: str
    source_kind: int
    source_layer: int
    row_begin: int
    row_count: int
    name_offset: int = 0


def pack_segment_record(s: SegmentRecord) -> bytes:
    name_bytes = s.name.encode("utf-8")
    return _SEGMENT_STRUCT.pack(
        s.source_kind,
        s.source_layer,
        s.row_begin,
        s.row_count,
        s.name_offset,
        len(name_bytes),
        fnv1a_64(name_bytes),
    )


def unpack_segment_record(buf: bytes) -> dict:
    vals = _SEGMENT_STRUCT.unpack(buf[: _SEGMENT_STRUCT.size])
    keys = [
        "source_kind",
        "source_layer",
        "row_begin",
        "row_count",
        "name_offset",
        "name_len",
        "name_hash",
    ]
    return dict(zip(keys, vals))


@dataclass
class FusionGroupRecord:
    group_id: int
    source_layer: int
    block_count: int
    shared_input_kind: int
    first_block_tensor_index: int
    payload_offset: int
    payload_bytes: int
    total_n: int
    shared_k: int


def pack_fusion_group_record(g: FusionGroupRecord) -> bytes:
    body = _FUSION_GROUP_STRUCT.pack(
        g.group_id,
        g.source_layer,
        g.block_count,
        g.shared_input_kind,
        g.first_block_tensor_index,
        g.payload_offset,
        g.payload_bytes,
        g.total_n,
        g.shared_k,
    )
    return body.ljust(FUSION_GROUP_RECORD_SIZE, b"\x00")


def unpack_fusion_group_record(buf: bytes) -> dict:
    vals = _FUSION_GROUP_STRUCT.unpack(buf[: _FUSION_GROUP_STRUCT.size])
    keys = [
        "group_id",
        "source_layer",
        "block_count",
        "shared_input_kind",
        "first_block_tensor_index",
        "payload_offset",
        "payload_bytes",
        "total_n",
        "shared_k",
    ]
    return dict(zip(keys, vals))


def build_string_table(entries: List[TensorEntry], segments: Optional[List[SegmentRecord]] = None) -> bytes:
    """Assign name_offset to block and segment records and return a NUL-terminated string table."""
    table = bytearray()
    records = list(entries)
    if segments is not None:
        records.extend(segments)
    for r in records:
        r.name_offset = len(table)
        table.extend(r.name.encode("utf-8"))
        table.append(0)
    return bytes(table)
