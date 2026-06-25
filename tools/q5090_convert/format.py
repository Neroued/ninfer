"""Binary (de)serialization for the q5090_w4g64_mixed_v1 packed file.

Exact byte layout: ../../docs/q5090_packed_file_format_v1.md (sections 2-4, 9).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import List

MAGIC = b"Q5090MIXEDV1\x00\x00\x00\x00"  # 16 bytes
assert len(MAGIC) == 16

VERSION = 1
ENDIAN_TAG = 0x01020304
HEADER_SIZE = 4096
MODULE_RECORD_SIZE = 64
TENSOR_ENTRY_SIZE = 128

PAYLOAD_ALIGN = 256       # per-tensor payload alignment
REGION_ALIGN = 4096       # payload region start alignment

# struct layouts (little-endian). See spec.
_HEADER_STRUCT = struct.Struct("<16s8I8Q14I32s")
_MODULE_STRUCT = struct.Struct("<IIQQQQII")
_ENTRY_STRUCT = struct.Struct("<IIQHHHH4I4IIHHQQIIII")

assert _HEADER_STRUCT.size == 200, _HEADER_STRUCT.size
assert _MODULE_STRUCT.size == 48, _MODULE_STRUCT.size
assert _ENTRY_STRUCT.size == 96, _ENTRY_STRUCT.size

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


def _pad4(values: List[int]) -> List[int]:
    out = list(values[:4]) + [1, 1, 1, 1]
    return out[:4]


@dataclass
class FileHeaderFields:
    tensor_count: int
    module_count: int
    layer_count: int
    flags: int
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
    sha256_index: bytes = b"\x00" * 32


def pack_header(h: FileHeaderFields) -> bytes:
    body = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION, ENDIAN_TAG, HEADER_SIZE,
        h.tensor_count, h.module_count, h.layer_count, h.flags, 0,
        h.module_index_offset, h.module_index_bytes,
        h.tensor_index_offset, h.tensor_index_bytes,
        h.string_table_offset, h.string_table_bytes,
        h.payload_offset, h.payload_bytes,
        h.hidden_size, h.intermediate_size, h.vocab_size,
        h.num_attention_heads, h.num_key_value_heads, h.head_dim,
        h.gdn_key_heads, h.gdn_value_heads, h.gdn_key_head_dim,
        h.gdn_value_head_dim, h.gdn_conv_width, h.full_attention_interval,
        h.max_position_embeddings, 0,
        h.sha256_index[:32].ljust(32, b"\x00"),
    )
    assert len(body) == _HEADER_STRUCT.size
    return body.ljust(HEADER_SIZE, b"\x00")


def unpack_header(buf: bytes) -> dict:
    vals = _HEADER_STRUCT.unpack(buf[: _HEADER_STRUCT.size])
    keys = [
        "magic", "version", "endian", "header_size",
        "tensor_count", "module_count", "layer_count", "flags", "reserved0",
        "module_index_offset", "module_index_bytes",
        "tensor_index_offset", "tensor_index_bytes",
        "string_table_offset", "string_table_bytes",
        "payload_offset", "payload_bytes",
        "hidden_size", "intermediate_size", "vocab_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "gdn_key_heads", "gdn_value_heads", "gdn_key_head_dim",
        "gdn_value_head_dim", "gdn_conv_width", "full_attention_interval",
        "max_position_embeddings", "reserved1", "sha256_index",
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
    load_policy: int
    flags: int = 0


def pack_module_record(m: ModuleRecord) -> bytes:
    body = _MODULE_STRUCT.pack(
        m.module_kind, m.module_version,
        m.tensor_index_begin, m.tensor_index_count,
        m.payload_offset, m.payload_bytes,
        m.load_policy, m.flags,
    )
    return body.ljust(MODULE_RECORD_SIZE, b"\x00")


def unpack_module_record(buf: bytes) -> dict:
    vals = _MODULE_STRUCT.unpack(buf[: _MODULE_STRUCT.size])
    keys = [
        "module_kind", "module_version", "tensor_index_begin",
        "tensor_index_count", "payload_offset", "payload_bytes",
        "load_policy", "flags",
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
    source_layer: int
    source_kind: int
    # filled in during writing:
    name_offset: int = 0
    payload_offset: int = 0
    payload_bytes: int = 0
    crc32: int = 0

    @property
    def ndim(self) -> int:
        return len(self.shape)


def pack_tensor_entry(e: TensorEntry) -> bytes:
    name_bytes = e.name.encode("utf-8")
    shape = _pad4(e.shape)
    padded = _pad4(e.padded_shape)
    body = _ENTRY_STRUCT.pack(
        e.name_offset, len(name_bytes), fnv1a_64(name_bytes),
        e.qtype, e.layout, e.module_kind, e.ndim,
        shape[0], shape[1], shape[2], shape[3],
        padded[0], padded[1], padded[2], padded[3],
        e.group_size, e.scale_dtype, 0,
        e.payload_offset, e.payload_bytes,
        e.source_layer, e.source_kind, e.crc32, 0,
    )
    return body.ljust(TENSOR_ENTRY_SIZE, b"\x00")


def unpack_tensor_entry(buf: bytes) -> dict:
    vals = _ENTRY_STRUCT.unpack(buf[: _ENTRY_STRUCT.size])
    keys = [
        "name_offset", "name_len", "name_hash",
        "qtype", "layout", "module_kind", "ndim",
        "shape0", "shape1", "shape2", "shape3",
        "padded0", "padded1", "padded2", "padded3",
        "group_size", "scale_dtype", "reserved0",
        "payload_offset", "payload_bytes",
        "source_layer", "source_kind", "crc32", "reserved1",
    ]
    d = dict(zip(keys, vals))
    d["shape"] = [d["shape0"], d["shape1"], d["shape2"], d["shape3"]][: d["ndim"]]
    d["padded_shape"] = [d["padded0"], d["padded1"], d["padded2"], d["padded3"]][: d["ndim"]]
    return d


def build_string_table(entries: List[TensorEntry]) -> bytes:
    """Assign name_offset to each entry and return the packed NUL-terminated table."""
    table = bytearray()
    for e in entries:
        e.name_offset = len(table)
        table.extend(e.name.encode("utf-8"))
        table.append(0)
    return bytes(table)
