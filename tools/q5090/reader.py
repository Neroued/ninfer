"""Canonical mmap-backed q5090 v4.1 reader."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import zlib
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Iterator

import torch

from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.packing import row_split_plane_sizes
from tools.q5090_convert.layouts import decode_row_split_quantized

from .codec import decode_tensor


@dataclass(frozen=True)
class Block:
    index: int
    name: str
    qtype: int
    layout: int
    module_kind: int
    shape: list[int]
    padded_shape: list[int]
    group_size: int
    scale_dtype: int
    payload_offset: int
    payload_bytes: int
    nibble_plane_bytes: int
    high_plane_bytes: int
    scale_plane_bytes: int
    source_layer: int
    source_kind: int
    crc32: int
    segment_begin: int
    segment_count: int
    fusion_group_id: int
    fusion_index: int


@dataclass(frozen=True)
class View:
    index: int
    name: str
    block: Block
    source_kind: int
    source_layer: int
    row_begin: int
    row_count: int


class Reader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = fmt.unpack_header(self._mmap[:fmt.HEADER_SIZE])
        self._validate_header()
        self.modules = self._read_fixed_table(
            "module_index_offset",
            "module_count",
            fmt.MODULE_RECORD_SIZE,
            fmt.unpack_module_record,
        )
        tensors = self._read_fixed_table(
            "tensor_index_offset",
            "tensor_count",
            fmt.TENSOR_ENTRY_SIZE,
            fmt.unpack_tensor_entry,
        )
        segments = self._read_fixed_table(
            "segment_index_offset",
            "segment_count",
            fmt.SEGMENT_RECORD_SIZE,
            fmt.unpack_segment_record,
        )
        self.fusions = self._read_fixed_table(
            "fusion_group_index_offset",
            "fusion_group_count",
            fmt.FUSION_GROUP_RECORD_SIZE,
            fmt.unpack_fusion_group_record,
        )
        self.tokenizers = self._read_fixed_table(
            "tokenizer_index_offset",
            "tokenizer_record_count",
            fmt.TOKENIZER_RECORD_SIZE,
            fmt.unpack_tokenizer_record,
        )
        table_start = self.header["string_table_offset"]
        table_end = table_start + self.header["string_table_bytes"]
        strings = self._mmap[table_start:table_end]
        for index, record in enumerate(tensors):
            record["name"] = self._name(strings, record, f"block[{index}]")
        for index, record in enumerate(segments):
            record["name"] = self._name(strings, record, f"segment[{index}]")
        self.blocks = [self._block(i, record) for i, record in enumerate(tensors)]
        self.entries = {block.name: block for block in self.blocks}
        if len(self.entries) != len(self.blocks):
            raise ValueError("duplicate q5090 block name")
        self.segments = segments
        self.views: dict[str, View] = {}
        for block in self.blocks:
            end = block.segment_begin + block.segment_count
            if end > len(segments):
                raise ValueError(f"{block.name}: segment range outside table")
            for index in range(block.segment_begin, end):
                segment = segments[index]
                view = View(
                    index,
                    segment["name"],
                    block,
                    segment["source_kind"],
                    segment["source_layer"],
                    segment["row_begin"],
                    segment["row_count"],
                )
                if view.name in self.views:
                    raise ValueError(f"duplicate q5090 segment name: {view.name}")
                self.views[view.name] = view
        self._validate_catalog()

    def _validate_header(self) -> None:
        h = self.header
        required = {
            "magic": fmt.MAGIC,
            "version": fmt.VERSION,
            "format_minor": fmt.FORMAT_MINOR,
            "endian": fmt.ENDIAN_TAG,
            "header_size": fmt.HEADER_SIZE,
            "tokenizer_record_size": fmt.TOKENIZER_RECORD_SIZE,
            "tokenizer_record_count": fmt.TOKENIZER_RECORD_COUNT,
        }
        bad = [f"{key}={h.get(key)!r}, expected {value!r}" for key, value in required.items()
               if h.get(key) != value]
        if bad:
            raise ValueError("invalid q5090 header: " + "; ".join(bad))
        if h["flags"] & fmt.FLAG_RESERVED_MASK:
            raise ValueError(f"q5090 reserved flags set: {h['flags']:#x}")
        sized_regions = (
            ("module_index_offset", h["module_count"] * fmt.MODULE_RECORD_SIZE, "module table"),
            ("tensor_index_offset", h["tensor_count"] * fmt.TENSOR_ENTRY_SIZE, "tensor table"),
            ("segment_index_offset", h["segment_count"] * fmt.SEGMENT_RECORD_SIZE, "segment table"),
            (
                "fusion_group_index_offset",
                h["fusion_group_count"] * fmt.FUSION_GROUP_RECORD_SIZE,
                "fusion table",
            ),
            (
                "tokenizer_index_offset",
                h["tokenizer_record_count"] * fmt.TOKENIZER_RECORD_SIZE,
                "tokenizer table",
            ),
            ("string_table_offset", h["string_table_bytes"], "string table"),
            ("tokenizer_data_offset", h["tokenizer_data_bytes"], "tokenizer data"),
            ("payload_offset", h["payload_bytes"], "payload"),
        )
        file_bytes = len(self._mmap)
        for offset_key, size, label in sized_regions:
            offset = h[offset_key]
            if offset < 0 or size < 0 or offset > file_bytes or size > file_bytes - offset:
                raise ValueError(f"q5090 {label} is outside the file")

    def _read_fixed_table(self, offset_key, count_key, size, unpack):
        offset = self.header[offset_key]
        records = []
        for _ in range(self.header[count_key]):
            records.append(unpack(self._mmap[offset:offset + size]))
            offset += size
        return records

    @staticmethod
    def _name(table: bytes, record: dict, label: str) -> str:
        begin = record["name_offset"]
        end = begin + record["name_len"]
        if begin < 0 or end >= len(table) or table[end] != 0:
            raise ValueError(f"{label}: invalid string range")
        raw = table[begin:end]
        if fmt.fnv1a_64(raw) != record["name_hash"]:
            raise ValueError(f"{label}: name hash mismatch")
        return raw.decode("utf-8")

    @staticmethod
    def _block(index: int, e: dict) -> Block:
        return Block(
            index=index,
            name=e["name"],
            qtype=e["qtype"],
            layout=e["layout"],
            module_kind=e["module_kind"],
            shape=e["shape"],
            padded_shape=e["padded_shape"],
            group_size=e["group_size"],
            scale_dtype=e["scale_dtype"],
            payload_offset=e["payload_offset"],
            payload_bytes=e["payload_bytes"],
            nibble_plane_bytes=e["nibble_plane_bytes"],
            high_plane_bytes=e["high_plane_bytes"],
            scale_plane_bytes=e["scale_plane_bytes"],
            source_layer=e["source_layer"],
            source_kind=e["source_kind"],
            crc32=e["crc32"],
            segment_begin=e["segment_begin"],
            segment_count=e["segment_count"],
            fusion_group_id=e["fusion_group_id"],
            fusion_index=e["fusion_index"],
        )

    def _validate_catalog(self) -> None:
        file_bytes = len(self._mmap)
        module_kinds = set(qt.MODULE_NAME)
        qtypes = set(qt.QTYPE_NAME)
        layouts = set(qt.LAYOUT_NAME)
        for block in self.blocks:
            if block.module_kind not in module_kinds:
                raise ValueError(f"{block.name}: unknown module kind {block.module_kind}")
            if block.qtype not in qtypes or block.layout not in layouts:
                raise ValueError(f"{block.name}: unknown qtype/layout")
            if not block.shape or len(block.shape) != len(block.padded_shape):
                raise ValueError(f"{block.name}: invalid shape rank")
            if any(dim <= 0 for dim in block.shape + block.padded_shape):
                raise ValueError(f"{block.name}: nonpositive shape")
            if any(padded < logical for logical, padded in zip(block.shape, block.padded_shape)):
                raise ValueError(f"{block.name}: padded shape is smaller than logical shape")
            if (
                block.payload_offset > file_bytes
                or block.payload_bytes > file_bytes - block.payload_offset
            ):
                raise ValueError(f"{block.name}: payload is outside the file")
        for view in self.views.values():
            if view.row_begin < 0 or view.row_count <= 0:
                raise ValueError(f"{view.name}: invalid row range")
            if view.row_begin + view.row_count > view.block.shape[0]:
                raise ValueError(f"{view.name}: row range exceeds block")
        seen_tokenizers = set()
        for record in self.tokenizers:
            kind = record["kind"]
            if kind not in fmt.TOKENIZER_KINDS or kind in seen_tokenizers:
                raise ValueError(f"invalid or duplicate tokenizer kind {kind}")
            seen_tokenizers.add(kind)
            self.tokenizer_data(kind)
        if seen_tokenizers != set(fmt.TOKENIZER_KINDS):
            raise ValueError("q5090 tokenizer catalog is incomplete")
        for index, module in enumerate(self.modules):
            begin = module["tensor_index_begin"]
            count = module["tensor_index_count"]
            if begin + count > len(self.blocks):
                raise ValueError(f"module[{index}]: tensor range outside catalog")
            module_blocks = self.blocks[begin:begin + count]
            if any(block.module_kind != module["module_kind"] for block in module_blocks):
                raise ValueError(f"module[{index}]: block module kind mismatch")
        for index, fusion in enumerate(self.fusions):
            first = fusion["first_block_tensor_index"]
            count = fusion["block_count"]
            if fusion["group_id"] not in qt.FUSION_GROUP_NAME or first + count > len(self.blocks):
                raise ValueError(f"fusion[{index}]: invalid group or block range")

    def payload(self, block: Block) -> bytes:
        begin = block.payload_offset
        return self._mmap[begin:begin + block.payload_bytes]

    def tokenizer_data(self, kind: int) -> bytes:
        matches = [record for record in self.tokenizers if record["kind"] == kind]
        if len(matches) != 1:
            raise KeyError(f"q5090 tokenizer kind {kind} is absent or duplicated")
        record = matches[0]
        begin = record["data_offset"]
        data = self._mmap[begin:begin + record["data_bytes"]]
        if zlib.crc32(data) & 0xFFFFFFFF != record["crc32"]:
            raise ValueError(f"q5090 tokenizer kind {kind} CRC mismatch")
        if hashlib.sha256(data).digest() != record["sha256"]:
            raise ValueError(f"q5090 tokenizer kind {kind} SHA256 mismatch")
        return data

    def row_geometry(self, block: Block) -> tuple[int, int, int, int, int, int, int, int]:
        if block.layout != qt.LAYOUT_ROW_SPLIT or block.qtype not in qt.QUANT_SPECS:
            raise ValueError(f"{block.name} is not quantized ROW_SPLIT")
        spec = qt.QUANT_SPECS[block.qtype]
        n, k = block.shape
        _, kp = block.padded_shape
        groups = kp // spec.group_size
        nibble, high_off, high, scale_off, scale, _ = row_split_plane_sizes(
            n, groups, spec.nibble_bytes_per_group, spec.high_bytes_per_group
        )
        if (nibble, high, scale) != (
            block.nibble_plane_bytes,
            block.high_plane_bytes,
            block.scale_plane_bytes,
        ):
            raise ValueError(f"{block.name}: ROW_SPLIT plane geometry mismatch")
        return (
            n,
            k,
            kp,
            groups * spec.nibble_bytes_per_group,
            groups * spec.high_bytes_per_group,
            groups * 2,
            high_off,
            scale_off,
        )

    def slice_rows(
        self,
        block: Block,
        row: int,
        count: int,
        payload: bytes | torch.Tensor | None = None,
    ) -> bytes | torch.Tensor:
        n, _, _, rn, rh, rs, high_off, scale_off = self.row_geometry(block)
        if row < 0 or count < 0 or row + count > n:
            raise IndexError(f"{block.name}: row slice outside tensor")
        source = self.payload(block) if payload is None else payload
        nibble = source[row * rn:(row + count) * rn]
        high = source[high_off + row * rh:high_off + (row + count) * rh]
        scale = source[scale_off + row * rs:scale_off + (row + count) * rs]
        npad = fmt.align_up(len(nibble), fmt.PAYLOAD_ALIGN) - len(nibble)
        hpad = fmt.align_up(len(high), fmt.PAYLOAD_ALIGN) - len(high) if len(high) else 0
        if isinstance(source, torch.Tensor):
            parts = [nibble]
            if npad:
                parts.append(torch.zeros(npad, dtype=torch.uint8, device=source.device))
            if len(high):
                parts.append(high)
                if hpad:
                    parts.append(torch.zeros(hpad, dtype=torch.uint8, device=source.device))
            parts.append(scale)
            return torch.cat(parts)
        return nibble + b"\0" * npad + high + b"\0" * hpad + scale

    def decode_block(
        self,
        block: Block,
        device: torch.device | str,
        *,
        payload: bytes | torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
        compiled: bool = True,
    ) -> torch.Tensor:
        return decode_tensor(
            self.payload(block) if payload is None else payload,
            block.qtype,
            block.layout,
            block.shape,
            block.padded_shape,
            device,
            output_dtype=dtype,
            compiled=compiled,
        )

    def decode_view(
        self,
        view: View,
        device: torch.device | str,
        *,
        payload: bytes | torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
        compiled: bool = True,
    ) -> torch.Tensor:
        block = view.block
        if view.row_begin == 0 and view.row_count == block.shape[0]:
            return self.decode_block(
                block, device, payload=payload, dtype=dtype, compiled=compiled
            )
        sliced = self.slice_rows(block, view.row_begin, view.row_count, payload)
        return decode_tensor(
            sliced,
            block.qtype,
            block.layout,
            [view.row_count, block.shape[1]],
            [view.row_count, block.padded_shape[1]],
            device,
            output_dtype=dtype,
            compiled=compiled,
        )

    def row_chunks(
        self,
        view: View,
        device: torch.device | str,
        rows: int,
        *,
        payload: bytes | torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
        compiled: bool = True,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        for row0 in range(0, view.row_count, rows):
            row1 = min(view.row_count, row0 + rows)
            sliced = self.slice_rows(view.block, view.row_begin + row0, row1 - row0, payload)
            yield row0, row1, decode_tensor(
                sliced,
                view.block.qtype,
                view.block.layout,
                [row1 - row0, view.block.shape[1]],
                [row1 - row0, view.block.padded_shape[1]],
                device,
                output_dtype=dtype,
                compiled=compiled,
            )

    def structural_dump(self, path: str | Path) -> None:
        def json_value(value):
            return value.hex() if isinstance(value, bytes) else value

        segments_by_block: dict[int, list[View]] = {}
        for view in self.views.values():
            segments_by_block.setdefault(view.block.index, []).append(view)
        blocks = []
        for block in self.blocks:
            block_segments = [
                {
                    "segment_index": view.index,
                    "name": view.name,
                    "source_kind": view.source_kind,
                    "source_layer": view.source_layer,
                    "row_begin": view.row_begin,
                    "row_count": view.row_count,
                }
                for view in sorted(
                    segments_by_block.get(block.index, []), key=lambda item: item.index
                )
            ]
            if qt.is_quant(block.qtype):
                spec = qt.QUANT_SPECS[block.qtype]
                probes = []
                n, k = block.shape
                for row, col in ((0, 0), (n // 2, k // 2), (n - 1, k - 1)):
                    payload = self.slice_rows(block, row, 1)
                    scale16, codes = decode_row_split_quantized(
                        payload, [1, block.padded_shape[1]], block.qtype, "cpu"
                    )
                    group, lane = divmod(col, spec.group_size)
                    code = int(codes[0, group, lane].item())
                    scale = float(scale16[0, group].float().item())
                    probes.append(
                        {"row": row, "col": col, "scale": scale, "q": code, "value": scale * code}
                    )
            else:
                tensor = self.decode_block(block, "cpu", dtype=torch.float32).reshape(-1)
                probes = []
                for index in (0, tensor.numel() // 2, tensor.numel() - 1):
                    if len(block.shape) == 1:
                        row, col = index, 0
                    else:
                        cols = prod(block.shape[1:])
                        row, col = index // cols, index % cols
                    probes.append({"row": row, "col": col, "value": float(tensor[index].item())})
            item = {
                "block_index": block.index,
                "name": block.name,
                "source_kind": block.source_kind,
                "source_layer": block.source_layer,
                "qtype": qt.QTYPE_NAME[block.qtype],
                "layout": qt.LAYOUT_NAME[block.layout],
                "shape": block.shape,
                "padded_shape": block.padded_shape,
                "payload_offset": block.payload_offset,
                "payload_bytes": block.payload_bytes,
                "nibble_plane_bytes": block.nibble_plane_bytes,
                "high_plane_bytes": block.high_plane_bytes,
                "scale_plane_bytes": block.scale_plane_bytes,
                "crc32": block.crc32,
                "fusion_group_id": block.fusion_group_id,
                "fusion_index": block.fusion_index,
                "segments": block_segments,
                "dequant_probes": probes,
            }
            blocks.append(item)
        fusions = []
        for fusion in self.fusions:
            first = fusion["first_block_tensor_index"]
            count = fusion["block_count"]
            fusions.append(
                {
                    "group_id": qt.FUSION_GROUP_NAME.get(fusion["group_id"], "NONE"),
                    "source_layer": fusion["source_layer"],
                    "block_count": count,
                    "shared_input_kind": fusion["shared_input_kind"],
                    "first_block_tensor_index": first,
                    "payload_offset": fusion["payload_offset"],
                    "payload_bytes": fusion["payload_bytes"],
                    "total_n": fusion["total_n"],
                    "shared_k": fusion["shared_k"],
                    "members": [block.name for block in self.blocks[first:first + count]],
                }
            )
        tokenizers = [
            {
                "kind": fmt.TOKENIZER_KIND_NAME[record["kind"]],
                "encoding": record["encoding"],
                "data_offset": record["data_offset"],
                "data_bytes": record["data_bytes"],
                "crc32": f"{record['crc32']:08x}",
                "sha256": record["sha256"].hex(),
            }
            for record in self.tokenizers
        ]
        out = {
            "format": "q5090_w4g64_mixed_v4_1",
            "file": str(self.path),
            "header": {key: json_value(value) for key, value in self.header.items()},
            "modules": self.modules,
            "blocks": blocks,
            "fusion_groups": fusions,
            "tokenizer": tokenizers,
        }
        path = Path(path)
        os.makedirs(path.parent or Path("."), exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
