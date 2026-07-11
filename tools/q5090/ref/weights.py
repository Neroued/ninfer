"""Static, budget-aware q5090 weight residency."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from tools.q5090.reader import Block, Reader
from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt

from .config import CFG


GIB = 1 << 30


@dataclass(frozen=True)
class MemoryPlan:
    free_bytes: int
    headroom_bytes: int
    fixed_bytes: int
    weight_budget_bytes: int
    decoded_bytes: int
    packed_bytes: int
    streamed_bytes: int
    decoded_blocks: int
    packed_blocks: int
    streamed_blocks: int

    def summary(self) -> str:
        gib = lambda value: value / GIB
        return (
            f"free={gib(self.free_bytes):.2f}GiB "
            f"headroom={gib(self.headroom_bytes):.2f}GiB "
            f"fixed={gib(self.fixed_bytes):.2f}GiB "
            f"weights={gib(self.decoded_bytes + self.packed_bytes):.2f}GiB "
            f"(decoded={gib(self.decoded_bytes):.2f}GiB/{self.decoded_blocks}, "
            f"packed={gib(self.packed_bytes):.2f}GiB/{self.packed_blocks}, "
            f"stream={gib(self.streamed_bytes):.2f}GiB/{self.streamed_blocks})"
        )


def estimate_fixed_bytes(capacity: int, kv_dtype: str, mtp: bool, prefill_chunk: int) -> int:
    kv_layers = CFG.full_layers + (1 if mtp else 0)
    if kv_dtype == "bf16":
        kv_per_layer_token = 2 * CFG.kv_heads * CFG.head_dim * 2
    else:
        kv_per_layer_token = 2 * CFG.kv_heads * (CFG.head_dim + CFG.head_dim // 64 * 2)
    kv = capacity * kv_layers * kv_per_layer_token
    ssm = CFG.gdn_layers * CFG.gdn_v_heads * CFG.gdn_k_dim * CFG.gdn_v_dim * 4
    conv = CFG.gdn_layers * CFG.conv_dim * (CFG.conv_width - 1) * 4
    # Largest model activation plus FLA/SDPA/compiler scratch. This is deliberately
    # conservative; actual peaks are reported after a run.
    scratch = max(2 * GIB, prefill_chunk * CFG.intermediate * 8)
    return kv + ssm + conv + scratch


class WeightStore:
    def __init__(
        self,
        reader: Reader,
        device: torch.device,
        *,
        capacity: int,
        kv_dtype: str,
        mtp: bool,
        draft_head: bool,
        compile_codec: bool,
        prefill_chunk: int,
        memory_bytes: int | None = None,
        headroom_bytes: int = 2 * GIB,
    ):
        self.reader = reader
        self.device = device
        self._packed: dict[int, torch.Tensor] = {}
        self._decoded: dict[int, torch.Tensor] = {}
        self._representation: dict[int, str] = {}
        self.compile_codec = compile_codec
        fixed = estimate_fixed_bytes(capacity, kv_dtype, mtp, prefill_chunk)
        if device.type == "cuda":
            with torch.cuda.device(device):
                free, _ = torch.cuda.mem_get_info()
        else:
            # CPU mode is a slow diagnostic fallback. Default to streaming so
            # it never attempts to cache the 50 GiB decoded text core.
            free = memory_bytes if memory_bytes is not None else fixed
            headroom_bytes = 0
        available = min(free, memory_bytes) if memory_bytes is not None else free
        budget = max(0, available - headroom_bytes - fixed)
        relevant_modules = {qt.MODULE_TEXT}
        if mtp:
            relevant_modules.add(qt.MODULE_MTP)
        if draft_head:
            relevant_modules.add(qt.MODULE_LM_HEAD_DRAFT)
        blocks = [block for block in reader.blocks if block.module_kind in relevant_modules]
        used = 0

        # Control tensors are tiny and frequently reused.
        controls = [block for block in blocks if block.layout == qt.LAYOUT_CONTIGUOUS]
        for block in controls:
            size = self._decoded_size(block)
            representation = "decoded" if used + size <= budget else "stream"
            self._representation[block.index] = representation
            if representation == "decoded":
                used += size

        quant = [block for block in blocks if block.layout == qt.LAYOUT_ROW_SPLIT]
        # Packed residency prevents per-token PCIe traffic. Largest blocks first
        # maximizes protected bytes when the whole text module does not fit.
        for block in sorted(quant, key=lambda item: item.payload_bytes, reverse=True):
            if used + block.payload_bytes <= budget:
                self._representation[block.index] = "packed"
                used += block.payload_bytes
            else:
                self._representation[block.index] = "stream"

        decoded = sum(
            self._decoded_size(block) for block in blocks
            if self._representation[block.index] == "decoded"
        )
        packed = sum(
            block.payload_bytes for block in blocks
            if self._representation[block.index] == "packed"
        )
        streamed = sum(
            block.payload_bytes for block in blocks
            if self._representation[block.index] == "stream"
        )
        self.plan = MemoryPlan(
            free_bytes=free,
            headroom_bytes=headroom_bytes,
            fixed_bytes=fixed,
            weight_budget_bytes=budget,
            decoded_bytes=decoded,
            packed_bytes=packed,
            streamed_bytes=streamed,
            decoded_blocks=sum(v == "decoded" for v in self._representation.values()),
            packed_blocks=sum(v == "packed" for v in self._representation.values()),
            streamed_blocks=sum(v == "stream" for v in self._representation.values()),
        )

    @staticmethod
    def _decoded_size(block: Block) -> int:
        dtype_size = 4 if block.qtype == qt.QT_FP32 else 4 if block.qtype == qt.QT_I32 else 2
        return prod(block.shape) * dtype_size

    def representation(self, block: Block) -> str:
        return self._representation.get(block.index, "stream")

    def _upload(self, block: Block) -> torch.Tensor:
        cached = self._packed.get(block.index)
        if cached is not None:
            return cached
        host = torch.frombuffer(bytearray(self.reader.payload(block)), dtype=torch.uint8)
        payload = host.to(self.device)
        if self.representation(block) == "packed":
            self._packed[block.index] = payload
        return payload

    def _block(self, block: Block) -> torch.Tensor:
        cached = self._decoded.get(block.index)
        if cached is not None:
            return cached
        representation = self.representation(block)
        payload = self._upload(block) if representation in {"packed", "decoded"} else None
        decoded = self.reader.decode_block(
            block,
            self.device,
            payload=payload,
            compiled=self.compile_codec,
        )
        if representation == "decoded":
            self._decoded[block.index] = decoded
        return decoded

    def block(self, name: str) -> torch.Tensor:
        return self._block(self.reader.entries[name])

    def tensor(self, name: str) -> torch.Tensor:
        view = self.reader.views[name]
        block = view.block
        if self.representation(block) == "decoded":
            decoded = self._block(block)
            if view.row_begin == 0 and view.row_count == block.shape[0]:
                return decoded
            return decoded[view.row_begin:view.row_begin + view.row_count]
        payload = self._upload(block) if self.representation(block) == "packed" else None
        return self.reader.decode_view(
            view,
            self.device,
            payload=payload,
            compiled=self.compile_codec,
        )

    def rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        view = self.reader.views[name]
        block = view.block
        if block.layout != qt.LAYOUT_ROW_SPLIT:
            raise ValueError(f"{name} is not row-addressable")
        if self.representation(block) == "decoded":
            return self._block(block).index_select(0, rows.to(self.device))
        payload = self._upload(block) if self.representation(block) == "packed" else None
        if payload is None:
            source = self.reader.payload(block)
            _, k, kp, rn, rh, rs, high_off, scale_off = self.reader.row_geometry(block)
            indices = [view.row_begin + int(row) for row in rows.cpu().tolist()]
            nibble = b"".join(source[index * rn:(index + 1) * rn] for index in indices)
            high = b"".join(
                source[high_off + index * rh:high_off + (index + 1) * rh]
                for index in indices
            )
            scale = b"".join(
                source[scale_off + index * rs:scale_off + (index + 1) * rs]
                for index in indices
            )
            npad = fmt.align_up(len(nibble), fmt.PAYLOAD_ALIGN) - len(nibble)
            hpad = fmt.align_up(len(high), fmt.PAYLOAD_ALIGN) - len(high) if high else 0
            from tools.q5090.codec import decode_tensor

            return decode_tensor(
                nibble + b"\0" * npad + high + b"\0" * hpad + scale,
                block.qtype,
                block.layout,
                [len(indices), k],
                [len(indices), kp],
                self.device,
                output_dtype=torch.bfloat16,
                compiled=self.compile_codec,
            )
        _, k, kp, rn, rh, rs, high_off, scale_off = self.reader.row_geometry(block)
        indices = rows.to(device=self.device, dtype=torch.long) + view.row_begin
        nibble = (
            payload[:block.nibble_plane_bytes]
            .reshape(block.shape[0], rn)
            .index_select(0, indices)
            .flatten()
        )
        high = (
            payload[high_off:high_off + block.high_plane_bytes]
            .reshape(block.shape[0], rh)
            .index_select(0, indices)
            .flatten()
            if rh
            else payload[:0]
        )
        scale = (
            payload[scale_off:scale_off + block.scale_plane_bytes]
            .reshape(block.shape[0], rs)
            .index_select(0, indices)
            .flatten()
        )
        npad = fmt.align_up(nibble.numel(), fmt.PAYLOAD_ALIGN) - nibble.numel()
        hpad = fmt.align_up(high.numel(), fmt.PAYLOAD_ALIGN) - high.numel() if high.numel() else 0
        parts = [nibble]
        if npad:
            parts.append(torch.zeros(npad, dtype=torch.uint8, device=self.device))
        if high.numel():
            parts.append(high)
            if hpad:
                parts.append(torch.zeros(hpad, dtype=torch.uint8, device=self.device))
        parts.append(scale)
        gathered = torch.cat(parts)
        from tools.q5090.codec import decode_tensor

        return decode_tensor(
            gathered,
            block.qtype,
            block.layout,
            [indices.numel(), k],
            [indices.numel(), kp],
            self.device,
            output_dtype=torch.bfloat16,
            compiled=self.compile_codec,
        )

    def chunks(self, name: str, rows: int = 8192):
        view = self.reader.views[name]
        block = view.block
        if self.representation(block) == "decoded":
            decoded = self._block(block)
            for row0 in range(0, view.row_count, rows):
                row1 = min(view.row_count, row0 + rows)
                yield row0, row1, decoded[
                    view.row_begin + row0:view.row_begin + row1
                ]
            return
        payload = self._upload(block) if self.representation(block) == "packed" else None
        yield from self.reader.row_chunks(
            view,
            self.device,
            rows,
            payload=payload,
            compiled=self.compile_codec,
        )
