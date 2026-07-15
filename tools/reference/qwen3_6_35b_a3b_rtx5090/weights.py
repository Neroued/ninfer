"""Budget-aware tensor materialization over the typed 35B binding."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterator, Sequence
import warnings

import torch

from tools.artifact import (
    decode_direct,
    dequantize_row_split,
    gather_row_planes,
    row_split_geometry,
    split_row_planes,
)

from .bindings import (
    ArtifactBinding,
    AxisView,
    LogicalRowView,
    PhysicalBlock,
    RowAddressable,
    WeightObject,
)
from .config import CFG


GIB = 1 << 30
PROGRAM_WORKSPACE_BYTES = 256 << 20
_DIRECT_BYTES = {"BF16": 2, "FP32": 4, "I32": 4}


@dataclass(frozen=True, slots=True)
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


def estimate_fixed_bytes(
    capacity: int,
    kv_dtype: str,
    *,
    text: bool,
    mtp: bool,
    prefill_chunk: int,
) -> int:
    """Estimate target state plus the bounded Text/MTP workspace."""

    if not text and not mtp:
        return 0
    kv_layers = (CFG.full_layers if text else 0) + (1 if mtp else 0)
    if kv_dtype == "bf16":
        kv_per_layer_token = 2 * CFG.kv_heads * CFG.head_dim * 2
    elif kv_dtype == "int8":
        kv_per_layer_token = 2 * CFG.kv_heads * (
            CFG.head_dim + CFG.head_dim // 64 * 2
        )
    else:
        raise ValueError(f"unsupported KV dtype: {kv_dtype!r}")
    kv = capacity * kv_layers * kv_per_layer_token
    if text:
        ssm = (
            CFG.gdn_layers
            * CFG.gdn_v_heads
            * CFG.gdn_k_dim
            * CFG.gdn_v_dim
            * 4
        )
        conv = (
            CFG.gdn_layers
            * CFG.conv_dim
            * (CFG.conv_width - 1)
            * CFG.conv_state_bytes
        )
    else:
        ssm = 0
        conv = 0
    workspace_chunks = max(1, (prefill_chunk + CFG.prefill_chunk - 1) // CFG.prefill_chunk)
    workspace = workspace_chunks * PROGRAM_WORKSPACE_BYTES
    return kv + ssm + conv + workspace


class WeightStore:
    """Materialize regular blocks and selected expert rows under one budget."""

    def __init__(
        self,
        binding: ArtifactBinding,
        device: torch.device | str,
        *,
        capacity: int,
        kv_dtype: str = "bf16",
        text: bool = True,
        mtp: bool = False,
        draft_head: bool = False,
        vision: bool = False,
        compile_codec: bool = False,
        prefill_chunk: int = CFG.prefill_chunk,
        memory_bytes: int | None = None,
        headroom_bytes: int = 2 * GIB,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        if memory_bytes is not None and memory_bytes < 0:
            raise ValueError("memory_bytes must be nonnegative")
        if headroom_bytes < 0:
            raise ValueError("headroom_bytes must be nonnegative")
        if not any((text, mtp, draft_head, vision)):
            raise ValueError("at least one weight component must be selected")

        self.binding = binding
        self.device = torch.device(device)
        self.compile_codec = compile_codec
        self._packed: dict[int, torch.Tensor] = {}
        self._decoded: dict[int, torch.Tensor] = {}
        self._axis_decoded: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        self._representation: dict[int, str] = {}
        self._expert_block_ids = frozenset(
            bank.block.tensor_id for bank in binding.expert_banks
        )

        selected: dict[int, PhysicalBlock] = {}

        def add(blocks: Sequence[PhysicalBlock]) -> None:
            selected.update((block.tensor_id, block) for block in blocks)

        if text:
            add(binding.blocks_for("text"))
        if mtp:
            add(binding.blocks_for("mtp"))
            add((binding.mtp.token_embedding, binding.mtp.full_output_head))
        if draft_head:
            add(binding.blocks_for("draft"))
        if vision:
            add(binding.blocks_for("vision"))
        blocks = tuple(selected[index] for index in sorted(selected))

        fixed = estimate_fixed_bytes(
            capacity,
            kv_dtype,
            text=text,
            mtp=mtp,
            prefill_chunk=prefill_chunk,
        )
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                free, _ = torch.cuda.mem_get_info()
        else:
            free = memory_bytes if memory_bytes is not None else fixed
            headroom_bytes = 0
        available = min(free, memory_bytes) if memory_bytes is not None else free
        budget = max(0, available - headroom_bytes - fixed)
        used = 0

        controls = [
            block for block in blocks if block.layout == "contiguous-le-v1"
        ]
        for block in controls:
            size = self._decoded_size(block)
            if block.component == "vision":
                self._representation[block.tensor_id] = "decoded"
                used += size
                continue
            representation = "decoded" if used + size <= budget else "stream"
            self._representation[block.tensor_id] = representation
            if representation == "decoded":
                used += size

        quantized = [
            block for block in blocks if block.layout == "row-split-k128-v1"
        ]
        for block in sorted(
            quantized, key=lambda item: item.payload_bytes, reverse=True
        ):
            if block.component == "vision":
                self._representation[block.tensor_id] = "stream"
            elif used + block.payload_bytes <= budget:
                self._representation[block.tensor_id] = "packed"
                used += block.payload_bytes
            else:
                self._representation[block.tensor_id] = "stream"

        decoded = sum(
            self._decoded_size(block)
            for block in blocks
            if self._representation[block.tensor_id] == "decoded"
        )
        packed = sum(
            block.payload_bytes
            for block in blocks
            if self._representation[block.tensor_id] == "packed"
        )
        streamed = sum(
            block.payload_bytes
            for block in blocks
            if self._representation[block.tensor_id] == "stream"
        )
        self.plan = MemoryPlan(
            free_bytes=free,
            headroom_bytes=headroom_bytes,
            fixed_bytes=fixed,
            weight_budget_bytes=budget,
            decoded_bytes=decoded,
            packed_bytes=packed,
            streamed_bytes=streamed,
            decoded_blocks=sum(
                value == "decoded" for value in self._representation.values()
            ),
            packed_blocks=sum(
                value == "packed" for value in self._representation.values()
            ),
            streamed_blocks=sum(
                value == "stream" for value in self._representation.values()
            ),
        )

    @staticmethod
    def _decoded_size(block: PhysicalBlock) -> int:
        return prod(block.shape) * _DIRECT_BYTES[block.format]

    @staticmethod
    def _physical(value: WeightObject) -> PhysicalBlock:
        return value if isinstance(value, PhysicalBlock) else value.block

    def representation(self, value: WeightObject) -> str:
        return self._representation.get(self._physical(value).tensor_id, "stream")

    @staticmethod
    def _readonly_tensor(payload: memoryview) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            return torch.frombuffer(payload, dtype=torch.uint8)

    def _stream_tensor(self, block: PhysicalBlock) -> torch.Tensor:
        host = self._readonly_tensor(self.binding.payload(block))
        return host if self.device.type == "cpu" else host.to(self.device)

    def _upload(self, block: PhysicalBlock) -> torch.Tensor:
        cached = self._packed.get(block.tensor_id)
        if cached is not None:
            return cached
        if self.device.type == "cpu":
            payload = torch.frombuffer(
                bytearray(self.binding.payload(block)), dtype=torch.uint8
            )
        else:
            payload = self._stream_tensor(block)
        if self.representation(block) == "packed":
            self._packed[block.tensor_id] = payload
        return payload

    def _decode_block(
        self,
        block: PhysicalBlock,
        *,
        dequant_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if block.tensor_id in self._expert_block_ids:
            raise ValueError("expert banks must be materialized by selected rows")
        cached = self._decoded.get(block.tensor_id)
        if cached is not None:
            return cached
        representation = self.representation(block)
        if block.layout == "contiguous-le-v1":
            source = (
                self._upload(block)
                if representation == "decoded"
                else self._stream_tensor(block)
            )
            decoded = decode_direct(
                source, block.format, block.shape, device=self.device
            )
            if representation == "stream" and self.device.type == "cpu":
                decoded = decoded.clone()
        else:
            source = (
                self._upload(block)
                if representation == "packed"
                else self._stream_tensor(block)
            )
            decoded = dequantize_row_split(
                source,
                block.format,
                block.shape,
                device=self.device,
                dtype=dequant_dtype,
                compiled=self.compile_codec,
            )
        if representation == "decoded":
            self._decoded[block.tensor_id] = decoded
        return decoded

    def tensor(
        self,
        value: WeightObject,
        *,
        dequant_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if isinstance(value, PhysicalBlock):
            return self._decode_block(value, dequant_dtype=dequant_dtype)
        if isinstance(value, AxisView):
            key = (value.block.tensor_id, value.axes)
            cached = self._axis_decoded.get(key)
            if cached is not None:
                return cached
            decoded = self._decode_block(
                value.block,
                dequant_dtype=dequant_dtype,
            ).permute(value.axes).contiguous()
            if tuple(decoded.shape) != value.shape:
                raise RuntimeError("axis view produced an unexpected shape")
            if self.representation(value.block) == "decoded":
                self._axis_decoded[key] = decoded
            return decoded

        block = value.block
        if block.layout == "contiguous-le-v1":
            return self._decode_block(
                block,
                dequant_dtype=dequant_dtype,
            )[value.row_begin : value.row_end]
        if self.representation(block) == "decoded":
            return self._decode_block(
                block,
                dequant_dtype=dequant_dtype,
            )[value.row_begin : value.row_end]
        geometry = row_split_geometry(block.format, block.shape)
        source = (
            self._upload(block)
            if self.representation(block) == "packed"
            else self.binding.payload(block)
        )
        planes = split_row_planes(
            source, geometry, value.row_begin, value.row_count
        )
        return dequantize_row_split(
            planes,
            block.format,
            value.shape,
            device=self.device,
            dtype=dequant_dtype,
            compiled=self.compile_codec,
        )

    @staticmethod
    def _row_span(value: RowAddressable) -> tuple[PhysicalBlock, int, int]:
        if isinstance(value, PhysicalBlock):
            if len(value.shape) != 2:
                raise ValueError("row access requires a rank-two block")
            return value, 0, value.shape[0]
        return value.block, value.row_begin, value.row_count

    @staticmethod
    def _relative_rows(
        rows: Sequence[int] | torch.Tensor,
        count: int,
        *,
        check_bounds: bool,
    ) -> Sequence[int] | torch.Tensor:
        if isinstance(rows, torch.Tensor):
            if rows.dim() != 1 or rows.numel() == 0:
                raise ValueError("rows must be a nonempty one-dimensional tensor")
            if rows.dtype not in (torch.int32, torch.int64):
                raise TypeError("row-index tensors must use int32 or int64")
            if check_bounds and (int(rows.min()) < 0 or int(rows.max()) >= count):
                raise IndexError("row index is outside the logical view")
            return rows
        indices = tuple(rows)
        if not indices:
            raise ValueError("at least one row is required")
        if check_bounds and (min(indices) < 0 or max(indices) >= count):
            raise IndexError("row index is outside the logical view")
        return indices

    def rows(
        self,
        value: RowAddressable,
        rows: Sequence[int] | torch.Tensor,
        *,
        dequant_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        block, row_begin, row_count = self._row_span(value)
        if block.layout != "row-split-k128-v1":
            raise ValueError("block is not row-addressable")
        relative = self._relative_rows(
            rows,
            row_count,
            check_bounds=isinstance(value, LogicalRowView),
        )
        if isinstance(relative, torch.Tensor):
            absolute: Sequence[int] | torch.Tensor = relative + row_begin
        else:
            absolute = tuple(row_begin + row for row in relative)

        if self.representation(block) == "decoded":
            indices = torch.as_tensor(
                absolute, dtype=torch.long, device=self.device
            )
            return self._decode_block(block).index_select(0, indices)
        source = (
            self._upload(block)
            if self.representation(block) == "packed"
            else self.binding.payload(block)
        )
        geometry = row_split_geometry(block.format, block.shape)
        planes = gather_row_planes(source, geometry, absolute)
        return dequantize_row_split(
            planes,
            block.format,
            (planes.rows, block.shape[1]),
            device=self.device,
            dtype=dequant_dtype,
            compiled=self.compile_codec,
        )

    def chunks(
        self,
        value: RowAddressable,
        rows: int = 8192,
        *,
        dequant_dtype: torch.dtype = torch.bfloat16,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        if rows <= 0:
            raise ValueError("chunk rows must be positive")
        block, row_begin, row_count = self._row_span(value)
        if block.layout != "row-split-k128-v1":
            raise ValueError("block is not row-addressable")
        if block.tensor_id in self._expert_block_ids:
            raise ValueError("expert banks are addressed by selected rows, not chunks")
        if self.representation(block) == "decoded":
            decoded = self._decode_block(block)
            for local_begin in range(0, row_count, rows):
                local_end = min(row_count, local_begin + rows)
                yield (
                    local_begin,
                    local_end,
                    decoded[row_begin + local_begin : row_begin + local_end],
                )
            return

        source = (
            self._upload(block)
            if self.representation(block) == "packed"
            else self.binding.payload(block)
        )
        geometry = row_split_geometry(block.format, block.shape)
        for local_begin in range(0, row_count, rows):
            local_end = min(row_count, local_begin + rows)
            count = local_end - local_begin
            planes = split_row_planes(
                source, geometry, row_begin + local_begin, count
            )
            decoded = dequantize_row_split(
                planes,
                block.format,
                (count, block.shape[1]),
                device=self.device,
                dtype=dequant_dtype,
                compiled=self.compile_codec,
            )
            yield local_begin, local_end, decoded

    def close(self) -> None:
        self._axis_decoded.clear()
        self._decoded.clear()
        self._packed.clear()


__all__ = [
    "GIB",
    "MemoryPlan",
    "PROGRAM_WORKSPACE_BYTES",
    "WeightStore",
    "estimate_fixed_bytes",
]
