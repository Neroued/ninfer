"""Qwen3.6-27B q5090 reference schedule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from tools.q5090.reader import Reader
from tools.q5090.tokenizer import Tokenizer
from tools.q5090_convert import format as fmt

from .config import CFG
from . import mtp as mtp_schedule
from .multimodal import MultimodalBatch
from .ops import attention, linear, rmsnorm
from .state import ModelState, StateSnapshot
from .tap import NullTap
from .text import run as run_text
from .vision import VisionEncoder, VisionOutput, VisionStats
from .weights import WeightStore


COMPILED_CODEC_MIN_TOKENS = 12


@dataclass
class ModelSnapshot:
    state: StateSnapshot
    last_hidden: torch.Tensor | None
    last_draft: int | None


class RefModel:
    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | torch.device = "cuda",
        memory_bytes: int | None = None,
        headroom_bytes: int = 2 << 30,
        kv_dtype: str = "bf16",
        prefill_chunk: int = CFG.prefill_chunk,
        mtp: bool = False,
        draft_head: bool = False,
        compile_codec: bool | None = None,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but the active PyTorch has no CUDA support")
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        if memory_bytes is not None and memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if headroom_bytes < 0:
            raise ValueError("headroom_bytes must be nonnegative")
        if draft_head and not mtp:
            raise ValueError("draft_head requires mtp")
        self.reader = Reader(weights)
        self.tokenizer = Tokenizer(self.reader)
        CFG.validate_header(self.reader.header)
        if mtp and not (self.reader.header["flags"] & fmt.FLAG_MTP_PRESENT):
            raise ValueError("MTP was requested but the q5090 artifact has no MTP module")
        if draft_head and not (
            self.reader.header["flags"] & fmt.FLAG_LM_HEAD_DRAFT_PRESENT
        ):
            raise ValueError("draft head was requested but the q5090 artifact has no draft module")
        self.memory_bytes = memory_bytes
        self.headroom_bytes = headroom_bytes
        self.kv_dtype = kv_dtype
        self.prefill_chunk = prefill_chunk
        self.mtp_enabled = mtp
        self.draft_head = draft_head
        self.compile_codec = compile_codec
        self.active_compile_codec: bool | None = None
        self.weights: WeightStore | None = None
        self.state: ModelState | None = None
        self.last_hidden: torch.Tensor | None = None
        self.last_draft: int | None = None
        self.vision_stats: VisionStats | None = None

    def prepare(self, capacity: int, *, compile_codec: bool | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if capacity > self.reader.header["max_position_embeddings"]:
            raise ValueError("capacity exceeds q5090 max_position_embeddings")
        self.weights = None
        self.state = None
        self.last_hidden = None
        self.last_draft = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if compile_codec is None:
            compile_codec = self.compile_codec if self.compile_codec is not None else False
        self.weights = WeightStore(
            self.reader,
            self.device,
            capacity=capacity,
            kv_dtype=self.kv_dtype,
            mtp=self.mtp_enabled,
            draft_head=self.draft_head,
            compile_codec=compile_codec,
            prefill_chunk=self.prefill_chunk,
            memory_bytes=self.memory_bytes,
            headroom_bytes=self.headroom_bytes,
        )
        self.state = ModelState(self.device, capacity, self.kv_dtype)
        self.last_hidden = None
        self.active_compile_codec = compile_codec

    def encode_vision(
        self,
        batch: MultimodalBatch,
        *,
        compile_codec: bool = True,
        attention_limit: int | None = None,
        tap=None,
    ) -> VisionOutput:
        encoder = VisionEncoder(
            self.reader,
            self.device,
            compile_codec=compile_codec,
            memory_bytes=self.memory_bytes,
            headroom_bytes=self.headroom_bytes,
            attention_limit=attention_limit,
        )
        try:
            output = encoder.encode(
                batch.pixel_values,
                batch.image_grid_thw,
                batch.pixel_values_videos,
                batch.video_grid_thw,
                tap=tap,
            )
        finally:
            encoder.close()
        self.vision_stats = output.stats
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return output

    def reset(self, capacity: int) -> None:
        self.vision_stats = None
        self.prepare(capacity, compile_codec=self.active_compile_codec)

    def _ready(self) -> tuple[WeightStore, ModelState]:
        if self.weights is None or self.state is None:
            raise RuntimeError("call prepare(), prefill(), or generate() before model operations")
        return self.weights, self.state

    def weight(self, name: str) -> torch.Tensor:
        weights, _ = self._ready()
        return weights.tensor(name)

    def block_weight(self, name: str) -> torch.Tensor:
        weights, _ = self._ready()
        return weights.block(name)

    def embed(self, ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        weights, _ = self._ready()
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(list(ids), device=self.device, dtype=torch.long)
        else:
            ids = ids.to(device=self.device, dtype=torch.long)
        return weights.rows("model.language_model.embed_tokens.weight", ids)

    @staticmethod
    def _tap(tap, name, value, *, phase, step, chunk, position):
        tap(
            name,
            value,
            phase=phase,
            step=step,
            chunk=chunk,
            position_begin=position,
        )

    def _gqa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        start: int,
        *,
        mtp: bool = False,
    ) -> torch.Tensor:
        _, state = self._ready()
        cache = state.mtp_kv if mtp else state.kv
        if cache.length != start:
            raise ValueError(
                f"KV append start {start} does not match resident length {cache.length}"
            )
        cache.write(layer, start, k, v)
        end = start + q.shape[0]
        k_all, v_all = cache.read(layer, end)
        return attention(q, k_all, v_all, causal=q.shape[0] > 1)

    def final_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight("model.language_model.norm.weight"))

    def logits_last(self, hidden: torch.Tensor, *, draft: bool = False) -> torch.Tensor:
        weights, _ = self._ready()
        last = hidden[-1:].to(torch.bfloat16)
        if draft:
            logits = linear(last, weights.block("lm_head_draft"))[0]
            ids = weights.tensor("lm_head_draft.idmap").long()
            full = torch.full((CFG.vocab,), -torch.inf, device=self.device, dtype=torch.float32)
            full[ids] = logits.float()
            return full
        block = self.reader.views["lm_head.weight"].block
        if weights.representation(block) == "decoded":
            return (last @ weights.block("lm_head.weight").t())[0]
        logits = torch.empty(CFG.vocab, device=self.device, dtype=torch.bfloat16)
        for row0, row1, weight in weights.chunks("lm_head.weight"):
            logits[row0:row1] = (last @ weight.t())[0]
        return logits

    def prefill(
        self,
        ids: Iterable[int],
        *,
        capacity: int | None = None,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> int:
        ids = list(ids)
        if not ids:
            raise ValueError("prefill ids must not be empty")
        if self.state is None:
            self.prepare(capacity or len(ids) + 1)
        _, state = self._ready()
        if state.position + len(ids) > state.capacity:
            raise ValueError("prefill exceeds prepared context capacity")
        tap = tap or NullTap()
        last_hidden = None
        for chunk, offset in enumerate(range(0, len(ids), self.prefill_chunk)):
            part = ids[offset:offset + self.prefill_chunk]
            start = state.position
            x = self.embed(part)
            self._tap(tap, "embed", x, phase="prefill", step=0, chunk=chunk, position=start)
            positions = torch.arange(
                start, start + len(part), device=self.device, dtype=torch.int32
            )
            x = run_text(
                self,
                x,
                positions,
                start,
                phase="prefill",
                step=0,
                chunk=chunk,
                tap=tap,
            )
            state.position += len(part)
            state.kv.length = state.position
            last_hidden = self.final_hidden(x)
            self._tap(
                tap,
                "final_norm",
                last_hidden,
                phase="prefill",
                step=0,
                chunk=chunk,
                position=start,
            )
            if self.mtp_enabled:
                known = min(len(part), len(ids) - 1 - offset)
                if known > 0:
                    self.mtp_forward(
                        ids[offset + 1:offset + 1 + known],
                        last_hidden[:known],
                        positions[:known],
                        start=state.mtp_kv.length,
                        sample=False,
                    )
        assert last_hidden is not None
        logits = self.logits_last(last_hidden)
        target = sampler(logits) if sampler is not None else int(torch.argmax(logits).item())
        self.last_hidden = last_hidden
        if self.mtp_enabled:
            _, draft = self.mtp_forward(
                [target],
                last_hidden[-1:],
                torch.tensor([state.position - 1], device=self.device, dtype=torch.int32),
                start=state.mtp_kv.length,
            )
            self.last_draft = draft
        self._tap(
            tap,
            "logits",
            logits,
            phase="prefill",
            step=0,
            chunk=chunk,
            position=state.position - 1,
        )
        return target

    def prefill_multimodal(
        self,
        batch: MultimodalBatch,
        vision: VisionOutput,
        *,
        capacity: int | None = None,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> int:
        ids = batch.input_ids.tolist()
        if not ids:
            raise ValueError("multimodal prompt must not be empty")
        if self.state is None:
            self.prepare(capacity or len(ids) + 1)
        _, state = self._ready()
        if state.position != 0:
            raise ValueError("multimodal prefill requires a fresh model state")
        if state.position + len(ids) > state.capacity:
            raise ValueError("multimodal prefill exceeds prepared context capacity")
        types = batch.mm_token_type_ids.flatten()
        if types.numel() != len(ids) or batch.position_ids.shape != (3, len(ids)):
            raise ValueError("multimodal token metadata shape mismatch")
        image_count = int((types == 1).sum().item())
        video_count = int((types == 2).sum().item())
        if image_count != (0 if vision.image_embeddings is None else vision.image_embeddings.shape[0]):
            raise ValueError("image placeholder/embedding count mismatch")
        if video_count != (0 if vision.video_embeddings is None else vision.video_embeddings.shape[0]):
            raise ValueError("video placeholder/embedding count mismatch")
        state.mrope = True
        state.rope_delta = batch.rope_delta
        tap = tap or NullTap()
        image_cursor = 0
        video_cursor = 0
        last_hidden = None
        last_positions = None
        for chunk, offset in enumerate(range(0, len(ids), self.prefill_chunk)):
            part = ids[offset:offset + self.prefill_chunk]
            start = state.position
            x = self.embed(part)
            part_types = types[offset:offset + len(part)].to(device=self.device)
            image_mask = part_types == 1
            video_mask = part_types == 2
            nimage = int(image_mask.sum().item())
            nvideo = int(video_mask.sum().item())
            if nimage:
                assert vision.image_embeddings is not None
                x[image_mask] = vision.image_embeddings[image_cursor:image_cursor + nimage]
                image_cursor += nimage
            if nvideo:
                assert vision.video_embeddings is not None
                x[video_mask] = vision.video_embeddings[video_cursor:video_cursor + nvideo]
                video_cursor += nvideo
            self._tap(tap, "embed", x, phase="prefill", step=0, chunk=chunk, position=start)
            positions = batch.position_ids[:, offset:offset + len(part)].to(
                device=self.device, dtype=torch.int32
            )
            x = run_text(
                self,
                x,
                positions,
                start,
                phase="prefill",
                step=0,
                chunk=chunk,
                tap=tap,
            )
            state.position += len(part)
            state.kv.length = state.position
            last_hidden = self.final_hidden(x)
            last_positions = positions
            self._tap(
                tap,
                "final_norm",
                last_hidden,
                phase="prefill",
                step=0,
                chunk=chunk,
                position=start,
            )
            if self.mtp_enabled:
                known = min(len(part), len(ids) - 1 - offset)
                if known > 0:
                    self.mtp_forward(
                        ids[offset + 1:offset + 1 + known],
                        last_hidden[:known],
                        positions[..., :known],
                        start=state.mtp_kv.length,
                        sample=False,
                    )
        assert last_hidden is not None and last_positions is not None
        if image_cursor != image_count or video_cursor != video_count:
            raise RuntimeError("not all visual embeddings were consumed during prefill")
        logits = self.logits_last(last_hidden)
        target = sampler(logits) if sampler is not None else int(torch.argmax(logits).item())
        self.last_hidden = last_hidden
        if self.mtp_enabled:
            _, draft = self.mtp_forward(
                [target],
                last_hidden[-1:],
                last_positions[..., -1:],
                start=state.mtp_kv.length,
            )
            self.last_draft = draft
        self._tap(
            tap,
            "logits",
            logits,
            phase="prefill",
            step=0,
            chunk=chunk,
            position=state.position - 1,
        )
        return target

    def _decode(
        self,
        token: int,
        *,
        step: int = 0,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        _, state = self._ready()
        if state.position >= state.capacity:
            raise ValueError("decode exceeds prepared context capacity")
        tap = tap or NullTap()
        start = state.position
        x = self.embed([token])
        self._tap(tap, "embed", x, phase="decode", step=step, chunk=0, position=start)
        if state.mrope:
            value = start + state.rope_delta
            positions = torch.full((3, 1), value, device=self.device, dtype=torch.int32)
        else:
            positions = torch.tensor([start], device=self.device, dtype=torch.int32)
        x = run_text(
            self,
            x,
            positions,
            start,
            phase="decode",
            step=step,
            chunk=0,
            tap=tap,
        )
        state.position += 1
        state.kv.length = state.position
        hidden = self.final_hidden(x)
        self._tap(tap, "final_norm", hidden, phase="decode", step=step, chunk=0, position=start)
        logits = self.logits_last(hidden)
        self.last_hidden = hidden
        self._tap(tap, "logits", logits, phase="decode", step=step, chunk=0, position=start)
        target = sampler(logits) if sampler is not None else int(torch.argmax(logits).item())
        if self.mtp_enabled:
            _, self.last_draft = self.mtp_forward(
                [target],
                hidden,
                positions,
                start=state.mtp_kv.length,
            )
        return target, hidden, logits

    def decode(
        self,
        token: int,
        *,
        step: int = 0,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> int:
        return self._decode(token, step=step, sampler=sampler, tap=tap)[0]

    def snapshot(self) -> ModelSnapshot:
        _, state = self._ready()
        return ModelSnapshot(
            state.snapshot(),
            None if self.last_hidden is None else self.last_hidden.clone(),
            self.last_draft,
        )

    def restore(self, snapshot: ModelSnapshot) -> None:
        _, state = self._ready()
        state.restore(snapshot.state)
        self.last_hidden = snapshot.last_hidden
        self.last_draft = snapshot.last_draft

    def target_verify(self, current: int, drafts: Iterable[int]) -> list[int]:
        """Greedy target verification using explicit recurrent state transitions.

        This is intentionally sequential: it is a correctness oracle for the
        speculative schedule, not a duplicate of the C++ small-T fused path.
        """
        drafts = list(drafts)
        accepted: list[int] = []
        token = current
        for draft in drafts:
            target = self.decode(token)
            if target != draft:
                return accepted + [target]
            accepted.append(draft)
            token = draft
        return accepted + [self.decode(token)]

    def mtp_forward(
        self,
        ids: Iterable[int],
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        start: int,
        sample: bool = True,
    ) -> tuple[torch.Tensor, int | None]:
        return mtp_schedule.forward(
            self,
            ids,
            hidden,
            positions,
            start=start,
            sample=sample,
        )

    def generate(
        self,
        prompt: Iterable[int],
        max_new_tokens: int,
        *,
        stop_token_ids: set[int] | None = None,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> list[int]:
        prompt = list(prompt)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if not prompt or max_new_tokens == 0:
            return []
        compile_codec = self.compile_codec
        if compile_codec is None:
            compile_codec = max_new_tokens >= COMPILED_CODEC_MIN_TOKENS
        self.vision_stats = None
        self.prepare(len(prompt) + max_new_tokens, compile_codec=compile_codec)
        token = self.prefill(prompt, sampler=sampler, tap=tap)
        output = [token]
        while len(output) < max_new_tokens and not (stop_token_ids and token in stop_token_ids):
            token = self.decode(token, step=len(output) - 1, sampler=sampler, tap=tap)
            output.append(token)
        return output

    def close(self) -> None:
        self.weights = None
        self.state = None
        self.last_hidden = None
        self.last_draft = None
        self.vision_stats = None
        self.active_compile_codec = None
        self.reader.close()

    def __enter__(self) -> "RefModel":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
