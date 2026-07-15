"""Qwen3.6-35B-A3B reference schedule over a native NInfer artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from tools.reference.qwen3_6.common.multimodal import MultimodalBatch
from tools.reference.qwen3_6.common.sampling import Sampler
from tools.reference.qwen3_6.common.tap import NullTap

from . import mtp as mtp_schedule
from .bindings import ArtifactBinding, RowAddressable, WeightObject
from .config import CFG
from .ops import attention, linear, rmsnorm
from .state import ModelState, StateSnapshot
from .text import run as run_text
from .vision import VisionEncoder, VisionOutput, VisionStats
from .weights import WeightStore


COMPILED_CODEC_MIN_TOKENS = 1


@dataclass
class ModelSnapshot:
    state: StateSnapshot
    last_hidden: torch.Tensor | None
    last_mtp_hidden: torch.Tensor | None
    last_draft: int | None


class RefModel:
    """Complete fixed Text/MoE/MTP compute path for the 35B-A3B artifact target."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | torch.device = "cuda",
        memory_bytes: int | None = None,
        headroom_bytes: int = 2 << 30,
        kv_dtype: str = "bf16",
        prefill_chunk: int = CFG.prefill_chunk,
        mtp_draft_tokens: int = 0,
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
        if not 0 <= mtp_draft_tokens <= 5:
            raise ValueError("mtp_draft_tokens must be in [0,5]")
        if draft_head and mtp_draft_tokens == 0:
            raise ValueError("draft_head requires MTP")

        self.binding = ArtifactBinding.open(weights)
        self.memory_bytes = memory_bytes
        self.headroom_bytes = headroom_bytes
        self.kv_dtype = kv_dtype
        self.prefill_chunk = prefill_chunk
        self.mtp_draft_tokens = mtp_draft_tokens
        self.mtp_enabled = mtp_draft_tokens > 0
        self.draft_head = draft_head
        self.compile_codec = compile_codec
        self.active_compile_codec: bool | None = None
        self.weights: WeightStore | None = None
        self.state: ModelState | None = None
        self.last_hidden: torch.Tensor | None = None
        self.last_mtp_hidden: torch.Tensor | None = None
        self.last_draft: int | None = None
        self.mtp_stats = mtp_schedule.MtpStats()
        self.vision_stats: VisionStats | None = None

    def prepare(self, capacity: int, *, compile_codec: bool | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if capacity > CFG.max_position_embeddings:
            raise ValueError("capacity exceeds the artifact profile's maximum context")
        self.weights = None
        self.state = None
        self.last_hidden = None
        self.last_mtp_hidden = None
        self.last_draft = None
        self.mtp_stats = mtp_schedule.MtpStats()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if compile_codec is None:
            compile_codec = self.compile_codec if self.compile_codec is not None else False
        self.weights = WeightStore(
            self.binding,
            self.device,
            capacity=capacity,
            kv_dtype=self.kv_dtype,
            text=True,
            mtp=self.mtp_enabled,
            draft_head=self.draft_head,
            vision=False,
            compile_codec=compile_codec,
            prefill_chunk=self.prefill_chunk,
            memory_bytes=self.memory_bytes,
            headroom_bytes=self.headroom_bytes,
        )
        self.state = ModelState(self.device, capacity, self.kv_dtype)
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
            self.binding,
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

    def weight(
        self,
        value: WeightObject,
        *,
        small_t: bool = False,
    ) -> torch.Tensor:
        weights, _ = self._ready()
        return weights.tensor(
            value,
            dequant_dtype=torch.float32 if small_t else torch.bfloat16,
        )

    def block_weight(
        self,
        value: WeightObject,
        *,
        small_t: bool = False,
    ) -> torch.Tensor:
        weights, _ = self._ready()
        return weights.tensor(
            value,
            dequant_dtype=torch.float32 if small_t else torch.bfloat16,
        )

    def rows(
        self,
        value: RowAddressable,
        rows: Iterable[int] | torch.Tensor,
        *,
        small_t: bool = False,
    ) -> torch.Tensor:
        weights, _ = self._ready()
        return weights.rows(
            value,
            rows,
            dequant_dtype=torch.float32 if small_t else torch.bfloat16,
        )

    def _positions(self, start: int, count: int) -> torch.Tensor:
        _, state = self._ready()
        values = torch.arange(
            start, start + count, device=self.device, dtype=torch.int32
        )
        if state.mrope:
            values = values + state.rope_delta
            return values.unsqueeze(0).expand(3, -1)
        return values

    def embed(self, ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        weights, _ = self._ready()
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(list(ids), device=self.device, dtype=torch.long)
        else:
            ids = ids.to(device=self.device, dtype=torch.long)
        return weights.rows(self.binding.text.token_embedding, ids)

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
        return rmsnorm(x, self.weight(self.binding.text.final_norm))

    def logits_last(self, hidden: torch.Tensor, *, draft: bool = False) -> torch.Tensor:
        weights, _ = self._ready()
        last = hidden[-1:].to(torch.bfloat16)
        if draft:
            head = self.binding.text.draft_head
            logits = linear(
                last,
                weights.tensor(head.weight, dequant_dtype=torch.float32),
                small_t=True,
            )[0]
            ids = weights.tensor(head.token_ids).long()
            full = torch.full((CFG.vocab,), -torch.inf, device=self.device, dtype=torch.float32)
            full[ids] = logits.float()
            return full

        head = self.binding.text.output_head
        if weights.representation(head) == "decoded":
            return linear(
                last,
                weights.tensor(head, dequant_dtype=torch.float32),
                small_t=True,
            )[0]
        logits = torch.empty(CFG.vocab, device=self.device, dtype=torch.bfloat16)
        for row0, row1, weight in weights.chunks(
            head,
            dequant_dtype=torch.float32,
        ):
            logits[row0:row1] = linear(last, weight, small_t=True)[0]
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
        last_chunk = 0
        for chunk, offset in enumerate(range(0, len(ids), self.prefill_chunk)):
            last_chunk = chunk
            part = ids[offset : offset + self.prefill_chunk]
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
                        ids[offset + 1 : offset + 1 + known],
                        last_hidden[:known],
                        positions[:known],
                        start=state.mtp_kv.length,
                        small_t=False,
                        sample=False,
                    )

        assert last_hidden is not None
        logits = self.logits_last(last_hidden)
        target = (
            sampler(logits)
            if sampler is not None
            else int(torch.argmax(logits[: CFG.token_domain]).item())
        )
        self.last_hidden = last_hidden
        if self.mtp_enabled:
            mtp_hidden, draft = self.mtp_forward(
                [target],
                last_hidden[-1:],
                torch.tensor([state.position - 1], device=self.device, dtype=torch.int32),
                start=state.mtp_kv.length,
                small_t=True,
            )
            self.last_mtp_hidden = mtp_hidden[-1:]
            self.last_draft = draft
        self._tap(
            tap,
            "logits",
            logits,
            phase="prefill",
            step=0,
            chunk=last_chunk,
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
        if len(ids) > state.capacity:
            raise ValueError("multimodal prefill exceeds prepared context capacity")
        types = batch.mm_token_type_ids.flatten()
        if types.numel() != len(ids) or batch.position_ids.shape != (3, len(ids)):
            raise ValueError("multimodal token metadata shape mismatch")

        state.mrope = True
        state.rope_delta = batch.rope_delta
        tap = tap or NullTap()
        last_hidden = None
        last_positions = None
        last_chunk = 0
        for chunk, offset in enumerate(range(0, len(ids), self.prefill_chunk)):
            last_chunk = chunk
            part = ids[offset : offset + self.prefill_chunk]
            start = state.position
            span_ids = ids[offset : min(len(ids), offset + len(part) + 1)]
            span = self.embed(span_ids)
            batch.scatter_visual_embeddings_(
                span,
                vision.image_embeddings,
                vision.video_embeddings,
                offset=offset,
            )
            x = span[: len(part)]
            self._tap(tap, "embed", x, phase="prefill", step=0, chunk=chunk, position=start)
            positions = batch.position_ids[:, offset : offset + len(part)].to(
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
                    shifted_ids = ids[offset + 1 : offset + 1 + known]
                    self.mtp_forward(
                        shifted_ids,
                        last_hidden[:known],
                        positions[..., :known],
                        start=state.mtp_kv.length,
                        small_t=False,
                        sample=False,
                        input_embeddings=span[1 : known + 1],
                    )

        assert last_hidden is not None and last_positions is not None
        logits = self.logits_last(last_hidden)
        target = (
            sampler(logits)
            if sampler is not None
            else int(torch.argmax(logits[: CFG.token_domain]).item())
        )
        self.last_hidden = last_hidden
        if self.mtp_enabled:
            mtp_hidden, draft = self.mtp_forward(
                [target],
                last_hidden[-1:],
                last_positions[..., -1:],
                start=state.mtp_kv.length,
                small_t=True,
            )
            self.last_mtp_hidden = mtp_hidden[-1:]
            self.last_draft = draft
        self._tap(
            tap,
            "logits",
            logits,
            phase="prefill",
            step=0,
            chunk=last_chunk,
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
        update_mtp: bool = True,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        _, state = self._ready()
        if state.position >= state.capacity:
            raise ValueError("decode exceeds prepared context capacity")
        tap = tap or NullTap()
        start = state.position
        x = self.embed([token])
        self._tap(tap, "embed", x, phase="decode", step=step, chunk=0, position=start)
        positions = self._positions(start, 1)
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
        target = (
            sampler(logits)
            if sampler is not None
            else int(torch.argmax(logits[: CFG.token_domain]).item())
        )
        if self.mtp_enabled and update_mtp:
            mtp_hidden, self.last_draft = self.mtp_forward(
                [target],
                hidden,
                positions,
                start=state.mtp_kv.length,
                small_t=True,
            )
            self.last_mtp_hidden = mtp_hidden[-1:]
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
            None if self.last_mtp_hidden is None else self.last_mtp_hidden.clone(),
            self.last_draft,
        )

    def restore(self, snapshot: ModelSnapshot) -> None:
        _, state = self._ready()
        state.restore(snapshot.state)
        self.last_hidden = snapshot.last_hidden
        self.last_mtp_hidden = snapshot.last_mtp_hidden
        self.last_draft = snapshot.last_draft

    def _draft_proposals(self, count: int) -> list[int]:
        _, state = self._ready()
        if count <= 0:
            return []
        if state.mtp_kv.length != state.position:
            raise RuntimeError("target and MTP cursors are not aligned at round start")
        if self.last_mtp_hidden is None or self.last_draft is None:
            raise RuntimeError("MTP proposal state is not initialized")

        drafts = [self.last_draft]
        hidden = self.last_mtp_hidden
        while len(drafts) < count:
            start = state.mtp_kv.length
            hidden, draft = self.mtp_forward(
                [drafts[-1]],
                hidden,
                self._positions(start, 1),
                start=start,
                small_t=True,
            )
            assert draft is not None
            drafts.append(draft)
        return drafts

    @staticmethod
    def _greedy(logits: torch.Tensor) -> int:
        return int(torch.argmax(logits[: CFG.token_domain]).item())

    def _verify_choice(
        self,
        logits: torch.Tensor,
        draft: int,
        provisional: list[int],
        sampler: Sampler | None,
    ) -> tuple[bool, int]:
        if sampler is None:
            target = self._greedy(logits)
            return target == draft, target

        distribution = sampler.distribution(logits, provisional=provisional)
        if sampler.config.temperature <= 0.0:
            target = int(distribution.indices[0].item())
            return target == draft, target
        if sampler.accept_draft(distribution, draft):
            return True, draft
        return False, sampler.sample_distribution(distribution, exclude=draft)

    def _bonus_token(
        self,
        logits: torch.Tensor,
        provisional: list[int],
        sampler: Sampler | None,
    ) -> int:
        if sampler is None:
            return self._greedy(logits)
        distribution = sampler.distribution(logits, provisional=provisional)
        if sampler.config.temperature <= 0.0:
            return int(distribution.indices[0].item())
        return sampler.sample_distribution(distribution)

    def _speculative_round(
        self,
        current: int,
        window: int,
        *,
        remaining_budget: int,
        stop_token_ids: set[int] | None,
        sampler: Sampler | None,
        step: int,
        tap,
    ) -> list[int]:
        _, state = self._ready()
        baseline = state.position
        drafts = self._draft_proposals(window)
        outputs: list[int] = []
        target_hiddens: list[torch.Tensor] = []
        accepted = 0
        token = current
        stopped = False
        rejected = False

        for column, draft in enumerate(drafts):
            _, hidden, logits = self._decode(
                token,
                step=step + column,
                sampler=None,
                tap=tap,
                update_mtp=False,
            )
            target_hiddens.append(hidden)
            accept, correction = self._verify_choice(
                logits, draft, outputs, sampler
            )
            if not accept:
                outputs.append(correction)
                rejected = True
                stopped = bool(stop_token_ids and correction in stop_token_ids)
                break
            outputs.append(draft)
            accepted += 1
            token = draft
            if stop_token_ids and draft in stop_token_ids:
                stopped = True
                break

        if not rejected and not stopped and accepted == len(drafts):
            _, hidden, logits = self._decode(
                token,
                step=step + len(drafts),
                sampler=None,
                tap=tap,
                update_mtp=False,
            )
            target_hiddens.append(hidden)
            bonus = self._bonus_token(logits, outputs, sampler)
            outputs.append(bonus)
            stopped = bool(stop_token_ids and bonus in stop_token_ids)

        outputs = mtp_schedule.truncate_at_stop(outputs, stop_token_ids)
        self.mtp_stats.record_round(len(drafts), accepted)
        if sampler is not None:
            sampler.commit(outputs)

        if len(target_hiddens) != len(outputs):
            raise RuntimeError("target verification did not produce one hidden per output")
        state.mtp_kv.rewind(baseline)
        rebuild_hidden, next_draft = self.mtp_forward(
            outputs,
            torch.cat(target_hiddens, dim=0),
            self._positions(baseline, len(outputs)),
            start=baseline,
            small_t=True,
            sample=(len(outputs) < remaining_budget and not stopped),
        )
        self.last_mtp_hidden = rebuild_hidden[-1:]
        self.last_draft = next_draft
        return outputs

    def _ordinary_output(
        self,
        current: int,
        *,
        sampler: Callable[[torch.Tensor], int] | None,
        step: int,
        tap,
    ) -> list[int]:
        if self.mtp_enabled:
            self.mtp_stats.record_fallback()
        token, _, _ = self._decode(current, step=step, sampler=sampler, tap=tap)
        return [token]

    def mtp_forward(
        self,
        ids: Iterable[int],
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        start: int,
        small_t: bool,
        sample: bool = True,
        input_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int | None]:
        return mtp_schedule.forward(
            self,
            ids,
            hidden,
            positions,
            start=start,
            small_t=small_t,
            sample=sample,
            input_embeddings=input_embeddings,
        )

    def _continue_generation(
        self,
        token: int,
        max_new_tokens: int,
        *,
        stop_token_ids: set[int] | None,
        sampler: Callable[[torch.Tensor], int] | None,
        tap,
    ) -> list[int]:
        output = [token]
        while len(output) < max_new_tokens and not (
            stop_token_ids and token in stop_token_ids
        ):
            remaining = max_new_tokens - len(output)
            _, state = self._ready()
            window = min(self.mtp_draft_tokens, remaining - 1)
            supports_speculative_sampling = sampler is None or isinstance(sampler, Sampler)
            can_run_round = (
                self.mtp_enabled
                and supports_speculative_sampling
                and window > 0
                and self.last_mtp_hidden is not None
                and self.last_draft is not None
                and state.mtp_kv.length == state.position
                and state.position + window + 1 <= state.capacity
            )
            if can_run_round:
                round_output = self._speculative_round(
                    token,
                    window,
                    remaining_budget=remaining,
                    stop_token_ids=stop_token_ids,
                    sampler=sampler if isinstance(sampler, Sampler) else None,
                    step=len(output) - 1,
                    tap=tap,
                )
            else:
                round_output = self._ordinary_output(
                    token,
                    sampler=sampler,
                    step=len(output) - 1,
                    tap=tap,
                )
                round_output = mtp_schedule.truncate_at_stop(
                    round_output, stop_token_ids
                )
            output.extend(round_output)
            token = round_output[-1]
        return output

    def continue_generation(
        self,
        token: int,
        max_new_tokens: int,
        *,
        stop_token_ids: set[int] | None = None,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> list[int]:
        """Continue from the first sampled token after an explicit prefill."""

        if max_new_tokens <= 0:
            return []
        return self._continue_generation(
            token,
            max_new_tokens,
            stop_token_ids=stop_token_ids,
            sampler=sampler,
            tap=tap,
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
        return self.continue_generation(
            token,
            max_new_tokens,
            stop_token_ids=stop_token_ids,
            sampler=sampler,
            tap=tap,
        )

    def generate_multimodal(
        self,
        batch: MultimodalBatch,
        vision: VisionOutput,
        max_new_tokens: int,
        *,
        stop_token_ids: set[int] | None = None,
        sampler: Callable[[torch.Tensor], int] | None = None,
        tap=None,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if batch.prompt_length == 0 or max_new_tokens == 0:
            return []
        compile_codec = self.compile_codec
        if compile_codec is None:
            compile_codec = max_new_tokens >= COMPILED_CODEC_MIN_TOKENS
        self.prepare(batch.prompt_length + max_new_tokens, compile_codec=compile_codec)
        token = self.prefill_multimodal(batch, vision, sampler=sampler, tap=tap)
        return self.continue_generation(
            token,
            max_new_tokens,
            stop_token_ids=stop_token_ids,
            sampler=sampler,
            tap=tap,
        )

    def close(self) -> None:
        self.weights = None
        self.state = None
        self.last_hidden = None
        self.last_mtp_hidden = None
        self.last_draft = None
        self.vision_stats = None
        self.active_compile_codec = None
        self.binding.close()

    def __enter__(self) -> "RefModel":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


__all__ = ["COMPILED_CODEC_MIN_TOKENS", "ModelSnapshot", "RefModel"]
