"""Command-line reference inference for the Qwen3.6-35B-A3B NInfer artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.reference.qwen3_6.common.frontend import Frontend
from tools.reference.qwen3_6.common.multimodal import MultimodalBatch, load_messages
from tools.reference.qwen3_6.common.sampling import Sampler, SamplingConfig
from tools.reference.qwen3_6.common.tap import FileTap

from .bindings import ArtifactBinding
from .config import CFG
from .model import COMPILED_CODEC_MIN_TOKENS, RefModel
from .vision import VisionStats


@dataclass(frozen=True, slots=True)
class PromptInput:
    token_ids: list[int]
    batch: MultimodalBatch | None
    rendered: str | None
    source: str


@dataclass(frozen=True, slots=True)
class Timings:
    load: float
    preprocess: float
    vision: float
    prepare: float
    prefill: float
    decode: float


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split()]


def parse_bytes(text: str | None) -> int | None:
    if text is None or text.strip().lower() == "auto":
        return None
    units = {"gib": 1 << 30, "mib": 1 << 20, "gb": 10**9, "mb": 10**6}
    lowered = text.strip().lower()
    for suffix, scale in units.items():
        if lowered.endswith(suffix):
            return int(float(lowered[: -len(suffix)]) * scale)
    return int(lowered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Qwen3.6-35B-A3B .ninfer artifact")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="single user message rendered by the artifact template")
    prompt.add_argument("--ids", help="comma/space-separated prompt token IDs")
    prompt.add_argument("--messages", help="Qwen messages JSON with optional image/video content")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable the Qwen thinking generation prompt",
    )
    parser.add_argument("--decode", type=int, default=512, help="maximum generated tokens")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory", default="auto")
    parser.add_argument("--headroom", default="2GiB")
    parser.add_argument("--prefill-chunk", type=int, default=CFG.prefill_chunk)
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default="bf16")
    parser.add_argument(
        "--mtp-draft-tokens",
        type=int,
        choices=range(6),
        default=0,
        metavar="0..5",
        help="MTP draft tokens proposed per verification round",
    )
    parser.add_argument(
        "--draft-head",
        action="store_true",
        help="use the optimized shortlist head for MTP proposals",
    )
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stop-ids",
        help="comma/space-separated override; defaults to artifact generation config",
    )
    parser.add_argument("--activation-dump")
    parser.add_argument("--dump-level", choices=("layer", "op"), default="layer")
    parser.add_argument(
        "--vision-attention-limit",
        type=int,
        help="reject prompts whose sum(T*(H*W)^2) exceeds this value",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.weights).suffix != ".ninfer":
        parser.error("--weights must name a .ninfer artifact")
    if args.decode < 0:
        parser.error("--decode must be nonnegative")
    if args.prefill_chunk <= 0:
        parser.error("--prefill-chunk must be positive")
    if args.vision_attention_limit is not None and args.vision_attention_limit <= 0:
        parser.error("--vision-attention-limit must be positive")
    if args.draft_head and args.mtp_draft_tokens == 0:
        parser.error("--draft-head requires --mtp-draft-tokens greater than zero")
    try:
        gpu_memory = parse_bytes(args.gpu_memory)
        headroom = parse_bytes(args.headroom)
    except ValueError as exc:
        parser.error(f"invalid memory size: {exc}")
    if gpu_memory is not None and gpu_memory <= 0:
        parser.error("--gpu-memory must be positive or auto")
    if headroom is None or headroom < 0:
        parser.error("--headroom must be nonnegative")


def _load_prompt(frontend: Frontend, args: argparse.Namespace) -> PromptInput:
    if args.ids is not None:
        try:
            token_ids = parse_ids(args.ids)
        except ValueError as exc:
            raise ValueError(f"invalid --ids: {exc}") from exc
        batch = None
        rendered = frontend.decode(token_ids, skip_special_tokens=False)
        source = args.ids
    else:
        if args.prompt is not None:
            batch = frontend.process_text(args.prompt, thinking=args.thinking)
            source = args.prompt
        else:
            batch = frontend.process(load_messages(args.messages), thinking=args.thinking)
            source = args.messages
        token_ids = batch.input_ids.tolist()
        rendered = frontend.decode(token_ids, skip_special_tokens=False)
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")
    return PromptInput(token_ids, batch, rendered, source)


def _sampling_config(frontend: Frontend, args: argparse.Namespace) -> SamplingConfig:
    generation = frontend.generation_config
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(getattr(generation, "temperature", 0.6))
    )
    top_p = (
        args.top_p
        if args.top_p is not None
        else float(getattr(generation, "top_p", 0.95))
    )
    top_k = (
        args.top_k
        if args.top_k is not None
        else int(getattr(generation, "top_k", 20))
    )
    return SamplingConfig(
        temperature=0.0 if args.greedy else temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_result(
    *,
    prompt: PromptInput,
    generated: Sequence[int],
    generated_text: str,
    stop_reason: str,
    timings: Timings,
    sampler: Sampler | None,
    model: RefModel | None,
    vision_stats: VisionStats | None,
) -> None:
    if model is not None and model.weights is not None:
        print("MEMORY_PLAN:", model.weights.plan.summary())
        print("CODEC:", "compiled" if model.active_compile_codec else "eager")
    if vision_stats is not None:
        print("VISION:", vision_stats.summary())
    if sampler is not None:
        print("SAMPLING:", sampler.summary())
    if model is not None and model.mtp_draft_tokens:
        stats = model.mtp_stats
        acceptance = (
            stats.accepted_tokens / stats.draft_tokens
            if stats.draft_tokens
            else 0.0
        )
        print(
            f"MTP: k={model.mtp_draft_tokens} rounds={stats.rounds} "
            f"fallback_steps={stats.fallback_steps} drafts={stats.draft_tokens} "
            f"accepted={stats.accepted_tokens} acceptance={acceptance:.4f} "
            f"accepted_per_pos={stats.accepted_per_pos[:model.mtp_draft_tokens]}"
        )
    decode_tokens = max(0, len(generated) - 1)
    decode_tps = decode_tokens / timings.decode if timings.decode > 0.0 else 0.0
    print(
        f"TIMING: load={timings.load:.3f}s preprocess={timings.preprocess:.3f}s "
        f"vision={timings.vision:.3f}s prepare={timings.prepare:.3f}s "
        f"prefill={timings.prefill:.3f}s decode={timings.decode:.3f}s "
        f"decode_tokens={decode_tokens} decode_tps={decode_tps:.3f}"
    )
    if model is not None and model.device.type == "cuda":
        print(
            "CUDA_PEAK: "
            f"allocated={torch.cuda.max_memory_allocated(model.device) / (1 << 30):.2f}GiB "
            f"reserved={torch.cuda.max_memory_reserved(model.device) / (1 << 30):.2f}GiB"
        )
    print("PROMPT_TOKEN_IDS:", prompt.token_ids)
    print("GENERATED_TOKEN_IDS:", list(generated))
    print("STOP_REASON:", stop_reason)
    print("GENERATED_TEXT:")
    print(generated_text)


def _stop_ids(
    parser: argparse.ArgumentParser,
    frontend: Frontend,
    override: str | None,
) -> set[int]:
    if override is None:
        return frontend.default_stop_token_ids
    try:
        return set(parse_ids(override))
    except ValueError as exc:
        parser.error(f"invalid --stop-ids: {exc}")


def _frontend_only(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Cheap decode-disabled path for inputs that cannot contain media."""

    load_started = time.perf_counter()
    with ArtifactBinding.open(args.weights) as binding:
        frontend = Frontend(binding)
        load_seconds = time.perf_counter() - load_started
        preprocess_started = time.perf_counter()
        try:
            prompt = _load_prompt(frontend, args)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        preprocess_seconds = time.perf_counter() - preprocess_started
        _print_result(
            prompt=prompt,
            generated=(),
            generated_text="",
            stop_reason="disabled",
            timings=Timings(load_seconds, preprocess_seconds, 0.0, 0.0, 0.0, 0.0),
            sampler=None,
            model=None,
            vision_stats=None,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    memory_bytes = parse_bytes(args.gpu_memory)
    headroom_bytes = parse_bytes(args.headroom)
    assert headroom_bytes is not None

    if args.decode == 0 and args.messages is None:
        _frontend_only(parser, args)
        return

    tap = FileTap(args.activation_dump, args.dump_level) if args.activation_dump else None
    compile_codec = args.decode >= COMPILED_CODEC_MIN_TOKENS
    load_started = time.perf_counter()
    with RefModel(
        args.weights,
        device=args.device,
        memory_bytes=memory_bytes,
        headroom_bytes=headroom_bytes,
        kv_dtype=args.kv_dtype,
        prefill_chunk=args.prefill_chunk,
        mtp_draft_tokens=args.mtp_draft_tokens,
        draft_head=args.draft_head,
        compile_codec=compile_codec,
    ) as model, torch.inference_mode():
        frontend = Frontend(model.binding)
        load_seconds = time.perf_counter() - load_started

        preprocess_started = time.perf_counter()
        try:
            prompt = _load_prompt(frontend, args)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        preprocess_seconds = time.perf_counter() - preprocess_started
        stops = _stop_ids(parser, frontend, args.stop_ids)

        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        vision_output = None
        vision_stats = None
        vision_seconds = 0.0
        if prompt.batch is not None and prompt.batch.has_vision:
            vision_started = time.perf_counter()
            vision_output = model.encode_vision(
                prompt.batch,
                compile_codec=compile_codec,
                attention_limit=args.vision_attention_limit,
            )
            _synchronize(model.device)
            vision_seconds = time.perf_counter() - vision_started
            vision_stats = vision_output.stats

        sampler = None
        generated: list[int] = []
        prepare_seconds = 0.0
        prefill_seconds = 0.0
        decode_seconds = 0.0
        if args.decode > 0:
            try:
                sampler = Sampler(
                    _sampling_config(frontend, args), CFG.token_domain, model.device
                )
            except ValueError as exc:
                parser.error(str(exc))
            prepare_started = time.perf_counter()
            model.prepare(
                len(prompt.token_ids) + args.decode,
                compile_codec=compile_codec,
            )
            _synchronize(model.device)
            prepare_seconds = time.perf_counter() - prepare_started

            prefill_started = time.perf_counter()
            if vision_output is not None:
                assert prompt.batch is not None
                token = model.prefill_multimodal(
                    prompt.batch,
                    vision_output,
                    sampler=sampler,
                    tap=tap,
                )
            else:
                token = model.prefill(
                    prompt.token_ids,
                    sampler=sampler,
                    tap=tap,
                )
            _synchronize(model.device)
            prefill_seconds = time.perf_counter() - prefill_started

            decode_started = time.perf_counter()
            generated = model.continue_generation(
                token,
                args.decode,
                stop_token_ids=stops,
                sampler=sampler,
                tap=tap,
            )
            _synchronize(model.device)
            decode_seconds = time.perf_counter() - decode_started

        generated_text = frontend.decode(generated, skip_special_tokens=True)
        stop_reason = (
            "disabled"
            if args.decode == 0
            else "stop_token"
            if generated and generated[-1] in stops
            else "length"
        )
        _print_result(
            prompt=prompt,
            generated=generated,
            generated_text=generated_text,
            stop_reason=stop_reason,
            timings=Timings(
                load_seconds,
                preprocess_seconds,
                vision_seconds,
                prepare_seconds,
                prefill_seconds,
                decode_seconds,
            ),
            sampler=sampler,
            model=model,
            vision_stats=vision_stats,
        )
        if tap is not None:
            tap.close(
                weights=str(Path(args.weights)),
                prompt_source=prompt.source,
                rendered_prompt=prompt.rendered,
                prompt_ids=prompt.token_ids,
                generated_token_ids=generated,
                generated_text=generated_text,
                stop_reason=stop_reason,
                thinking=args.thinking if args.ids is None else None,
                sampling=None if sampler is None else sampler.summary(),
                kv_dtype=args.kv_dtype,
                prefill_chunk=args.prefill_chunk,
                mtp_draft_tokens=args.mtp_draft_tokens,
                draft_head=args.draft_head,
                vision=vision_stats is not None,
            )


if __name__ == "__main__":
    main()
