"""Command line entry point for q5090 reference inference."""

from __future__ import annotations

import argparse
import time

import torch

from .model import COMPILED_CODEC_MIN_TOKENS, RefModel
from .chat import render_prompt
from .config import CFG
from .multimodal import Processor, load_messages
from .sampling import Sampler, SamplingConfig
from .tap import FileTap


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split()]


def parse_bytes(text: str | None) -> int | None:
    if text is None or text == "auto":
        return None
    units = {"gib": 1 << 30, "mib": 1 << 20, "gb": 10**9, "mb": 10**6}
    lowered = text.strip().lower()
    for suffix, scale in units.items():
        if lowered.endswith(suffix):
            return int(float(lowered[:-len(suffix)]) * scale)
    return int(lowered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="single user message rendered as Qwen3.6 chat")
    prompt.add_argument("--ids", help="comma/space-separated prompt token ids")
    prompt.add_argument("--messages", help="structured Qwen messages JSON with image/video content")
    parser.add_argument(
        "--processor",
        help="local Hugging Face checkpoint directory; required with --messages",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable the Qwen thinking generation prompt",
    )
    parser.add_argument("--decode", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory", default="auto")
    parser.add_argument("--headroom", default="2GiB")
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--kv-dtype", choices=["bf16", "int8"], default="bf16")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--draft-head", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stop-ids",
        help="comma/space-separated override; defaults to embedded generation config",
    )
    parser.add_argument("--structural-dump")
    parser.add_argument("--activation-dump")
    parser.add_argument("--dump-level", choices=["layer", "op"], default="layer")
    parser.add_argument(
        "--vision-attention-limit",
        type=int,
        help="reject prompts whose sum(T*(H*W)^2) exceeds this value",
    )
    args = parser.parse_args()
    if args.decode < 0:
        parser.error("--decode must be nonnegative")
    if args.vision_attention_limit is not None and args.vision_attention_limit <= 0:
        parser.error("--vision-attention-limit must be positive")
    tap = FileTap(args.activation_dump, args.dump_level) if args.activation_dump else None
    load_start = time.perf_counter()
    with RefModel(
        args.weights,
        device=args.device,
        memory_bytes=parse_bytes(args.gpu_memory),
        headroom_bytes=parse_bytes(args.headroom) or 0,
        kv_dtype=args.kv_dtype,
        prefill_chunk=args.prefill_chunk,
        mtp=args.mtp,
        draft_head=args.draft_head,
    ) as model, torch.inference_mode():
        load_seconds = time.perf_counter() - load_start
        if args.structural_dump:
            model.reader.structural_dump(args.structural_dump)
        multimodal = None
        rendered_prompt = None
        if args.messages is not None:
            if not args.processor:
                parser.error("--processor is required with --messages")
            processor = Processor(
                args.processor,
                expected_token_ids={
                    "<|vision_start|>": 248053,
                    "<|vision_end|>": 248054,
                    "<|image_pad|>": 248056,
                    "<|video_pad|>": 248057,
                },
            )
            multimodal = processor.process(load_messages(args.messages), thinking=args.thinking)
            prompt_ids = multimodal.input_ids.tolist()
            rendered_prompt = processor.impl.tokenizer.decode(
                prompt_ids, skip_special_tokens=False
            )
        else:
            rendered_prompt = None if args.prompt is None else render_prompt(
                args.prompt, thinking=args.thinking
            )
            prompt_ids = parse_ids(args.ids) if args.ids is not None else model.tokenizer.encode(
                rendered_prompt
            )
        if not prompt_ids:
            parser.error("prompt must encode to at least one token")
        stops = set(
            parse_ids(args.stop_ids)
            if args.stop_ids is not None
            else model.tokenizer.default_stop_token_ids
        )
        sampling = SamplingConfig(
            temperature=0.0 if args.greedy else args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            seed=args.seed,
        )
        try:
            sampler = Sampler(sampling, CFG.vocab, model.device)
        except ValueError as exc:
            parser.error(str(exc))
        compile_codec = args.decode >= COMPILED_CODEC_MIN_TOKENS
        vision_output = None
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        if multimodal is not None and (multimodal.image_tokens or multimodal.video_tokens):
            vision_output = model.encode_vision(
                multimodal,
                compile_codec=compile_codec,
                attention_limit=args.vision_attention_limit,
            )
        prepare_start = time.perf_counter()
        model.prepare(
            len(prompt_ids) + max(1, args.decode),
            compile_codec=compile_codec,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        prepare_seconds = time.perf_counter() - prepare_start
        tokens = []
        prefill_start = time.perf_counter()
        if args.decode > 0:
            token = (
                model.prefill_multimodal(multimodal, vision_output, sampler=sampler, tap=tap)
                if vision_output is not None
                else model.prefill(prompt_ids, sampler=sampler, tap=tap)
            )
            tokens.append(token)
        vision_output = None
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        prefill_seconds = time.perf_counter() - prefill_start
        decode_start = time.perf_counter()
        while len(tokens) < args.decode and tokens[-1] not in stops:
            tokens.append(
                model.decode(
                    tokens[-1],
                    step=len(tokens) - 1,
                    sampler=sampler,
                    tap=tap,
                )
            )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        decode_seconds = time.perf_counter() - decode_start
        print("MEMORY_PLAN:", model.weights.plan.summary())
        if model.vision_stats is not None:
            print("VISION:", model.vision_stats.summary())
        print("CODEC:", "compiled" if model.active_compile_codec else "eager")
        print("SAMPLING:", sampler.summary())
        print(
            f"TIMING: load={load_seconds:.3f}s prepare={prepare_seconds:.3f}s "
            f"prefill={prefill_seconds:.3f}s decode={decode_seconds:.3f}s "
            f"decode_tokens={max(0, len(tokens) - 1)}"
        )
        if model.device.type == "cuda":
            print(
                "CUDA_PEAK: "
                f"allocated={torch.cuda.max_memory_allocated(model.device) / (1 << 30):.2f}GiB "
                f"reserved={torch.cuda.max_memory_reserved(model.device) / (1 << 30):.2f}GiB"
            )
        generated_text = model.tokenizer.decode(tokens, skip_special_tokens=True)
        stop_reason = (
            "disabled"
            if args.decode == 0
            else "stop_token"
            if tokens and tokens[-1] in stops
            else "length"
        )
        print("PROMPT_TOKEN_IDS:", prompt_ids)
        print("GENERATED_TOKEN_IDS:", tokens)
        print("STOP_REASON:", stop_reason)
        print("GENERATED_TEXT:")
        print(generated_text)
        if args.mtp:
            print("MTP_LAST_DRAFT:", model.last_draft)
        if tap:
            tap.close(
                weights=str(model.reader.path),
                prompt_text=args.prompt if args.prompt is not None else args.messages,
                rendered_prompt=rendered_prompt,
                prompt_ids=prompt_ids,
                generated_token_ids=tokens,
                generated_text=generated_text,
                stop_reason=stop_reason,
                thinking=args.thinking if args.ids is None else None,
                sampling=sampler.summary(),
                kv_dtype=args.kv_dtype,
                prefill_chunk=args.prefill_chunk,
            )


if __name__ == "__main__":
    main()
