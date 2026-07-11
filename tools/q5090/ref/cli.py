"""Command line entry point for q5090 reference inference."""

from __future__ import annotations

import argparse
import time

import torch

from .model import COMPILED_CODEC_MIN_TOKENS, RefModel
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
    prompt.add_argument("--prompt", help="raw prompt text encoded by the embedded tokenizer")
    prompt.add_argument("--ids", help="comma/space-separated prompt token ids")
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory", default="auto")
    parser.add_argument("--headroom", default="2GiB")
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--kv-dtype", choices=["bf16", "int8"], default="bf16")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--draft-head", action="store_true")
    parser.add_argument(
        "--stop-ids",
        help="comma/space-separated override; defaults to embedded generation config",
    )
    parser.add_argument("--structural-dump")
    parser.add_argument("--activation-dump")
    parser.add_argument("--dump-level", choices=["layer", "op"], default="layer")
    args = parser.parse_args()
    if args.decode < 0:
        parser.error("--decode must be nonnegative")
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
        prompt_ids = (
            parse_ids(args.ids) if args.ids is not None else model.tokenizer.encode(args.prompt)
        )
        if not prompt_ids:
            parser.error("prompt must encode to at least one token")
        stops = set(
            parse_ids(args.stop_ids)
            if args.stop_ids is not None
            else model.tokenizer.default_stop_token_ids
        )
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        prepare_start = time.perf_counter()
        model.prepare(
            len(prompt_ids) + max(1, args.decode),
            compile_codec=args.decode >= COMPILED_CODEC_MIN_TOKENS,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        prepare_seconds = time.perf_counter() - prepare_start
        tokens = []
        prefill_start = time.perf_counter()
        if args.decode > 0:
            token = model.prefill(prompt_ids, tap=tap)
            tokens.append(token)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        prefill_seconds = time.perf_counter() - prefill_start
        decode_start = time.perf_counter()
        while len(tokens) < args.decode and tokens[-1] not in stops:
            tokens.append(model.decode(tokens[-1], step=len(tokens) - 1, tap=tap))
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        decode_seconds = time.perf_counter() - decode_start
        print("MEMORY_PLAN:", model.weights.plan.summary())
        print("CODEC:", "compiled" if model.active_compile_codec else "eager")
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
        print("PROMPT_TOKEN_IDS:", prompt_ids)
        print("GENERATED_TOKEN_IDS:", tokens)
        print("GENERATED_TEXT:")
        print(generated_text)
        if args.mtp:
            print("MTP_LAST_DRAFT:", model.last_draft)
        if tap:
            tap.close(
                weights=str(model.reader.path),
                prompt_text=args.prompt,
                prompt_ids=prompt_ids,
                generated_token_ids=tokens,
                generated_text=generated_text,
                kv_dtype=args.kv_dtype,
                prefill_chunk=args.prefill_chunk,
            )


if __name__ == "__main__":
    main()
