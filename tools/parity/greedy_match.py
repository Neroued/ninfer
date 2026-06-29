#!/usr/bin/env python3
"""L4 greedy token parity against the approved q5090 v1 snapshot."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.parity import hf_reference as hfref  # noqa: E402
from tools.parity.ref_model import RefModel  # noqa: E402

DEFAULT_SNAPSHOT = ROOT / "profiles/e2e/m3-output-gate.json"
DEFAULT_FIXTURE = ROOT / "bench/fixtures/prompts/cn_short.ids"
DEFAULT_STOP = {248046, 248044}


def is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def first_divergence(a: list[int], b: list[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def load_snapshot_tokens(path: Path, case: str, repeat_index: int) -> list[int]:
    if repeat_index < 0:
        raise ValueError("--repeat must be nonnegative")
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    for item in report.get("cases", []):
        if item.get("name") != case:
            continue
        repeats = item.get("repeats", [])
        if repeat_index >= len(repeats):
            raise ValueError(f"{path}: case {case!r} has no repeat {repeat_index}")
        tokens = repeats[repeat_index].get("generated_token_ids")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(is_nonnegative_int(x) for x in tokens)
        ):
            raise ValueError(
                f"{path}: case {case!r} repeat {repeat_index} "
                "generated_token_ids must be nonempty nonnegative ints"
            )
        return list(tokens)
    raise ValueError(f"{path}: missing case {case!r}")


def infer_case(fixture: Path) -> str:
    name = fixture.name
    if name.endswith(".ids"):
        return name[:-4]
    return fixture.stem


def require_selected_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda":
        raise RuntimeError("q5090 greedy parity requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for q5090 greedy parity, but this Python has CPU-only PyTorch. "
            "Run with a CUDA environment such as "
            "/home/neroued/miniconda3/envs/py311/bin/python or "
            "/home/neroued/miniconda3/envs/vllm-bench/bin/python."
        )
    return selected


def unload_hf(tokenizer, model) -> None:
    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hf_report(
    hf_dir: Path,
    prompt: list[int],
    max_new: int,
    v2_tokens: list[int],
    snapshot_tokens: Optional[list[int]],
    gpu_mem: str,
    cpu_mem: str,
) -> None:
    try:
        t0 = time.perf_counter()
        tok, model = hfref.load_hf_model(hf_dir, gpu_mem=gpu_mem, cpu_mem=cpu_mem)
        hf_tokens = hfref.hf_greedy_tokens(
            model, prompt, max_new, stop_token_ids=set(DEFAULT_STOP)
        )
        print(f"HF greedy_s={time.perf_counter() - t0:.1f} tokens={len(hf_tokens)}")
        unload_hf(tok, model)
    except Exception as exc:  # report-only path
        print(f"HF first divergence unavailable: {exc}", file=sys.stderr)
        return

    div_v2 = first_divergence(v2_tokens[: len(hf_tokens)], hf_tokens)
    if div_v2 is None:
        print(f"HF first divergence vs v2: none in {len(hf_tokens)} tokens")
    else:
        print(
            f"HF first divergence vs v2: index={div_v2} "
            f"v2={v2_tokens[div_v2] if div_v2 < len(v2_tokens) else 'EOF'} "
            f"hf={hf_tokens[div_v2] if div_v2 < len(hf_tokens) else 'EOF'}"
        )

    if snapshot_tokens is not None:
        div_snap = first_divergence(snapshot_tokens[: len(hf_tokens)], hf_tokens)
        if div_snap is None:
            print(f"HF first divergence vs snapshot: none in {len(hf_tokens)} tokens")
        else:
            print(
                f"HF first divergence vs snapshot: index={div_snap} "
                f"snapshot={snapshot_tokens[div_snap] if div_snap < len(snapshot_tokens) else 'EOF'} "
                f"hf={hf_tokens[div_snap] if div_snap < len(hf_tokens) else 'EOF'}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--hf", default=None, help="local HF bf16 model directory for report-only divergence")
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--tokens", type=int, default=128, help="requested decode budget")
    ap.add_argument("--snapshot-report", default=str(DEFAULT_SNAPSHOT))
    ap.add_argument("--case", default=None)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resident", default="auto", choices=["auto", "gpu", "stream"])
    ap.add_argument("--run-hf-report", action="store_true", help="also run slow HF greedy divergence report")
    ap.add_argument("--hf-gpu-mem", default="26GiB")
    ap.add_argument("--hf-cpu-mem", default="80GiB")
    args = ap.parse_args()

    weights = Path(args.weights)
    fixture = Path(args.fixture)
    if not args.snapshot_report:
        raise ValueError("--snapshot-report is required for the L4 gate")
    snapshot_report = Path(args.snapshot_report)
    case = args.case or infer_case(fixture)
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not fixture.exists():
        raise FileNotFoundError(fixture)
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    device = require_selected_device(args.device)

    prompt = hfref.read_ids(fixture)
    snapshot_tokens = load_snapshot_tokens(snapshot_report, case, args.repeat)
    compare_len = len(snapshot_tokens)
    print(
        f"snapshot={snapshot_report} case={case} repeat={args.repeat} "
        f"snapshot_tokens={compare_len} requested_tokens={args.tokens}"
    )

    t0 = time.perf_counter()
    model = RefModel(weights, device=str(device), cache_globals=False, resident=args.resident)
    try:
        with torch.inference_mode():
            v2_tokens = model.forward(prompt, compare_len)
    finally:
        model.q5090.close()
    print(f"v2_greedy_s={time.perf_counter() - t0:.1f} tokens={len(v2_tokens)}")
    print(f"v2: {' '.join(str(x) for x in v2_tokens)}")

    failed = False
    if snapshot_tokens is not None:
        expected = snapshot_tokens[:compare_len]
        got = v2_tokens[:compare_len]
        div = first_divergence(got, expected)
        if div is None and len(got) == len(expected):
            print(f"PASS snapshot token match length={compare_len}")
        else:
            failed = True
            print(
                f"TOKEN MISMATCH index={div} "
                f"v2={got[div] if div is not None and div < len(got) else 'EOF'} "
                f"snapshot={expected[div] if div is not None and div < len(expected) else 'EOF'}",
                file=sys.stderr,
            )

    if args.hf and args.run_hf_report:
        hf_dir = Path(args.hf)
        if hf_dir.exists():
            run_hf_report(
                hf_dir,
                prompt,
                compare_len,
                v2_tokens,
                snapshot_tokens,
                args.hf_gpu_mem,
                args.hf_cpu_mem,
            )
        else:
            print(f"HF first divergence unavailable: missing HF dir {hf_dir}", file=sys.stderr)
    elif args.hf:
        print("HF first divergence skipped: snapshot gate is authoritative")
    else:
        print("HF first divergence unavailable: --hf not provided")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
