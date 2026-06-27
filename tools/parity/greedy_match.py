#!/usr/bin/env python3
"""Real-weight greedy token parity gate for M2."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.parity.ref_model import D, RefModel, parse_prompt  # noqa: E402


def run_engine(engine_exe: Path, weights: Path, prompt: list[int], decode: int, max_context: int):
    cmd = [
        str(engine_exe),
        str(weights),
        "--max-context",
        str(max_context),
        "--max-new",
        str(decode),
        *[str(x) for x in prompt],
    ]
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    tokens = parse_prompt(proc.stdout.strip())
    match = re.search(r"tok_s=([0-9.]+)", proc.stderr)
    tok_s = float(match.group(1)) if match else float("nan")
    return tokens, tok_s, proc.stderr.strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64, copy=False).reshape(-1)
    bf = b.astype(np.float64, copy=False).reshape(-1)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0.0:
        return 1.0 if np.count_nonzero(af - bf) == 0 else 0.0
    return float(np.dot(af, bf) / denom)


def dump_first_divergent_layer(
    dump_exe: Path,
    weights: Path,
    prompt: list[int],
    mismatch_index: int,
    model: RefModel,
) -> None:
    with tempfile.TemporaryDirectory(prefix="qus-layer-dump-") as tmp:
        out_dir = Path(tmp)
        subprocess.check_call(
            [
                str(dump_exe),
                str(weights),
                str(out_dir),
                str(mismatch_index),
                *[str(x) for x in prompt],
            ],
            cwd=ROOT,
        )
        dumps: dict[str, torch.Tensor] = {}
        model.forward(prompt, mismatch_index + 1, dumps=dumps)
        first_bad = None
        for layer in range(64):
            cpp = np.fromfile(out_dir / f"layer_{layer:02d}.f32", dtype="<f4")
            if cpp.size % D != 0:
                raise ValueError(f"bad dump shape for layer {layer}: {cpp.size} floats")
            cpp = cpp.reshape(cpp.size // D, D)
            ref = dumps[f"layer_{layer}"].numpy()
            cos = cosine(cpp, ref)
            print(f"layer {layer:02d} cosine={cos:.8f}")
            if first_bad is None and cos < 0.999:
                first_bad = (layer, cos)
        if first_bad is not None:
            print(f"first divergent layer: {first_bad[0]} cosine={first_bad[1]:.8f}")
        else:
            print("no per-layer cosine below 0.999; mismatch is likely final norm/lm_head/argmax")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus"))
    ap.add_argument("--prompt", default="1 2 3 4")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine-exe", default=str(ROOT / "build/src/qus"))
    ap.add_argument("--dump-exe", default=str(ROOT / "build/tests/qus_layer_dump"))
    ap.add_argument("--max-context", type=int, default=64)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"SKIP real q5090 weights absent: {weights}")
        return
    engine_exe = Path(args.engine_exe)
    dump_exe = Path(args.dump_exe)
    if not engine_exe.exists():
        raise FileNotFoundError(f"missing engine executable: {engine_exe}")
    if not dump_exe.exists():
        raise FileNotFoundError(f"missing layer dump executable: {dump_exe}")

    prompt = parse_prompt(args.prompt)
    engine_tokens, tok_s, engine_stats = run_engine(
        engine_exe, weights, prompt, args.decode, args.max_context
    )
    print(f"engine: {' '.join(str(x) for x in engine_tokens)}")
    print(f"engine_stats: {engine_stats}")

    t0 = time.perf_counter()
    model = RefModel(weights, device=args.device, cache_globals=False)
    with torch.inference_mode():
        oracle_tokens = model.forward(prompt, args.decode)
    oracle_s = time.perf_counter() - t0
    print(f"oracle: {' '.join(str(x) for x in oracle_tokens)}")
    print(f"oracle_elapsed_s={oracle_s:.3f}")
    print(f"decode_tok_s={tok_s:.6g}")

    if engine_tokens != oracle_tokens:
        mismatch = next(
            i for i, pair in enumerate(zip(engine_tokens, oracle_tokens)) if pair[0] != pair[1]
        )
        print(
            f"TOKEN MISMATCH index={mismatch} engine={engine_tokens[mismatch]} "
            f"oracle={oracle_tokens[mismatch]}",
            file=sys.stderr,
        )
        dump_first_divergent_layer(dump_exe, weights, prompt, mismatch, model)
        raise SystemExit(1)
    print("PASS greedy token match")


if __name__ == "__main__":
    main()
