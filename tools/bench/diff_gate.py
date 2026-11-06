#!/usr/bin/env python3
"""ninfer prefill DIFF gate: measure ninfer-serve vs the locked sglang baseline.

One self-contained judge for the Qwen3.6-35B-A3B prefill-2x optimization task.

Methodology (identical to the 2026-11-06 evaluation that produced the baseline):
- Anthropic /v1/messages streaming, thinking disabled, greedy, 512 output tokens
- unique-nonce prose prompts defeat prefix/radix caching on every request
- per point: prefill throughput  = sum(prompt_tokens) / max(TTFT of the wave)
             decode throughput   = mean per-stream (completion-1)/(e2e-ttft)
- point prefill ratio = ninfer prefill throughput / sglang baseline throughput
- primary metric: prefill_ratio_min  (target >= 2.0 on every point)
- guard metric:    decode_keep_ratio_min (ninfer decode vs ninfer baseline >= 0.95)

Matrices (n = requests per point, matches the baseline wave shapes exactly):
- quick: 32K x C=1/6, 200K x C=1/6      (~10 min, tuning iterations)
- full:  2K/32K/128K/200K x C=1/2/4/6  (~50 min, e2e gate)

The script starts its own ninfer-serve on GATE_GPU (default 5) and stops it at
exit. It never launches sglang: the sglang baseline is frozen in
tools/bench/baselines_35b.json (measured once, 2026-11-06, GPU5, sglang 0.5.16
with the run_qwen36_35b_new_sglang.sh flags).

Exit codes: 0 pass / 1 perf or functional failure / 2 environment error.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = "/home/users/humingqing/qwen/ninfer-models/qwen3_6_35b_a3b.ninfer"
TOKENIZER = "/home/users/humingqing/qwen/Qwen3.6-35B-A3B-NVFP4-ninfer"
BASELINES = Path(__file__).resolve().parent / "baselines_35b.json"
SERVE = REPO / "build/rtx5090-release/apps/ninfer-serve"
MODEL_ID = "qwen3.6-35b-a3b"
OUT_TOKENS = 512

# (ctx, concurrency, requests) — request counts mirror the baseline waves.
SMOKE_MATRIX = [
    (2048, 1, 2), (32768, 2, 4),
]
QUICK_MATRIX = [
    (32768, 1, 3), (32768, 6, 12),
    (204800, 1, 3), (204800, 6, 6),
]
FULL_MATRIX = [
    (2048, 1, 3), (2048, 2, 4), (2048, 4, 8), (2048, 6, 12),
    (32768, 1, 3), (32768, 2, 4), (32768, 4, 8), (32768, 6, 12),
    (131072, 1, 3), (131072, 2, 4), (131072, 4, 8), (131072, 6, 6),
    (204800, 1, 3), (204800, 2, 4), (204800, 4, 8), (204800, 6, 6),
]

PREFILL_TARGET = 2.0
DECODE_KEEP_TARGET = 0.95

_FILLER = (
    "The transformer architecture processes sequences through stacked attention layers. "
    "Each layer projects its input into query, key and value vectors, then mixes information "
    "across positions using softmax-normalized dot products. Residual connections and "
    "normalization keep gradients stable across depth. Feed-forward sublayers expand and "
    "re-project features channel-wise, giving the network per-position nonlinearity. "
    "Training minimizes next-token cross-entropy over large corpora; inference reuses the "
    "resulting representations to predict continuations one token at a time. "
)

import requests  # noqa: E402  (kept after docstring for fast --help)


def build_prompt(tokenizer, target_tokens: int, nonce: str) -> str:
    head = f"Document {nonce}: "
    body_units = max(1, (target_tokens - 64) // 100 + 1)
    text = head + (_FILLER * (body_units // 1 + 1))
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens - 32:
        text = tokenizer.decode(ids[: target_tokens - 32])
    tail = (
        f"\n\nAbove is document {nonce}. Write a detailed, thorough summary of at "
        f"least 450 words covering every topic it mentions. Do not stop early, "
        f"do not think; answer directly."
    )
    return text + tail


def one_request(url: str, prompt: str, timeout: float) -> dict:
    payload = {
        "model": MODEL_ID,
        "max_tokens": OUT_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
    }
    rec = {"ok": False}
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/v1/messages", json=payload,
                             timeout=timeout, stream=True)
        rec["http_status"] = resp.status_code
        if resp.status_code != 200:
            rec["error"] = resp.text[:300]
            return rec
        ttft = None
        usage_in = None
        usage_out = None
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            try:
                chunk = json.loads(raw[5:].strip())
            except json.JSONDecodeError:
                continue
            kind = chunk.get("type")
            if kind == "message_start":
                usage_in = chunk["message"].get("usage", {}).get("input_tokens")
            elif kind == "content_block_delta":
                delta = chunk.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            elif kind == "message_delta":
                usage_out = (chunk.get("usage") or {}).get("output_tokens")
        rec.update(ok=True, ttft_s=ttft, e2e_s=time.perf_counter() - t0,
                   prompt_tokens=usage_in, completion_tokens=usage_out)
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


def run_point(url: str, tokenizer, ctx: int, conc: int, n: int, timeout: float) -> dict:
    run_id = uuid.uuid4().hex[:8]
    prompts = [build_prompt(tokenizer, ctx, f"{run_id}-g-{i:04d}") for i in range(n)]
    t0 = time.perf_counter()
    if conc == 1:
        recs = [one_request(url, p, timeout) for p in prompts]
    else:
        with ThreadPoolExecutor(max_workers=conc) as pool:
            recs = list(pool.map(lambda p: one_request(url, p, timeout), prompts))
    wall = time.perf_counter() - t0

    ok = [r for r in recs if r.get("ok") and r.get("ttft_s")]
    failed = len(recs) - len(ok)
    if not ok:
        return {"ctx": ctx, "c": conc, "n": n, "failed": failed,
                "error": recs[0].get("error", "no successful request")}
    prompt_total = sum(r["prompt_tokens"] or 0 for r in ok)
    max_ttft = max(r["ttft_s"] for r in ok)
    decodes = []
    for r in ok:
        out = r.get("completion_tokens")
        if out and r["e2e_s"] > r["ttft_s"]:
            decodes.append((out - 1) / (r["e2e_s"] - r["ttft_s"]))
    return {
        "ctx": ctx, "c": conc, "n": n, "failed": failed,
        "prefill_tps": prompt_total / max_ttft if max_ttft else None,
        "decode_tps_mean": sum(decodes) / len(decodes) if decodes else None,
        "prompt_tokens_total": prompt_total,
        "max_ttft_s": max_ttft,
        "wall_s": wall,
    }


def gpu_free_mib(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    return int(out)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def free_port(base: int) -> int | None:
    for offset in range(9):
        port = base + offset
        if port_free(port):
            return port
    return None


def pick_gpu(requested: int | None) -> int:
    """Requested GPU, else the first idle card among the identical 5090Ds."""
    candidates = [requested] if requested is not None else [5, 4, 6, 3]
    for gpu in candidates:
        if gpu is None:
            continue
        if gpu_free_mib(gpu) <= 2000:
            return gpu
    sys.exit("environment error: no free GPU among "
             f"{[g for g in candidates if g is not None]}")


def start_server(gpu: int, port: int, log_path: Path):
    if not SERVE.exists():
        sys.exit(f"environment error: {SERVE} not built")
    used = gpu_free_mib(gpu)
    if used > 2000:
        sys.exit(f"environment error: GPU {gpu} has {used} MiB in use; "
                 f"free the card or point GATE_GPU at a free one")
    if not port_free(port):
        sys.exit(f"environment error: port {port} is busy")
    toml = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    toml.write(f"""# diff_gate auto-generated serving config.
# prefill_chunk is deliberately NOT set: it resolves from the engine/model
# default, making it a legitimate tuning surface (change the default in the
# source or artifact config; this judge must not pin it).
[model]
path = "{ARTIFACT}"

[engine]
max_context = 262144
max_concurrency = 6
kv_dtype = "fp8"
context_capacity = "auto"
memory_headroom = "256MiB"
cuda_graph = true

[speculative]
backend = "mtp"
draft_tokens = 3
proposal_head = "optimized"

[server]
host = "127.0.0.1"
port = {port}
model_id = "{MODEL_ID}"
""")
    toml.close()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LD_LIBRARY_PATH"] = (
        "/home/opt/nvidia_lib:/root/local-deps/usr/lib:"
        + env.get("LD_LIBRARY_PATH", ""))
    env.pop("CUDA_MODULE_LOADING", None)
    proc = subprocess.Popen(
        [str(SERVE), "--config", toml.name],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + 300
    url = f"http://127.0.0.1:{port}"
    import requests as rq
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"environment error: ninfer-serve exited during startup; "
                     f"see {log_path}")
        try:
            if rq.get(f"{url}/health", timeout=2).status_code == 200:
                return proc, url
        except Exception:
            pass
        time.sleep(3)
    proc.terminate()
    sys.exit(f"environment error: server not healthy in 300s; see {log_path}")


def stop_server(proc) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=("smoke", "quick", "full"),
                        default=os.environ.get("GATE_MATRIX", "full"))
    parser.add_argument("--gpu", type=int, default=None,
                        help="explicit GPU; default GATE_GPU or first free card")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GATE_PORT", "18081")))
    parser.add_argument("--enforce", action="store_true",
                        help="fail (exit 1) when perf targets are not met; "
                             "without it only request failures fail")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    baselines = json.loads(BASELINES.read_text())
    matrices = {"smoke": SMOKE_MATRIX, "quick": QUICK_MATRIX,
                "full": FULL_MATRIX}
    matrix = matrices[args.matrix]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)

    out_dir = os.environ.get("AUTODEVX_ANALYZE_OUTPUT_DIR")
    log_path = Path(out_dir or tempfile.gettempdir()) / "gate_server.log"
    gpu = pick_gpu(args.gpu if args.gpu is not None
                   else int(os.environ["GATE_GPU"]) if os.environ.get("GATE_GPU")
                   else None)
    port = free_port(args.port)
    if port is None:
        sys.exit("environment error: no free port in the gate range")
    proc, url = start_server(gpu, port, log_path)
    print(f"[gate] matrix={args.matrix} gpu={gpu} port={port} "
          f"enforce={args.enforce}", flush=True)
    points = {}
    try:
        for ctx, conc, n in matrix:
            point = run_point(url, tokenizer, ctx, conc, n, args.request_timeout)
            key = f"ctx{ctx}_c{conc}"
            base = baselines["points"].get(key)
            if base is None:
                sys.exit(f"environment error: no baseline for {key}")
            prefill_ratio = (point["prefill_tps"] / base["sglang_prefill_tps"]
                             if point.get("prefill_tps") else None)
            decode_keep = (point["decode_tps_mean"] / base["ninfer_decode_tps"]
                           if point.get("decode_tps_mean") else None)
            points[key] = {**point, "prefill_ratio": prefill_ratio,
                           "decode_keep": decode_keep}
            print(f"[gate] {key}: prefill {point.get('prefill_tps') and round(point['prefill_tps'])} tok/s "
                  f"ratio={prefill_ratio and round(prefill_ratio, 3)} "
                  f"decode_keep={decode_keep and round(decode_keep, 3)} "
                  f"failed={point['failed']}", flush=True)
    finally:
        stop_server(proc)

    functional_ok = all(p["failed"] == 0 for p in points.values())
    ratios = [p["prefill_ratio"] for p in points.values() if p["prefill_ratio"]]
    keeps = [p["decode_keep"] for p in points.values() if p["decode_keep"]]
    metrics = {
        "prefill_ratio_min": min(ratios) if ratios else None,
        "decode_keep_ratio_min": min(keeps) if keeps else None,
        "functional_ok": functional_ok,
        "points": points,
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "points"},
                     indent=2), flush=True)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (Path(__file__).resolve().parent / "gate_last_result.json").write_text(
        json.dumps(metrics, indent=2))

    if not functional_ok:
        return 1
    if args.enforce:
        if metrics["prefill_ratio_min"] is None or \
                metrics["prefill_ratio_min"] < PREFILL_TARGET:
            return 1
        if metrics["decode_keep_ratio_min"] is None or \
                metrics["decode_keep_ratio_min"] < DECODE_KEEP_TARGET:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
