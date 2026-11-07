#!/usr/bin/env python3
"""One-command prefill/decode ladder with a recorded environment (issue #30).

Runs a fixed set of context sizes against a running capy-flash server, each size with a
deterministic self-contained prompt (no corpus to download), and reports, as the median of a few
passes, the cold-prefill and decode rates from the server's own `timings` block. It also records
the machine and the run configuration, so two people's rows are actually comparable: GPUs, driver,
CUDA, PCIe link width, whether it runs under WSL2, CPU and RAM, and — when you pass the server's
startup log — the P2P state and the expanded launch line the server itself printed.

    # 1. start the server (see the README for the build and the artifact), keeping its log:
    build/apps/ninfer-serve <artifact>.ninfer --profile flash-next --host 127.0.0.1 --port 8098 \
        --model-id capy-flash 2>&1 | tee server.log
    # 2. run the ladder:
    python3 bench/capy_flash/ladder.py http://127.0.0.1:8098 capy-flash result.json \
        --server-log server.log --note "2x RTX 5070 Ti, bare metal"

It writes result.json (one comparable row) and prints a Markdown table. `check_result.py` validates
a row before it is shared. Only the Python standard library is used, and the server does the work,
so this runs anywhere the server does.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import time
import urllib.request

SCHEMA = "capy-flash-ladder/1"
DEFAULT_SIZES = "8000,16000,32000,64000,128000"
# A fixed word list: the prompt filler is deterministic given (size, pass), so the cold leg of one
# pass never hits the prefix cache left by another, and anyone's run uses the same text.
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar "
    "papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu north south east west "
    "river mountain forest desert ocean valley meadow canyon glacier harbor island summit prairie"
).split()


def filler(approx_tokens, seed):
    """A deterministic pseudo-random word stream of about `approx_tokens` tokens. These short words
    tokenize at ~1.4 tokens each on this family, so emit ~0.72 word per target token; the server
    reports the real count, which check_result.py keeps within 0.5-1.6x of the target."""
    state = (seed * 2654435761) & 0xFFFFFFFF
    out = []
    for _ in range(int(approx_tokens * 0.72)):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(WORDS[state % len(WORDS)])
    return " ".join(out)


def ask(url, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning_effort": "none",
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    key = os.environ.get("CAPY_API_KEY")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.time() - t0


def run_size(url, model, size, passes, max_tokens, timeout, seed_base):
    """Cold prefill (max_tokens=1, a different prompt each pass so the cold leg is cold) and decode
    (the warm prompt again), median of `passes` over the server's timings."""
    prefill_tps, decode_tps, accept, prompt_tokens = [], [], [], []
    for p in range(passes):
        prompt = filler(size, seed_base + size + p * 1_000_003)
        cold, _ = ask(url, model, prompt, 1, timeout)
        ti = cold.get("timings") or {}
        if ti.get("prompt_per_second"):
            prefill_tps.append(ti["prompt_per_second"])
        prompt_tokens.append((cold.get("usage") or {}).get("prompt_tokens"))
        warm, _ = ask(url, model, prompt, max_tokens, timeout)
        tw = warm.get("timings") or {}
        if tw.get("predicted_per_second"):
            decode_tps.append(tw["predicted_per_second"])
        if tw.get("draft_n"):
            accept.append(tw["draft_n_accepted"] / tw["draft_n"])
    med = lambda xs: round(statistics.median(xs), 1) if xs else None  # noqa: E731
    return {
        "target_tokens": size,
        "prompt_tokens": next((t for t in prompt_tokens if t), None),
        "passes": passes,
        "prefill_tps_median": med(prefill_tps),
        "prefill_tps_all": [round(x, 1) for x in prefill_tps],
        "decode_tps_median": med(decode_tps),
        "decode_tps_all": [round(x, 1) for x in decode_tps],
        "mtp_accept_median": round(statistics.median(accept), 3) if accept else None,
    }


def sh(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        return ""


def environment():
    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "wsl2": "microsoft" in sh(["uname", "-r"]).lower()
        or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"),
    }
    smi = sh(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader",
        ]
    )
    env["gpus"] = [line.strip() for line in smi.splitlines() if line.strip()]
    # The driver's max CUDA is in the nvidia-smi header, not a queryable field on every driver.
    m = re.search(r"CUDA Version:\s*([0-9.]+)", sh(["nvidia-smi"]))
    env["cuda_driver"] = m.group(1) if m else ""
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    p2p = ansi.sub("", sh(["nvidia-smi", "topo", "-p2p", "r"]))
    env["p2p_matrix"] = [line.rstrip() for line in p2p.splitlines() if "GPU" in line][:8]
    for line in open("/proc/meminfo", errors="replace") if os.path.exists("/proc/meminfo") else []:
        if line.startswith("MemTotal:"):
            env["ram_gib"] = round(int(line.split()[1]) / (1024 * 1024), 1)
    env["cpu"] = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in (open("/proc/cpuinfo", errors="replace") if os.path.exists("/proc/cpuinfo") else [])
            if ln.startswith("model name")
        ),
        platform.processor(),
    )
    return env


def from_server_log(path):
    """The lines the server itself printed that decide what the numbers mean: the expanded launch
    line, the P2P state, the host-mapped weights line and the KV capacity line."""
    if not path or not os.path.exists(path):
        return {}
    text = open(path, errors="replace").read()
    out = {}
    m = re.search(r"\bp2p (on|off[^\n]*)", text)
    if m:
        out["p2p"] = m.group(0)
    for key, pat in {
        "host_mapped": r"host-mapped weights[^\n]*",
        "capacity": r"capacity \| KV[^\n]*",
        "tensor_parallel": r"tensor parallel \| tp[^\n]*",
        "engine_ready": r"engine ready[^\n]*",
    }.items():
        m = re.search(pat, text)
        if m:
            out[key] = m.group(0).strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="capy-flash prefill/decode ladder with environment")
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("out", help="result JSON to write")
    ap.add_argument("--sizes", default=DEFAULT_SIZES, help="comma-separated target prompt tokens")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--server-log", default="", help="the server's startup log (for p2p/profile)")
    ap.add_argument("--note", default="", help="free-text note, e.g. the launch config in words")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results = []
    for size in sizes:
        row = run_size(
            args.url, args.model, size, args.passes, args.max_tokens, args.timeout, args.seed_base
        )
        results.append(row)
        print(
            f"{size:>8} tok  prefill {row['prefill_tps_median']}  decode {row['decode_tps_median']}"
            f"  (prompt {row['prompt_tokens']})",
            flush=True,
        )

    record = {
        "schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "note": args.note,
        "config": {
            "sizes": sizes,
            "passes": args.passes,
            "max_tokens": args.max_tokens,
        },
        "environment": environment(),
        "server": from_server_log(args.server_log),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)

    print("\n| context | prefill tok/s | decode tok/s | MTP accept | prompt tokens |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['target_tokens']} | {r['prefill_tps_median']} | {r['decode_tps_median']} | "
            f"{r['mtp_accept_median']} | {r['prompt_tokens']} |"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
