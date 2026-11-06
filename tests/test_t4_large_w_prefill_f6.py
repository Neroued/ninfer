"""T4 acceptance tests for F6 (large-W prefill kernels), F7 (decode guard), F8 (correctness).

Only public contracts are used: the Op suite runner (ninfer_softmax_attention_test), the
ninfer-serve HTTP API, ninfer-perplexity, and the recorded profile/bench artifacts under
.agent-tmp/ninfer-35b-prefill-2x. Prefill-ratio thresholds belong to the tuning phase and are
never asserted here.

Heavy work (two servers and two perplexity runs) happens once in the module fixture `runs`,
using two idle GPUs from {3, 4, 5, 6} concurrently.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".agent-tmp/ninfer-35b-prefill-2x"
PROFILE = TASK / "profile"
CAND = Path(os.environ.get("NINFER_T4_CANDIDATE_BUILD", ROOT / "build/rtx5090-dev"))
BASE = TASK / "baseline-a754fcf3/build/rtx5090-release"
OP_TEST = CAND / "tests/ninfer_softmax_attention_test"
ARTIFACT = Path("/home/users/humingqing/qwen/ninfer-models/qwen3_6_35b_a3b.ninfer")
LIBS = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"
CORPUS_STREAM = ROOT / "eval/corpora/perplexity-1m/data/pg19/00.txt"
GEOMETRY = "d256-h16-kv2"
MODEL_ID = "qwen3.6-35b-a3b"
GEN_TOKENS = 512
FIXED_NONCES = ["fixed-0000", "fixed-0001"]
PROMPT_CTX = 2048


def _env(gpu=None):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _free_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return []
    free = []
    for line in out.stdout.splitlines():
        idx, used = (int(x) for x in line.split(","))
        if idx in (3, 4, 5, 6) and used <= 2000:
            free.append(idx)
    return free


def _op_list(fmt):
    if not OP_TEST.exists():
        pytest.skip(f"{OP_TEST} not built")
    gpus = _free_gpus()
    res = subprocess.run([str(OP_TEST), "--list", "--format", fmt], capture_output=True,
                         text=True, env=_env(gpus[0] if gpus else None), timeout=60)
    assert res.returncode == 0, res.stderr
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def _has_dims(case):
    return bool(re.search(r"/t\d+/base\d+/bound\d+/", case))


def _case_dims(case):
    t = int(re.search(r"/t(\d+)/", case).group(1))
    base = int(re.search(r"/base(\d+)/", case).group(1))
    bound = int(re.search(r"/bound(\d+)/", case).group(1))
    return t, base, bound


def _run_cases(fmt, cases, timeout):
    gpus = _free_gpus()
    if not gpus:
        pytest.skip("no idle GPU in {3,4,5,6}")
    for case in cases:
        res = subprocess.run([str(OP_TEST), "--format", fmt, "--case", case],
                             capture_output=True, text=True, env=_env(gpus[0]), timeout=timeout)
        assert res.returncode == 0, f"{case}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}"


# ---- F6 ------------------------------------------------------------------------------------


def test_f6_op_suite_declares_large_w_and_long_history_cases():
    cases = [c for c in _op_list("fp8") if f"/{GEOMETRY}/" in c and _has_dims(c)]
    dims = [_case_dims(c) for c in cases]
    widths = {t for t, _, _ in dims}
    assert {2048, 4096} <= widths, f"W=2048/4096 cases missing, have {sorted(widths)}"
    assert any(t >= 2048 and max(base, bound) >= 131072 for t, base, bound in dims), \
        "no W>=2048 case with history >= 131072"
    assert any(t > 128 and t % 128 != 0 for t in widths), "no non-tile-multiple W above 128"


def test_f6_production_fp8_gqa2_public_entry_cases_pass_against_fp32_reference():
    cases = [c for c in _op_list("fp8")
             if f"/{GEOMETRY}/" in c and _has_dims(c) and "/envelope_updates" not in c]
    big = [c for c in cases if _case_dims(c)[0] >= 1024]
    long_history = [c for c in cases
                    if _case_dims(c)[0] > 16 and max(_case_dims(c)[1:]) >= 131072]
    selected = sorted(set(big + long_history))
    assert selected, "no large-W production cases declared"
    _run_cases("fp8", selected[:8], timeout=120)


def _load_jsonl(path):
    if not path.exists():
        pytest.skip(f"{path} missing")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_f6_profile_records_cover_200k_and_128k_with_time_shares():
    fields = ["tool", "commit", "gpu", "ctx", "concurrency", "prefill_chunk", "attention_pct",
              "gdn_pct", "moe_pct", "linear_pct", "host_gap_pct", "total_s"]
    records = _load_jsonl(PROFILE / "records.jsonl")
    for rec in records:
        for f in fields:
            assert f in rec, f"{f} missing in {rec}"
        for f in fields[6:]:
            assert math.isfinite(rec[f]) and rec[f] >= 0
        shares = sum(rec[f] for f in fields[6:11])
        assert 95.0 <= shares <= 105.0, f"shares sum to {shares}"
        assert 0 < rec["total_s"] < 600
    assert any(r["ctx"] == 204800 and r["concurrency"] == 1 for r in records)
    assert any(r["ctx"] == 131072 and r["concurrency"] == 1 for r in records)


def test_f6_bench_records_report_baseline_and_candidate_for_same_workload():
    records = _load_jsonl(PROFILE / "bench_records.jsonl")
    assert records
    for rec in records:
        assert {"bench", "workload", "metric", "baseline", "candidate"} <= set(rec)
        assert {"history", "heads"} <= set(rec["workload"])
        assert {"W", "T"} & set(rec["workload"])
        assert math.isfinite(rec["baseline"]) and rec["baseline"] > 0
        assert math.isfinite(rec["candidate"]) and rec["candidate"] > 0
    assert any((r["workload"].get("W") or r["workload"].get("T") or 0) >= 2048 for r in records)
