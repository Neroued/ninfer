"""K3 acceptance tests for F6 (long-context large-W prefill kernels).

Two groups:
  * Op suites: the shipped Op test runners (independent FP32 references) for the routed_swiglu,
    causal_softmax_attention and gated_delta_net public entries. Only a small, fixed set of cases
    is run: large-W / long-history production shapes (FP8 KV, [256,16,2]) and the decode / MTP
    verify shapes (B=1..8, W<=16) that must not change.
  * Recorded artifacts: prefill-profile.md and moe-before-after.md. Only presence, finiteness,
    consistency and plausibility are checked. prefill_ratio >= 2.0 belongs to the tuning phase and
    is not asserted anywhere here.
Suites and artifacts that are absent skip rather than fail.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".agent-tmp/ninfer-35b-prefill-2x"
TESTS_BIN = ROOT / "build/rtx5090-dev/tests"
PROFILE = TASK / "profile/prefill-profile.md"
MOE_BA = TASK / "bench/moe-before-after.md"

LD_PATH = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"
CASE_TIMEOUT_S = 90

ATTN = "ninfer_softmax_attention_test"
GDN = "ninfer_gated_delta_net_test"
MOE = "ninfer_moe_test"

# FP8 KV, D=256, Hq=16, Hkv=2 (the production GQA shape).
ATTN_CASES = [
    # large W with 8K history, non-multiple of tile history/bound
    "causal_append/fp8/d256-h16-kv2/t1024/base8192/bound9216/seed611/fragmented",
    # W=65 / 129 are not tile multiples
    "causal_batch/fp8/d256-h16-kv2/w65/b1/seed1304/contexts_8191",
    "causal_batch/fp8/d256-h16-kv2/w129/b1/seed619/contexts_17",
    # decode / MTP verify: W<=16 with B=1..8, including zero history
    "causal_batch/fp8/d256-h16-kv2/w1/b3/seed614/contexts_0_31_63",
    "causal_batch/fp8/d256-h16-kv2/w7/b3/seed616/contexts_17_4097_64",
    "causal_batch/fp8/d256-h16-kv2/w16/b3/seed610/contexts_17_1025_64",
    "causal_batch/fp8/d256-h16-kv2/w16/b8/seed618/contexts_128_64_32_8_2_17_63_127",
]
GDN_CASES = [
    "gdn/35b recurrent/h16/v32/t15/normalized/disjoint",
    "gdn/qwen-35b-a3b/h16/v32/t1025/raw/alias",
    "gdn/prefill boundary/h1/v3/t4096/normalized/disjoint",
    "gdn/batch/35b ordinary/b8/normalized/s8/s9/s10/s11/s12/s13/s14/s15/d8/d9/d10/d11/d12/d13/d14/d15",
]


def _env() -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LD_PATH + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def _binary(name: str) -> Path:
    path = TESTS_BIN / name
    if not path.is_file():
        pytest.skip(f"{path} is not built")
    return path


def _run(name: str, *args: str) -> subprocess.CompletedProcess:
    exe = _binary(name)
    proc = subprocess.run(
        [str(exe), *args], cwd=TESTS_BIN, env=_env(), capture_output=True, text=True, timeout=CASE_TIMEOUT_S
    )
    if "no CUDA" in proc.stderr or "no CUDA-capable" in proc.stdout:
        pytest.skip("no usable CUDA device")
    return proc


def _list(name: str, *args: str) -> list[str]:
    proc = _run(name, "--list", *args)
    assert proc.returncode == 0, proc.stderr
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _expect_ok(proc: subprocess.CompletedProcess) -> None:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-2000:]
    assert re.search(r"OK \d+ cases", out), out[-2000:]


# --------------------------------------------------------------------------------------
# Op suites
# --------------------------------------------------------------------------------------


def test_routed_swiglu_suite_passes_including_large_t():
    """Whole routed_swiglu Op suite (~10 s) passes, and it declares large-T groupwise Q4/Q5 and Q4/Q6."""
    cases = _list(MOE)
    assert any(c.startswith("moe/q4_q5/complete/") and c.endswith("t4097") for c in cases)
    assert any(c.startswith("moe/q4_q6/complete/") for c in cases)
    _expect_ok(_run(MOE))


@pytest.mark.parametrize("case", ATTN_CASES)
def test_causal_softmax_attention_fp8_gqa_cases(case):
    """Public causal attention entry, FP8 KV and [256,16,2]: large W and decode/MTP W<=16 match the reference."""
    assert case in _list(ATTN, "--format", "fp8"), f"case no longer declared: {case}"
    _expect_ok(_run(ATTN, "--case", case))


@pytest.mark.parametrize("case", GDN_CASES)
def test_gated_delta_net_cases(case):
    """Public gated_delta_net entry: recurrent (decode), chunked prefill and batch cases match the reference."""
    assert case in _list(GDN)
    _expect_ok(_run(GDN, "--case", case))


# --------------------------------------------------------------------------------------
# Profile artifact (recorded before the kernel work)
# --------------------------------------------------------------------------------------


def _profile_text() -> str:
    if not PROFILE.is_file():
        pytest.skip(f"missing {PROFILE.relative_to(ROOT)}")
    return PROFILE.read_text()


def _pct(cell: str) -> float:
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)%\s*", cell)
    assert m, f"not a percentage: {cell!r}"
    return float(m.group(1))


def _profile_rows() -> dict[tuple[str, str], dict[str, str]]:
    lines = _profile_text().splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("| ctx |"))
    header = [c.strip() for c in lines[header_idx].strip("|").split("|")]
    rows = {}
    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        row = dict(zip(header, cells))
        rows[(row["ctx"], row["C"])] = row
    return rows


def test_profile_records_128k_and_200k_c1_with_class_shares():
    text = _profile_text()
    assert re.search(r"^commit:\s*[0-9a-f]{7,40}", text, re.M), "commit missing"
    scope = re.search(r"^scope:\s*(.+)$", text, re.M)
    assert scope and re.search(r"nsys|gpu-timing", scope.group(1)), "scope must name nsys or gpu-timing"
    assert re.search(r"^workload:", text, re.M), "workload missing"

    rows = _profile_rows()
    for key in (("128K", "1"), ("200K", "1")):
        assert key in rows, f"profile row {key} missing"
        row = rows[key]
        assert int(row["prompt tokens"]) > 0
        total_ms = float(row["total_ms (stream interval)"])
        assert math.isfinite(total_ms) and total_ms > 0
        shares = {k: _pct(row[k]) for k in ("attention", "gdn", "moe", "linear", "other")}
        assert abs(sum(shares.values()) - 100.0) <= 1.5, shares
        assert 0.0 <= _pct(row["host_gap"]) < 20.0
        tps = float(row["tok/s"])
        assert math.isfinite(tps) and 0 < tps < 1e6
    # prompt lengths match the requested contexts (128K ~ 131072, 200K ~ 204800)
    assert 120_000 <= int(rows[("128K", "1")]["prompt tokens"]) <= 131_072 + 2048
    assert 190_000 <= int(rows[("200K", "1")]["prompt tokens"]) <= 204_800 + 2048


def test_profile_lists_top_kernels_for_long_context():
    text = _profile_text()
    for ctx in ("128K C=1", "200K C=1"):
        m = re.search(rf"## top kernels: {re.escape(ctx)}\n\n(.*?)(?:\n\n|\Z)", text, re.S)
        assert m, f"top kernels section for {ctx} missing"
        rows = [l for l in m.group(1).splitlines() if l.startswith("|")][2:]
        assert len(rows) >= 3
        pcts = [float(r.strip("|").split("|")[0]) for r in rows]
        assert all(0 < p <= 100 for p in pcts) and pcts == sorted(pcts, reverse=True)


# --------------------------------------------------------------------------------------
# Before/after artifact (Op bench only claims Op performance)
# --------------------------------------------------------------------------------------


def _moe_op_rows() -> list[tuple[str, int, float, float, float]]:
    if not MOE_BA.is_file():
        pytest.skip(f"missing {MOE_BA.relative_to(ROOT)}")
    rows = []
    for line in MOE_BA.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"q\d-q\d", cells[0]) and cells[1].isdigit():
            delta = float(cells[4].rstrip("%"))
            rows.append((cells[0], int(cells[1]), float(cells[2]), float(cells[3]), delta))
    return rows


def test_moe_before_after_documents_baseline_workload_metric_and_consistent_delta():
    text = MOE_BA.read_text() if MOE_BA.is_file() else pytest.skip("missing moe-before-after.md")
    assert re.search(r"^baseline_commit:\s*[0-9a-f]{7,40}", text, re.M)
    assert "ninfer_moe_bench" in text and "--cache cold" in text, "workload/measurement conditions missing"
    assert re.search(r"\bus\b", text), "metric unit missing"
    assert re.search(r"Op (bench )?proves Op", text), "scope statement for Op bench missing"

    rows = _moe_op_rows()
    assert {r[1] for r in rows} >= {2048, 4096, 8192}
    for codec, t, before, after, delta in rows:
        assert math.isfinite(before) and math.isfinite(after) and before > 0 and after > 0
        expected = (after - before) / before * 100.0
        assert abs(expected - delta) <= 0.15, (codec, t, expected, delta)
        assert -50.0 < delta < 50.0, "implausible change"


def test_moe_before_after_reports_non_improved_ranges_and_other_ops():
    text = MOE_BA.read_text() if MOE_BA.is_file() else pytest.skip("missing moe-before-after.md")
    assert re.search(r"not improved", text, re.I)
    # attention / GDN are covered explicitly (changed or recorded as untouched)
    assert re.search(r"attention", text, re.I) and re.search(r"GDN", text)
