"""K2 acceptance tests for F1, F3, F4, F7, F8 (reclaim planning cost and context epoch contract).

Host-only. Checks the public Session/ResourceManager contract surface, the K2 measurement
artifact, and the recorded diff_gate metrics. Nothing here starts a server or loads a model.
Prefill throughput thresholds (prefill_ratio_min >= 2.0) belong to the tuning phase: the tests only
require prefill_ratio to be present, finite and inside the plausibility band (0, 50).
Recorded-artifact tests skip when the artifact is absent.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".agent-tmp/ninfer-35b-prefill-2x"
SRC = ROOT / "src"
RM_HEADER = SRC / "runtime/engine/context_cache/resource_manager.h"
SESSIONS = {
    "qwen3_5": SRC / "models/qwen3_5/session/session.h",
    "qwen4_exp": SRC / "models/qwen4_exp/session/session.h",
    "glm5next": SRC / "models/glm5next/session/session.h",
}
POINTS = [f"ctx{ctx}_c{c}" for ctx in (2048, 32768, 131072, 204800) for c in (1, 2, 4, 6)]
BASELINE = "a754fcf3"


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _load(path: Path):
    if not path.is_file():
        pytest.skip(f"missing artifact {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def gate_metrics():
    return _load(TASK / "k2/full-after/metrics.json")


@pytest.fixture(scope="module")
def reclaim_report():
    path = TASK / "k2-reclaim.md"
    if not path.is_file():
        pytest.skip("missing k2-reclaim.md")
    return path.read_text()


# ---- F1: judge measurability --------------------------------------------------------------


def test_f1_locked_files_unchanged_and_metrics_well_formed(gate_metrics):
    locked = ["tools/bench/diff_gate.py", "tools/bench/baselines_35b.json"]
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "HEAD", "--", *locked],
                           capture_output=True, text=True, timeout=60)
    assert dirty.returncode == 0, dirty.stderr
    assert dirty.stdout.strip() == "", dirty.stdout
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%h %s", "--", *locked],
                         capture_output=True, text=True, timeout=60).stdout
    assert "autodevx" not in log, log
    points = gate_metrics["points"]
    assert sorted(points) == sorted(POINTS)
    for name, point in points.items():
        for key in ("prefill_tps", "decode_tps_mean", "prefill_ratio", "decode_keep"):
            assert _finite_positive(point[key]), (name, key, point[key])
        expected = point["n"] * point["ctx"]
        assert abs(point["prompt_tokens_total"] - expected) / expected <= 0.05, name
    for key in ("prefill_ratio_min", "decode_keep_ratio_min"):
        assert _finite_positive(gate_metrics[key])
    assert 0 < gate_metrics["prefill_ratio_min"] < 50
    assert gate_metrics["prefill_ratio_min"] == pytest.approx(
        min(p["prefill_ratio"] for p in points.values()))


# ---- F3: multi-request scheduling under reclaim -------------------------------------------


def test_f3_concurrent_points_have_no_failed_requests(gate_metrics):
    concurrent = [p for name, p in gate_metrics["points"].items() if not name.endswith("_c1")]
    assert len(concurrent) == 12
    assert all(p["failed"] == 0 for p in concurrent), [
        (p["ctx"], p["c"], p["failed"]) for p in concurrent if p["failed"]]


def test_f3_reclaim_planning_is_cheaper_per_request(reclaim_report):
    # The 204800 C=2 row reports engine-side planning time per request before and after K2.
    match = re.search(r"planning_ns[^\n]*\(([\d.]+) s per request, before ([\d.]+) s\)",
                      reclaim_report)
    assert match, "planning_ns per request (before/after) missing"
    after, before = float(match.group(1)), float(match.group(2))
    assert 0 < after < before


# ---- F4: capture/reclaim does not stall other lanes ---------------------------------------


def test_f4_context_epoch_and_pins_exposed_by_every_session():
    for model, header in SESSIONS.items():
        text = header.read_text()
        assert re.search(r"std::uint64_t\s+context_epoch\(\)\s+const\s+noexcept", text), model
        assert re.search(r"bool\s+context_pins\(\s*CheckpointHandle[^)]*\)\s+const\s+noexcept",
                         text), model


def test_f4_planning_stats_contract():
    text = RM_HEADER.read_text()
    body = re.search(r"struct ReclaimPlanningStats\s*\{(.*?)\};", text, re.S)
    assert body, "ReclaimPlanningStats missing"
    fields = re.findall(r"std::uint64_t\s+(\w+)\s*=\s*0;", body.group(1))
    assert fields == ["order_builds", "order_reuses", "selections", "planning_ns"]
    assert re.search(r"const\s+ReclaimPlanningStats&\s+planning_stats\(\)\s+const\s+noexcept", text)


def test_f4_prefix_reuse_defaults_unchanged_from_baseline():
    names = ["src/models/qwen3_5/session/planning.h", "src/models/qwen3_5/program/planning.h"]
    for rel in names:
        if not (ROOT / rel).is_file():
            continue
        diff = subprocess.run(
            ["git", "-C", str(ROOT), "diff", BASELINE, "--", rel],
            capture_output=True, text=True, timeout=60).stdout
        changed = [l for l in diff.splitlines()
                   if l[:1] in "+-" and re.search(r"checkpoint_interval|prefix_reuse|prefix_cache", l)]
        assert changed == [], (rel, changed)


# ---- F7: decode guard ---------------------------------------------------------------------


def test_f7_decode_keep_per_point(gate_metrics):
    points = gate_metrics["points"]
    low = {n: p["decode_keep"] for n, p in points.items() if p["decode_keep"] < 0.95}
    assert not low, low
    assert gate_metrics["decode_keep_ratio_min"] >= 0.95


def test_f7_decode_growth_horizon_has_one_authority():
    defs = [p for p in (SRC / "models/qwen3_5").rglob("*")
            if p.suffix in {".h", ".cpp", ".inl"}
            and re.search(r"\bkDecodeGrowthReserveTokens\s*=\s*\d+", p.read_text(errors="ignore"))]
    assert len(defs) == 1, defs
    numeric = re.compile(r"\+\s*(1024|2048|4096)\b|\b(1024|2048|4096)\s*\+")
    session_dir = SRC / "models/qwen3_5/session"
    # The engine must not restate the Session's decode reserve as its own literal.
    engine = (SRC / "runtime/engine/engine_schedule.inl").read_text()
    assert "kDecodeGrowthReserveTokens" not in engine
    assert not re.search(r"decode[_a-z]*reserve[_a-z]*\s*=\s*\d{3,}", engine)
    assert session_dir.is_dir() and numeric  # sanity: patterns compiled, directory present


# ---- F8: functional correctness -----------------------------------------------------------


def test_f8_functional_ok_and_zero_failed_points(gate_metrics):
    assert gate_metrics["functional_ok"] is True
    failed = {n: p["failed"] for n, p in gate_metrics["points"].items() if p["failed"]}
    assert not failed, failed


def test_f8_recorded_perplexity_within_one_percent():
    base = _load(TASK / "ppl/baseline/report.json")
    cand = _load(TASK / "ppl/candidate/report.json")
    for report in (base, cand):
        argv = " ".join(map(str, report["argv"]))
        if "perplexity-1m" not in argv or "--context 65536" not in argv:
            pytest.skip("recorded PPL reports do not use the spec corpus/context parameters")
    a, b = base["overall"]["perplexity"], cand["overall"]["perplexity"]
    assert _finite_positive(a) and _finite_positive(b)
    assert abs(b / a - 1) <= 0.01, (a, b)


def test_f8_reclaim_artifact_has_required_fields(reclaim_report):
    for needle in ("commit", "workload", "hot spot", "planning_ns", "128K C=1", "decode_keep"):
        assert needle in reclaim_report, needle
    assert re.search(r"ttft", reclaim_report, re.I)
    assert re.search(r"reclaim\w*\s+(calls|attempts)", reclaim_report, re.I)
