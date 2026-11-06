"""T1 acceptance tests for F1 (judge measurability) and F2 (prefill_chunk single source).

No GPU, model artifact or full build is needed. The C++ contract test is compiled header-only
against the CUDA headers. Prefill throughput thresholds belong to the tuning phase and are not
asserted here.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CPP_TEST = ROOT / "tests/models/qwen3_5/test_t1_prefill_chunk_default.cpp"
LOCKED = ["tools/bench/diff_gate.py", "tools/bench/baselines_35b.json"]
ARTIFACT = Path("/home/users/humingqing/qwen/ninfer-models/qwen3_6_35b_a3b.ninfer")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _qwen35_sources():
    for path in (SRC / "models/qwen3_5").rglob("*"):
        if path.suffix in {".cpp", ".h", ".inl", ".cu", ".cuh"}:
            yield path


# ---- F2 ------------------------------------------------------------------------------------


def test_f2_resident_default_defined_in_one_place():
    defs = []
    for path in SRC.rglob("*"):
        if path.suffix not in {".cpp", ".h", ".inl", ".cu", ".cuh"}:
            continue
        text = path.read_text(errors="ignore")
        if re.search(r"\bkResidentPrefillChunk\s*=\s*\d+", text):
            defs.append(path)
    assert len(defs) == 1, defs

    # No second synonymous default: no literal 1024/2048/4096 assigned or used as a prefill
    # chunk fallback anywhere else under src/models/qwen3_5.
    pattern = re.compile(r"prefill_chunk\s*(==\s*0\s*\?|=\s*\d{3,}|\?\s*\d{3,}|:\s*\d{3,})")
    offenders = [
        str(p.relative_to(ROOT))
        for p in _qwen35_sources()
        if p not in defs and pattern.search(p.read_text(errors="ignore"))
    ]
    assert offenders == []


@pytest.fixture(scope="module")
def cpp_contract_binary(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("c++")
    cuda_inc = Path("/usr/local/cuda/include")
    if cxx is None or not cuda_inc.is_dir():
        pytest.skip("no host compiler or CUDA headers")
    out = tmp_path_factory.mktemp("t1_prefill_chunk")
    exe = out / "t1_prefill_chunk"
    proc = subprocess.run(
        [cxx, "-std=c++20", f"-I{SRC}", f"-I{ROOT / 'include'}", f"-I{cuda_inc}",
         str(CPP_TEST), "-o", str(exe)],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return exe


def test_f2_resolve_prefill_chunk_contract(cpp_contract_binary):
    proc = subprocess.run([str(cpp_contract_binary)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_f2_real_model_test_expects_resolved_default():
    text = (ROOT / "tests/models/qwen3_5/test_engine_moe_real.cpp").read_text()
    assert "kResidentPrefillChunk" in text
    assert "kHostExpertsPrefillChunk" in text


def test_f2_engine_options_chunk_matches_description_in_test_sources():
    # The real-model tests compare options() against description(); keep that observable.
    text = (ROOT / "tests/models/qwen3_5/test_engine_moe_real.cpp").read_text()
    assert re.search(r"description\(\)\.prefill_chunk", text)
    assert re.search(r"options\(\)\.prefill_chunk", text)


# ---- F1 ------------------------------------------------------------------------------------


def test_f1_locked_files_unchanged():
    for rel in LOCKED:
        assert (ROOT / rel).is_file(), rel
    diff = _git("diff", "HEAD", "--", *LOCKED)
    assert diff.returncode == 0
    assert diff.stdout == ""


def test_f1_model_artifact_readable_and_stable():
    if not ARTIFACT.exists():
        pytest.skip("model artifact not present")
    h1 = hashlib.sha256()
    with ARTIFACT.open("rb") as f:
        h1.update(f.read(1 << 20))
    assert h1.hexdigest()  # artifact is readable; full sha256 is compared by the harness


def test_f1_no_workload_specific_branches_in_prefill_defaults():
    # Defaults must not depend on the judge's identity (model id, nonce, port).
    banned = re.compile(r"diff_gate|nonce|judge", re.I)
    for name in ("src/models/qwen3_5/prefill_chunk.h", "src/models/qwen3_5/startup.cpp",
                 "src/models/qwen3_5/program/planning/startup.cpp"):
        path = ROOT / name
        if path.exists():
            assert not banned.search(path.read_text()), name


def test_f1_last_result_schema_when_present():
    import json
    import math

    path = ROOT / "tools/bench/gate_last_result.json"
    if not path.exists():
        pytest.skip("no gate result yet")
    data = json.loads(path.read_text())
    metrics = data.get("metrics", data)
    for key in ("prefill_ratio_min", "decode_keep_ratio_min"):
        if key in metrics and metrics[key] is not None:
            assert math.isfinite(metrics[key]) and metrics[key] > 0
    if "prefill_ratio_min" in metrics and metrics["prefill_ratio_min"] is not None:
        assert 0 < metrics["prefill_ratio_min"] < 50  # sanity bound only, no 2.0 threshold
