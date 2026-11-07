"""K1 acceptance tests for F2 (prefill_chunk single authority) and F5 (packed prefill contract).

Host-only: the C++ contract test is compiled header-only against the CUDA headers. The real-model
startup check runs only with NINFER_K1_REAL=1 plus NINFER_TEST_ARTIFACT and a built test binary.
Prefill throughput thresholds belong to the tuning phase and are not asserted here.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CPP = ROOT / "tests/runtime/test_k1_prefill_pack_pacing.cpp"
SRC_SUFFIX = {".cpp", ".h", ".inl", ".cu", ".cuh"}


def _sources(base: Path):
    return [p for p in base.rglob("*") if p.suffix in SRC_SUFFIX]


@pytest.fixture(scope="module")
def contract_binary(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("c++")
    cuda_inc = Path("/usr/local/cuda/include")
    if cxx is None or not cuda_inc.is_dir():
        pytest.skip("no host compiler or CUDA headers")
    exe = tmp_path_factory.mktemp("k1") / "k1_contract"
    proc = subprocess.run(
        [cxx, "-std=c++20", f"-I{SRC}", f"-I{ROOT / 'include'}", f"-I{cuda_inc}", str(CPP),
         "-o", str(exe)], capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return exe


# ---- F2 ------------------------------------------------------------------------------------


def test_f2_default_has_one_definition_and_no_synonyms():
    defs = [p for p in _sources(SRC)
            if re.search(r"\bkResidentPrefillChunk\s*=\s*\d+", p.read_text(errors="ignore"))]
    assert len(defs) == 1, defs
    # Startup and planning take the resolved value; neither may carry a numeric fallback.
    fallback = re.compile(r"prefill_chunk\s*(==\s*0\s*\?|\?\s*\d{3,}|:\s*\d{3,}|=\s*\d{3,})")
    offenders = [str(p.relative_to(ROOT)) for p in _sources(SRC / "models/qwen3_5")
                 if p not in defs and fallback.search(p.read_text(errors="ignore"))]
    assert offenders == []


def test_f2_resolved_default_contract(contract_binary):
    proc = subprocess.run([str(contract_binary)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_f2_tests_keep_explicit_values_and_drop_old_default():
    moe = (ROOT / "tests/models/qwen3_5/test_engine_moe_real.cpp").read_text()
    assert re.search(r"prefill_chunk\s*=\s*2048", moe)
    assert re.search(r"options\(\)\.prefill_chunk", moe)
    assert re.search(r"description\(\)\.prefill_chunk", moe)
    assert "kResidentPrefillChunk" in moe  # default expectation derives from the authority
    for name in ("test_engine_prefix_real.cpp", "test_engine_dflash_real.cpp"):
        text = (ROOT / "tests/models/qwen3_5" / name).read_text()
        assert re.search(r"prefill_chunk\s*=\s*1024", text), name
    # No stale expectation that the Resident default is 1024.
    stale = re.compile(r"(kResidentPrefillChunk\s*==\s*1024|resident[^\n]*default[^\n]*\b1024\b)",
                       re.I)
    for p in (ROOT / "tests").rglob("*.cpp"):
        assert not stale.search(p.read_text(errors="ignore")), p


@pytest.mark.skipif(os.environ.get("NINFER_K1_REAL") != "1", reason="opt-in real model run")
def test_f2_real_default_startup():
    artifact = os.environ.get("NINFER_TEST_ARTIFACT")
    exe = ROOT / "build/rtx5090-dev/tests/ninfer_qwen3_5_engine_moe_real_test"
    if not artifact or not exe.exists():
        pytest.skip("artifact or built test missing")
    env = dict(os.environ, LD_LIBRARY_PATH="/home/opt/nvidia_lib:/root/local-deps/usr/lib")
    proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=280, env=env)
    assert proc.returncode in (0, 77), proc.stderr[-2000:]


# ---- F5 ------------------------------------------------------------------------------------


def test_f5_scheduler_and_pack_contract(contract_binary):
    # Same binary as F2; it also checks pack capacity, segment fallback, charging and regimes.
    proc = subprocess.run([str(contract_binary)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stderr


def test_f5_op_header_documents_segment_range_and_rejection():
    text = (ROOT / "include/ninfer/ops/softmax_attention.h").read_text()
    assert "cu_seqlens" in text and re.search(r"segment", text, re.I)
    assert re.search(r"throw|reject|invalid", text, re.I)


def test_f5_sessions_expose_pack_capacity_and_op_bench_exists():
    for model in ("qwen3_5", "qwen4_exp", "glm5next"):
        text = (SRC / "models" / model / "session/session.h").read_text()
        assert "prefill_pack_tokens()" in text, model
    assert list((ROOT / "bench/ops").glob("packed_*bench.cu"))


def test_f5_noise_artifacts_are_finite_when_present():
    base = ROOT / ".agent-tmp/ninfer-35b-prefill-2x/noise"
    runs = sorted(base.glob("*.json")) if base.is_dir() else []
    if not runs:
        pytest.skip("no noise artifact yet")
    for path in runs:
        data = json.loads(path.read_text())
        for key in ("prefill_ratio_min", "decode_keep_ratio_min"):
            if key in data and data[key] is not None:
                assert math.isfinite(data[key]) and 0 < data[key] < 50, (path, key)
        for name, point in data.get("points", {}).items():
            for key in ("prefill_ratio", "decode_keep"):
                if point.get(key) is not None:
                    assert math.isfinite(point[key]), (path, name, key)
