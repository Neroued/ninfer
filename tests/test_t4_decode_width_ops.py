"""T4 F7 acceptance: decode-width (W <= 16, MTP verify shapes) attention Op cases still pass.

Runs the public Op suite runner on the production FP8 and K8V4 GQA-2 geometry against its
independent FP32 reference. A few cases per format keep the cost low.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAND = Path(os.environ.get("NINFER_T4_CANDIDATE_BUILD", ROOT / "build/rtx5090-dev"))
OP_TEST = CAND / "tests/ninfer_softmax_attention_test"
LIBS = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"
GEOMETRY = "d256-h16-kv2"


def _env(gpu):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _idle_gpu():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    for line in out.stdout.splitlines():
        idx, used = (int(x) for x in line.split(","))
        if idx in (3, 4, 5, 6) and used <= 2000:
            return idx
    return None


def _width(case):
    return int(re.search(r"/t(\d+)/", case).group(1))


@pytest.mark.parametrize("fmt", ["fp8", "k8v4"])
def test_f7_decode_width_attention_cases_pass(fmt):
    if not OP_TEST.exists():
        pytest.skip(f"{OP_TEST} not built")
    gpu = _idle_gpu()
    if gpu is None:
        pytest.skip("no idle GPU in {3,4,5,6}")
    listing = subprocess.run([str(OP_TEST), "--list", "--format", fmt], capture_output=True,
                             text=True, env=_env(gpu), timeout=60)
    assert listing.returncode == 0, listing.stderr
    cases = [c.strip() for c in listing.stdout.splitlines()
             if f"/{GEOMETRY}/" in c and re.search(r"/t\d+/", c) and _width(c) <= 16]
    assert cases, f"no decode-width {fmt} cases declared"
    picks = [c for c in cases if "envelope_updates" in c][:3] + cases[:2]
    for case in sorted(set(picks)):
        res = subprocess.run([str(OP_TEST), "--format", fmt, "--case", case],
                             capture_output=True, text=True, env=_env(gpu), timeout=120)
        assert res.returncode == 0, f"{case}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}"
