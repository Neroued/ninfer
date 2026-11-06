"""T4 F7/F8 acceptance: Graph+MTP stay active, greedy correctness against baseline a754fcf3.

One module fixture starts the candidate and baseline ninfer-serve on two idle GPUs (judge toml
content), sends two fixed prompts (+ one repeat on the candidate), then runs both
ninfer-perplexity binaries on a corpus slice. Prefill-ratio thresholds are not asserted.
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
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".agent-tmp/ninfer-35b-prefill-2x"
CAND = Path(os.environ.get("NINFER_T4_CANDIDATE_BUILD", ROOT / "build/rtx5090-dev"))
BASE = TASK / "baseline-a754fcf3/build/rtx5090-release"
ARTIFACT = Path("/home/users/humingqing/qwen/ninfer-models/qwen3_6_35b_a3b.ninfer")
LIBS = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"
CORPUS = ROOT / "eval/corpora/perplexity-1m/data/pg19/00.txt"
MODEL_ID = "qwen3.6-35b-a3b"
GEN_TOKENS = 512
NONCES = ["fixed-0000", "fixed-0001"]
PROMPT_CTX = 2048
TOML = """[model]
path = "{artifact}"

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
model_id = "{model}"
"""


def _env(gpu):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _idle_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    pairs = [tuple(int(x) for x in line.split(",")) for line in out.stdout.splitlines()]
    return [i for i, used in pairs if i in (3, 4, 5, 6) and used <= 2000]


def _stream(requests, url, prompt):
    payload = {"model": MODEL_ID, "max_tokens": GEN_TOKENS, "temperature": 0.0, "stream": True,
               "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(f"{url}/v1/messages", json=payload, stream=True, timeout=300)
    text, tokens, deltas = [], None, 0
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data:"):
            continue
        chunk = json.loads(raw[5:].strip())
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                deltas += 1
                text.append(delta.get("text", ""))
        elif chunk.get("type") == "message_delta":
            tokens = (chunk.get("usage") or {}).get("output_tokens")
    return {"status": resp.status_code, "text": "".join(text), "completion_tokens": tokens,
            "deltas": deltas}


def _session(build, gpu, prompts, repeat, work, tag):
    import requests
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg, log = work / f"{tag}.toml", work / f"{tag}.log"
    cfg.write_text(TOML.format(artifact=ARTIFACT, port=port, model=MODEL_ID))
    proc = subprocess.Popen([str(build / "apps/ninfer-serve"), "--config", str(cfg)],
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, env=_env(gpu))
    url, out = f"http://127.0.0.1:{port}", {"results": [], "repeat": None, "metrics": ""}
    try:
        deadline = time.time() + 240
        while True:
            assert proc.poll() is None, f"{tag} exited: {log.read_text()[-1500:]}"
            assert time.time() < deadline, f"{tag} not healthy"
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(2)
        out["results"] = [_stream(requests, url, p) for p in prompts]
        if repeat:
            out["repeat"] = _stream(requests, url, prompts[0])
        out["metrics"] = requests.get(f"{url}/metrics", timeout=10).text
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    out["log"] = log.read_text()
    return out


def _perplexity(build, gpu, text, out_dir):
    cmd = [str(build / "apps/ninfer-perplexity"), str(ARTIFACT), "--text", str(text),
           "--context", "16384", "--stride", "8192", "--kv-dtype", "fp8",
           "--output", str(out_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=_env(gpu), timeout=240)
    assert res.returncode == 0, res.stderr[-1500:]
    return json.loads((out_dir / "report.json").read_text())["overall"]["perplexity"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    for need in (CAND / "apps/ninfer-serve", CAND / "apps/ninfer-perplexity",
                 BASE / "apps/ninfer-serve", BASE / "apps/ninfer-perplexity", ARTIFACT, CORPUS):
        if not need.exists():
            pytest.skip(f"{need} missing")
    gpus = _idle_gpus()
    if len(gpus) < 2:
        pytest.skip("need two idle GPUs in {3,4,5,6}")
    try:
        from transformers import AutoTokenizer
        import requests  # noqa: F401
    except ImportError:
        pytest.skip("transformers/requests unavailable")
    spec = importlib.util.spec_from_file_location("diff_gate_ro", ROOT / "tools/bench/diff_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    tok = AutoTokenizer.from_pretrained(gate.TOKENIZER, trust_remote_code=False)
    prompts = [gate.build_prompt(tok, PROMPT_CTX, n) for n in NONCES]
    work = tmp_path_factory.mktemp("t4")
    text = work / "ppl.txt"
    text.write_bytes(CORPUS.read_bytes()[:150_000])
    out = {"tok": tok}
    with cf.ThreadPoolExecutor(2) as pool:
        c = pool.submit(_session, CAND, gpus[0], prompts, True, work, "cand")
        b = pool.submit(_session, BASE, gpus[1], prompts, False, work, "base")
        out["cand"], out["base"] = c.result(), b.result()
        c = pool.submit(_perplexity, CAND, gpus[0], text, work / "ppl_cand")
        b = pool.submit(_perplexity, BASE, gpus[1], text, work / "ppl_base")
        out["ppl_cand"], out["ppl_base"] = c.result(), b.result()
    return out


def test_f7_cuda_graph_and_mtp_active_with_nonzero_acceptance(runs):
    log = runs["cand"]["log"]
    assert "CUDA graphs ready" in log
    assert re.search(r"speculative\s+backend\s+mtp", log)
    for line in log.splitlines():
        if any(k in line.lower() for k in ("graph", "mtp", "speculative")):
            assert not re.search(r"disabled|fallback|degrad|unsupported", line.lower()), line
    m = re.search(r"spec_decode_accepted_tokens_total(?:\{[^}]*\})?\s+(\d+)", runs["cand"]["metrics"])
    assert m and int(m.group(1)) > 0, "no accepted speculative tokens in /metrics"


def test_f8_candidate_requests_succeed_with_positive_tokens_and_text(runs):
    for res in runs["cand"]["results"]:
        assert res["status"] == 200 and res["deltas"] > 0
        assert res["completion_tokens"] and res["completion_tokens"] > 0
        assert res["text"].strip()


def test_f8_full_length_count_not_below_baseline(runs):
    def full(side):
        return sum(r["completion_tokens"] == GEN_TOKENS for r in runs[side]["results"])
    assert full("cand") >= full("base")


def test_f8_first_32_tokens_match_baseline_commit(runs):
    tok = runs["tok"]
    for cand, base in zip(runs["cand"]["results"], runs["base"]["results"]):
        assert (tok.encode(cand["text"], add_special_tokens=False)[:32]
                == tok.encode(base["text"], add_special_tokens=False)[:32])


def test_f8_greedy_is_deterministic_on_candidate(runs):
    first, again = runs["cand"]["results"][0], runs["cand"]["repeat"]
    assert again["text"] == first["text"]
    assert again["completion_tokens"] == first["completion_tokens"]


def test_f8_perplexity_within_one_percent_of_baseline(runs):
    base, cand = runs["ppl_base"], runs["ppl_cand"]
    assert math.isfinite(base) and math.isfinite(cand) and base > 1
    assert abs(cand - base) / base <= 0.01, (base, cand)
