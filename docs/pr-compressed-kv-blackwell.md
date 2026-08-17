# PR: Live compressed-KV cache quantization on the Blackwell (sm_120a) path

**Branch:** `feat/compressed-kv-blackwell-5090`
**Base:** upstream `master` @ `32c9881` (or later)
**Scope:** compressed-KV cache (live KV-quantization) for the Qwen3.6 engine,
ported from the NVIDIA Ada (RTX 4090 / sm_89) fork onto the official
Blackwell (sm_120a / RTX 5090) codebase.

---

## What this adds

Four new live KV-quantization cache modes, selectable via `--kv-dtype`
(server and CLI):

| mode        | key storage                                   | value storage    | typical use      |
|-------------|-----------------------------------------------|------------------|------------------|
| `rk8v4`     | rotated int8 key, group-64 fp16 scale          | packed int4 V    | high-fidelity    |
| `rk4v4`     | rotated int4 key (packed)                      | packed int4 V    | long-context     |
| `rk4v4-e8`  | general Conway-Sloane **E8 lattice** key codes | packed int4 V    | long-context     |
| `rk2v4-e8`  | two-stage **E8 240-root** key codes, 2 bits/dim| packed int4 V    | max compression  |

The E8 modes are the headline: algebraic nearest-point projection onto the
Conway-Sloane E8 = D8 ∪ (D8 + ½·¹) lattice lets the K cache run at
**2–4 bits per dimension** instead of Full-Precision, so a 24 GB mobile GPU
can fit **262,144-token context** on a ~27B-parameter model — things a
bf16 or even a plain int8 KV cache can't hold at that length.

Commits:
1. `feat(ops)` — E8 lattice / root codecs, packed-rotation math, and the
   templated attention kernels, launchers, wrapper + data-model foundation.
2. `feat(serve)` — server `--kv-dtype` parsing + logging labels.
3. `feat(engine)` — runtime storage routing through the qwen3.6 planner.
4. `test(kv)` — standalone correctness oracle + build wiring + README.

## Why

24 GB-class Blackwell-class mobile GPUs (GeForce RTX 5090, sm_120a) can
serve long-context LLMs if the KV cache is compressed at attention time.
This brings the tried E8 compressed-KV machinery to the official Blackwell
tree so the market-facing architecture is the target-supported one.

## Attribution

Full credit goes to the original implementer:

- **UDPSendToFailed/ninfer-4090** — a fork of **Neroued/ninfer** target
  the NVIDIA Ada (sm_89 / GeForce RTX 4090) generation. **All of the
  compressed-KV cache design — the Hadamard-rotated K/V, the packed
  int4 V, and the E8 Conway-Sloane lattice / 240-root codec mathematics —
  originates here. Full credit to this fork's author.**
- itself derived from **Don-Chad/ninfer-3090**.

This branch is a **port of that source of truth onto the official
Blackwell (sm_120a) codebase**. The E8 codecs, the packing/rotation math,
and the registry of `rk*` modes are unchanged from the 4090 fork; the
changes here are the Blackwell target and the integration layer
(launchers, wrapper, runtime storage wiring). Attribution is also carried
in each commit body and in `tools/test_kv/README.md`.

## Verification evidence

Shipped with this PR is a standalone correctness oracle
(`tools/test_kv`, branch commit 4):

```text
NInfer: True E8 Conway-Sloane Lattice Microbenchmark
Corpus: 1,000,000 tokens, 256-dim, 5 embedded needles @ 5/25/50/75/95%

Method 1: Two-Stage E8 Root Codec (2-bit)  -> cosine ~100.0 vs FP32, needles PASSED
Method 2: General Conway-Sloane E8 Lattice -> cosine ~100.0 vs FP32, needles PASSED
Needle Retrieval Ranking: 5/5 [PASSED - 100% RETRIEVED]
```

- **100% needle retrieval** across 1M tokens for both codecs — the needles
  (high-magnitude directional key/query targets) are recovered at their
  exact indices by the quantized attention returning the same top scores
  as the full-FP32 reference.
- On this hardware an end-to-end sanity run of `rk2v4-e8` served a
  Qwen3.6-27B NVFP4 model **generating correctly at 262,144-token
  context** on a 24 GB-class Blackwell-class GPU.

### Reproduce

```sh
nvcc -arch=sm_120a -O3 tools/test_kv/test_e8_codec.cu \
     tools/test_kv/verify_1m_retrieval.cu -o verify_1m_retrieval
./verify_1m_retrieval            # 1,000,000 tokens default
./verify_1m_retrieval 2000000    # larger corpus, if VRAM allows
```

(or `cmake --build build --target ninfer_kv_e8_verify`; see
`tools/test_kv/README.md`).

## Scope / non-goals

- Adds the **E8 codec + compressed-KV storage** to the Qwen3.6 engine
  path. No changes to weights, no new formats, no changes to the
  Full-Precision path (defaults unchanged).
- Only the Blackwell (sm_120a) architecture is supported by NInfer and by
  this port (the reuse is source-level; there is no sm_89 target here).
- Building/running the oracle and the engine are unchanged otherwise.