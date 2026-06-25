# qwen3.6-ultraspeed — Master Design & Goal Document

> Status: **design / spec phase** (no implementation yet).
> Date: 2026-06-25.
> This document is the single source of truth for *what we are building and why*. It guides
> all subsequent implementation. Kernel-level micro-decisions are deliberately left open;
> architecture, scope, and boundaries are fixed here.

---

## 1. Purpose & success criteria

Build a from-scratch C++/CUDA inference engine that runs **Qwen3.6-27B** **as fast as
physically possible** under one fixed scenario:

- **Single user, single sequence** (batch size = 1, no concurrent requests).
- **Single RTX 5090** (Blackwell, sm_120, 32 GB GDDR7, ~1.79 TB/s).

Because the scenario is fixed, we optimize for it exclusively — no dynamic compute graph,
no general-purpose model zoo (cf. llama.cpp). We hand-write the computation for *this one
model* and specialize aggressively.

### Metrics

| Metric | Priority | Definition |
|---|---|---|
| **Decode throughput** | **Primary (headline)** | tokens/sec, batch=1, single-token autoregression |
| Prefill / TTFT | Secondary (tracked) | time-to-first-token and prefill tok/s for long prompts |

### Roofline anchor

At batch=1, every linear layer is a **GEMV that is memory-bandwidth-bound on the weights**.
A dense forward reads ~all weights once per token:

```
~14–15 GB (4-bit weights) ÷ 1.79 TB/s ≈ 7.8–8.4 ms/token  ⇒  ~120–130 tok/s ceiling
```

The entire project is a campaign to approach this ceiling. (Note: `lm_head` (5120×248320)
is read *every* decode step — its precision is itself a bandwidth lever; see §9.)

---

## 2. Hardware & toolchain target

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5090, Blackwell, **sm_120**, 32 GB, ~1.79 TB/s, FP8 + FP4 tensor cores |
| CUDA | 13.1 |
| Host compiler | gcc 13.3 |
| Build | CMake ≥ 3.28 |
| OS | Linux (WSL2) |

We target sm_120 specifically and may use Blackwell-only features (FP8/FP4 tensor cores,
TMA, large shared memory) where they help.

---

## 3. Target model — frozen reference (`qwen3_5` architecture)

`Qwen3.6-27B` is internally `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`): a
**hybrid-attention, multimodal model with built-in MTP**. v1 freezes to the **text decoder
only**. These dimensions are the "固化" truth everything specializes to.

### 3.1 Text decoder

| Field | Value |
|---|---|
| num_hidden_layers | **64** |
| hidden_size | 5120 |
| intermediate_size (SwiGLU) | 17408 |
| hidden_act | silu (SwiGLU) |
| norm | RMSNorm, eps 1e-6 |
| vocab_size | **248320** |
| tie_word_embeddings | **false** (separate `lm_head`) |
| max_position_embeddings | 262144 (256K); **v1 targets 128K** |
| torch_dtype | bfloat16 |

### 3.2 Hybrid attention — 3:1 pattern (`full_attention_interval = 4`)

64 layers: every 4th layer is full softmax attention, the other three are linear attention.
⇒ **48 linear-attention layers + 16 full-attention layers.**

**Linear-attention layers (Gated DeltaNet):**

| Field | Value |
|---|---|
| linear_conv_kernel_dim | 4 (causal conv1d) |
| linear_num_key_heads | 16 |
| linear_key_head_dim | 128 |
| linear_num_value_heads | 48 |
| linear_value_head_dim | 128 |
| output_gate_type | swish |
| state dtype | **fp32** (`mamba_ssm_dtype`) |
| state size | fixed, **context-independent** |

**Full-attention layers (GQA):**

| Field | Value |
|---|---|
| num_attention_heads (Q) | 24 |
| num_key_value_heads (KV) | 4 (GQA) |
| head_dim | 256 |
| partial_rotary_factor | 0.25 (64 of 256 dims rotated) |
| rope | MRoPE, sections [11, 11, 10], interleaved |
| rope_theta | 1e7 |
| attn_output_gate | true |

### 3.3 Components present in checkpoint but OUT of v1
- **MTP** (`mtp_num_hidden_layers: 1`) — self-speculative decoding; planned later.
- **Vision tower** (depth 27, hidden 1152) + `mmproj` — multimodal; planned later.

### 3.4 Parameter budget (~26B params, ~52 GB bf16 across 15 safetensors shards)
- 64 transformer layers ≈ ~24B params.
- `embed_tokens` (248320×5120) ≈ 1.27B — gathered (1 row) at decode → memory cost only.
- `lm_head` (5120×248320) ≈ 1.27B — **read fully every decode step** → bandwidth-critical.

---

## 4. Scope & boundaries

### 4.1 In scope (v1)
- Text-only decoder forward (prefill + decode).
- **token-ids in → token-ids out**; tokenizer/detokenizer done **outside** (Python/HF).
- **Greedy (argmax)** next-token selection.
- **128K** context, **bf16 KV**.
- **W4A16** linear layers (bf16 activations, 4-bit weights dequantized to bf16).
- Python offline tooling (quantize + relayout/pack → one fixed weight file).
- C++/CUDA runtime: load the fixed file + run inference.

### 4.2 Frozen (固化)
- Model dimensions / shape, the 3:1 hybrid layer pattern.
- On-disk weight-file layout (the Python↔C++ contract).
- The model card (`qwen3_6_27b`) as the static compute graph.
- Kernel-fusion targets specialized to this model's shapes.

### 4.3 Flexible (自由)
- Kernel **API surface**: generic interfaces with multiple impls + a dispatcher.
- Memory-pool sizes, max-context, future KV-precision knob.
- Pluggable future model cards reusing L0/L1.

### 4.4 Explicitly OUT (deferred, in priority order)
1. **MTP self-speculative decode** (planned; deferred so single-token throughput can be
   optimized without accept-rate confounding the measurements).
2. fp8/fp4 **prefill** kernels.
3. **256K** context / **fp8 KV**.
4. Full sampler (temperature / top-k / top-p / RNG).
5. C++ tokenizer.
6. Vision / multimodal path.
7. Multi-GPU, tensor/pipeline parallelism, continuous batching, paged attention.

### 4.5 Decisions log (from brainstorm)
| # | Decision | Choice |
|---|---|---|
| 1 | Primary metric | Decode tok/s primary; prefill/TTFT tracked |
| 2 | Architecture | Dense (hybrid linear+full attention) |
| 3 | Multimodal | Text-only v1; don't preclude vision later |
| 4 | MTP | Planned later; single-token first |
| 5 | Compute precision | bf16 activations + W4A16 baseline; fp8/fp4 prefill later |
| 6 | Context / KV | 128K, bf16 KV; pool runtime-sized |
| 7 | 固化/自由 | Model-card graph + generic kernel API + routed specialized impls |
| 8 | Runtime I/O | ids in/out, tokenizer in Python, greedy |
| 9 | Weight pipeline | Python (quant+pack) → fixed file; C++ only loads |
| 10 | Execution model | Optimization ladder, correctness-gated |

---

## 5. System architecture — 3 layers

```
┌─────────────────────────────────────────────────────────────┐
│ L2  Model card / static graph   (src/model/qwen3_6_27b.cpp)  │
│     embed → 64-layer loop (3:1 dispatch) → norm → lm_head     │
│     owns prefill/decode drivers + KV/state lifecycle          │
├─────────────────────────────────────────────────────────────┤
│ L1  Kernels: generic API  ⟂  specialized impl + routing      │
│     rmsnorm · w4a16_gemm · gqa_attention · gdn_linear_attn    │
│     causal_conv1d · swiglu · rope_mrope · argmax              │
│     (clean header per op; dispatcher routes by dims + phase)  │
├─────────────────────────────────────────────────────────────┤
│ L0  Infra (model-agnostic, reusable)                         │
│     memory pool (device + pinned host) · weight-file loader   │
│     KV & GDN-state allocators · workspace arena · streams     │
│     tensor/view type                                          │
└─────────────────────────────────────────────────────────────┘
```

**Principle:** freedom at the API surface, 固化 in the implementation/fusion and the model
card. Adding another model later = a new L2 card reusing L0/L1, never an engine rewrite.
No dynamic graph is built at runtime — the schedule is the C++ in the model card.

### Kernel API/impl/dispatch split
Each operator is:
- a **public header** (`include/qus/kernels/...`) defining a generic, shape-parameterized API;
- one or more **CUDA implementations** (`src/kernels/.../*.cu`), possibly fused/specialized;
- a **dispatcher** that selects the best impl given known dims + phase (prefill vs decode).

A CUDA Graph (later) simply records whatever impls the dispatcher chose for the decode step.

---

## 6. Component breakdown

### L0 — Infra
- **MemoryPool** — device + pinned-host suballocation; large initial reservation, sub-buffers.
- **WeightStore / Loader** — mmap the fixed weight file; expose typed weight views; place into
  device buffers; validate header/dims against the model card's `constexpr` config.
- **KVCache** — contiguous per-(full-attn-layer) K/V buffers sized for `max_context`.
- **StateStore** — fixed-size GDN recurrent + conv state per linear-attn layer (fp32).
- **WorkspaceArena** — per-step scratch (bump allocator, reset each step).
- **Stream/handle/event** wrappers; error-checking macros.
- **Tensor** — lightweight shape/stride/dtype view (no ownership).

### L1 — Kernels (API + routed impls)
- `rmsnorm` (fp32 accumulate)
- `w4a16_gemm` / `w4a16_gemv` — 4-bit weight dequant + matmul; the decode workhorse
- `gqa_attention` — full-attn: prefill (flash-style) + decode (single-query) paths
- `gdn_linear_attn` — chunked/parallel (prefill) + recurrent (decode) paths
- `causal_conv1d` — kernel-4 short conv for linear-attn layers
- `rope_mrope` — partial RoPE (0.25) with MRoPE sections
- `swiglu` — fused gate·up + down projections
- attention output gate, residual adds, embedding gather, `argmax`

### L2 — Model card
- `constexpr ModelConfig` (the frozen dims).
- `prefill(prompt_ids) -> first_token`, `decode_step(prev_token) -> next_token`.
- Layer loop dispatching linear vs full attention on the 3:1 pattern.
- KV/state lifecycle and position bookkeeping.

### Runtime / driver
- Engine init (load weights, size pools), the prefill+decode loop, bench harness
  (`ids in → ids out`).

---

## 7. Data flow

### Prefill (prompt, many tokens, compute-bound)
```
embed gather
for layer in 0..63:
    h = rmsnorm(x)
    if full_attn(layer):  attn = gqa_attention_prefill(h)   # fills KV
    else:                 attn = gdn_linear_attn_chunked(h)  # folds prompt into state
    x = x + attn_output_gate(attn)
    h = rmsnorm(x)
    x = x + swiglu(h)
x = rmsnorm(x)
logits = lm_head(x[last])     # only last position
first_token = argmax(logits)
```

### Decode (1 token/step, memory-bound)
```
embed gather (1 token)
for layer in 0..63:
    h = rmsnorm(x)
    if full_attn(layer):  attn = gqa_attention_decode(h)     # append 1 KV slot, attend window
    else:                 attn = gdn_linear_attn_recurrent(h)# in-place state update
    x = x + attn_output_gate(attn)
    h = rmsnorm(x)
    x = x + swiglu(h)
x = rmsnorm(x)
logits = lm_head(x)
next_token = argmax(logits)
```
The decode step's kernel sequence is the unit later captured by CUDA Graph / fused / merged
into a megakernel.

---

## 8. Memory management (@128K context, bf16 KV)

| Region | Size | Notes |
|---|---|---|
| 4-bit weights (+ scales, + hi-precision sensitive tensors) | ~14–15 GB | loaded once |
| Full-attn KV (16 layers) @128K | ~8 GB | 64 KB/token; runtime-sized pool |
| GDN recurrent + conv state | ~0.15–0.2 GB | fixed-size, context-independent |
| Activations / workspace arena | ~1–2 GB | reused across layers |
| **Total** | **~24–26 GB / 32 GB** | headroom for fp8-KV / 256K later |

- Weights mmap'd from the one fixed file into device buffers (no runtime relayout).
- KV/state pools sized at startup from `(max_context, dims)`.
- Workspace arena reset every step; no per-step `cudaMalloc`.

---

## 9. Numerics & precision

| Quantity | Precision |
|---|---|
| Master activations | bf16 |
| Linear layers | **W4A16** (4-bit weights → bf16 dequant matmul) |
| KV cache | bf16 |
| GDN recurrent/conv state | fp32 |
| Norms / softmax / reductions | fp32 accumulate |

- No runtime/dynamic quantization — the file is pre-quantized and pre-laid-out.
- fp8/fp4 activation kernels are a **prefill-only** later optimization.
- **Open lever:** `lm_head` precision. Kept high-precision it costs ~2.5 GB/token of decode
  bandwidth; at 4-bit ~0.64 GB/token. To be decided by quality-vs-speed measurement.

---

## 10. Weight pipeline & file-format contract

**Two halves, one contract:**

```
bf16 safetensors ─┐
                  │  (Python, in-repo, offline)
                  ▼
        quantize (→ stable 4-bit, sensitive tensors high-precision)
                  ▼
        relayout / pack (→ kernel-optimal: weight permutation,
                          4-bit interleave, scale layout)
                  ▼
        ONE fixed weight file  ──────────►  C++/CUDA runtime: mmap + load + run
                                            (no quant, no relayout at runtime)
```

- The **file format** is a stable, self-describing container: header (magic, version, dims,
  per-tensor dtype/layout/offset/scale metadata) + tensor blobs. Spec lives in
  `tools/weight_format.md`.
- Python owns quantization *and* layout; everything in the file is pre-shaped for the kernels.
- The C++ loader validates the header against the model card's `constexpr` config and fails
  fast on mismatch.

---

## 11. Performance methodology — the optimization ladder

Profiler-driven, **correctness-gated**, in strict order:

1. **Correctness baseline** — eager driver, simplest kernels. Validate (see §12).
2. **Per-kernel optimization** — drive each kernel to its roofline (ncu: occupancy, memory
   throughput, stall reasons).
3. **Inter-kernel fusion** — kill global round-trips & launches (e.g. dequant+GEMV,
   RMSNorm+QKV, RoPE+KV-write, SwiGLU, gate+residual).
4. **CUDA Graph replay / persistent megakernel / any effective means** — collapse launch
   overhead for the decode step.

Tools: **nsys** (timeline, launch overhead), **ncu** (per-kernel roofline),
**compute-sanitizer** (correctness gate). Never optimize ahead of correctness.

---

## 12. Validation & correctness

- **Per-layer numerical parity**: dump activations from HF/vLLM for fixed inputs; compare
  cosine similarity / max-abs-error per layer.
- **End-to-end greedy token-match**: identical prompt + greedy must match vLLM & llama.cpp
  token-for-token (greedy determinism makes this nearly free).
- **compute-sanitizer** clean (memcheck/racecheck) at each stage.
- **Perf tracking** vs the §1 roofline at every ladder stage.

---

## 13. Repository layout

```
qwen3.6-ultraspeed/
  README.md                 # simple overview + quickstart
  docs/design.md            # this document
  CMakeLists.txt
  tools/                    # Python offline tooling (in-repo)
    quantize/               # bf16 safetensors -> stable 4-bit
    pack/                   # stable 4-bit -> kernel-optimal fixed file
    weight_format.md        # the file-format spec (Python<->C++ contract)
  include/qus/
    core/                   # mem pool, allocators, tensor, loader (public headers)
    kernels/                # kernel API headers
    model/                  # model config + model-card interface
  src/
    core/                   # infra impl
    kernels/                # cuda impls + dispatch
      gemm/ attention/ gdn/ norm/ rope/ activation/ sampling/
    model/qwen3_6_27b.cpp   # the model card / static graph
    runtime/                # engine: load + prefill + decode loop
    main.cpp                # bench driver: ids in -> ids out
  tests/                    # parity + kernel unit tests
  bench/                    # benchmark scripts
```

---

## 14. Roadmap / milestones

- **M0 — Skeleton:** repo, CMake, L0 infra (memory pool, tensor, loader stub), model config.
- **M1 — Weight pipeline:** Python quantize + pack; file format; C++ loader; load full model.
- **M2 — Correctness baseline:** eager forward for all op types; greedy decode;
  per-layer parity + greedy token-match vs reference.
- **M3 — Per-kernel optimization:** each kernel to roofline (W4A16 GEMV, GQA, GDN, conv,
  RoPE, SwiGLU, argmax).
- **M4 — Fusion:** fuse adjacent ops; reduce launches/round-trips.
- **M5 — Launch-overhead elimination:** CUDA Graph decode replay; explore megakernel.
- **Later:** MTP speculative decode → fp8/fp4 prefill → 256K/fp8-KV → sampler → tokenizer →
  vision → (multi-GPU/batching, only if ever needed).

---

## 15. Open questions (to resolve during implementation, non-blocking)
- `lm_head` precision (decode-bandwidth vs quality) — decide by measurement.
- Exact GDN state layout & chunk size for the prefill scan.
- W4A16 weight packing/interleave layout that best feeds the GEMV access pattern.
- Whether the embedding table stays bf16 (memory) or 4-bit.
