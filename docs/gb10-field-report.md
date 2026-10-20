# GB10 field report — measurements and feedback from a real dgx_spark machine

Status: information only. This report carries no code changes. It documents what the
`baristahaus/ninfer-gb10` fork (a GB10 port of NInfer) has measured and learned while running the
engine on a real GB10 workstation, checks upstream's Spark roofline calibration against this
machine's measurements, and offers data upstream does not publish yet.

## 1. The machine

- NVIDIA GB10 (Grace Blackwell), aarch64, 48 SMs, 24 MiB L2, 121.6 GiB unified LPDDR5x shared by
  CPU and GPU (host sees ~121 GiB).
- 20-core Arm CPU. CUDA 13.0.88, driver 580.173.02.
- Measured sustained GPU streaming read from `cudaMalloc`: **246 GB/s** on an idle machine
  (`tools/gb10/memory_probe.cu` in the fork; NVIDIA's DGX Spark specification is 273 GB/s).
- Derived compute rates: NVFP4 dense ~500 TFLOP/s (half of the 1 PFLOP sparse rating), BF16 dense
  ~125 TFLOP/s, FP32 ~31 TFLOP/s. These are derivations, not datasheet figures.

## 2. A possible discrepancy in the Spark roofline

1. **Roofline calibration.** `bench/common/roofline.h` records 257.42 GB/s sustained read,
   calibrated on a real GB10 (per `bench/README.md`, a 48-SM machine — matching this one). This
   machine measures **246 GB/s** sustained device streaming read with `tools/gb10/memory_probe.cu`
   on an idle machine (CUDA 13.0.88, driver 580.173.02). That is a 4.6 percent spread — enough to
   matter when a claim sits at 90+ percent of roofline. Worth one line in the roofline header
   about which probe and which driver produced 257.42, so downstream forks can tell a method
   difference from a unit difference.
2. **Grid caps.** Some Spark grid constants are larger than their RTX 5090 counterparts (for
   example `src/ops/elementwise/residual_add.cu`: 16384 versus 4096 block cap). The planners that
   take a runtime `sms` argument (for example
   `src/ops/softmax_attention/qsa/indexer/dgx_spark/plan.cpp`) adapt correctly to this 48-SM
   machine; the hard caps are worth one look to confirm they are bandwidth-driven rather than
   SM-count-derived.

The fork's full hardware record, including the probe tool, lives at
`tools/bench/hardware/gb10.json` and `tools/gb10/` in `baristahaus/ninfer-gb10`.

## 3. Flash-Next 125B-A6B decode on GB10 — we are at the roofline

- Artifact: FP8 MTP artifact (70.5 GiB resident), single stream, greedy, CUDA Graph decode,
  short context (512 generated tokens).
- Measured decode: **39–40 tok/s without MTP**. The bandwidth check closes: roughly 6–7 GB of
  active weights per step at 246 GB/s implies a ceiling near 38 tok/s. Single-stream decode on
  this class of machine is memory-bound end to end; kernel-side wins must come from
  overhead-shaped paths (launch efficiency, attention at long context, speculative round
  overhead), not bandwidth.
- Long-context record from the fork's performance page: 32.6/31.5 tok/s (C1/C2) without
  speculation; prefill ~2.5k tok/s from 8K to 64K context.

Implication for the Spark lane: the shape-catalog kernels cannot raise single-stream
bandwidth-bound decode above the roofline, so their value concentrates in the small-T launch
schedule, the MTP draft/verify round, and long-context attention. That matches where the fork's
profiling puts the decode time (MoE ~22%, QSA ~22%, HyperConnection ~21% at 8K context).

## 4. MTP acceptance and draft-token economics (Flash-Next, FP8 MTP artifact)

Eight fixed greedy prompts x 128 new tokens, draft-tokens 2, `lm-head-draft`:

| Metric | Value |
|---|---|
| Mean acceptance rate | 19.9% (range 12.4–34.7%) |
| Mean accept length | 1.40 tok/round (1.25–1.69) |
| Draft position 1 vs 2 accepted | 18 vs 5 (run 1) |
| Per-class best/worst | code generation 34.7%; expository text 12–15% |

Draft-token economics (same prompts, 512 tokens, greedy):

| Config | tok/s | vs no-MTP |
|---|---|---|
| draft-tokens 2 | 37 | 0.95x (net loss) |
| draft-tokens 1 | 40 | 1.03x |
| no MTP | 39 | 1.00x |

Two conclusions that may be useful for `qwen4_exp` MTP design:

1. At ~1.40 tok/round the T=1 propose forward plus the wider verify batch costs more than the
   acceptance gain on this machine. The cheapest lever is round overhead (overlap or shrink the
   propose pass), not acceptance rate.
2. The second draft position is almost never accepted. Proposal-head quality at position 2 is the
   acceptance-side lever; this supports the direction of the optimized full-vocab Q4 proposal head
   upstream added. If upstream has per-position acceptance data for the Q4 head, a comparison
   against the numbers above would be interesting.

Methodology note: the fork verified the whole MTP draft layer (QSA, MoE, proposal head,
hyperconnection mix) offline against engine dumps with a torch replica and exact oracle formulas,
then confirmed the captured decode path is byte-identical to the uncaptured path on a real model.
The dump harness and battery scripts are in `tools/gb10/` of the fork.

## 5. Determinism and KV findings

- **C1 greedy is bitwise deterministic; C4 greedy is not run-to-run deterministic**, with or
  without MTP, on this machine. Anything that compares whole-batch outputs across runs should not
  assume batch determinism at C>1.
- **FP8 KV halves the KV footprint at no decode-speed cost** versus BF16 KV in the fork's paired
  screens (FP8 can change generated continuations; throughput is unchanged).
- A CUDA-graph-captured decode round produced byte-identical output and identical acceptance
  summaries versus the uncaptured path on a real model — worth knowing when adding the segmented
  per-round graphs upstream introduced.

## 6. Operational notes for Spark-class boxes

- **ncu plus large unified-memory allocations can crash the node.** Nsight Compute fails to
  profile a process that holds a large allocation (minimal repro: `cudaMalloc` of ~70 GiB plus a
  trivial kernel), and profiling a large-model process without free memory headroom exhausted the
  unified pool and corrupted driver state (required a reboot). Recommendation worth encoding in
  docs: on unified-memory machines, use nsys (trace-only) for large models and reserve ncu for
  small targets with free memory exceeding model size plus headroom.
- **A faulted CUDA process can leave a phantom utilization counter**: `nvidia-smi` reports the
  last live utilization (for example 96%) at idle power (<=17 W) with no compute process listed.
  Cross-check power before treating the GPU as busy; the driver channel clears on reboot.
- **Single-instance discipline**: the 121 GiB pool fits one large model load at a time. Pre-launch
  free-memory checks should count pinned/registered host bytes as device capacity — upstream's
  `NINFER_SHARED_PHYSICAL_MEMORY` budget accounting already does this correctly and deserves the
  same rule in ops tooling.

## 7. Questions for upstream

1. The roofline records 257.42 GB/s sustained read for `dgx_spark`; this machine measures 246 GB/s
   with the same method class. Which probe and driver produced 257.42? Should the platform header
   carry the measured roofline so per-target claims stay reproducible?
2. Has upstream measured per-position MTP acceptance on `qwen4_exp`? If yes, do the position-2
   numbers above match the Q4 proposal head's profile?
3. Is there interest in a follow-up PR carrying the fork's EXL3 routed-bank route (per-expert
   trellis banks with directory plane and per-expert extents, validated by an independent
   numerical battery)? It composes with the Spark no-offload policy: a 125B-A6B model's banks at
   2–3 bpw fit on-device at 33–47 GiB.

## 8. What the fork can share as follow-on code PRs

- The GB10 memory probe and hardware record (`tools/gb10/`, `tools/bench/hardware/gb10.json`).
- The MTP dump/parity harness and acceptance battery (`tools/gb10/`), adapted to upstream's tree.
- The EXL3 bank route (quantize, pack, bind) with its synthetic and real-campaign test evidence.
