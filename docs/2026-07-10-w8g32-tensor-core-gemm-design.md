# W8G32 Tensor Core GEMM architecture — SM120 roofline design

Date: 2026-07-10

Target: Qwen3.6-27B MTP prefill on one RTX 5090 (GB202, compute capability 12.0)

Status: implemented and measured; see `2026-07-10-w8g32-tensor-core-gemm-implementation-report.md`

Implementation note: finite tuning rejected the proposed direct-register producer/consumer
pipeline. The production kernel uses the same BF16-MMA numerical contract and tile, but a
cooperative raw-code `cp.async.cg` pipeline was faster and safer. The implementation report records
the final architecture, measured deviations, and acceptance evidence; this document remains the
original design and gate definition.

## 1. Decision

Add a **dedicated W8G32 Large-T kernel family**. Do not route W8G32 through the existing
Q4/Q5/Q6 MMA implementation and do not add W8 branches to that implementation. The only shared
pieces should be small, format-independent PTX helpers such as `cp.async`, `ldmatrix`, BF16
`mma.sync`, shared-address conversion, and barrier wrappers.

The production math path is:

```text
W8G32 codes + FP16 scales --on-chip dequantize/round--> BF16 A fragments
BF16 x ------------------------------------------------> BF16 B fragments
BF16 A x BF16 B --mma.sync.m16n8k16, FP32 accumulate--> BF16 output
```

The primary kernel is a warp-specialized, two-stage producer/consumer pipeline:

```text
                       stage s                         stage s^1
             +-------------------------+     +-------------------------+
4 producer   | cp.async BF16 x -> Bs   |     | fill the next Bs        |
warps        | load W8 + scale          | <-> | dequant W8 -> BF16 As   |
             | dequant W8 -> BF16 As   |     |                         |
             +------------+------------+     +------------+------------+
                          | named barrier                  |
                          v                                v
8 consumer   +---------------------------------------------------------+
warps        | ldmatrix As/Bs -> mma.sync.m16n8k16 -> FP32 accumulators |
             +---------------------------------------------------------+
```

At the main `BM=64, BN=128, BK=64` tile, double-buffered `As+Bs` uses exactly 48 KiB:

- `As[2][64][64]` BF16: 16 KiB;
- `Bs[2][128][64]` BF16: 32 KiB;
- W8 codes and FP16 scales are loaded directly into producer registers, so there is no raw-weight
  shared-memory staging plane.

The RTX 5090 exposes 100 KiB shared memory per SM and 99 KiB opt-in per block. Including the CUDA
per-block reservation, two 48 KiB CTAs fit on one SM. Two resident CTAs provide 16 consumer MMA
warps and 8 producer warps per SM. The register gate is `<= 85 registers/thread` for the 384-thread,
two-CTA configuration; the implementation must not trade the second resident CTA for a larger
accumulator tile.

The first implementation uses BF16 Tensor Cores, not INT8 Tensor Cores. W8G32 is a weight storage
format; the public operator contract is `W8G32 x BF16 -> BF16`. Native INT8 MMA would require a new
activation quantization contract and a per-32-K rescale of every integer partial. That is a separate
accuracy/performance experiment, not the production baseline.

## 2. Scope

### 2.1 In scope

- A standalone W8G32 BF16-MMA kernel for every valid Large-T MTP linear shape.
- Fast full-tile paths for the real MTP dimensions and a predicated edge path.
- Shape-aware tile selection for the current critical matrices:
  - MTP FC: `[M,K]=[5120,10240]`;
  - MTP K projection: `[1024,5120]`;
  - MTP V projection: `[1024,5120]`.
- Coverage of the other W8G32 operator shapes even though the KV-only prompt schedule no longer
  executes their Large-T forms:
  - fused attention input `[14336,5120]`;
  - O projection `[5120,6144]`;
  - fused MLP gate/up `[34816,5120]`;
  - MLP down `[5120,17408]`.
- A fused shared-input K/V variant after the single-GEMM kernel reaches its roofline gate.
- Real operator numerical tests, Compute Sanitizer, per-shape benchmark sweeps, NCU roofline
  captures, and full-engine MTP prefill confirmation.

### 2.2 Out of scope

- Changing q5090 v4.1, W8G32 group size, scale dtype, row-split layout, or converter output.
- Pre-dequantizing and permanently storing a BF16 copy of W8 weights.
- Changing MTP decode/T=1 away from its current memory-bound path.
- Quantizing BF16 activations in the production kernel.
- Adding CUTLASS as a runtime or build dependency.
- Cluster launch, persistent CLC scheduling, TMA tensor maps, or split-K before metrics prove that
  the simpler static schedule cannot meet the gate.

## 3. Evidence and current bottleneck

The implemented MTP KV-only prompt schedule leaves exactly three Large-T W8G32 launches per
1024-token chunk: FC, K, and V. In the matched BF16-KV Nsys trace they cost:

| launch | grid in current small-T kernel | duration |
|---|---:|---:|
| FC `[5120,10240,1024]` | `640 x 128`, 256 threads | 9.263 ms |
| K `[1024,5120,1024]` | `128 x 128`, 256 threads | 0.709 ms |
| V `[1024,5120,1024]` | `128 x 128`, 256 threads | 0.992 ms |
| total | 3 launches | 10.964 ms |

The isolated FC median is 8.034 ms and 13.36 useful TFLOP/s. NCU reports:

- zero HMMA/tensor-pipe instructions;
- 81.82% SM throughput spent on CUDA-core work;
- 49.73% achieved occupancy, limited by 66 registers/thread;
- only 4.68% DRAM throughput and 40.11% L2 throughput;
- 128 token tiles at `T=1024`, so the current eight-token kernel repeatedly replays work that a
  GEMM tile should reuse.

For comparison, the existing Q4 BF16-MMA kernel on `[34816,5120,1024]` reaches 168.88 useful
TFLOP/s and 82.59% tensor-pipe activity. The W8 problem is therefore an absent backend, not a small
parameter-tuning miss.

Artifacts:

- `profiles/ncu/mtp-prefill-overhead-20260710/report.md`;
- `profiles/ncu/mtp-prefill-overhead-20260710/*.ncu-rep`;
- `profiles/nsys/mtp-prefill-kv-only-20260710/pp1024_tg256_bf16_k3.sqlite`;
- `profiles/bench/mtp-prefill-kv-only-20260710/report.md`.

## 4. Hardware and roofline model

The exact profiled device properties are:

| property | value |
|---|---:|
| GPU | GeForce RTX 5090 / GB202 / CC 12.0 |
| SM count | 170 |
| L2 | 96 MiB |
| registers | 64 Ki 32-bit registers/SM |
| maximum resident warps | 48/SM |
| shared memory | 100 KiB/SM, 99 KiB opt-in/block |
| measured stream-copy ceiling | 1.508 TB/s |
| measured BF16 `mma.sync` ceiling | 196-221 TFLOP/s across existing runs |

For repository layout `out[M,Ntok] = W[M,K] * x[K,Ntok]`, count:

```text
FLOPs       = 2 * M * Ntok * K
weight bytes = M * K * (1 + 2/32)        # one int8 code + one FP16 scale/G32
x bytes      = 2 * K * Ntok
out bytes    = 2 * M * Ntok
AI_min       = FLOPs / (weight + x + out)
roof         = min(measured_BF16_TC, measured_HBM * AI_min)
```

At a concurrently measured 200 TFLOP/s and 1.508 TB/s:

| shape | T | useful FLOPs | minimum bytes | AI | roof | ideal floor |
|---|---:|---:|---:|---:|---:|---:|
| FC `[5120,10240]` | 64 | 6.711 GF | 57.672 MB | 116.4 | 175.5 TF/s | 38.2 us |
| FC `[5120,10240]` | 128 | 13.422 GF | 59.638 MB | 225.1 | 200 TF/s | 67.1 us |
| FC `[5120,10240]` | 1024 | 107.374 GF | 87.163 MB | 1231.9 | 200 TF/s | 536.9 us |
| K or V `[1024,5120]` | 64 | 0.671 GF | 6.357 MB | 105.6 | 159.2 TF/s | 4.2 us |
| K or V `[1024,5120]` | 128 | 1.342 GF | 7.143 MB | 187.9 | 200 TF/s | 6.7 us |
| K or V `[1024,5120]` | 1024 | 10.737 GF | 18.153 MB | 591.5 | 200 TF/s | 53.7 us |

The `T=1024` targets are compute-bound. `T<=64` is memory/launch-bound and needs a smaller tile or
the existing small-T backend. The dispatch crossover must be measured per shape; `T>16` is not
automatically an efficient MMA region for the small-M K/V projection.

The logical-byte roofline is only the first gate. NCU must also report actual DRAM/L2 sectors. The
FC W8 payload is 53.1 MiB and FC input is 20 MiB, so their combined live set fits in the 96 MiB L2.
K/V payloads are much smaller. For the 180.6 MiB fused gate/up payload, tile raster order must retain
one weight row tile across token tiles rather than assuming the entire matrix fits in L2.

## 5. Numerical contract

### 5.1 Current path

The correctness-first CUDA-core path is approximately:

```text
out = BF16( sum_k FP32(int8_code * FP16_scale) * FP32(BF16_x) )
```

### 5.2 New BF16-MMA path

The Tensor Core path is:

```text
w_bf16 = BF16_RN(FP32(int8_code) * FP32(FP16_scale))
out    = BF16_RN( sum_k FP32(w_bf16 * BF16_x) )
```

The only intentional precision change is the BF16 rounding of each dequantized weight before the
MMA. Activations remain BF16, accumulation remains FP32, scales are still decoded exactly from FP16,
and the output is still rounded to BF16.

The implementation must not:

- dynamically quantize activations;
- approximate or downcast the FP16 scale before the FP32 multiply;
- use BF16 scale multiplication with two roundings;
- change NaN/zero-scale/sign-extension behavior;
- use reduced-precision accumulation;
- enable an accuracy-changing fast path based only on performance.

This precision change is more important for W8 than Q4: BF16's seven explicit precision bits are
close to the effective precision of an eight-bit code. Operator testing therefore has two oracles:

1. **Kernel oracle:** CPU/GPU reference explicitly rounds dequantized W8 values to BF16 before an
   FP32 dot. This catches layout, scale, tail, and MMA-fragment errors.
2. **Model oracle:** existing FP64 W8 dequant reference without early BF16 rounding. This measures
   the actual precision delta and must pass the repository's Tensor Core linear tolerance.

If the MTP quality gate fails, the first precision rescue is a two-term weight decomposition:

```text
w_hi = BF16(w)
w_lo = BF16(w - FP32(w_hi))
W*x  = mma(w_hi, x) + mma(w_lo, x)
```

That variant nearly restores FP32-dequantized weight precision but doubles MMA work. It is a
fallback experiment, not part of the fast production path and not allowed unless real MTP quality
evidence requires it.

## 6. Weight layout and dequantization

W8G32 row-split is already well matched to the target:

- codes are `[row][group][32 signed int8]`;
- every code group is one aligned 32-byte transaction;
- two adjacent G32 groups exactly fill `BK=64`;
- their two FP16 scales are adjacent and can be loaded as one 32-bit scale pair;
- every real MTP K is a multiple of 128;
- a segment view advances both code and scale pointers by complete rows, so the same kernel handles
  standalone FC and the K/V slices of the fused attention block.

No offline repack is needed. For each `(row, BK=64)` producer task:

1. lane 0 loads the two FP16 scales as one aligned `u32` and broadcasts it;
2. lanes 0-15 each load two adjacent signed codes as one aligned `u16`;
3. sign-extend both bytes;
4. convert to FP32, multiply by the selected FP16-derived FP32 scale;
5. round the pair once with `__floats2bfloat162_rn`;
6. write one BF16 pair into the XOR-swizzled `As` stage.

Sixteen active lanes issue one coalesced 32-byte code transaction per group. The other lanes remain
available for independent row tasks in later loop iterations; do not add lane-to-lane code shuffles
unless SASS/NCU proves they reduce producer time.

The shared layout uses the existing conflict-free 8-BF16-group XOR scheme:

```text
swizzle(row, col) = (((col >> 3) ^ (row & 7)) << 3) | (col & 7)
```

The same transform is applied when producers store `As/Bs` and when consumers form `ldmatrix`
addresses. The W8 kernel owns its copy of the policy; only the tiny swizzle helper may be shared if
it remains format-independent.

## 7. Main throughput kernel

### 7.1 Tile

The first roofline candidate is:

| level | tile / assignment |
|---|---|
| CTA output | `BM=64, BN=128` |
| K step | `BK=64` = two W8G32 groups = four BF16 `k16` MMA steps |
| consumer warp output | `WM=64, WN=16` |
| consumer warps | 8 (`1 x 8` over the CTA output tile) |
| producer warps | 4 |
| threads | 384 |
| accumulators | 32 FP32 values/consumer thread |
| stages | 2 |
| shared memory | 48 KiB |
| requested residency | 2 CTAs/SM |

The 8 consumers reuse every `As` fragment across two `n8` fragments and every `Bs` fragment across
four `m16` fragments. Each consumer warp issues 32 `m16n8k16` MMAs per BK step; the CTA issues 256.

### 7.2 Producer/consumer ping-pong

Use two named full-CTA barriers, one per stage. The barrier sequence itself provides both readiness
and overwrite safety:

```text
producer: fill stage 0 -> barrier 0 -> fill stage 1 -> barrier 1 -> fill stage 0 -> ...
consumer: wait barrier 0 -> MMA stage 0 -> barrier 1 -> MMA stage 1 -> barrier 0 -> ...
```

After `barrier 0` releases, consumers read stage 0 while producers fill stage 1. Consumers finish
stage 0 before arriving at `barrier 1`, so producers may safely overwrite stage 0 after barrier 1
releases. No block-wide barrier is placed inside the consumer MMA interval.

Within a producer stage:

1. cooperatively issue 16-byte `cp.async` copies for `Bs[stage]`;
2. commit the async group;
3. synchronously load codes/scales and dequantize W into `As[stage]` while the B copies progress;
4. wait for the async group;
5. arrive at the stage barrier.

This overlaps three pipelines:

- LDGSTS/global-to-shared traffic for BF16 x;
- scalar integer/FP32/BF16 conversion work for W8 dequantization;
- Tensor Core MMA for the other stage.

The kernel uses `__launch_bounds__(384, 2)`. A build that spills or allocates more than 85
registers/thread does not satisfy the architecture even if one microbenchmark happens to improve;
losing the second CTA removes the phase diversity needed to cover producer and barrier latency.

### 7.3 Fragment schedule

Consumers use:

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for each 16x16 A fragment;
- `ldmatrix.sync.aligned.m8n8.x2.shared.b16` for each 16x8 B fragment;
- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`;
- two register fragment sets so the next `ldmatrix` can be issued before the current fragment's MMA
  dependency chain completes, provided the register gate still holds.

There are independent FP32 accumulator chains across the four M fragments and two N fragments.
This is enough ILP to saturate the tensor subpipe without increasing `WN` and doubling accumulator
pressure.

### 7.4 Epilogue

The first implementation converts FP32 accumulators to BF16 and uses the documented MMA lane layout
for direct global stores. Critical full tiles have no predicates.

A shared-memory vectorized epilogue is added only if NCU shows store-sector inefficiency or a real
epilogue bottleneck. It may overlay a no-longer-used stage, but it must not increase the steady-state
48 KiB allocation or add a barrier to every K step.

## 8. Small-M and boundary configurations

One tile cannot be roofline-efficient for both FC and `[1024,5120]` K/V. The implementation remains
one algorithm family with a small, compile-time configuration set; it is not a runtime autotuner.

Seed candidates:

| policy | CTA tile | producer/consumer warps | shared | intended region |
|---|---|---:|---:|---|
| throughput | `64x128x64` | `4 + 8` | 48 KiB | FC and large M, `T>=128` |
| balanced | `64x64x64` | `2 + 4` | 24 KiB | medium T / large M |
| small-M | `32x64x64` | `2 + 4` | 24 KiB | K/V and other `M=1024` |
| narrow-tail | `32x32x64` | `2 + 2` | <=16 KiB | measured crossover/tail only |

For K/V at `T=1024`, `32x64` creates `32 x 16 = 512` CTAs, roughly three CTAs per SM, instead of
the 128 CTAs in the current path. Four consumer warps per CTA and at least three active CTAs provide
12+ MMA warps per active SM with only 16 FP32 accumulators per consumer thread.

Tile selection is frozen from benchmark data for the finite Qwen3.6 shape set. The production code
must not retain an environment-variable tile selector. A temporary benchmark-only selector may be
used during the sweep and must be deleted when dispatch is finalized.

## 9. Fused K/V specialization

After the standalone small-M kernel passes numerical and roofline gates, add an MTP-only paired
kernel because K and V consume the same normalized activation:

```text
w8g32_mtp_kv_gemm(ah, k_weight, v_weight, k_out, v_out)
```

The paired CTA owns `32 K rows + 32 V rows` by `64 tokens`:

- one double-buffered `Bs` is shared by both projections;
- separate double-buffered `As_k` and `As_v` planes hold the two weights;
- four consumer warps accumulate K and four accumulate V;
- four producer warps stage x and both weights;
- one launch replaces two launches;
- the useful-FLOP roofline is computed over both outputs, not by hiding work in a fusion metric.

The proposed shared footprint is 32 KiB:

```text
Bs[2][64][64]                  = 16 KiB
As_k[2][32][64] + As_v[...]   = 16 KiB
```

Two resident 384-thread CTAs are required initially to keep the register budget practical. A third
CTA is allowed only if ptxas reports no spills and the measured tensor-pipe result improves.

This specialization belongs in its own `.cu/.cuh`; it reuses fragment and barrier helpers but does
not add paired-output branches to the standalone kernel.

## 10. Edge handling and dispatch

### 10.1 Full fast path

All production MTP dimensions satisfy:

- `M` is a multiple of 32/64 for the selected tile;
- `K` is a multiple of 128 and therefore `BK=64`;
- `T=1024` is a multiple of every candidate BN.

Instantiate a `FullTiles=true` kernel that has no M/N/K predicates in the mainloop or epilogue.

### 10.2 Predicated path

The same algorithm supports a non-multiple final prompt chunk:

- out-of-range x copies zero-fill shared memory;
- out-of-range W rows zero-fill `As`;
- output stores predicate M and Ntok;
- padded K codes may be read, but logical `k>=K` activation values are zero;
- a non-16-byte-aligned logical x row uses a scalar/vector tail rather than an unaligned 16-byte
  `cp.async`.

Generic odd-K descriptors remain correct. They need not meet the MTP roofline gate. If the tail
path becomes disproportionately complex, odd K continues to use the correct small-T backend while
all real Qwen3.6 K values use the MMA backend.

### 10.3 Regime dispatch

Keep the existing W8 T=1/small-T implementation. Replace only the measured Large-T region. The
final crossover is selected independently for FC/large-M and K/V/small-M from:

```text
T = 2, 3, 4, 5, 6, 8, 16, 17, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024
```

For `T>=128`, the critical production policy must use the MMA kernel. Below that point, dispatch is
whichever implementation has lower cold median without a correctness compromise.

## 11. Raster order, cache reuse, and split-K

Use a flattened static CTA grid with explicit `(tile_m, tile_n)` mapping. Benchmark two raster
orders:

- Ntok-major: reuses x while streaming all M weight tiles;
- M-major in small groups: keeps a W row tile hot across several Ntok tiles.

FC's W+x live set fits L2, so the lower-overhead order wins. The full fused gate/up W8 payload does
not fit L2, so grouped M-major order is required to avoid re-reading the whole weight from DRAM for
every token tile.

Do not use split-K for the T=1024 critical shapes:

- their output grids already expose hundreds of CTAs;
- split-K adds a large FP32 partial workspace and a reduction kernel;
- it multiplies output traffic and interferes with CUDA Graph/workspace simplicity.

For very narrow T, first reduce BN/BM. Split-K is allowed only if the resulting grid still cannot
occupy the device and a measured end-to-end win exceeds the reduction/workspace cost.

## 12. Why native INT8 MMA is not the primary kernel

SM120 supports `mma.sync.aligned.m16n8k32...s8.s8.s32`, but the second operand is BF16 today. A
native integer path would require:

1. quantize every `[token,G32]` activation group to int8;
2. store an activation scale for every `(token,group)`;
3. run one int8 MMA per G32 group into a fresh S32 fragment;
4. convert the S32 partial to FP32;
5. multiply by `w_scale[row,group] * x_scale[token,group]`;
6. accumulate that rescaled partial into FP32 output accumulators.

The S32 accumulator cannot contract across groups because the scale product changes with row,
token, and group. The path therefore adds an activation-quantization kernel or fused quantization,
S32-to-FP32 conversions, a scale outer product, FP32 FMAs every K32, and additional registers. It
also changes MTP numerics more than BF16 weight dequantization.

SM120's new block-scaled narrow-precision MMA formats do not directly consume this artifact's
signed-int8 codes, arbitrary FP16 W scales, and BF16 activation. Reinterpreting W8G32 as FP8 or
changing scales to UE8M0 would change the file/model contract.

An INT8-MMA prototype is allowed only after the BF16-MMA kernel reaches roofline. It must be measured
as an end-to-end optional policy with the same MTP quality gates; it cannot silently replace the
precision-first W8G32 path.

## 13. Code ownership

Expected implementation boundaries:

- `src/kernels/linear/gemm/linear_rowsplit_w8g32_gemm_mma.cuh`
  - W8-only dequant, stage layout, warp-specialized standalone kernel, fragment helpers local to W8.
- `src/kernels/linear/gemm/linear_rowsplit_w8g32_gemm_mma.cu`
  - full/tail instantiations, shape/T dispatch, launch attributes, error checks.
- `src/kernels/linear/gemm/linear_rowsplit_w8g32_kv_gemm_mma.{cuh,cu}`
  - later paired K/V specialization.
- the existing CUDA kernel helper seam
  - reuse only genuinely format-independent `cp.async` support; keep named barriers, W8 stage
    layout, `ldmatrix`, and BF16 MMA helpers local until a second real consumer justifies moving
    them.
- `src/kernels/linear/plan/linear_plan.{h,cpp}`
  - explicit W8G32 MMA policy, a real `MtpKV1024x5120` shape family, and shape-aware crossover.
- `src/kernels/linear/linear.cpp`
  - route W8G32 Large-T to the new launcher; keep T1/small-T direct.
- `include/qus/kernels/linear.h`
  - add the paired K/V entry point only when the paired kernel is integrated.
- `bench/linear_op_bench.cu`
  - add the actual `[1024,5120]` MTP K/V target and retain simultaneous stream/TC ceilings.
- `tests/kernels/test_linear.cpp`
  - numerical oracle, real/edge shapes, and BF16-MMA tolerance coverage.
- `src/model/qwen3_6_27b.cpp`
  - switch the two K/V calls to the paired operator only after its independent gate passes.

Do not add a generic GEMM framework, runtime autotuner, compatibility flag, or a second artifact
layout.

## 14. Performance gates

### 14.1 Roofline definition

For each benchmark row, measure the stream and BF16 MMA ceilings in the same process/run. Define:

```text
compute_floor = useful_flops / measured_tc_flops
memory_floor  = minimum_unique_bytes / measured_stream_bytes
launch_floor  = median empty-kernel/event overhead for the same stream
ideal_time    = launch_floor + max(compute_floor, memory_floor)
efficiency    = ideal_time / measured_kernel_time
```

For NCU, replace minimum bytes with actual DRAM/L2 sectors when diagnosing hierarchy limits. NCU
replay time is diagnostic only; absolute acceptance uses the unprofiled benchmark.

### 14.2 Mandatory gates

Critical `T=1024` gates:

- standalone FC useful throughput `>= 90%` of the simultaneously measured BF16 Tensor Core ceiling;
- standalone K and V each `>= 85%`, or paired K/V aggregate `>= 90%`;
- combined FC + K/V GPU time `<= 0.85 ms` at a 196-221 TFLOP/s measured ceiling;
- tensor-pipe/HMMA instruction count is nonzero and matches the mathematical tile count;
- FC tensor-pipe active `>= 90%`; paired/small-M tensor-pipe active `>= 85%`;
- no local-memory spills;
- throughput kernel `<=85` registers/thread and two resident CTAs/SM;
- no more than 15% excess DRAM traffic over the cache-aware expected stream on FC/K/V;
- no regression in Q4/Q5/Q6 paths or W8 T=1/small-T.

The fixed `0.85 ms` total corresponds to roughly 13x over the current 10.964 ms Nsys total and
leaves headroom for launch/epilogue overhead around the approximately 0.64-0.73 ms mathematical
floor observed at current clocks.

For `T=17..127`:

- dispatch must be no slower than the current W8 implementation;
- the chosen backend should reach `>=80%` of the launch-aware hierarchical roofline once the
  operation is large enough for its compute/memory floor to exceed launch noise;
- tail-token waste from BN padding must be reported, not hidden in a useful-TFLOP number.

### 14.3 Full-engine gate

On CUDA Graph `qus_bench` with BF16 and INT8 KV:

- MTP acceptance rate/length and generated token stream must satisfy the established MTP E2E and
  quality contract;
- `pp1024+tg256` Large-T W8 time must fall from about 10.5-11.0 ms to `<=0.85 ms`;
- prefill wall time must improve in both KV modes;
- no decode regression, because decode stays on the existing W8 T=1 kernel;
- long prompts (`pp8192`, `pp32768`) must show the per-chunk gain without a workspace increase.

## 15. Profiling protocol

### 15.1 Benchmark sweep

Build Release with line info and run cold-cache medians for all five W8 blocks plus actual K/V:

```bash
cmake --build build --target qus_linear_op_bench qus_linear_test -j

./build/bench/qus_linear_op_bench \
  --shape MtpFc5120x10240 --qtype W8G32 \
  --t-sweep 17,32,64,96,128,256,512,1024 \
  --warmup 3 --repeat 20 --copy-repeat 8 \
  --csv-out profiles/w8g32-tc-gemm/fc.csv

./build/bench/qus_linear_op_bench \
  --shape MtpKV1024x5120 --qtype W8G32 \
  --t-sweep 17,32,64,96,128,256,512,1024 \
  --warmup 3 --repeat 20 --copy-repeat 8 \
  --csv-out profiles/w8g32-tc-gemm/kv.csv
```

During tuning, record clocks, power/thermal limit state, stream ceiling, TC ceiling, cold median,
warm median, p95, useful TFLOP/s, and roofline efficiency. Never compare a kernel from one clock
state with a ceiling from another run.

### 15.2 NCU

First prove the regex captures the intended kernel:

```bash
ncu --force-overwrite \
  -o profiles/ncu/w8g32-tc-gemm/fc_t1024_basic \
  --set basic \
  --kernel-name regex:'linear_rowsplit_w8g32_gemm_mma_kernel' \
  --launch-skip 0 --launch-count 1 --replay-mode application \
  ./build/bench/qus_linear_op_bench \
    --shape MtpFc5120x10240 --qtype W8G32 --t-sweep 1024 \
    --warmup 0 --repeat 1 --copy-repeat 1 --stream-ceiling-gbs 1508.2
```

Then collect only the needed sections:

```bash
ncu --force-overwrite \
  -o profiles/ncu/w8g32-tc-gemm/fc_t1024_detailed \
  --section SpeedOfLight --section Occupancy \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'linear_rowsplit_w8g32_gemm_mma_kernel' \
  --launch-skip 0 --launch-count 1 --replay-mode application \
  ./build/bench/qus_linear_op_bench \
    --shape MtpFc5120x10240 --qtype W8G32 --t-sweep 1024 \
    --warmup 0 --repeat 1 --copy-repeat 1 --stream-ceiling-gbs 1508.2
```

Required report fields:

- intended kernel name, grid/block, launch skip/count, replay mode;
- duration, useful TFLOP/s, SM and tensor-pipe throughput;
- achieved/theoretical occupancy, blocks/SM, warps/SM;
- registers/thread, static/dynamic shared memory, spills;
- DRAM/L2/L1 sectors, hit rates, and throughput;
- issue slots, eligible warps, top warp stalls;
- producer vs consumer source/SASS hotspots when the tensor gate is missed.

Use Nsys again only after microbench gates pass, to verify launch count, prefill phase attribution,
CUDA Graph behavior, and the paired K/V integration.

## 16. Verification

### 16.1 Numerical

Allowed high-value kernel coverage:

- W8G32 G32 code/scale addressing, including negative codes, `-127`, zero scale, and scale extremes;
- full and edge tiles around `M={31,32,33,63,64,65}`, `T={17,31,32,63,64,127,128,1023,1024}`;
- logical K tails around 32/64/128 boundaries in the generic path;
- actual `[5120,10240]` FC and `[1024,5120]` K/V shapes at representative T;
- multiple deterministic seeds and nonuniform FP16 scales;
- standalone and fused K/V parity against the BF16-dequant kernel oracle;
- tolerance against the existing high-precision W8 oracle.

### 16.2 Safety

- build the affected targets;
- run `qus_linear_test` and affected MTP E2E tests;
- run Compute Sanitizer memcheck on full-tile, tail-tile, segment-view K/V, and paired K/V cases;
- verify no use-after-workspace-rewind in the engine path;
- inspect ptxas register/spill output for every production instantiation.

### 16.3 Model quality

- deterministic prompt/token parity for canonical MTP E2E fixtures;
- BF16/INT8 KV variants;
- acceptance rate and mean accepted length on the existing representative corpus;
- explicit comparison of BF16-MMA weight rounding against the prior W8 path;
- reject the fast path if the quality delta exceeds the project's established MTP threshold, even
  if the operator norm test passes.

## 17. Implementation sequence

1. Add actual K/V shape to `linear_op_bench`, capture fresh current baselines and simultaneous
   ceilings.
2. Land a standalone W8 full/tail BF16-MMA correctness kernel in separate files.
3. Replace raw-weight shared staging with direct register dequantization into swizzled BF16 As.
4. Implement the `4 producer + 8 consumer`, two-stage named-barrier pipeline and enforce the
   two-CTA resource gate.
5. Sweep only the finite tile set and freeze FC/large-M and K/V/small-M dispatch.
6. Pass operator numerical, sanitizer, and NCU roofline gates.
7. Add the separate paired K/V kernel and switch only the MTP KV-only prompt path.
8. Re-run CUDA Graph full-engine BF16/INT8 KV sweeps and matched Nsys captures.
9. Investigate TMA, persistent scheduling, or native INT8 MMA only for a metric-proven remaining
   ceiling gap.

## 18. Definition of done

The work is complete only when all of the following are true:

- W8G32 Large-T no longer launches `linear_rowsplit_gemm_smallt_kernel`;
- FC and K/V execute Tensor Core MMA with the stated accuracy contract;
- the critical T=1024 kernels pass the roofline, residency, no-spill, and traffic gates;
- arbitrary final prompt chunks are correct and sanitizer-clean;
- W8 decode/small-T and all Q4/Q5/Q6 results are unchanged;
- BF16 and INT8 KV full-engine MTP correctness/quality gates pass;
- matched Nsys shows `<=0.85 ms` for the three former Large-T W8 launches, or the two launches after
  K/V fusion;
- the final code contains no tuning environment flag, compatibility route, duplicated old
  Large-T W8 policy, or unused experimental INT8 path.

## 19. Hardware references

- NVIDIA PTX ISA: BF16 `mma.sync.m16n8k16`, INT8 `mma.sync.m16n8k32`, and fragment layouts:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>
- NVIDIA Blackwell tuning guide: CC 12.0 occupancy and shared-memory limits:
  <https://docs.nvidia.com/cuda/blackwell-tuning-guide/>
- NVIDIA CUDA Programming Guide compute-capability tables:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- NVIDIA CUTLASS Blackwell documentation, used only as an architecture reference for SM120 tile and
  pipeline organization:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms>
