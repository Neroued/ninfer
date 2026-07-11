# W8G32 Tensor Core GEMM implementation report

Date: 2026-07-10

Target: Qwen3.6-27B MTP prefill on RTX 5090 (GB202, CC 12.0)

Status: implemented, integrated, numerically verified, sanitizer-clean, and profiled

## Outcome

W8G32 Large-T no longer uses the CUDA-core small-T GEMM. It has an independent BF16 Tensor Core
backend plus a separate paired MTP K/V kernel. BF16/Q4/Q5/Q6 code is unchanged.

At `T=1024`:

| operation | before | final | useful throughput | result |
|---|---:|---:|---:|---|
| FC `[5120,10240]` | 8.312 ms | 603.52 us | 177.91 TFLOP/s | 13.77x faster; 90.93% of same-run TC ceiling |
| K + V `[1024,5120]` | two standalone launches | 155.01 us paired | 138.54 TFLOP/s aggregate | one launch; activation staged once |
| FC + K/V in Nsys | 10.964 ms | about 0.760 ms | — | 14.4x faster; passes the 0.85 ms gate |

The paired kernel is limited by the small problem grid: 256 CTAs give 0.75 waves across 170 SMs.
NCU reports about 12 active warps/SM and the measured 138 TFLOP/s is about 92% of the corresponding
coverage-adjusted compute ceiling. Wider/narrower paired tiles were slower because padding,
predication, or scale traffic exceeded the occupancy benefit.

## Production architecture

### Standalone FC / general Large-T W8G32

- Independent source files: `linear_rowsplit_w8g32_gemm_mma.{cuh,cu}`.
- CTA tile: `BM=64, BN=128, BK=64`; warp tile: `WM=64, WN=16`; eight warps.
- Small-M standalone tile: `BM=32, BN=128, BK=64`; warp tile `32x16`.
- BF16 `mma.sync.m16n8k16` with FP32 accumulators and BF16 output rounding.
- Two BF16 activation stages, one raw W8 code plane, one four-K-tile FP16 scale slab, and one
  BF16 dequantized-weight plane.
- `cp.async.cg` bypasses low-value L1 allocation for activation, code, and vector scale traffic.
- A four-way unrolled K loop and two fragment register sets overlap `ldmatrix` with HMMA.
- A half-warp owns one weight row: each lane loads two adjacent signed codes and stores one
  `__nv_bfloat162`. The two per-element operations remain FP32 multiply followed by BF16_RN.

The final FC instantiation uses 256 threads, 46.08 KB static shared memory, and 119 registers/thread.
There are no local-memory spills and two CTAs reside per SM. The original `<=85` register gate was
specific to the rejected 384-thread producer/consumer design; the correct two-CTA register ceiling
for the production 256-thread architecture is 128.

### Paired MTP K/V

- Independent source files: `linear_rowsplit_w8g32_kv_gemm_mma.{cuh,cu}`.
- CTA tile: `BM=32, BN=128, BK=64`, eight warps.
- K and V keep independent raw-code, scale, BF16-dequant, and FP32 accumulator state.
- The activation tile is copied once and consumed by both projections.
- K and V preserve their original accumulation order; there is no cross-projection reduction.
- The model prefill path now calls `linear_w8g32_kv_pair`; T<=16 and unsupported alignment cases
  execute the two existing small-T linears.

The paired instantiation uses 256 threads, 41.98 KB static shared memory, 104 registers/thread, no
spills, and at most two CTAs/SM.

## Dispatch and coverage

The explicit W8G32 MMA policy is selected for Large-T row-split weights. The fast vector-scale path
requires:

- logical `K % 8 == 0` for aligned 16-byte activation rows;
- `padded_K % 256 == 0` so every scale row is 16-byte aligned.

Shapes that do not satisfy these conditions use the existing correct small-T/general kernel. This
is a correctness route, not a compatibility alias. All production Qwen3.6 shapes satisfy the fast
conditions.

The benchmark now registers `MtpKV1024x5120` and supports `--paired-kv`. The measured sweep covers
`T={17,32,64,96,128,256,512,1024}` for FC, standalone K/V, and paired K/V.

## Numerical behavior

No accuracy-changing optimization was added beyond the design's intentional early BF16 rounding:

```text
w_bf16 = BF16_RN(FP32(int8_code) * FP32(FP16_scale))
out    = BF16_RN(sum_k FP32(w_bf16 * x_bf16))
```

The following performance changes are bit-preserving for that contract:

- cache policy (`cp.async.ca` to `cp.async.cg`);
- loading adjacent codes as `u16` and storing the two independent results as BF16x2;
- caching eight adjacent FP16 scales for four BK=64 iterations;
- K-loop/fragment software pipelining;
- pairing K and V while retaining separate accumulators.

Focused full/tail standalone and paired cases have rel-L2 `0.0017–0.0023` against the existing
high-precision W8 oracle, below the Tensor Core linear tolerance of `0.004`. The full
`qus_linear_test` passes.

## Performance evidence

Final cold-cache microbenchmark, one process/run:

| row | median | useful TFLOP/s | measured TC ceiling | TC % |
|---|---:|---:|---:|---:|
| FC T=512 | 308.54 us | 174.00 | 195.65 | 88.93% |
| FC T=1024 | 603.52 us | 177.91 | 195.65 | 90.93% |
| paired K/V T=512 | 114.66 us | 93.65 | 205.28 | 45.62% |
| paired K/V T=1024 | 155.01 us | 138.54 | 205.28 | 67.49% |

NCU replay time is diagnostic only. Final reports show:

| metric | FC | paired K/V |
|---|---:|---:|
| tensor FP pipeline, active-cycle view | 78.28% | 69.58% |
| registers/thread | 119 | 104 |
| static shared | 46.08 KB | 41.98 KB |
| theoretical / achieved occupancy | 33.33% / 30.72% | 33.33% / 25.15% |
| L2 hit rate | 95.73% | 93.96% |
| local-memory spills | 0 | 0 |

## Full-engine result

CUDA Graph `pp1024+tg256`, MTP K=3, warmup 1, three repetitions:

| KV dtype | prefill before | prefill final | change | decode regression | MTP statistics |
|---|---:|---:|---:|---|---|
| BF16 | 418.29 ms | 402.56 ms | -3.76% | none | exact match |
| INT8 | 424.24 ms | 402.88 ms | -5.03% | none | exact match |

Draft tokens, accepted tokens, accepted-per-position, acceptance rate/length, round count, and
fallback count match the pre-change reports exactly for both KV dtypes.

Matched Nsys captures show exactly one FC W8 MMA launch and one paired K/V launch in prefill:

| KV dtype | FC | paired K/V | Large-T W8 total | prefill NVTX wall |
|---|---:|---:|---:|---:|
| BF16 | 608.33 us | 151.91 us | 760.23 us | 402.62 ms |
| INT8 | 607.72 us | 152.13 us | 759.85 us | 404.60 ms |

## Verification and artifacts

Commands:

```bash
cmake --build build -j --target qus_linear_test qus_linear_op_bench qus_bench \
  qus_engine_mtp_e2e_test
./build/tests/qus_linear_test
./build/tests/qus_engine_mtp_e2e_test batched
QUS_LINEAR_TEST_W8G32_ONLY=1 compute-sanitizer --tool memcheck \
  --error-exitcode=99 ./build/tests/qus_linear_test
```

Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.

Artifacts:

- `profiles/w8g32-tc-gemm-20260710/final/` — FC, standalone K/V, and paired K/V sweeps;
- `profiles/ncu/w8g32-tc-gemm-20260710/final/` — final FC and paired NCU reports;
- `profiles/bench/w8g32-tc-gemm-20260710/` — BF16/INT8 CUDA Graph reports;
- `profiles/nsys/w8g32-tc-gemm-20260710/` — Nsys reports, SQLite exports, summaries, and stats;
- `profiles/w8g32-tc-gemm-20260710/verification/compute_sanitizer_final.txt` — final-source
  sanitizer log.
