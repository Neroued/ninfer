# GQA INT8 prefill kernel redesign — native-S8 QK, producer/consumer PV

- Date: 2026-07-10
- Status: proposed implementation design
- Scope: the prompt-scale GQA path (`T > 6`) for Qwen3.6-27B on one RTX 5090. The KV format,
  public operator API, model schedule, and the existing BF16 prefill algorithm are fixed.

## 0. Decision

The current `gqa_attention_prefill_kernel<Quantized>` must be replaced by two independently
compiled kernels:

- `gqa_attention_prefill_bf16_kernel`: the current optimized BF16 kernel, preserved as its own
  implementation and tuning surface;
- `gqa_attention_prefill_i8_kernel`: a new INT8-native kernel with a different CTA size, shared
  memory layout, register strategy, load pipeline, and Tensor Core instruction mix.

They may share only fixed shape/index constants and leaf PTX helpers. They must not share a kernel
body, KV staging policy, shared-memory arena, warp schedule, or a `KvCodec`/`Quantized` template.
The dtype dispatch remains in the launcher. There is no compatibility alias and no runtime fallback
from the new INT8 kernel to the old templated body.

The first implementation target is:

- native `mma.sync.m16n8k32.s32.s8.s8.s32` for QK;
- Q quantized on-chip per `(query row, 64 dimensions)`;
- K kept INT8 from cache to Tensor Core, with no K dequantization;
- V codes/scales loaded asynchronously and dequantized once into a BF16 shared tile;
- BF16 Tensor Core PV;
- `Br=64`, `Bc=64`, 16 warps/CTA;
- four QK/softmax producer warps, one per 16-row query tile;
- all 16 warps acting as PV consumers, four output-dimension slices per row tile;
- at most 120 registers/thread, at most 92 KiB shared memory, one 16-warp CTA/SM, and no spills.

This is the right first target for maximum Tensor Core duty without introducing a second, unproven
quantization of the softmax probabilities. A full-S8 PV extension is defined in §11, but it is not a
runtime alternative and is not part of the first landing.

## 1. Why the current architecture must be replaced

### 1.1 Measured T=1024 context sweep

Command, run once for each KV dtype:

```bash
build/bench/qus_gqa_attention_bench \
  --append-prompt-baseline --tokens 1024 \
  --context 0,8192,32768,65536,131072,262144 \
  --kv-dtype <bf16|int8> --warmup 5 --repeat 30
```

Each value is the per-launch median of the full GQA prompt operation (KV fill plus attention):

| Existing context | BF16 | INT8 | INT8 / BF16 latency | BF16 useful TFLOP/s | INT8 useful TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1178 ms | 0.2883 ms | 2.448x | 109.51 | 44.74 |
| 8,192 | 1.4122 ms | 3.6996 ms | 2.620x | 155.12 | 59.21 |
| 32,768 | 5.2976 ms | 19.9320 ms | 3.762x | 158.10 | 42.02 |
| 65,536 | 10.4680 ms | 46.1833 ms | 4.412x | 158.79 | 35.99 |
| 131,072 | 21.5818 ms | 101.9371 ms | 4.723x | 153.44 | 32.49 |
| 262,144 | 42.5332 ms | 203.7662 ms | 4.791x | 155.41 | 32.44 |

A reverse-order check reproduced every BF16 median within 0.40% and every INT8 median within 1.31%.
The regression is structural, not a clock-order artifact.

INT8 reduces the benchmark's modeled tile KV bytes by about 48.4% (at 262K, 98,508 MiB to
50,793 MiB), yet becomes 4.79x slower. The current path is therefore not near a memory-bandwidth
roofline; it is unable to turn compressed storage into useful execution throughput.

### 1.2 Nsight Compute baseline

The representative profiler point is `T=1024, context=32768`, attention-only, with application
replay and one matching launch:

```bash
ncu --force-overwrite --target-processes all --replay-mode application \
  --section SpeedOfLight --section Occupancy \
  --section SchedulerStats --section WarpStateStats --section InstructionStats \
  --kernel-name regex:'gqa_attention_prefill_kernel' \
  --launch-skip 0 --launch-count 1 \
  -o /tmp/gqa_prefill_<dtype>_t1024_c32768_sched \
  build/bench/qus_gqa_attention_bench \
    --append-prompt-attention-only --tokens 1024 --context 32768 \
    --kv-dtype <dtype> --warmup 0 --repeat 1
```

NCU matched `gqa_attention_prefill_kernel<1>` for INT8 and
`gqa_attention_prefill_kernel<0>` for BF16. Replay duration is profiler-inflated and is used only
for the within-capture comparison; the standalone sweep above owns performance acceptance.

| Metric | BF16 | current INT8 | Consequence |
|---|---:|---:|---|
| Registers/thread | 249 | 250 | INT8 gets no register relief |
| Dynamic shared memory/block | 98.30 KiB | 98.30 KiB | INT8 gets no storage/occupancy benefit |
| Active warps/SM | 4.00 | 4.00 | one warp per scheduler |
| Achieved occupancy | 8.33% | 8.33% | latency is fully exposed |
| Compute (SM) throughput | 62.61% | 17.40% | INT8 dilutes the compute issue stream |
| Memory throughput | 30.79% | 6.84% | INT8 is not memory-bound |
| DRAM throughput | 1.38% | 0.20% | compressed bytes are not the limiter |
| Issued instructions | 0.884B | 2.020B | INT8 executes 2.28x as many instructions |
| Issued warp/scheduler/cycle | 0.11 | 0.07 | most issue slots are empty |
| Cycles/issued instruction | 8.92 | 15.09 | INT8 dependency latency is much worse |
| No eligible warp | 88.80% | 93.36% | no latency-hiding reserve |
| Leading stall | math-pipe throttle, 4.7 cycles | long scoreboard, 9.0 cycles | INT8 waits on synchronous cache loads |

The INT8 leading stall accounts for 59.5% of the average 15.1 cycles between issued instructions.
Static resource inspection agrees with NCU: the compiled INT8 and BF16 kernels use 250 and 249
registers/thread respectively and both opt into the same 96 KiB Q+K+V tile.

### 1.3 Source-level cause

The current template preserves the BF16 dataflow and inserts a codec operation into it:

1. BF16 stages each 16-byte K/V vector with `cp.async` and overlaps the transfer with Tensor Core
   work.
2. INT8 calls `gqa_prefill_stage_kv_i8`, which synchronously loads an FP16 scale and eight INT8
   codes, converts `int8 -> float -> bf16x2`, then writes a 16-byte BF16 vector to shared memory.
3. This is done for both K and V and for every `(q_head, q_block, key tile)`.
4. The surrounding `async_copy_commit/wait` cannot make this conversion asynchronous; the helper
   has completed before the kernel reaches the Tensor Core work.
5. After paying that cost, INT8 executes exactly the same BF16 QK and BF16 PV MMA loops as BF16.

The design has all of INT8's conversion cost, none of its native Tensor Core benefit, and none of its
shared-memory or occupancy benefit. Scheduling tweaks inside the template cannot fix those three
properties.

## 2. Goals and non-goals

### 2.1 Goals

1. Make INT8 prefill at least as fast as BF16 over the complete T=1024 context sweep, and materially
   faster at long context.
2. Keep the Tensor Core pipe as the dominant utilized pipeline at long context, with at least 85%
   tensor-pipe active utilization at the representative NCU point.
3. Raise residency from four to sixteen active warps/SM without local-memory spills.
4. Preserve the existing per-token, per-64-dimension INT8 KV cache codes and FP16 scales bit-for-bit.
5. Preserve causal/bottom-right alignment, append semantics, output dtype, and public API.
6. Leave BF16 kernel behavior and performance unaffected by INT8-specific tuning.

### 2.2 Non-goals

- No generic attention runtime, arbitrary head dimension, arbitrary head counts, or model family.
- No KV cache relayout and no second persistent cache representation.
- No change to `DType::I8`, `KVCache`, group size 64, or the cache scale dtype.
- No CUTLASS dependency.
- No runtime BF16 fallback for INT8 KV.
- No simultaneous old/new INT8 implementations after selection.
- No TMA requirement in the first landing. TMA remains a measured tuning option, not an architectural
  dependency (§10.4).
- No full-INT8 PV in the first landing (§11).

## 3. Hard code-organization boundary

The intended file layout is:

| File | Ownership |
|---|---|
| `gqa_attention_prefill_common.cuh` | fixed shape/index constants, swizzles, `ldmatrix`, BF16/S8 MMA leaf helpers, packing and `exp2` helpers only |
| `gqa_attention_prefill_bf16.cuh` | current BF16 fill and attention kernel bodies, BF16 shared-memory layout, BF16 pipeline |
| `gqa_attention_prefill_i8.cuh` | INT8 fill, Q quantization, INT8 arena, native-S8 QK, producer/consumer schedule, BF16 PV |
| `gqa_attention_prefill.cu` | dtype dispatch and two explicit launch functions |

`gqa_attention_prefill.cuh` is removed rather than kept as an alias. The launcher includes the three
new headers directly.

The common header must not contain:

- a `template<bool Quantized>` kernel;
- a `KvCodec`, loader policy, or format-dependent stage loop;
- a shared-memory arena definition;
- a warp-role mapping;
- a combined BF16/INT8 launch helper.

The dispatch is deliberately explicit:

```text
gqa_attention_prompt_launch
  kv.dtype == BF16
    -> gqa_attention_prefill_fill_bf16_kernel
    -> gqa_attention_prefill_bf16_kernel
  kv.dtype == I8
    -> gqa_attention_prefill_fill_i8_kernel
    -> gqa_attention_prefill_i8_kernel
```

`gqa_attention_prompt_attention_launch`, used by the attention-only benchmark, performs the same
dtype dispatch but skips the fill kernel. Public wrapper validation remains unchanged.

This separation is not cosmetic. It allows INT8 to change block size, launch bounds, smem opt-in,
swizzle, Bc, stage count, and instruction selection without adding an `if constexpr` or resource
side effect to BF16.

## 4. INT8 fill kernel

The prompt operation must first quantize the new K/V chunk into the cache because attention reads
all history, including the newly appended tokens, from one uniform cache representation. This stays
a separate kernel; fusing it into the attention CTA would require an invalid grid-wide ordering.

The current fill uses one 64-thread CTA per `(token, kv_head, group)` and shared-memory reductions.
The redesigned fill copies the proven decode quantizer organization:

- one warp owns one `(token, kv_head, 64-d group)`;
- lane `l` reads dimensions `l` and `l+32` for K and V;
- warp shuffles reduce the two absmax values;
- lane 0 writes the FP16 K/V scales;
- all lanes write two K codes and two V codes;
- a 256-thread CTA handles eight independent groups; no shared memory and no block-wide barrier.

This preserves code/scale bits exactly while reducing launch blocks and removing shared-memory
reduction overhead. It matters mainly at context 0, where the attention work is smallest. It is not
allowed to change the cache quantization formula to improve kernel convenience.

## 5. Chosen INT8 attention math

### 5.1 On-chip Q quantization

Q remains BF16 at the operator boundary. Each query row is quantized in shared memory using four
independent 64-dimension groups:

```text
qs[row,g]  = max(abs(Q[row, 64g:64g+64])) / 127
Qi8[row,d] = clamp(round(Q[row,d] / qs[row,g(d)]), -127, 127)
```

`qs` is FP32, matching the current native-S8 decode path. It is an ephemeral kernel value and does
not change the cache format. All 16 warps cooperate over the 256 `(row, group)` quantization units;
each warp reduction consumes two values/lane. Q is quantized once per CTA before the key loop and
remains in a swizzled INT8 shared tile.

### 5.2 Native-S8 QK

K codes and K scales are loaded from cache to shared memory with `cp.async`. K is never converted to
BF16. For each 64-dimension group:

```text
score[row,key] += qs[row,g] * ks[key,g]
                  * int32_dot(Qi8[row,g], Ki8[key,g])
```

One group uses two `m16n8k32.s32.s8.s8.s32` operations. With `Bc=64`, a producer warp executes:

```text
4 groups * 2 k32 steps * 8 key n-tiles = 64 S8 MMA instructions / key tile
```

The BF16 kernel executes 16 k16 steps times 8 n-tiles = 128 BF16 MMA instructions for the same QK
work. Native S8 therefore halves issued QK MMA instructions and deletes K dequantization entirely.
The int32 accumulator is exact; each group's result is converted to FP32 and accumulated with the
two scales.

Q fragments are loaded one group at a time. They are not all kept live in registers across the key
loop.

### 5.3 Online softmax

Each of the four producer warps owns 16 query rows for the lifetime of the CTA. It performs the same
stable online softmax contract as BF16:

- FP32 running row max `m` and row sum `l`;
- bottom-right causal mask;
- full-tile fast path and one diagonal/tail path;
- scale folded into `exp2` using `scale * log2(e)`;
- FFMA-form exponent arguments;
- deferred normalization in the epilogue.

The producer writes the current unnormalized probability tile as BF16 into a swizzled `P[Br,Bc]`
shared buffer and writes the per-row output rescale `alpha` to shared memory. There is no inter-warp
softmax reduction because each 16-row tile has exactly one producer warp.

### 5.4 BF16 PV

V is quantized per `(key, 64-d output group)`. Its scale varies along the contracted key dimension,
so it cannot be factored out of a plain INT8 PV dot product. The first landing therefore:

1. asynchronously loads V codes and V scales;
2. dequantizes V once from shared INT8 to a swizzled BF16 shared tile;
3. executes `P_bf16 * V_bf16` with `m16n8k16.bf16` Tensor Core instructions.

Only V is dequantized. The work is distributed across the 12 non-producer warps while the four
producer warps execute native-S8 QK and softmax. The dequantization is no longer a serialized stage
on every compute warp.

All 16 warps then consume P/V. Four warps are assigned to each 16-row tile; each owns one contiguous
64-dimension output slice. A consumer therefore carries:

```text
16 rows * 64 output dims / 32 lanes = 32 FP32 accumulator registers/thread
```

The current kernel carries 128 FP32 output accumulator registers/thread. This four-way split is the
main register and occupancy unlock.

## 6. CTA geometry and warp roles

### 6.1 Grid

Keep one CTA per `(q_head, q_block)`:

```text
grid.x = ceil(T / 64)
grid.y = 24 query heads
block  = 512 threads = 16 warps
kv_head = q_head / 6
```

The mapping preserves independent FA2-style query-row ownership and avoids partial output/reducer
workspace. At T=1024 the grid is 384 CTAs, or 2.26 one-CTA-per-SM waves on the 170-SM RTX 5090.

Fusing all six GQA query heads into one CTA is rejected. It would improve K/V reuse but would need
24 producer warps before PV splitting, exceed practical register capacity, and make causal/tail load
balance worse. The existing 96 MB L2 already absorbs most repeated K/V reads; the measured failure
is instruction latency, not DRAM traffic.

### 6.2 Warp map

| Warp IDs | QK/softmax role | PV role |
|---|---|---|
| 0..3 | producer for row tiles 0..3 | output slice 0 (dims 0..63) of row tiles 0..3 |
| 4..7 | V-dequant workers | output slice 1 (dims 64..127) of row tiles 0..3 |
| 8..11 | V-dequant workers | output slice 2 (dims 128..191) of row tiles 0..3 |
| 12..15 | V-dequant workers | output slice 3 (dims 192..255) of row tiles 0..3 |

The PV mapping is:

```text
row_tile = warp_id % 4
d_slice  = warp_id / 4
```

Producer warps are not permanently excluded from PV; they own slice 0. During the QK phase, the 12
other warps perform useful V conversion instead of waiting.

### 6.3 Register target

Compile with `__launch_bounds__(512, 1)` and treat the following as hard gates:

- ptxas register count at most 120/thread (128 is the architectural edge for 512 threads);
- local load/store bytes zero;
- shared spill bytes zero.

The expected dominant live state per consumer is 32 FP32 PV accumulators plus fragments. Producer
warps additionally hold one 64-key score tile and one Q group at a time, but not a full 256-d Q
fragment set. If producers exceed the cap, reduce score fragment liveness before changing block
geometry.

## 7. Shared-memory layout

Starting `Br=64`, `Bc=64` budget:

| Region | Shape / dtype | Bytes | Lifetime |
|---|---|---:|---|
| Q codes | `[64,256]` INT8 swizzled | 16,384 | full kernel |
| Q scales | `[64,4]` FP32 | 1,024 | full kernel |
| K codes | `[64,256]` INT8 swizzled | 16,384 | current/next tile |
| V codes | `[64,256]` INT8 row-major | 16,384 | current/next tile |
| V dequant | `[64,256]` BF16 swizzled | 32,768 | current PV |
| P | `[64,64]` BF16 swizzled | 8,192 | current PV |
| K/V scales | `2 * [64,4]` FP16 | 1,024 | current/next tile |
| alpha/final-l | `2 * [64]` FP32 | 512 | current tile / epilogue |
| **Total** | | **92,672 B = 90.5 KiB** | |

Alignment and barrier metadata must keep the final opt-in allocation at or below 92 KiB. This fits
under the current 98,304-byte dynamic-smem limit while leaving enough margin for alignment.

The K/V code arena is single-buffered but pipelined by lifetime aliasing:

- after QK has consumed K codes and V conversion has consumed V codes, both 16 KiB code regions are
  dead;
- the next tile's K/V codes and scales are prefetched into those regions while current PV reads only
  the separate BF16 V tile;
- the next V conversion overwrites the BF16 V tile only after current PV completes.

This achieves tile-to-tile overlap without allocating a second 32 KiB code arena.

## 8. Key-tile pipeline

Initialization:

1. all warps quantize Q and write Q scales;
2. issue `cp.async` for K/V codes and scales of key tile 0;
3. wait and synchronize once.

Steady-state key tile:

```text
producer warps 0..3                  worker warps 4..15
-------------------                  ------------------
load one Q group at a time           read V codes/scales from shared
S8 QK + per-group rescale            dequant INT8 V -> BF16 V tile
causal mask + online softmax
write BF16 P + alpha
                    \                /
                     CTA barrier #1
                     code buffers are now free
                     issue cp.async for next K/V codes + scales
all 16 warps: rescale output accumulator by alpha
all 16 warps: BF16 Tensor Core PV for one 64-d output slice
wait next cp.async; CTA barrier #2
```

The target is two CTA-wide barriers per key tile, down from a serialized load/convert/compute chain.
Named barriers may replace CTA barriers only after the simple schedule is correct and NCU identifies
barrier stalls as a top limiter.

The code/scales copy uses a fixed 16-byte page mapping. Each of 512 threads issues about four code
copies for a complete K+V tile. K destination addresses use the same packed-INT8-as-b16 swizzle as
the native-S8 decode kernel; V codes stay row-major for vector dequant, and the BF16 destination uses
the PV `ldmatrix.trans` swizzle. Tail vectors are explicitly zeroed.

## 9. MMA and instruction accounting

For one producer warp and one `Bc=64` key tile:

| Work | Current BF16-style INT8 | New INT8 |
|---|---:|---:|
| QK MMA instructions | 128 BF16 | 64 S8 |
| PV MMA instructions for 16 rows x 256 dims | 128 BF16 | 128 BF16, split over 4 warps |
| K dequant vectors | 2,048/CTA tile | 0 |
| V dequant vectors | 2,048/CTA tile on compute warps | 2,048/CTA tile on 12 overlapping workers |

Ignoring softmax and conversion, the Tensor Core instruction count for the mathematical attention
work drops from 256 to 192 per row warp, a 25% reduction. If issue throughput were otherwise equal,
that gives a 1.33x compute-side speedup. The real design has additional headroom because it removes K
conversion, lowers dependencies, and provides four active warps/scheduler instead of one.

The existing benchmark's `tflops_pct` divides all useful QK+PV FLOPs by the BF16/FP32-accumulate
peak. That remains a useful cross-kernel throughput number, but it is not a valid hardware
utilization percentage for a mixed S8-QK/BF16-PV kernel. Actual Tensor Core utilization must come
from NCU's tensor-pipe metrics (§13).

## 10. Deliberately rejected alternatives

### 10.1 Keep BF16 MMA and only improve dequant staging

This could recover some of the 4.8x regression, but K conversion would remain pure overhead and QK
would still execute 128 BF16 MMA instructions/tile. It does not use the cache's native representation
and has a lower ceiling. Rejected.

### 10.2 Template a codec into the BF16 kernel

This is the current failure mode. Shared launch bounds and smem layout force INT8 to inherit BF16's
one-warp-per-scheduler schedule. It also makes every INT8 change a BF16 regression risk. Rejected.

### 10.3 Fuse multiple query heads per CTA

Sharing K/V across two or six GQA heads looks attractive, but output accumulators and query/score
state scale with query heads while the register file does not. The 16-warp single-head design already
fills four warps/scheduler and reads mostly from L2, so head fusion attacks a non-leading cost.
Rejected for the first implementation.

### 10.4 Require TMA immediately

TMA could reduce copy/address instructions and provide hardware swizzling, but the project currently
has no TMA helper layer, and the new 512-thread CTA needs only about four 16-byte K/V copy issues per
thread/tile. Start with the proven conflict-free `cp.async` page mapping. Promote the INT8 copy to a
TMA/mbarrier ring only if NCU shows one of the following after the producer/consumer kernel lands:

- long-scoreboard remains a top-two stall;
- K/V issue/address instructions exceed 10% of issued instructions;
- tensor-pipe active is below 85% while registers, barriers, and V conversion are not limiting.

TMA is an INT8-file-only change and must not alter the BF16 pipeline.

### 10.5 Use a split-KV reducer for T=1024

Long context supplies abundant work inside every CTA and the grid already has 384 blocks. A split
reducer would materialize partial `[T,H,D]` outputs and stats, adding large workspace traffic and a
second kernel. Decode needs split-KV because T is tiny; prefill does not. Rejected.

## 11. Deferred full-S8 PV extension

V's scale varies by key and output group:

```text
V[row_key,d] ~= vs[row_key,g(d)] * Vi8[row_key,d]
```

For output group `g`, define:

```text
Pg[q,key,g] = P[q,key] * vs[key,g]
```

Then quantize `Pg` per `(query row, key tile, output group)` and compute:

```text
O[q,d in g] += pgs[q,g] * int32_dot(Pgi8[q,:,g], Vi8[:,d])
```

This makes PV native S8 and removes the 32 KiB BF16 V tile. It is feasible for prefill because each
of the four PV warps assigned to a row tile already owns exactly one 64-d V group. However, it adds
four probability-scale reductions and a second INT8 round-trip to every key tile. That may reduce
rather than increase tensor duty, and it changes attention numerics beyond the cache round-trip.

It is considered only if the first kernel is correct and one of these measured conditions holds:

- V conversion is at least 15% of issued instructions or is the leading stall;
- BF16 PV tensor issue is not the dominant pipe despite at least 16 active warps/SM;
- the first kernel cannot reach 190 useful TFLOP/s at contexts 32K and above.

Before implementation, a CPU oracle and model-level accuracy experiment must quantify the added P
quantization error. If selected, full-S8 PV replaces the mixed kernel; it is not kept behind a runtime
flag.

## 12. Numerical contract

The cache append contract is unchanged and remains bit-exact:

- FP16-rounded `absmax/127` scale per token/head/64-d group;
- round-to-nearest-even codes clamped to `[-127,127]`;
- identical K/V code and scale arrays before and after the redesign.

Native-S8 QK intentionally changes the attention arithmetic to match decode:

1. Q is quantized per row/group with an FP32 scale.
2. K participates as exact `int8_code * stored_fp16_scale` in group-wise int32 dot products; it is
   not rounded to BF16 after dequantization.
3. V is `int8_code * stored_fp16_scale`, rounded to BF16 in shared memory before BF16 PV.
4. Newly appended K/V follows the same cache round-trip as history.

The INT8 prefill CPU oracle must therefore use:

```text
Q_ref = per-row/group INT8 Q round-trip in FP32
K_ref = code * FP16 scale in FP32
V_ref = bf16(code * FP16 scale)
```

and then perform fp64 causal attention. This is the same arithmetic contract already used by the
native-S8 decode oracle, extended from T<=6 to prompt T.

There is no silent numerical fallback. If model accuracy fails, the native-S8 prefill design does
not ship until its quantization contract is revised and revalidated.

## 13. Verification and acceptance

### 13.1 Correctness

Attention numerical tests are allowed by the repository testing policy because they protect a real
CUDA numerical contract with a mathematical oracle.

Required cases:

- INT8 prefill `T=65` and `T=192`, preserving the existing non-tile-multiple cases;
- at least one append case with nonzero existing context and more than one key tile;
- causal diagonal and future-token isolation;
- zero-scale groups and large-magnitude softmax stress;
- bit-exact K/V codes and FP16 scales;
- output comparison to the native-S8-Q/BF16-V fp64 oracle with
  `Tolerance::attention_bf16()`;
- prefill -> decode consistency under INT8 KV.

Do not add source-shape tests. Register, SASS, and resource constraints are build/profile review
checks, not unit tests.

Commands:

```bash
cmake --build build --target qus_gqa_attention_test qus_gqa_attention_bench -j
ctest --test-dir build -R gqa_attention --output-on-failure

compute-sanitizer --tool memcheck build/tests/qus_gqa_attention_test
compute-sanitizer --tool racecheck build/tests/qus_gqa_attention_test
compute-sanitizer --tool initcheck build/tests/qus_gqa_attention_test
```

### 13.2 Standalone performance gate

Use the same 5-warmup/30-repeat sweep that exposed the regression, for both full prompt and
attention-only modes. The checked-in BF16 kernel is the comparison baseline; do not substitute old
historical numbers if its measured median moves.

Hard gate, based on the current BF16 medians:

| Existing context | Current BF16 | INT8 hard maximum | INT8 target |
|---:|---:|---:|---:|
| 0 | 0.1178 ms | 0.130 ms | <= 0.120 ms |
| 8,192 | 1.4122 ms | 1.412 ms | <= 1.25 ms |
| 32,768 | 5.2976 ms | 4.768 ms (0.90x BF16) | <= 4.238 ms (0.80x) |
| 65,536 | 10.4680 ms | 9.421 ms | <= 8.374 ms |
| 131,072 | 21.5818 ms | 19.424 ms | <= 17.265 ms |
| 262,144 | 42.5332 ms | 38.280 ms | <= 34.027 ms |

The long-context target corresponds to roughly 190-198 useful TFLOP/s with the benchmark's
mathematical FLOP definition. Context 0 has a looser hard gate because INT8 must quantize the new
cache contents before attention.

### 13.3 NCU gate

Capture at `T=1024, context=32768` and one long point after the standalone benchmark is green.
The intended kernel name must be unambiguous: `gqa_attention_prefill_i8_kernel`.

Required final metrics:

- registers/thread <= 120;
- dynamic shared memory <= 92 KiB;
- local and shared spilling = 0;
- achieved active warps/SM >= 15.5 (target geometry: 16);
- tensor-pipe active >= 85% at long context and is the highest-utilized compute pipeline;
- issued instructions at C=32768 <= 1.5B (at least 25% below the current 2.020B; V conversion
  remains in the first landing);
- cycles/issued instruction <= 6;
- no-eligible cycles <= 65%;
- long-scoreboard is not more than 25% of cycles between issued instructions;
- no shared-bank-conflict regression on the K/P/V `ldmatrix` paths.

These are architecture gates, not micro-optimizing suggestions. If the kernel still has four active
warps or a long-scoreboard majority, it has reproduced the old failure in a new file.

### 13.4 End-to-end and accuracy gate

After the operator gate:

- run `qus_bench` BF16/INT8 KV prefill at 1024 and representative longer prompts;
- confirm the operator gain survives all attention layers and does not regress full-model prefill;
- run the existing greedy/judge BF16-KV vs INT8-KV comparison;
- explicitly compare the new INT8 prefill outputs against the already native-S8 decode behavior at
  the prefill/decode boundary.

## 14. Benchmark/report adjustments

Keep `useful_flops` and `tflops` unchanged so BF16 and INT8 wall-time results remain directly
comparable. Clarify the report semantics:

- `tflops_pct` is BF16-peak-normalized useful throughput, not mixed-kernel hardware utilization;
- record `math_mode=bf16_qk_bf16_pv` or `s8_qk_bf16_pv` in JSON/CSV;
- record `qk_mma_dtype`, `pv_mma_dtype`, and the modeled QK/PV MMA counts;
- use NCU tensor-pipe metrics, not `tflops_pct`, for the >=85% utilization gate;
- keep full-prompt and attention-only results together so fill overhead is visible.

These fields are user-visible report schema and require the existing benchmark schema test to be
updated only for the real downstream contract.

## 15. Implementation sequence

1. **Split without changing BF16.** Move common leaf helpers, create explicit BF16 names, delete the
   template and old header, build/test, compare BF16 SASS resources and the six-point sweep.
2. **Replace the INT8 fill.** Port the warp quantizer organization from decode; prove bit-exact cache
   codes/scales and measure context-0 fill cost.
3. **Bring up Q quantization and native-S8 QK.** Reuse the proven decode S8 fragment contract and
   update the prefill oracle before optimizing the pipeline.
4. **Land the 16-warp producer/consumer kernel.** Add the shared arena, V conversion workers, P/alpha
   handoff, four-way PV output split, and vectorized epilogue.
5. **Pipeline next-tile copies.** Alias dead K/V code buffers during current PV, reduce to two
   barriers/key tile, then run sanitizers.
6. **Tune only from evidence.** Compare Bc=64 against Bc=32 only if tails/barriers dominate; consider
   TMA only under §10.4; preserve one selected production policy per measured regime.
7. **Run operator, NCU, end-to-end, and accuracy gates.** Remove temporary variants and document the
   final measured configuration.

The CUDA kernel body, launcher resources, oracle, and benchmark are tightly coupled and should land
in this order. Independent review is required before completion: one pass for index/lifetime and
barrier correctness, and one pass for numerical semantics/performance evidence.

## 16. Risks and controls

| Risk | Control |
|---|---|
| 512-thread CTA exceeds register budget | 32-register PV slice; group-at-a-time Q fragments; <=120-reg hard gate |
| V dequant does not fit under S8 QK | 12 dedicated worker warps; measure issued mix; full-S8 PV only under §11 gate |
| next-tile prefetch overwrites live codes | explicit lifetime phases; two CTA barriers; memcheck/racecheck/initcheck |
| INT8 swizzle feeds wrong S8 fragment bytes | reuse decode's proven packed-int8-as-b16 fragment contract; numerical oracle |
| Q quantization changes model behavior | native-S8 prefill oracle plus model accuracy gate; no silent fallback |
| context-0 loses to fill overhead | warp-only fill kernel and separate attention-only/full-prompt reporting |
| BF16 regresses during split | BF16 SASS/resource comparison and six-point A/B before INT8 work continues |
| mixed-S8/BF16 roofline is misreported | separate useful throughput from NCU tensor-pipe utilization |
| tuning variants become permanent complexity | measured dispatcher thresholds only; delete losing implementations |

## 17. Definition of done

The redesign is complete only when all of the following are true:

1. BF16 and INT8 prefill are separate named kernels and separate source bodies.
2. BF16 numerical tests, resource usage, and benchmark medians remain within measurement noise.
3. INT8 cache codes/scales remain bit-exact.
4. INT8 prompt output passes the native-S8-Q/BF16-V fp64 oracle and model accuracy gate.
5. All three compute-sanitizer tools are clean.
6. INT8 passes every hard latency gate in §13.2.
7. The long-context kernel has at least 15.5 active warps/SM, no spills, and at least 85% tensor-pipe
   active utilization.
8. NCU shows the tensor pipe—not synchronous dequant loads—as the binding execution resource.
9. Temporary kernels, compatibility branches, and losing tuning variants are deleted.
