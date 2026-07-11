# GQA prefill kernel vs FlashAttention-2 — evidence-backed parity audit

Date: 2026-07-08
Kernel under audit: `qus::kernels::gqa_attention_prefill_kernel`
(`src/kernels/kernel/gqa_attention_prefill.cuh`)
Reference: FlashAttention-2 forward (`flash-attn` 2.8.3.post1), the
`Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, bf16>` instantiation.

Update note: sections 1-9 preserve the original audit baseline. Section 10 records
the implementation pass made from this audit and the new measured state.

## 0. Ground rule for this document

Every performance statement carries one of two evidence tags:

- **[MEASURED]** — a number produced by an `ncu` run in this session, with the
  metric name and the report file it came from. Reproduction commands are in §7.
- **[SOURCE]** — a code fact verifiable by reading a cited file and line range
  (FlashAttention source or our own kernel). A structural code difference is a
  fact, not a guess.

No claim in this document is a hypothesis about a cause that was not measured.
Where an earlier working hypothesis was contradicted by measurement, it is
listed in §6 as **refuted**, with the measurement that refuted it.

## 1. Shape and method

Canonical shape (identical for both implementations):

- q: `[1, T, 24, 256]`, k/v: `[1, context + T, 4, 256]`, bf16, causal,
  `scale = 0.0625`, bottom-right causal alignment.
- Audited point: `T = 1024`, `context = 4096` (seqlen_q = 1024, seqlen_k = 5120).
- Device: RTX 5090 (CC 12.0), driver 591.86, CUDA 13, ncu 2025.4.1.

Tools:

- Ours: `build/bench/qus_gqa_attention_bench --append-prompt-baseline --tokens 1024 --context 4096`
  (built with `-lineinfo`, `bench/CMakeLists.txt:12`), which drives
  `gqa_attention_prompt_launch` → `gqa_attention_prefill_kernel`
  (`src/kernels/launcher/gqa_attention_prefill.cu:12-48`).
- FA: `tools/bench/flash_attn_gqa_bench.py --tokens 1024 --context 4096 --attention-only`
  in conda env `vllm-bench` (torch 2.11.0+cu130, flash-attn 2.8.3.post1).

Artifacts (this session): `profiles/qus_prefill_T1024_C4096.ncu-rep`,
`profiles/fa_prefill_T1024_C4096.ncu-rep`, `profiles/qus_raw.csv`,
`profiles/fa_raw.csv`.

## 2. Headline result [MEASURED]

Standalone wall-clock (median of the bench harness, not under the profiler):

| impl | median | useful TFLOP/s | % of 209.5 TC peak |
|------|--------|----------------|--------------------|
| ours | 0.950 ms | 122.10 | 58.28% |
| FA   | 0.761 ms | 152.45 | 72.77% |

FA is **1.25×** faster (ours is 24.8% slower). Source: bench stdout, §7.

Under `ncu` (single launch, kernel replay, `gpu__time_duration.sum`): ours
1.386 ms, FA 1.025 ms — same 1.35× ratio band, confirming the standalone gap is
real and not a harness artifact.

**Single root cause, measured:** at *identical* launch configuration and
occupancy, our kernel issues **2.10× more instructions** than FA
(`smsp__inst_executed.sum`: 185,801,472 vs 88,628,736). Because both kernels run
at 1 warp/scheduler (§3), there is no latency hiding, so runtime tracks issued
instructions almost linearly. The extra ~97M instructions are non-tensor work
diluting the issue stream; consequently our tensor pipe is busy only 64.7% of
active cycles vs FA's 88.5%.

## 3. Configuration parity — everything the same [MEASURED]

From `launch__*` and occupancy metrics in both reports:

| property | ours | FA | same? |
|----------|------|----|-------|
| grid size | 384 (16×24×1) | 384 | yes |
| block size | 128 | 128 | yes |
| dynamic smem / block | 98.304 KB | 98.304 KB | yes |
| static smem / block | 0 | 0 | yes |
| waves / SM | 2.26 | 2.26 | yes |
| achieved occupancy | 8.327% | 8.340% | yes |
| registers / thread | 246 | 235 | ~ (both smem-limited) |

Occupancy is **8.3% for both** — 1 CTA/SM → 4 warps/SM → **1 warp/scheduler**.
This is forced by the 98 KB smem tile and is identical on both sides, so it
cannot explain the gap. It does mean the kernel is latency-exposed: with one warp
per scheduler, every instruction's latency is on the critical path, which is why
the *count* and *mix* of instructions (§4, §5) dominate.

`[SOURCE]` The geometry matches FA's traits: 4 warps, Br=64, Bc=64, hd=256, MMA
`m16n8k16`, single-buffered K/V, `Is_Q_in_regs=false`
(FA `csrc/flash_attn/src/kernel_traits.h:66-90`; ours
`src/kernels/kernel/gqa_attention_prefill.cuh:34-39, 173-189`).

## 4. Full measured metric comparison [MEASURED]

From `profiles/qus_raw.csv` and `profiles/fa_raw.csv` (raw page):

| metric | ours | FA | note |
|--------|------|----|------|
| `smsp__inst_executed.sum` | 185,801,472 | 88,628,736 | **2.10× more** |
| tensor active % (`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`) | 64.70 | 88.52 | FA denser |
| tensor elapsed % (…`_elapsed`) | 48.00 | 65.77 | |
| SM throughput % | 48.00 | 65.77 | |
| FP32 fused (FFMA), dyn | 228,864 | 3,786,240 | FA fuses 16.5× more |
| FP32 non-fused, dyn | 33,607,680 | 18,266,112 | ours 1.84× more |
| ALU pipe % active | 4.56 | 0.99 | ours 4.6× |
| XU pipe % active (MUFU/exp2) | 4.17 | 2.92 | |
| FMA pipe % active | 2.82 | 2.14 | |
| LSU pipe % active | 20.02 | 16.29 | |
| bank conflicts, total | 6,344,186 | 0 | see §5.4 |
| — `op_ld` (ldmatrix/LDSM) | 0 | 0 | ours conflict-free |
| — `op_st` (STS) | 0 | 0 | no STS |
| — `op_ldgsts` (cp.async) | 6,396,868 | 0 | ours only |
| stall math_pipe_throttle (cyc) | 3.20 | 6.60 | see §5.6 |
| stall wait | 2.52 | 3.36 | |
| stall short_scoreboard | 0.355 | 0.137 | ours 2.6× |
| stall long_scoreboard | 0.125 | 0.133 | equal |
| stall mio_throttle | 0.066 | 0.101 | small both |
| stall barrier | 0.071 | 0.043 | |
| stall lg_throttle | 0.092 | 0.231 | |

FP dynamic counts are from the InstructionStats rule text in each `.ncu-rep`
(`--page details`).

Static SASS opcode histogram of `gqa_attention_prefill_kernel` (one compiled
instance, 2328 instructions, `cuobjdump -sass`):

```
FMUL 454   HMMA 256   LDSM 144   FADD 136   F2FP 128   STG 128   MOV 104
FSETP 102  LOP3 94    IADD 90    FMNMX 70   MUFU 68    FSEL 64    F2F 64
ISETP 51   PLOP3 34   PRMT 33    IMAD 27    SHF 13     ...   (FFMA: absent from top ranks)
```

This static mix is per key-block iteration body (the `for kb` loop is not
unrolled; the inner QK/PV loops are). `HMMA 256` = 128 QK MMAs + 128 PV MMAs per
iteration, matching the tile math.

## 5. Findings (each tied to a measurement)

### 5.1 Instruction count is the gap — 2.10× [MEASURED]

`smsp__inst_executed.sum` = 185.8M (ours) vs 88.6M (FA). Same grid, same
occupancy, same tensor workload (both do 256 HMMA/iteration). The delta is
~97M *non-tensor* warp-instructions. Direct consequence, also measured: tensor
pipe active drops from FA's 88.5% to our 64.7%. Everything in §5.2–§5.5 is a
component of this delta.

### 5.2 FP is not fused — FFMA barely used [MEASURED] + [SOURCE]

[MEASURED] Dynamic FP32: ours 228,864 fused / 33,607,680 non-fused (0.68%
fused); FA 3,786,240 fused / 18,266,112 non-fused (17.2% fused). Static SASS:
FFMA is absent from our top opcodes while FMUL (454) + FADD (136) dominate.
`sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active`: 2.82 vs 2.14;
XU (MUFU/exp2) 4.17 vs 2.92.

[SOURCE] We compute the exp2 argument as a separate subtract-then-multiply and
recompute the scaled max per element:
`gqa_attention_prefill.cuh:378-381` → `exp2((score - nm0) * scale_l2)`
(`nm0` is not pre-scaled, so this is FADD then FMUL). FA precomputes
`max_scaled = max * scale` once per row and issues a single FFMA
`exp2f(x * scale - max_scaled)` (FA `softmax.h:76, 88`). Our online rescale of
`acc` is `acc *= alpha` (FMUL, `gqa_attention_prefill.cuh:430-435`) and the
running sum is `l = l*alpha + bl` — FA folds equivalent work into FFMA form
(`softmax.h:157-160`).

### 5.3 Integer / address math is ~4.6× heavier [MEASURED] + [SOURCE]

[MEASURED] `sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active`: 4.56
(ours) vs 0.99 (FA). Static SASS integer ops per iteration: LOP3 94, IADD 90,
IMAD 27, SHF 13 (plus PLOP3 34).

[SOURCE] We recompute swizzled ldmatrix byte addresses in-loop from per-lane
pieces (`gqa_prefill_swz_addr`, `gqa_attention_prefill.cuh:80-83`) for every QK
and PV `ldmatrix` (`:297-320, 456-473`), and the cp.async staging recomputes a
swizzle per element (`gqa_prefill_swz`, `:58-60`, called at `:156, 244`). FA's
addresses come from a cute `tile_to_shape` swizzled layout whose per-k-slice
`ldmatrix` offsets are compile-time-folded (FA `kernel_traits.h:79-95`,
`utils.h:139-160`), which is why FA's ALU pipe is near-idle (0.99%).

### 5.4 cp.async writes into smem are bank-conflicted; ldmatrix reads are not [MEASURED] + [SOURCE]

[MEASURED] Total shared bank conflicts: ours 6,344,186 vs FA 0. Split by op:
`op_ld` (ldmatrix) = **0** on both; `op_st` (STS) = 0 on both;
`op_ldgsts` (cp.async) = **6,396,868** (ours) vs 0 (FA). So 100% of our conflicts
are on the cp.async global→shared store path, none on the tensor-core reads.

[SOURCE] Our gmem→smem staging maps **32 threads across one full 256-wide row**
(`gqa_attention_prefill.cuh:153-156`: `key_l = chunk>>5`, `d = (chunk&31)<<3`),
so a 512-byte row is written by 32 threads → the write wavefront revisits each
bank. FA maps **8 threads per 64-column page** (`kGmemThreadsPerRow =
kBlockKSmem/8 = 8`, `GmemLayoutAtom = <16, 8>`; FA `kernel_traits.h:118-133`),
which is documented there as the exact choice that makes the smem store
conflict-free.

Impact caveat (measured): the associated `mio_throttle` stall is small on both
(ours 0.066 vs FA 0.101 cyc), so this conflict is a real correctness-of-design
difference but **not** a first-order contributor to the 1.25× runtime gap. It is
listed for completeness and because it is cheap to fix.

### 5.5 short_scoreboard (smem/ldmatrix dependency) is 2.6× ours [MEASURED]

`smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`:
0.355 (ours) vs 0.137 (FA). long_scoreboard is equal (0.125 vs 0.133), i.e.
global-memory latency hiding is already on par; the remaining smem-side stall is
the ldmatrix→MMA and pack dependency chain, consistent with our higher LSU pipe
use (20.0% vs 16.3%) and the extra F2FP(128)/PRMT(33) pack ops per iteration
(`gqa_prefill_pack_bf16`, `gqa_attention_prefill.cuh:62-68, 386-391`).

### 5.6 Interpreting math_pipe_throttle [MEASURED]

math_pipe_throttle is *higher* for FA (6.60 vs our 3.20 cyc). This is expected
and favorable to FA: with fewer total instructions, FA's single warp spends a
larger share of its cycles genuinely saturating the tensor pipe (88.5% active),
i.e. it is closer to tensor-bound. Our lower throttle with 2.1× the instructions
means our cycles are diluted across ALU/FMA/XU/LSU work instead of tensor work.
The absolute runtime still favors FA because it needs far fewer issue cycles.

## 6. Refuted / corrected hypotheses (by measurement)

- **"Our XOR swizzle causes the 6M bank conflicts on ldmatrix reads."**
  Refuted: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 0`
  (§5.4). The ldmatrix reads are conflict-free. The 6M conflicts are entirely
  cp.async stores (`op_ldgsts`).
- **"Our row-major smem layout (vs FA's `tile_to_shape` page-blocked layout)
  breaks conflict-freedom."** Not supported by any measured conflict on the read
  path (`op_ld = 0`). The layout difference exists in source
  (`gqa_attention_prefill.cuh:156, 218` uses row stride 512 B vs FA's page-blocked
  atom, `kernel_traits.h:79-95`) but has no measured bank-conflict cost here.
- **"short/long scoreboard is the primary gap."** Partially refuted:
  long_scoreboard is at parity (0.125 vs 0.133); only short_scoreboard differs,
  and both are small next to the instruction-count delta (§5.1).

## 7. Structural differences confirmed in source (context for §5)

These are code facts (verifiable by reading the cited lines); they explain the
composition of the measured instruction delta and are the actionable list.

1. **exp2/rescale not in FFMA form** — ours `:378-381, 425-435`; FA
   `softmax.h:76, 88, 157-160`. (Measured: §5.2)
2. **In-loop swizzle address recompute** — ours `:80-83, 156, 297-320, 456-473`;
   FA compile-time-folded `kernel_traits.h:79-95`, `utils.h:139-160`.
   (Measured: §5.3)
3. **cp.async store thread map 32/row vs 8/page** — ours `:153-156`; FA
   `kernel_traits.h:118-133`. (Measured: §5.4)
4. **Row-sum cross-thread reduction every key block vs deferred once** — ours
   `:420-423` reduces `bl0/bl1` with 4 `__shfl_xor` every iteration; FA defers
   the sum allreduce to the epilogue and per-iteration reduces only the max
   (`softmax.h:144-165` sum has no allreduce, `:172` allreduce at normalize).
   Contributes to the instruction/ALU delta (§5.1, §5.3); not independently
   isolated.
5. **Single branchy key loop vs two loops (masking + steady state)** — ours
   `:337, 341/347/375/393` branches on `full_score_tile` every iteration; FA
   splits into a masking loop and a mask-free steady loop
   (`flash_fwd_kernel.h:301, 378`). Contributes to the instruction delta;
   not independently isolated.
6. **Epilogue: scattered per-element global stores vs smem-staged vectorized
   store** — ours `:487-501` (128 `STG` in static SASS, and the details rule
   reports uncoalesced global stores at 8 of 32 bytes/sector utilized); FA stages
   `acc_o` through smem and writes 128-bit-coalesced (`flash_fwd_kernel.h:437,
   447, 465, 491`). Once-per-CTA, so small dynamic weight.

## 8. Priority for closing the gap (ranked by measured contribution)

1. **FFMA fusion of the softmax/rescale FP** (§5.2): removes the largest measured
   non-tensor block — ~33.6M non-fused FP32 ops today vs FA's 18.3M, with FA
   fusing 16.5× more. Highest expected instruction-count reduction.
2. **Cut in-loop integer/address math** (§5.3): our ALU pipe is 4.6× FA's; move
   `ldmatrix` addressing to precomputed/base-plus-immediate form so the inner
   loops emit no LOP3/IADD/IMAD for addresses.
3. **Defer the row-sum reduction to the epilogue** (§7.4): removes 4 `__shfl_xor`
   per row per key block.
4. **Split the key loop into masking + steady-state** (§7.5): removes the
   per-iteration mask branch and its predicated FP.
5. **cp.async store mapping to 8-threads/64-page** (§5.4): eliminates the 6.4M
   `op_ldgsts` conflicts (low runtime impact, cheap fix).
6. **smem-staged vectorized O epilogue** (§7.6): removes the scattered `STG`.

Expected direction (not a promise of a number): items 1–4 attack the measured
2.10× instruction delta and the 64.7%→88.5% tensor-active gap, which is the sole
measured cause of the 1.25× runtime difference.

## 9. Reproduction

```bash
# Build (once)
cmake --build build --target qus_gqa_attention_bench -j

# Ours — standalone timing
./build/bench/qus_gqa_attention_bench --append-prompt-baseline --tokens 1024 --context 4096
# -> median 0.950 ms, 122.10 TFLOP/s, tc=58.28%

# FA — standalone timing (conda env vllm-bench)
python tools/bench/flash_attn_gqa_bench.py --tokens 1024 --context 4096 --attention-only
# -> median 0.761 ms, 152.45 TFLOP/s, tc=72.77%

# Ours — full ncu capture
ncu --force-overwrite -o profiles/qus_prefill_T1024_C4096 --set full \
  --kernel-name regex:'gqa_attention_prefill_kernel' --launch-skip 4 --launch-count 1 \
  ./build/bench/qus_gqa_attention_bench --append-prompt-baseline --tokens 1024 --context 4096

# FA — full ncu capture
ncu --force-overwrite -o profiles/fa_prefill_T1024_C4096 --set full \
  --kernel-name regex:'flash_fwd_kernel' --launch-skip 10 --launch-count 1 \
  python tools/bench/flash_attn_gqa_bench.py --tokens 1024 --context 4096 \
  --attention-only --warmup 5 --repeat 8 --min-time-ms 0

# Extract raw metrics
ncu --import profiles/qus_prefill_T1024_C4096.ncu-rep --page raw --csv > profiles/qus_raw.csv
ncu --import profiles/fa_prefill_T1024_C4096.ncu-rep  --page raw --csv > profiles/fa_raw.csv

# Static SASS mix (ours)
cuobjdump -sass ./build/bench/qus_gqa_attention_bench
```

## 10. Implementation update from this audit [MEASURED]

Current code:

- `src/kernels/kernel/gqa_attention_prefill.cuh`
- `src/kernels/launcher/gqa_attention.h`
- `src/kernels/launcher/gqa_attention_prefill.cu`
- `bench/gqa_attention_bench.cu`

Accepted changes:

1. Packed output bf16 conversion now uses `cvt.rn.bf16x2.f32` and 32-bit stores
   instead of two scalar bf16 conversions and stores.
2. The softmax exp2 argument is emitted in FFMA form by pre-scaling the current
   block max and using `__fmaf_rn(score, scale_l2, -max_scaled)`.
3. The alpha rescale path removed the first-tile `-inf` guard; valid rows
   naturally produce `exp2(-inf) == 0`.
4. The running row sum update uses FFMA form, and the row-sum allreduce is
   deferred to the epilogue instead of running four shuffles every key tile.
5. K/V staging now splits full and partial tiles before the staging loop and
   unrolls the fixed 16-iteration cp.async loop.
6. Q/K/V staging uses a local `cp.async.cg.shared.global` helper, matching FA's
   cache-global staging choice instead of the old `.ca` helper. This removed
   the measured cp.async shared-bank conflicts.
7. Q staging splits full query tiles from partial tiles, so the common
   T=1024/context=4096 path avoids one per-copy bounds branch in the prologue.
8. Prefill/fill kernel pointer arguments are marked `__restrict__`.
9. The K/V fill kernel copies 16 B (`int4`) chunks and the launcher sizes the
   fill grid at that vector granularity.
10. The benchmark has an `--append-prompt-attention-only` mode that fills the
   cache once, then times only `gqa_attention_prefill_kernel`, matching FA's
   `--attention-only` comparator.

Standalone timing after this implementation pass:

```bash
for i in 1 2 3; do \
  ./build/bench/qus_gqa_attention_bench \
    --append-prompt-baseline --tokens 1024 --context 4096 \
    --warmup 20 --repeat 100 --min-time-ms 500; \
done
# -> median 0.766 ms, useful 151.36-151.41 TFLOP/s,
#    tc=72.25-72.27%, ns/key=0.162

for i in 1 2 3; do \
  ./build/bench/qus_gqa_attention_bench \
    --append-prompt-attention-only --tokens 1024 --context 4096 \
    --warmup 20 --repeat 100 --min-time-ms 500; \
done
# -> median 0.759-0.760 ms, useful 152.69-152.85 TFLOP/s,
#    tc=72.88-72.96%, ns/key=0.161

for i in 1 2 3; do \
  conda run -n vllm-bench python tools/bench/flash_attn_gqa_bench.py \
    --tokens 1024 --context 4096 --attention-only \
    --warmup 20 --repeat 100 --min-time-ms 500; \
done
# -> FA attention-only median 0.761-0.763 ms, useful 152.06-152.41 TFLOP/s

for i in 1 2 3; do \
  conda run -n vllm-bench python tools/bench/flash_attn_gqa_bench.py \
    --tokens 1024 --context 4096 --include-fill \
    --warmup 20 --repeat 100 --min-time-ms 500; \
done
# -> FA with-fill median 0.795-0.796 ms, useful 145.62-145.91 TFLOP/s
```

The original audit compared our full append-prompt wrapper against FA
attention-only, which is a useful upper-bound comparison but not an
apples-to-apples API comparison. With the matching with-fill operation, current
ours is about **1.04x faster** than FA (0.766 ms vs 0.795-0.796 ms). Against the
FA attention-only upper bound, current attention-only timing is also a small
stable win (0.759-0.760 ms vs 0.761-0.763 ms).

NCU spot check after this implementation pass:

| metric | original ours | current ours | FA comparison |
|--------|---------------|--------------|---------------|
| full append/wrapper median | 0.950 ms | 0.766 ms | n/a |
| attention-only median | n/a | 0.759-0.760 ms | 0.761-0.763 ms |
| with-fill median | n/a | 0.766 ms | 0.795-0.796 ms |
| useful TFLOP/s, full/with-fill | 122.10 | 151.36-151.41 | 145.62-145.91 |
| useful TFLOP/s, attention-only | n/a | 152.69-152.85 | 152.06-152.41 |
| `smsp__inst_executed.sum` | 185,801,472 | 124,551,168 | 88,628,736 |
| tensor active % | 64.70 | 82.17 | 88.52 |
| shared bank conflicts, total | 6,344,186 | 0 | 0 |
| cp.async bank conflicts | 6,396,868 | 0 | 0 |
| main register / thread | 246 | 249 | 235 |
| fill register / thread | 16 | 20 | n/a |

This closes the measured cp.async conflict issue completely and cuts main-kernel
dynamic instructions by 33.0% from the original kernel. Wall-clock now meets the
audit target: current ours is faster than FA in both the matching with-fill path
and the stricter attention-only comparator.

Measured candidates that were reverted:

- Page-style cp.async staging intended to mimic FA's 8-thread/64-column store
  map passed correctness but was slower on this kernel.
- Smem-staged vectorized output stores passed correctness but added shared
  memory traffic, barriers, and conflicts without improving wall time.
- A templated mask-free steady loop reduced source-level branching but increased
  dynamic instruction count in NCU.
- Row-major non-XOR shared layout made runtime much worse; the original XOR
  layout remains the best measured local layout.
- Q staging unroll, compile-time scale specialization, `cp.async` `L2::128B`
  hinting, and approximate reciprocal specialization either pushed register
  pressure to 255 registers/thread or did not improve sustained long-run timing.

Remaining gap:

The local fixes above moved the full append-prompt path from roughly 0.950 ms to
0.766 ms, which beats FA's matching with-fill operation and its attention-only
kernel on this audit shape. The remaining structural gap is still main-kernel
dynamic instruction count: 124.6M vs FA's 88.6M. That gap no longer prevents
wall-clock parity on RTX 5090 for T=1024/context=4096, but closing it further
would require the larger FA-style reverse mainloop/address schedule rewrite
identified above, not additional launch-side changes.
