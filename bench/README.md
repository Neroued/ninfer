# Benchmarks

`ninfer_bench` measures the complete public `ninfer::Engine` route against a `.ninfer` artifact.
The `bench/ops/` `ninfer_<op>_bench` executables measure central Op contracts and their specialized
CUDA implementations for ncu/nsys work. Target benchmarks measure Program/model composition.
Correctness and model parity live outside this directory; development rules are in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --parallel --target ninfer_bench
```

## Product benchmark

The benchmark slices exact token counts from `bench/fixtures/bench_corpus.ids`, calls
`Engine::prepare_tokens()`, then calls `Engine::generate()` once for each repetition. It does not
have a private prefill/decode loop and does not call target implementation interfaces.

The matrix contains three independently measured test kinds:

- `pp{P}` prepares `P` tokens and requests one output token. This is the smallest request that runs
  the model; `prefill t/s` is `P / GenerationTimings.prefill_seconds`.
- `tg{G}` prepares a one-token seed outside the reported phase and requests `G+1` output tokens.
  The begin-round token belongs to prefill, leaving exactly `G` tokens in the reported decode
  phase.
- `pp{P}+tg{G}` uses the same `G+1` convention after a `P`-token prefill and reports both phase
  rates from the same generation call.

All benchmark requests use raw output, disable model-default stops, and disable prefix reuse. This
keeps the requested token count exact without adding another generation path. When CUDA Graph is
enabled and the matrix contains decode work, one ordinary public generation request primes the
decode graph before warmups and measured repetitions.

## CLI

```text
ninfer_bench --weights <artifact.ninfer>
          [--corpus <ids-path>]
          [-p, --n-prompt <list>]
          [-n, --n-gen <list>]
          [-pg, --prompt-gen <P,G;P,G...>]
          [-r, --repetitions <n>] [--warmup <n>]
          [--max-ctx <tokens>] [--prefill-chunk <tokens>]
          [--kv-dtype <bf16|int8>]
          [--mtp-draft-tokens <0..5>] [--lm-head-draft]
          [--device <id>] [--no-cuda-graph] [--profile-measured]
          [-o, --output <table|json|csv>] [--output-file <path>]
```

With no `-p`, `-n`, or `-pg`, the matrix is `pp512` and `tg128`.

Example:

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1
```

`bf16` selects BF16 KV storage and `int8` selects INT8 group-64 KV storage. MTP is enabled with
`--mtp-draft-tokens`; `--lm-head-draft` selects the optimized proposal head. CUDA Graph decode is
enabled by default.

`--profile-measured` is a benchmark-only profiler boundary. It requires exactly one selected test
and `-r 1`, synchronizes after warmup, and brackets only the measured repetition with
`cudaProfilerStart/Stop`. Use it with an Nsight Systems `cudaProfilerApi` capture range so artifact
load, graph construction, and warmup do not enter topology counts.

## Text linear Op benchmarks

`ninfer_linear_op_bench` exposes the registered 27B fused LinearSwiGLU and Q5 LinearAdd contracts
through their production dispatch at the default `T=1024` prefill extent:

```bash
./build/bench/ninfer_linear_op_bench \
  --shape MlpGateUp34816x5120 --qtype Q4 --linear-swiglu --t-sweep 1024
./build/bench/ninfer_linear_op_bench \
  --shape MlpDown5120x17408 --qtype Q5 --linear-add --t-sweep 1024
./build/bench/ninfer_linear_op_bench \
  --shape Out5120x6144 --qtype Q5 --linear-add --t-sweep 1024
```

The benchmark records the selected physical route, kernel variant, cold-cache timing, measured
tensor-core ceiling, and useful/executed throughput.

## GDN control-projection Op benchmark

`ninfer_gdn_gating_proj_bench` measures the registered BF16 control projection. With `--35b
--norm-control`, it measures the complete 35B RMSNorm, normalized hidden output, A/B projection,
gate transform, and beta transform contract. `--candidate auto` uses production dispatch;
`--candidate composed` is the explicit RMSNorm-plus-control comparison. Every row reports the
selected route and transient workspace after a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_gdn_gating_proj_bench
./build/bench/ninfer_gdn_gating_proj_bench \
  --35b --norm-control --candidate auto \
  -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --warmup 10 --repeat 200
./build/bench/ninfer_gdn_gating_proj_bench \
  --35b --norm-control --candidate composed \
  -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --warmup 10 --repeat 200
```

## Gated-delta snapshot Op benchmark

`ninfer_gated_delta_rule_bench --small-t` measures the stateful recurrent snapshot contract with
17 state slots and exact `T=1..16`. Every row reports the complete selected route. Its useful-byte
rate includes raw Q/K/V, g/beta, output, the selected initial FP32 state read, and all `T` FP32
snapshot writes; it does not inflate the rate with cache-line or implementation traffic.
`--qk-norm fused` is the production one-kernel route. `composed` explicitly runs two L2Norm kernels
and the pre-normalized recurrent kernel as a containing-layer control.

```bash
cmake --build build --parallel --target ninfer_gated_delta_rule_bench ninfer_gdn_layer_bench
./build/bench/ninfer_gated_delta_rule_bench \
  --small-t --H_v 32 --qk-norm fused \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --warmup 20 --repeat 500 --csv
./build/bench/ninfer_gated_delta_rule_bench \
  --small-t --H_v 32 --qk-norm composed \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --warmup 20 --repeat 500 --csv
./build/bench/ninfer_gdn_layer_bench \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --route fused --norm-control fused --qk-norm fused --warmup 20 --repeat 500
```

## Input-projection Op benchmark

`ninfer_input_proj_bench` measures the exact Qwen3.6-27B Attention and GDN input-projection shapes.
Attention production uses the two parent projections and its benchmark-only control uses the
former four logical projections. GDN production writes directly into the pitched final output;
its controls isolate projection time and the former materialize-plus-two-copy composition. All
timed operands are allocated before measurement, and each sample is preceded by a 256 MiB L2 flush.
Production accepts the Text token extent through the benchmark allocation limit; controls remain
limited to their Small-T domain.

```bash
cmake --build build --parallel --target ninfer_input_proj_bench
./build/bench/ninfer_input_proj_bench \
  --op all --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,128,129,1024 \
  --warmup 5 --repeat 50 --csv-out profiles/bench/input_proj.csv
```

The four-projection and materialize/copy controls exist only in this benchmark and are not
production-callable routes.

## Bidirectional GQA Op benchmark

`ninfer_bidirectional_gqa_attention_bench` measures the read-only, non-causal Q32/KV8/D128
attention contract for `T=1..16`. Every timed invocation is a CUDA Graph replay. `--cold-cache`
flushes 256 MiB before each sample outside the timed interval; `--route direct|split` and the tile
and split overrides expose candidate controls, while `--route production` uses measured dispatch.

```bash
cmake --build build --parallel --target ninfer_bidirectional_gqa_attention_bench
./build/bench/ninfer_bidirectional_gqa_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,2048,8192,32768,131072,196608,262144 \
  --route production --cold-cache --warmup 10 --repeat 61
./build/bench/ninfer_bidirectional_gqa_attention_bench \
  --tokens 16 --context 262144 --route production --cold-cache --profile-once
```

The reported useful roofline counts one read of context K/V, query Q/K/V, and one output write,
plus the full QK and PV FLOPs. It intentionally excludes implementation scratch traffic.

## Symmetric sliding-window GQA Op benchmark

`ninfer_swa_bench` measures the read-only, non-causal Q32/KV8/D128 attention contract over a
4096-slot cyclic BF16 context and a complete temporary query block for `T=1..16`. Every timed
invocation is a CUDA Graph replay. `--cold-cache` flushes 256 MiB before each sample outside the
timed interval. `--route direct|split`, `--key-block`, and `--split-capacity` expose candidate
controls; `--route production` uses the measured dispatch.

```bash
cmake --build build --parallel --target ninfer_swa_bench
./build/bench/ninfer_swa_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,32,64,96,128,4095,4096,8192,262144 \
  --route production --cold-cache --warmup 10 --repeat 61
./build/bench/ninfer_swa_bench \
  --tokens 16 --context 4096 --route production --cold-cache --profile-once
```

The useful roofline counts the admitted cyclic context once, query Q/K/V, one output write, and
the complete QK/PV FLOPs. It intentionally excludes split/reduce scratch traffic; the
implementation-traffic comparison is reported separately in the Op qualification.

## Device-count K/V prefix-append benchmark

`ninfer_kv_cache_append_prefix_bench` measures exact BF16 D128/KV8 prefix publication into linear
and 4096-slot cyclic cache layouts. `T` fixes the captured launch envelope, while `C` is the device
commit count and determines the exact bytes written. Every timed invocation is one CUDA Graph
replay. `--cold-cache` flushes 256 MiB before each sample outside the timed interval, and
`--route flat16|flat32|persistent32|token` exposes private candidate controls.

```bash
cmake --build build --parallel --target ninfer_kv_cache_append_prefix_bench
./build/bench/ninfer_kv_cache_append_prefix_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --counts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --layout all --route production --cold-cache --warmup 10 --repeat 61
./build/bench/ninfer_kv_cache_append_prefix_bench \
  --tokens 16 --counts 16 --layout cyclic --route production --profile-once
```

Useful traffic is the exact committed K/V input read plus cache write, 8192 bytes per committed
token. `C=0` still exercises the captured device-count launch but has zero useful bytes.

## Masked-block preparation benchmark

`ninfer_prepare_masked_block_bench` measures the exact I32 anchor/mask block transform for every
registered `B=2..16`. Every timed invocation is one CUDA Graph replay. An untimed 256 MiB write
conditions GPU clocks; hot mode primes the Op once after that write, while `--cold-cache` measures
immediately after it. `--route warp32|block64|block128|block256` exposes private candidate
controls, and `--route production` uses measured dispatch.

```bash
cmake --build build --parallel --target ninfer_prepare_masked_block_bench
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --route all
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --route production --cold-cache --warmup 20 --repeat 101
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 16 --route production --profile-once
```

Useful traffic is `(2+2B)*4` bytes: two device scalar reads and two complete I32 output writes.
This benchmark qualifies block preparation and route selection only; it neither includes nor
claims the deferred embedding fusion or complete-round performance.

## 35B W8 input-projection Op benchmark

`ninfer_w8_input_proj_bench` measures the registered 35B-A3B target's W8 Attention
`[9216,2048]`, the companion Q/K/V Attention `[6144,2048]`, and GDN `[12288,2048]`
multi-output Ops. Production writes independent contiguous consumer allocations directly. The
controls expose each compiled SIMT/MMA candidate plus the semantically equivalent parent Linear
alone and parent Linear followed by four, three, or two column extracts. Each timed sample is
preceded by a 256 MiB L2 flush.

`--op companion-attention` covers the exact `T=1..16` proposal domain and the tuned prefill
dispatch. `--production-only` suppresses candidates for a compact boundary sweep; `--profile`
launches the selected production kernel once for an NCU capture.

`--op gdn-snapshot` measures the complete stateful GDN input projection, causal convolution, SiLU,
Q/K/V split, z projection, and snapshot publication contract. It reports the selected production
route and an explicit five-launch composed control at every requested T.

```bash
cmake --build build --parallel --target ninfer_w8_input_proj_bench
./build/bench/ninfer_w8_input_proj_bench \
  --op all --t-sweep 1,2,4,8,12,13,16,17,32,64,128,129,256,512,1024 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_input_proj_final.csv

./build/bench/ninfer_w8_input_proj_bench \
  --op gdn-snapshot --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --warmup 5 --repeat 50 --csv-out profiles/bench/gdn_input_snapshot.csv

./build/bench/ninfer_w8_input_proj_bench \
  --op companion-attention \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,32,33,64,65,96,97,128,256,512,896,1024 \
  --production-only --warmup 20 --repeat 100 \
  --csv-out profiles/bench/w8_companion_attention.csv
```

The executable isolates these Op contracts; end-to-end 35B-A3B measurement uses `ninfer_bench`.

## 35B sparse-MoE dFlash benchmark

`ninfer_sparse_moe_bench` measures the complete routed-plus-shared post-mixer Op for the 35B Text
Q4+Q5/Q6 profiles and the MTP W8+W8 profile. Small-T rows report the selected number of D3
paths/CTA (`p`) and D4 output rows/CTA (`r`). `--matrix` includes the former three-path/one-row
schedule as a same-process control.

```bash
cmake --build build --parallel --target ninfer_sparse_moe_bench
for codec in q4-q5 q4-q6 w8-w8; do
  for tokens in $(seq 1 16); do
    ./build/bench/ninfer_sparse_moe_bench \
      --scope full --candidate production --codec "$codec" --tokens "$tokens" \
      --distribution trace-like --warmup 5 --repeat 50
  done
done

./build/bench/ninfer_sparse_moe_bench \
  --matrix --tokens 16 --distribution trace-like --warmup 5 --repeat 80
```

Each cold sample flushes 256 MiB before the timed interval. `trace-like` is the primary
single-sequence verification distribution; `independent` and `same` bound zero and complete expert
overlap. Candidate names `small-t-p{1,3,9}-r{1,2,4}` expose the compiled schedule matrix without
changing public dispatch.

## Target MTP round benchmark

`ninfer_qwen3_6_27b_mtp_round_bench` measures the registered target's native proposal and
verification round without introducing a second generation controller. It loads the same `.ninfer`
artifact through the target-private package facade, prepares a real prompt with that target's
Frontend, and reports draft/accept statistics for the target-owned MTP schedule:

```bash
cmake --build build --parallel --target ninfer_qwen3_6_27b_mtp_round_bench
./build/bench/ninfer_qwen3_6_27b_mtp_round_bench \
  --artifact out/qwen3_6_27b.ninfer
```

## 35B target-side speculative round benchmark

`ninfer_qwen3_6_35b_a3b_target_speculative_round_bench` measures the shared target verification,
greedy acceptance, accepted-hidden selection, and target-state publication transaction without
including MTP or dFlash proposal work. It uses the real 35B Text weights, target KV cache, and 17
GDN snapshot slots when `K=15`. The default sweep covers `K=1..15` in eager and CUDA Graph modes;
`--accepted-drafts A` forces one acceptance frontier for every selected K:

```bash
cmake --build build --parallel \
  --target ninfer_qwen3_6_35b_a3b_target_speculative_round_bench
./build/bench/ninfer_qwen3_6_35b_a3b_target_speculative_round_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer \
  --context 128 --draft-tokens 7,15 --mode both
```

The reported `target_side_effective_tok_s` is `(A+1)/target-side latency`. It deliberately excludes
proposal generation and dFlash-specific context maintenance and is not an end-to-end speed claim.

## Token-decision Op benchmarks

The G1 benchmark covers the Qwen3.6-35B full physical vocabulary with 248077 valid rows at
`C=1..16`, plus the 131072-row shortlist. Its `--control` route reads the same rotating payload and
uses the same launch grid without computing argmax, which provides the fixed-work comparison used
by the benchmark comparison. `--candidate-block` forces a tiled-atomic CTA geometry:

```bash
cmake --build build --parallel --target ninfer_argmax_bench ninfer_sampling_select_bench
./build/bench/ninfer_argmax_bench
./build/bench/ninfer_argmax_bench --control
./build/bench/ninfer_argmax_bench --candidate-block 128
```

The G2/G3 benchmark uses physical rows 248320, valid token domain 248077, optional occurrence
counts, and every MTP window `K=1..5`. With no arguments it runs the full greedy/stochastic matrix;
individual routes are suitable for Nsight Compute capture:

```bash
./build/bench/ninfer_sampling_select_bench --matrix
./build/bench/ninfer_sampling_select_bench --sample --mode stochastic --top-k 20
./build/bench/ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5 --top-k 20
```

## 35B dFlash attention qualification

The GQA benchmark covers exact verify widths `T=1..16`, both KV codecs, append-and-attend and
already-cached attention, and prints the selected context-dependent route:

```bash
./build/bench/ninfer_gqa_attention_bench --append-small-t --geometry 35b \
  --kv-dtype bf16 --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192
./build/bench/ninfer_gqa_attention_bench --cached-small-t --geometry 35b \
  --kv-dtype int8 --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192
```

The complete 35B attention mixer benchmark uses cold-cache CUDA Graph replay. Its
`prompt-control` route preserves the former prompt implementation for a same-build layer A/B:

```bash
./build/bench/ninfer_attention_layer_bench --geometry 35b --kv-dtype bf16 \
  --context 128,1024,8192 --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
./build/bench/ninfer_attention_layer_bench --geometry 35b --kv-dtype bf16 \
  --attention-route prompt-control --context 1024,8192 --t-sweep 7,8,9,10,11,12,13,14,15,16
```

## Pointwise Op benchmarks

The Section 5 benchmarks cover the complete Qwen3.6-35B pointwise matrix. Default invocation runs
all registered small, established, maximum-video, and maximum-image shapes. `--control` preserves
the selected kernel topology and payload while replacing the mathematical operation with minimal
bitwise work:

```bash
cmake --build build --parallel --target \
  ninfer_residual_add_bench ninfer_sigmoid_mul_bench \
  ninfer_gelu_bench ninfer_add_bias_bench

./build/bench/ninfer_residual_add_bench [--patches P] [--control]
./build/bench/ninfer_sigmoid_mul_bench \
  [--tokens T[,T...]] [--control | --candidate-block B]
./build/bench/ninfer_position_bench \
  [--tokens T[,T...]] [--candidate-block B] [--cold-graph] [--warmup N] [--repeat N]
./build/bench/ninfer_gelu_bench [--mode tanh|exact --columns C] [--control]
./build/bench/ninfer_add_bias_bench [--d D --columns C] [--control]
```

Aligned registered shapes use 16-byte BF16 packs in the cache-sized regime. GELU and AddBias
select their BF16x2 streaming routes for larger Vision items; odd or unaligned repository-internal
test shapes exercise the scalar fallbacks.

## Reports

Table, JSON, and CSV reports all identify the selected target, artifact, Engine configuration,
load summary, memory capacity, KV payload, workspace peak, phase throughput, and speculative
statistics. JSON schema version 8 records the public value objects directly:

- `load`: target, load/upload time, file/H2D/staging bytes, tensor count, and resource count;
- `memory`: weights/sequence/workspace arenas, planned context, KV storage, and KV payload;
- each repetition's `timings`: prepare, Vision, prefill, decode, and total seconds;
- each repetition's `speculative`: window, rounds, drafted/accepted tokens, fallbacks, and per-position
acceptance.

`decode_output_tok_s` counts the requested `G` decode outputs. `decode_engine_tok_s` uses the
Program's speculative round statistics, so it also describes work performed by a final partially
committed speculative round. Reports also contain the command and machine information needed to
interpret a local measurement.

Raw reports and profiler captures remain local under `profiles/bench`, `profiles/ncu`, and
`profiles/nsys`.
