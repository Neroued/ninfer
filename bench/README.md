# Benchmarks

`ninfer_bench` measures the complete public `ninfer::Engine` route against a `.ninfer` artifact.
The `bench/ops/` `ninfer_<op>_bench` executables measure central Op contracts and their specialized
CUDA implementations for ncu/nsys work. Target benchmarks measure Program/model composition.
Correctness and model parity live outside this directory; development rules are in
[`../docs/op-development.md`](../docs/op-development.md).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j --target ninfer_bench
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
          [--device <id>] [--no-cuda-graph]
          [-o, --output <table|json|csv>] [--output-file <path>]
```

With no `-p`, `-n`, or `-pg`, the matrix is `pp512` and `tg128`.

Example:

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1
```

`bf16` selects BF16 KV storage and `int8` selects INT8 group-64 KV storage. MTP is enabled with
`--mtp-draft-tokens`; `--lm-head-draft` selects the optimized proposal head. CUDA Graph decode is
enabled by default.

## Target MTP round benchmark

`ninfer_qwen3_6_27b_mtp_round_bench` measures the registered target's native proposal and
verification round without introducing a second generation controller. It loads the same `.ninfer`
artifact through the target-private package facade, prepares a real prompt with that target's
Frontend, and reports draft/accept statistics for the target-owned MTP schedule:

```bash
cmake --build build -j --target ninfer_qwen3_6_27b_mtp_round_bench
./build/bench/ninfer_qwen3_6_27b_mtp_round_bench \
  --artifact out/qwen3_6_27b_rtx5090.ninfer
```

## Token-decision Op benchmarks

The G1 benchmark covers the Qwen3.6-35B full physical vocabulary with 248077 valid rows at
`C=1..6`, plus the 131072-row shortlist. Its `--control` route reads the same rotating payload and
uses the same launch grid without computing argmax, which provides the fixed-work comparison used
by the retained qualification report:

```bash
cmake --build build -j --target ninfer_argmax_bench ninfer_sampling_select_bench
./build/bench/ninfer_argmax_bench
./build/bench/ninfer_argmax_bench --control
```

The G2/G3 benchmark uses physical rows 248320, valid token domain 248077, optional occurrence
counts, and every MTP window `K=1..5`. With no arguments it runs the full greedy/stochastic matrix;
individual routes are suitable for Nsight Compute capture:

```bash
./build/bench/ninfer_sampling_select_bench --matrix
./build/bench/ninfer_sampling_select_bench --sample --mode stochastic --top-k 20
./build/bench/ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5 --top-k 20
```

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
