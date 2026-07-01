# Benchmarks

This directory contains benchmark binaries. `qus_bench` is the real-weight throughput tool; the
`qus_<op>_bench` binaries are per-kernel microbenchmarks. Correctness and per-layer parity are NOT
handled here; they live under [`tools/parity`](../tools/parity).

## Benchmark types

Per-op benchmarks are the `qus_<op>_bench` binaries. They run one kernel family at Qwen3.6-27B
shapes and are the entry point for ncu/nsys analysis. Their stdout numbers are convenience
readouts; optimization claims require profiler evidence plus before/after `qus_bench` reports.

`qus_bench` is the real-weight throughput benchmark, modeled on `llama-bench`. It drives
`Engine::load()`, `Engine::prefill()`, and `Engine::decode_step()` directly and measures prefill
(`pp`) and decode (`tg`) throughput separately.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target qus_bench
```

## Meaningful token corpus

`qus_bench` reads token ids only; it has no tokenizer dependency. Prefill of an exact length `P`
is the first `P` ids of a committed, meaningful corpus:

```text
bench/fixtures/bench_corpus.ids
bench/fixtures/bench_corpus.manifest.json
```

Bake or verify the corpus with [`tools/bench/make_bench_corpus.py`](../tools/bench/README.md). The
committed corpus is ~64k tokens and bounds the largest prefill length; re-bake with a larger
`--tokens`, or pass `--source-text` to tokenize a downloaded book/dataset, to go higher (memory
permitting).

## Test model

Three test kinds, each measured independently over `repetitions` timed runs (plus discarded
`warmup` runs). A warmup run is required for honest decode numbers because the first `decode_step`
captures the CUDA graph; measured runs then reflect steady graph replay.

- `pp{P}` — prefill `P` meaningful tokens. `prefill t/s = P / prefill_time`.
- `tg{G}` — prefill a 1-token seed (untimed), then time exactly `G` decode steps. `decode t/s = G / decode_time`. Decode ignores stop tokens, so the length is exact.
- `pp{P}+tg{G}` — prefill `P`, then decode `G` in one sequence; reports both rates (decode runs at context offset `P`).

Timing boundary is `host_visible_phase_end`: `prefill()` and each `decode_step()` synchronize and
read back their token before the timer stops, so decode t/s includes per-step device-to-host
readback. `max_ctx` is auto-sized to the largest test requirement (`pp`→P, `tg`→G+1, `pp+tg`→P+G)
unless `--max-ctx` overrides it; a `--max-ctx` smaller than a test needs is rejected.

## CLI

```text
qus_bench --weights <q5090-path>
          [--corpus <ids-path>]              # default: bench/fixtures/bench_corpus.ids
          [-p, --n-prompt <list>]            # pp tests, e.g. 512 or 128,512,2048
          [-n, --n-gen <list>]               # tg tests, e.g. 128
          [-pg, --prompt-gen <P,G;P,G...>]   # combined pp+tg tests
          [-r, --repetitions <n>]            # default 5
          [--warmup <n>]                     # default 1, discarded
          [--max-ctx <tokens>]               # default: auto = max test requirement
          [--work-bytes <bytes>]             # prefill workspace arena size (raise for long prefills)
          [--device <id>] [--no-cuda-graph]
          [-o, --output <table|json|csv>]    # default table
          [--output-file <path>]             # default stdout
```

With no `-p/-n/-pg`, the default matrix is `pp512` and `tg128`. Progress lines are written to
`stderr` with the `[qus_bench]` prefix and are not part of the output artifact.

Example:

```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  -p 512,2048 -n 128 -pg 2048,128 -r 5 --warmup 1
```

## Output

Default `table` prints an identity/config header followed by one row per test with prefill and
decode t/s (`mean ± stddev`; the stddev is omitted for a single repetition) and the `work peak`
column (high-water workspace-arena usage for that test). `--output json` and `--output csv` write
machine-readable results, including `workspace_peak_bytes` per test.

Prefill workspace scales with the prompt length, so long prefills can overflow the default
workspace arena (`std::bad_alloc`). Raise `--work-bytes` accordingly; the reported `work peak` /
`workspace_peak_bytes` tells you how much a given prefill length actually needed, so you can size
it. On a 32 GB card with the ~16 GB q5090 model, prefills into the tens of thousands of tokens are
workspace- and memory-bound (e.g. `pp16384` needs roughly 5 GiB of workspace).

JSON shape (`schema_version: 1`, `artifact_type: "qus_bench_report"`):

```json
{
  "schema_version": 1,
  "artifact_type": "qus_bench_report",
  "tool": "qus_bench",
  "command": "",
  "git_commit": "",
  "worktree_dirty": false,
  "environment": {"gpu_name": "", "cuda_runtime_version": "", "cuda_driver_version": "", "device_id": 0},
  "weights": {"path": "", "file_size_bytes": 0},
  "config": {"max_ctx": 0, "work_bytes": 0, "decode_path": "cuda_graph", "repetitions": 5,
             "warmup": 1, "timing_boundary": "host_visible_phase_end", "corpus_path": "",
             "corpus_tokens": 0},
  "tests": [
    {
      "label": "pp2048+tg128", "kind": "pp+tg", "n_prompt": 2048, "n_gen": 128,
      "prefill_tok_s_mean": 0.0, "prefill_tok_s_stddev": 0.0,
      "decode_tok_s_mean": 0.0, "decode_tok_s_stddev": 0.0,
      "prefill_time_s_mean": 0.0, "decode_time_s_mean": 0.0,
      "workspace_peak_bytes": 0,
      "reps": [{"prefill_time_s": 0.0, "prefill_tok_s": 0.0, "decode_time_s": 0.0, "decode_tok_s": 0.0}]
    }
  ]
}
```

`kind` is `pp`, `tg`, or `pp+tg`. Phase fields that do not apply to a test kind are `null`
(a `pp` test has null decode fields; a `tg` test has null prefill fields). CSV columns are
`label,kind,n_prompt,n_gen,repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,decode_tok_s_mean,decode_tok_s_stddev,prefill_time_s_mean,decode_time_s_mean`,
with empty cells for inapplicable phases.

## Artifacts

Raw reports are local and stay out of git:

```text
profiles/bench/
```

Profiler outputs are local:

```text
profiles/ncu/
profiles/nsys/
```

## Performance claims

A valid performance claim needs both local per-op/profiler evidence for the touched kernel family
and before/after `qus_bench` reports for the relevant test matrix. Any published number must state
the command, artifact path, git commit, q5090 identity, and dirty/clean worktree state.
