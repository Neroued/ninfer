# Benchmarks

This directory contains benchmark binaries. `ninfer_bench` is the real-weight throughput tool; the
`ninfer_<op>_bench` binaries are per-kernel microbenchmarks. Correctness and per-layer parity are NOT
handled here; they live under [`tools/parity`](../tools/parity).

## Benchmark types

Per-op benchmarks are the `ninfer_<op>_bench` binaries. They run one kernel family at Qwen3.6-27B
shapes and are the entry point for ncu/nsys analysis. Their stdout numbers are convenience
readouts; optimization claims require profiler evidence plus before/after `ninfer_bench` reports.

`ninfer_bench` is the real-weight throughput benchmark, modeled on `llama-bench`. It drives
`Engine::load()`, `Engine::prefill()`, and `Engine::decode_step()` directly and measures prefill
(`pp`) and decode (`tg`) throughput separately.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j --target ninfer_bench
```

## Meaningful token corpus

`ninfer_bench` reads token ids only; it has no tokenizer dependency. Prefill of an exact length `P`
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
`warmup` runs). When CUDA graph decode is enabled and the matrix contains decode work, `ninfer_bench`
also primes the decode graph once before timed repetitions so small `tg` cases do not include graph
capture in their measured time.

- `pp{P}` — prefill `P` meaningful tokens. `prefill t/s = P / prefill_time`.
- `tg{G}` — prefill a 1-token seed (untimed), then time exactly `G` `decode_step()` calls. Decode
  ignores stop tokens, so the requested output length is exact.
- `pp{P}+tg{G}` — prefill `P`, then decode `G` in one sequence; reports both rates (decode runs at
  context offset `P`).

Decode reports two token/s views:

- `decode_output_tok_s` — caller-visible requested output tokens divided by decode time (`G / time`).
- `decode_engine_tok_s` — engine-produced tokens divided by decode time. This equals output tokens
  for non-MTP and can exceed output tokens for MTP when the final round produces pending tokens.

Timing boundary is `host_visible_phase_end`: `prefill()` and each `decode_step()` synchronize and
read back their token before the timer stops, so decode rates include per-step device-to-host
readback. `max_ctx` is auto-sized to the largest test requirement. With MTP enabled, the auto size
also leaves capacity for prefill draft preparation and full MTP decode rounds; with CUDA graph decode
it also leaves capacity for graph priming. A `--max-ctx` smaller than a test needs is rejected.

## CLI

```text
ninfer_bench --weights <q5090-path>
          [--corpus <ids-path>]              # default: bench/fixtures/bench_corpus.ids
          [-p, --n-prompt <list>]            # pp tests, e.g. 512 or 128,512,2048
          [-n, --n-gen <list>]               # tg tests, e.g. 128
          [-pg, --prompt-gen <P,G;P,G...>]   # combined pp+tg tests
          [-r, --repetitions <n>]            # default 5
          [--warmup <n>]                     # default 1, discarded
          [--max-ctx <tokens>]               # default: auto = max test requirement
          [--prefill-chunk <tokens>]         # default 1024, must be a multiple of 128
          [--kv-dtype <bf16|int8>]           # default bf16
          [--work-bytes <bytes>]             # optional workspace override
          [--device <id>] [--no-cuda-graph]
          [--mtp-draft-tokens <0..5>]
          [--lm-head-draft]                   # requires MTP; loads v4.2 LM_HEAD_DRAFT
          [-o, --output <table|json|csv>]    # default table
          [--output-file <path>]             # default stdout
```

With no `-p/-n/-pg`, the default matrix is `pp512` and `tg128`. Progress lines are written to
`stderr` with the `[ninfer_bench]` prefix and are not part of the output artifact.

Example:

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  -p 512,2048 -n 128 -pg 2048,128 -r 5 --warmup 1
```

## Output

Default `table` prints an identity/config header followed by one row per test with prefill t/s,
decode output t/s, decode engine t/s (`mean ± stddev`; the stddev is omitted for a single
repetition), MTP acceptance, MTP round/fallback counts, and `work peak` (high-water workspace-arena
usage for that test). `--output json` and `--output csv` write machine-readable results, including
`config.prefill_chunk`, KV cache dtype/payload, MTP/draft-head mode, selected q5090 modules and
H2D/resident bytes, graph-prime metadata, and `workspace_peak_bytes` per test.

The default workspace arena is derived from `--prefill-chunk`, not prompt length. `work peak` /
`workspace_peak_bytes` should stay flat as prompt length grows at a fixed prefill chunk. Use
`--work-bytes` only as an explicit experiment override.

JSON shape (`schema_version: 7`, `artifact_type: "ninfer_bench_report"`):

```json
{
  "schema_version": 7,
  "artifact_type": "ninfer_bench_report",
  "tool": "ninfer_bench",
  "command": "",
  "git_commit": "",
  "worktree_dirty": false,
  "environment": {"gpu_name": "", "cuda_runtime_version": "", "cuda_driver_version": "", "device_id": 0},
  "weights": {"path": "", "file_size_bytes": 0, "h2d_bytes": 0,
              "device_resident_bytes": 0, "resident_modules": ["TEXT_CORE"]},
  "config": {"max_ctx": 0, "prefill_chunk": 1024, "kv_dtype": "bf16",
             "kv_quant_group": 0, "kv_cache_payload_bytes": 0,
             "mtp_draft_tokens": 0, "lm_head_draft": false, "work_bytes": 0,
             "decode_path": "cuda_graph",
             "decode_graph_prime": {"requested": true, "primed": true, "decode_steps": 2},
             "repetitions": 5, "warmup": 1,
             "timing_boundary": "host_visible_phase_end", "corpus_path": "",
             "corpus_tokens": 0},
  "tests": [
    {
      "label": "pp2048+tg128", "kind": "pp+tg", "n_prompt": 2048, "n_gen": 128,
      "prefill_tok_s_mean": 0.0, "prefill_tok_s_stddev": 0.0,
      "decode_output_tok_s_mean": 0.0, "decode_output_tok_s_stddev": 0.0,
      "decode_engine_tok_s_mean": 0.0, "decode_engine_tok_s_stddev": 0.0,
      "prefill_time_s_mean": 0.0, "decode_time_s_mean": 0.0,
      "workspace_peak_bytes": 0,
      "mtp": {"enabled": false, "k": 0, "draft_tokens": 0, "accepted_tokens": 0,
              "acceptance_rate": null, "acceptance_length": null,
              "rounds": 0, "fallback_steps": 0, "accepted_per_pos": []},
      "reps": [{"prefill_time_s": 0.0, "prefill_tok_s": 0.0,
                "decode_time_s": 0.0, "decode_output_tokens": 128,
                "decode_engine_tokens": 128,
                "decode_output_tok_s": 0.0, "decode_engine_tok_s": 0.0,
                "mtp": {"enabled": false, "k": 0, "draft_tokens": 0,
                        "accepted_tokens": 0, "acceptance_rate": null,
                        "acceptance_length": null, "rounds": 0,
                        "fallback_steps": 0, "accepted_per_pos": []}}]
    }
  ]
}
```

`kind` is `pp`, `tg`, or `pp+tg`. Phase fields that do not apply to a test kind are `null`
(a `pp` test has null decode fields; a `tg` test has null prefill fields). CSV columns are
`label,kind,n_prompt,n_gen,prefill_chunk,mtp_draft_tokens,lm_head_draft,q5090_h2d_bytes,q5090_resident_bytes,decode_path,kv_dtype,kv_quant_group,kv_cache_payload_bytes,decode_graph_primed,decode_graph_prime_steps,mtp_rounds,mtp_fallback_steps,mtp_acceptance_rate,repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,decode_output_tok_s_mean,decode_output_tok_s_stddev,decode_engine_tok_s_mean,decode_engine_tok_s_stddev,prefill_time_s_mean,decode_time_s_mean,workspace_peak_bytes`,
with empty cells for inapplicable phase rates.

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
and before/after `ninfer_bench` reports for the relevant test matrix. Any published number must state
the command, artifact path, git commit, q5090 identity, and dirty/clean worktree state.
