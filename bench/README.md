# Benchmarks

This directory contains benchmark binaries. The authoritative pre-M3 benchmark, report, and readiness
contract is `docs/m2.8-pre-m3-standard.md`.

## Benchmark Types

Per-op benchmarks are the existing `qus_<op>_bench` binaries. They run one kernel family at Qwen3.6-27B
shapes and are the entry point for ncu/nsys analysis. Their stdout numbers are convenience readouts;
M3 optimization claims require profiler evidence plus before/after e2e JSON reports.

`qus_e2e_bench` is the M2.8 real-weight benchmark target to add. It must drive `Engine::load()`,
`Engine::prefill()`, and `Engine::decode_step()` directly, write one JSON report object per invocation,
and use the schema in `docs/bench/e2e-report-schema.md`. It must not use `Engine::generate()` for
primary timing.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build an individual benchmark target when iterating:

```bash
cmake --build build -j --target qus_argmax_bench
```

`qus_e2e_bench` is not registered yet. When M2.8 implements it, it should build as a benchmark target
but remain outside default real-weight CTest execution.

## Prompt Fixtures

Canonical e2e prompts live under:

```text
bench/fixtures/prompts/
```

Each case uses a committed pair:

```text
<case>.txt
<case>.ids
```

`.txt` is UTF-8 source text for review and regeneration. `.ids` is the canonical C++ benchmark input:
whitespace-separated decimal token ids, with no comments and no inline metadata.

The fixture set manifest is required:

```text
bench/fixtures/prompts/m2.8-v1.manifest.json
```

The manifest records tokenizer provenance, case names, prompt token counts, and hashes for each
`.txt`/`.ids` pair. `qus_e2e_bench` reads `.ids` only and has no tokenizer dependency.

## E2E CLI Contract

The required M2.8 CLI is:

```text
qus_e2e_bench \
  --weights <q5090-path> \
  --output-json <report-path> \
  --case <name>:<prompt-ids-path>:<max-new-tokens> \
  [--case <name>:<prompt-ids-path>:<max-new-tokens> ...] \
  [--warmup-repeats <n>] \
  [--repeats <n>] \
  [--max-ctx <tokens>] \
  [--device <cuda-device>] \
  [--eos-token-id <id>]
```

Smoke baseline minimum:

```text
cn_short, max_new_tokens >= 8, repeats >= 1
```

M3 gate baseline minimum:

```text
cn_short, max_new_tokens >= 128, warmup_repeats >= 1, repeats >= 3
long_2k, prompt_tokens >= 2048, repeats >= 1
```

## Artifacts

Raw e2e reports are local:

```text
profiles/e2e/
```

Decoded text sidecars are written beside a report:

```text
profiles/e2e/<report-stem>.decoded/
```

Profiler outputs are local:

```text
profiles/ncu/
profiles/nsys/
```

Committed official baseline summaries live under:

```text
docs/bench/baselines/
```

Large raw reports and profiler artifacts stay out of git. The committed summary must include enough
metadata to audit the readiness decision from git alone.

## Report Comparison

The planned M2.8 comparison tool is:

```text
tools/bench/compare_e2e_reports.py
```

Default hard failures include schema/artifact mismatches, missing required cases, changed fixture
identity, non-ok status, and generated token id mismatch for fixed q5090 + fixed prompt + greedy config.
Performance and memory regressions are warnings by default unless promoted by CLI flags.

## Performance Claims

A valid M3 performance claim needs both:

- local per-op/profiler evidence for the touched kernel family;
- before/after `qus_e2e_bench` JSON reports for the relevant fixture cases.

Any published number must state the command, artifact path, git commit, q5090 identity where relevant,
and dirty/clean worktree state.
