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

`qus_e2e_bench` is registered as a benchmark target. It builds with `cmake --build build -j --target
qus_e2e_bench` and remains outside default real-weight CTest execution.

## Prompt Fixtures

Canonical e2e prompts live under:

```text
bench/fixtures/prompts/
```

Each case uses a committed pair:

```text
<case>.messages.json
<case>.ids
```

`.messages.json` is the chat-message source for review and regeneration. `.ids` is the canonical C++
benchmark input: whitespace-separated decimal token ids, with no comments and no inline metadata.

The fixture set manifest is required:

```text
bench/fixtures/prompts/m2.8-v1.manifest.json
```

The manifest records tokenizer provenance, case names, prompt token counts, and hashes for each
`.messages.json`/`.ids` pair. `qus_e2e_bench` reads `.ids` only and has no tokenizer dependency.
Python fixture generation renders ids with
`tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` from a local
Qwen3.6 tokenizer path; tokenizer-level special tokens are not added separately.

Regenerate or check fixtures with:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --fixture-dir bench/fixtures/prompts

python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --fixture-dir bench/fixtures/prompts \
  --check
```

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
  --stop-token-id 248046 \
  --stop-token-id 248044
```

Minimal local error-report smoke:

```bash
printf '1 2 3\n' > /tmp/qus_e2e_smoke.ids
./build/bench/qus_e2e_bench \
  --weights /tmp/missing.qus \
  --output-json profiles/e2e/error-smoke.json \
  --case smoke:/tmp/qus_e2e_smoke.ids:1 \
  --max-ctx 3 \
  --stop-token-id 248046 \
  --stop-token-id 248044
```

The command exits nonzero and writes a schema-v1 error report when the output path is writable.

Smoke baseline minimum:

```text
cn_short, max_new_tokens >= 96, repeats >= 1, decoded clean output nonempty
```

M3 output gate baseline minimum:

```text
cn_short, en_short, code_short, math_short
max_new_tokens >= 96 for each short case
warmup_repeats >= 1, repeats >= 3
decoded clean output nonempty for each short case
```

M3 prefill gate baseline minimum:

```text
long_2k, prompt_tokens >= 2048, max_new_tokens == 1, repeats >= 1
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

Each decoded repeat writes raw and cleaned text:

```text
repeat_<n>.raw.txt
repeat_<n>.clean.txt
```

Generate decoded sidecars with:

```bash
python3 tools/bench/decode_e2e_report.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --report profiles/e2e/example.json
```

Decoded text is for human smoke review only, not an automated correctness gate.
Smoke and M3 output-gate summaries require decoded sidecars and reject empty clean decoded output.

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

Compare two local raw e2e reports with:

```bash
python3 tools/bench/compare_e2e_reports.py \
  --baseline profiles/e2e/baseline.json \
  --candidate profiles/e2e/candidate.json \
  --output-json profiles/e2e/compare.json
```

Default hard failures include schema/artifact mismatches, missing required cases, changed case identity
including fixture or generation config fields, non-ok status, changed q5090 identity when token comparison
is enabled, and generated token id mismatch for fixed q5090 + fixed prompt + greedy config.

Performance and memory regressions are warnings by default. Promote them with:

```bash
python3 tools/bench/compare_e2e_reports.py \
  --baseline profiles/e2e/baseline.json \
  --candidate profiles/e2e/candidate.json \
  --fail-on-performance-regression \
  --fail-on-memory-regression
```

Create a committed baseline summary from a local raw report with:

Use a redacted decoded manifest for committed summaries. Current decode tooling writes `tokenizer_path` as
an empty string by default, so the generated `manifest.json` can be passed directly. If you are working with
older or hand-written decoded manifests, verify that `tokenizer.tokenizer_path` is empty before summary
generation; the summary tool rejects unredacted manifests.

```bash
python3 tools/bench/make_baseline_summary.py \
  --report profiles/e2e/m3-output-gate.json \
  --output docs/bench/baselines/m3-output-gate-summary.json \
  --baseline-class m3_output_gate \
  --decoded-manifest profiles/e2e/m3-output-gate.decoded/manifest.json
```

For the long prefill gate:

```bash
python3 tools/bench/make_baseline_summary.py \
  --report profiles/e2e/m3-prefill-gate.json \
  --output docs/bench/baselines/m3-prefill-gate-summary.json \
  --baseline-class m3_prefill_gate
```

## Performance Claims

A valid M3 performance claim needs both:

- local per-op/profiler evidence for the touched kernel family;
- before/after `qus_e2e_bench` JSON reports for the relevant fixture cases.

Any published number must state the command, artifact path, git commit, q5090 identity where relevant,
and dirty/clean worktree state.
