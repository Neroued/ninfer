# tools/bench

Offline helper for the `qus_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

## Corpus baker

`qus_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, ids sha256, source
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `QUS_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Verify the committed .ids + manifest agree (hash + token count only; no tokenizer or source
# needed, so a corpus baked from a downloaded source stays CI-verifiable).
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## Current-state performance matrix

`run_qus_bench_matrix.py` runs the layered current-state `qus_bench` matrix and stores all raw
reports under ignored `profiles/bench/`. The matrix treats `--mtp-draft-tokens 3` as the primary
MTP hypothesis, keeps `k=0` and `k=5` as controls, and sweeps `k=0..5` on representative
context-decode cases.

```bash
# Inspect commands without running the model.
python3 tools/bench/run_qus_bench_matrix.py --preset core --dry-run

# Main current-state run. Builds qus_bench first, writes JSON reports and summary.csv.
python3 tools/bench/run_qus_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_qus_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_qus_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/current-state-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check.
