# NInfer Perplexity Corpus

`ninfer-ppl-1m-v1` is the fixed text corpus for NInfer causal-perplexity evaluation. It contains
16 independent UTF-8 streams from four domains. The streams are not concatenated during scoring;
the first token of each stream is the only token without a causal score.

| Domain | Source | Streams |
|---|---|---:|
| `english_reference` | WikiText-2 raw test | 4 |
| `english_long_form` | PG-19 test books | 4 |
| `chinese_reference` | Chinese Wikipedia | 4 |
| `ninfer_code` | NInfer production C++/CUDA source | 4 |

The `full` mode contains all 16 streams and is roughly a one-million-token workload. The `quick`
mode selects stream `00` from every domain and is roughly a quarter of that size. Exact bytes,
order, and SHA-256 values are recorded in `manifest.json`. `provenance/*.jsonl` maps every output
stream back to source records or file byte ranges. See `THIRD_PARTY_NOTICES.md` before redistributing
the corpus.

The streams were initially sized with the Qwen3.6 tokenizer, without special tokens, only to keep
their workloads in the intended rough range. `provenance/generation.json` records that build-time
input and its digest. Tokenizer identity and exact token counts are not part of the corpus contract.
Every evaluation tokenizes the fixed text with its artifact's tokenizer and reports the resulting
counts.

## Verification

```bash
python3 tools/perplexity/prepare_corpus.py \
  --check
```

Verification is offline and tokenizer-independent. It checks UTF-8, file bytes and hashes,
provenance byte coverage, fixed stream order, and the `quick`/`full` definitions.

## Rebuilding

Generation requires explicit local copies of the pinned upstream inputs and performs no network
access:

```bash
python3 tools/perplexity/prepare_corpus.py \
  --tokenizer /path/to/Qwen3.6-27B \
  --wikitext-snapshot /path/to/Salesforce-wikitext-snapshot \
  --pg19-snapshot /path/to/deepmind-pg19-metadata-snapshot \
  --pg19-assets /path/to/pg19-assets \
  --wikipedia-snapshot /path/to/wikimedia-wikipedia-snapshot \
  --ninfer-repo /path/to/ninfer \
  --output /empty/output/directory
```

The tool validates every fixed upstream input against its expected digest before producing output.
Changing any source, selection rule, stream boundary, or prepared byte requires a new corpus
identity; `ninfer-ppl-1m-v1` must not be silently regenerated with different content.
