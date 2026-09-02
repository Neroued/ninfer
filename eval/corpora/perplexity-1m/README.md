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
mode selects stream `00` from every domain and is roughly a quarter of that size. `manifest.json`
records the stream order and paths; `provenance/*.jsonl` maps every output stream back to source
records or file byte ranges. See `THIRD_PARTY_NOTICES.md` before redistributing the corpus.

The streams were initially sized with the Qwen3.6 tokenizer, without special tokens, only to keep
their workloads in the intended rough range. Tokenizer identity and exact token counts are not part
of the corpus contract. Every evaluation tokenizes the committed text with its artifact's tokenizer
and reports the resulting counts. A changed corpus uses a new corpus identity.
