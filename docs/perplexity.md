# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

For single-device and dual-Spark configuration files, see [deployment examples](../examples/configs/).
The evaluator uses the [shared startup configuration](startup-configuration.md), including TOML,
command-line overrides and distributed deployment. It consumes model path, device, context limit,
KV format, Context capacity, prefill chunk, GPU timing and log level. Generation-only settings in
the same file do not apply. The Engine owns causal-scoring execution: one request, fresh window
state, no speculative decoding, CUDA Graph or cross-request prefix reuse.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/rtx5090-release/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.
The PPL entry defaults to FP8 KV; explicit TOML or command-line settings override it.

`--context` defines the metric's scoring window. `--max-context` sets the Engine capacity limit:
it defaults to the scoring window when absent from both TOML and the command line. An explicit
limit is preserved and must accommodate the scoring window. `--prefill-chunk` controls execution
inside a window and does not change the window/stride protocol.

```bash
./build/rtx5090-release/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

For distributed scoring, pass the existing deployment configuration to the same product entry.
The coordinator starts the configured workers and returns the model's complete-vocabulary scores.
For example, with a two-Spark configuration in `ninfer.local.toml`:

```bash
./build/dgx-spark-dev/apps/ninfer-perplexity --config ninfer.local.toml \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick \
  --max-context 4096 --kv-dtype fp8
```

Corpus, window, stride, quick mode and output-directory options are command-line settings;
they are not additional TOML sections. Model/deployment support is determined by the Engine's
model implementation.

Run `./build/rtx5090-release/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The report identifies the artifact's architecture, public name, actual weight formats and prefill
signature alongside the workload and numerical results. `configuration`, `config_path`, `argv`
and `environment` retain startup inputs; `execution` records the scoring window, stride and actual
Engine context limit, prefill chunk, KV format and Context capacity. `startup.ranks` contains each
rank's node/device identity, hardware, budget and resource observations;
`startup.total_allocations` contains their totals. Resource data uses the shared Engine reporting
format rather than assuming a single device.
