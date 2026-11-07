# Agentic replay (capy-flash benchmark)

The replay behind the speed table of the main README: 3 synthetic coding-agent sessions (`sessions.json`, 53 turns,
system prompt 8.6k tokens, prompts growing to ~81k tokens, ~3.8k new tokens per turn), each turn sent as a full
chat request with the whole history, streamed, temperature 0, 256 output tokens. A server with a prefix cache
re-reads only the new part of each turn. It measures what an agent feels: time to first token, decode speed and
total time.

```
# 1. start the server under test (capy-flash below, see the main README for the build and the artifact)
CUDA_VISIBLE_DEVICES=0,1 build/apps/ninfer-serve <artifact>.ninfer --profile flash-next --host 127.0.0.1 \
  --port 8098 --model-id capy-flash --greedy --request-log-jsonl reqlog.jsonl
#    (one GPU, 64 GB edition: --profile flash-next-1gpu; the profiles are listed in docs/serving.md#profiles)
# 2. replay, then summarize
python3 bench/capy_flash/replay.py http://127.0.0.1:8098 capy-flash bench/capy_flash/sessions.json out.jsonl 256
python3 bench/capy_flash/summarize.py out.jsonl --ninfer-request-log reqlog.jsonl
```

Compare engines with the **server-side** decode rate: a server that streams tokens in bursts makes the client-side
rate look higher than it is. For Strata, start its OpenAI server (`python -m serve.server --engine strata --config
<config> --port <port>`) with the same model id in the replay and pass its engine log to `summarize.py
--strata-engine-log`. The published Strata 0.1.44.1 runs used two GPUs (layer split auto, `--pipeline-windows 2`,
`--pcie-frac 0.2` for IQ3_S, its NVFP4 default otherwise), `--max-context 131072`, int8 KV, MTP (`--spec 4
--spec-min-p 0.5`), greedy; for the NVFP4 pack `STRATA_NO_ARENA_THP=1` (without it the 63 GiB expert arena is
requested with transparent huge pages and, on a machine with THP defrag `madvise`, compaction stalls the load).

`gen_sessions.py` documents how `sessions.json` was built (from a source tree and the model tokenizer); the published
numbers all used the `sessions.json` committed here.

## Long-context decode and greedy identity (`long_context.py`)

Decode speed at fixed context sizes, one fresh conversation per size (a distinct slice of a long text cut to the
size with the model tokenizer, then a summarization request, temperature 0), and a greedy probe whose output files
compare byte for byte between two configurations of the same binary (for example a placement option on and off,
with `NINFER_QWEN4_VALIDATION=1 NINFER_QWEN4_EXPERT_FREEZE=all` so that expert residency does not differ). Used for
active KV in RAM (`--kv-ram-active`, [design](../../docs/maintainer/active-kv-in-ram.md)).

```
TOK=<model dir>/tokenizer.json; WIKI=<wikitext-2-raw>
python3 bench/capy_flash/long_context.py decode http://127.0.0.1:8098 capy-flash lc.jsonl --text $WIKI/wiki.train.raw \
  --tokenizer $TOK --sizes 16000,64000,128000,250000 --max-tokens 512 --server-log serve.log
python3 bench/capy_flash/long_context.py summarize reqlog.jsonl lc.jsonl
python3 bench/capy_flash/long_context.py greedy http://127.0.0.1:8098 capy-flash greedy.jsonl --text $WIKI/wiki.valid.raw \
  --tokenizer $TOK --sizes 1000,3000,12000,30000,60000
```

`summarize` reads the server-side decode rate (completion tokens / decode seconds) from the request log and, with
the probe output, the expert slots per layer per GPU and the KV bytes per GPU of an `--elastic` server. Each `decode`
request carries an `X-Ninfer-Client` label unique to its run and size, which the probe records and the request log
keeps: `summarize` pairs a request with its probe by that label, so other requests in the log and earlier runs
appended to the probe file do not shift the pairing.

SSE errors or missing `finish_reason`/`[DONE]` are recorded in `error`; the next size still runs.
`decode` uses monotonic response timings; `wall_s` excludes the post-response wait for server logs.

## Portable ladder with a recorded environment (`ladder.py`, issue #30)

One command to produce a row that is comparable with other people's machines, without downloading a
corpus or a tokenizer. It runs a fixed set of context sizes against a running server, each with a
deterministic self-contained prompt, and reports — as the median of a few passes — the cold-prefill
and decode rates from the server's own `timings` block. Alongside the numbers it records the
machine (GPUs, driver, PCIe link width, WSL2, CPU, RAM) and, when given the server's startup log,
the P2P state and the expanded launch line the server printed, since those decide what the numbers
mean.

```
# start the server keeping its log, then:
python3 bench/capy_flash/ladder.py http://127.0.0.1:8098 capy-flash result.json \
  --server-log server.log --note "2x RTX 5070 Ti, bare metal"      # default sizes 8k-128k
python3 bench/capy_flash/check_result.py result.json               # validate before sharing
```

Defaults: sizes `8000,16000,32000,64000,128000`, 3 passes, 256 decode tokens. The cold-prefill leg
uses a different prompt each pass so it never hits the prefix cache. If the server is started with
`--api-key`, set `CAPY_API_KEY` for the client. `check_result.py` refuses a row that is missing the
environment that makes it comparable or whose prompt-token counts do not match the targets — the
gate for a shared results log.
