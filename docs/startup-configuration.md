# Startup configuration

`ninfer`, `ninfer-serve`, `ninfer_bench` and `ninfer-perplexity` share TOML 1.0 configuration and
command-line overrides. Use [deployment examples](../examples/configs/) for a starting configuration;
[ninfer.example.toml](../ninfer.example.toml) lists every field.

```bash
cp examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml ninfer.local.toml
# Edit model.path.
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
./build/rtx5090-release/apps/ninfer --config ninfer.local.toml --prompt "Hello" --max-tokens 1024
```

Configuration is printed at INFO level before loading, with file/command-line/default sources.
After preparation, the log reports resolved resources and rank information. API keys are redacted.
Personal TOML files are ignored by Git; the root reference and `examples/configs/*.toml` are tracked.

## Sources and paths

Settings resolve in this order: entry defaults, file, explicit command-line values. Unspecified
sampling and template options use model defaults. Serve request fields override the server's
generation defaults. CLI makes one request; it retains explicit cache/concurrency settings from
the file even though its own default profile is smaller.

| Default | CLI | Serve |
|---|---|---|
| Per-request context limit | 8192 | 8192 |
| Shared Device Context capacity | auto | auto |
| Active requests | 1 | 1 |
| Output limit | 128 | 8192; requests can override |
| Prefix reuse | false | true |
| Host context capacity | 0 B | RTX 5090: 8 GiB + eight native StateImages; Spark: 0 B |
| Seed | 0 | Random per request |

[PPL](perplexity.md) consumes model path, device, max context, KV type, Context capacity, memory
headroom, prefill chunk, GPU timing, log level and distributed deployment. Its default KV is FP8;
its Engine context limit follows the scoring window unless explicitly set. Generation settings
do not affect scoring.
The [inference benchmark](../bench/README.md#engine-微测) uses the shared launch options and its own
measurement controls.

File paths in TOML are relative to the config file. Command-line paths are relative to the working
directory. Media paths inside a CLI messages file are relative to that messages file. Distributed
worker/artifact paths are absolute paths on each node. Strings do not expand `~`, environment
variables or shell expressions.

Unknown keys and wrong types are errors. Fields for another entry are parsed but do not enable
that entry's behavior: for example, `[cli]` does not inject messages into Serve. Unused input paths
are not opened.

### Values and overrides

- Capacities accept integer bytes or strings with `B`, `KiB`, `MiB`, `GiB`; durations accept integer
  milliseconds or `ms`/`s` strings. Fractional units must resolve to whole bytes/milliseconds.
- `model` clears a sampling/template override. `default` restores the entry-specific default where
  the field allows it. `auto`, `none` and `random` have the field-specific meanings listed below.
- Zero, false and empty lists are explicit values. Omission keeps the inherited value.
- Specify at most one `--config` and one positional model path. The positional path overrides
  `model.path`. The position of `--config` among arguments does not affect precedence.
- For repeated scalar flags, the last value wins. TOML itself rejects duplicate keys.
- Boolean flags have positive/negative forms and accept `--flag=true|false`. Optional booleans also
  accept the `model`/`default` values listed below.
- The first command-line stop-list flag replaces the file's corresponding list; subsequent flags
  append. `--clear-stops` clears all three CLI stop lists.
- `--prompt` and `--messages` are alternatives. A command-line choice replaces the file's input
  choice. Likewise a command-line output constraint replaces the complete file constraint.
- `--spec none` clears inherited draft/head settings. Explicitly pairing `none` with nonzero drafts
  or an optimized head is an error.
- Thinking and CLI reasoning effort are merged together; contradictory explicit values at the same
  level are rejected. Supported efforts and thinking modes depend on the model.

`--help` / `-h` prints help without reading a config or starting the Engine.

## Model and execution

| TOML field | CLI flag | Values / default | Meaning |
|---|---|---|---|
| `model.path` | positional path | Required path | v3 artifact entry file |
| `model.chat_template` | `--chat-template` | Path / `""` | Override the stored Jinja template; empty uses the artifact |
| `model.vision` | `--vision` / `--no-vision` | bool / false | Load Vision and enable image/video requests |
| `engine.device` | `--device` | Nonnegative int / 0 | Local device; distributed ranks declare devices separately |
| `engine.max_context` | `--max-context` | Positive uint32 / 8192 | Per-request input plus generation limit |
| `engine.kv_dtype` | `--kv-dtype` | `bf16/int8/fp8/nvfp4/k8v4` / `bf16` | Runtime KV representation; GLM latent KV supports BF16 and FP8 |
| `engine.context_capacity` | `--context-capacity` | Positive bytes or `auto` / `auto` | Shared Device KV/State capacity per rank |
| `engine.memory_headroom` | `--memory-headroom` | Nonnegative bytes / `512MiB` | Per-rank startup planning allowance; `0B` adds no headroom |
| `engine.prefill_chunk` | `--prefill-chunk` | Positive multiple of 128 or `default` | Qwen4Exp, GLM5Next and CPU expert offload default to 4096; other paths to 1024; bounded by max context |
| `engine.cuda_graph` | `--cuda-graph` / `--no-cuda-graph` | bool / true | CUDA Graph decode; prefill and Vision encoding remain eager |
| `engine.max_concurrency` | `--max-concurrency` | 1..8 / 1 | Maximum active request lanes, across the deployment |
| `engine.max_pending_requests` | `--max-pending-requests` | Positive uint32 / 16 | Additional waiting requests |
| `engine.pending_timeout` | `--pending-timeout` | Positive duration / `30s` | Deadline before execution; not a generation time limit |

Weight formats are stored in the artifact and are independent of `kv_dtype`. Optional components
must be present when enabled. Supported model/deployment combinations are listed in the
[README](../README.md#supported-models-and-deployments).

## Context and memory

`max_context` is a logical limit. `context_capacity` is a physical byte budget shared by KV histories
and continuation state. With `auto`, startup plans weights, workspace, Graph/communication storage,
Vision and other required resources before selecting Context capacity. Explicit capacity, such as
`"8GiB"`, requests that many bytes per rank. Both forms must accommodate the model's minimum working
set and fit the available budget.

`memory_headroom` applies to both automatic and explicit Context capacity. Each rank deducts it
once from its current planning allowance, after accounting for the device budget and mandatory
Host capacity. Startup allocations may consume this allowance; it is not a required amount of
free memory after initialization. For example, `--memory-headroom 256MiB` overrides the TOML value.
Setting `0B` removes only this allowance: weights, Graph/communication storage, workspace and
other required resources still count, and actual allocations must fit the device budget.

The backing is fixed at startup; KV and State allocations within it follow demand. Prefixes share
history where possible. Pressure can reclaim retained prefixes or pause requests for snapshot/replay
recovery. The startup **token-capacity estimate** translates the pool into a useful context scale
while accounting for active state. It is an estimate for the selected model/KV/spec configuration,
not an independent reservation or a per-request guarantee. Context bytes are reported per rank; the
token estimate accounts for the deployment's
TP/attention-owner placement.

| TOML field | CLI flag | Values / default | Meaning |
|---|---|---|---|
| `context_cache.prefix_reuse` | `--prefix-reuse` / `--no-prefix-reuse` | bool or `default` | Cross-request prefix lookup and retention; CLI false, Serve true |
| `context_cache.checkpoint_interval` | `--checkpoint-interval` | Positive uint32 / 8192 | Target spacing of intermediate prefill checkpoints; execution and semantic boundaries determine exact positions |
| `context_cache.host_capacity` | `--host-cache-capacity` | Nonnegative bytes or `default` | Shared pinned Host context budget on RTX 5090; defaults in the table above |
| `context_cache.cost_presets` | `--context-cost-presets` | Path / `""` | Optional context-cost JSON for reuse/restoration decisions |
| `vision.memory_budget` | `--vision-memory-budget` | Positive bytes / `1GiB` | Per-rank Vision encoder activations and output window; excludes weights and Host staging |
| `media.cache_capacity` | `--media-cache-capacity` | Nonnegative bytes / `1GiB` | Retained prepared-media cache; zero disables retention |
| `media.live_capacity` | `--media-live-capacity` | Positive bytes / `2GiB` | All live prepared patch payloads, including retained media counted once |
| `media.preprocess_threads` | `--media-preprocess-threads` | 1..64 or `auto` / `auto` | CPU media preprocessing; auto uses at most 16 workers |

Host context holds State, KV, paused snapshots and in-flight destinations. It excludes CPU expert
weights, PLE, media and other process memory. Spark uses Device context and requires Host context
capacity to be zero; shared-memory planning also accounts for Host resources such as live media.
Disabling prefix reuse does not disable pause/recovery or release the configured backing.

Vision is selected at startup. Its bounded execution window processes media before the corresponding
text segment consumes the features. The media budgets cover CPU preprocessing/cache storage and
are separate from the GPU Vision budget. Processor-specific media limits still apply.

## Speculative decoding

| TOML field | CLI flag | Values / default | Meaning |
|---|---|---|---|
| `speculative.backend` | `--spec` | `none/mtp/dflash/dflash2` / `none` | One backend per Engine |
| `speculative.draft_tokens` | `--draft-tokens` | 0 when none; MTP 1..5; DFlash/DFlash2 1..15 | Number of proposed tokens |
| `speculative.proposal_head` | `--proposal-head` | `full/optimized` / `full` | Draft head; requires a backend and the selected weights |

The [CLI guide](cli.md#speculative-decoding) describes model-specific backends and optimized heads.

## Generation defaults

These fields form the CLI request or the Serve defaults. HTTP requests can override Serve defaults.
Absent sampling fields retain the model preset selected for the request's thinking mode.

| TOML field | CLI flag | Values / default |
|---|---|---|
| `generation.max_tokens` | `--max-tokens` | Positive uint32 or `default`; CLI 128, Serve 8192 |
| `generation.thinking` | `--thinking` / `--no-thinking` | bool or `model` / `model` |
| `generation.preserve_thinking` | `--preserve-thinking` / `--no-preserve-thinking` | bool or `model` / `model` |
| `generation.thinking_budget` | `--thinking-budget` | Positive uint32 or `none` / `none` |
| `generation.sampling.temperature` | `--temperature` | [0,2] or `model`; zero selects greedy |
| `generation.sampling.top_k` | `--top-k` | Integer [0,20] or `model`; zero uses the current limit of 20 |
| `generation.sampling.top_p` | `--top-p` | [0,1] or `model` |
| `generation.sampling.min_p` | `--min-p` | [0,1] or `model` |
| `generation.sampling.presence_penalty` | `--presence-penalty` | [-2,2] or `model` |
| `generation.sampling.frequency_penalty` | `--frequency-penalty` | [-2,2] or `model` |
| `generation.sampling.seed` | `--seed` | uint64, `random` or `default`; CLI 0, Serve random per request |

TOML seeds above INT64_MAX use decimal strings, for example `"18446744073709551615"`.
Output limits include accepted model tokens and inserted thinking-control tokens.
A thinking budget limits model-origin thinking, with model-specific closing behavior; see
[thinking and sampling](cli.md#thinking-and-sampling). Serve can retain a default budget for requests
that enable thinking even if its default mode is disabled. CLI rejects an explicitly disabled mode
paired with a positive budget.

## CLI input and output

| TOML field | CLI flag | Values / default |
|---|---|---|
| `cli.prompt` | `--prompt` | Nonempty string; alternative to messages |
| `cli.messages` | `--messages` | JSON file path; alternative to prompt |
| `cli.reasoning_effort` | `--reasoning-effort` | `model/none/minimal/low/medium/high/xhigh/max`; model decides supported values |
| `cli.raw_output` | `--raw-output` / `--no-raw-output` | bool / false |
| `cli.print_token_ids` | `--print-token-ids` / `--no-print-token-ids` | bool / false |
| `cli.stop_token_ids` | `--stop-token-id` (repeatable) | Nonnegative int32 array / [] |
| `cli.stop` | `--stop` (repeatable) | Nonempty content strings / [] |
| `cli.reasoning_stop` | `--reasoning-stop` (repeatable) | Nonempty reasoning strings / [] |
| `cli.constraint.kind` | `--constraint none` or a flag below | `none/grammar/json_object/json_schema/regex/choice` / `none` |
| `cli.constraint.grammar_file` | `--grammar-file` | Path for `grammar`; entry rule `root` |
| `cli.constraint.json_schema_file` | `--json-schema-file` | Path for `json_schema` |
| `cli.constraint.regex` | `--regex` | String for `regex`; empty permits only empty content |
| `cli.constraint.choices` | `--choice` (repeatable) | Nonempty array for `choice`; an entry may be empty |

`--json-object` selects `kind = "json_object"` without another field. Set only fields for the selected
kind. `--constraint none` clears a file constraint. See [CLI constraints](cli.md#constrained-output)
for examples and interactions with tools and stops.

## HTTP and observability

| TOML field | CLI flag | Values / default | Meaning |
|---|---|---|---|
| `server.host` | `--host` | Nonempty string / `127.0.0.1` | HTTP listen address |
| `server.port` | `--port` | 1..65535 / 8080 | HTTP port |
| `server.api_key` | `--api-key` | String / `""` | Empty disables authentication |
| `server.model_id` | `--model-id` | String / `""` | Public alias; empty uses artifact metadata |
| `server.cors` | `--cors` / `--no-cors` | bool / false | Cross-origin responses |
| `server.max_request_size` | `--max-request-size` | Positive bytes / `384MiB` | HTTP body limit |
| `server.responses.max_records` | `--response-store-max-records` | Positive uint64 / 1024 | Process-local Responses history count |
| `server.responses.max_size` | `--response-store-max-size` | Positive bytes / `256MiB` | Responses history size, separate from Context |
| `logging.level` | `--log-level` | `trace/debug/info/warning/error/critical/off` / `info` | Diagnostic verbosity |
| `logging.stats_interval` | `--stats-interval` | Nonnegative duration / `5s` | Serve aggregate log interval; zero disables it |
| `logging.request_jsonl` | `--request-log-jsonl` | Path / `""` | Serve structured request log; empty disables it |
| `observability.gpu_timing` | `--gpu-timing` / `--no-gpu-timing` | bool / false | CUDA stream timing at existing completion boundaries |

See [Serve metrics](serving.md#live-metrics) and [structured logs](serving.md#structured-request-log)
for observations and their units. CUDA stream durations are separate from request wall times.

## Distributed deployment

Use a [dual-Spark example](../examples/configs/README.md#two-sparks). Start CLI or Serve once on the
coordinator node. It starts local workers directly and remote workers through noninteractive SSH,
passes each worker its rank and deployment, and owns worker shutdown. Clients connect to one HTTP
server; workers are internal execution processes.

Both nodes need matching builds with `NINFER_ENABLE_DISTRIBUTED=ON`, NCCL, the artifact entry and all
its parts. `model.path` is visible to the coordinator; each node's `artifact` is its local copy of
the same model. Keep each worker's build `lib/` alongside `apps/`.

Distributed settings are TOML fields; there are no `--tp`/`--ep` command-line aliases.

| Field | Value / default | Meaning |
|---|---|---|
| `distributed.coordinator_node` | Node ID / 0 | Node where CLI/Serve runs |
| `distributed.control_host` | Required nonempty address | Coordinator address reachable by all workers |
| `distributed.control_port` | 0..65535 / 0 | Control listener; zero selects a free port |
| `distributed.connect_timeout` | Positive duration / `60s` | Worker connection timeout |
| `distributed.shutdown_timeout` | Positive duration / `30s` | Worker shutdown timeout |
| `distributed.parallel.tp` | Positive integer / 1 | Tensor parallel size |
| `distributed.parallel.ep` | Positive integer / 1 | Expert parallel size; 1 with TP2 selects expert TP |
| `distributed.parallel.adp` | Positive integer / 1 | Attention DP within the execution group |
| `distributed.parallel.dp` | Positive integer / 1 | Independent model replicas; current models require 1 |
| `distributed.parallel.pp` | Positive integer / 1 | Pipeline parallel size; current models require 1 |
| `distributed.nodes[].id` | Required unique ID | Node identity |
| `distributed.nodes[].ssh` | SSH destination | Required for remote nodes; omitted for the coordinator node |
| `distributed.nodes[].worker` | Required absolute path | Product `ninfer-worker` on that node |
| `distributed.nodes[].artifact` | Required absolute path | Artifact entry on that node |
| `distributed.nodes[].environment` | String-to-string table / empty | Worker environment, including NCCL network selection |
| `distributed.ranks[].rank` | Required unique integer | Global rank; contiguous from 0 |
| `distributed.ranks[].node` | Required node ID | Hosting node |
| `distributed.ranks[].device` | Nonnegative integer / 0 | CUDA ordinal on that node |

Current dual-Spark placements:

| Model | TP | EP | ADP |
|---|---:|---:|---:|
| Flash Next | 2 | 1 or 2 | 1 |
| GLM5Next | 2 | 1 or 2 | 2 |

With `ep = 1`, each rank holds the TP shard of every expert. With `ep = 2`, ranks own different
whole experts while other TP partitions remain. GLM attention DP places each request's MLA history
on an attention owner within the same model execution group; it is not an independent model replica.
The concurrency setting is the group-wide active-request limit.

NCCL handles GPU collectives. `control_host` selects the control listener, not the NCCL data path.
Use per-node `environment` entries to select the intended network/HCA; the examples show their
placement. Startup logs identify rank/node phases and actual resources. One-device execution needs
no `[distributed]` section.

## Environment controls

| Variable | Behavior |
|---|---|
| `NINFER_VRAM_BUDGET_MIB` | Positive integer MiB cap on the Engine's device allocation budget; useful for RTX 5090 expert offload |
| `NINFER_CUDA_SYNC` | `spin` (default), `blocking`, `yield`, or `auto`; CUDA host-wait scheduling |
| `NINFER_KPOOL_INDEXER_BF16` | GLM indexer uses FP8 by default; `1` selects BF16 at startup, independently of latent `kv_dtype` |

`spin` prioritizes host wakeup latency; `blocking` sleeps while waiting. `auto` selects CUDA's
scheduling heuristic. These environment controls are separate from TOML. Set the GLM indexer override
on the coordinator before launch; the selected format is passed to every rank. CUDA synchronization,
VRAM budget and `NCCL_*` environment settings are forwarded to workers; per-node environment entries
override the forwarded values.
