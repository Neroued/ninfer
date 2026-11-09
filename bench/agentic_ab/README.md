# Agentic A/B benchmark

A closed-loop replay of agentic coding traffic (three interleaved sessions, eleven subagents,
compaction, retries, an aborted request; 130 requests per seed) against two `ninfer-serve` builds
on the same model and GPUs. Metrics come from each serve's own `--request-log-jsonl`.

**Origin and credits.** The suite was written for
[Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom) (commit `535fe587`,
2026-09-26, author Ian Ranson), a fork of Neroued/ninfer under the Apache License 2.0, to compare
that fork with upstream on Windows. Its description follows [below](#the-original-suite-windows-upstream--windows-port-vs-wallawalla47ninfer-custom),
unchanged. This tree adds a Linux launcher and small analyzer changes; `NOTICE` records the
provenance.

| File | Status here |
|---|---|
| `workload.py` | Unchanged (one docstring paragraph). The observations still come from ninfer-custom at `e48a0d28`, so they are byte-identical to the original suite's. |
| `runner.py` | Client unchanged; it now also records timestamped `workload_start`, `signal` and `workload_end` events. Its Windows orchestration is kept but not used here. |
| `analyze.py` | Same metrics. New: arms named in `config.json` (baseline first, with labels), a cross-check of the serve's `req#N done` log lines against the request log, a phase table, retry/abort/failure counts, a seed-to-seed spread table in the combined report. Windows runs render as before. |
| `linux.py`, `run_linux.sh` | New: the Linux launcher. |
| `build_control.bat` | Unchanged; Windows only. |

## Running it on Linux

```bash
bench/agentic_ab/run_linux.sh \
  --arm-a /path/to/build-a/apps/ninfer-serve --label-a A \
  --arm-b /path/to/build-b/apps/ninfer-serve --label-b B \
  --model /path/to/model.ninfer \
  --profile agentic --seeds 42,43,44 --out /path/to/out
```

Requirements: Python 3.8 or newer (standard library only), `git`, `nvidia-smi`, a systemd user
session (`systemd-run --user`), and a clone of ninfer-custom for the corpus:

```bash
git clone --bare https://github.com/Wallawalla47/ninfer-custom /path/ninfer-custom-corpus.git
export AB_CORPUS_REPO=/path/ninfer-custom-corpus.git   # AB_CORPUS_COMMIT defaults to e48a0d28
```

A full run takes hours; start it in its own unit and follow `runner.log`:

```bash
systemd-run --user --unit=agab-run --collect -p WorkingDirectory=$PWD \
  bash -c 'bench/agentic_ab/run_linux.sh ... > run.log 2>&1'
```

For each seed, in ABBA order (the first seed runs A then B, the second B then A, and so on), each
arm:

1. waits until the GPUs are free: no CUDA compute process and no active user unit whose name
   contains a `--wait-units` fragment (default `tune-`), polling every 2 min for at most
   `--gpu-wait-s` (3 h);
2. starts `<exe> <artifact> --host 127.0.0.1 --port 8091 --model-id qwen27b <profile flags>
   --request-log-jsonl <arm>/request_log.jsonl` as the transient unit `agab-a` or `agab-b`, with
   stdout and stderr in `<arm>/serve.log`, and waits for `/health`;
3. sends one priming request (no workload seed, never analysed), then runs the workload;
4. stops the unit and writes `arm.json`: wall time, client failures, load time, the build
   (sha256 of the executable, `git describe` of its checkout), GPU clocks before and after, and
   any other CUDA process seen while the arm ran (sampled every 30 s; flagged in the report).

Then `analyze.py` writes the seed's `report.md` and `summary.json`; after the last seed the
combined `report.md` goes to the output directory. `--resume` skips arms whose `arm.json` is
complete, to continue an interrupted run. `--dry-run` prints the plans, the order, the full serve
command and the GPU state, and starts nothing.

| Option | Default | Meaning |
|---|---|---|
| `--arm-a`, `--arm-b` | required | the two `ninfer-serve` executables; A is the baseline |
| `--label-a`, `--label-b` | `git describe` of the build's checkout | names in the report |
| `--model` | required | the `.ninfer` artifact, the same for both arms |
| `--profile` | `agentic` | named flag set (below) |
| `--flags`, `--extra-flags` | | replace or extend the profile |
| `--seeds` | `42,43,44` | workload seeds; each replays different observations |
| `--order` | `abba` | `ab` runs A first on every seed |
| `--scale` | `1.0` | stretches the session loops (0.3 is a smoke run) |
| `--port`, `--host`, `--model-id` | `8091`, `127.0.0.1`, `qwen27b` | serve endpoint |
| `--unit-prefix` | `agab` | units `<prefix>-<arm>` (`<prefix>-a`, `<prefix>-b` with `--arm-a/--arm-b`) |
| `--load-timeout`, `--request-timeout` | 900 s, 3600 s | `/health` wait; client socket timeout per request |
| `--agent-max-tokens` | 64000 (or `AB_AGENT_MAX_TOKENS`) | `max_tokens` the client sends on every agent turn; written to `config.json` and `arm.json`, and per request to `client.jsonl` |
| `--arms-file` | | JSON list of arms, baseline first, replacing `--arm-a`/`--arm-b` (below) |
| `--arm-order` | | comma-separated arm names to run on every seed, in that order (default: all arms, forward on even seed positions and reversed on odd ones) |
| `--keep-going` | off | an arm that fails to start or run is recorded in its `arm.json` and the next arm runs |

**More than two arms, or arms that differ.** `--arms-file` takes a list such as

```json
[{"name": "prod", "label": "v0.4.6", "exe": "/path/ninfer-serve", "flags": "--tp 2 ... --max-context 262144 --kv-capacity 327680"},
 {"name": "kv360", "exe": "/path/ninfer-serve", "flags": "... --kv-capacity 360448"},
 {"name": "mt16k", "exe": "/path/ninfer-serve", "flags": "...", "agent_max_tokens": 16000}]
```

`flags` is the full flag string of that arm (default: the profile plus `--extra-flags`), so arms may
differ in context, KV capacity or any other flag; the client's context guard follows each arm's own
`--max-context`. `agent_max_tokens` overrides `--agent-max-tokens` for that arm. Units are
`<prefix>-<name>`. Every arm is compared with the first one. A seed may be run arm by arm
(`--seeds 42 --arm-order kv360 --resume`): `config.json` keeps the arms and times of the earlier
invocations, and the seed's report is written as soon as the baseline and one more arm are complete.

Profiles (both arms always get the same flags):

| Profile | Flags |
|---|---|
| `prod` | the production launch line: `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 --spec mtp --draft-tokens 3 --vision --vision-device 0 --max-vision-tokens 4096`, plus `--lm-head-draft` |
| `agentic` | `prod` with `--max-concurrency 4 --device-state-slots 8`, so subagents and sessions decode together when their KV reservations fit |

Every profile also gets `--pending-timeout-ms 3600000`, above the 10-minute default: a request
that waits for admission behind long decodes must not be rejected, and the workload keeps up to
eight requests in flight.

Output layout: `<out>/runner.log`, `<out>/report.md` (combined), and per seed
`<out>/seed-<n>/{plan.json,config.json,report.md,summary.json}` with `a/` and `b/` each holding
`client.jsonl`, `request_log.jsonl`, `serve.log` and `arm.json`.

### What differs from the Windows runner

| Windows runner (`runner.py`) | Linux launcher (`linux.py`) |
|---|---|
| treatment = this fork, control = upstream + Windows port, optional alt arm | arm A and arm B, any two builds |
| flags read from a deploy `.bat`, control drops flags its `--help` lacks | one explicit flag profile for both arms |
| `--max-context` calibrated on the control; `--host-cache-mib` translated into the control's explicit host-cache flags | `--max-context` from the profile; no translation (at `--tp 2` the Host tiers are off in both arms) |
| arms in a fixed order on every seed | ABBA over the seeds |
| `Popen` + `taskkill`, readiness on `/v1/models`, GPU idle = memory below 6 GB | systemd user unit, readiness on `/health`, GPU idle = no compute process and no busy unit |
| build hash from the serve log | sha256 of the executable and `git describe` of its checkout |

### What the report measures here

The serve's request log on this tree has the schema `analyze.py` reads (`schema_version` 21:
`request_done` with `result`, `timings_seconds`, `engine_timing.queue_wait_seconds`,
`speculative`; `throughput` with `decode_batch`, `host_work.work_class_seconds`, `tokens`,
`context_cache.pressure`; `server_start` with `engine` and `memory`), so every metric of the
original suite is computed the same way. Not available: ngram drafting (`ngram_*` fields; these
builds have none; the report says so), the Host cache tiers (0 at `--tp 2`), and a build hash in
the serve log (taken from `arm.json` instead). The serve's human-readable `req#N done` lines
(prompt, output, cache, TTFT, decode, `mtp accepted a/b`) are parsed and cross-checked against
the request log for every request.

New measurements: monotonic output/tool latency, n/median/p90/max, thinking and parser diagnostics.
**What three seeds resolve.** The first run on this tree (v0.1.1 against v0.2.0, two RTX 5070 Ti,
`--profile agentic`, seeds 42-44, about 47 min per arm) resolved a 1-2 % difference only in the
engine rows: prefill tok/s on the requests with no cache hit in either arm (+1.2 to +2.0 % per
seed) and decode rounds/s with one request decoding (+1.0 to +3.2 %). Wall time, TTFT and output
tok/s moved by -8 to +11 % between seeds with the same builds, because the sampled output volume
(completion tokens -18 to +28 %) and the resulting cache evictions change with every run. With a
196,608-token KV pool and no Host tier, the three sessions of this workload do not all stay
cached: 31-33 % of prompt tokens came from the cache, and about 40 of the 75 continuing
main-session turns prefilled their whole prompt again.

---

## The original suite (Windows): upstream + Windows port vs Wallawalla47/ninfer-custom

A black-box A/B benchmark of two `ninfer-serve` builds serving the same model on the same GPU,
driven by a closed-loop replay of real agentic coding traffic over the OpenAI chat-completions
API. The fork runs its default hybrid prefix cache; an optional third arm runs the fork build with
the original prefix cache (`--use-original-prefix-caching`). It produces a
README-style comparison table covering:

- **prefix-cache hits** (tokens served from cache, continuing turns that had to re-prefill);
- **time to first token**, split into continuing-session turns (cache retention) and new long
  prompts (prefill speed);
- **raw prefill tok/s** on requests that had no cache hit in any arm;
- **output tok/s** per second of engine decode time, with one request decoding, two requests
  decoding, and at the run's own batching, each with a 95 % interval, split into engine speed
  (decode rounds/s) and speculative acceptance (tokens per round).

Metrics come from each serve's own request log (`--request-log-jsonl`); the client only supplies
the request classification.

### Files

| File | Purpose |
|---|---|
| `workload.py` | Deterministic scenario: personas, observations and session timeline (`python workload.py` prints a summary). |
| `runner.py` | Calibrates a common `--max-context`, runs the treatment, the optional alt arm, then the control, drives the workload, calls the analyzer. |
| `analyze.py` | `python analyze.py <run_dir>`: joins client and serve logs, writes `report.md` and `summary.json`; `--aggregate` combines several seeds' runs. |
| `build_control.bat` | Windows build of the control's `ninfer-serve` from a control checkout. |

Each run writes to `profiles/bench/agentic_ab/<timestamp>/` (gitignored): `runner.log`,
`config.json`, `plan.json`, `report.md`, `summary.json`, and per arm `client.jsonl`,
`request_log.jsonl`, `serve.log`, `arm.json`. A run over several seeds has one such directory per
seed (`seed-<n>/`) and the combined `report.md` above them.

### What the workload models

The shape comes from the deploy folder's production request logs (~3,300 requests from Qwen Code
and Claude Code style clients): prompts of 30K-210K tokens (median ~100K) that grow by a few
hundred to a few thousand tokens per turn, short tool-calling answers (median ~380 tokens,
p90 ~4,000) with thinking on, `max_tokens: 64000` on every agent turn, periodic compaction,
subagent fan-outs, whole-history side calls, retries, aborted requests, and several sessions
sharing one serve with up to eight requests in flight.

One run replays three interleaved sessions plus eleven subagents (130 requests at scale 1.0):

| Actor | Persona | What it does |
|---|---|---|
| A | Claude-Code-like (20 tools, ~13K-token system+tools) | Resumes a ~85K-token C++ session (cold), runs a tool loop, fans out 4 explore subagents, loops on to ~125K, compacts, restarts from the summary, fans out 3 more subagents, then waits for B and C to finish and wraps up alone: a last loop, a review subagent and a long pull-request write-up. |
| B | Qwen-Code-like (17 tools, ~11K) | Resumes a ~60K Python session, loops (a user retry, a loop-detection side call), launches two research subagents at once, sits idle while A compacts, comes back, loops, sends a second side call, has its older tool results cleared by the client (all but the last 20), loops. |
| C | Claude-Code-like | Resumes a ~25K tests/docs session, loops, receives a pasted ~16K-token CI log that the client aborts after 1.5 s and re-sends, launches a review subagent, loops. |
| explore-1..7, survey-1..2, review-1..2 | Subagent personas (6 tools, ~12-15K shared prefix) | 4-5 read-only research or review turns each, final report. |

The three resumes run one after another before the sessions start interleaving, so they are
clean, isolated cold prefills of ~25K, ~55K and ~80K tokens. Four agent turns ask the model to
rewrite or edit a file it has just read (copy-heavy output). B's concurrent subagent pair and A's
solo wrap-up give every arm sustained stretches of two-request and one-request decoding, whatever
its prefill speed does to the interleaving elsewhere.

**Closed loop, identical observations.** Every observation the client sends (tool results, user
messages, subagent reports, the post-compaction summary) is generated before the run from real
files, diffs and history of this repository at a pinned commit (`AB_CORPUS_COMMIT`, default
`e48a0d28`), so every arm receives byte-identical observations. The assistant turns are each
arm's own streamed output (reasoning, content and tool calls), fed back as an agent client does;
that is what makes the engine's private-endpoint reuse behave as in production. Observations
answer the model's tool calls by id whatever it asked for, so every arm sees the same token
growth. Each request carries a `seed` that is identical in every arm; it also tags the request
in the serve log.

**Lock-step main sessions.** Once resumed, A, B and C take their turns in rounds: a session
sends its next request only when every other session in the rounds is also ready to send, and
within a round the scripted tool and user delays, identical in every arm, set the order. A
session leaves the rounds while it waits on its subagents or on another session, and when it
finishes, so the rounds never stall. Without them, a faster build runs one session ahead of the
others and changes which session's prefix is evicted at the device-KV limit, so the cache rows
would partly measure speed.

**Sampling.** Every request sends `temperature 1.0`, `top_p 0.95`, `top_k 20`, as 97 % of the
production requests did.

What differs between arms is only what should: the model's sampled text (the kernels differ
numerically, so outputs diverge after a few tokens), and therefore interleaving and queueing.
Rates, cache fractions and per-class averages are comparable; absolute completion totals are not.
Sampled answer lengths are heavy-tailed, as in production: the longest 8 % of turns write about
45 % of all output, and which turns run long changes with every run, so one run is one draw.
Several workload seeds (`--seeds`) show how much a result depends on the particular session.

### Fairness

- **Control = upstream at the commit the fork has merged, plus only the Windows port.** A
  control built from an older upstream would credit the fork with upstream's own newer work.
- **Same launch configuration.** The treatment runs the deploy folder's launch bat
  (`AB_LAUNCH_BAT`, default the official-artifact launcher) plus `AB_TREATMENT_EXTRA_FLAGS`
  (default `--fast-prefill-kernel`, from the production launcher). The control runs the same
  flags minus the ones its `--help` does not advertise.
- **Same host RAM.** Upstream has no `--host-cache-mib`. The runner reads the split the fork
  resolved at startup (host state slots, host KV bytes, private continuations, long anchors,
  shared prefixes) and passes the control those exact values as explicit flags.
- **Same context.** Upstream keeps a fixed 1 GiB of VRAM spare under `--kv-capacity auto`, so it
  may not start at the bat's context. The runner tries the bat's value first, then 200000,
  180000, 170000, 160000, and runs every arm at the first one the control starts with. The
  fork's larger device KV at that context (`--vram-headroom-mib 0` and its measured CUDA Graph
  allowance) is part of what is being compared and is shown in the report header.

### Running it

1. Stop any server on the port and let the GPU go idle (the runner checks both and refuses
   otherwise; it never starts or stops a production server).
2. Build the control:

   ```bat
   git worktree add -b ab/upstream-windows-port <dir>\src origin/master
   git -C <dir>\src cherry-pick <windows-port-commit>
   set AB_CONTROL_SRC=<dir>\src
   set AB_CONTROL_BUILD=<dir>\build
   build_control.bat configure
   build_control.bat build
   ```

3. Run both arms (about 35 minutes on an RTX 5090, including model loads):

   ```bat
   set AB_CONTROL_EXE=<dir>\build\apps\Release\ninfer-serve.exe
   python runner.py
   ```

   `--arms treatment,alt,control` adds the original-prefix-cache arm (about 50 minutes; the arms
   always run in that order). The control's host cache is translated from the split the fork's
   original cache resolves for the same `--host-cache-mib`, read from one extra startup of the
   fork before the first seed. `--seeds 42,43,44` runs every arm once per workload seed, seed by seed, into
   `<out>/seed-<n>`, and writes a combined report to `<out>/report.md`; each seed replays
   different observations. `--ctx N` skips calibration, `--scale F` stretches or shrinks the
   session loops (0.3 is a quick smoke run), `--dry-run` prints the plans and flags.
   `python analyze.py <run_dir>` re-renders a report from whichever arms the run directory holds;
   every arm is compared with the control. `python analyze.py --aggregate <out> <run_dir>...`
   re-renders the combined report.

| Variable | Default |
|---|---|
| `AB_LAUNCH_BAT` | `<AB_DEPLOY>\LaunchQwen3.8-27B-official-dflash2-ngram.bat` |
| `AB_DEPLOY` | `E:\NInfer-Deploy-V3` |
| `AB_MODEL` | the model path in the launch bat |
| `AB_TREATMENT_EXE` | `build-windows\apps\Release\ninfer-serve.exe` in this checkout |
| `AB_CONTROL_EXE` | `bench\agentic_ab\control\build\apps\Release\ninfer-serve.exe` |
| `AB_TREATMENT_EXTRA_FLAGS` | `--fast-prefill-kernel` |
| `AB_ALT_EXTRA_FLAGS` | `--use-original-prefix-caching` (added to the treatment's flags) |
| `AB_HOST` / `AB_PORT` | `127.0.0.1` / `8080` |
| `AB_OUT` | `profiles\bench\agentic_ab` in this checkout |
| `AB_CORPUS_REPO` / `AB_CORPUS_COMMIT` | this checkout / `e48a0d28` |

A run is marked invalid (non-zero exit, warning in the report) if any workload request fails.
The report also flags when the client's context guard had to clear old tool results to keep a
prompt 24K tokens under `--max-context`.

### Reading the report

- **TTFT** includes queueing: two lanes serve up to seven requests in flight. The report also
  gives the average queue wait and TTFT without it.
- **Brackets on the TTFT and cache rows** are 95 % block-bootstrap intervals over stretches of
  10 consecutive requests: how much the number moves with which stretches of the run it
  contains. They cannot show how differently another run would interleave; the combined report
  over several seeds shows that, as the mean with the min-max over seeds, and each arm's change
  against the control computed per seed.
- **Continuing-session turns** are tool-loop turns, retries, the post-idle turn, the
  post-history-edit turn and the abort retry; cache retention decides their TTFT. **New long
  prompts** are the resumes, compaction and side calls; prefill speed decides theirs.
- **Re-prefilled turns** are split into main-session and subagent turns: losing a 100K main
  session costs far more tokens than losing a 15K subagent, so the two tell different stories.
- **Prefill tok/s with no cache hit** is token-weighted (total prefilled tokens over total
  prefill time) over requests that had no hit in any arm and prefilled at least 4,096 tokens,
  with a separate 32K+ row; the per-request table shows them by size.
- **Output tok/s** is the decode tokens the server committed per second of the engine's own
  decode time: device wait plus host work of decode rounds, from the serve's ~5 s throughput
  records. Prefill chunks and idle time do not dilute it, and ngram copies count as output.
  The *one request* and *two requests* rows use only the records in which every decode round ran
  that many requests, so a build's batching mix cannot move them. The *all* row takes every decode
  record at the batching the run produced, so faster prefill that keeps both lanes decoding shows
  up there. Brackets are 95 % block-bootstrap intervals over ~30 s stretches of decoding: how much
  the rate moves with which turns happened to decode. A row with less than ~150 s of decoding
  shows its seconds instead of an interval.
- **Output tok/s = decode rounds/s × tokens per round.** Rounds/s is the engine's own speed at
  that batch size (kernels and host work) and varies only a few percent across a run. Tokens per
  round is speculative acceptance: it moves with what the model happened to write (a file copy
  accepts several times more than fresh reasoning) and carries most of the interval on output
  tok/s. Compare rounds/s for engine speed and tokens per round for drafting; each arm samples
  its own text, so acceptance also differs between repeated runs of one build. The notes give
  each arm's decode seconds per row and per-request completion over decode wall time, which is
  what one stream saw, including other lanes' batching and prefill.
- **Output volume** is sampled, not controlled: the notes give each arm's completion and thinking
  totals and how many turns ran into the thinking budget. A few long turns shift cache
  pressure, batching and wall time, so compare seeds before attributing them to a build.
- **Where cache hits came from** and **cache pressure** (eviction, degradation, spill and
  host/device transfer counters summed from the serve's throughput records) explain the cache
  rows.
