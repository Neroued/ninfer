# Capability Evaluation

`eval/` contains the repository-local capability evaluation coordinator. It can evaluate this
project's server, another local OpenAI-compatible service, or a remote online model. The inference
engine is only one possible target; its single-sequence limitation is represented by
`max_concurrency: 1`, not built into the framework.

EvalScope is the first real evaluation backend. The coordinator, configuration, logging, progress,
resume, and result contracts do not import or depend on EvalScope. The deterministic `mock` backend
can exercise those contracts without a model service or network access.

## Environment

Create the isolated environment with the repository's canonical Python:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m venv eval/.venv
eval/.venv/bin/python -m pip install -r eval/requirements.txt
```

The pinned initial stack is EvalScope 1.9.0, `bfcl-eval==2025.10.27.1`, and the BFCL runtime
dependency `soundfile==0.14.0`. Dataset and model caches remain owned by their upstream libraries.
Installing dependencies does not download the Qwen model or create a `.ninfer` artifact.

## Configuration

See [`configs/capability-suite.yaml`](configs/capability-suite.yaml) for the initial AIME25,
AIME26, GPQA-Diamond, and BFCL-v4 suites, and [`configs/mock-suite.yaml`](configs/mock-suite.yaml)
for a network-free example.

A target defines the model service:

```yaml
targets:
  model_api:
    protocol: openai_chat
    base_url: http://127.0.0.1:18080/v1
    model: qwen3.6-27b
    api_key_env: MODEL_API_KEY   # optional; omit for an unauthenticated endpoint
    max_concurrency: 1
    request:
      timeout_seconds: 3600
      retries: 2
```

API keys must come from environment variables. Literal `api_key`, `Authorization`, and
`x-api-key` configuration is rejected so secrets cannot enter saved effective configurations.

Concurrency has two levels:

- `runtime.max_parallel_jobs` controls concurrently active dataset jobs;
- target `max_concurrency` caps aggregate requests to that endpoint;
- optional job `max_concurrency` caps how many target slots one job may reserve.

For EvalScope, the granted job slots become `eval_batch_size`. Multiple jobs sharing a target can
never reserve more slots than the target capacity. Set the local `ninfer-serve` target to one; set a
larger explicit value for an online service that supports it.

Portable generation settings live under `generation`. Evaluator-specific controls live under
`backend_args`; unknown fields are rejected rather than silently ignored.

## Commands

Set `PYTHONPATH` because this is a repository-local package:

```bash
export PYTHONPATH="$PWD/eval"
```

Validate configuration and installed runtime dependencies:

```bash
eval/.venv/bin/python -m ninfer_eval validate \
  --config eval/configs/capability-suite.yaml --suite smoke
```

Show expected work without making model requests:

```bash
eval/.venv/bin/python -m ninfer_eval plan \
  --config eval/configs/capability-suite.yaml --suite reasoning
```

Add `--check-runtime` to resolve configured secret environment variables and check pinned backend
packages.

Run the network-free coordinator check:

```bash
eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/mock-suite.yaml --suite all
```

Run the small real-endpoint matrix before a formal evaluation:

```bash
eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/capability-suite.yaml --suite smoke
```

Then run the full reasoning and BFCL suites independently:

```bash
eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/capability-suite.yaml --suite reasoning

SERPAPI_API_KEY=... eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/capability-suite.yaml --suite bfcl_full
```

BFCL-v4 full evaluation contains 5,106 samples. Multi-turn samples can make more than one model
request. Its Web Search subsets require `SERPAPI_API_KEY`; `memory_vector` may download an upstream
model, which the example explicitly acknowledges with `allow_network_downloads: true`.

Inspect and resume a run:

```bash
eval/.venv/bin/python -m ninfer_eval status --run eval/runs/<run-id>
eval/.venv/bin/python -m ninfer_eval resume --run eval/runs/<run-id>
eval/.venv/bin/python -m ninfer_eval summarize --run eval/runs/<run-id>
```

Resume rejects a changed effective configuration or backend version. Completed jobs are skipped;
an incomplete EvalScope job reuses its own prediction cache when available.

## Progress And Logs

TTY runs use a live display with dataset phase, completed/total units, elapsed time, rate, and ETA.
Non-TTY runs print periodic heartbeats without ANSI cursor control. Unknown totals remain `?`; the
framework does not invent a percentage or ETA.

Every run is stored below `eval/runs/<timestamp>-<config-hash>/`:

| Artifact | Purpose |
|---|---|
| `effective-config.yaml` | validated, secret-free effective configuration |
| `manifest.json` | git state, environment, backend versions, target and concurrency provenance |
| `state.json` | atomically updated operational and resume state |
| `events.jsonl` | append-only structured progress and lifecycle events |
| `run.log` | human-readable timestamps, progress, retries, and failures |
| `backends/<job>/` | unchanged backend-native predictions, logs, cache, and reports |
| `summary.json` | versioned normalized result contract |
| `summary.md` | compact human-readable score table |

The sample-retention policy is recorded in the manifest. API keys and known secret values are
redacted from coordinator events and task snapshots.

## Scores

Each benchmark remains independently reportable. The framework does not average AIME, GPQA, and
BFCL into an invented cross-benchmark score.

- AIME25 and AIME26 report rule-scored accuracy over 30 samples each.
- GPQA-Diamond reports accuracy over 198 samples.
- BFCL-v4 reports its official `agentic`, `multi_turn`, `live`, `non_live`, `hallucination`, and
  `overall` values when the full score-bearing suite is complete.

A partial or failed job makes the run `partial` or `failed`; an incomplete BFCL run is never labeled
as the official full BFCL score.

## Adding Evaluations

An ordinary EvalScope dataset needs only another configured job:

```yaml
- id: new_dataset
  backend: evalscope
  dataset: evalscope_dataset_name
  target: model_api
  generation:
    temperature: 0
  backend_args:
    subset_list: [subset_name]
```

An evaluator that does not use EvalScope implements the four-method backend protocol in
`ninfer_eval/backends/base.py`, registers one stable name in `backends/registry.py`, retains its raw
artifacts, and returns the normalized `DatasetResult`. The coordinator and summary writer do not
need benchmark-specific changes.

## Exit Status

| Code | Meaning |
|---:|---|
| 0 | completed successfully, or status query for an active run |
| 2 | invalid configuration or missing configured secret |
| 3 | missing/incompatible backend dependency |
| 4 | partial evaluation |
| 5 | failed evaluation or missing run artifact |
| 6 | cancelled evaluation |

## Verification

```bash
PYTHONPATH=eval eval/.venv/bin/python -m py_compile $(rg --files eval/ninfer_eval -g '*.py')
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/mock-suite.yaml --suite all
```
