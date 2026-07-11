# General Evaluation Framework Design And Implementation Plan

> Status: completed on 2026-07-12.
>
> This document records the design and implementation plan for the evaluation framework under
> `eval/`. Stable operation now belongs in `eval/README.md`; this archived plan is retained as
> implementation provenance.

## 1. Goal

Build a repository-local capability-evaluation framework that can run reproducible evaluation
suites against this project's OpenAI-compatible server, other local model servers, and remote
online services.

The framework must:

- make endpoint URL, optional API key, model id, request settings, and concurrency configurable;
- support adding datasets without changing the orchestration core;
- treat EvalScope as one evaluation backend rather than as the framework itself;
- provide useful terminal progress, persistent logs, elapsed time, rate, and ETA where the backend
  exposes enough information;
- retain raw backend evidence while producing a small, versioned normalized summary;
- resume safely when a backend supports it, without reusing results from a different effective
  configuration;
- initially support `aime25`, `aime26`, `gpqa_diamond`, and `bfcl_v4` through EvalScope.

## 2. Non-Goals

The first implementation will not:

- add batching or concurrency to `qus-serve`;
- make the C++ inference runtime support other models;
- implement a distributed scheduler, multi-host worker pool, queue service, or Web UI;
- reimplement benchmark datasets or scoring already owned by EvalScope or another evaluator;
- force every backend into one universal per-sample prompt or conversation representation;
- synthesize an overall score across unrelated benchmarks unless a suite explicitly defines and
  documents an aggregation formula;
- automatically install packages, download large models, or acquire paid external-service keys;
- promise precise ETA when the backend cannot report a reliable total or when a sample expands into
  an unknown number of agent/tool turns.

Performance stress testing is separate from capability evaluation. Per-request latency and token
counts may be recorded when available, but this framework's first contract is correctness and
reproducible scores, not load generation.

## 3. Current Facts

- `eval/` exists and is empty.
- The canonical Python is `/home/neroued/miniconda3/envs/py311/bin/python`.
- EvalScope, `bfcl-eval`, and the OpenAI Python package are not installed in that environment.
- `qus-serve` exposes OpenAI Chat Completions and supports function tools, but one Engine serializes
  generation. A target for this server therefore uses `max_concurrency: 1`.
- Concurrency is a property of a configured target/run, not a framework invariant. A remote service
  may use a larger value.
- EvalScope 1.9.0 supports the initial benchmark set and exposes `eval_batch_size`, a progress
  tracker, backend work directories, prediction/review caches, and structured reports.
- BFCL-v4 uses `bfcl-eval==2025.10.27.1`, includes ordinary, live, multi-turn, web-search, and memory
  subsets, and has its own official aggregate score.
- `127.0.0.1:8080` is currently occupied by `tunnel-client`; no particular local port is assumed by
  this design.

## 4. Focused Reading List

Read before implementation:

- `AGENTS.md`;
- `docs/design.md`, for repository ownership boundaries;
- `docs/serving.md`, for this server's OpenAI, sampling, thinking, and tool-call behavior;
- EvalScope's pinned-version `TaskConfig`, progress tracker, report model, AIME/GPQA adapters, and
  BFCL-v4 adapter;
- the BFCL-v4 usage and aggregate-score documentation.

No q5090, model-math, or CUDA documentation is required because this work does not change those
surfaces.

## 5. Ownership And Boundaries

All implementation lives under `eval/` and is Python-only.

```text
eval CLI / coordinator
    |
    +-- configuration and secret resolution
    +-- run manifest and state
    +-- progress and event logging
    +-- backend registry
    |     +-- EvalScope backend
    |     +-- mock backend used for contract verification
    |     `-- future native or third-party backends
    `-- normalized report writer
```

The coordinator owns lifecycle and cross-backend contracts. A backend owns evaluator-specific
configuration, execution, progress extraction, resume behavior, and result conversion.

The inference service is an external target. The framework does not special-case `qus-serve`; its
single-sequence restriction is expressed only in the target configuration.

## 6. Proposed Repository Layout

```text
eval/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── local-qus.yaml
│   └── capability-suite.yaml
├── qus_eval/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── coordinator.py
│   ├── events.py
│   ├── logging.py
│   ├── progress.py
│   ├── result.py
│   ├── secrets.py
│   └── backends/
│       ├── base.py
│       ├── registry.py
│       ├── mock.py
│       └── evalscope.py
├── tests/
│   ├── fixtures/
│   ├── test_config.py
│   ├── test_coordinator.py
│   ├── test_progress.py
│   └── test_report.py
└── runs/                         # ignored, one directory per run
```

Tests are limited to observable configuration, progress/event, resume-safety, and report-schema
contracts. They must not scan implementation source or lock private call order.

## 7. Configuration Model

YAML is the user-facing format. Configuration is split into reusable targets and suites, while one
file may contain both for simple usage.

```yaml
schema_version: 1

targets:
  local_qus:
    protocol: openai_chat
    base_url: http://127.0.0.1:18080/v1
    model: qwen3.6-27b
    api_key_env: QUS_API_KEY       # optional; omit when the endpoint needs no key
    max_concurrency: 1
    request:
      timeout_seconds: 3600
      retries: 2

  remote_model:
    protocol: openai_chat
    base_url: https://example.invalid/v1
    model: remote-model-id
    api_key_env: REMOTE_API_KEY
    max_concurrency: 8
    request:
      timeout_seconds: 600
      retries: 5

suites:
  capability:
    jobs:
      - id: aime25
        backend: evalscope
        dataset: aime25
        target: local_qus
        max_concurrency: 1          # optional per-job cap
        generation:
          temperature: 0.6
          top_p: 0.95
          max_tokens: 16384
          seed: 42
        backend_args:
          judge_strategy: rule

      - id: bfcl_v4
        backend: evalscope
        dataset: bfcl_v4
        target: local_qus
        generation:
          temperature: 0
        backend_args:
          is_fc_model: true
          underscore_to_dot: true
          serpapi_api_key_env: SERPAPI_API_KEY

runtime:
  max_parallel_jobs: 1
  runs_dir: eval/runs
  progress:
    enabled: true
    refresh_seconds: 1
    heartbeat_seconds: 30
  samples:
    retention: all                # all | errors | none
```

### 7.1 Target fields

- `protocol`: initially `openai_chat`; future protocols require a target client or backend that
  explicitly supports them.
- `base_url`: API root such as `http://host:port/v1`, never a hard-coded project default.
- `model`: model id sent to the endpoint and recorded in reports.
- `api_key_env`: optional environment-variable name. Absence means no key. Literal secrets in YAML
  are rejected in the initial implementation.
- `max_concurrency`: positive integer limiting in-flight requests for this target.
- `request`: default timeout, retries, retry interval, and optional non-secret headers.

### 7.2 Suite and job fields

A suite is an ordered list of independently reportable jobs. A job selects one backend, one
dataset, and, when that backend needs a model service, one target. A backend may declare that no
target is required, for example for a local scorer that consumes existing predictions.
`generation` contains portable settings; `backend_args` is an explicitly backend-owned mapping.

Unknown portable fields are errors. Unknown `backend_args` are validated by the selected backend.
This avoids silently ignoring misspelled sampling or dataset options.

### 7.3 Concurrency semantics

Two independent limits exist:

- `runtime.max_parallel_jobs`: how many dataset jobs the coordinator may execute at once;
- `target.max_concurrency`: maximum aggregate in-flight model requests to a target;
- optional job `max_concurrency`: the most target slots that one job may reserve.

The initial coordinator executes jobs sequentially by default. For EvalScope, the effective target
slot grant maps to `eval_batch_size`. If parallel jobs are enabled, the scheduler reserves an
integer number of slots for each job before it starts. The sum of active reservations for jobs
sharing a target may never exceed `target.max_concurrency`. A job waits when no requested slot is
available. This reservation model is enforceable even when a third-party backend owns its internal
request threads and cannot call a coordinator semaphore for every request.

The requested and granted concurrency are written to the manifest and terminal header. They are
never inferred from the endpoint type.

## 8. Core Interfaces

The core uses a small backend protocol rather than benchmark-specific classes.

```python
class EvaluationBackend(Protocol):
    name: str

    def validate(self, job: JobConfig, target: TargetConfig | None) -> None: ...
    def plan(self, context: RunContext) -> WorkPlan: ...
    def run(self, context: RunContext, emit: EventSink) -> BackendRun: ...
    def normalize(self, run: BackendRun) -> DatasetResult: ...
```

- `validate` rejects unsupported settings before any paid or long-running request.
- `plan` describes known datasets/subsets, progress totals, required external dependencies, and
  resume capability.
- `run` owns interaction with the evaluator and writes raw artifacts below its assigned work
  directory.
- `normalize` converts backend reports into the repository's result schema without deleting or
  rewriting raw evidence.

The coordinator does not import EvalScope. Only `backends/evalscope.py` may import it. Backend
modules register by stable name through an explicit registry; discovery does not execute arbitrary
filesystem modules.

### 8.1 Extension cases

- Adding an ordinary EvalScope dataset: add a job to YAML; no core code change.
- Adding an EvalScope dataset needing special arguments: extend only the EvalScope backend's
  validation/normalization table.
- Adding a dataset evaluated by another library: add one backend module implementing the four
  methods and register it.
- Adding a fully local custom scorer: implement a native backend; no OpenAI endpoint is required if
  that backend does not need one.

The first release includes a deterministic mock backend so coordinator behavior can be verified
without EvalScope, a dataset download, a model endpoint, or network access. This is also the
architectural check that EvalScope is optional.

## 9. EvalScope Backend

The EvalScope backend performs the following translation:

- target `base_url`, `model`, and resolved key -> EvalScope API model configuration;
- target `max_concurrency` -> `eval_batch_size`;
- portable generation fields -> `generation_config`;
- job dataset and `backend_args` -> `datasets` and `dataset_args`;
- backend work directory -> EvalScope `work_dir`/`use_cache`;
- framework seed -> EvalScope `seed`;
- progress tracking -> EvalScope `enable_progress_tracker=True`;
- raw EvalScope reports -> normalized `DatasetResult`.

The adapter pins and records the installed EvalScope and benchmark-plugin versions. It rejects a
requested feature that the pinned version does not support instead of guessing an alternate flag.

Initial dataset policies:

| Dataset | Primary output | Required policy |
|---|---|---|
| `aime25` | accuracy, correct/30 | rule scoring, fixed generation settings |
| `aime26` | accuracy, correct/30 | rule scoring, fixed generation settings |
| `gpqa_diamond` | accuracy, correct/198 | zero-shot, fixed framework seed for shuffled choices |
| `bfcl_v4` | official overall and category scores | `is_fc_model`, subset completeness, BFCL dependency version |

BFCL's internal parallel function-call categories do not change HTTP request concurrency. Web and
memory subsets declare their external prerequisites during `plan`; a missing prerequisite blocks
the affected formal job before inference rather than producing a misleading zero score.

## 10. Run Lifecycle And State

Each invocation creates an immutable run identity from timestamp plus a short configuration hash:

```text
eval/runs/20260712T120000Z-3a19c842/
├── effective-config.yaml
├── manifest.json
├── state.json
├── events.jsonl
├── run.log
├── backends/
│   ├── aime25/
│   └── bfcl_v4/
├── summary.json
└── summary.md
```

Lifecycle:

1. load and validate configuration;
2. resolve environment-backed secrets without persisting them;
3. query backend plans and validate dependencies;
4. create manifest and effective configuration with secrets redacted;
5. run jobs under configured concurrency limits;
6. persist state transitions and progress events atomically;
7. normalize completed backend outputs;
8. write JSON and Markdown summaries;
9. finish as `completed`, `partial`, `failed`, or `cancelled`.

`state.json` is operational state and may be replaced atomically. `events.jsonl` is append-only
evidence. A SIGINT requests graceful cancellation, lets the backend checkpoint when supported,
writes `cancelled`, and exits nonzero.

### 10.1 Resume safety

Resume requires the original run directory. Before reusing backend cache, the coordinator compares
the stored fingerprint of:

- target URL excluding credentials, model id, and protocol;
- dataset, subsets, limit/repeats, and backend arguments;
- generation parameters and seed;
- effective concurrency when it affects backend behavior;
- backend and plugin versions;
- scoring mode and judge configuration.

A mismatch is rejected with a field-level explanation. Users start a new run instead of forcing
incompatible cache reuse. Backends that cannot resume declare that in `WorkPlan` and restart only
their own job.

## 11. Progress And Logging

### 11.1 Event model

Backends emit structured events; terminal rendering and persistent logging consume the same stream.

```python
ProgressEvent(
    timestamp,
    run_id,
    job_id,
    phase,             # planning | loading | inference | scoring | reporting
    completed,
    total,             # optional
    unit,              # samples | subsets | requests | turns
    message,
    metrics,           # optional rate/latency/token counters
)
```

Other event kinds cover job start/end, retry, warning, dependency failure, and artifact creation.
Every event has a stable schema version.

### 11.2 Terminal display

On a TTY, display:

- overall jobs completed/total;
- one active row per job with dataset, subset/phase, completed/total, percent, elapsed, rate, and ETA;
- target/model and effective concurrency;
- retry/error counters;
- the most recent meaningful status message.

On non-TTY output, emit one concise progress line at the configured heartbeat interval instead of
ANSI cursor control. CI logs therefore remain readable.

ETA is shown only when `total` is known and at least several units have completed. Agentic and
multi-turn jobs label the progress unit explicitly; sample ETA is not presented as request ETA.

### 11.3 EvalScope progress bridge

The adapter enables EvalScope's file progress tracker and polls its structured progress state. It
also emits phase changes around dataset loading, inference, review, and report generation. It does
not parse arbitrary human log lines as a correctness contract. If a pinned EvalScope component
lacks structured granular progress, the UI shows phase, elapsed time, and an indeterminate state.

### 11.4 Persistent logs

- `run.log`: human-readable timestamps, levels, job ids, progress heartbeats, retries, and errors;
- `events.jsonl`: machine-readable event stream used for later diagnosis and timing analysis;
- backend-native logs: retained unchanged below `backends/<job>/`;
- `manifest.json`: environment and reproducibility metadata;
- `summary.json`: stable downstream-consumed score contract.

API keys, authorization headers, secret environment values, and URL credentials are always
redacted. Prompt/response retention follows `runtime.samples.retention`; the manifest records the
policy. Exceptions include stack traces in backend logs, but secret-bearing request headers are
never included.

## 12. Normalized Result Contract

`summary.json` has an explicit `schema_version`. Its minimum dataset record is:

```json
{
  "job_id": "gpqa_diamond",
  "backend": "evalscope",
  "dataset": "gpqa_diamond",
  "status": "completed",
  "primary_metric": "accuracy",
  "metrics": {"accuracy": 0.6768},
  "counts": {
    "planned": 198,
    "completed": 198,
    "scored": 198,
    "failed": 0,
    "skipped": 0
  },
  "duration_seconds": 12345.6,
  "artifacts": ["backends/gpqa_diamond/reports/report.json"]
}
```

`metrics` may contain nested category results such as BFCL's `agentic`, `multi_turn`, `live`,
`non_live`, `hallucination`, and `overall`. Backend-specific details remain in raw artifacts.

The framework reports each benchmark independently. A suite-level aggregate is omitted unless the
suite declares a named formula and all required inputs are complete. BFCL's official internal
aggregate is retained because it is part of that benchmark's contract.

## 13. Manifest And Reproducibility

The manifest records:

- run id, timestamps, hostname, platform, Python version, and framework version;
- repository commit and dirty state;
- redacted effective configuration and its hash;
- target protocol, redacted URL, model id, and effective concurrency;
- evaluator/backend/plugin versions;
- dataset ids, subsets, sample counts, cache locations, and available revisions/hashes;
- generation and scoring settings;
- for the local engine when supplied by the user, artifact path/manifest identity and server launch
  settings;
- status, errors, retry counts, duration, and artifact paths.

The framework must not infer the local q5090 artifact from `out/`; an optional explicitly configured
artifact identity is recorded only as provenance.

## 14. CLI Contract

Initial commands:

```bash
eval/.venv/bin/python -m qus_eval validate --config eval/configs/capability-suite.yaml
eval/.venv/bin/python -m qus_eval plan --config eval/configs/capability-suite.yaml --suite capability
eval/.venv/bin/python -m qus_eval run --config eval/configs/capability-suite.yaml --suite capability
eval/.venv/bin/python -m qus_eval status --run eval/runs/<run-id>
eval/.venv/bin/python -m qus_eval resume --run eval/runs/<run-id>
eval/.venv/bin/python -m qus_eval summarize --run eval/runs/<run-id>
```

`validate` performs no model requests. `plan` may inspect local package versions and endpoint health
but must identify every network action it performs. `run` prints the run directory immediately so
the user can inspect logs from another shell.

Exit codes distinguish successful completion, invalid configuration, dependency/preflight failure,
partial evaluation, runtime failure, and cancellation. Exact numeric values are documented in
`eval/README.md` with the implementation.

## 15. Failure Policy

- Configuration and missing required dependency errors fail before starting model inference.
- Authentication and systematic schema incompatibility stop the affected job quickly.
- Transient transport failures use the configured bounded retry policy and emit retry events.
- A scored model-format failure remains a benchmark result when the evaluator defines it that way;
  it is not rewritten as infrastructure failure.
- Backend crashes or missing reports mark the job failed and preserve its work directory.
- Formal summaries are `partial` if any required job/subset is incomplete. They must not present a
  partial BFCL run as the official full score.
- `ignore_errors` is not a global default. Any future best-effort mode must remain explicit and be
  recorded in the summary.

## 16. Implementation Phases

### Phase 1: Core Contracts And Mock Backend

Implement configuration loading/validation, secret resolution, backend registry, run directories,
manifest/state/event schemas, normalized reports, and the deterministic mock backend.

Verification:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m venv eval/.venv
eval/.venv/bin/python -m pip install -r eval/requirements.txt
eval/.venv/bin/python -m py_compile $(rg --files eval/qus_eval -g '*.py')
eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
eval/.venv/bin/python -m qus_eval validate --config eval/configs/capability-suite.yaml
```

Definition of done:

- mock jobs run without EvalScope installed or imported;
- malformed configurations and literal secrets are rejected clearly;
- summary, manifest, state, and event files validate against the implemented schema contracts;
- cancelling and resuming a mock run demonstrates safe state transitions.

### Phase 2: Progress, Logging, And Concurrency

Implement TTY/non-TTY progress rendering, heartbeat logs, structured events, retry/error counters,
target concurrency limits, and atomic state updates.

Definition of done:

- a bounded mock workload demonstrates concurrency 1 and concurrency greater than 1;
- observed maximum in-flight work never exceeds the configured target limit;
- terminal progress and `events.jsonl` agree on completed/total counts;
- unknown totals render without a fabricated percentage or ETA;
- logs contain no configured secret values.

### Phase 3: EvalScope Reasoning Backend

Pin EvalScope, implement the adapter, and support full and limited runs for AIME25, AIME26, and
GPQA-Diamond.

Definition of done:

- `validate` and `plan` work without starting inference;
- `limit: 2` smoke runs complete against an explicitly supplied OpenAI-compatible endpoint;
- `max_concurrency` reaches EvalScope as `eval_batch_size`;
- EvalScope progress and raw artifacts appear beneath the job work directory;
- normalized sample counts and accuracy agree with the raw EvalScope report;
- AIME rule scoring does not invoke an unconfigured external judge.

### Phase 4: BFCL-v4 Integration

Add BFCL dependency/preflight validation, subset configuration, external-key handling, progress
bridging, official category normalization, and cache/resume behavior.

Definition of done:

- smoke coverage includes simple, multiple/parallel, irrelevance, and multi-turn categories;
- missing SerpAPI or memory prerequisites block only configurations that require those formal
  subsets and are reported before expensive inference;
- native function-call model failures are preserved as scored BFCL outcomes;
- a full report includes every required score-bearing subset and the official aggregate;
- `live_relevance` handling matches the pinned BFCL contract.

### Phase 5: Documentation And Final Review

Write `eval/README.md`, example configurations for a serial local endpoint and a concurrent remote
endpoint, dependency bootstrap instructions, score interpretation, resume instructions, and a
troubleshooting section.

Perform a final review of:

- concurrency composition and target isolation;
- secret redaction and sample-retention policy;
- cache fingerprints and resume safety;
- normalized report/schema correctness;
- evidence that the core does not import EvalScope;
- BFCL completeness and aggregate semantics;
- progress behavior for known and unknown totals.

Final verification:

```bash
eval/.venv/bin/python -m py_compile $(rg --files eval/qus_eval -g '*.py')
eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
eval/.venv/bin/python -m qus_eval run --config eval/tests/fixtures/mock-suite.yaml --suite all
eval/.venv/bin/python -m qus_eval summarize --run <mock-run-dir>
rg -n 'evalscope' eval/qus_eval --glob '!backends/evalscope.py' --glob '!backends/__init__.py'
git diff --check
```

The `rg` command is a focused architecture review aid, not a unit test.

## 17. Definition Of Done

The framework is ready for its first formal evaluation when:

- the same coordinator runs a mock backend without EvalScope and the initial datasets through the
  EvalScope backend;
- endpoint, optional key, model id, timeouts, retries, and concurrency are configuration-driven;
- adding an ordinary EvalScope dataset requires only configuration;
- a non-EvalScope backend can be added without changing coordinator or report code;
- progress remains useful in TTY and log-only environments;
- every run has redacted configuration, manifest, state, events, raw artifacts, and normalized
  summaries;
- interrupted compatible runs resume, while incompatible cache reuse is rejected;
- the initial smoke matrix passes and the exact commands/configuration are recorded;
- documentation links and stale references are checked and `git diff --check` passes.

## 18. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Backend concurrency combines unexpectedly | Separate job and target limits; one shared target limiter |
| EvalScope API/report changes | Pin version; isolate imports and normalization in one adapter |
| Progress appears stuck during opaque phases | Phase events and timed heartbeats; indeterminate display when totals are unavailable |
| Agentic ETA is misleading | Label units; calculate ETA only from reliable completed/total data |
| Online keys leak into artifacts | Environment-only secrets and centralized redaction tests |
| Cache produces scores for a changed config | Full effective-config fingerprint and field-level resume rejection |
| BFCL external failures become false zero scores | Preflight dependencies; distinguish infrastructure from scored model failures |
| Generic design grows into a framework for hypothetical needs | Keep the four-method adapter boundary; add backend-specific behavior only when a real evaluator requires it |

## 19. Archive And Evidence

While implementation is active, this file remains under `docs/plans/`. After completion:

- move this plan to `docs/archive/optimization-era/plans/`;
- add the archived path to `docs/archive/README.md` only if a new archive grouping is needed;
- keep stable usage and configuration contracts in `eval/README.md`;
- keep raw run data under ignored `eval/runs/`;
- preserve any selected dated formal-evaluation summary with its commit, dirty state, target,
  backend versions, full command, configuration hash, and artifact identity.
