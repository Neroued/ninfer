# QUS INT8 Full Evaluation and llama.cpp Comparison Plan

Status: proposed, not started

## Goal

Measure the current QUS INT8 service on the complete AIME25, AIME26, and GPQA-Diamond datasets and
on every BFCL-v4 subset except Web Search, then compare it with the local llama.cpp mainline service
on the same RTX 5090. The final report must separate capability, single-request latency, service
throughput, speculative-decoding behavior, memory, and energy instead of collapsing them into one
score.

## Non-goals

- This is not a pure runtime comparison with bit-identical weights. QUS uses q5090 mixed W4G64
  (effective text 4.8716 bpw, W8G32 MTP, shortlisted Q4 draft head); llama.cpp uses GGUF Q4_K_M
  with its embedded MTP head. Capability differences are therefore end-to-end artifact plus runtime
  differences.
- Do not update llama.cpp, regenerate either weight artifact, download a new model, or modify
  llama.cpp source during the benchmark.
- Do not average AIME, GPQA, and BFCL into an invented global capability score.
- Kernel profiling is out of scope unless the service-level results expose a specific unexplained
  bottleneck.

## Current facts

### Hardware and repositories

- GPU: NVIDIA RTX 5090, 32,607 MiB, driver 591.86.
- CPU: Ryzen 9 9950X3D, 16 cores / 32 threads; host RAM 93 GiB.
- Free filesystem space: approximately 417 GiB.
- QUS source: `/home/neroued/qwen3.6-ultraspeed`, commit
  `154ad5534f3c01d997d22be69319356eb35708d9`, currently dirty because the evaluation framework is
  uncommitted. The manifest must record the dirty diff identity.
- llama.cpp source: `/home/neroued/llama.cpp-mainline`, commit
  `07d937828636e305bc0cfe738b288f9ab05ff748`, with an unrelated untracked `qus-compare/`
  directory that must remain untouched.
- The existing llama.cpp binaries report old build commit `0eca4d4`, and `llama-server` is absent.
  Rebuilding the server from the frozen current source is mandatory.

### Artifacts

- QUS: `out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus`, 17,505,990,144 bytes.
- llama.cpp:
  `/home/neroued/models/llm/qwen/Qwen3.6-27B/gguf-q4_k_m-mtp/Qwen3.6-27B-Q4_K_M-mtp.gguf`,
  16,998,723,232 bytes, SHA-256
  `c2275978182b91ec0f0a2e334e37e4fbfc8385eb9b3cdb6d5d4f7e23fce3b557`.
- GGUF metadata: architecture `qwen35`, 65 blocks, native context 262,144,
  `qwen35.nextn_predict_layers=1`, file type Q4_K_M. The MTP weights are in the same GGUF; do not
  pass a separate `--spec-draft-model`.

### Complete evaluation workload

| Benchmark | Samples | Primary metric | Notes |
|---|---:|---|---|
| AIME25 | 30 | accuracy | sampled thinking |
| AIME26 | 30 | accuracy | sampled thinking |
| GPQA-Diamond | 198 | accuracy | sampled thinking |
| BFCL-v4 no-web | 4,906 | selected-subset accuracy and category scores | no official full OVERALL |

The selected BFCL-v4 workload includes 800 multi-turn samples. Their test definitions contain 3,380
user turns before tool-execution steps are counted. The two Web Search subsets, 200 score-bearing
variants in total, are deliberately excluded from both engines. The 465 memory variants remain and
require the BFCL memory dependencies and may initialize a vector model.

The current environment has EvalScope 1.9.0, bfcl-eval 2025.10.27.1, soundfile 0.14.0,
sentence-transformers 5.6.0, and faiss-cpu 1.11.0. No SerpAPI key is needed for the selected
no-web workload.

## Benchmark contract

### Common model behavior

Use one benchmark contract for both endpoints:

- model id: `qwen3.6-27b`;
- context per active sequence: 65,536 tokens;
- thinking: enabled;
- MTP draft maximum: 3;
- reasoning sampling: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0,
  frequency penalty 0.0, seed 42;
- reasoning maximum output: 65,000 tokens;
- BFCL: greedy temperature 0, presence/frequency penalty 0, seed 42, maximum output 8,192 tokens;
- no context shifting or silent context reduction;
- same EvalScope and BFCL package versions and the same cached dataset files;
- sample retention `all`.

The 65,000 reasoning budget is intentional: the prior 16,384-token failures completed correctly at
19,556 and 33,317 tokens under QUS INT8. Prompt plus output must remain below the 65,536 per-slot
context.

llama.cpp has extra samplers enabled by default. Disable min-p, DRY, XTC, typical-p, adaptive
sampling, and repeat penalty. Set `repeat-last-n=-1` so presence/frequency penalties see the full
generated history. Configure the sampler order as `penalties;temperature;top_k;top_p` to be as close
as possible to QUS. The implementations still need not produce identical sampled sequences.

### KV comparison contract

- QUS uses `--kv-dtype int8`.
- llama.cpp uses `-ctk q8_0 -ctv q8_0`; its MTP context also uses q8_0 K/V.
- These are both 8-bit KV modes but not a promise of identical quantization mathematics. Record the
  difference in the final report.

### Two performance lanes

1. **Latency lane:** QUS concurrency 1 versus llama.cpp `-np 1`. This is the apples-to-apples
   single-active-sequence comparison.
2. **Capacity lane:** QUS concurrency 1 versus the highest validated llama.cpp slot count from
   `N in {2, 4}`. This measures service throughput. It must not be presented as a single-request
   kernel speedup.

Full capability inference for llama.cpp may use the selected capacity-lane concurrency after a
determinism gate. A fixed representative workload must also be run at `-np 1` so latency and decode
speed remain comparable with QUS.

## Phase 1: Freeze and build

Record repository status, build configuration, binary hashes, artifact hashes, driver, CUDA, GPU,
CPU, and timestamp in the run manifest.

Rebuild only the required llama.cpp targets from the frozen source:

```bash
cd /home/neroued/llama.cpp-mainline
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server llama-bench -j 16
build/bin/llama-server --version
```

Verification:

- `llama-server --version` names commit `07d9378`, not `0eca4d4`;
- `ldd build/bin/llama-server` resolves the CUDA 13 and local llama/ggml libraries;
- no source or artifact files changed;
- `sha256sum` is recorded for both server binaries and both model artifacts.

## Phase 2: Define reproducible server launchers

### QUS

```bash
cd /home/neroued/qwen3.6-ultraspeed
build/src/qus-serve \
  out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --host 127.0.0.1 --port 18080 --model-id qwen3.6-27b \
  --max-context 65536 --prefill-chunk 1024 \
  --kv-dtype int8 \
  --mtp-draft-tokens 3 --lm-head-draft \
  --default-max-tokens 65000 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.0 --frequency-penalty 0.0 --seed 42
```

QUS remains concurrency 1. Capture stdout/stderr to a run-owned server log because each request log
contains prompt/generated tokens, TTFT, prefill rate, decode rate, wall time, MTP tokens per round,
and acceptance percentage.

### llama.cpp

Let `N` be the slot count and set total context to `65536 * N`, because llama.cpp divides `-c`
across slots.

```bash
cd /home/neroued/llama.cpp-mainline
build/bin/llama-server \
  -m /home/neroued/models/llm/qwen/Qwen3.6-27B/gguf-q4_k_m-mtp/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  -a qwen3.6-27b --host 127.0.0.1 --port 18080 \
  -ngl all -sm none -mg 0 --fit off -fa on \
  -c $((65536 * N)) -np "$N" -cb \
  -b 2048 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 --spec-draft-n-min 0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --jinja -rea on --reasoning-format deepseek \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --samplers 'penalties;temperature;top_k;top_p' \
  --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 \
  --repeat-last-n -1 --repeat-penalty 1.0 \
  --presence-penalty 1.0 --frequency-penalty 0.0 --seed 42 \
  --cache-prompt --no-context-shift \
  --metrics --perf --no-ui --offline --timeout 3600
```

Use `--fit off` so llama.cpp cannot silently reduce context or GPU residency. At startup, require
the log to show `n_ctx_slot = 65536`, MTP enabled with `n_max=3`, full GPU offload, and the requested
q8_0 target/draft KV types.

Do not assume `-ub 512` is optimal. Before the full run, test `-ub 512` and `-ub 1024` at N=1 using
the same prompt set, retain the faster stable value, and record the choice. Do not tune on final
accuracy.

## Phase 3: Protocol, template, and quality gates

Before any long run, execute these gates against each server:

1. Health and model listing.
2. Token-count parity on ten saved messages: plain AIME, GPQA, single tool call, parallel tool call,
   and multi-turn tool history. Use QUS count-tokens behavior and llama.cpp
   `/v1/chat/completions/input_tokens` or `/apply-template` plus `/tokenize`.
3. Confirm thinking is returned separately as `reasoning_content` and final text remains in
   `content`.
4. Confirm native OpenAI `tool_calls`, parallel calls, tool history, and `finish_reason=tool_calls`
   on both servers.
5. Run the 16-sample existing smoke suite and inspect every prediction/review, not just summary
   scores.
6. Repeat the llama smoke at N=1 and candidate N. Require no score changes, protocol errors,
   truncations, or invalid tool-call parsing before allowing concurrent full inference.

Exact validation entry points:

```bash
cd /home/neroued/qwen3.6-ultraspeed
PYTHONPATH=eval eval/.venv/bin/python -m qus_eval validate \
  --config eval/configs/full-compare.yaml --suite qus_smoke --check-runtime
PYTHONPATH=eval eval/.venv/bin/python -m qus_eval run \
  --config eval/configs/full-compare.yaml --suite qus_smoke
```

Use the corresponding `llama_smoke` suite after switching servers. A gate failure stops the run;
do not consume thousands of samples while the chat template or tool parser is mismatched.

## Phase 4: Select llama.cpp concurrency

Test this matrix with q8_0 target and draft KV:

| N | Total `-c` | Per-slot context | Purpose |
|---:|---:|---:|---|
| 1 | 65,536 | 65,536 | latency baseline |
| 2 | 131,072 | 65,536 | moderate batching |
| 4 | 262,144 | 65,536 | maximum candidate within native context |

For each N, run at least 16 representative requests containing short BFCL, GPQA, and long-thinking
AIME prompts. Record startup VRAM, peak VRAM, aggregate output tok/s, requests/s, p50/p95 latency,
MTP accepted/drafted tokens, deferred requests, and errors.

Select the largest N that:

- starts without auto-fit or CPU layer fallback;
- leaves at least 2 GiB of GPU memory headroom at peak;
- reports 65,536 context per slot;
- completes the test with zero OOM, protocol, and timeout failures;
- preserves smoke scores relative to N=1;
- improves aggregate output throughput by at least 15% over the previous N.

If N=4 fails any condition, use N=2. If N=2 fails, use N=1.

## Phase 5: Evaluation configuration

Add `eval/configs/full-compare.yaml` with separate QUS and llama targets/suites. Keep
`runtime.max_parallel_jobs: 1`; concurrency is inside each EvalScope job via the target capacity.

Reasoning jobs:

```yaml
generation:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  presence_penalty: 1.0
  frequency_penalty: 0.0
  max_tokens: 65000
  seed: 42
```

BFCL job:

```yaml
generation:
  temperature: 0
  presence_penalty: 0.0
  frequency_penalty: 0.0
  max_tokens: 8192
  seed: 42
  parallel_tool_calls: true
backend_args:
  subset_list:
    - simple_python
    - simple_java
    - simple_javascript
    - multiple
    - parallel
    - parallel_multiple
    - irrelevance
    - live_simple
    - live_multiple
    - live_parallel
    - live_parallel_multiple
    - live_irrelevance
    - live_relevance
    - multi_turn_base
    - multi_turn_miss_func
    - multi_turn_miss_param
    - multi_turn_long_context
    - memory_kv
    - memory_vector
    - memory_rec_sum
  is_fc_model: true
  underscore_to_dot: true
  allow_network_downloads: true
```

Target capacities:

- QUS: `max_concurrency: 1`;
- llama latency suite: `max_concurrency: 1`;
- llama capacity/full suite: selected N.

The effective configuration and server launch command must be stored in every run directory. Do not
reuse a run after changing N, context, sampler, server binary, or artifact.

## Phase 6: Telemetry and result contract

The existing score summary is insufficient for a performance comparison. Before the full run, add
or provide a postprocessor that records:

### Capability

- AIME25, AIME26, and GPQA exact accuracy and per-sample result;
- BFCL selected-sample accuracy plus the valid AGENTIC-memory, MULTI_TURN, LIVE, NON_LIVE,
  HALLUCINATION, and per-subset scores; never label the partial aggregate as official OVERALL;
- Wilson/binomial confidence intervals and paired disagreement counts;
- truncation count, empty-final-answer count, API error count, parse-error count;
- output-token distribution and finish-reason distribution.

### Performance

- model load/warmup time;
- total suite wall time and active model time;
- requests/s, samples/s, aggregate output tok/s, and correct samples/hour;
- per-request latency p50/p90/p95/p99;
- prompt and output token p50/p95/max;
- prefill tok/s and decode tok/s, reported separately;
- MTP accepted/drafted ratio and accepted tokens per target round;
- prompt-cache hit tokens where available;
- peak VRAM, average GPU utilization, average/peak power, and estimated energy per output token.

Sample GPU telemetry once per second into CSV:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm \
  --format=csv,noheader,nounits -l 1
```

Poll llama.cpp `/metrics` and `/slots` during its run. Preserve QUS request logs and llama.cpp
server logs. Because EvalScope normalizes responses, add raw-wire capture or a redacted request-log
sidecar if engine-specific response timing/MTP fields would otherwise be discarded.

## Phase 7: Full execution order

1. Pre-cache datasets and BFCL memory dependencies without a model server.
2. Run QUS full reasoning: 258 samples, concurrency 1.
3. Run llama full reasoning at selected N.
4. Run the fixed latency workload with llama N=1 and QUS N=1, three repetitions each, alternating
   engine order after warmup.
5. Run QUS BFCL-v4 no-web, 4,906 samples, concurrency 1.
6. Run llama BFCL-v4 no-web at selected safe N.
7. Resume only through `qus_eval resume`; never change a run's effective config.
8. Generate and independently review the paired comparison report.

Commands after each server is ready:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m qus_eval run \
  --config eval/configs/full-compare.yaml --suite qus_reasoning_full

PYTHONPATH=eval eval/.venv/bin/python -m qus_eval run \
  --config eval/configs/full-compare.yaml --suite llama_reasoning_full

PYTHONPATH=eval eval/.venv/bin/python -m qus_eval run \
  --config eval/configs/full-compare.yaml --suite qus_bfcl_no_web

PYTHONPATH=eval eval/.venv/bin/python -m qus_eval run \
  --config eval/configs/full-compare.yaml --suite llama_bfcl_no_web
```

## Runtime and storage estimate

Do not schedule from smoke extrapolation alone. First run 10 samples per reasoning dataset and 10
samples from each BFCL family, then calculate ETA from observed output tokens, model-call count, and
wall time.

Initial planning range on this machine:

- QUS full reasoning: approximately 4-8 hours;
- QUS BFCL no-web: approximately 6-12 hours because multi-turn/memory samples expand to many model
  calls;
- llama.cpp may reduce wall time through N=2 or N=4, but only the pilot can establish scaling;
- allow a continuous 18-24 hour window for all full runs, retries, switching servers, and review;
- reserve at least 10 GiB for predictions, reviews, logs, telemetry, and comparison artifacts.

## Comparison rules

The final report has four distinct tables:

1. **Capability:** same-item scores and deltas by dataset/subset.
2. **Single-stream efficiency:** QUS C=1 versus llama C=1.
3. **Service capacity:** QUS C=1 versus llama selected N, including N and per-slot context in every
   row.
4. **Resource cost:** artifact size, load time, peak VRAM, power/energy, and host RAM.

For each accuracy delta, include the number of QUS-only correct, llama-only correct, both correct,
and both wrong samples. Inspect a stratified set of disagreements and all truncations/protocol
errors. Do not attribute score differences solely to the runtime because the quantized artifacts
are different.

## Risks and stop conditions

- llama server version mismatch: rebuild before any score-bearing request.
- Prompt/token-count mismatch: stop and diagnose template/tokenizer behavior before the full run.
- llama N=4 OOM or context shrink: fall back according to the concurrency gate; never enable
  auto-fit.
- Reasoning repeatedly reaches 65,000 tokens: preserve output, count it as truncation, and do not
  silently increase only one engine's budget.
- BFCL tool-call parser failure in smoke: fix protocol parity before the BFCL no-web run.
- Thermal/background interference: stop if another GPU process appears; warm up before measurements.

## Definition of done

- All 258 reasoning samples and all 4,906 selected BFCL samples are completed or explicitly
  accounted for on both engines.
- No completed run has a mismatched binary/artifact/config fingerprint.
- Raw predictions, reviews, effective config, server log, GPU telemetry, and normalized summary are
  retained for each run.
- The report contains all four comparison tables, paired score analysis, truncation/error audit,
  and MTP/resource metrics.
- A separate final review verifies sampler, per-slot context, MTP, tool protocol, external dependency,
  and concurrency claims against the saved evidence.
- Verification passes:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
PYTHONPATH=eval eval/.venv/bin/python -m qus_eval validate \
  --config eval/configs/full-compare.yaml --suite qus_reasoning_full --check-runtime
PYTHONPATH=eval eval/.venv/bin/python -m qus_eval validate \
  --config eval/configs/full-compare.yaml --suite llama_reasoning_full --check-runtime
git diff --check
```

When work is complete or abandoned, move this plan to
`docs/archive/optimization-era/plans/2026-07-12-full-int8-evaluation-llamacpp-comparison.md`, add it
to the archive index if required, and keep stable evaluation behavior in `eval/README.md` rather
than in the archived plan.
