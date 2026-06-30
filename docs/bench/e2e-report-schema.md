# M2.8 E2E Report Schema v1

This is the current focused schema reference for `qus_e2e_bench`, decoded sidecars, and committed
baseline summaries. The historical M2.8 standard that introduced this schema is archived at
[`../archive/pre-optimization/m2.8-pre-m3-standard.md`](../archive/pre-optimization/m2.8-pre-m3-standard.md).

## Artifact Types

| Value | Meaning |
|---|---|
| `qus_e2e_benchmark_report` | Raw e2e benchmark JSON report. |
| `qus_decoded_text_artifacts` | Sidecar manifest for decoded text files. |
| `qus_e2e_baseline_summary` | Committed audit summary for an official smoke, M3 output-gate, or M3 prefill-gate baseline. |
| `qus_e2e_compare_result` | Optional structured result from comparing two raw e2e reports. |

## Raw E2E Report

Each `qus_e2e_bench` invocation writes one JSON object, not JSONL:

```json
{
  "schema_version": 1,
  "artifact_type": "qus_e2e_benchmark_report",
  "status": "ok",
  "run": {
    "binary": "qus_e2e_bench",
    "command": "",
    "git_commit": "",
    "worktree_dirty": false,
    "load_time_s": 0.0
  },
  "environment": {},
  "engine": {},
  "weights": {},
  "memory": {},
  "summary": {},
  "cases": []
}
```

`run.load_time_s` is measured once per process, before warmup and measured generation repeats. It is not
duplicated inside repeat records.

## Environment

Required when available:

```json
{
  "environment": {
    "cuda_runtime_version": "",
    "cuda_driver_version": "",
    "gpu_name": "",
    "device_id": 0
  }
}
```

## Engine

Required fields:

```json
{
  "engine": {
    "max_context": 0,
    "workspace_lifetime_policy": "block_scoped_mixer_mlp_rewind",
    "decode_metric": "decode_tok_s",
    "decode_path": "cuda_graph",
    "sampling_location": "device_argmax",
    "token_readback": "per_step_sync_d2h",
    "includes_token_readback": true,
    "timing_boundary": "host_visible_phase_end"
  }
}
```

The current official/runtime workspace lifetime policy is `block_scoped_mixer_mlp_rewind`.
Historical P6 short smoke artifacts may record `step_reset`; those are legacy smoke observations and do
not represent the current M3 gate package policy.

`decode_metric` is path-neutral and always names the repeat throughput field. `decode_path` records
the engine execution path that produced the decode timing: `cuda_graph` for the production default, or
`eager` when the engine is explicitly run without CUDA graph replay.

## Weights

Required fields:

```json
{
  "weights": {
    "q5090_path": "",
    "q5090_file_size_bytes": 0,
    "q5090_sha256": "",
    "q5090_conv1d_layout": "runtime_native_conv_dim_by_kernel",
    "load_strategy": "full_file_host_vector_then_h2d_payload_upload",
    "default_weight_arena_policy": "q5090_file_size_plus_256MiB",
    "estimated_host_file_buffer_bytes": 0,
    "selected_modules": {
      "text_core": true,
      "mtp": false,
      "vision": false
    },
    "q5090_loaded_payload_bytes": 0,
    "weight_arena_capacity_bytes": 0,
    "weight_arena_used_bytes": 0,
    "weight_arena_peak_used_bytes": 0,
    "weight_arena_slack_bytes": 0,
    "weight_payload_to_arena_used_overhead_bytes": 0
  }
}
```

Raw benchmark reports may leave `q5090_sha256` empty so `qus_e2e_bench` does not synchronously rescan
large q5090 files before the first case runs. Committed baseline summaries still require q5090 SHA256;
`tools/bench/make_baseline_summary.py` computes it from `q5090_path` when the raw report field is empty.

## Memory

Required shape:

```json
{
  "memory": {
    "accounting_scope": "engine_arenas_only",
    "hidden_device_allocations": true,
    "arenas": [
      {
        "name": "weights",
        "present": true,
        "capacity_bytes": 0,
        "used_bytes": 0,
        "peak_used_bytes": 0
      }
    ],
    "q5090_loaded_payload_bytes": 0,
    "q5090_tensor_count": 0,
    "q5090_quant_count": 0,
    "known_exclusions": []
  }
}
```

The official M3-ready baseline must use:

```json
{
  "accounting_scope": "engine_owned_device_arenas_complete",
  "hidden_device_allocations": false
}
```

This memory accounting does not include host q5090 file buffers, CUDA driver/runtime internals,
profiler overhead, OS/process RSS, or ignored local `profiles/` artifacts.

## Cases

Each case records fixture identity, generation config, repeat policy, raw repeat records, summaries, and
memory/timing information:

```json
{
  "name": "cn_short",
  "fixture_set": "m2.8-v1",
  "fixture_manifest_path": "bench/fixtures/prompts/m2.8-v1.manifest.json",
  "fixture_manifest_sha256": "",
  "prompt_format": "qwen3.6-chat-template",
  "messages_path": "bench/fixtures/prompts/cn_short.messages.json",
  "messages_sha256": "",
  "rendered_prompt_sha256": "",
  "prompt_ids_path": "bench/fixtures/prompts/cn_short.ids",
  "prompt_ids_sha256": "",
  "prompt_tokens": 0,
  "add_generation_prompt": true,
  "add_special_tokens": false,
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "requested_max_new_tokens": 128,
  "stop_token_ids": [248046, 248044],
  "max_context": 0,
  "decode_loop_tokens_requested": 127,
  "required_max_context": 0,
  "warmup_repeats": 1,
  "measured_repeats": 3,
  "repeats": [],
  "summary": {}
}
```

Before running a case:

```text
decode_loop_tokens_requested = max(max_new_tokens - 1, 0)
required_max_ctx = prompt_tokens + decode_loop_tokens_requested
```

`qus_e2e_bench` must reject the case before running if `max_ctx < required_max_ctx`.

## Repeats

At minimum, each measured repeat stores:

```json
{
  "repeat_index": 0,
  "prefill_time_s": 0.0,
  "decode_time_s": 0.0,
  "e2e_excluding_load_time_s": 0.0,
  "prompt_tokens": 0,
  "prefill_output_tokens": 1,
  "decode_loop_tokens": 0,
  "generated_tokens_total": 1,
  "prefill_prompt_tok_s": 0.0,
  "decode_tok_s": null,
  "decode_tok_s_valid": false,
  "e2e_excluding_load_tok_s": 0.0,
  "stop_reason": "max_new_tokens",
  "generated_token_ids": [],
  "memory": {
    "arenas": []
  }
}
```

Required timing formulas:

```text
e2e_excluding_load_time_s = prefill_time_s + decode_time_s
decode_tok_s = decode_loop_tokens / decode_time_s
prefill_prompt_tok_s = prompt_tokens / prefill_time_s
e2e_excluding_load_tok_s = generated_tokens_total / e2e_excluding_load_time_s
```

If `decode_loop_tokens == 0`, set `decode_tok_s` to `null` and
`decode_tok_s_valid` to `false`. Do not omit the field, write `NaN`, or use `0` as a fake
throughput. If `decode_loop_tokens > 0`, `decode_tok_s_valid` is `true`.

The prefill token counts in `generated_tokens_total` and e2e throughput. It does not count in
decode-loop throughput.

When `decode_path` is `cuda_graph`, reports intended for graph replay benchmarking should use
`--warmup-repeats >= 1`. The warmup repeat drives the one-time decode module warmup and graph capture
before measured repeats, so `decode_tok_s` reflects steady graph replay rather than capture overhead.

## Error Reports

Fatal errors exit nonzero. If `--output-json` is writable, the benchmark should emit:

```json
{
  "schema_version": 1,
  "artifact_type": "qus_e2e_benchmark_report",
  "status": "error",
  "error": {
    "phase": "load",
    "message": ""
  }
}
```

## Decoded Sidecar Manifest

Decoded text is human-smoke-only and is not an automated correctness gate.

Allowed `readability_gate` values:

| Value | Meaning |
|---|---|
| `human_smoke_only` | Decoded text exists for human inspection only. |
| `not_run` | Decode/readability tooling was not run for this report or summary. |

Manifest shape:

```json
{
  "artifact_type": "qus_decoded_text_artifacts",
  "source_report": "profiles/e2e/example.json",
  "readability_gate": "human_smoke_only",
  "tokenizer": {
    "tokenizer_source": "local_hf",
    "tokenizer_model_id": "Qwen/Qwen3.6-27B",
    "tokenizer_path": "",
    "tokenizer_json_sha256": "",
    "tokenizer_config_sha256": "",
    "special_tokens_map_sha256": "",
    "chat_template_jinja_sha256": "",
    "generation_config_sha256": ""
  },
  "artifacts": [
    {
      "case_index": 0,
      "case_name": "cn_short",
      "repeat_index": 0,
      "raw_text_path": "profiles/e2e/example.decoded/case0/repeat_0.raw.txt",
      "clean_text_path": "profiles/e2e/example.decoded/case0/repeat_0.clean.txt",
      "raw_text_chars": 12,
      "clean_text_chars": 10,
      "clean_text_sha256": "",
      "clean_text_nonempty_after_strip": true
    }
  ]
}
```

`smoke` and `m3_output_gate` summaries require decoded manifests and reject output cases whose clean
decoded text is empty after stripping whitespace. This is a readability smoke check only, not semantic
grading.

## Committed Baseline Summary

Raw reports stay local under `profiles/e2e/`. Official smoke, M3 output-gate, or M3 prefill-gate
summaries are committed under `docs/bench/baselines/` and use:

```json
{
  "artifact_type": "qus_e2e_baseline_summary",
  "schema_version": 1,
  "baseline_class": "m3_output_gate",
  "source_report_path": "profiles/e2e/example.json",
  "source_report_sha256": "",
  "command": "",
  "git_commit": "",
  "worktree_dirty": false,
  "q5090": {},
  "cases": [],
  "timing_summary": {},
  "memory_summary": {},
  "hidden_device_allocations": false,
  "workspace_lifetime_policy": "block_scoped_mixer_mlp_rewind",
  "decode_path": "cuda_graph",
  "tokenizer": {},
  "readability_gate": "not_run"
}
```

`baseline_class` is `smoke`, `m3_output_gate`, or `m3_prefill_gate`. Smoke summaries do not satisfy
M3 readiness. `m3_output_gate` and `m3_prefill_gate` are both required for M3 readiness.
`decoded_manifest_path` and `decoded_manifest_sha256` are present when `--decoded-manifest` is
provided. `smoke` and `m3_output_gate` require a decoded manifest; `m3_prefill_gate` may use
`readability_gate: not_run`.
Any decoded manifest referenced by a committed summary must be the redacted form: local tokenizer paths
must be empty, and raw unredacted decoded manifests stay local.

With `--decoded-manifest`, the summary also includes:

```json
{
  "decoded_manifest_path": "profiles/e2e/example.decoded/manifest.json",
  "decoded_manifest_sha256": ""
}
```

## Comparison Identity Fields

Report comparison depends on stable identity fields:

- `schema_version`;
- `artifact_type`;
- git commit and dirty/clean state;
- q5090 path, file size, and SHA256 for committed official baseline summaries;
- fixture set, manifest SHA256, case names, prompt format, messages path/SHA256, prompt ids path/SHA256,
  and prompt token count;
- chat-template identity, including chat-template Jinja SHA256 and generation-config SHA256;
- generation config, including `requested_max_new_tokens` and `stop_token_ids`;
- workspace lifetime policy;
- decode path;
- memory accounting scope.

For fixed q5090 + fixed chat-template prompt ids + fixed stop-token list + greedy generation config,
generated token ids must match across measured repeats and across compared reports unless token comparison
is explicitly disabled.

## Compare Result

`tools/bench/compare_e2e_reports.py` writes this optional JSON object when `--output-json` is passed:

```json
{
  "artifact_type": "qus_e2e_compare_result",
  "schema_version": 1,
  "status": "ok",
  "failures": [],
  "warnings": []
}
```

`status` is `ok`, `warning`, or `fail`. Warnings become failures only when the matching promotion flag is
used.

Baseline summary generation is handled by `tools/bench/make_baseline_summary.py`. It validates the raw
report schema first and preserves the raw report path and SHA256 so the committed summary can be audited
against the local artifact that produced it.
