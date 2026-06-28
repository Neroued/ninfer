# M2.8 E2E Baseline Summaries

This directory is reserved for committed, auditable summaries of official M2.8/M3 e2e baseline runs.
Summaries must be evidence for Qwen3.6 chat-template fixtures, not older pre-chat-template fixtures.

Raw e2e reports remain local under `profiles/e2e/` because `profiles/` is ignored and may contain
large profiler artifacts. Each committed summary here must include enough information to audit the
baseline without relying on chat history:

- raw report path and SHA256;
- generation command;
- git commit;
- q5090 identity and SHA256;
- prompt fixture set and case summary;
- `prompt_format` for each case;
- `stop_token_ids`, including `[248046, 248044]` for Qwen3.6 chat-template baselines;
- tokenizer chat-template and generation-config hashes;
- timing summary;
- memory summary;
- hidden allocation status;
- workspace policy;
- tokenizer provenance.

Smoke runs do not satisfy the M3 gate. Official M3 readiness requires both `m3_output_gate` and
`m3_prefill_gate` summaries, following `docs/m2.8-pre-m3-standard.md` and
`docs/bench/e2e-report-schema.md`.

Committed summary files use:

```json
{
  "artifact_type": "qus_e2e_baseline_summary",
  "schema_version": 1,
  "baseline_class": "smoke"
}
```

`baseline_class` is `smoke`, `m3_output_gate`, or `m3_prefill_gate`.
