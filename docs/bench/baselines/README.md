# M2.8 E2E Baseline Summaries

This directory is reserved for committed, auditable summaries of official M2.8/M3-gate e2e baseline
runs.

Raw e2e reports remain local under `profiles/e2e/` because `profiles/` is ignored and may contain
large profiler artifacts. Each committed summary here must include enough information to audit the
baseline without relying on chat history:

- raw report path and SHA256;
- generation command;
- git commit;
- q5090 identity and SHA256;
- prompt fixture set and case summary;
- timing summary;
- memory summary;
- hidden allocation status;
- workspace policy;
- tokenizer provenance.

Smoke runs do not satisfy the M3 gate. Official M3-gate summaries must follow
`docs/m2.8-pre-m3-standard.md` and `docs/bench/e2e-report-schema.md`.

Committed summary files use:

```json
{
  "artifact_type": "qus_e2e_baseline_summary",
  "schema_version": 1,
  "baseline_class": "smoke"
}
```

`baseline_class` is either `smoke` or `m3_gate`.
