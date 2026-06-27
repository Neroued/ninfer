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
`docs/m2.8-pre-m3-standard.md`.
