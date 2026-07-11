# Optimization-Era Documentation

This archive contains the dated work that moved the project from the M2.8 baseline through the
specialized linear backend, q5090 v2/v3/v4 evolution, CUDA Graph decode, MTP, sampling, INT8 KV,
serving, and native Vision integration.

The files preserve designs, decisions, implementation reports, and evidence as originally written.
They are historical even when their own headers say “current”, “approved”, “normative”, or
“pending”. For the implemented system, start at [`../../README.md`](../../README.md).

## Contents

- root Markdown files — dated designs/reports plus superseded long-lived design documents;
- `design.md`, `qwen3.6-27b-architecture.md`, and `q5090_packed_file_format_v4.md` — snapshots of
  those documents immediately before the documentation cleanup, retained so historical plans keep
  their original context;
- `plans/` — completed implementation plans and roadmaps;
- `bench/` — benchmark and profiler reports;
- `roofline/` — historical roofline investigations;
- `q5090_*v1/v2/v3*` files — retired artifact specifications and migration material.

No backward compatibility is implied by preserving a retired format specification here.
