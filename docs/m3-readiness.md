# M3 Readiness

> Status: pending regeneration of the Qwen3.6 chat-template M2.8 baseline.
> Scope: placeholder for the official M3 entry evidence package for qwen3.6-ultraspeed.

The previous pre-chat-template baseline evidence is no longer valid for M3 readiness. The controller must
rerun the official baseline with `.messages.json` prompt fixtures rendered through the Qwen3.6 chat
template, fixed prompt ids, and stop tokens `[248046, 248044]`.

Final readiness will be filled after the real q5090 run produces a valid local raw report and a committed
chat-template baseline summary under `docs/bench/baselines/`.

## Pending Evidence

- Local raw e2e report under `profiles/e2e/`.
- Committed baseline summary with `prompt_format`, `stop_token_ids`, tokenizer chat-template hash, and
  generation-config hash.
- Required `cn_short` and `long_2k` gate cases from `docs/m2.8-pre-m3-standard.md`.
- Memory accounting, hidden allocation status, workspace lifetime policy, q5090 identity, and first-wave
  M3 target ordering from the regenerated run.
