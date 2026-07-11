# Sampling

Both `qus` (CLI) and `qus-serve` decode with a real sampler by default. This
replaces pure `argmax` decoding, which Qwen3 explicitly warns falls into endless
repetition in thinking mode (observed as a 12k-token tool-follow-up loop with an
abnormally high MTP acceptance rate).

## Defaults (Qwen3 thinking)

When no sampling fields are given, both front-ends use the Qwen3 thinking
recommendation:

| Field | Default |
|-------|---------|
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `top_k` | `20` |
| `presence_penalty` | `1.0` |
| `frequency_penalty` | `0.0` |
| `min_p` | `0.0` (disabled) |

`qus-serve` honors per-request OpenAI sampling fields (`temperature`, `top_p`,
`top_k`, `presence_penalty`, `frequency_penalty`, `seed`); any omitted field
falls back to the server default above. `top_k` and `min_p` are Qwen/vLLM
extensions, not part of the OpenAI schema.

### `top_k` is clamped to 20

The sampler builds its truncated distribution from at most **20 candidates** (the
Qwen3.6 thinking default). A request `top_k` in `[1, 20]` is honored exactly; a
`top_k <= 0` (no explicit limit) or a `top_k > 20` both select the top 20 logits.
`top_p` and `min_p` then operate **within that top-20 set**. This is a deliberate
scope choice for the specialized engine, not a bug: larger `top_k` semantics are
not supported.

## Greedy / parity (`--greedy`, `temperature 0`)

`temperature <= 0` is an **exact greedy bypass**: the sampler kernels run the same
`argmax` reduction as before and are **bit-identical** to pre-sampler output
(lowest-index tie-break). Use it for deterministic parity:

- `qus --greedy ...` and `qus-serve --greedy ...` force `temperature 0` regardless
  of the request.
- A request with `temperature: 0` selects the same greedy path on a sampling
  server.
- Two `--greedy` runs of the same prompt produce identical tokens.

The `Engine` / `SamplingConfig` default is greedy, so anything that drives the
engine directly (parity tests such as `tests/test_engine_real_file.cpp`, the
PyTorch reference in `tools/parity/`) stays deterministic by construction. Only
the two front-ends flip to sampling. Any **manual** CLI-driven greedy comparison
must pass `--greedy`.

## Reproducibility (seed)

Sampling uses a counter-based RNG keyed by `(seed, absolute_position, purpose)`,
so a fixed seed reproduces the exact token stream.

- **CLI**: default seed is a fixed constant (`0`) so demo runs are reproducible;
  override with `--seed N`.
- **serve**: a request `seed` is honored; otherwise the operator may pin one with
  `--seed N`. With neither set, each request draws a fresh random seed so
  regenerations of the same prompt differ.

## MTP interaction (distribution-correct speculative sampling)

Sampling stays compatible with MTP speculative decoding at no meaningful
per-token cost:

- The **draft/propose path is unchanged** (greedy, one-hot `q`). The expensive
  MTP propose/AR forward passes are untouched.
- Only the **accept kernel** changes. With `temperature > 0` it reads the
  already-computed verify logits `[vocab, k+1]` and runs rejection sampling: accept
  draft `d_i` when `u_i < p_i(d_i)`; on rejection, resample from the residual
  distribution; when all drafts are accepted, draw the bonus token from `p_k`.
  With a greedy draft, `p_i` is the exact target distribution, so this is
  distribution-correct speculative sampling. When `presence_penalty` or
  `frequency_penalty` is active, each verify column `i` is scored with the
  global counts **plus a round-local overlay** of the drafts already accepted in
  this round (`drafts[0..i-1]`, statically known because column `i` is only
  consumed when every earlier draft accepted), so a token repeated within a
  round is penalized exactly as the per-token decode path would penalize it (see
  [`docs/2026-07-03-mtp-round-algorithm.md`](2026-07-03-mtp-round-algorithm.md) §8
  and [`docs/2026-07-03-mtp-vllm-reference.md`](2026-07-03-mtp-vllm-reference.md)
  §4). It is the same count of O(vocab) passes as the argmax verify it replaces.
- `temperature <= 0` keeps the original greedy compare (`target == draft`) exactly.

Because sampling injects diversity, the committed-token distribution matches
direct target sampling and the acceptance rate drops from the degenerate ~88% of
the repetition loop to a healthy level (~50% observed on open-ended prompts),
which is expected and correct.

## CUDA graph safety

The sampler config lives in a device buffer (not passed by value), and the RNG is
a pure function of device scalars, so a captured decode graph replays across
requests without recapture. `Engine::set_sampling` refreshes the device config on
the host between requests; `qus`/`qus-serve` call it (via the text runner) before
each prefill, so config never leaks between requests on a shared engine.

## Not supported in v1

- `n > 1`, `logprobs`.
- `logit_bias` (parsed by the OpenAI schema, but not applied by the sampler).
- `min_p` beyond a disabled default.
