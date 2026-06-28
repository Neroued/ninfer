# Qwen3.6 Chat-Template E2E Input Design

Date: 2026-06-28

## Summary

M2.8 e2e prompt fixtures must be replaced with Qwen3.6-27B chat-template-rendered token
ids. The C++ benchmark continues to consume `.ids` files only. Python tooling owns all tokenizer,
chat template, thinking-mode, rendered-prompt, and generation-config provenance work.

The current raw-text `.txt -> tokenizer.encode(..., add_special_tokens=False) -> .ids` pipeline is
not a valid representation of how Qwen3.6-27B is normally served. It omits message roles,
`<|im_start|>` / `<|im_end|>` wrappers, assistant generation prompt, thinking-mode controls, and
the generation-config EOS policy. The existing raw fixtures and their baseline reports are no
longer meaningful and will be replaced in-place.

## Evidence

Local Qwen3.6-27B tokenizer files under
`/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16` define:

- `eos_token`: `<|im_end|>`
- `model_max_length`: `262144`
- `chat_template`: a Jinja template that renders messages with Qwen chat markers
- `generation_config.json` EOS list: `[248046, 248044]`

Confirmed token ids from the local tokenizer:

| Token | Id |
|---|---:|
| `<|im_start|>` | `248045` |
| `<|im_end|>` | `248046` |
| `<think>` | `248068` |
| `</think>` | `248069` |
| `<|endoftext|>` | `248044` |

The Qwen3.6 model card shows the supported serving path as chat-completions/messages and
`apply_chat_template(..., add_generation_prompt=True)`. It also states that Qwen3.6 defaults to
thinking mode and that non-thinking mode is selected through `chat_template_kwargs` with
`enable_thinking=false`.

The local vLLM source uses the same model of operation:

- Chat completions default `add_generation_prompt=true`.
- `add_special_tokens=false` is the default because the chat template owns special tokens.
- request-level `chat_template_kwargs` are passed into `tokenizer.apply_chat_template`.
- default server-level chat template kwargs can set `{"enable_thinking": false}`.
- the Qwen3 parser treats `enable_thinking=true` as the default reasoning mode.
- `generation_config.eos_token_id` may be an int or a list and is folded into the stop token set.

The local llama.cpp source also treats chat template rendering as a first-class boundary. It has
template inputs for `add_generation_prompt` and `enable_thinking`, and it reads model chat templates
from model metadata or explicit overrides.

## Decisions

### D1. C++ Benchmark Input Boundary

`qus_e2e_bench` remains tokenizer-free. It continues to read prompt token ids from `.ids` files.
No C++ tokenizer, Jinja renderer, Hugging Face dependency, or Python runtime dependency is added to
the benchmark binary.

Rationale: this preserves the current M2.8 runtime boundary while still allowing the input ids to
represent real Qwen3.6 chat prompts.

### D2. Fixture Source Format

Each canonical fixture case is sourced from committed messages JSON, not from raw text. The
messages format follows the OpenAI/Qwen chat-completions shape:

```json
[
  {
    "role": "user",
    "content": "Explain the difference between prefill and decode in three concise Chinese paragraphs."
  }
]
```

Text-only fixtures are the scope for this design. Multimodal, tool-calling, document/RAG, and
preserved historical assistant reasoning fixtures are out of scope for the first replacement.

The existing fixture `.txt` files are replaced by `.messages.json` files. Human-readable text mirrors
are not part of the first replacement so the committed fixture source remains unambiguous.

### D3. Fixture Set Replacement

The existing `m2.8-v1` fixture set is replaced in-place. The old raw-text `m2.8-v1` meaning is
invalid after this change.

The current HEAD after implementation must not contain committed baseline summaries or readiness
claims based on the old raw-text fixture ids. They must be deleted or regenerated from the new
chat-template fixtures.

### D4. Chat Template Rendering

Python fixture tooling renders ids with the local Hugging Face tokenizer:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=False,
)
```

The rendered prompt must include the assistant generation prefix and the non-thinking marker
sequence emitted by the Qwen3.6 chat template:

```text
<|im_start|>assistant
<think>

</think>

```

`enable_thinking=false` is the default and the required M3 gate setting. This makes decoded output
represent the assistant answer instead of mostly reasoning content.

### D5. Special Tokens

Python fixture tooling must not add extra tokenizer-level special tokens on top of the chat
template. The chat template owns the Qwen chat markers.

The equivalent policy is:

```text
add_special_tokens = false
```

### D6. EOS and Stop Token Policy

The e2e benchmark stop policy is upgraded from one optional EOS token to a list of stop token ids.
For Qwen3.6-27B, the default stop token ids come from `generation_config.json`:

```json
[248046, 248044]
```

Where:

- `248046` is `<|im_end|>`
- `248044` is `<|endoftext|>`

The canonical C++ CLI exposes repeated `--stop-token-id <id>` flags. The report and schema record
the normalized list in argument order with duplicate ids removed.

For compatibility, the legacy `--eos-token-id <id>` flag remains as a deprecated alias for exactly
one stop token. New docs and generated commands use `--stop-token-id` only. The canonical report
field is `stop_token_ids`.

### D7. Manifest Provenance

The prompt fixture manifest must make chat-template rendering auditable from git. Each case records:

- case name
- messages JSON path
- `.ids` path
- prompt token count
- messages SHA256
- rendered prompt SHA256
- `.ids` SHA256
- prompt format: `qwen3.6-chat-template`
- `add_generation_prompt: true`
- `add_special_tokens: false`
- `chat_template_kwargs: {"enable_thinking": false}`

The manifest-level tokenizer block records:

- tokenizer source: `local_hf`
- tokenizer model id: `Qwen/Qwen3.6-27B`
- tokenizer path redacted in committed files
- `tokenizer.json` SHA256
- `tokenizer_config.json` SHA256
- `special_tokens_map.json` SHA256
- `chat_template.jinja` SHA256, if present
- `generation_config.json` SHA256

The manifest-level generation block records:

- stop token ids
- decoded stop token names where known
- sampling policy note: M2.8 benchmark is greedy, so Qwen sampling defaults are provenance only

### D8. Rendered Prompt Artifacts

The canonical committed input remains `.ids`. Rendered prompt text is not required to be committed
for every case, but tooling must be able to emit it for audit and decode sidecars.

Decoded sidecars should include, for each case:

- generated text with `skip_special_tokens=false`
- generated text with `skip_special_tokens=true`
- the rendered prompt text or a path to it
- the same prompt-format and chat-template metadata as the manifest

This keeps correctness based on token ids while making human review practical.

### D9. Report Schema

Raw e2e reports must replace or extend the old prompt fields with chat-template-aware identity:

```json
{
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
  "stop_token_ids": [248046, 248044]
}
```

Comparison tooling must treat changes to these fields as case identity changes.

### D10. Old Baselines

Existing raw-text baseline summaries, decoded sidecars, and readiness evidence are invalid after
the fixture replacement. The implementation must regenerate the M3 gate evidence using:

- chat-template-rendered `m2.8-v1` fixtures
- `enable_thinking=false`
- Qwen3.6 generation-config stop token ids
- current real q5090 weights

The final readiness document must not claim M3 readiness from old raw-id reports.

## Data Flow

1. Developer edits `<case>.messages.json`.
2. Python tooling loads the local Qwen3.6 tokenizer with `local_files_only=true`.
3. Python tooling renders messages with Qwen3.6 chat template, `add_generation_prompt=true`, and
   `enable_thinking=false`.
4. Python tooling writes canonical `.ids`.
5. Python tooling writes the manifest with tokenizer, chat-template, generation-config, messages,
   rendered-prompt, and ids provenance.
6. `qus_e2e_bench` reads `.ids` and stop token ids.
7. `qus_e2e_bench` runs `load`, `prefill`, and `decode_step`.
8. The report records prompt identity, stop policy, timing, generated token ids, memory, and
   determinism.
9. Decode tooling decodes generated ids and emits human-review sidecars.
10. Summary/readiness tooling validates the new fixture identity and rejects old raw-text evidence.

## Error Handling

Fixture tooling must fail if:

- a required messages JSON file is missing;
- messages JSON is not a non-empty array;
- message roles are not supported for the initial text-only scope;
- `apply_chat_template` returns no token ids;
- `enable_thinking` is not exactly `false` for the M3 gate fixture set;
- tokenizer or generation config hashes cannot be computed from existing local files;
- committed manifest data differs from regenerated data in check mode.

Benchmark tooling must fail before model execution if:

- no stop token ids are provided and no documented default can be resolved;
- stop token ids are malformed;
- `max_ctx < prompt_tokens + max(max_new_tokens - 1, 0)`;
- case identity fields in the fixture manifest cannot be read.

Comparison and summary tooling must fail if:

- a report lacks `prompt_format`;
- `prompt_format` is not `qwen3.6-chat-template` for M3 gate artifacts;
- `chat_template_kwargs.enable_thinking` is not `false`;
- stop token ids differ between baseline and candidate reports;
- old raw-text baseline artifacts are used as current M3 gate evidence.

## Testing Strategy

Unit tests cover:

- messages JSON parsing and validation;
- `apply_chat_template` invocation with `add_generation_prompt=true` and `enable_thinking=false`;
- manifest hash fields and check-mode stale detection;
- stop token id parsing and normalization in C++;
- report serialization of `stop_token_ids` and prompt-format identity;
- comparison failure on prompt-format, messages hash, rendered prompt hash, or stop-token changes;
- summary rejection of old raw-text fixture evidence.

Integration verification covers:

- regenerating fixtures from the local Qwen3.6 tokenizer;
- running a short real-weight smoke with the new `cn_short` fixture;
- decoding generated output and checking it is readable assistant content, not primarily a thinking
  block;
- regenerating the M3 gate baseline summary from new fixtures.

No correctness gate depends on decoded text semantics. Decoded text remains a human smoke aid.

## Non-Goals

- Adding a C++ tokenizer.
- Adding Jinja rendering to C++.
- Supporting multimodal chat fixtures in this replacement.
- Supporting tool-call fixtures in this replacement.
- Preserving old raw-text baseline artifacts as current evidence.
- Matching vLLM sampling output; the benchmark remains greedy.

## Approval State

Approved design decisions:

- C++ consumes ids only.
- Python fills chat template, thinking, EOS, and provenance gaps.
- Default M3 gate fixture mode is `enable_thinking=false`.
- Old raw-text fixture data and baseline evidence are not preserved as current evidence.

Open questions: none.
