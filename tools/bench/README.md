# Bench Tools

These tools support the M2.8 pre-M3 benchmark standard.

## Tokenizer Policy

Tokenizer-dependent commands use a local Hugging Face tokenizer directory. Resolution order:

1. `--tokenizer-path`
2. `QUS_TOKENIZER_PATH`
3. fail

The tools use `local_files_only=True` and must not download from the network.

Prompt fixture generation uses Qwen3.6 chat-template sources: each case is a `.messages.json` chat
message file paired with a canonical `.ids` file. Token ids are rendered with
`tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` from the local
tokenizer path; tokenizer-level special tokens are not added separately.

## Regenerate Fixtures

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --fixture-dir bench/fixtures/prompts
```

Check committed fixtures:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --fixture-dir bench/fixtures/prompts \
  --check
```

## Decode E2E Reports

```bash
python3 tools/bench/decode_e2e_report.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --report profiles/e2e/example.json
```

Decoded text is human-smoke-only. Each decoded repeat writes `repeat_<n>.raw.txt` and
`repeat_<n>.clean.txt` sidecars. Correctness gates use token ids and report comparison.

E2E benchmark invocations for Qwen3.6 chat-template fixtures should pass both stop tokens:

```bash
--stop-token-id 248046 --stop-token-id 248044
```
