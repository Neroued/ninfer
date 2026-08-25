# GPU runbook: decode micro-opts A/B (#69 + #67)

This round is implementation and compile only. Do not boot a server or acquire the
GPU lock until the coordinator schedules an exclusive window. The Qwen3.8-27B
NVFP4 artifact is about 20 GiB and does not fit the 16 GiB compact cap.

## Arms

| Arm | Git | Binary |
|---|---|---|
| A baseline | `9dda66511c81e72686ba6b610256625a8af603a7` (`feat/prefix-seed-store` without the thinking dialect) | rebuild `ninfer-serve` from that commit |
| B treatment | `task/issue-1-decode-micro-opts` HEAD | rebuild `ninfer-serve` from this branch |

Do not base either arm on the vLLM-dialect merge. Decode A/B must stay pure.

Rebuild image `ninfer:seedstore` from the arm under test. Coordinator owns the
rebuild and the `:8018` lane. Never stop production containers
(`sglang-qwen38` on `:8016`, embeddings, whisper).

## Server flags (identical both arms)

No `--preserve-thinking`. Model ID `qwen3.8-27b`. JSONL log required.

```text
--host 127.0.0.1 --port 8018
--max-context 131072 --kv-capacity 1048576 --max-concurrency 8
--spec mtp --draft-tokens 5 --lm-head-draft
--kv-dtype int8 --prefill-chunk 2048 --vision --cors
--prefix-cache-mib 4096
--request-log-jsonl /tmp/ninfer-decode-ab.jsonl
```

## A/B protocol (binding)

- Identical config both arms.
- At least **3 full process boots per arm** (6 boots total). Interleave
  A/B/A/B/A/B in one exclusive GPU session so thermal/clock drift is visible.
- Client load: **c=1, n=16**. One in-flight request. Sixteen serial Chat
  Completions per boot.
- Prompts: production-shaped chat from `examples/cli/messages/scenario_*.json`
  (code, story, translation, structured). Cycle the twelve scenario fixtures
  and repeat the first four to make sixteen. `max_completion_tokens=256`.
  Do not send `enable_thinking` / `chat_template_kwargs` (this arm predates
  the dialect).
- Speculative: MTP-5 as in the server flags above. Do **not** pass `--greedy`
  on the A/B throughput boots (MTP acceptance luck is why boot variance is
  large).
- Metric: per-request decode tok/s from the process log line
  `decode=<rate>` which is `(completion_tokens - 1) / timings_seconds.decode`.
  Boot score = median of the 16 request rates. Arm score = median of the 3
  boot scores. Also record all three boot scores so spread is visible.
- **Noise floor:** boot-to-boot decode on this card has been 133.5–155.9 tok/s
  from stochastic MTP acceptance. A **single-boot** delta under about 10% is
  noise. Do not accept or reject on one boot.
- Pass: arm B median is not a regression versus arm A after three boots.
  The upstream claims (+3.7% MoE L2 prefetch, +2.3% node removal) were
  measured on an RTX 5090 at T=1; they are unverified on the PRO 6000.
  Adopt only if the 3-boot median does not regress. Drop the pick if it
  regresses.

### Throughput commands

```bash
BASE=http://127.0.0.1:8018
MODEL=qwen3.8-27b
PROMPTS=(
  examples/cli/messages/scenario_code_cuda.json
  examples/cli/messages/scenario_code_python.json
  examples/cli/messages/scenario_code_typescript.json
  examples/cli/messages/scenario_story_zh_scifi.json
  examples/cli/messages/scenario_story_en_mystery.json
  examples/cli/messages/scenario_story_zh_dialogue.json
  examples/cli/messages/scenario_translation_zh_en.json
  examples/cli/messages/scenario_translation_en_zh.json
  examples/cli/messages/scenario_translation_markdown.json
  examples/cli/messages/scenario_structured_jsonl.json
  examples/cli/messages/scenario_structured_csv.json
  examples/cli/messages/scenario_structured_sql.json
)
# Repeat first four to reach n=16.
for i in $(seq 0 15); do
  msg="${PROMPTS[$((i % 12))]}"
  python3 - "$BASE" "$MODEL" "$msg" <<'PY'
import json, sys, urllib.request
base, model, path = sys.argv[1], sys.argv[2], sys.argv[3]
body = {
  "model": model,
  "messages": json.load(open(path, encoding="utf-8")),
  "max_completion_tokens": 256,
  "stream": False,
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=600) as r:
    json.load(r)
PY
done
```

Parse `/tmp/ninfer-decode-ab.jsonl` events with `"event":"request_done"`:

```text
decode_tok_s = (result.completion_tokens - 1) / timings_seconds.decode
```

## Greedy bit-identity: cold vs seeded

Run **once per arm**, not as part of the 3-boot throughput. Use `--greedy`
in addition to the flags above (same prefix-cache). Cold-start the process.

Prompt: a two-message chat so the seed frontier is the first user turn.
Send the same body twice.

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Reply with the single word ping."}],
  "max_completion_tokens": 32,
  "temperature": 0,
  "seed": 0
}
```

Pass only if all of:

1. Both HTTP 200.
2. `choices[0].message.content` is byte-identical across the two responses.
3. First `request_done.prefix_reuse_path` is `full_reset`.
4. Second `request_done.prefix_reuse_path` is `seed_prefix` (or a restore_*
   path). `full_reset` on the second request means the seed-store oracle
   failed — fail the pick even if content matches by chance.
5. Repeat the same pair on arm A and arm B. Content must match **across
   arms** as well (prefetch and node folding must not change tokens).

If (2) or (5) fails, drop the pick. If (4) fails, the seed-store contract
regressed and the pick is not adoptable.

### Long greedy generation

A 32-token probe cannot catch the 1-ulp class recorded in `src/ops/kernel/rope.cuh`:
in-kernel rotary drift that "surfaces as a diverged token deep inside long greedy
generations." Add one long greedy request per arm, same cold-start process as
the short probe (or a third request after it).

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a detailed technical explanation of speculative decoding with MTP, including a worked numeric example."}],
  "max_completion_tokens": 2048,
  "temperature": 0,
  "seed": 0
}
```

Send it twice (cold, then seeded) on arm A and arm B. Pass only if:

1. All four HTTP 200, `finish_reason` is `stop` or `length`.
2. `choices[0].message.content` is byte-identical across the four responses
   (both arms, cold and seeded).
3. Seeded `prefix_reuse_path` is not `full_reset`.

A mismatch anywhere in the 2048-token body fails the pick even if the 32-token
probe passed.

## GPU lock

Only the coordinator runs this. Before any process that allocates GPU
memory: `nvidia-smi` shows at least 20 GiB free; `mkdir`
`C:\Users\igorl\.ninfer-gpu.lock` (retry 60 s up to 30 min); remove the
lock directory after, success or failure. Never stop or restart the
production docker stack.
