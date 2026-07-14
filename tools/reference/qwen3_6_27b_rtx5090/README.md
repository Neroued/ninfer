# Qwen3.6-27B RTX 5090 Python reference

This is the complete target-private Text, Vision, MTP, sampling, state, and weight-residency
reference over a native `.ninfer` artifact. It retains the previous reference computation and
performance mechanisms while replacing the q5090 v4.2 file boundary with typed artifact bindings.

It does not need the original Hugging Face checkpoint at inference time. `Frontend` materializes the
tokenizer, chat template, generation defaults, and image/video processor resources embedded in the
artifact, then delegates those functions to Transformers.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.reference.qwen3_6_27b_rtx5090 \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 512
```

The input is exactly one of `--prompt`, `--ids`, or `--messages`. Structured messages may contain
images and videos in the normal Transformers format. Thinking is enabled by default and can be
disabled with `--no-thinking`.

MTP is disabled by default. Enable one to five draft positions with
`--mtp-draft-tokens 1..5`; `--draft-head` selects the artifact's optimized proposal head. Target
verification always uses the full output head. The CLI reports round counts, per-position accepted
drafts, fallback steps, timing, memory planning, and peak CUDA allocation.

Important runtime controls include:

- `--gpu-memory auto|24GiB` and `--headroom 2GiB`;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`;
- `--greedy` or sampling overrides for temperature, top-p, top-k, and penalties;
- `--vision-attention-limit N`;
- `--activation-dump DIR --dump-level layer|op`.

Quantized Text matrices retain the decoded/packed/streamed residency plan and compiled low-bit
codec. Vision decodes large matrices one at a time and releases its weight store before Text weight
preparation. Multimodal MTP uses the composed Vision embedding for the shifted input, including at
prefill chunk boundaries.

Target-specific numerical and current-Engine parity commands live in
`tools/parity/qwen3_6_27b_rtx5090/`.
