# Qwen3.6-35B-A3B Python reference

This is the complete target-private artifact-native Python reference for the accepted
`qwen3_6_35b_a3b` artifact profile. It runs the 40-layer Text decoder, top-8 routed and
gated shared experts, Vision tower and 2048-wide merger, one-layer sparse-MoE MTP model, sampling,
and persistent KV/GDN state directly from the `.ninfer` object layouts.

The accepted artifact also contains the target-private DFlash companion tensors. The reference
binds their complete typed weight views and can include them in an explicit weight-memory plan, but
the current `RefModel` does not make them resident and has no DFlash state, schedule, or execution
path.

The reference is an independent diagnostic implementation for the registered artifact. The C++
Engine target is registered separately; this Python route is not its generated-token golden and
does not define equality across different numerical paths. It does not need the original Hugging
Face checkpoint at inference time: tokenizer, generation, template, image, and video resources are
read from the artifact.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
python3 \
  -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The input is exactly one of `--prompt`, `--ids`, or `--messages`. Structured messages may contain
images and videos in the normal Transformers format. Thinking is enabled by default and can be
disabled with `--no-thinking`.

MTP is disabled by default. Enable one to five draft positions with `--mtp-draft-tokens 1..5`;
`--draft-head` selects the artifact's optimized proposal head. Target verification always uses the
full output head. The CLI reports proposal acceptance, timing, memory planning, Vision work, and
peak CUDA allocation.

Important runtime controls include:

- `--gpu-memory auto|24GiB` and `--headroom 2GiB`;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`;
- `--greedy` or sampling overrides;
- `--vision-attention-limit N`;
- `--activation-dump DIR --dump-level layer|op`.

The sparse-MoE path resolves router ids first and materializes only the row spans of experts that
receive tokens. It never expands a complete routed bank. Vision processes each image or video item
independently, retains only its merged BF16 embeddings on the host, and releases its streaming
weight store before Text preparation. Prompt chunks transfer only the required image/video rows;
multimodal MTP uses the same composed Vision embedding for shifted inputs.

Decode, MTP proposal/verification, and both output heads use the contract's FP32-dequant Small-T
projection profile. Text/MTP prefill retains the separate BF16 MMA-weight rounding boundary.
