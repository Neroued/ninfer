# Qwen3.6-27B target parity tools

These report-only commands compare the native `.ninfer` reference, the source BF16 Vision tower,
and the still-current `.qus` C++ Engine where their boundaries overlap.

Compare structured activation dumps:

```bash
python -m tools.parity.qwen3_6_27b_rtx5090.activations \
  /tmp/reference-dump /tmp/candidate-dump
```

Compare current C++ preprocessing with the library frontend embedded in `.ninfer`:

```bash
python -m tools.parity.qwen3_6_27b_rtx5090.preprocess \
  --engine-weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --artifact out/qwen3_6_27b_rtx5090.ninfer \
  --messages messages.json --no-thinking
```

Compare quantized artifact Vision activations with the source BF16 tower:

```bash
python -m tools.parity.qwen3_6_27b_rtx5090.vision \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --model-dir /path/to/Qwen3.6-27B/base-hf-bf16 \
  --messages messages.json --ninfer-dump /tmp/ninfer-vision
```

These diagnostics report numerical differences. They do not require exact generated-token equality
between independent Python and C++ execution paths.
