# Qwen3.6-27B target parity tools

These tools compare the independent `.ninfer` Python reference, the C++ target diagnostic, and the
source BF16 Vision tower at matching semantic boundaries.

Create a C++ activation dump with the target-private tool:

```bash
build/tools/ninfer-qwen3_6_27b-dump \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --ids 248045,846,198,5834,248046,198 --decode 2 --greedy \
  --activation-dump /tmp/cpp-dump
```

Create the corresponding Python reference dump, then compare structured records:

```bash
python -m tools.parity.qwen3_6_27b_rtx5090.activations \
  /tmp/reference-dump /tmp/cpp-dump
```

The Text comparator is a gate. Its checked-in layer-tap rules require relative RMS/cosine of
`0.01/0.9999` for embeddings, `0.25/0.94` for every mixer, MLP, and final norm, and `0.35/0.90`
for logits. Missing or unexpected tensors, a missing requested phase, shape/size mismatch,
non-finite values, an unregistered tap, or a tolerance failure produces a nonzero exit. Reports
retain max-absolute, RMS, relative-RMS, and cosine metrics.

Compare quantized artifact Vision activations with the source BF16 tower:

```bash
python -m tools.parity.qwen3_6_27b_rtx5090.vision \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --model-dir /path/to/Qwen3.6-27B/base-hf-bf16 \
  --messages messages.json --ninfer-dump /tmp/ninfer-vision
```

These diagnostics report numerical differences. They do not require exact generated-token equality
between independent Python and C++ execution paths.

Run the mixed image/video frontend, Vision, composed-embedding, and `k=1` full-head MTP gate
sequentially (the Python model is released before the C++ diagnostic starts):

```bash
python -m tools.parity.qwen3_6_27b_rtx5090.vision_mtp \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --cpp build/tools/ninfer-qwen3_6_27b-dump \
  --messages messages.json --prefill-chunk 1024 --kv-dtype bf16 \
  --mtp-draft-tokens 1 --proposal-head full \
  --work-dir /tmp/vision-mtp --output /tmp/vision-mtp/report.json
```

The runner requires exact frontend token IDs, token types, three-axis positions, and `rope_delta`.
It selects the diagnostic's compact `vision-mtp` dump level, which records the representative
Vision tensors and first composed Text embedding without writing unrelated Text-layer tensors.
Activation rules are fixed in `vision_mtp.py`: patch embedding uses relative RMS/cosine limits
`0.03/0.999`, blocks 0/13/26 use `0.06/0.995`, `0.18/0.97`, and `0.25/0.94`, and the Vision
merger plus composed Text embedding use `0.25/0.94`. Missing records, shape mismatches, non-finite
values, tolerance failures, or failure to execute a `k=1` proposal round produce a nonzero exit.
