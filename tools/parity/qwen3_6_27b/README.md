# Qwen3.6-27B target parity tools

These tools compare the independent `.ninfer` Python reference, the C++ target diagnostic, and the
source BF16 Vision tower at matching semantic boundaries.

Create matching Python and C++ activation dumps. Both executions and the comparator must receive
the same K:V cache format:

```bash
KV_DTYPE=bf16:int8
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ninfer \
  --ids 248045,846,198,5834,248046,198 --decode 2 --greedy \
  --prefill-chunk 1024 --kv-dtype "$KV_DTYPE" \
  --activation-dump /tmp/reference-dump

build/tools/ninfer-qwen3_6_27b-dump \
  --weights out/qwen3_6_27b.ninfer \
  --ids 248045,846,198,5834,248046,198 --decode 2 --greedy \
  --prefill-chunk 1024 --kv-dtype "$KV_DTYPE" \
  --activation-dump /tmp/cpp-dump

python -m tools.parity.qwen3_6_27b.activations \
  /tmp/reference-dump /tmp/cpp-dump --kv-dtype "$KV_DTYPE"
```

The Text comparator is a gate. Its checked-in layer-tap rules require relative RMS/cosine of
`0.01/0.9999` for embeddings, `0.25/0.94` for every mixer, MLP, and final norm, and `0.35/0.90`
for logits. Missing or unexpected tensors, a missing requested phase, shape/size mismatch,
non-finite values, an unregistered tap, or a tolerance failure produces a nonzero exit. Reports
retain max-absolute, RMS, relative-RMS, and cosine metrics. A missing or different `kv_dtype` in
either manifest is rejected before numerical comparison; cross-format differences are not valid
cross-runtime parity evidence and are never hidden by a looser tolerance.

Compare quantized artifact Vision activations with the source BF16 tower:

```bash
python -m tools.parity.qwen3_6_27b.vision \
  --weights out/qwen3_6_27b.ninfer \
  --model-dir /path/to/Qwen3.6-27B/base-hf-bf16 \
  --messages messages.json --ninfer-dump /tmp/ninfer-vision
```

These diagnostics report numerical differences. They do not require exact generated-token equality
between independent Python and C++ execution paths.

Run the mixed image/video frontend, Vision, composed-embedding, and `k=1` full-head MTP gate
sequentially (the Python model is released before the C++ diagnostic starts):

```bash
python -m tools.parity.qwen3_6_27b.vision_mtp \
  --weights out/qwen3_6_27b.ninfer \
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
