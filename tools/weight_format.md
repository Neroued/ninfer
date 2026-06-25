# Weight File Format — Python packer ↔ C++ runtime contract

> Status: **stub — to be specified.** The single fixed file the C++ runtime mmaps/loads (no
> runtime quant/relayout). Produced by `tools/pack/pack.py`; consumed by `qus::WeightStore`
> (`include/qus/core/weight_store.h`).

## TODO: specify
- **Header:** magic, version, endianness, model id + dims (validated against `ModelConfig`).
- **Tensor table:** per tensor — name/role, layer index, dtype / quant-layout tag, shape,
  byte offset, scale offset/shape, group size.
- **Payload:** packed-4bit blobs + scales + high-precision (bf16) "sensitive" tensors, in
  kernel-optimal layout; per-blob alignment (e.g. 256 B).

See `docs/qwen3.6-27b-architecture.md` §13 (tensor map + offline transforms) and
`docs/l0-infrastructure-design.md` §5.2 (WeightStore).
