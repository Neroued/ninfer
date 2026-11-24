# ThinkingCap Indexed88 v7.0 — NInfer Upstream Runtime

## Overview

This release ports the ThinkingCap 135K Indexed88 MTP5 direct proposal path to
NInfer upstream base `1882c6adbc98043fcf89f3589aa5addb3742777a`.
It is a specialized NInfer runtime for the compatible ThinkingCap 135K Q8
artifact, not a universal replacement for upstream NInfer.

## Features

- 135,168-row physical Q8 output head with hidden dimension 5,120
- Indexed88 direct proposal selection using 88,064 ranked token IDs
- MTP speculative decoding with up to five draft tokens
- NVIDIA Blackwell SM120a build, CUDA Graphs and NVFP4 KV support
- Upstream context-cache and checkpoint runtime

**Not included:** internal Stage 4A/4B frequency, coverage, full-head probe,
BIN telemetry or numeric-audit code.

## Reproducibility and requirements

- Upstream base: `1882c6adbc98043fcf89f3589aa5addb3742777a`
- Ranking asset: `assets/indexed88/selected_88064_135k_v64_40_R2.i32le`
- Ranking SHA256: `b578828c0b517ba47abe90a130fda7cde5a56d94add1afeac7623e7c8f6234b4`
- Ranking entries: 88,064 little-endian signed 32-bit integers
- Required environment: `NINFER_MTP_INDEXED_SHADOW=1`,
  `NINFER_MTP_INDEXED_MODE=indexed96`,
  `NINFER_MTP_INDEXED_RANKING=/absolute/path/to/ranking.i32le`
- Use `--spec mtp --draft-tokens 5 --proposal-head full --kv-dtype nvfp4`
  and a suitable `--prefill-chunk` (2048 was validated).
- The `indexed96` environment value is the implementation's mode selector;
  the ranking itself contains 88,064 IDs.

## Validation

A controlled 1,024-token Stage-3 vs public-v7 comparison on an NVIDIA RTX PRO
4000 Blackwell returned identical content and reasoning SHA256 hashes, token
counts, speculative draft counts (1,170) and accepted draft counts (788).
The public release candidate reported 107.99 tokens/s decode versus 108.76
for Stage 3. This single short test is not a statistical performance claim.

A separate 256-token smoke request completed successfully with MTP enabled.
The public release candidate must be validated against the target hardware
and artifact before deployment.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_TARGET=rtx5090 -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
cmake --build build --parallel 6 --target ninfer ninfer-serve
```

Alternatively, use the repository Dockerfile. Build tooling, GPU drivers,
and compatible CUDA runtime are required. No model weights are included.

## Provenance

Source tag: `thinkingcap-indexed88-v7.0-135k`.
Previous historical release: `thinkingcap-indexed88-v6.4-135k-r2`.
Model repository: https://huggingface.co/Schestex/ThinkingCap-Qwen3.8-27B-NInfer

The compatible experimental 135K Q8 artifact and its license are separate
from this source-code release. Review repository and model licensing before use.
