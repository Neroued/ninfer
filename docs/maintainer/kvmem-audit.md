# KVMem integration and validation boundaries

## Supported execution

`--kvmem-window-pages` opts Qwen-family inference into a retrieval-managed GPU
working set. Zero retains dense execution. The window counts 64-token pages per
lane; one or two active lanes are supported. Model weights and the Host KV budget
are shared, while recurrent state, retrieval features and replay cursors belong
to each lane. Logical context capacity does not imply that all history is resident.

The supported combinations are ordinary decoding, MTP, and DFlash2 companions
whose draft layers are all local attention, with text or vision input. First-generation
DFlash, full-attention draft companions, scoring and more than two sparse lanes
are rejected. Dense execution retains its own existing capability rules.

## State and placement

- Capture accumulates represented pre-RoPE, post-normalization Q/K features.
  Model-owned 128-token retrieval blocks collect mean keys; the final user query
  supplies the query feature. New admissions reset retrieval ownership.
- Prefill maps only the next chunk's growth and retires committed overflow at
  chunk boundaries. Startup and request entitlement account for each lane's
  window, growth space and backend lead, rather than the whole logical history.
- Host replicas retain the packed device representation. Page tables mark absent
  pages with `kPagedKVPageHole`; attention never dereferences or assigns probability
  mass to such pages. All append destinations must be resident.
- Retrieval retains sink/recent pages and complete media groups. Mandatory media
  that cannot fit the window is rejected before execution. Original positions,
  including MRoPE coordinates, survive placement changes.
- Query replay restores recurrent and KV state and advances one bounded chunk at
  a time. Service projection includes replay work, including requests with only
  one output token available. Cancellation becomes visible at worker boundaries.
- Vision sessions retain the payloads required by replay. DFlash prefill uses its
  own per-chunk ingress with the correct state destination and backend table row;
  decode ingress is not borrowed for prefill or query replay.

## Numerical and model validation

The attention tests compare represented BF16/INT8/FP8/NVFP4/K8V4 cache inputs with
an independently decoded FP64 softmax oracle. Hole cases cover grouped decode,
tiled prompts, empty split ranges, mixed valid lengths, lane permutation and CUDA
Graph replay. Cache bytes, inert output columns and allocation guards are checked
separately. The feature accumulation Op also has an independent FP64 oracle.

[The E2E pipeline](../../tests/e2e/README.md) exercises actual generation, historical
recall, rolling windows, retrieval restore, cancellation/reuse and two-lane text/
vision isolation. DFlash2 coverage requires seven drafts and observed zero, partial
and full acceptance. A successful HTTP response alone is not a passing case.

Packed page copies can preserve stored bytes, but retrieval-selected attention is
not mathematically equivalent to dense attention over all history. Numerical Op
tests and controlled model regressions do not establish broad quality equivalence,
two full 256K visual requests, or an end-to-end performance gain. Qualification
applies only to the binary, artifact, configuration and workload in its report.
