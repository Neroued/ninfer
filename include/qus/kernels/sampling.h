#pragma once
// qus::kernels — next-token selection: argmax over the 248320-vocab logits (greedy, v1).
// A full sampler (temp/top-k/top-p) is a deferred add-on. See docs/qwen3.6-27b-architecture.md §10.5.
// NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): int argmax_vocab(const Tensor& logits, stream);

}  // namespace qus::kernels
