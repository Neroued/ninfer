#pragma once
// qus::kernels — W4A16 GEMM / GEMV (the decode workhorse): bf16 activations x packed-4bit
// weights (+scales) -> bf16. Generic API; specialized impls routed by dims+phase.
// Used by: all linear projections + lm_head. See docs/qwen3.6-27b-architecture.md §10.1.
// NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): void w4a16_gemm(const Tensor& a, const QuantWeight& w, Tensor& out, stream);
//             (decode: T=1 GEMV memory-bound; prefill: T>1 GEMM compute-bound)

}  // namespace qus::kernels
