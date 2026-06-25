#pragma once
// qus::kernels — activations / elementwise: silu_and_mul (SwiGLU: silu(gate)*up),
// sigmoid_gate_mul (full-attn output gate), residual add (often fused into norm),
// embedding_gather. See docs/qwen3.6-27b-architecture.md §6.6/§10.2.
// NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): silu_and_mul; sigmoid_gate_mul; embedding_gather.

}  // namespace qus::kernels
