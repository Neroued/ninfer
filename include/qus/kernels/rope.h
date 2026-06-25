#pragma once

// qus::kernels — partial RoPE + (interleaved) MRoPE: rotate the first rotary_dim=64 of 256
// head dims (NeoX split), pass through the rest; mrope_section [11,11,10]; applied to q,k
// AFTER q/k-norm. See docs/qwen3.6-27b-architecture.md §6.4/§8/§10.3.
// NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): rope_partial_mrope(q, k, positions, ...). v1 may use plain partial RoPE
//             (text-only) and add the 3-axis split with vision.

} // namespace qus::kernels
