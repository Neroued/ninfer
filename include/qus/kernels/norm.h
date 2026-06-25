#pragma once

// qus::kernels — RMSNorm variants: gemma_rmsnorm (1+w, fp32 var; optional fused residual add;
// also q/k-norm head-wise) and gated_rmsnorm (GDN output: plain w, * SiLU(z), over dv=128).
// See docs/qwen3.6-27b-architecture.md §5/§6.2/§10.2.
// NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): gemma_rmsnorm(x[,residual], weight) -> out;  gated_rmsnorm(o, z, weight) -> out.

} // namespace qus::kernels
