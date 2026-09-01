#pragma once

// Split-projection NVFP4 SwiGLU on Volta tensor cores (sm_70 only): two independent QPN2 launches
// -- one per weight half, using the unmodified plain-QPN2 kernel -- into fp32 scratch, then a
// small combine kernel applies silu(gate) * up in fp32 before the single BF16 round.
//
// Why not the fused kernel (nvfp4_linear_swiglu_volta_qpn.cuh)? Measured against it directly on
// the gate_up shape at T=4: the fused kernel reads 225.4 GB/s across the whole weight; two
// separate QPN2 passes at QPN2's own tuned schedule read 371.6 GB/s each. The fused kernel pays
// for computing both projections inside one CTA -- double the accumulators, double the decode
// registers -- and that costs more than the shared activation load saves. Splitting keeps each
// launch at QPN2's own measured-fastest register/occupancy profile; the only new cost is a
// trivial elementwise combine kernel (output-sized traffic, negligible next to the weight stream)
// and one extra kernel launch.
//
// Why fp32 scratch rather than composing linear() + silu_mul() (which was tried and reverted for
// the fused kernel's problem too): the same reasoning holds regardless of which QPN kernel writes
// the projection. linear()'s A16 output is BF16, and silu(gate) * up compounds two independently
// BF16-rounded operands multiplicatively, failing the correctness test by ~0.8% against a ~0.35%
// tolerance. Fp32 scratch is the fix either way -- it just also happens to fix the speed problem,
// because it lets each projection run through QPN2's own kernel unmodified instead of a doubled
// one.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

// Reads two fp32 [kIntermediate, T] planes (token-major, matching Nvfp4Fp32ContiguousOutput's
// store layout) and writes silu(gate) * up as BF16. Grid-stride over the whole [kIntermediate, T]
// extent; this is output-sized traffic, not weight-sized, so a simple 1D launch is enough.
__global__ void nvfp4_swiglu_fp32_combine_kernel(const float* __restrict__ gate,
                                                  const float* __restrict__ up,
                                                  __nv_bfloat16* __restrict__ out,
                                                  std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16_rn(silu(gate[i]) * up[i]);
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
