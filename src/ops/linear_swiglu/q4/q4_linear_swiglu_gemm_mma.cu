#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"

#include "ops/common/math.h"
#include "core/device.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using GateUpCfg = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;

void launch_folded(Q4KernelVariant variant, const Tensor& x, const Weight& weight, Tensor& out,
                   cudaStream_t stream) {
    constexpr int PM = GateUpCfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, GateUpCfg::BN)));
    if (variant == Q4KernelVariant::Full) {
        q4_linear_swiglu_mma_split_half_pair_kernel<GateUpCfg, true>
            <<<grid, GateUpCfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], t,
                weight.padded_shape[1]);
    } else if (variant == Q4KernelVariant::Predicated) {
        q4_linear_swiglu_mma_split_half_pair_kernel<GateUpCfg, false>
            <<<grid, GateUpCfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], t,
                weight.padded_shape[1]);
    } else {
        throw std::invalid_argument("q4 linear_swiglu MMA requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(Q4KernelVariant variant, const Tensor& x,
                                                          const Weight& weight, Tensor& out,
                                                          cudaStream_t stream) {
    launch_folded(variant, x, weight, out, stream);
}

} // namespace ninfer::ops::detail
