#include "kernels/linear/gemm/linear_rowsplit_q4_gate_up_silu_gemm_mma_folded.cuh"

#include "kernels/common/math.h"
#include "ninfer/core/device.h"

namespace ninfer::kernels::detail {
namespace {

using GateUpCfg = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;

void launch_folded(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = GateUpCfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, GateUpCfg::BN)));
    const bool full_tiles = (out.ne[0] % PM) == 0 && (t % GateUpCfg::BN) == 0;
    if (full_tiles) {
        linear_rowsplit_q4_gate_up_silu_gemm_mma_folded_kernel<GateUpCfg, true>
            <<<grid, GateUpCfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], t,
                weight.padded_shape[1]);
    } else {
        linear_rowsplit_q4_gate_up_silu_gemm_mma_folded_kernel<GateUpCfg, false>
            <<<grid, GateUpCfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], t,
                weight.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear_rowsplit_q4_gate_up_silu_gemm_mma_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& out, cudaStream_t stream) {
    launch_folded(x, weight, out, stream);
}

} // namespace ninfer::kernels::detail
