#pragma once
#include "core/device.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

namespace ninfer::ops::detail {
// Exact column counts retain the measured SIMT register and instruction footprint.
template <int InputRows, int Tokens, int KWarps>
void launch_q5_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert(InputRows % 1024 == 0);
    static_assert(KWarps == 2 || KWarps == 4);
    if constexpr (KWarps == 2) {
        q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Tokens, InputRows / 1024,
                                            InputRows><<<weight.n, 64, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            weight.n, InputRows, x.ne[1], weight.padded_shape[1], InputRows / 1024);
    } else {
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, InputRows / 1024,
                                            InputRows><<<weight.n, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            nullptr, weight.n, static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16)), InputRows,
            x.ne[1], weight.padded_shape[1], InputRows / 1024);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
