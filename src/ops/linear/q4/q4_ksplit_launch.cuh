#pragma once

#include <stdexcept>
#include "core/device.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"

namespace ninfer::ops::detail {
template <int OutputRows, int InputRows, int Capacity>
void launch_q4_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<OutputRows, InputRows>;
    static_assert(OutputRows % Q4KSplitMmaSchedule::kRowsPerCta == 0);
    if (weight.padded_shape[1] != InputRows)
        throw std::invalid_argument("q4 K-split: padded K differs from geometry");
    q4_ksplit_mma_kernel<Geometry, (Capacity + 7) / 8 * 8, Capacity, Q4KSplitStoreEpilogue,
                         Q4KSplitIdentityRows, true>
        <<<OutputRows / 16, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            {}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
