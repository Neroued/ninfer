#pragma once

#include "core/device.h"
#include "ops/linear/q8/q8_ksplit_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

// Shape files select the capacity and physical schedule. The launcher supplies the live T,
// so every T in the selected interval reuses this compiled instance.
template <class Geometry, int ColumnCapacity, class Schedule,
          class Epilogue = Q8KSplitStoreEpilogue>
void launch_q8_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    if (weight.padded_shape[1] != Geometry::kInputRows) {
        throw std::invalid_argument("q8 K-split: padded K differs from the registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_ksplit_mma_kernel<Geometry, ColumnCapacity, Schedule, Q8ContiguousOutput, Epilogue,
                         Q8KSplitIdentityRows, false, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Epilogue{},
            Q8KSplitIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
