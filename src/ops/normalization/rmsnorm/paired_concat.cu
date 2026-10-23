#include "ops/normalization/rmsnorm/kernel/fused.cuh"
#include "core/device.h"
#include <algorithm>

namespace ninfer::ops::detail::fused_rmsnorm {
void launch_paired(Arguments a, bool vector_aligned, cudaStream_t stream) {
    const int d = a.dimension, m = a.columns;
    const auto grid = static_cast<unsigned>(std::min<std::int64_t>(2LL * m, 2147483647));
    if (d == 4096 && vector_aligned) {
        if ((m >= 16 && m <= 64) || m >= 1024)
            vector_kernel<Form::Paired, 4096, 512, 8><<<grid, 512, 0, stream>>>(a);
        else
            vector_kernel<Form::Paired, 4096, 256, 8><<<grid, 256, 0, stream>>>(a);
    } else
        generic_kernel<Form::Paired><<<grid, 256, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::fused_rmsnorm
