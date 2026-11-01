#include "ops/normalization/rmsnorm/kernel/fused.cuh"
#include "core/device.h"
#include <algorithm>

namespace ninfer::ops::detail::fused_rmsnorm {
void launch_residual(Arguments a, bool vector_aligned, cudaStream_t stream) {
    const int d = a.dimension, m = a.columns;
    if (d == 4096 && vector_aligned)
        vector_kernel<Form::Residual, 4096, 512, 8><<<m, 512, 0, stream>>>(a);
    else if (d == 1024 && vector_aligned) {
        if (m > 16 && m <= 256)
            vector_kernel<Form::Residual, 1024, 256, 4><<<m, 256, 0, stream>>>(a);
        else
            vector_kernel<Form::Residual, 1024, 128, 8><<<m, 128, 0, stream>>>(a);
    } else
        generic_kernel<Form::Residual><<<m, 256, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::fused_rmsnorm
