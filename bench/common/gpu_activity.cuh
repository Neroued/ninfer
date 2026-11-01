#pragma once

#include "core/device.h"

namespace ninfer::bench {
namespace detail {
__device__ inline unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Register-only work gives warm and cold measurements the same activity prelude.
// It neither warms the operands nor evicts their cache lines, and does not lock clocks.
static __global__ void condition_kernel(unsigned long long nanoseconds) {
    const auto begin = global_nanoseconds();
    float value      = float(threadIdx.x) + 1.F;
    do {
#pragma unroll
        for (int i = 0; i < 16; ++i) value = __fmaf_rn(value, 1.000001F, 0.000001F);
        asm volatile("" : "+f"(value));
    } while (global_nanoseconds() - begin < nanoseconds);
}
} // namespace detail

inline void condition_gpu(CudaExecutionView execution, int microseconds) {
    if (!microseconds) return;
    detail::condition_kernel<<<2 * execution.multiprocessor_count(), 256, 0, execution.stream()>>>(
        static_cast<unsigned long long>(microseconds) * 1000);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::bench
