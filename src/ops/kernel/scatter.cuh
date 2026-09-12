#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void scatter_bf16x8_kernel(const uint4* src, const std::int32_t* indices, uint4* dst,
                                      std::int32_t vectors_per_column) {
    const std::int32_t src_col  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col  = indices[src_col];
    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * vectors_per_column;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * vectors_per_column;
    for (std::int32_t vector = static_cast<std::int32_t>(threadIdx.x); vector < vectors_per_column;
         vector += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + vector] = src[src_base + vector];
    }
}

__global__ void scatter_bf16x2_kernel(const __nv_bfloat162* src, const std::int32_t* indices,
                                      __nv_bfloat162* dst, std::int32_t pairs) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col = indices[src_col];

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * pairs;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * pairs;
    for (std::int32_t pair = static_cast<std::int32_t>(threadIdx.x); pair < pairs;
         pair += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + pair] = src[src_base + pair];
    }
}

__global__ void scatter_scalar_kernel(const __nv_bfloat16* src, const std::int32_t* indices,
                                      __nv_bfloat16* dst, std::int32_t d) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col = indices[src_col];

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * d;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * d;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < d;
         row += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + row] = src[src_base + row];
    }
}

inline constexpr int kScatterBatchThreads = 128;

// A 2 KiB feature tile exposes parallelism even when the block has few columns.
__launch_bounds__(kScatterBatchThreads) __global__
    void scatter_bf16_batch_kernel(const uint4* __restrict__ source,
                                   const std::int32_t* __restrict__ lanes,
                                   const std::int32_t* __restrict__ valid_columns,
                                   uint4* __restrict__ destination, std::int32_t vectors_per_column,
                                   std::int32_t width, std::int64_t destination_column_stride,
                                   std::int64_t destination_lane_stride) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t batch  = static_cast<std::int32_t>(blockIdx.y);
    if (column >= valid_columns[batch]) return;
    const std::int64_t source_base =
        (static_cast<std::int64_t>(batch) * width + column) * vectors_per_column;
    const std::int64_t destination_base =
        static_cast<std::int64_t>(lanes[batch]) * destination_lane_stride +
        static_cast<std::int64_t>(column) * destination_column_stride;
    // Keep the tile loop CTA-uniform; only the store has a per-thread tail.
    for (std::int64_t base = static_cast<std::int64_t>(blockIdx.z) * kScatterBatchThreads;
         base < vectors_per_column;
         base += static_cast<std::int64_t>(gridDim.z) * kScatterBatchThreads) {
#pragma unroll
        for (int offset = threadIdx.x; offset < kScatterBatchThreads;
             offset += kScatterBatchThreads) {
            const std::int64_t vector = base + offset;
            if (vector < vectors_per_column)
                destination[destination_base + vector] = source[source_base + vector];
        }
    }
}

} // namespace ninfer::ops
