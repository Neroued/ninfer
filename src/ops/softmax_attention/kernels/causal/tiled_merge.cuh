#pragma once

#include "ops/softmax_attention/kernels/causal/ragged.h"

#include "ops/kv_cache/kernels/hadamard.cuh"

#include "core/device.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include <math_constants.h>
#include <stdexcept>

namespace ninfer::ops::detail {

// One warp combines up to 32 partitions for a complete D256 row, including
// K8V4's inverse rotation in FP32.
#include "ops/softmax_attention/kernels/causal/tiled_merge.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/tiled_merge.inl"
#undef NINFER_CAUSAL_RAGGED

template <class G, bool InverseRotation, class... Metadata>
void launch_causal_tiled_merge(const CausalAttentionOperands& p, const std::int32_t* valid_columns,
                               CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream, Metadata... metadata) {
    if (partition.capacity < 1 || partition.capacity > common::kWarpSize)
        throw std::invalid_argument("tiled attention merge: partitions exceed one warp");
    const dim3 grid       = sizeof...(Metadata) == 0 ? dim3(common::div_up(p.width * G::QHeads, 8))
                                                     : dim3(p.width, G::QHeads / 8);
    constexpr auto kernel = [] {
        if constexpr (sizeof...(Metadata) == 0)
            return causal_tiled_merge_kernel<G, InverseRotation>;
        else
            return causal_tiled_merge_ragged_kernel<G, InverseRotation>;
    }();
    kernel<<<grid, 256, 0, stream>>>(partial, p.positions, valid_columns, p.width, partition, p.out,
                                     metadata...);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
