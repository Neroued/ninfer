#pragma once

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
template <class G, bool InverseRotation>
__launch_bounds__(256) __global__
    void causal_tiled_merge_kernel(CausalPartialView partial, const std::int32_t* positions,
                                   const std::int32_t* valid_columns, int width,
                                   CausalKvPartition partition, __nv_bfloat16* out) {
    const int lane  = threadIdx.x & 31;
    const int row   = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int head  = row % G::QHeads;
    const int token = row / G::QHeads;
    if (token >= width) return;
    if (valid_columns && token >= valid_columns[0]) {
#pragma unroll
        for (int d = lane; d < 256; d += 32)
            causal_store_output(out + causal_q_index<G>(head, d, token), 0.0F);
        return;
    }
    const int splits        = partition.active(positions[width - 1] + 1);
    const auto stat         = causal_stat_index<G>(head, token, lane, width);
    const float m           = lane < splits ? partial.maximum[stat] : -CUDART_INF_F;
    const float l           = lane < splits ? partial.sum[stat] : 0.0F;
    const float maximum     = common::warp_max(m);
    const float weight      = l > 0.0F ? expf(m - maximum) : 0.0F;
    const float denominator = common::warp_sum(l * weight);
    float values[8]{};
    for (int split = 0; split < splits; ++split) {
        const float w     = __shfl_sync(0xffffffffU, weight, split);
        const auto offset = causal_partial_index<G>(head, lane, token, split, width);
#pragma unroll
        for (int i = 0; i < 8; ++i) values[i] += partial.acc[offset + i * 32] * w;
    }
    const float inverse = denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
#pragma unroll
    for (float& value : values) value *= inverse;
    if constexpr (InverseRotation) kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        causal_store_output(out + causal_q_index<G>(head, lane + i * 32, token), values[i]);
}

template <class G, bool InverseRotation>
void launch_causal_tiled_merge(const CausalAttentionOperands& p, const std::int32_t* valid_columns,
                               CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream) {
    if (partition.capacity < 1 || partition.capacity > common::kWarpSize)
        throw std::invalid_argument("tiled attention merge: partitions exceed one warp");
    causal_tiled_merge_kernel<G, InverseRotation>
        <<<common::div_up(p.width * G::QHeads, 8), 256, 0, stream>>>(
            partial, p.positions, valid_columns, p.width, partition, p.out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
