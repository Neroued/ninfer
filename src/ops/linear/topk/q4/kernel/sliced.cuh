#pragma once
#include "ops/common/cuda/reduce.cuh"
#include "core/weight.h"
#include "ops/linear/topk/dispatch.h"

#include "core/device.h"
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"

#include <cstdint>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

struct Q4TopKOutput {
    std::uint64_t* partial_keys;
    const std::int32_t* row_to_global_ids;
    std::int32_t producer_groups;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const int group = row / kLinearTopK;
        const int rank  = row % kLinearTopK;
        const std::int64_t offset =
            (static_cast<std::int64_t>(token) * producer_groups + group) * kLinearTopK + rank;
        partial_keys[offset] = common::score_id_order_key(value, row_to_global_ids[row]);
    }
};

} // namespace
} // namespace ninfer::ops::detail
