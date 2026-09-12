#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cub/block/block_merge_sort.cuh>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

inline constexpr int kMergeThreads        = 256;
inline constexpr int kMergeItemsPerThread = 2;

using MergeSort = cub::BlockMergeSort<std::uint64_t, kMergeThreads, kMergeItemsPerThread>;

__device__ __forceinline__ std::int64_t partial_offset(std::int32_t column, std::int32_t group,
                                                       std::int32_t rank,
                                                       std::int32_t group_stride) {
    return (static_cast<std::int64_t>(column) * group_stride + group) * kLinearTopK + rank;
}

template <bool Finalize>
__global__ __launch_bounds__(kMergeThreads, 2) void linear_topk_merge_kernel(
    const std::uint64_t* __restrict__ input_keys, std::uint64_t* __restrict__ output_keys,
    std::int32_t* __restrict__ group_done, std::int32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_scores, std::int32_t input_groups, std::int32_t output_groups) {
    const std::int32_t output_group = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t column       = static_cast<std::int32_t>(blockIdx.y);
    const int tid                   = static_cast<int>(threadIdx.x);

    __shared__ typename MergeSort::TempStorage sort_storage;
    __shared__ std::int32_t is_last;
    std::uint64_t keys[kMergeItemsPerThread];

#pragma unroll
    for (int item = 0; item < kMergeItemsPerThread; ++item) {
        const int p             = tid * kMergeItemsPerThread + item;
        const int local_partial = p / kLinearTopK;
        const int rank          = p - local_partial * kLinearTopK;
        const int input_group   = output_group * kLinearTopKMergeFanIn + local_partial;
        keys[item]              = input_group < input_groups
                                      ? input_keys[partial_offset(column, input_group, rank, input_groups)]
                                      : 0;
    }

    MergeSort(sort_storage).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
    for (int item = 0; item < kMergeItemsPerThread; ++item) {
        const int rank = tid * kMergeItemsPerThread + item;
        if (rank < kLinearTopK) {
            output_keys[partial_offset(column, output_group, rank, output_groups)] = keys[item];
        }
    }
    if constexpr (!Finalize) { return; }
    if (tid < kLinearTopK / kMergeItemsPerThread) { __threadfence(); }
    __syncthreads();

    if (tid == 0) { is_last = atomicAdd(group_done + column, 1) + 1 == output_groups; }
    __syncthreads();
    if (is_last == 0) { return; }

#pragma unroll
    for (int item = 0; item < kMergeItemsPerThread; ++item) {
        const int p     = tid * kMergeItemsPerThread + item;
        const int group = p / kLinearTopK;
        const int rank  = p - group * kLinearTopK;
        keys[item]      = group < output_groups
                              ? output_keys[partial_offset(column, group, rank, output_groups)]
                              : 0;
    }
    __syncthreads();
    MergeSort(sort_storage).Sort(keys, ScoreIdOrderGreater{});

#pragma unroll
    for (int item = 0; item < kMergeItemsPerThread; ++item) {
        const int rank = tid * kMergeItemsPerThread + item;
        if (rank < kLinearTopK) {
            const std::uint64_t key = keys[item];
            const std::int64_t out  = static_cast<std::int64_t>(column) * kLinearTopK + rank;
            candidate_ids[out]      = id_from_order_key(key);
            candidate_scores[out]   = score_from_order_key(key);
        }
    }
    if (tid == 0) { group_done[column] = 0; }
}

} // namespace

void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(workspace.group_done.data, 0, workspace.group_done.bytes(), stream));
    if (workspace.secondary_groups != 0) {
        const dim3 first_grid(static_cast<unsigned>(workspace.merge_groups),
                              static_cast<unsigned>(workspace.columns), 1U);
        linear_topk_merge_kernel<false><<<first_grid, kMergeThreads, 0, stream>>>(
            static_cast<const std::uint64_t*>(workspace.partial_keys.data),
            static_cast<std::uint64_t*>(workspace.group_keys.data), nullptr, nullptr, nullptr,
            workspace.producer_groups, workspace.merge_groups);
        CUDA_CHECK(cudaGetLastError());

        const dim3 final_grid(static_cast<unsigned>(workspace.secondary_groups),
                              static_cast<unsigned>(workspace.columns), 1U);
        linear_topk_merge_kernel<true><<<final_grid, kMergeThreads, 0, stream>>>(
            static_cast<const std::uint64_t*>(workspace.group_keys.data),
            static_cast<std::uint64_t*>(workspace.secondary_keys.data),
            static_cast<std::int32_t*>(workspace.group_done.data),
            static_cast<std::int32_t*>(candidate_ids.data),
            static_cast<float*>(candidate_scores.data), workspace.merge_groups,
            workspace.secondary_groups);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 grid(static_cast<unsigned>(workspace.merge_groups),
                    static_cast<unsigned>(workspace.columns), 1U);
    linear_topk_merge_kernel<true><<<grid, kMergeThreads, 0, stream>>>(
        static_cast<const std::uint64_t*>(workspace.partial_keys.data),
        static_cast<std::uint64_t*>(workspace.group_keys.data),
        static_cast<std::int32_t*>(workspace.group_done.data),
        static_cast<std::int32_t*>(candidate_ids.data), static_cast<float*>(candidate_scores.data),
        workspace.producer_groups, workspace.merge_groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
