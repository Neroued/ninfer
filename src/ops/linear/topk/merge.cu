#include "ops/common/cuda/reduce.cuh"
#include "ops/linear/topk/dispatch.h"

#include "core/device.h"
#include "ops/linear/topk/workspace.h"

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

template <bool KeysOnly, bool Accumulate>
__device__ void publish_keys(const std::uint64_t (&keys)[kMergeItemsPerThread], int column,
                             int* candidate_ids, float* candidate_scores,
                             std::uint64_t* candidate_keys) {
    const int tid = threadIdx.x;
    if constexpr (Accumulate) {
        if (tid < 32) {
            const auto even = __shfl_sync(common::kFullWarpMask, keys[0], tid / 2);
            const auto odd  = __shfl_sync(common::kFullWarpMask, keys[1], tid / 2);
            auto key        = tid < kLinearTopK
                                  ? (tid & 1 ? odd : even)
                                  : candidate_keys[std::int64_t(column) * kLinearTopK + 31 - tid];
#pragma unroll
            for (int distance = 16; distance; distance /= 2) {
                const auto other = __shfl_xor_sync(common::kFullWarpMask, key, distance);
                key              = (tid & distance) ? min(key, other) : max(key, other);
            }
            if (tid < kLinearTopK) {
                const auto offset = std::int64_t(column) * kLinearTopK + tid;
                if constexpr (KeysOnly)
                    candidate_keys[offset] = key;
                else {
                    candidate_ids[offset]    = common::id_from_order_key(key);
                    candidate_scores[offset] = common::score_from_order_key(key);
                }
            }
        }
    } else {
#pragma unroll
        for (int item = 0; item < kMergeItemsPerThread; ++item) {
            const int rank = tid * kMergeItemsPerThread + item;
            if (rank < kLinearTopK) {
                const std::uint64_t key = keys[item];
                const std::int64_t out  = static_cast<std::int64_t>(column) * kLinearTopK + rank;
                if constexpr (KeysOnly)
                    candidate_keys[out] = key;
                else {
                    candidate_ids[out]    = common::id_from_order_key(key);
                    candidate_scores[out] = common::score_from_order_key(key);
                }
            }
        }
    }
}

template <bool KeysOnly, bool Accumulate>
__global__ __launch_bounds__(kMergeThreads, 2) void linear_topk_single_merge_kernel(
    const std::uint64_t* input, int input_groups, int* ids, float* scores, std::uint64_t* output) {
    __shared__ typename MergeSort::TempStorage storage;
    const int column = blockIdx.x;
    std::uint64_t keys[kMergeItemsPerThread];
#pragma unroll
    for (int item = 0; item < kMergeItemsPerThread; ++item) {
        const int rank = int(threadIdx.x) * kMergeItemsPerThread + item;
        keys[item]     = rank < input_groups * kLinearTopK
                             ? input[std::int64_t(column) * input_groups * kLinearTopK + rank]
                             : 0;
    }
    MergeSort(storage).Sort(keys, common::ScoreIdOrderGreater{});
    publish_keys<KeysOnly, Accumulate>(keys, column, ids, scores, output);
}

template <bool Finalize, bool KeysOnly = false, bool Accumulate = false>
__global__ __launch_bounds__(kMergeThreads, 2) void linear_topk_merge_kernel(
    const std::uint64_t* __restrict__ input_keys, std::uint64_t* __restrict__ output_keys,
    std::int32_t* __restrict__ group_done, std::int32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_scores, std::uint64_t* __restrict__ candidate_keys,
    std::int32_t input_groups, std::int32_t output_groups) {
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

    MergeSort(sort_storage).Sort(keys, common::ScoreIdOrderGreater{});
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
    MergeSort(sort_storage).Sort(keys, common::ScoreIdOrderGreater{});

    publish_keys<KeysOnly, Accumulate>(keys, column, candidate_ids, candidate_scores,
                                       candidate_keys);
    if (tid == 0) { group_done[column] = 0; }
}

} // namespace

template <bool KeysOnly, bool Accumulate = false>
void launch_merge(const LinearTopKWorkspace& workspace, int* ids, float* scores,
                  std::uint64_t* keys, cudaStream_t stream) {
    if (workspace.producer_groups <= kLinearTopKMergeFanIn) {
        linear_topk_single_merge_kernel<KeysOnly, Accumulate>
            <<<workspace.columns, kMergeThreads, 0, stream>>>(
                static_cast<const std::uint64_t*>(workspace.partial_keys.data),
                workspace.producer_groups, ids, scores, keys);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(workspace.group_done.data, 0, workspace.group_done.bytes(), stream));
    if (workspace.secondary_groups != 0) {
        const dim3 first_grid(static_cast<unsigned>(workspace.merge_groups),
                              static_cast<unsigned>(workspace.columns), 1U);
        linear_topk_merge_kernel<false><<<first_grid, kMergeThreads, 0, stream>>>(
            static_cast<const std::uint64_t*>(workspace.partial_keys.data),
            static_cast<std::uint64_t*>(workspace.group_keys.data), nullptr, nullptr, nullptr,
            nullptr, workspace.producer_groups, workspace.merge_groups);
        CUDA_CHECK(cudaGetLastError());

        const dim3 final_grid(static_cast<unsigned>(workspace.secondary_groups),
                              static_cast<unsigned>(workspace.columns), 1U);
        linear_topk_merge_kernel<true, KeysOnly, Accumulate>
            <<<final_grid, kMergeThreads, 0, stream>>>(
                static_cast<const std::uint64_t*>(workspace.group_keys.data),
                static_cast<std::uint64_t*>(workspace.secondary_keys.data),
                static_cast<std::int32_t*>(workspace.group_done.data), ids, scores, keys,
                workspace.merge_groups, workspace.secondary_groups);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 grid(static_cast<unsigned>(workspace.merge_groups),
                    static_cast<unsigned>(workspace.columns), 1U);
    linear_topk_merge_kernel<true, KeysOnly, Accumulate><<<grid, kMergeThreads, 0, stream>>>(
        static_cast<const std::uint64_t*>(workspace.partial_keys.data),
        static_cast<std::uint64_t*>(workspace.group_keys.data),
        static_cast<std::int32_t*>(workspace.group_done.data), ids, scores, keys,
        workspace.producer_groups, workspace.merge_groups);
    CUDA_CHECK(cudaGetLastError());
}

void launch_linear_topk_merge(const LinearTopKWorkspace& workspace, Tensor& ids, Tensor& scores,
                              cudaStream_t stream) {
    launch_merge<false>(workspace, static_cast<int*>(ids.data), static_cast<float*>(scores.data),
                        nullptr, stream);
}

void launch_linear_topk_merge_final(const LinearTopKWorkspace& workspace, Tensor& keys, Tensor& ids,
                                    Tensor& scores, cudaStream_t stream) {
    launch_merge<false, true>(workspace, static_cast<int*>(ids.data),
                              static_cast<float*>(scores.data),
                              static_cast<std::uint64_t*>(keys.data), stream);
}

void launch_linear_topk_merge_keys(const LinearTopKWorkspace& workspace, Tensor& keys,
                                   cudaStream_t stream) {
    launch_merge<true>(workspace, nullptr, nullptr, static_cast<std::uint64_t*>(keys.data), stream);
}

void launch_linear_topk_merge_keys_accumulate(const LinearTopKWorkspace& workspace, Tensor& keys,
                                              cudaStream_t stream) {
    launch_merge<true, true>(workspace, nullptr, nullptr, static_cast<std::uint64_t*>(keys.data),
                             stream);
}
} // namespace ninfer::ops::detail
