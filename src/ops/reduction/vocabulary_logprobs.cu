#include "ninfer/ops/target_logprobs.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/math.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <algorithm>

namespace ninfer::ops {
namespace {
using namespace common;
constexpr int kTile = 8192, kThreads = 256, kColumnSlab = 65535;

int partials(int rows) { return (rows - 1) / kTile + 1; }

__global__ void stats_kernel(const __nv_bfloat16* logits, const int* targets, int rows, int live,
                             int begin, int pieces, float* stats) {
    const int col = blockIdx.y, start = blockIdx.x * kTile;
    const auto base = std::int64_t(col) * rows;
    float values[kTile / kThreads], maximum = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < kTile / kThreads; ++i) {
        const auto row = std::int64_t(start) + threadIdx.x + i * kThreads;
        values[i]      = row < live ? __bfloat162float(logits[base + row]) : -CUDART_INF_F;
        maximum        = fmaxf(maximum, values[i]);
    }
    maximum   = block_max_broadcast<kThreads>(maximum);
    float sum = 0;
#pragma unroll
    for (int i = 0; i < kTile / kThreads; ++i)
        if (std::int64_t(start) + threadIdx.x + i * kThreads < live)
            sum += __expf(values[i] - maximum);
    __shared__ float warp_sums[kThreads / 32];
    sum = block_reduce_sum<kThreads>(sum, warp_sums);
    if (threadIdx.x == 0) {
        const auto out    = (std::int64_t(col) * pieces + blockIdx.x) * 3;
        const auto target = std::int64_t(targets[col]) - begin;
        stats[out]        = maximum;
        stats[out + 1]    = sum;
        stats[out + 2]    = target >= start && target < std::int64_t(start) + kTile && target < live
                                ? __bfloat162float(logits[base + target])
                                : 0;
    }
}

// The same stable aggregation joins private tiles and public vocabulary partitions.
// strides are in scalar elements; their values do not alter the represented statistic.
template <bool Final>
__global__ void merge_kernel(const float* input, float* output, int pieces,
                             std::int64_t piece_stride, std::int64_t column_stride, int columns) {
    const int col = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (col >= columns) return;
    const auto* in = input + col * column_stride;
    float maximum  = -CUDART_INF_F;
    for (int i = lane; i < pieces; i += 32) maximum = fmaxf(maximum, in[i * piece_stride]);
    maximum   = warp_max(maximum);
    float sum = 0, target = 0;
    for (int i = lane; i < pieces; i += 32) {
        const auto* part = in + i * piece_stride;
        if (part[1] > 0) sum += part[1] * __expf(part[0] - maximum);
        target += part[2];
    }
    sum    = warp_reduce_sum(sum);
    target = warp_reduce_sum(target);
    if (lane == 0) {
        if constexpr (Final)
            output[col] = target - maximum - logf(sum);
        else {
            output[std::int64_t(col) * 3]     = maximum;
            output[std::int64_t(col) * 3 + 1] = sum;
            output[std::int64_t(col) * 3 + 2] = target;
        }
    }
}
} // namespace

std::size_t vocabulary_logprob_stats_workspace_capacity_bytes(int physical_rows, int max_columns) {
    if (physical_rows <= 0 || max_columns <= 0)
        throw std::invalid_argument("vocabulary_logprob_stats: invalid profile");
    return physical_rows <= kTile
               ? 0
               : std::size_t(partials(physical_rows)) * std::min(max_columns, kColumnSlab) * 3 * 4;
}

void vocabulary_logprob_stats(const Tensor& logits, const Tensor& target_ids, int token_begin,
                              int token_domain, Tensor& stats, MemoryArena& workspace,
                              CudaExecutionView execution) {
    constexpr const char* op = "vocabulary_logprob_stats";
    const int rows = logits.ne[0], cols = logits.ne[1];
    const auto bytes = vocabulary_logprob_stats_workspace_capacity_bytes(rows, cols);
    if (token_begin < 0 || token_domain <= 0)
        throw std::invalid_argument("vocabulary_logprob_stats: invalid domain");
    operand::tensor(logits, DType::BF16, {rows, cols}, 2, op, "logits", execution.device());
    operand::tensor(target_ids, DType::I32, {cols}, 4, op, "targets", execution.device());
    operand::tensor(stats, DType::FP32, {3, cols}, 4, op, "stats", execution.device());
    operand::device(workspace.device(), execution.device(), op, "workspace");
    for (const auto* input : {&logits, &target_ids})
        operand::disjoint(operand::range(stats, op), operand::range(*input, op), op);
    auto scope     = workspace.scope();
    float* output  = static_cast<float*>(stats.data);
    float* partial = output;
    if (bytes) {
        const auto scratch = workspace.alloc_bytes(bytes);
        const auto range   = operand::range(scratch.data, scratch.bytes, op);
        for (const auto* input : {&logits, &target_ids, static_cast<const Tensor*>(&stats)})
            operand::disjoint(range, operand::range(*input, op), op);
        partial = static_cast<float*>(scratch.data);
    }
    const int pieces = partials(rows),
              live   = std::max(0, std::min(rows, token_domain - token_begin));
    for (std::int64_t first = 0; first < cols; first += kColumnSlab) {
        const int width = int(std::min(std::int64_t(cols) - first, std::int64_t(kColumnSlab)));
        auto* result    = output + first * 3;
        auto* tiles     = pieces == 1 ? result : partial;
        stats_kernel<<<dim3(pieces, width), kThreads, 0, execution.stream()>>>(
            static_cast<const __nv_bfloat16*>(logits.data) + first * rows,
            static_cast<const int*>(target_ids.data) + first, rows, live, token_begin, pieces,
            tiles);
        if (pieces > 1)
            merge_kernel<false><<<(width - 1) / 4 + 1, 128, 0, execution.stream()>>>(
                tiles, result, pieces, 3, std::int64_t(pieces) * 3, width);
    }
    CUDA_CHECK(cudaGetLastError());
}

void target_logprobs_from_stats(const Tensor& stats, Tensor& output, CudaExecutionView execution) {
    constexpr const char* op = "target_logprobs_from_stats";
    const int cols = stats.ne[1], ranks = stats.ne[2];
    if (cols <= 0 || ranks < 1 || ranks > 16)
        throw std::invalid_argument("target_logprobs_from_stats: invalid C/R");
    operand::tensor(stats, DType::FP32, {3, cols, ranks}, 4, op, "stats", execution.device());
    operand::tensor(output, DType::FP32, {cols}, 4, op, "output", execution.device());
    operand::disjoint(operand::range(stats, op), operand::range(output, op), op);
    merge_kernel<true><<<(cols - 1) / 4 + 1, 128, 0, execution.stream()>>>(
        static_cast<const float*>(stats.data), static_cast<float*>(output.data), ranks,
        std::int64_t(cols) * 3, 3, cols);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
