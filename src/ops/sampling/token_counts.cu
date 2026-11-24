#include "ninfer/ops/sampling.h"
#include "ops/common/host/operands.h"
#include <algorithm>
#include <stdexcept>
#include "ops/common/numeric/math.h"
#include "core/device.h"

namespace ninfer::ops {
using namespace common;

namespace {

__global__ void increment_prefix_counts_kernel(const int* ids, const int* lengths, const int* slots,
                                               int* counts, int tokens, std::int64_t slot_stride) {
    const int row = blockIdx.x, n = lengths[row];
    if (n == 0) return;
    auto* destination = counts + std::int64_t(slots[row]) * slot_stride;
    for (int t = threadIdx.x; t < n; t += blockDim.x)
        atomicAdd(destination + ids[std::int64_t(row) * tokens + t], 1);
}

__global__ void increment_token_counts_kernel(const std::int32_t* token_ids, std::int32_t count,
                                              std::int32_t* token_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { atomicAdd(&token_counts[token_ids[index]], 1); }
}

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = token_ids.ne[0];
    increment_token_counts_kernel<<<div_up(count, kBlock), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token_ids.data), count,
        static_cast<std::int32_t*>(token_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void increment_token_counts(const Tensor& token_ids, Tensor& token_counts,
                            CudaExecutionView execution) {
    if (token_ids.device != execution.device() || token_counts.device != execution.device()) {
        throw std::invalid_argument(
            "increment_token_counts: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (token_ids.dtype != DType::I32 || token_ids.ne[0] <= 0 || token_ids.ne[1] != 1 ||
        token_ids.ne[2] != 1 || token_ids.ne[3] != 1 || !token_ids.is_contiguous() ||
        token_ids.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_ids must be a contiguous non-empty I32 vector");
    }
    if (token_counts.dtype != DType::I32 || token_counts.ne[0] <= 0 || token_counts.ne[1] != 1 ||
        token_counts.ne[2] != 1 || token_counts.ne[3] != 1 || !token_counts.is_contiguous() ||
        token_counts.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_counts must be a contiguous non-empty I32 vector");
    }
    increment_token_counts_launch(token_ids, token_counts, stream);
}

void increment_token_counts_batch(const Tensor& token_ids, const Tensor& valid_columns,
                                  const Tensor& slots, Tensor& token_counts,
                                  CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "increment_token_counts_batch";
    const int t = token_ids.ne[0], b = token_ids.ne[1], v = token_counts.ne[0],
              s = token_counts.ne[1];
    if (t <= 0 || b < 1 || b > 8 || v <= 0 || s <= 0)
        throw std::invalid_argument("increment_token_counts_batch: invalid shape");
    operand::tensor(token_ids, DType::I32, {t, b}, 4, op, "IDs", execution.device());
    operand::tensor(valid_columns, DType::I32, {b}, 4, op, "valid columns", execution.device());
    operand::tensor(slots, DType::I32, {b}, 4, op, "slots", execution.device());
    operand::slotted_tensor(token_counts, DType::I32, {v, s}, 1, 4, op, "counts",
                            execution.device());
    for (const auto* input : {&token_ids, &valid_columns, &slots})
        if (operand::overlaps(operand::blocks(token_counts, op), operand::range(*input, op)))
            throw std::invalid_argument("increment_token_counts_batch: input/output alias");
    increment_prefix_counts_kernel<<<b, 128, 0, execution.stream()>>>(
        static_cast<const int*>(token_ids.data), static_cast<const int*>(valid_columns.data),
        static_cast<const int*>(slots.data), static_cast<int*>(token_counts.data), t,
        token_counts.nb[1] / 4);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
