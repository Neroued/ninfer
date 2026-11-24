#include "ninfer/ops/sum_contributions.h"
#include "ops/common/host/operands.h"

#include <cuda_bf16.h>

namespace ninfer::ops {
namespace {
template <bool Two, bool Residual>
__global__ void sum_kernel(const float* first, const float* second, const __nv_bfloat16* residual,
                           __nv_bfloat16* destination, std::int64_t count) {
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        float value = first[i];
        if constexpr (Two) { value += second[i]; }
        if constexpr (Residual) { value += __bfloat162float(residual[i]); }
        destination[i] = __float2bfloat16_rn(value);
    }
}
} // namespace

void sum_contributions(std::span<const Tensor> contributions, const Tensor* residual,
                       Tensor& destination, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "sum_contributions";
    if (contributions.empty() || contributions.size() > 2 ||
        execution.multiprocessor_count() <= 0) {
        throw std::invalid_argument(
            "sum_contributions: requires one or two inputs and valid execution");
    }
    const auto count = operand::numel(destination, op);
    if (count <= 0) { throw std::invalid_argument("sum_contributions: shape must be positive"); }
    const auto shape = {destination.ne[0], destination.ne[1], destination.ne[2], destination.ne[3]};
    operand::tensor(destination, DType::BF16, shape, 2, op, "destination", execution.device());
    const auto output_range = operand::range(destination, op);
    for (const auto& input : contributions) {
        operand::tensor(input, DType::FP32, shape, 4, op, "contribution", execution.device());
        operand::disjoint(operand::range(input, op), output_range, op);
    }
    if (residual) {
        operand::tensor(*residual, DType::BF16, shape, 2, op, "residual", execution.device());
        if (residual->data != destination.data) {
            operand::disjoint(operand::range(*residual, op), output_range, op);
        }
    }
    const auto* first = static_cast<const float*>(contributions[0].data);
    const auto* second =
        contributions.size() == 2 ? static_cast<const float*>(contributions[1].data) : nullptr;
    const auto* skip = residual ? static_cast<const __nv_bfloat16*>(residual->data) : nullptr;
    auto* output     = static_cast<__nv_bfloat16*>(destination.data);
    const int blocks = static_cast<int>(
        std::min<std::int64_t>((count + 255) / 256, execution.multiprocessor_count() * 8));
    const auto launch = [&]<bool Two, bool Residual>() {
        sum_kernel<Two, Residual>
            <<<blocks, 256, 0, execution.stream()>>>(first, second, skip, output, count);
    };
    if (second) {
        if (residual) {
            launch.template operator()<true, true>();
        } else {
            launch.template operator()<true, false>();
        }
    } else {
        if (residual) {
            launch.template operator()<false, true>();
        } else {
            launch.template operator()<false, false>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
