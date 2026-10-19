#include "ninfer/ops/sequence.h"
#include "ops/common/host/operands.h"

namespace ninfer::ops {
namespace {
__global__ void shift_kernel(const int* ids, const int* positions, const int* rope,
                             const int* prefix, int* output, int* output_positions,
                             int* output_rope, int* lengths, int* last, int* active, int width) {
    const int b = blockIdx.x, a = prefix[b], n = max(a - 1, 0), w = width - 1;
    if (threadIdx.x == 0) {
        lengths[b] = n;
        last[b]    = a - 1;
        active[b]  = a ? 0 : -1;
    }
    for (int j = threadIdx.x; j < w; j += blockDim.x) {
        const int source            = j < n ? j : 0;
        output[b * w + j]           = ids[b * width + (j < n ? j + 1 : 0)];
        output_positions[b * w + j] = positions[b * width + source];
        for (int axis = 0; axis < 3; ++axis)
            output_rope[(b * 3 + axis) * w + j] = rope[(b * 3 + axis) * width + source];
    }
}
} // namespace

void prepare_shifted_prefix(const Tensor& ids, const Tensor& positions, const Tensor& rope,
                            const Tensor& prefix, Tensor& shifted_ids, Tensor& shifted_positions,
                            Tensor& shifted_rope, Tensor& lengths, Tensor& last_columns,
                            Tensor& active_columns, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "prepare_shifted_prefix";
    const auto w = ids.ne[0], b = ids.ne[1];
    if (w < 2 || b < 1 || b > 8)
        throw std::invalid_argument("prepare_shifted_prefix: invalid shape");
    for (const auto* tensor : {&ids, &positions})
        operand::tensor(*tensor, DType::I32, {w, b}, 4, op, "input", execution.device());
    operand::tensor(rope, DType::I32, {w, 3, b}, 4, op, "RoPE", execution.device());
    operand::tensor(prefix, DType::I32, {b}, 4, op, "prefix", execution.device());
    for (const auto* tensor : {&shifted_ids, &shifted_positions})
        operand::tensor(*tensor, DType::I32, {w - 1, b}, 4, op, "shifted input",
                        execution.device());
    operand::tensor(shifted_rope, DType::I32, {w - 1, 3, b}, 4, op, "shifted RoPE",
                    execution.device());
    for (const auto* tensor : {&lengths, &last_columns, &active_columns})
        operand::tensor(*tensor, DType::I32, {b}, 4, op, "output metadata", execution.device());
    std::array<operand::Range, 6> outputs{
        operand::range(shifted_ids, op),  operand::range(shifted_positions, op),
        operand::range(shifted_rope, op), operand::range(lengths, op),
        operand::range(last_columns, op), operand::range(active_columns, op)};
    operand::disjoint(outputs, "prepare_shifted_prefix: outputs overlap");
    for (auto output : outputs)
        for (const auto* input : {&ids, &positions, &rope, &prefix})
            if (operand::overlaps(output, operand::range(*input, op)))
                throw std::invalid_argument("prepare_shifted_prefix: input/output alias");
    shift_kernel<<<b, 128, 0, execution.stream()>>>(
        static_cast<const int*>(ids.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(rope.data), static_cast<const int*>(prefix.data),
        static_cast<int*>(shifted_ids.data), static_cast<int*>(shifted_positions.data),
        static_cast<int*>(shifted_rope.data), static_cast<int*>(lengths.data),
        static_cast<int*>(last_columns.data), static_cast<int*>(active_columns.data), w);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
