#include "ninfer/ops/sequence.h"
#include "ops/common/host/operands.h"

namespace ninfer::ops {
namespace {
__global__ void pair_kernel(const uint4* hidden, const uint4* previous, const int* ids,
                            const int* positions, const int* prefix, const int* skip, uint4* paired,
                            int* paired_ids, int* paired_positions, int* lengths, int* last,
                            int* active, int vectors, int width, int output_width) {
    const int j = blockIdx.x, b = blockIdx.y, a = prefix[b], n = max(a - skip[b], 0);
    const int i = j + skip[b];
    for (int d = threadIdx.x; d < vectors; d += blockDim.x) {
        uint4 value{};
        if (j < n)
            value = i ? hidden[(std::int64_t(b) * width + i - 1) * vectors + d]
                      : previous[std::int64_t(b) * vectors + d];
        paired[(std::int64_t(b) * output_width + j) * vectors + d] = value;
    }
    if (threadIdx.x == 0) {
        paired_ids[b * output_width + j]       = j < n ? ids[b * width + i] : 0;
        paired_positions[b * output_width + j] = j < n ? positions[b * width + i] - 1 : 0;
        if (j == 0) {
            lengths[b] = n;
            last[b]    = a - 1;
            active[b]  = a ? 0 : -1;
        }
    }
}
} // namespace

void prepare_causal_pairs(const Tensor& hidden, const Tensor& previous, const Tensor& ids,
                          const Tensor& positions, const Tensor& prefix, const Tensor& skip,
                          const CausalPairOutputs& out, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "prepare_causal_pairs";
    const int d = hidden.ne[0], w = hidden.ne[1], b = hidden.ne[2], p = out.hidden.ne[1];
    if (d <= 0 || d % 8 || w < 1 || p < 1 || b < 1 || b > 8)
        throw std::invalid_argument("prepare_causal_pairs: invalid shape");
    operand::tensor(hidden, DType::BF16, {d, w, b}, 16, op, "hidden", execution.device());
    operand::tensor(previous, DType::BF16, {d, b}, 16, op, "previous", execution.device());
    for (const auto* v : {&ids, &positions})
        operand::tensor(*v, DType::I32, {w, b}, 4, op, "input metadata", execution.device());
    for (const auto* v : {&prefix, &skip})
        operand::tensor(*v, DType::I32, {b}, 4, op, "row metadata", execution.device());
    operand::tensor(out.hidden, DType::BF16, {d, p, b}, 16, op, "paired hidden",
                    execution.device());
    for (const auto* v : {&out.ids, &out.positions})
        operand::tensor(*v, DType::I32, {p, b}, 4, op, "paired metadata", execution.device());
    for (const auto* v : {&out.lengths, &out.last_columns, &out.active_columns})
        operand::tensor(*v, DType::I32, {b}, 4, op, "output metadata", execution.device());
    const std::array outputs{
        operand::range(out.hidden, op),       operand::range(out.ids, op),
        operand::range(out.positions, op),    operand::range(out.lengths, op),
        operand::range(out.last_columns, op), operand::range(out.active_columns, op)};
    operand::disjoint(outputs, "prepare_causal_pairs: outputs overlap");
    for (const auto output : outputs)
        for (const auto* input : {&hidden, &previous, &ids, &positions, &prefix, &skip})
            if (operand::overlaps(output, operand::range(*input, op)))
                throw std::invalid_argument("prepare_causal_pairs: input/output alias");
    pair_kernel<<<dim3(p, b), 128, 0, execution.stream()>>>(
        static_cast<const uint4*>(hidden.data), static_cast<const uint4*>(previous.data),
        static_cast<const int*>(ids.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(prefix.data), static_cast<const int*>(skip.data),
        static_cast<uint4*>(out.hidden.data), static_cast<int*>(out.ids.data),
        static_cast<int*>(out.positions.data), static_cast<int*>(out.lengths.data),
        static_cast<int*>(out.last_columns.data), static_cast<int*>(out.active_columns.data), d / 8,
        w, p);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
