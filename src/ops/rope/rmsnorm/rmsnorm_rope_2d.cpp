#include "ninfer/ops/rmsnorm_rope.h"
#include "ops/common/host/operands.h"
#include "ops/rope/kernels/axial64.h"
#include <cmath>

namespace ninfer::ops {
void rmsnorm_rope_2d(const Tensor& positions, const Tensor& q_weight, const Tensor& k_weight,
                     float epsilon, float theta, Tensor& q, Tensor& k,
                     CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "rmsnorm_rope_2d";
    const int tokens         = q.ne[2];
    if (tokens < 0 || !(epsilon > 0) || !std::isfinite(epsilon) || !(theta > 0) ||
        !std::isfinite(theta))
        throw std::invalid_argument("rmsnorm_rope_2d: invalid extent, epsilon or theta");
    for (const auto* t : {&q, &k}) {
        operand::dtype(*t, DType::BF16, op, "q/k");
        operand::shape(*t, {64, 16, tokens, 1}, op, "q/k");
        if (tokens)
            operand::tensor(*t, DType::BF16, {64, 16, tokens}, 4, op, "q/k", execution.device());
    }
    for (const auto* t : {&q_weight, &k_weight}) {
        operand::dtype(*t, DType::BF16, op, "gain");
        operand::shape(*t, {64, 1, 1, 1}, op, "gain");
        if (tokens) operand::tensor(*t, DType::BF16, {64}, 2, op, "gain", execution.device());
    }
    operand::dtype(positions, DType::I32, op, "positions");
    operand::shape(positions, {tokens, 2, 1, 1}, op, "positions");
    if (!tokens) return;
    operand::tensor(positions, DType::I32, {tokens, 2}, 4, op, "positions", execution.device());
    operand::disjoint(operand::range(q, op), operand::range(k, op), op);
    for (const auto* output : {&q, &k})
        for (const auto* input : {&positions, &q_weight, &k_weight})
            operand::disjoint(operand::range(*output, op), operand::range(*input, op), op);
    const detail::AxialRope64Arguments a{static_cast<const int*>(positions.data),
                                         static_cast<const __nv_bfloat16*>(q_weight.data),
                                         static_cast<const __nv_bfloat16*>(k_weight.data),
                                         static_cast<__nv_bfloat16*>(q.data),
                                         static_cast<__nv_bfloat16*>(k.data),
                                         1024,
                                         1024,
                                         tokens,
                                         16,
                                         16,
                                         -std::log2(theta) / 16.F,
                                         epsilon};
    detail::launch_normalized_axial64(a, execution.stream());
}
} // namespace ninfer::ops
