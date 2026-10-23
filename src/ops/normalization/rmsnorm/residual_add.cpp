#include "ninfer/ops/residual_add_rmsnorm.h"
#include "ops/normalization/rmsnorm/fused_launch.h"
#include "ops/common/host/operands.h"
#include <array>
#include <cmath>

namespace ninfer::ops {
void residual_add_rmsnorm(const Tensor& residual, const Tensor& branch, const Tensor& weight,
                          float epsilon, Tensor& residual_out, Tensor& normalized_out,
                          CudaExecutionView execution) {
    namespace operand        = common::operand;
    namespace kernel         = detail::fused_rmsnorm;
    constexpr const char* op = "residual_add_rmsnorm";
    const int d = residual.ne[0], m = residual.ne[1];
    if (d <= 0 || m < 0 || !(epsilon > 0) || !std::isfinite(epsilon))
        throw std::invalid_argument("residual_add_rmsnorm: invalid extent or epsilon");
    std::uintptr_t alignment = 0;
    for (const auto* t : std::array<const Tensor*, 5>{&residual, &branch, &weight, &residual_out,
                                                      &normalized_out}) {
        const int columns = t == &weight ? 1 : m;
        operand::dtype(*t, DType::BF16, op, "operand must be BF16");
        operand::shape(*t, {d, columns, 1, 1}, op, "operand");
        if (m) operand::tensor(*t, DType::BF16, {d, columns}, 2, op, "operand", execution.device());
        alignment |= reinterpret_cast<std::uintptr_t>(t->data);
    }
    if (!m) return;
    const auto r = operand::range(residual_out, op), n = operand::range(normalized_out, op);
    operand::disjoint(r, n, op);
    for (const auto* input : {&residual, &branch, &weight}) {
        const auto range = operand::range(*input, op);
        operand::disjoint(n, range, op);
        if (input != &residual || residual_out.data != residual.data)
            operand::disjoint(r, range, op);
    }
    kernel::Arguments a{static_cast<const __nv_bfloat16*>(residual.data),
                        static_cast<const __nv_bfloat16*>(branch.data),
                        static_cast<const __nv_bfloat16*>(weight.data),
                        nullptr,
                        nullptr,
                        static_cast<__nv_bfloat16*>(normalized_out.data),
                        static_cast<__nv_bfloat16*>(residual_out.data),
                        d,
                        m,
                        epsilon};
    kernel::launch_residual(a, !(alignment & 15), execution.stream());
}
} // namespace ninfer::ops
