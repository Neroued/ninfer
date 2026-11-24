#include "ninfer/ops/paired_rmsnorm_concat.h"
#include "ops/normalization/rmsnorm/fused_launch.h"
#include "ops/common/host/operands.h"
#include <cmath>
#include <limits>

namespace ninfer::ops {
void paired_rmsnorm_concat(const Tensor& embedding, const Tensor& hidden,
                           const Tensor& embedding_weight, const Tensor& hidden_weight,
                           const Tensor& embedding_enabled, float epsilon, Tensor& out,
                           CudaExecutionView execution) {
    namespace operand        = common::operand;
    namespace kernel         = detail::fused_rmsnorm;
    constexpr const char* op = "paired_rmsnorm_concat";
    const int d = embedding.ne[0], m = embedding.ne[1];
    if (d <= 0 || d > std::numeric_limits<int>::max() / 2 || m < 0 || !(epsilon > 0) ||
        !std::isfinite(epsilon))
        throw std::invalid_argument("paired_rmsnorm_concat: invalid extent or epsilon");
    const auto check = [&](const Tensor& t, DType dtype, int rows, int columns, const char* name) {
        operand::dtype(t, dtype, op, name);
        operand::shape(t, {rows, columns, 1, 1}, op, name);
        if (m)
            operand::tensor(t, dtype, {rows, columns}, dtype_size(dtype), op, name,
                            execution.device());
    };
    check(embedding, DType::BF16, d, m, "embedding");
    check(hidden, DType::BF16, d, m, "hidden");
    check(embedding_weight, DType::BF16, d, 1, "embedding gain");
    check(hidden_weight, DType::BF16, d, 1, "hidden gain");
    check(embedding_enabled, DType::I32, m, 1, "embedding enabled");
    check(out, DType::BF16, d * 2, m, "output");
    if (!m) return;
    std::uintptr_t alignment = reinterpret_cast<std::uintptr_t>(out.data);
    for (const auto* input :
         {&embedding, &hidden, &embedding_weight, &hidden_weight, &embedding_enabled}) {
        operand::disjoint(operand::range(out, op), operand::range(*input, op), op);
        if (input != &embedding_enabled) alignment |= reinterpret_cast<std::uintptr_t>(input->data);
    }
    kernel::Arguments a{static_cast<const __nv_bfloat16*>(embedding.data),
                        static_cast<const __nv_bfloat16*>(hidden.data),
                        static_cast<const __nv_bfloat16*>(embedding_weight.data),
                        static_cast<const __nv_bfloat16*>(hidden_weight.data),
                        static_cast<const int*>(embedding_enabled.data),
                        static_cast<__nv_bfloat16*>(out.data),
                        nullptr,
                        d,
                        m,
                        epsilon};
    kernel::launch_paired(a, !(alignment & 15), execution.stream());
}
} // namespace ninfer::ops
