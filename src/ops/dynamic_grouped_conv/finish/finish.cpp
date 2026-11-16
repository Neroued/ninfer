#include "ops/dynamic_grouped_conv/finish/launch.h"
#include "ops/dynamic_grouped_conv/operands.h"
#include "ops/linear/common/operands.h"
#include "core/platform.h"
#include <array>

namespace ninfer::ops {
void linear_dynamic_grouped_conv(const Tensor& x, const Weight& projection, const Tensor& base,
                                 const Tensor& delta, Tensor& partial,
                                 CudaExecutionView execution) {
#if NINFER_TARGET_SPARK
    namespace operand        = common::operand;
    constexpr const char* op = "linear dynamic grouped conv";
    if (projection.n != 4096 || (projection.k != 2048 && projection.k != 6144))
        throw std::invalid_argument("linear dynamic grouped conv: unsupported projection");
    const int width = x.ne[1], batch = x.ne[2], hidden = projection.n;
    detail::dynamic_grouped_conv::validate_interval(width, width, batch, batch, op);
    operand::tensor(x, DType::BF16, {projection.k, width, batch}, 16, op, "input",
                    execution.device());
    operand::tensor(partial, DType::BF16, {hidden, width, batch}, 16, op, "partial",
                    execution.device());
    operand::tensor(base, DType::BF16, {hidden, 2, 2}, 16, op, "base", execution.device());
    operand::tensor(delta, DType::BF16, {hidden / 16, 2, width, batch}, 16, op, "delta",
                    execution.device());
    const auto flat_x   = x.view({projection.k, width * batch});
    const auto flat_out = partial.view({hidden, width * batch});
    detail::validate_bf16_projection(flat_x, projection, {&flat_out, 1}, nullptr, execution, op);
    const std::array ranges{
        operand::range(x, op), operand::range(base, op), operand::range(delta, op),
        operand::range(partial, op),
        operand::range(projection.qdata, std::size_t(hidden) * projection.k * 2, op)};
    operand::disjoint(ranges, op);
    detail::dynamic_grouped_conv::finish::launch(x, projection, base, delta, partial,
                                                 execution.stream());
#else
    throw std::invalid_argument("linear dynamic grouped conv: unsupported target");
#endif
}
} // namespace ninfer::ops
