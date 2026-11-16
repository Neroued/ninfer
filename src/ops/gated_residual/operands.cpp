#include "ops/gated_residual/operands.h"
#include "ops/common/host/operands.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::hc {
namespace {
void require(bool value, const char* op, const char* reason) {
    if (!value) throw std::invalid_argument(std::string(op) + ": " + reason);
}

common::operand::Range check(const Tensor& t, DType dtype, int a, int b, int c, Device device,
                             const char* op) {
    common::operand::device(t.device, device, op, "operand");
    require(t.dtype == dtype, op, "invalid operand dtype");
    require(t.ne[0] == a && t.ne[1] == b && t.ne[2] == c && t.ne[3] == 1, op,
            "invalid operand shape");
    require(t.is_contiguous() && common::operand::aligned(t.data, 16), op,
            "operands require contiguous, 16-byte-aligned backing");
    return common::operand::range(t.data, t.bytes(), op);
}

bool same_view(const Tensor& a, const Tensor& b) {
    if (a.data != b.data || a.dtype != b.dtype) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] != b.ne[d] || a.nb[d] != b.nb[d]) return false;
    return true;
}
} // namespace

Operands validate(Form form, const Tensor& input, const Tensor* block, const Tensor* injection,
                  const Tensor* weight, float epsilon, Tensor* residual, Tensor* normalized,
                  CudaExecutionView execution, const char* op) {
    const bool expand = form == Form::ExpandNorm;
    const int width   = input.ne[0];
    validate_norm_combine_width(width);
    const auto m = input.ne[expand ? 1 : 2];
    require(m > 0, op, "columns must be positive");
    if (weight)
        require(epsilon > 0 && std::isfinite(epsilon), op, "epsilon must be positive and finite");
    std::array<common::operand::Range, 4> inputs;
    int count = 1;
    inputs[0] = check(input, DType::BF16, width, expand ? m : kStreams, expand ? 1 : m,
                      execution.device(), op);
    if (block) inputs[count++] = check(*block, DType::BF16, width, m, 1, execution.device(), op);
    if (injection)
        inputs[count++] = check(*injection, DType::FP32, kStreams, m, 1, execution.device(), op);
    if (weight)
        inputs[count++] = check(*weight, DType::BF16, width, kStreams, 1, execution.device(), op);
    const bool alias = residual && !expand && same_view(input, *residual);
    if (residual) {
        const auto output =
            check(*residual, DType::BF16, width, kStreams, m, execution.device(), op);
        for (int i = alias ? 1 : 0; i < count; ++i)
            common::operand::disjoint(output, inputs[i], op);
    }
    if (normalized) {
        const auto output =
            check(*normalized, DType::BF16, width, kStreams, m, execution.device(), op);
        for (int i = 0; i < count; ++i) common::operand::disjoint(output, inputs[i], op);
        if (residual)
            common::operand::disjoint(
                output, common::operand::range(residual->data, residual->bytes(), op), op);
    }
    return {input.data,
            block ? block->data : nullptr,
            injection ? static_cast<const float*>(injection->data) : nullptr,
            weight ? weight->data : nullptr,
            residual ? residual->data : nullptr,
            normalized ? normalized->data : nullptr,
            m,
            width,
            epsilon};
}
} // namespace ninfer::ops::detail::hc
