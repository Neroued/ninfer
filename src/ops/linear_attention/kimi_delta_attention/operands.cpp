#include "ops/linear_attention/kimi_delta_attention/operands.h"
#include "ops/common/host/operands.h"

#include <cmath>

namespace ninfer::ops::detail::kimi_delta_attention {
Geometry validate_inputs(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                         float lower_bound, float scale, const Tensor& out, Device device) {
    constexpr const char* op = "kimi_delta_attention";
    const Geometry p{q.ne[1], v.ne[1], q.ne[2], q.ne[3]};
    if (!valid_heads(p.qk_heads, p.value_heads) || p.tokens < 1 || p.batch < 1 || p.batch > 8)
        throw std::invalid_argument("kimi_delta_attention: invalid head/token/batch geometry");
    if (!std::isfinite(lower_bound) || lower_bound < -5 || lower_bound > 0 ||
        !std::isfinite(scale) || std::abs(scale - 1.0F / std::sqrt(128.0F)) > 1.0e-6F)
        throw std::invalid_argument("kimi_delta_attention: invalid decay bound or query scale");
    for (const auto* t : {&q, &k})
        common::operand::tensor(*t, DType::BF16, {128, p.qk_heads, p.tokens, p.batch}, 16, op,
                                "Q/K", device);
    for (const auto* t : {&v, &g, &out})
        common::operand::tensor(*t, DType::BF16, {128, p.value_heads, p.tokens, p.batch}, 16, op,
                                "value/gate/output", device);
    common::operand::tensor(beta, DType::BF16, {p.value_heads, p.tokens, p.batch}, 2, op, "beta",
                            device);
    common::operand::tensor(a_log, DType::FP32, {p.value_heads}, 4, op, "A_log", device);
    common::operand::tensor(dt_bias, DType::FP32, {128, p.value_heads}, 4, op, "dt_bias", device);
    for (const auto* t : {&q, &k, &v, &g, &beta, &a_log, &dt_bias})
        common::operand::disjoint(common::operand::range(*t, op), common::operand::range(out, op),
                                  op);
    return p;
}

void validate_state(ConstStateFieldView state, const Geometry& p, Device device) {
    constexpr const char* op = "kimi_delta_attention";
    if (state.field.layers != 1)
        throw std::invalid_argument("kimi_delta_attention: select one State layer");
    common::operand::state_field(state, DType::FP32, {128, 128, p.value_heads}, 16, op,
                                 "recurrent state", device);
    common::operand::shape(state.records.byte_offsets, {p.batch, 1, 1, 1}, op, "State offsets");
}
} // namespace ninfer::ops::detail::kimi_delta_attention
