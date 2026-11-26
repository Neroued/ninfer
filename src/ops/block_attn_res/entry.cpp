#include "launch.h"

#include "ops/common/host/operands.h"

#include <cmath>
#include <limits>

namespace ninfer::ops {
namespace operand = common::operand;

namespace attn_res_detail {

void require_disjoint(const Tensor& output, std::initializer_list<const Tensor*> inputs,
                      const char* op) {
    const auto written = operand::range(output, op);
    for (const auto* input : inputs) {
        if (input->data) operand::disjoint(written, operand::range(*input, op), op);
    }
}

Mix prepare_mix(const Tensor& stack, const Tensor& addend, const Tensor& queries,
                const Tensor& norm_weight, const BlockAttnResParameters& params, Tensor& out, int d,
                int width, int batch, Device device, const char* op) {
    const auto columns = std::int64_t(width) * batch;
    if (d <= 0 || width <= 0 || batch <= 0 || columns > std::numeric_limits<int>::max() ||
        queries.ne[1] <= 0 || (stack.data && stack.ne[2] <= 0)) {
        throw std::invalid_argument(std::string(op) + ": invalid attention dimensions");
    }
    for (const auto value : {params.query_scale, params.rms_eps, params.norm_eps}) {
        if (!std::isfinite(value) || value <= 0.0f)
            throw std::invalid_argument(std::string(op) + ": expected finite positive parameters");
    }
    const int m = static_cast<int>(columns), q = queries.ne[1];
    const std::size_t align = d % 8 == 0 ? 16 : 2;
    if (stack.data)
        operand::tensor(stack, DType::BF16, {d, m, stack.ne[2]}, align, op, "stack", device);
    if (addend.data)
        operand::tensor(addend, DType::BF16, {d, width, batch}, align, op, "addend", device);
    operand::tensor(queries, DType::FP32, {d, q}, std::max(align, alignof(float)), op, "queries",
                    device);
    if (norm_weight.data)
        operand::tensor(norm_weight, DType::BF16, {d, q}, align, op, "norm weight", device);
    const auto dtype = norm_weight.data ? DType::BF16 : DType::FP32;
    operand::tensor(out, dtype, {d, m, q}, std::max(align, dtype_size(dtype)), op, "mixed output",
                    device);
    require_disjoint(out, {&stack, &addend, &queries, &norm_weight}, op);
    return {static_cast<const std::uint16_t*>(stack.data),
            static_cast<const std::uint16_t*>(addend.data),
            static_cast<const float*>(queries.data),
            static_cast<const std::uint16_t*>(norm_weight.data),
            out.data,
            d,
            m,
            stack.data ? stack.ne[2] : 0,
            q,
            1.0f / float(d),
            params};
}

} // namespace attn_res_detail

void block_attn_res(const Tensor& stack, const Tensor& partial, const Tensor& addend,
                    const Tensor& queries, const Tensor& norm_weight,
                    const BlockAttnResParameters& params, Tensor& out,
                    CudaExecutionView execution) {
    constexpr const char* op = "block_attn_res";
    const auto d = partial.ne[0], m = partial.ne[1];
    const auto mix = attn_res_detail::prepare_mix(stack, addend, queries, norm_weight, params, out,
                                                  d, m, 1, execution.device(), op);
    operand::tensor(partial, DType::BF16, {d, m}, d % 8 == 0 ? 16 : 2, op, "partial",
                    execution.device());
    attn_res_detail::require_disjoint(out, {&partial}, op);
    attn_res_detail::launch(mix, static_cast<const std::uint16_t*>(partial.data), execution);
}

} // namespace ninfer::ops
