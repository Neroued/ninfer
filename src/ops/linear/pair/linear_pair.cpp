#include "ninfer/ops/linear_pair.h"
#include "ops/linear/pair/dispatch.h"
#include "ops/common/host/operands.h"
#include "ops/linear/projection/q8/q8_dispatch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace operand = common::operand;

void require_matrix(const Tensor& t, int rows, int tokens, CudaExecutionView execution) {
    if (t.device != execution.device()) {
        throw std::invalid_argument("linear_pair: operand belongs to another device");
    }
    if (t.dtype != DType::BF16 || t.ne[0] != rows || t.ne[1] != tokens || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || !operand::aligned(t.data, 16))
        throw std::invalid_argument("linear_pair: expected aligned contiguous BF16 matrix");
}

void require_nonoverlap(const Tensor& x0, const Tensor& x1, const Weight& w0, const Weight& w1,
                        const Tensor& y0, const Tensor& y1) {
    const auto range = [](const void* ptr, std::size_t bytes) {
        return operand::range(ptr, bytes, "linear_pair");
    };
    const auto first  = operand::range(y0, "linear_pair"),
               second = operand::range(y1, "linear_pair");
    operand::disjoint(first, second, "linear_pair");
    const std::size_t elements = std::size_t(w0.n) * w0.k;
    const auto bytes           = elements * (w0.qtype == QType::BF16 ? 2 : 1);
    for (const auto& input : {operand::range(x0, "linear_pair"), operand::range(x1, "linear_pair"),
                              range(w0.qdata, bytes), range(w1.qdata, bytes)}) {
        operand::disjoint(first, input, "linear_pair");
        operand::disjoint(second, input, "linear_pair");
    }
    if (w0.qtype == QType::Q8_G32_FP16)
        for (const auto* scales : {w0.scales, w1.scales}) {
            operand::disjoint(first, range(scales, elements / 32 * 2), "linear_pair");
            operand::disjoint(second, range(scales, elements / 32 * 2), "linear_pair");
        }
}
} // namespace

std::size_t linear_pair_workspace_capacity_bytes(const Weight& first, const Weight& second,
                                                 std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("linear_pair: invalid token interval");
    detail::validate_pair_weights(first, second, max_tokens);
    return 0;
}

std::size_t linear_pair_workspace_capacity_bytes(LinearProfile first, LinearProfile second,
                                                 std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens || first.qtype != second.qtype ||
        first.n != second.n || first.k != second.k)
        throw std::invalid_argument("linear_pair: invalid matrix profiles or token interval");
    if (first.qtype == QType::BF16) {
        if ((first.n != 4096 && first.n != 8192) || first.k != 128)
            throw std::invalid_argument("linear_pair: unsupported BF16 geometry");
    } else if (first.qtype == QType::Q8_G32_FP16 && first.n == 1024)
        (void)detail::select_linear_pair(first.k, max_tokens);
    else
        throw std::invalid_argument("linear_pair: unsupported encoding or geometry");
    return 0;
}

void linear_pair(const Tensor& x, const Tensor& second_input, const Weight& first,
                 const Weight& second, Tensor& first_out, Tensor& second_out,
                 CudaExecutionView execution) {
    require_matrix(x, first.k, x.ne[1], execution);
    require_matrix(second_input, second.k, x.ne[1], execution);
    require_matrix(first_out, first.n, x.ne[1], execution);
    require_matrix(second_out, second.n, x.ne[1], execution);
    detail::validate_pair_weights(first, second, x.ne[1]);
    if (first.device != execution.device() || second.device != execution.device()) {
        throw std::invalid_argument("linear_pair: weights belong to another device");
    }
    require_nonoverlap(x, second_input, first, second, first_out, second_out);
    if (first.qtype == QType::BF16)
        detail::launch_bf16_pair(x, second_input, first, second, first_out, second_out,
                                 execution.stream());
    else if (x.data == second_input.data)
        detail::select_linear_pair(x.ne[0], x.ne[1])(x, first, second, first_out, second_out,
                                                     execution.stream());
    else {
        const auto launch = detail::select_q8_a16_launch(first.n, first.k, x.ne[1]);
        launch(x, first, first_out, execution.stream());
        launch(second_input, second, second_out, execution.stream());
    }
}
} // namespace ninfer::ops
