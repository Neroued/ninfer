#include "ninfer/ops/linear_bias.h"
#include "ops/linear/bias/dispatch.h"
#include "ops/linear/bias/mxfp8/dispatch.h"
#include "ops/linear/bias/nvfp4_mul/dispatch.h"
#include "ops/linear/common/operands.h"
#include "core/platform.h"

namespace ninfer::ops::detail::linear_bias {

Launch select_launch(int rows, int k, int columns) {
    if (columns <= 0) throw std::invalid_argument("linear_bias: columns must be positive");
#if NINFER_TARGET_SPARK
    if (rows == 1024 && k == 1176) return select_n1024_k1176(columns);
    if (rows == 1024 && k == 1024) return select_n1024_k1024(columns);
    if (rows == 1024 && k == 4096) return select_n1024_k4096(columns);
    if (rows == 4096 && k == 4096) return select_n4096_k4096(columns);
#endif
    throw std::invalid_argument("linear_bias: unsupported weight geometry");
}

} // namespace ninfer::ops::detail::linear_bias

namespace ninfer::ops {
namespace {

void execute(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out, LinearPolicy policy,
             MemoryArena* workspace, CudaExecutionView execution) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("linear_bias: invalid compute policy");
    if (workspace && workspace->device() != execution.device())
        throw std::invalid_argument("linear_bias: workspace belongs to another device");
    if (out.ne[0] != w.n) throw std::invalid_argument("linear_bias: invalid output rows");
    if (w.qtype == QType::BF16) {
        detail::validate_bf16_projection(x, w, {&out, 1}, &bias, execution, "linear_bias");
        detail::linear_bias::select_launch(w.n, w.k, x.ne[1])(x, w, bias, out, execution.stream());
        return;
    }
    detail::validate_projection_operands(x, w, {&out, 1}, &bias, execution, "linear_bias");
    switch (w.qtype) {
    case QType::MXFP8:
        detail::mxfp8_bias_dispatch(x, w, bias, out, policy, workspace, execution.stream());
        return;
    case QType::NVFP4_MUL:
        detail::nvfp4_mul_bias_dispatch(x, w, bias, out, policy, execution.stream());
        return;
    default:
        throw std::invalid_argument("linear_bias: unsupported weight format");
    }
}

} // namespace

std::size_t linear_bias_workspace_capacity_bytes(QType qtype, std::int32_t rows,
                                                 std::int32_t input_rows, LinearPolicy policy,
                                                 std::int32_t first, std::int32_t last) {
    if (!valid_linear_policy(policy) || first <= 0 || last < first)
        throw std::invalid_argument("linear_bias workspace: invalid policy or token interval");
    switch (qtype) {
    case QType::BF16:
        (void)detail::linear_bias::select_launch(rows, input_rows, first);
        (void)detail::linear_bias::select_launch(rows, input_rows, last);
        return 0;
    case QType::MXFP8:
        return detail::mxfp8_bias_workspace_capacity_bytes(rows, input_rows, policy, first, last);
    case QType::NVFP4_MUL:
        return detail::nvfp4_mul_bias_workspace_capacity_bytes(rows, input_rows, policy, first,
                                                               last);
    default:
        throw std::invalid_argument("linear_bias workspace: unsupported weight format");
    }
}

void linear_bias(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                 LinearPolicy policy, MemoryArena& workspace, CudaExecutionView execution) {
    execute(x, weight, bias, out, policy, &workspace, execution);
}

void linear_bias(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out,
                 CudaExecutionView execution) {
    execute(x, w, bias, out, LinearPolicy::A16Only, nullptr, execution);
}

} // namespace ninfer::ops
