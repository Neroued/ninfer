#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/nvfp4_mul/dispatch.h"
#include "ops/linear/common/operands.h"
#include "core/platform.h"

#include <cmath>

namespace ninfer::ops {
namespace {

void execute(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out, float limit,
             LinearPolicy policy, MemoryArena* workspace, CudaExecutionView execution) {
    (void)linear_swiglu_bias_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, policy,
                                                      x.ne[1], x.ne[1]);
    if (out.ne[0] != weight.n / 2)
        throw std::invalid_argument("linear_swiglu bias: invalid output rows");
    if (!std::isfinite(limit) || limit <= 0)
        throw std::invalid_argument("linear_swiglu bias: clamp limit must be finite and positive");
    if (workspace && workspace->device() != execution.device())
        throw std::invalid_argument("linear_swiglu bias: workspace belongs to another device");
    if (weight.qtype == QType::NVFP4_MUL) {
        detail::validate_projection_operands(x, weight, {&out, 1}, &bias, execution,
                                             "linear_swiglu bias");
        detail::nvfp4_mul_swiglu_dispatch(x, weight, &bias, out, limit, true, policy,
                                          execution.stream());
        return;
    }
#if NINFER_TARGET_SPARK
    detail::validate_bf16_projection(x, weight, {&out, 1}, &bias, execution, "linear_swiglu bias");
    detail::launch_bf16_bias_swiglu_n8192_k1024(x, weight, bias, out, limit, execution.stream());
#endif
}

} // namespace

std::size_t linear_swiglu_bias_workspace_capacity_bytes(QType type, std::int32_t rows,
                                                        std::int32_t k, LinearPolicy policy,
                                                        std::int32_t first, std::int32_t last) {
    if (!valid_linear_policy(policy) || first <= 0 || last < first)
        throw std::invalid_argument("linear_swiglu bias workspace: invalid policy or interval");
#if NINFER_TARGET_SPARK
    if (rows == 8192 && k == 1024) {
        if (type == QType::BF16) return 0;
        if (type == QType::NVFP4_MUL)
            return detail::nvfp4_mul_swiglu_workspace_capacity_bytes(rows, k, policy, first, last);
    }
#endif
    throw std::invalid_argument("linear_swiglu bias: unsupported weight geometry");
}

void linear_swiglu(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                   float limit, LinearPolicy policy, MemoryArena& workspace,
                   CudaExecutionView execution) {
    execute(x, weight, bias, out, limit, policy, &workspace, execution);
}

void linear_swiglu(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                   float limit, CudaExecutionView execution) {
    execute(x, weight, bias, out, limit, LinearPolicy::A16Only, nullptr, execution);
}

} // namespace ninfer::ops
