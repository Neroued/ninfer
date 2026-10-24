#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/common/operands.h"
#include "core/platform.h"

#include <cmath>

namespace ninfer::ops {

void linear_swiglu(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                   float limit, CudaExecutionView execution) {
#if NINFER_TARGET_SPARK
    if (weight.n != 8192 || weight.k != 1024 || out.ne[0] != weight.n / 2)
        throw std::invalid_argument("linear_swiglu bias: unsupported weight/output geometry");
    if (!std::isfinite(limit) || limit <= 0)
        throw std::invalid_argument("linear_swiglu bias: clamp limit must be finite and positive");
    detail::validate_bf16_projection(x, weight, {&out, 1}, &bias, execution, "linear_swiglu bias");
    detail::launch_bf16_bias_swiglu_n8192_k1024(x, weight, bias, out, limit, execution.stream());
#else
    throw std::invalid_argument("linear_swiglu bias: unsupported weight geometry");
#endif
}

} // namespace ninfer::ops
