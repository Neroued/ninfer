#include "ninfer/ops/linear_bias.h"
#include "ops/linear/bias/dispatch.h"
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

void linear_bias(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out,
                 CudaExecutionView execution) {
    const auto launch = detail::linear_bias::select_launch(w.n, w.k, x.ne[1]);
    if (out.ne[0] != w.n) throw std::invalid_argument("linear_bias: invalid output rows");
    detail::validate_bf16_projection(x, w, {&out, 1}, &bias, execution, "linear_bias");
    launch(x, w, bias, out, execution.stream());
}

} // namespace ninfer::ops
