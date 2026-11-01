#include "ninfer/ops/linear_split.h"
#include "ops/linear/split/dispatch.h"
#include "ops/linear/common/operands.h"
#include "core/platform.h"

namespace ninfer::ops {

void linear_split(const Tensor& x, const Weight& weight, const Tensor& bias,
                  std::span<const Tensor> outputs, CudaExecutionView execution) {
#if NINFER_TARGET_SPARK
    if (weight.n != 3072 || weight.k != 1024 || outputs.size() != 3)
        throw std::invalid_argument(
            "linear_split bias: unsupported parent geometry or segmentation");
    for (const auto& out : outputs)
        if (out.ne[0] != 1024)
            throw std::invalid_argument("linear_split bias: invalid output rows");
    detail::validate_bf16_projection(x, weight, outputs, &bias, execution, "linear_split bias");
    detail::linear_split::launch_bf16_bias_n3072_k1024(x, weight, bias, outputs,
                                                       execution.stream());
#else
    throw std::invalid_argument("linear_split bias: unsupported parent geometry or segmentation");
#endif
}

} // namespace ninfer::ops
