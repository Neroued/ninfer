#include "ops/linear/swiglu/nvfp4_mul/dispatch.h"

#include "core/platform.h"
#include "ops/common/host/operands.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail {

Nvfp4MulSwiGluLaunch select_nvfp4_mul_swiglu_launch(int rows, int k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("NVFP4_MUL linear_swiglu: invalid policy");
#if NINFER_TARGET_SPARK
    if (rows == 8192 && k == 1024) return launch_nvfp4_mul_swiglu_n8192_k1024;
    if (rows == 20480 && k == 4096) return launch_nvfp4_mul_swiglu_n20480_k4096;
#endif
    throw std::invalid_argument("NVFP4_MUL linear_swiglu: unsupported weight geometry");
}

std::size_t nvfp4_mul_swiglu_workspace_capacity_bytes(int rows, int k, LinearPolicy policy,
                                                      int first, int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("NVFP4_MUL linear_swiglu workspace: invalid token interval");
    (void)select_nvfp4_mul_swiglu_launch(rows, k, policy);
    return 0;
}

void nvfp4_mul_swiglu_dispatch(const Tensor& x, const Weight& weight, const Tensor* bias,
                               Tensor& out, float limit, bool clamped, LinearPolicy policy,
                               cudaStream_t stream) {
    (void)common::validate_nvfp4_mul_weight(weight, "NVFP4_MUL linear_swiglu");
    if (x.ne[1] <= 0)
        throw std::invalid_argument("NVFP4_MUL linear_swiglu: tokens must be positive");
    if (clamped && (!std::isfinite(limit) || limit <= 0))
        throw std::invalid_argument("NVFP4_MUL linear_swiglu: invalid clamp limit");
    if (bias != nullptr && (weight.n != 8192 || weight.k != 1024 || !clamped))
        throw std::invalid_argument("NVFP4_MUL linear_swiglu: unsupported bias geometry");
    select_nvfp4_mul_swiglu_launch(weight.n, weight.k, policy)(x, weight, bias, out, limit, clamped,
                                                               stream);
}

} // namespace ninfer::ops::detail
