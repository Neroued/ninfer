#include "ops/linear/bias/nvfp4_mul/dispatch.h"

#include "core/platform.h"
#include "ops/common/host/operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Nvfp4MulBiasRoute select_nvfp4_mul_bias_route(int rows, int k, int tokens, LinearPolicy policy) {
    if (tokens <= 0 || !valid_linear_policy(policy))
        throw std::invalid_argument("NVFP4_MUL linear_bias: invalid tokens or policy");
#if NINFER_TARGET_SPARK
    if (rows == 1024 && k == 4096) return select_nvfp4_mul_bias_n1024_k4096(tokens);
#endif
    throw std::invalid_argument("NVFP4_MUL linear_bias: unsupported weight geometry");
}

std::size_t nvfp4_mul_bias_workspace_capacity_bytes(int rows, int k, LinearPolicy policy, int first,
                                                    int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("NVFP4_MUL linear_bias workspace: invalid token interval");
    (void)select_nvfp4_mul_bias_route(rows, k, first, policy);
    (void)select_nvfp4_mul_bias_route(rows, k, last, policy);
    return 0;
}

void nvfp4_mul_bias_dispatch(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                             LinearPolicy policy, cudaStream_t stream) {
    (void)common::validate_nvfp4_mul_weight(weight, "NVFP4_MUL linear_bias");
    select_nvfp4_mul_bias_route(weight.n, weight.k, x.ne[1], policy)
        .launch(x, weight, bias, out, stream);
}

} // namespace ninfer::ops::detail
