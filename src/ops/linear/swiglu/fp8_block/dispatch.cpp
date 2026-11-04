#include "ops/linear/swiglu/fp8_block/dispatch.h"

#include "core/platform.h"
#include "ops/common/host/operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Fp8BlockSwiGluLaunch select_fp8_block_swiglu_launch(int rows, int k, int tokens,
                                                    LinearPolicy policy) {
    if (tokens <= 0 || !valid_linear_policy(policy))
        throw std::invalid_argument("block FP8 linear_swiglu: invalid tokens or policy");
#if NINFER_TARGET_SPARK
    if (rows == 2048 && k == 4096) return select_fp8_block_swiglu_n2048_k4096(tokens);
#endif
    throw std::invalid_argument("block FP8 linear_swiglu: unsupported weight geometry");
}

std::size_t fp8_block_swiglu_workspace_capacity_bytes(int rows, int k, LinearPolicy policy,
                                                      int first, int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("block FP8 linear_swiglu workspace: invalid token interval");
    (void)select_fp8_block_swiglu_launch(rows, k, first, policy);
    (void)select_fp8_block_swiglu_launch(rows, k, last, policy);
    return 0;
}

void fp8_block_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                               bool clamped, LinearPolicy policy, cudaStream_t stream) {
    (void)common::validate_fp8_block_fp32_weight(weight, "block FP8 linear_swiglu");
    select_fp8_block_swiglu_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, limit,
                                                                        clamped, stream);
}

} // namespace ninfer::ops::detail
