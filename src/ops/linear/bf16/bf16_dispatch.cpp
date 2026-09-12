#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool n256_k5120 = n == 256 && k == 5120;
    const bool supported_problem =
        (n == 14336 && k == 5120) || (n == 5120 && k == 6144) || n256_k5120;
    if (!supported_problem || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    if (n256_k5120) { return launch_bf16_n256_k5120; }
    if (t == 1) { return launch_bf16_decode; }
    const std::int32_t small_t_end =
        n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
    if (t <= small_t_end) { return launch_bf16_small_t; }
    return launch_bf16_mma;
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
