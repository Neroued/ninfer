#include "ops/linear/projection/q8/q8_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy)(x, w, out, stream);
}

} // namespace ninfer::ops::detail
