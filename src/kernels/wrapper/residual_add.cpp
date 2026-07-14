// ninfer::kernels - residual_add wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/kernel-development.md §2.
#include "ninfer/kernels/residual_add.h"

#include "kernels/launcher/residual_add.h" // detail::residual_add_launch

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels {
namespace {

std::int64_t numel_allow_zero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("residual_add: y/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("residual_add: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream) {
    if (y.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("residual_add: y/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (y.ne[d] != x.ne[d]) {
            throw std::invalid_argument("residual_add: y/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (!y.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("residual_add: y/x must be contiguous");
    }
    if (y.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("residual_add: y/x data must be non-null");
    }

    detail::residual_add_launch(y, x, stream); // single variant -> direct dispatch
}

} // namespace ninfer::kernels
