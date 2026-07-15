// ninfer::ops - l2norm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/l2norm.h"

#include "ops/launcher/l2norm.h" // detail::l2norm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("l2norm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("l2norm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) { throw std::invalid_argument("l2norm: x/out shapes must match"); }
    }
}

} // namespace

void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("l2norm: x/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("l2norm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");
    require_same_shape(x, out);
    if (n == 0) { return; }

    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("l2norm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("l2norm: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("l2norm: x/out data must be non-null");
    }

    detail::l2norm_launch(x, eps, out, stream);
}

} // namespace ninfer::ops
