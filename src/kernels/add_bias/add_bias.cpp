#include "kernels/add_bias/add_bias.h"

#include "kernels/add_bias/launch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels {
namespace {

std::int64_t checked_numel(const Tensor& t) {
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] < 0) { throw std::invalid_argument("add_bias: negative dimension"); }
        if (t.ne[i] == 0) { return 0; }
        if (n > std::numeric_limits<std::int64_t>::max() / t.ne[i]) {
            throw std::overflow_error("add_bias: tensor size overflows int64");
        }
        n *= t.ne[i];
    }
    return n;
}

} // namespace

void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    if (bias.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("add_bias: bias/x must be BF16");
    }
    const std::int64_t n = checked_numel(x);
    (void)checked_numel(bias);
    if (bias.ne[0] != x.ne[0] || bias.ne[1] != 1 || bias.ne[2] != 1 || bias.ne[3] != 1) {
        throw std::invalid_argument("add_bias: bias must have shape [x.ne[0]]");
    }
    if (n == 0) { return; }
    if (!bias.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("add_bias: bias/x must be contiguous");
    }
    if (bias.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("add_bias: bias/x data must be non-null");
    }
    detail::add_bias_launch(bias, x, stream);
}

} // namespace ninfer::kernels
