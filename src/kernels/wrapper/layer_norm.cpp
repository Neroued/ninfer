#include "ninfer/kernels/layer_norm.h"

#include "kernels/launcher/layer_norm.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels {
namespace {

std::int64_t checked_numel(const Tensor& t) {
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] < 0) { throw std::invalid_argument("layer_norm: negative dimension"); }
        if (t.ne[i] == 0) { return 0; }
        if (n > std::numeric_limits<std::int64_t>::max() / t.ne[i]) {
            throw std::overflow_error("layer_norm: tensor size overflows int64");
        }
        n *= t.ne[i];
    }
    return n;
}

bool vector_of(const Tensor& t, std::int32_t d) {
    return t.ne[0] == d && t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

} // namespace

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
                cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || bias.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("layer_norm: x/weight/bias/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("layer_norm: eps must be positive and finite");
    }
    const std::int64_t n = checked_numel(x);
    (void)checked_numel(weight);
    (void)checked_numel(bias);
    (void)checked_numel(out);
    for (int i = 0; i < 4; ++i) {
        if (out.ne[i] != x.ne[i]) {
            throw std::invalid_argument("layer_norm: x/out shapes must match");
        }
    }
    if (!vector_of(weight, x.ne[0]) || !vector_of(bias, x.ne[0])) {
        throw std::invalid_argument("layer_norm: weight/bias must have shape [x.ne[0]]");
    }
    if (n == 0) { return; }
    if (!x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("layer_norm: tensors must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || bias.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("layer_norm: tensor data must be non-null");
    }
    detail::layer_norm_launch(x, weight, bias, eps, out, stream);
}

} // namespace ninfer::kernels
