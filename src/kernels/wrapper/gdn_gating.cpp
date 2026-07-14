// ninfer::kernels - gdn_gating wrapper: public api validation and launcher dispatch.
#include "ninfer/kernels/gdn_gating.h"

#include "kernels/launcher/gdn_gating.h" // detail::gdn_gating_launch

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::kernels {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("gdn_gating: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("gdn_gating: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_gate_shape(const Tensor& t, const char* label) {
    if (t.ne[0] != 48 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("gdn_gating: ") + label +
                                    " must have shape [48,T]");
    }
}

void require_vector48_shape(const Tensor& t, const char* label) {
    if (t.ne[0] != 48 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("gdn_gating: ") + label +
                                    " must have shape [48]");
    }
}

void require_same_gate_shape(const Tensor& ref, const Tensor& t, const char* label) {
    for (int d = 0; d < 4; ++d) {
        if (ref.ne[d] != t.ne[d]) {
            throw std::invalid_argument(std::string("gdn_gating: ") + label +
                                        " shape must match a");
        }
    }
}

} // namespace

void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream) {
    if (a.dtype != DType::BF16 || b.dtype != DType::BF16) {
        throw std::invalid_argument("gdn_gating: a/b must be BF16");
    }
    if (A_log.dtype != DType::FP32 || dt_bias.dtype != DType::FP32) {
        throw std::invalid_argument("gdn_gating: A_log/dt_bias must be FP32");
    }
    if (g.dtype != DType::FP32 || beta.dtype != DType::FP32) {
        throw std::invalid_argument("gdn_gating: g/beta must be FP32");
    }

    const std::int64_t n = numel_allow_zero(a, "a");
    (void) numel_allow_zero(b, "b");
    (void) numel_allow_zero(A_log, "A_log");
    (void) numel_allow_zero(dt_bias, "dt_bias");
    (void) numel_allow_zero(g, "g");
    (void) numel_allow_zero(beta, "beta");

    require_gate_shape(a, "a");
    require_gate_shape(b, "b");
    require_gate_shape(g, "g");
    require_gate_shape(beta, "beta");
    require_vector48_shape(A_log, "A_log");
    require_vector48_shape(dt_bias, "dt_bias");
    require_same_gate_shape(a, b, "b");
    require_same_gate_shape(a, g, "g");
    require_same_gate_shape(a, beta, "beta");
    if (n == 0) { return; }

    if (!a.is_contiguous() || !b.is_contiguous() || !A_log.is_contiguous() ||
        !dt_bias.is_contiguous() || !g.is_contiguous() || !beta.is_contiguous()) {
        throw std::invalid_argument("gdn_gating: all tensors must be contiguous");
    }
    if (a.data == nullptr || b.data == nullptr || A_log.data == nullptr || dt_bias.data == nullptr ||
        g.data == nullptr || beta.data == nullptr) {
        throw std::invalid_argument("gdn_gating: all tensor data pointers must be non-null");
    }

    detail::gdn_gating_launch(a, b, A_log, dt_bias, g, beta, stream);
}

} // namespace ninfer::kernels
