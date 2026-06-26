// qus::kernels - linear wrapper: public api validation and qtype/T dispatch.
#include "qus/kernels/linear.h"

#include "kernels/launcher/linear.h" // detail::linear_dense_*_launch
#include "qus/core/weight.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error("linear: weight payload size overflows uint64");
    }
    return a * b;
}

void require_matrix_shapes(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k) {
        throw std::invalid_argument("linear: x K dimension must match weight.k");
    }
    if (out.ne[0] != w.n) {
        throw std::invalid_argument("linear: out N dimension must match weight.n");
    }
    if (out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: out T dimension must match x");
    }
}

void require_dense_metadata(const Weight& w) {
    if (w.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("linear: dense weight must be Contiguous");
    }
    if (w.ndim != 2) { throw std::invalid_argument("linear: dense weight must be 2-D [N,K]"); }
    if (w.shape[0] <= 0 || w.shape[1] <= 0) {
        throw std::invalid_argument("linear: dense weight shape must be positive");
    }
    if (w.shape[0] != w.n || w.shape[1] != w.k) {
        throw std::invalid_argument("linear: dense weight shape must match n/k");
    }
    if (w.group != 0 || w.group_size != 0) {
        throw std::invalid_argument("linear: dense weight group must be zero");
    }
    if (w.q5090_scale_dtype != ScaleDType::None) {
        throw std::invalid_argument("linear: dense weight scale dtype must be None");
    }
    for (int d = 0; d < 4; ++d) {
        if (w.padded_shape[d] != w.shape[d]) {
            throw std::invalid_argument("linear: dense weight padded shape must match shape");
        }
    }
    const std::uint64_t elem_size = (w.qtype == QType::FP32_CTRL) ? 4u : 2u;
    const std::uint64_t expected  = checked_mul_u64(
        checked_mul_u64(static_cast<std::uint64_t>(w.n), static_cast<std::uint64_t>(w.k)),
        elem_size);
    if (w.payload_bytes != 0 && w.payload_bytes < expected) {
        throw std::invalid_argument("linear: dense payload is too small");
    }
}

bool is_empty_T(const Tensor& x, const Tensor& out) { return x.ne[1] == 0 || out.ne[1] == 0; }

void require_non_empty_tensors(const Tensor& x, const Tensor& out) {
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("linear: x/out data must be non-null");
    }
}

} // namespace

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }

    (void)numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");

    switch (w.qtype) {
    case QType::BF16_CTRL:
    case QType::FP32_CTRL: {
        require_dense_metadata(w);
        require_matrix_shapes(x, w, out);
        if (is_empty_T(x, out)) { return; }
        require_non_empty_tensors(x, out);
        if (w.qdata == nullptr) {
            throw std::invalid_argument("linear: dense weight data must be non-null");
        }
        const Tensor dense = as_dense(w);
        if (x.ne[1] == 1) {
            detail::linear_dense_gemv_launch(x, dense, out, stream);
        } else {
            detail::linear_dense_gemm_launch(x, dense, out, stream);
        }
    } break;
    default:
        throw std::invalid_argument("linear: unsupported weight qtype");
    }
}

} // namespace qus::kernels
