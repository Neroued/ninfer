#include "qus/kernels/linear.h"

#include "kernels/linear/gemv/linear_lowbit_gemv.h"
#include "kernels/linear/plan/linear_plan.h"
#include "kernels/linear/reference/linear_generic.h"
#include "qus/core/weight.h"   // as_dense

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

std::int32_t align_up_checked(std::int32_t x, std::int32_t m, const char* label) {
    const std::int64_t xi = x;
    const std::int64_t mi = m;
    const std::int64_t y  = ((xi + mi - 1) / mi) * mi;
    if (y > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error(std::string("linear: ") + label +
                                  " padded shape overflows int32");
    }
    return static_cast<std::int32_t>(y);
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

void require_weight_2d(const Weight& w, const char* label) {
    if (w.ndim != 2) {
        throw std::invalid_argument(std::string("linear: ") + label + " weight must be 2-D [N,K]");
    }
    if (w.shape[0] <= 0 || w.shape[1] <= 0) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " weight shape must be positive");
    }
    if (w.shape[0] != w.n || w.shape[1] != w.k) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " weight shape must match n/k");
    }
    if (w.shape[2] != 1 || w.shape[3] != 1 || w.padded_shape[2] != 1 || w.padded_shape[3] != 1) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " weight trailing dimensions must be 1");
    }
}

void require_dense_metadata(const Weight& w) {
    if (w.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("linear: dense weight must be Contiguous");
    }
    require_weight_2d(w, "dense");
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

void require_tile_lowbit_metadata(const Weight& w, const char* label, std::uint64_t bytes_per_row) {
    if (w.layout != QuantLayout::TileN64K64) {
        throw std::invalid_argument(std::string("linear: ") + label + " weight must be TileN64K64");
    }
    require_weight_2d(w, label);
    if (w.group != 64 || w.group_size != 64) {
        throw std::invalid_argument(std::string("linear: ") + label + " weight group must be 64");
    }
    if (w.q5090_scale_dtype != ScaleDType::FP16) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " weight scale dtype must be FP16");
    }
    if (w.padded_shape[0] != align_up_checked(w.shape[0], 64, label) ||
        w.padded_shape[1] != align_up_checked(w.shape[1], 64, label)) {
        throw std::invalid_argument(std::string("linear: ") + label + " padded shape is invalid");
    }
    const std::uint64_t nt       = static_cast<std::uint64_t>(w.padded_shape[0] / 64);
    const std::uint64_t kg       = static_cast<std::uint64_t>(w.padded_shape[1] / 64);
    const std::uint64_t tilew    = checked_mul_u64(64u, bytes_per_row) + 64u * 2u;
    const std::uint64_t expected = checked_mul_u64(checked_mul_u64(nt, kg), tilew);
    if (w.payload_bytes < expected) {
        throw std::invalid_argument(std::string("linear: ") + label + " payload is too small");
    }
}

bool is_empty_T(const Tensor& x, const Tensor& out) { return x.ne[1] == 0 || out.ne[1] == 0; }

bool is_contiguous_allow_zero(const Tensor& t) {
    std::int64_t expected = static_cast<std::int64_t>(dtype_size(t.dtype));
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] < 0) { return false; }
        if (t.ne[i] != 1 && t.nb[i] != expected) { return false; }
        if (i < 3 && t.ne[i] != 1) {
            if (t.ne[i] == 0) {
                expected = 0;
            } else if (expected > std::numeric_limits<std::int64_t>::max() / t.ne[i]) {
                return false;
            } else {
                expected *= t.ne[i];
            }
        }
    }
    return true;
}

void require_tensor_strides(const Tensor& x, const Tensor& out) {
    if (!is_contiguous_allow_zero(x) || !is_contiguous_allow_zero(out)) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
}

void require_tensor_data(const Tensor& x, const Tensor& out) {
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("linear: x/out data must be non-null");
    }
}

void require_aligned_16(const void* p, const char* label) {
    if ((reinterpret_cast<std::uintptr_t>(p) & 0xfu) != 0) {
        throw std::invalid_argument(std::string("linear: dense ") + label +
                                    " data must be 16-byte aligned");
    }
}

void require_dense_alignment(const Tensor& x, const Weight& w, const Tensor& out) {
    require_aligned_16(x.data, "x");
    require_aligned_16(w.qdata, "weight");
    require_aligned_16(out.data, "out");
}

} // namespace

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");

    // --- validation (identical to the legacy wrapper) ---
    switch (w.qtype) {
    case QType::BF16_CTRL:
    case QType::FP32_CTRL:
        require_dense_metadata(w);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.qdata == nullptr) {
            throw std::invalid_argument("linear: dense weight data must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        require_dense_alignment(x, w, out);
        break;
    case QType::Q4G64_F16S:
        require_tile_lowbit_metadata(w, "Q4G64_F16S", 32u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.payload == nullptr && w.qdata == nullptr) {
            throw std::invalid_argument("linear: Q4G64_F16S payload must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::Q5G64_F16S:
        require_tile_lowbit_metadata(w, "Q5G64_F16S", 40u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.payload == nullptr && w.qdata == nullptr) {
            throw std::invalid_argument("linear: Q5G64_F16S payload must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::Q6G64_F16S:
        require_tile_lowbit_metadata(w, "Q6G64_F16S", 48u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.payload == nullptr && w.qdata == nullptr) {
            throw std::invalid_argument("linear: Q6G64_F16S payload must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    default:
        throw std::invalid_argument("linear: unsupported weight qtype");
    }

    // --- classify + dispatch (M3 framework) ---
    const detail::LinearFormat fmt    = detail::classify_format(w);
    const detail::ShapeFamily  shape  = detail::classify_shape(w.n, w.k);
    const detail::LinearRegime regime = detail::classify_regime(fmt, shape, x.ne[1]);
    const detail::LinearPlan   plan   = detail::resolve_plan(detail::LinearPlanKey{fmt, shape, regime});

    switch (plan.policy) {
    case detail::LinearPolicyId::GenericLowbitGemv:
        detail::linear_generic_lowbit_gemv_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::GenericLowbitGemm:
        detail::linear_generic_lowbit_gemm_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::TunedLowbitGemv:
        detail::linear_tuned_lowbit_gemv_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::GenericDenseGemv: {
        const Tensor dense = as_dense(w);
        detail::linear_generic_dense_gemv_launch(x, dense, out, stream);
        break;
    }
    case detail::LinearPolicyId::GenericDenseGemm: {
        const Tensor dense = as_dense(w);
        detail::linear_generic_dense_gemm_launch(x, dense, out, stream);
        break;
    }
    }
}

} // namespace qus::kernels
