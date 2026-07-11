#include "qus/kernels/linear.h"
#include "qus/kernels/linear_pair.h"

#include "kernels/common/math.h"
#include "kernels/linear/plan/linear_plan.h"
#include "kernels/linear/gemv/linear_rowsplit_gemv_attn_in_7168.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_qk_4096.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_vz_6144.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_lm_head.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_out_6144.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_proj_6144.cuh"
#include "kernels/linear/reference/linear_generic.h"
#include "qus/core/weight.h" // as_dense

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

std::uint64_t align_up_u64(std::uint64_t x, std::uint64_t m) {
    const std::uint64_t add = m - 1;
    if (x > std::numeric_limits<std::uint64_t>::max() - add) {
        throw std::overflow_error("linear: aligned size overflows uint64");
    }
    return round_up(x, m);
}

std::int32_t align_up_checked(std::int32_t x, std::int32_t m, const char* label) {
    const std::int64_t xi = x;
    const std::int64_t mi = m;
    const std::int64_t y  = round_up(xi, mi);
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
    if (w.qhigh != nullptr || w.high_plane_bytes != 0) {
        throw std::invalid_argument("linear: dense weight high plane must be null");
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

void require_row_split_lowbit_metadata(const Weight& w, const char* label, std::int32_t group_size,
                                       std::uint64_t nibble_bytes_per_group,
                                       std::uint64_t high_bytes_per_group) {
    if (w.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(std::string("linear: ") + label + " weight must be RowSplit");
    }
    require_weight_2d(w, label);
    if (w.group != group_size || w.group_size != static_cast<std::uint32_t>(group_size)) {
        throw std::invalid_argument(std::string("linear: ") + label + " weight group is invalid");
    }
    if (w.q5090_scale_dtype != ScaleDType::FP16) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " weight scale dtype must be FP16");
    }
    if (w.padded_shape[0] != w.shape[0] ||
        w.padded_shape[1] != align_up_checked(w.shape[1], 128, label)) {
        throw std::invalid_argument(std::string("linear: ") + label + " padded shape is invalid");
    }
    const std::uint64_t kg = static_cast<std::uint64_t>(w.padded_shape[1] / group_size);
    const std::uint64_t nibble_plane_bytes = checked_mul_u64(
        checked_mul_u64(static_cast<std::uint64_t>(w.n), kg), nibble_bytes_per_group);
    const std::uint64_t high_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(w.n), kg), high_bytes_per_group);
    const std::uint64_t scale_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(w.n), kg), 2u);
    const std::uint64_t high_plane_off = align_up_u64(nibble_plane_bytes, 256);
    const std::uint64_t high_aligned   = align_up_u64(high_plane_bytes, 256);
    if (high_aligned > std::numeric_limits<std::uint64_t>::max() - high_plane_off) {
        throw std::overflow_error("linear: row-split payload size overflows uint64");
    }
    const std::uint64_t scale_plane_off = high_plane_off + high_aligned;
    if (scale_plane_bytes > std::numeric_limits<std::uint64_t>::max() - scale_plane_off) {
        throw std::overflow_error("linear: row-split payload size overflows uint64");
    }
    const std::uint64_t expected_payload = scale_plane_off + scale_plane_bytes;
    if (w.payload_bytes < expected_payload) {
        throw std::invalid_argument(std::string("linear: ") + label + " payload is too small");
    }
    if (w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " nibble and scale planes must be non-null");
    }
    if (high_bytes_per_group == 0) {
        if (w.qhigh != nullptr || w.high_plane_bytes != 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " high plane must be null");
        }
    } else if (w.qhigh == nullptr || w.high_plane_bytes < high_plane_bytes) {
        throw std::invalid_argument(std::string("linear: ") + label +
                                    " high plane must be non-null");
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

void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
            cudaStream_t stream) {
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
        require_row_split_lowbit_metadata(w, "Q4G64_F16S", 64, 32u, 0u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::Q5G64_F16S:
        require_row_split_lowbit_metadata(w, "Q5G64_F16S", 64, 32u, 8u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::Q6G64_F16S:
        require_row_split_lowbit_metadata(w, "Q6G64_F16S", 64, 32u, 16u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::W8G32_F16S:
        require_row_split_lowbit_metadata(w, "W8G32_F16S", 32, 32u, 0u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    default:
        throw std::invalid_argument("linear: unsupported weight qtype");
    }

    // --- classify + dispatch (M3 framework) ---
    const detail::LinearFormat fmt  = detail::classify_format(w);
    const detail::ShapeFamily shape = detail::classify_shape(w.n, w.k);
    detail::LinearRegime regime     = detail::classify_regime(fmt, shape, x.ne[1]);
    // The LargeT bf16 tensor-core mma GEMM streams x with 16B cp.async (8 bf16 per
    // token row), which requires k % 8 == 0 (else the per-token base k*col is not
    // 16-byte aligned and the last k%8 elements would be dropped). Every Qwen3.6
    // shape has k a multiple of 128; for any other k, fall back to the small-T
    // GEMM, which is correct for all k.
    const bool mma_routed_format = fmt == detail::LinearFormat::Q4G64_RowSplit ||
                                   fmt == detail::LinearFormat::Q5G64_RowSplit ||
                                   fmt == detail::LinearFormat::Q6G64_RowSplit ||
                                   fmt == detail::LinearFormat::W8G32_RowSplit;
    if (mma_routed_format && regime == detail::LinearRegime::LargeT && (w.k % 8) != 0) {
        regime = detail::LinearRegime::SmallT;
    }
    if (fmt == detail::LinearFormat::W8G32_RowSplit && (w.padded_shape[1] % 256) != 0) {
        regime = detail::LinearRegime::SmallT;
    }
    const detail::LinearPlan plan = detail::resolve_plan(detail::LinearPlanKey{fmt, shape, regime});

    switch (plan.policy) {
    case detail::LinearPolicyId::RowsplitLowbitGemmSmallt:
        detail::linear_rowsplit_gemm_smallt_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::RowsplitLowbitGemmMma:
        detail::linear_rowsplit_gemm_mma_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::RowsplitW8G32GemmMma:
        detail::linear_rowsplit_w8g32_gemm_mma_launch(x, w, out, stream);
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
    case detail::LinearPolicyId::MlpGateUp34816Q4RowsplitGemv:
        detail::linear_rowsplit_gemv_mlp_gate_up_34816_q4_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::AttnInQKV7168Q4RowsplitGemv:
        detail::linear_rowsplit_gemv_attn_in_7168_q4_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::AttnInQKV7168Q5RowsplitGemv:
        detail::linear_rowsplit_gemv_attn_in_7168_q5_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::GdnInQK4096Q4RowsplitGemv:
        detail::linear_rowsplit_gemv_gdn_in_qk_4096_q4_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::MlpDownQ5RowsplitGemv:
        detail::linear_rowsplit_gemv_mlp_down_q5_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::LmHeadQ6RowsplitGemv:
        detail::linear_rowsplit_gemv_lm_head_q6_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::LmHeadQ4RowsplitGemv:
        detail::linear_rowsplit_gemv_lm_head_q4_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::Proj6144Q5RowsplitGemv:
        detail::linear_rowsplit_gemv_proj_6144_q5_launch(x, w, out, ws, stream);
        break;
    case detail::LinearPolicyId::Out6144Q5RowsplitGemv:
        detail::linear_rowsplit_gemv_out_6144_q5_launch(x, w, out, ws, stream);
        break;
    }
}

void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, WorkspaceArena& ws, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || first_out.dtype != DType::BF16 ||
        second_out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_pair: x/out tensors must be BF16");
    }
    if (first_weight.qtype != second_weight.qtype) {
        throw std::invalid_argument("linear_pair: weight qtypes must match");
    }
    switch (first_weight.qtype) {
    case QType::W8G32_F16S:
        require_row_split_lowbit_metadata(first_weight, "first W8G32_F16S", 32, 32u, 0u);
        require_row_split_lowbit_metadata(second_weight, "second W8G32_F16S", 32, 32u, 0u);
        break;
    case QType::Q5G64_F16S:
        require_row_split_lowbit_metadata(first_weight, "first Q5G64_F16S", 64, 32u, 8u);
        require_row_split_lowbit_metadata(second_weight, "second Q5G64_F16S", 64, 32u, 8u);
        break;
    default:
        throw std::invalid_argument("linear_pair: unsupported weight qtype");
    }
    require_matrix_shapes(x, first_weight, first_out);
    require_matrix_shapes(x, second_weight, second_out);
    require_tensor_strides(x, first_out);
    require_tensor_strides(x, second_out);
    if (first_weight.n != second_weight.n || first_weight.k != second_weight.k ||
        first_weight.padded_shape[1] != second_weight.padded_shape[1]) {
        throw std::invalid_argument("linear_pair: weight shapes must match");
    }
    if (is_empty_T(x, first_out)) { return; }
    require_tensor_data(x, first_out);
    require_tensor_data(x, second_out);

    if (first_weight.qtype == QType::Q5G64_F16S && x.ne[1] == 1 && first_weight.n == 6144 &&
        first_weight.k == 5120) {
        detail::linear_rowsplit_gemv_gdn_in_vz_6144_q5_launch(x, first_weight, second_weight,
                                                              first_out, second_out, stream);
        return;
    }
    if (first_weight.qtype == QType::W8G32_F16S && x.ne[1] > 16 && (first_weight.k % 8) == 0 &&
        (first_weight.padded_shape[1] % 256) == 0) {
        detail::linear_rowsplit_w8g32_kv_gemm_mma_launch(x, first_weight, second_weight, first_out,
                                                         second_out, stream);
        return;
    }
    linear(x, first_weight, first_out, ws, stream);
    linear(x, second_weight, second_out, ws, stream);
}

} // namespace qus::kernels
