#include "ninfer/ops/gdn_gating_proj.h"

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_bf16_weight(const Weight& w, std::int32_t rows, std::int32_t input_rows,
                         const char* name) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(input_rows) *
                                        sizeof(std::uint16_t);
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous ||
        w.payload_bytes < payload_bytes || w.ndim != 2 || w.n != rows || w.k != input_rows ||
        w.shape[0] != rows || w.shape[1] != input_rows || w.padded_shape[0] != rows ||
        w.padded_shape[1] != input_rows || w.qhigh != nullptr || w.scales != nullptr ||
        w.high_plane_bytes != 0 || w.group != 0 || w.group_size != 0 || !aligned_to(w.qdata, 16)) {
        throw std::invalid_argument(std::string("gdn_gating_proj: invalid ") + name);
    }
}

Weight bf16_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    const std::size_t row_bytes = static_cast<std::size_t>(parent.k) * sizeof(std::uint16_t);
    const auto* data            = static_cast<const std::uint8_t*>(parent.qdata) +
                       static_cast<std::size_t>(row_begin) * row_bytes;
    Weight view          = parent;
    view.payload         = data;
    view.payload_bytes   = static_cast<std::uint64_t>(rows) * row_bytes;
    view.qdata           = data;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    view.n               = rows;
    return view;
}

void require_vector_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* op,
                           const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_sequence_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t tokens,
                             const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

} // namespace

std::size_t gdn_gating_proj_workspace_bytes(std::int32_t max_tokens) {
    return detail::bf16_gdn_gating_capacity_workspace_bytes(max_tokens);
}

void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream) {
    constexpr const char* op  = "gdn_gating_proj";
    const std::int32_t tokens = x.ne[1];
    require_sequence_tensor(x, DType::BF16, 5120, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, 48, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 48, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 48, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 48, tokens, op, "beta");
    require_bf16_weight(a_weight, 48, 5120, "a_weight");
    require_bf16_weight(b_weight, 48, 5120, "b_weight");

    detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     cudaStream_t stream) {
    constexpr const char* op  = "gdn_gating_proj";
    const std::int32_t tokens = x.ne[1];
    require_sequence_tensor(x, DType::BF16, 2048, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, 32, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 32, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 32, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 32, tokens, op, "beta");
    require_bf16_weight(ab_weight, 64, 2048, "ab_weight");

    const Weight a_weight = bf16_row_view(ab_weight, 0, 32);
    const Weight b_weight = bf16_row_view(ab_weight, 32, 32);
    detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

} // namespace ninfer::ops
