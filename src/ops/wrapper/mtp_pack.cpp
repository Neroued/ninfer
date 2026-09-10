#include "ninfer/ops/mtp_pack.h"
#include "ops/launcher/mtp_pack.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, const char* op,
                   const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

// The contract says [D], and ops::rmsnorm holds callers to it. A length check alone would let
// [1,D] or [D/2,2] through here and take the fused route on an operand the composed path refuses.
void require_weight_vector(const Tensor& t, std::int32_t d, const char* op, const char* name) {
    require_shape(t, d, 1, op, name);
}

} // namespace

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_pack_fc_input";
    require_bf16_contiguous_nonnull(embedding_norm, op, "embedding_norm");
    require_bf16_contiguous_nonnull(hidden_norm, op, "hidden_norm");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t rows   = embedding_norm.ne[0];
    const std::int32_t tokens = embedding_norm.ne[1];
    if (rows <= 0) { throw std::invalid_argument("mtp_pack_fc_input: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("mtp_pack_fc_input: T must be positive"); }
    require_shape(embedding_norm, rows, tokens, op, "embedding_norm");
    require_shape(hidden_norm, rows, tokens, op, "hidden_norm");
    require_shape(out, 2 * rows, tokens, op, "out");

    detail::mtp_pack_fc_input_launch(embedding_norm, hidden_norm, out, stream);
}

namespace {

void require_stem_operands(const Tensor& embedding, const Tensor& embedding_weight,
                           const Tensor& hidden, const Tensor& hidden_weight, const Tensor& out) {
    constexpr const char* op = "mtp_norm_pack_fc_input";
    require_bf16_contiguous_nonnull(embedding, op, "embedding");
    require_bf16_contiguous_nonnull(embedding_weight, op, "embedding_weight");
    require_bf16_contiguous_nonnull(hidden, op, "hidden");
    require_bf16_contiguous_nonnull(hidden_weight, op, "hidden_weight");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t rows   = embedding.ne[0];
    const std::int32_t tokens = embedding.ne[1];
    if (rows <= 0) { throw std::invalid_argument("mtp_norm_pack_fc_input: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("mtp_norm_pack_fc_input: T must be positive"); }
    require_shape(embedding, rows, tokens, op, "embedding");
    require_shape(hidden, rows, tokens, op, "hidden");
    require_shape(out, 2 * rows, tokens, op, "out");
    require_weight_vector(embedding_weight, rows, op, "embedding_weight");
    require_weight_vector(hidden_weight, rows, op, "hidden_weight");
}

} // namespace

// The fused Ops stand in for a composition that contains ops::rmsnorm, so they owe the same
// contract on eps that ops::rmsnorm enforces (src/ops/wrapper/rmsnorm.cpp). Without this a
// non-positive or non-finite eps would reach the kernel and come back as NaN instead of a throw.
void require_normalization_eps(float eps, const char* op) {
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(std::string(op) + ": eps must be positive and finite");
    }
}

bool mtp_norm_pack_fc_input_supported(const Tensor& embedding, const Tensor& embedding_weight,
                                      const Tensor& hidden, const Tensor& hidden_weight,
                                      const Tensor& out) {
    require_stem_operands(embedding, embedding_weight, hidden, hidden_weight, out);
    return detail::mtp_norm_pack_fc_input_admits(embedding.ne[0], embedding, embedding_weight,
                                                 hidden, hidden_weight, out);
}

void mtp_norm_pack_fc_input(const Tensor& embedding, const Tensor& embedding_weight,
                            const Tensor& hidden, const Tensor& hidden_weight, Tensor& out,
                            float eps, cudaStream_t stream) {
    require_stem_operands(embedding, embedding_weight, hidden, hidden_weight, out);
    require_normalization_eps(eps, "mtp_norm_pack_fc_input");
    if (!detail::mtp_norm_pack_fc_input_admits(embedding.ne[0], embedding, embedding_weight, hidden,
                                               hidden_weight, out)) {
        throw std::invalid_argument("mtp_norm_pack_fc_input: unsupported operand profile");
    }
    detail::mtp_norm_pack_fc_input_launch(embedding, embedding_weight, hidden, hidden_weight, out,
                                          eps, stream);
}

namespace {

void require_residual_norm_operands(const Tensor& delta, const Tensor& residual,
                                    const Tensor& weight, const Tensor& out) {
    constexpr const char* op = "mtp_residual_norm";
    require_bf16_contiguous_nonnull(delta, op, "delta");
    require_bf16_contiguous_nonnull(residual, op, "residual");
    require_bf16_contiguous_nonnull(weight, op, "weight");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t rows   = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    if (rows <= 0) { throw std::invalid_argument("mtp_residual_norm: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("mtp_residual_norm: T must be positive"); }
    require_shape(delta, rows, tokens, op, "delta");
    require_shape(residual, rows, tokens, op, "residual");
    require_shape(out, rows, tokens, op, "out");
    require_weight_vector(weight, rows, op, "weight");
}

} // namespace

bool mtp_residual_norm_supported(const Tensor& delta, const Tensor& residual, const Tensor& weight,
                                 const Tensor& out) {
    require_residual_norm_operands(delta, residual, weight, out);
    return detail::mtp_residual_norm_admits(residual.ne[0], delta, residual, weight, out);
}

void mtp_residual_norm(const Tensor& delta, Tensor& residual, const Tensor& weight, Tensor& out,
                       float eps, cudaStream_t stream) {
    require_normalization_eps(eps, "mtp_residual_norm");
    require_residual_norm_operands(delta, residual, weight, out);
    if (!detail::mtp_residual_norm_admits(residual.ne[0], delta, residual, weight, out)) {
        throw std::invalid_argument("mtp_residual_norm: unsupported operand profile");
    }
    detail::mtp_residual_norm_launch(delta, residual, weight, out, eps, stream);
}

void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_split_attn_in";
    require_bf16_contiguous_nonnull(attn_in, op, "attn_in");
    require_bf16_contiguous_nonnull(q, op, "q");
    require_bf16_contiguous_nonnull(k, op, "k");
    require_bf16_contiguous_nonnull(gate, op, "gate");
    require_bf16_contiguous_nonnull(v, op, "v");
    const std::int32_t tokens = attn_in.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("mtp_split_attn_in: T must be positive"); }
    require_shape(attn_in, 14336, tokens, op, "attn_in");
    if (q.ne[0] != 256 || q.ne[1] != 24 || q.ne[2] != tokens || q.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for q");
    }
    if (k.ne[0] != 256 || k.ne[1] != 4 || k.ne[2] != tokens || k.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for k");
    }
    if (gate.ne[0] != 256 || gate.ne[1] != 24 || gate.ne[2] != tokens || gate.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for gate");
    }
    if (v.ne[0] != 256 || v.ne[1] != 4 || v.ne[2] != tokens || v.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for v");
    }

    detail::mtp_split_attn_in_launch(attn_in, q, k, gate, v, stream);
}

} // namespace ninfer::ops
