#include "qus/kernels/mtp_pack.h"

#include "kernels/launcher/mtp_pack.h"

#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_bf16_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::BF16) { throw std::invalid_argument(std::string(op) + ": " + name + " must be BF16"); }
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

} // namespace

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_pack_fc_input";
    require_bf16_contiguous_nonnull(embedding_norm, op, "embedding_norm");
    require_bf16_contiguous_nonnull(hidden_norm, op, "hidden_norm");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t tokens = embedding_norm.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("mtp_pack_fc_input: T must be positive"); }
    require_shape(embedding_norm, 5120, tokens, op, "embedding_norm");
    require_shape(hidden_norm, 5120, tokens, op, "hidden_norm");
    require_shape(out, 10240, tokens, op, "out");

    detail::mtp_pack_fc_input_launch(embedding_norm, hidden_norm, out, stream);
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

} // namespace qus::kernels

