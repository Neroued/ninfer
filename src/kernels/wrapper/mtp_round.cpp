#include "qus/kernels/mtp_round.h"

#include "kernels/launcher/mtp_round.h"
#include "qus/model/model.h"

#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_dtype(const Tensor& t, DType dtype, const char* op, const char* name) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid dtype for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

void require_scalar(const Tensor& t, DType dtype, const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (t.ne[0] != 1 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid scalar shape for " + name);
    }
}

void require_vector(const Tensor& t, DType dtype, std::int32_t n, const char* op,
                    const char* name) {
    require_dtype(t, dtype, op, name);
    if (n <= 0 || t.ne[0] != n || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
}

} // namespace

void mtp_prepare_verify_inputs(const Tensor& token, const Tensor& drafts, const Tensor& length,
                               Tensor& window_base, Tensor& verify_ids, Tensor& positions,
                               cudaStream_t stream) {
    constexpr const char* op = "mtp_prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    require_scalar(token, DType::I32, op, "token");
    require_vector(drafts, DType::I32, k, op, "drafts");
    require_scalar(length, DType::I32, op, "length");
    require_scalar(window_base, DType::I32, op, "window_base");
    require_vector(verify_ids, DType::I32, k + 1, op, "verify_ids");
    require_vector(positions, DType::I32, k + 1, op, "positions");
    detail::mtp_prepare_verify_inputs_launch(token, drafts, length, window_base, verify_ids,
                                             positions, stream);
}

void mtp_accept_tokens(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
                       Tensor& length, Tensor& token, Tensor& sampled_out, Tensor& num_sampled,
                       Tensor& accepted, Tensor& ar_pos, Tensor& stats,
                       const SamplingConfig* config, cudaStream_t stream) {
    constexpr const char* op = "mtp_accept_tokens";
    const std::int32_t k     = drafts.ne[0];
    require_vector(target_tokens, DType::I32, k + 1, op, "target_tokens");
    require_dtype(logits, DType::BF16, op, "logits");
    if (logits.ne[0] <= 0 || logits.ne[1] < k + 1 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("mtp_accept_tokens: logits must be [vocab, >=k+1]");
    }
    require_vector(drafts, DType::I32, k, op, "drafts");
    require_scalar(length, DType::I32, op, "length");
    require_scalar(token, DType::I32, op, "token");
    require_vector(sampled_out, DType::I32, k + 1, op, "sampled_out");
    require_scalar(num_sampled, DType::I32, op, "num_sampled");
    require_scalar(accepted, DType::I32, op, "accepted");
    require_scalar(ar_pos, DType::I32, op, "ar_pos");
    require_vector(stats, DType::I64, model::kStepStatsCounters, op, "stats");
    if (config == nullptr) {
        throw std::invalid_argument("mtp_accept_tokens: config must be non-null");
    }
    detail::mtp_accept_tokens_launch(target_tokens, logits, drafts, length, token, sampled_out,
                                     num_sampled, accepted, ar_pos, stats, config, stream);
}

void mtp_prepare_shifted_ids(const Tensor& verify_ids, const Tensor& token,
                             const Tensor& accepted, Tensor& shifted_ids, cudaStream_t stream) {
    constexpr const char* op = "mtp_prepare_shifted_ids";
    const std::int32_t T     = verify_ids.ne[0];
    require_vector(verify_ids, DType::I32, T, op, "verify_ids");
    require_scalar(token, DType::I32, op, "token");
    require_scalar(accepted, DType::I32, op, "accepted");
    require_vector(shifted_ids, DType::I32, T, op, "shifted_ids");
    detail::mtp_prepare_shifted_ids_launch(verify_ids, token, accepted, shifted_ids, stream);
}

void mtp_gather_hidden_row(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                           cudaStream_t stream) {
    constexpr const char* op = "mtp_gather_hidden_row";
    require_dtype(hidden, DType::BF16, op, "hidden");
    if (hidden.ne[0] != model::kCfg.hidden || hidden.ne[1] <= 0 || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1) {
        throw std::invalid_argument("mtp_gather_hidden_row: invalid shape for hidden");
    }
    require_scalar(accepted, DType::I32, op, "accepted");
    require_dtype(out, DType::BF16, op, "out");
    if (out.ne[0] != model::kCfg.hidden || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("mtp_gather_hidden_row: invalid shape for out");
    }
    detail::mtp_gather_hidden_row_launch(hidden, accepted, out, stream);
}

void mtp_remap_draft_token(Tensor& draft_token, const std::int32_t* id_map, std::int32_t n,
                           cudaStream_t stream) {
    constexpr const char* op = "mtp_remap_draft_token";
    require_scalar(draft_token, DType::I32, op, "draft_token");
    if (id_map == nullptr || n <= 0) {
        throw std::invalid_argument("mtp_remap_draft_token: id_map must be non-null and n>0");
    }
    detail::mtp_remap_draft_token_launch(draft_token, id_map, n, stream);
}

void mtp_increment_i32(Tensor& scalar, cudaStream_t stream) {
    constexpr const char* op = "mtp_increment_i32";
    require_scalar(scalar, DType::I32, op, "scalar");
    detail::mtp_increment_i32_launch(scalar, stream);
}

void mtp_count_fallback_step(Tensor& stats, cudaStream_t stream) {
    constexpr const char* op = "mtp_count_fallback_step";
    require_vector(stats, DType::I64, model::kStepStatsCounters, op, "stats");
    detail::mtp_count_fallback_step_launch(stats, stream);
}

void mtp_reset_gdn_initial_slot(Tensor& gdn_initial_slot, cudaStream_t stream) {
    constexpr const char* op = "mtp_reset_gdn_initial_slot";
    require_scalar(gdn_initial_slot, DType::I32, op, "gdn_initial_slot");
    detail::mtp_reset_gdn_initial_slot_launch(gdn_initial_slot, stream);
}

void mtp_set_gdn_initial_slot_from_accepted(const Tensor& accepted, Tensor& gdn_initial_slot,
                                            cudaStream_t stream) {
    constexpr const char* op = "mtp_set_gdn_initial_slot_from_accepted";
    require_scalar(accepted, DType::I32, op, "accepted");
    require_scalar(gdn_initial_slot, DType::I32, op, "gdn_initial_slot");
    detail::mtp_set_gdn_initial_slot_from_accepted_launch(accepted, gdn_initial_slot, stream);
}

} // namespace qus::kernels
