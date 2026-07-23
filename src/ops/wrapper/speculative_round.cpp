#include "ninfer/ops/speculative_round.h"
#include "ops/launcher/speculative_round.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
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

void require_vector_at_least(const Tensor& t, DType dtype, std::int32_t n, const char* op,
                             const char* name) {
    require_dtype(t, dtype, op, name);
    if (n <= 0 || t.ne[0] < n || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
}

} // namespace

void speculative_prepare_verify_inputs(const Tensor& token, const Tensor& drafts,
                                       const Tensor& length, Tensor& verify_ids, Tensor& positions,
                                       cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    if (k < 1) { throw std::invalid_argument("speculative_prepare_verify_inputs: K must be >=1"); }
    require_scalar(token, DType::I32, op, "token");
    require_vector(drafts, DType::I32, k, op, "drafts");
    require_scalar(length, DType::I32, op, "length");
    require_vector(verify_ids, DType::I32, k + 1, op, "verify_ids");
    require_vector(positions, DType::I32, k + 1, op, "positions");
    detail::speculative_prepare_verify_inputs_launch(token, drafts, length, verify_ids, positions,
                                                     stream);
}

void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, Tensor& length, Tensor& token,
                                      Tensor& sampled_out, Tensor& num_sampled, Tensor& accepted,
                                      Tensor& stats, std::int32_t token_domain,
                                      const SamplingConfig* config, WorkspaceArena& workspace,
                                      cudaStream_t stream) {
    constexpr const char* op = "speculative_accept_greedy_drafts";
    const std::int32_t k     = drafts.ne[0];
    if (k < 1) { throw std::invalid_argument("speculative_accept_greedy_drafts: K must be >=1"); }
    require_vector(target_tokens, DType::I32, k + 1, op, "target_tokens");
    require_dtype(logits, DType::BF16, op, "logits");
    if (logits.ne[0] <= 0 || logits.ne[1] < k + 1 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: logits must be [physical_rows, >=k+1]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: token_domain must be in [1, logits.ne[0]]");
    }
    require_vector(drafts, DType::I32, k, op, "drafts");
    require_scalar(length, DType::I32, op, "length");
    require_scalar(token, DType::I32, op, "token");
    require_vector(sampled_out, DType::I32, k + 1, op, "sampled_out");
    require_scalar(num_sampled, DType::I32, op, "num_sampled");
    require_scalar(accepted, DType::I32, op, "accepted");
    require_vector_at_least(stats, DType::I64, 4 + k, op, "stats");
    if (config == nullptr) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: config must be non-null");
    }
    auto scratch_scope       = workspace.scope();
    const std::size_t bytes  = sampling_workspace_bytes(token_domain, k + 1);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::speculative_accept_greedy_drafts_launch(target_tokens, logits, drafts, length, token,
                                                    sampled_out, num_sampled, accepted, stats,
                                                    token_domain, config, scratch, stream);
}

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                                        cudaStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_dtype(hidden, DType::BF16, op, "hidden");
    if (hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid hidden shape");
    }
    require_scalar(accepted, DType::I32, op, "accepted");
    require_dtype(out, DType::BF16, op, "out");
    if (out.ne[0] != hidden.ne[0] || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid output shape");
    }
    detail::speculative_select_accepted_hidden_launch(hidden, accepted, out, stream);
}

void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map, std::int32_t n,
                              cudaStream_t stream) {
    constexpr const char* op = "proposal_remap_token_ids";
    require_dtype(proposal_tokens, DType::I32, op, "proposal_tokens");
    if (proposal_tokens.ne[0] <= 0 || proposal_tokens.ne[1] != 1 || proposal_tokens.ne[2] != 1 ||
        proposal_tokens.ne[3] != 1) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: proposal_tokens must be a non-empty vector");
    }
    if (id_map == nullptr || n <= 0) {
        throw std::invalid_argument("proposal_remap_token_ids: id_map must be non-null and n>0");
    }
    detail::proposal_remap_token_ids_launch(proposal_tokens, id_map, n, stream);
}

} // namespace ninfer::ops
