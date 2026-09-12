#include "ninfer/ops/speculative_round.h"
#include "ops/launcher/speculative_round.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSparseMaxDrafts    = 15;
constexpr std::int32_t kSparseCandidates   = 16;
constexpr std::int32_t kSparsePhysicalRows = 248320;
constexpr std::int32_t kSparseTokenDomain  = 248077;
constexpr std::int32_t kSparseMaxBatch     = 8;

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

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                    const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
}

void require_tensor3(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1,
                     std::int32_t n2, const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (n0 <= 0 || n1 <= 0 || n2 <= 0 || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

} // namespace

std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(std::int32_t token_domain,
                                                                      std::int32_t min_drafts,
                                                                      std::int32_t max_drafts,
                                                                      std::int32_t min_batch,
                                                                      std::int32_t max_batch) {
    if (token_domain <= 0 || min_drafts <= 0 || max_drafts < min_drafts || min_batch <= 0 ||
        max_batch < min_batch || max_drafts == std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("speculative accept workspace: invalid draft interval");
    }
    const std::size_t row_bytes =
        sampling_workspace_capacity_bytes(token_domain, min_drafts + 1, max_drafts + 1);
    if (row_bytes != 0 &&
        static_cast<std::size_t>(max_batch) > std::numeric_limits<std::size_t>::max() / row_bytes) {
        throw std::overflow_error("speculative accept workspace capacity overflows size_t");
    }
    return row_bytes * static_cast<std::size_t>(max_batch);
}

std::size_t speculative_accept_sparse_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, SpeculativeAcceptExecutionEnvelope envelope, std::int32_t min_drafts,
    std::int32_t max_drafts, std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain != kSparseTokenDomain || min_drafts < 1 || max_drafts < min_drafts ||
        max_drafts > kSparseMaxDrafts || min_batch < 1 || max_batch < min_batch ||
        max_batch > kSparseMaxBatch) {
        throw std::invalid_argument("sparse speculative accept workspace: unsupported profile");
    }
    if (envelope.all_rows_greedy_without_penalties) { return 0; }
    const std::size_t row_bytes =
        sampling_workspace_capacity_bytes(token_domain, min_drafts + 1, max_drafts + 1);
    return row_bytes * static_cast<std::size_t>(max_batch);
}

void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("speculative_prepare_verify_inputs: K must be >=1"); }
    if (batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_inputs: B must be >=1");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(base_positions, DType::I32, batch, op, "base_positions");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    require_matrix(positions, DType::I32, k + 1, batch, op, "positions");
    detail::speculative_prepare_verify_inputs_launch(
        anchors, drafts, base_positions, current_extents, verify_ids, positions, stream);
}

void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_ids";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_ids: K and B must be positive");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    detail::speculative_prepare_verify_ids_launch(anchors, drafts, current_extents, verify_ids,
                                                  stream);
}

void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, Tensor& accepted,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "speculative_accept_greedy_drafts";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("speculative_accept_greedy_drafts: K must be >=1"); }
    if (batch < 1) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: B must be >=1");
    }
    require_matrix(target_tokens, DType::I32, k + 1, batch, op, "target_tokens");
    require_dtype(logits, DType::BF16, op, "logits");
    if (logits.ne[0] <= 0 || logits.ne[1] != k + 1 || logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: logits must be [physical_rows,K+1,B]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: token_domain must be in [1, logits.ne[0]]");
    }
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(lengths, DType::I32, batch, op, "lengths");
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(licensed_tokens, DType::I32, k + 1, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: configs must be non-null");
    }
    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch, batch);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::speculative_accept_greedy_drafts_launch(
        target_tokens, logits, drafts, current_extents, lengths, anchors, licensed_tokens,
        licensed_counts, accepted, token_domain, configs, scratch, stream);
}

void speculative_accept_sparse_drafts(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
    Tensor& round_lengths, Tensor& round_anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted_drafts, std::int32_t token_domain, const SamplingConfig* configs,
    SpeculativeAcceptExecutionEnvelope envelope, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "speculative_accept_sparse_drafts";
    if (token_domain != kSparseTokenDomain) {
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: token_domain must be 248077");
    }
    const std::int32_t k = drafts.ne[0];
    if (k < 1 || k > kSparseMaxDrafts)
        throw std::invalid_argument("speculative_accept_sparse_drafts: K must be 1..15");
    const std::int32_t columns = k + 1;
    const std::int32_t batch   = drafts.ne[1];
    if (batch < 1 || batch > kSparseMaxBatch) {
        throw std::invalid_argument("speculative_accept_sparse_drafts: B must be 1..8");
    }
    require_matrix(target_tokens, DType::I32, columns, batch, op, "target_tokens");
    require_tensor3(logits, DType::BF16, kSparsePhysicalRows, columns, batch, op, "logits");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_tensor3(candidate_ids, DType::I32, kSparseCandidates, k, batch, op, "candidate_ids");
    require_tensor3(proposal_q, DType::FP32, kSparseCandidates, k, batch, op, "proposal_q");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(round_lengths, DType::I32, batch, op, "round_lengths");
    require_vector(round_anchors, DType::I32, batch, op, "round_anchors");
    require_matrix(licensed_tokens, DType::I32, columns, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted_drafts, DType::I32, batch, op, "accepted_drafts");
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_sparse_drafts: configs must be non-null");
    }

    auto scratch_scope      = workspace.scope();
    const std::size_t bytes = speculative_accept_sparse_drafts_workspace_capacity_bytes(
        token_domain, envelope, k, k, batch, batch);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::speculative_accept_sparse_drafts_launch(
        target_tokens, logits, drafts, candidate_ids, proposal_q, current_extents, round_lengths,
        round_anchors, licensed_tokens, licensed_counts, accepted_drafts, token_domain, configs,
        envelope.all_rows_greedy_without_penalties, scratch, stream);
}

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_dtype(hidden, DType::BF16, op, "hidden");
    if (hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid hidden shape");
    }
    require_vector(selectors, DType::I32, hidden.ne[2], op, "selectors");
    require_dtype(out, DType::BF16, op, "out");
    if (out.ne[0] != hidden.ne[0] || out.ne[1] != hidden.ne[2] || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid output shape");
    }
    detail::speculative_select_accepted_hidden_launch(hidden, selectors, out, stream);
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
