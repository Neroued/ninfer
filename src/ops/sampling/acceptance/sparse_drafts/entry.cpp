#include "ninfer/ops/speculative_accept.h"
#include "ops/sampling/acceptance/sparse_drafts/plan.h"
#include "ops/sampling/acceptance/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSparseMaxDrafts    = 15;
constexpr std::int32_t kSparsePhysicalRows = 248320;
constexpr std::int32_t kSparseTokenDomain  = 248077;

} // namespace

std::size_t speculative_accept_sparse_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, SpeculativeAcceptExecutionEnvelope envelope, std::int32_t min_drafts,
    std::int32_t max_drafts, std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain != kSparseTokenDomain || min_drafts < 1 || max_drafts < min_drafts ||
        max_drafts > kSparseMaxDrafts || min_batch < 1 || max_batch < min_batch) {
        throw std::invalid_argument("sparse speculative accept workspace: unsupported profile");
    }
    if (envelope.all_rows_greedy_without_penalties) { return 0; }
    const std::size_t row_bytes = sparse_drafts_detail::capacity(token_domain, max_drafts + 1);
    return row_bytes * static_cast<std::size_t>(max_batch);
}

void speculative_accept_sparse_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& candidate_ids,
                                      const Tensor& proposal_q, const Tensor& current_extents,
                                      const Tensor& base_positions, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, std::int32_t token_domain,
                                      const SamplingConfig* configs,
                                      SpeculativeAcceptExecutionEnvelope envelope,
                                      const Tensor& live_rows, MemoryArena& workspace,
                                      CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: workspace belongs to another device");
    }
    constexpr const char* op = "speculative_accept_sparse_drafts";
    if (token_domain != kSparseTokenDomain) {
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: token_domain must be 248077");
    }
    const int steps   = drafts.ne[0];
    const int columns = licensed_tokens.ne[0];
    const int batch   = licensed_tokens.ne[1];
    if (steps < 1 || steps > kSparseMaxDrafts || columns < 2 || columns > kSparseMaxDrafts + 1)
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: invalid proposal or verification width");
    if (batch < 1) {
        throw std::invalid_argument("speculative_accept_sparse_drafts: B must be positive");
    }
    common::operand::tensor(target_tokens, DType::I32, {columns, batch}, 1, op, "target_tokens",
                            execution.device());
    common::operand::tensor(logits, DType::BF16, {kSparsePhysicalRows, columns, batch}, 1, op,
                            "logits", execution.device());
    common::operand::strided_tensor(drafts, DType::I32, {steps, batch}, 4, op, "drafts",
                                    execution.device());
    sampling_detail::sparse_candidate_plane(candidate_ids, DType::I32, steps, batch, op,
                                            "candidate_ids", execution.device());
    sampling_detail::sparse_candidate_plane(proposal_q, DType::FP32, steps, batch, op, "proposal_q",
                                            execution.device());
    common::operand::tensor(current_extents, DType::I32, {batch}, 1, op, "current_extents",
                            execution.device());
    common::operand::tensor(base_positions, DType::I32, {batch}, 1, op, "base_positions",
                            execution.device());
    common::operand::tensor(licensed_tokens, DType::I32, {columns, batch}, 1, op, "licensed_tokens",
                            execution.device());
    common::operand::tensor(licensed_counts, DType::I32, {batch}, 1, op, "licensed_counts",
                            execution.device());
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_sparse_drafts: configs must be non-null");
    }
    if (live_rows.data) {
        common::operand::tensor(live_rows, DType::I32, {batch}, 4, op, "live rows",
                                execution.device());
    }

    auto scratch_scope       = workspace.scope();
    const auto plan          = sparse_drafts_detail::select_plan(token_domain, columns,
                                                                 envelope.all_rows_greedy_without_penalties);
    const std::size_t bytes  = plan.workspace.bytes * static_cast<std::size_t>(batch);
    const MemorySpan scratch = bytes == 0 ? MemorySpan{} : workspace.alloc_bytes(bytes);
    sampling_detail::check_outputs({&licensed_tokens, &licensed_counts},
                                   {&target_tokens, &logits, &drafts, &candidate_ids, &proposal_q,
                                    &current_extents, &base_positions, &live_rows},
                                   configs, batch, op, scratch);
    sparse_drafts_detail::launch(target_tokens, logits, drafts, candidate_ids, proposal_q,
                                 current_extents, base_positions, licensed_tokens, licensed_counts,
                                 token_domain, configs, live_rows, plan, scratch, execution);
}

} // namespace ninfer::ops
