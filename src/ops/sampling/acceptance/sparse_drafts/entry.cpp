#include "ninfer/ops/speculative_accept.h"
#include "ops/sampling/acceptance/sparse_drafts/plan.h"
#include "ops/common/host/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSparseMaxDrafts    = 15;
constexpr std::int32_t kSparseCandidates   = 16;
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
    const std::int32_t k = drafts.ne[0];
    if (k < 1 || k > kSparseMaxDrafts)
        throw std::invalid_argument("speculative_accept_sparse_drafts: K must be 1..15");
    const std::int32_t columns = k + 1;
    const std::int32_t batch   = drafts.ne[1];
    if (batch < 1) {
        throw std::invalid_argument("speculative_accept_sparse_drafts: B must be positive");
    }
    common::operand::tensor(target_tokens, DType::I32, {columns, batch}, 1, op, "target_tokens",
                            execution.device());
    common::operand::tensor(logits, DType::BF16, {kSparsePhysicalRows, columns, batch}, 1, op,
                            "logits", execution.device());
    common::operand::strided_tensor(drafts, DType::I32, {k, batch}, 1, op, "drafts",
                                    execution.device());
    common::operand::tensor(candidate_ids, DType::I32, {kSparseCandidates, k, batch}, 1, op,
                            "candidate_ids", execution.device());
    common::operand::tensor(proposal_q, DType::FP32, {kSparseCandidates, k, batch}, 1, op,
                            "proposal_q", execution.device());
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
        for (const auto* output : {&licensed_tokens, &licensed_counts})
            common::operand::disjoint(common::operand::range(live_rows, op),
                                      common::operand::range(*output, op), op);
    }

    auto scratch_scope       = workspace.scope();
    const auto plan          = sparse_drafts_detail::select_plan(token_domain, k + 1,
                                                                 envelope.all_rows_greedy_without_penalties);
    const std::size_t bytes  = plan.workspace.bytes * static_cast<std::size_t>(batch);
    const MemorySpan scratch = bytes == 0 ? MemorySpan{} : workspace.alloc_bytes(bytes);
    if (live_rows.data && scratch.data)
        common::operand::disjoint(
            common::operand::range(live_rows, "speculative_accept_sparse_drafts"),
            common::operand::range(scratch.data, scratch.bytes, "speculative_accept_sparse_drafts"),
            "speculative_accept_sparse_drafts");
    sparse_drafts_detail::launch(target_tokens, logits, drafts, candidate_ids, proposal_q,
                                 current_extents, base_positions, licensed_tokens, licensed_counts,
                                 token_domain, configs, live_rows, plan, scratch, execution);
}

} // namespace ninfer::ops
