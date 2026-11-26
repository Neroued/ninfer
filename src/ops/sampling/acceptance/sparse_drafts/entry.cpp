#include "ninfer/ops/speculative_accept.h"
#include "ops/sampling/acceptance/sparse_drafts/plan.h"
#include "ops/sampling/acceptance/operands.h"
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSparseMaxDrafts = 15;

std::size_t batch_capacity(std::size_t row_bytes, int batch) {
    const auto rows = static_cast<std::size_t>(batch);
    if (row_bytes > std::numeric_limits<std::size_t>::max() / rows)
        throw std::overflow_error("sparse speculative accept workspace: batch size overflow");
    return row_bytes * rows;
}

} // namespace

std::size_t speculative_accept_sparse_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, SpeculativeAcceptExecutionEnvelope envelope, std::int32_t min_drafts,
    std::int32_t max_drafts, std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain < 1 || min_drafts < 1 || max_drafts < min_drafts ||
        max_drafts > kSparseMaxDrafts || min_batch < 1 || max_batch < min_batch) {
        throw std::invalid_argument("sparse speculative accept workspace: unsupported profile");
    }
    if (envelope.all_rows_greedy_without_penalties) { return 0; }
    using namespace sparse_drafts_detail;
    if (!use_streaming(token_domain))
        return batch_capacity(select_plan(token_domain, max_drafts + 1, max_batch).row_bytes,
                              max_batch);

    std::size_t high_water = 0;
    for (int columns = min_drafts + 1; columns <= max_drafts + 1; ++columns) {
        // Beyond this bound the partition count is constant and capacity grows with B.
        const int crossover = kTargetSms * kMaxStreamingWaves / columns;
        const int last      = std::min(max_batch, crossover);
        for (int batch = min_batch; batch <= last; ++batch) {
            const auto plan = select_plan(token_domain, columns, batch);
            high_water      = std::max(high_water, batch_capacity(plan.row_bytes, batch));
        }
        const auto plan = select_plan(token_domain, columns, max_batch);
        high_water      = std::max(high_water, batch_capacity(plan.row_bytes, max_batch));
    }
    return high_water;
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
    const int steps          = drafts.ne[0];
    const int columns        = licensed_tokens.ne[0];
    const int batch          = licensed_tokens.ne[1];
    if (steps < 1 || steps > kSparseMaxDrafts || columns < 2 || columns > kSparseMaxDrafts + 1)
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: invalid proposal or verification width");
    if (batch < 1) {
        throw std::invalid_argument("speculative_accept_sparse_drafts: B must be positive");
    }
    common::operand::tensor(target_tokens, DType::I32, {columns, batch}, 1, op, "target_tokens",
                            execution.device());
    common::operand::dtype(logits, DType::BF16, op, "logits");
    common::operand::contiguous_nonnull(logits, op, "logits", execution.device());
    if (logits.ne[0] < 1 || logits.ne[1] != columns || logits.ne[2] != batch || logits.ne[3] != 1)
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: logits must be [physical_rows,W,B]");
    if (token_domain < 1 || token_domain > logits.ne[0])
        throw std::invalid_argument(
            "speculative_accept_sparse_drafts: token_domain must be in [1, logits.ne[0]]");
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
    const auto plan          = sparse_drafts_detail::select_plan(token_domain, columns, batch,
                                                                 envelope.all_rows_greedy_without_penalties,
                                                                 execution.multiprocessor_count());
    const std::size_t bytes  = batch_capacity(plan.row_bytes, batch);
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
