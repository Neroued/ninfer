#include "ninfer/ops/speculative_accept.h"
#include "ops/sampling/acceptance/greedy_drafts/plan.h"
#include "ops/common/host/operands.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {

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
        greedy_drafts_detail::capacity(token_domain, min_drafts + 1, max_drafts + 1);
    if (row_bytes != 0 &&
        static_cast<std::size_t>(max_batch) > std::numeric_limits<std::size_t>::max() / row_bytes) {
        throw std::overflow_error("speculative accept workspace capacity overflows size_t");
    }
    return row_bytes * static_cast<std::size_t>(max_batch);
}

void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      const Tensor& base_positions, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, std::int32_t token_domain,
                                      const SamplingConfig* configs, MemoryArena& workspace,
                                      CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: workspace belongs to another device");
    }
    const auto stream        = execution.stream();
    constexpr const char* op = "speculative_accept_greedy_drafts";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || k == std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: K must be >=1");
    }
    if (batch < 1) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: B must be >=1");
    }
    common::operand::tensor(target_tokens, DType::I32, {k + 1, batch}, 1, op, "target_tokens",
                            execution.device());
    common::operand::dtype(logits, DType::BF16, op, "logits");
    common::operand::contiguous_nonnull(logits, op, "logits", execution.device());
    if (logits.ne[0] <= 0 || logits.ne[1] != k + 1 || logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: logits must be [physical_rows,K+1,B]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: token_domain must be in [1, logits.ne[0]]");
    }
    common::operand::strided_tensor(drafts, DType::I32, {k, batch}, 1, op, "drafts",
                                    execution.device());
    common::operand::tensor(current_extents, DType::I32, {batch}, 1, op, "current_extents",
                            execution.device());
    common::operand::tensor(base_positions, DType::I32, {batch}, 1, op, "base_positions",
                            execution.device());
    common::operand::tensor(licensed_tokens, DType::I32, {k + 1, batch}, 1, op, "licensed_tokens",
                            execution.device());
    common::operand::tensor(licensed_counts, DType::I32, {batch}, 1, op, "licensed_counts",
                            execution.device());
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: configs must be non-null");
    }
    auto scratch_scope       = workspace.scope();
    const auto plan          = greedy_drafts_detail::select_plan(token_domain, k + 1);
    const std::size_t bytes  = plan.workspace.bytes * static_cast<std::size_t>(batch);
    const MemorySpan scratch = bytes == 0 ? MemorySpan{} : workspace.alloc_bytes(bytes);
    greedy_drafts_detail::launch(target_tokens, logits, drafts, current_extents, base_positions,
                                 licensed_tokens, licensed_counts, token_domain, configs, plan,
                                 scratch, stream);
}

} // namespace ninfer::ops
