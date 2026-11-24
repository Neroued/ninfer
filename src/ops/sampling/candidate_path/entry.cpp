#include "ops/common/host/operands.h"
#include "ninfer/ops/candidate_selector.h"

#include "ops/sampling/candidate_path/plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kCandidates   = 16;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kCodebookRows = 248320;

} // namespace

std::size_t candidate_selector_path_workspace_capacity_bytes(int min_steps, int max_steps,
                                                             int min_batch, int max_batch) {
    if (min_steps < 1 || max_steps > 15 || max_steps < min_steps || min_batch < 1 ||
        max_batch > 8 || max_batch < min_batch)
        throw std::invalid_argument("selector workspace: invalid K/B interval");
    WorkspaceLayoutBuilder layout;
    for (int k = min_steps; k <= max_steps; ++k)
        for (int b = min_batch; b <= max_batch; ++b) {
            auto scope = layout.scope();
            (void)candidate_path_detail::allocate_selector_workspace(
                layout, candidate_path_detail::select_candidate_path(k, b), k, b);
        }
    return layout.peak_bytes();
}

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, MemoryArena& workspace,
                             CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("candidate_selector_path: workspace belongs to another device");
    }
    const auto stream             = execution.stream();
    const std::int32_t batch_size = candidate_ids.ne[2];
    const std::int32_t kSteps     = candidate_ids.ne[1];
    if (kSteps < 1 || kSteps > 15)
        throw std::invalid_argument("candidate_selector_path: K must be in [1,15]");
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("candidate_selector_path: B must be in [1,8]");
    }
    common::operand::tensor(candidate_ids, DType::I32, {kCandidates, kSteps, batch_size, 1}, 16,
                            "candidate_selector_path", "candidate_ids", execution.device());
    common::operand::tensor(unary_scores, DType::FP32, {kCandidates, kSteps, batch_size, 1}, 16,
                            "candidate_selector_path", "unary_scores", execution.device());
    common::operand::tensor(projected_hidden, DType::BF16, {kRank, kSteps, batch_size, 1}, 16,
                            "candidate_selector_path", "projected_hidden", execution.device());
    common::operand::tensor(anchors, DType::I32, {batch_size, 1, 1, 1}, 16,
                            "candidate_selector_path", "anchors", execution.device());
    common::operand::tensor(predecessor_codebook, DType::BF16, {kRank, kCodebookRows, 1, 1}, 16,
                            "candidate_selector_path", "predecessor_codebook", execution.device());
    common::operand::tensor(successor_codebook, DType::BF16, {kRank, kCodebookRows, 1, 1}, 16,
                            "candidate_selector_path", "successor_codebook", execution.device());
    common::operand::tensor(base_positions, DType::I32, {batch_size, 1, 1, 1}, 16,
                            "candidate_selector_path", "base_positions", execution.device());
    common::operand::tensor(drafts, DType::I32, {kSteps, batch_size, 1, 1}, 16,
                            "candidate_selector_path", "drafts", execution.device());
    common::operand::tensor(proposal_q, DType::FP32, {kCandidates, kSteps, batch_size, 1}, 16,
                            "candidate_selector_path", "proposal_q", execution.device());
    if (!common::operand::aligned(configs, alignof(SamplingConfig))) {
        throw std::invalid_argument("candidate_selector_path: invalid configs");
    }
    const auto op = "candidate_selector_path";
    const std::array ranges{
        common::operand::range(candidate_ids, op),
        common::operand::range(unary_scores, op),
        common::operand::range(projected_hidden, op),
        common::operand::range(anchors, op),
        common::operand::range(predecessor_codebook, op),
        common::operand::range(successor_codebook, op),
        common::operand::range(base_positions, op),
        common::operand::range(drafts, op),
        common::operand::range(proposal_q, op),
        common::operand::range(configs, batch_size * sizeof(SamplingConfig), op)};
    common::operand::disjoint(ranges, op);

    candidate_path_detail::execute_candidate_path(
        candidate_ids, unary_scores, projected_hidden, anchors, predecessor_codebook,
        successor_codebook, base_positions, configs, drafts, proposal_q, workspace, stream);
}

} // namespace ninfer::ops
