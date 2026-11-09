#include "ops/common/host/operands.h"
#include "ninfer/ops/candidate_selector.h"
#include "ops/sampling/candidate_path/plan.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops {
namespace {
constexpr const char* kOp = "candidate_selector_path";

void validate_common(int steps, int batch, const Tensor& hidden, const Tensor& anchors,
                     const Tensor& predecessor, const Tensor& successor, int token_domain,
                     const Tensor& positions, const SamplingConfig* configs, const Tensor& drafts,
                     const Tensor& q, MemoryArena& workspace, CudaExecutionView execution) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("candidate_selector_path: K must be 1..15, B must be 1..8");
    if (workspace.device() != execution.device())
        throw std::invalid_argument("candidate_selector_path: workspace belongs to another device");
    const int rows = predecessor.ne[1];
    if (token_domain < 1 || token_domain > rows)
        throw std::invalid_argument("candidate_selector_path: invalid token domain");
    const auto tensor = [&](const Tensor& t, DType type, std::initializer_list<int> shape,
                            const char* name) {
        common::operand::tensor(t, type, shape, 16, kOp, name, execution.device());
    };
    tensor(hidden, DType::BF16, {256, steps, batch}, "projected hidden");
    tensor(anchors, DType::I32, {batch}, "anchors");
    tensor(predecessor, DType::BF16, {256, rows}, "predecessor codebook");
    tensor(successor, DType::BF16, {256, rows}, "successor codebook");
    tensor(positions, DType::I32, {batch}, "positions");
    tensor(drafts, DType::I32, {steps, batch}, "drafts");
    tensor(q, DType::FP32, {16, steps, batch}, "proposal q");
    if (!common::operand::aligned(configs, alignof(SamplingConfig)))
        throw std::invalid_argument("candidate_selector_path: invalid configs");
}

void execute(CandidateSelectorInput input, const Tensor& ids, const Tensor& unary,
             const Tensor& keys, const Tensor& hidden, const Tensor& anchors,
             const Tensor& predecessor, const Tensor& successor, int token_domain,
             const Tensor& positions, const SamplingConfig* configs, Tensor& drafts, Tensor& q,
             MemoryArena& workspace, CudaExecutionView execution) {
    const int steps = ids.ne[1], batch = ids.ne[2];
    validate_common(steps, batch, hidden, anchors, predecessor, successor, token_domain, positions,
                    configs, drafts, q, workspace, execution);
    common::operand::tensor(ids, DType::I32, {16, steps, batch}, 16, kOp, "candidate IDs",
                            execution.device());
    if (input == CandidateSelectorInput::Candidates)
        common::operand::tensor(unary, DType::FP32, {16, steps, batch}, 16, kOp, "unary scores",
                                execution.device());
    else
        common::operand::tensor(keys, DType::I64, {16, steps, batch, 2}, 8, kOp, "gathered keys",
                                execution.device());
    const std::array ranges{
        common::operand::range(ids, kOp),
        common::operand::range(input == CandidateSelectorInput::Candidates ? unary : keys, kOp),
        common::operand::range(hidden, kOp),
        common::operand::range(anchors, kOp),
        common::operand::range(predecessor, kOp),
        common::operand::range(successor, kOp),
        common::operand::range(positions, kOp),
        common::operand::range(drafts, kOp),
        common::operand::range(q, kOp),
        common::operand::range(configs, batch * sizeof(SamplingConfig), kOp)};
    common::operand::disjoint(ranges, kOp);
    candidate_path_detail::execute_candidate_path(input, ids, unary, keys, hidden, anchors,
                                                  predecessor, successor, positions, configs,
                                                  drafts, q, workspace, execution.stream());
}
} // namespace

std::size_t candidate_selector_path_workspace_capacity_bytes(CandidateSelectorInput input,
                                                             int min_steps, int max_steps,
                                                             int min_batch, int max_batch) {
    if (min_steps < 1 || max_steps > 15 || max_steps < min_steps || min_batch < 1 ||
        max_batch > 8 || max_batch < min_batch)
        throw std::invalid_argument("selector workspace: invalid K/B interval");
    WorkspaceLayoutBuilder layout;
    for (int k = min_steps; k <= max_steps; ++k)
        for (int b = min_batch; b <= max_batch; ++b) {
            auto scope = layout.scope();
            (void)candidate_path_detail::allocate_selector_workspace(
                layout, candidate_path_detail::select_candidate_path(input, k, b), k, b);
        }
    return layout.peak_bytes();
}

void candidate_selector_path(const Tensor& ids, const Tensor& unary, const Tensor& hidden,
                             const Tensor& anchors, const Tensor& predecessor,
                             const Tensor& successor, int token_domain, const Tensor& positions,
                             const SamplingConfig* configs, Tensor& drafts, Tensor& q,
                             MemoryArena& workspace, CudaExecutionView execution) {
    execute(CandidateSelectorInput::Candidates, ids, unary, {}, hidden, anchors, predecessor,
            successor, token_domain, positions, configs, drafts, q, workspace, execution);
}

void candidate_selector_path(const Tensor& keys, const Tensor& hidden, const Tensor& anchors,
                             const Tensor& predecessor, const Tensor& successor, int token_domain,
                             const Tensor& positions, const SamplingConfig* configs, Tensor& ids,
                             Tensor& drafts, Tensor& q, MemoryArena& workspace,
                             CudaExecutionView execution) {
    execute(CandidateSelectorInput::GatheredKeys, ids, {}, keys, hidden, anchors, predecessor,
            successor, token_domain, positions, configs, drafts, q, workspace, execution);
}
} // namespace ninfer::ops
