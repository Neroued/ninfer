#include "ops/common/host/operands.h"
#include "ops/sampling/candidate_path/plan.h"
#include "ops/sampling/candidate_path/launch.h"
#include <stdexcept>

namespace ninfer::ops::candidate_path_detail {
SelectorRoute select_candidate_path(int steps, int batch) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("invalid selector K/B");
    return steps <= 4 ? SelectorRoute::Direct : SelectorRoute::Lattice;
}

void execute_candidate_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                            const Tensor& projected_hidden, const Tensor& anchors,
                            const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                            const Tensor& base_positions, const SamplingConfig* configs,
                            Tensor& drafts, Tensor& proposal_q, MemoryArena& workspace,
                            cudaStream_t stream) {
    const auto route = select_candidate_path(candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope       = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, route, candidate_ids.ne[1], candidate_ids.ne[2]);
    const Tensor* live[]{
        &candidate_ids,      &unary_scores,   &projected_hidden, &anchors,   &predecessor_codebook,
        &successor_codebook, &base_positions, &drafts,           &proposal_q};
    for (const auto* work : {&scratch.edges}) {
        if (!work->data) continue;
        const auto range = common::operand::range(*work, "candidate_selector_path workspace");
        for (const auto* tensor : live)
            common::operand::disjoint(range,
                                      common::operand::range(*tensor, "candidate_selector_path"),
                                      "candidate_selector_path workspace");
        common::operand::disjoint(
            range,
            common::operand::range(configs, candidate_ids.ne[2] * sizeof(SamplingConfig),
                                   "candidate_selector_path"),
            "candidate_selector_path workspace");
    }
    launch_candidate_path(route, candidate_ids, unary_scores, projected_hidden, anchors,
                          predecessor_codebook, successor_codebook, base_positions, configs, drafts,
                          proposal_q, scratch, stream);
}
} // namespace ninfer::ops::candidate_path_detail
