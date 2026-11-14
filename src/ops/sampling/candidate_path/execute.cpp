#include "ops/common/host/operands.h"
#include "ops/sampling/candidate_path/plan.h"
#include "ops/sampling/candidate_path/launch.h"
#include <stdexcept>
#include <climits>

namespace ninfer::ops::candidate_path_detail {
int maximum_lattice_batch(CandidateSelectorInput input, int steps) {
    switch (input) {
    case CandidateSelectorInput::Candidates:
        return steps > 4 ? 7 : 0;
    case CandidateSelectorInput::GatheredKeys:
        return steps > 5 ? 2 : 0;
    }
    throw std::invalid_argument("invalid selector input");
}

SelectorPlan select_candidate_path(CandidateSelectorInput input, int steps, int batch) {
    if ((input != CandidateSelectorInput::Candidates &&
         input != CandidateSelectorInput::GatheredKeys) ||
        steps < 1 || steps > 15 || batch < 1 || std::int64_t(steps) * batch > INT_MAX)
        throw std::invalid_argument("invalid selector K/B");
    if (batch > maximum_lattice_batch(input, steps)) return {SelectorRoute::Direct, 16};
    const int warps =
        input == CandidateSelectorInput::Candidates || (batch == 1 && steps <= 9) ? 16 : 8;
    return {SelectorRoute::Lattice, warps};
}

void execute_candidate_path(CandidateSelectorInput input, const Tensor& candidate_ids,
                            const Tensor& unary_scores, const Tensor& keys,
                            const Tensor& projected_hidden, const Tensor& anchors,
                            const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                            const Tensor& base_positions, const SamplingConfig* configs,
                            Tensor& drafts, Tensor& proposal_q, const Tensor& live_rows,
                            MemoryArena& workspace, cudaStream_t stream) {
    const auto plan = select_candidate_path(input, candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope      = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, plan, candidate_ids.ne[1], candidate_ids.ne[2]);
    const Tensor* live[]{&candidate_ids,
                         input == CandidateSelectorInput::Candidates ? &unary_scores : &keys,
                         &projected_hidden,
                         &anchors,
                         &predecessor_codebook,
                         &successor_codebook,
                         &base_positions,
                         &drafts,
                         &proposal_q};
    for (const auto* work : {&scratch.edges}) {
        if (!work->data) continue;
        const auto range = common::operand::range(*work, "candidate_selector_path workspace");
        for (const auto* tensor : live)
            common::operand::disjoint(range,
                                      common::operand::range(*tensor, "candidate_selector_path"),
                                      "candidate_selector_path workspace");
        if (live_rows.data)
            common::operand::disjoint(range,
                                      common::operand::range(live_rows, "candidate_selector_path"),
                                      "candidate_selector_path workspace");
        common::operand::disjoint(
            range,
            common::operand::range(configs, candidate_ids.ne[2] * sizeof(SamplingConfig),
                                   "candidate_selector_path"),
            "candidate_selector_path workspace");
    }
    launch_candidate_path(plan, input, candidate_ids, unary_scores, keys, projected_hidden, anchors,
                          predecessor_codebook, successor_codebook, base_positions, configs, drafts,
                          proposal_q, live_rows, scratch, stream);
}
} // namespace ninfer::ops::candidate_path_detail
