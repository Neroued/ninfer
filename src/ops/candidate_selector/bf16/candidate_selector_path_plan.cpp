#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include <stdexcept>

namespace ninfer::ops::detail {
SelectorRoute candidate_selector_path_route(int steps, int batch) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("invalid selector K/B");
    return steps <= 4 ? SelectorRoute::Direct : SelectorRoute::Lattice;
}

const char* selector_route_name(SelectorRoute route) {
    return route == SelectorRoute::Direct ? "direct.t512" : "lattice.t512";
}

const char* candidate_selector_path_route_name(int steps, int batch) {
    return selector_route_name(candidate_selector_path_route(steps, batch));
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream) {
    const auto route = candidate_selector_path_route(candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope       = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, route, candidate_ids.ne[1], candidate_ids.ne[2]);
    const Tensor* live[]{
        &candidate_ids,      &unary_scores,   &projected_hidden, &anchors,   &predecessor_codebook,
        &successor_codebook, &base_positions, &drafts,           &proposal_q};
    for (const auto* work : {&scratch.edges}) {
        if (!work->data) continue;
        const auto begin = reinterpret_cast<std::uintptr_t>(work->data),
                   end   = begin + work->bytes();
        for (const auto* tensor : live) {
            const auto tb = reinterpret_cast<std::uintptr_t>(tensor->data);
            if (begin < tb + tensor->bytes() && tb < end)
                throw std::invalid_argument("selector workspace overlaps operand");
        }
        const auto cb = reinterpret_cast<std::uintptr_t>(configs);
        if (begin < cb + candidate_ids.ne[2] * sizeof(SamplingConfig) && cb < end)
            throw std::invalid_argument("selector workspace overlaps configs");
    }
    candidate_selector_path_launch(route, candidate_ids, unary_scores, projected_hidden, anchors,
                                   predecessor_codebook, successor_codebook, base_positions,
                                   configs, drafts, proposal_q, scratch, stream);
}
} // namespace ninfer::ops::detail
