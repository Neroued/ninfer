#include "ninfer/ops/candidate_selector.h"

#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kCandidates   = 16;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kCodebookRows = 248320;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
    }
}

struct Range {
    const void* pointer;
    std::size_t bytes;
    const char* label;
};

bool overlaps(const Range& lhs, const Range& rhs) {
    const std::uintptr_t lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.pointer);
    const std::uintptr_t rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.pointer);
    if (lhs.bytes > std::numeric_limits<std::uintptr_t>::max() - lhs_begin ||
        rhs.bytes > std::numeric_limits<std::uintptr_t>::max() - rhs_begin) {
        throw std::overflow_error("candidate_selector_path: operand range overflows");
    }
    return lhs_begin < rhs_begin + rhs.bytes && rhs_begin < lhs_begin + lhs.bytes;
}

void require_nonoverlap(const Tensor& candidate_ids, const Tensor& unary_scores,
                        const Tensor& projected_hidden, const Tensor& anchors,
                        const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                        const Tensor& base_positions, const SamplingConfig* configs,
                        const Tensor& drafts, const Tensor& proposal_q) {
    const std::array<Range, 10> ranges{{
        {candidate_ids.data, candidate_ids.bytes(), "candidate_ids"},
        {unary_scores.data, unary_scores.bytes(), "unary_scores"},
        {projected_hidden.data, projected_hidden.bytes(), "projected_hidden"},
        {anchors.data, anchors.bytes(), "anchors"},
        {predecessor_codebook.data, predecessor_codebook.bytes(), "predecessor_codebook"},
        {successor_codebook.data, successor_codebook.bytes(), "successor_codebook"},
        {base_positions.data, base_positions.bytes(), "base_positions"},
        {configs, static_cast<std::size_t>(candidate_ids.ne[2]) * sizeof(SamplingConfig),
         "configs"},
        {drafts.data, drafts.bytes(), "drafts"},
        {proposal_q.data, proposal_q.bytes(), "proposal_q"},
    }};
    for (std::size_t first = 0; first < ranges.size(); ++first) {
        for (std::size_t second = first + 1; second < ranges.size(); ++second) {
            if (overlaps(ranges[first], ranges[second])) {
                throw std::invalid_argument(std::string("candidate_selector_path: ") +
                                            ranges[first].label + " overlaps " +
                                            ranges[second].label);
            }
        }
    }
}

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
            (void)detail::allocate_selector_workspace(
                layout, detail::candidate_selector_path_route(k, b), k, b);
        }
    return layout.peak_bytes();
}

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const std::int32_t batch_size = candidate_ids.ne[2];
    const std::int32_t kSteps     = candidate_ids.ne[1];
    if (kSteps < 1 || kSteps > 15)
        throw std::invalid_argument("candidate_selector_path: K must be in [1,15]");
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("candidate_selector_path: B must be in [1,8]");
    }
    require_tensor(candidate_ids, DType::I32, kCandidates, kSteps, batch_size, 1, "candidate_ids");
    require_tensor(unary_scores, DType::FP32, kCandidates, kSteps, batch_size, 1, "unary_scores");
    require_tensor(projected_hidden, DType::BF16, kRank, kSteps, batch_size, 1, "projected_hidden");
    require_tensor(anchors, DType::I32, batch_size, 1, 1, 1, "anchors");
    require_tensor(predecessor_codebook, DType::BF16, kRank, kCodebookRows, 1, 1,
                   "predecessor_codebook");
    require_tensor(successor_codebook, DType::BF16, kRank, kCodebookRows, 1, 1,
                   "successor_codebook");
    require_tensor(base_positions, DType::I32, batch_size, 1, 1, 1, "base_positions");
    require_tensor(drafts, DType::I32, kSteps, batch_size, 1, 1, "drafts");
    require_tensor(proposal_q, DType::FP32, kCandidates, kSteps, batch_size, 1, "proposal_q");
    if (!aligned_to(configs, alignof(SamplingConfig))) {
        throw std::invalid_argument("candidate_selector_path: invalid configs");
    }
    require_nonoverlap(candidate_ids, unary_scores, projected_hidden, anchors, predecessor_codebook,
                       successor_codebook, base_positions, configs, drafts, proposal_q);

    detail::candidate_selector_path_dispatch(
        candidate_ids, unary_scores, projected_hidden, anchors, predecessor_codebook,
        successor_codebook, base_positions, configs, drafts, proposal_q, workspace, stream);
}

} // namespace ninfer::ops
