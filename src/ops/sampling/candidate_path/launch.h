#pragma once
#include "ops/sampling/candidate_path/plan.h"

namespace ninfer::ops::candidate_path_detail {
void launch_candidate_path(SelectorRoute route, const Tensor& candidate_ids,
                           const Tensor& unary_scores, const Tensor& projected_hidden,
                           const Tensor& anchors, const Tensor& predecessor_codebook,
                           const Tensor& successor_codebook, const Tensor& base_positions,
                           const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                           const SelectorWorkspace& workspace, cudaStream_t stream);
}
