#pragma once
#include "core/tensor.h"
#include "core/layout.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <cuda_runtime.h>

namespace ninfer::ops::candidate_path_detail {
enum class SelectorRoute { Direct, Lattice };

struct SelectorWorkspace {
    Tensor edges;
};

SelectorRoute select_candidate_path(int steps, int batch);

template <class Allocator>
SelectorWorkspace allocate_selector_workspace(Allocator& allocator, SelectorRoute route, int steps,
                                              int batch) {
    SelectorWorkspace out;
    if (route == SelectorRoute::Lattice)
        out.edges = allocator.alloc(DType::FP32, {16, 16, steps, batch}, 256);
    return out;
}

void execute_candidate_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                            const Tensor& projected_hidden, const Tensor& anchors,
                            const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                            const Tensor& base_positions, const SamplingConfig* configs,
                            Tensor& drafts, Tensor& proposal_q, MemoryArena& workspace,
                            cudaStream_t stream);
} // namespace ninfer::ops::candidate_path_detail
