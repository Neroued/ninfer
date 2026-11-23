#pragma once

#include "models/common/parallel_execution.h"
#include "ninfer/ops/vocabulary.h"

namespace ninfer::models {

// Borrowed rank-local logits and shared sampling resources. All rows use the same
// collective sequence; their sampling law is read from Device SamplingConfig.
struct DistributedVocabulary {
    static constexpr int candidates          = 20;
    static constexpr int candidates_with_lse = candidates + 1;
    std::uint32_t begin, domain;
    MemorySpan local, gathered;
    ParallelExecution& peer;
    MemoryArena& workspace;
    CudaExecutionView cuda;

    void sample(const Tensor& logits, const ops::SamplingConfig*, const Tensor& valid,
                const Tensor& live_rows, const Tensor& positions, int purpose, Tensor& ids) const;
    void sample(const Tensor& logits, const ops::SamplingConfig*, const Tensor& valid,
                const Tensor& live_rows, const Tensor& positions, int purpose, Tensor& ids,
                Tensor& top1_probabilities) const;
    void accept(const Tensor& logits, const ops::SamplingConfig*, const Tensor& valid,
                const Tensor& live_rows, const Tensor& drafts, const Tensor& extents,
                const Tensor& positions, Tensor& licensed, Tensor& counts) const;

    void accept_sparse(const Tensor& logits, const ops::SamplingConfig*, const Tensor& valid,
                       const Tensor& live_rows, const Tensor& drafts, const Tensor& candidate_ids,
                       const Tensor& q, const Tensor& extents, const Tensor& positions,
                       Tensor& licensed, Tensor& counts) const;

private:
    template <bool RelativeLse = false>
    [[nodiscard]] Tensor collect(const Tensor& logits, const ops::SamplingConfig*,
                                 const Tensor& valid, const Tensor& drafts) const;
};

} // namespace ninfer::models
