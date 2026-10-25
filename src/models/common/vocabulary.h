#pragma once

#include "models/common/parallel_execution.h"
#include "ninfer/ops/vocabulary.h"

namespace ninfer::models {

// Borrowed rank-local logits and shared sampling resources. All rows use the same
// collective sequence; their sampling law is read from Device SamplingConfig.
struct DistributedVocabulary {
    static constexpr int candidates = 20;
    std::uint32_t begin, domain;
    MemorySpan local, gathered;
    ParallelExecution& peer;
    MemoryArena& workspace;
    CudaExecutionView cuda;

    void sample(const Tensor& logits, const ops::SamplingConfig*, const Tensor& valid,
                const Tensor& positions, int purpose, Tensor& ids) const;
    void accept(const Tensor& logits, const ops::SamplingConfig*, const Tensor& valid,
                const Tensor& drafts, const Tensor& extents, Tensor& frontiers, Tensor& anchors,
                Tensor& licensed, Tensor& counts, Tensor& accepted) const;

private:
    [[nodiscard]] Tensor collect(const Tensor& logits, const ops::SamplingConfig*,
                                 const Tensor& valid, const Tensor& drafts) const;
};

} // namespace ninfer::models
