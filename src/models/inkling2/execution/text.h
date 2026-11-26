#pragma once

#include "models/common/vision/execution.h"
#include "models/inkling2/execution/relative_attention.h"

#include <optional>
#include <vector>

namespace ninfer::models::inkling2::execution {

struct TextExecution {
    MemoryArena& workspace;
    AttentionExecution attention;
};

class TextRun {
public:
    TextRun(const Parameters&, TextExecution, BatchView, StateUpdate, CudaExecutionView);

    void stem(std::optional<models::vision::EmbeddingView> visual = {});
    void forward(Tensor& final_hidden, const FullAttentionSubmit& submit = {});

    // Computes the exact FullAttentionReadout bindings for batched CUDA Graph capture.
    [[nodiscard]] std::vector<FullAttentionReadout>
    plan_full_readouts(ops::ContextBucketRange buckets) const;

private:
    [[nodiscard]] RelativeAttentionRun attention_run(const LogicalLayerConfig&) const;

    const Parameters& parameters_;
    TextExecution execution_;
    BatchView batch_;
    StateUpdate update_;
    CudaExecutionView cuda_;
    std::int32_t columns_;
    MemoryArena::Scope scope_;
    workspace::MainRoots roots_;
};

} // namespace ninfer::models::inkling2::execution
