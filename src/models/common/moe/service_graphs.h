#pragma once

#include "models/common/graph.h"
#include "models/common/speculative/round_plan.h"

#include <map>

namespace ninfer {
class MemoryArena;
}

namespace ninfer::models::moe {

// A service pass owns only expert backing. A captured pass can serve every local decode
// shape with the same number of transport blocks, independently of its attention geometry.
class ExpertServiceGraphs {
public:
    explicit ExpertServiceGraphs(const distributed::Group& group) : communication({&group}) {}

    using Pass = std::function<void(bool draft, std::uint32_t columns)>;
    void prepare(const BatchExecutionPlan&, const speculative::DecodeGraphRoutes&,
                 GraphPreparation&, MemoryArena&, CudaExecutionView, const Pass&);
    void launch(bool draft, std::uint32_t columns, CudaExecutionView);

    void clear() {
        graphs_.clear();
        reset_workspace_peak();
    }

    [[nodiscard]] std::size_t workspace_peak() const noexcept { return workspace_peak_; }

    void reset_workspace_peak() noexcept { workspace_peak_ = 0; }

    ParallelExecution communication;
private:
    struct Entry {
        GraphSegment segment;
        std::size_t workspace_bytes = 0;
    };

    std::map<std::pair<bool, int>, Entry> graphs_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::moe
