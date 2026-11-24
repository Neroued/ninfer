#pragma once

#include "core/context_pool.h"
#include "models/glm5next/program/planning.h"

namespace ninfer::models::glm5next::detail {
class RankProgramImpl;

[[nodiscard]] std::size_t graph_context_bytes(const ExecutionPlan&);

class GraphInputs {
public:
    explicit GraphInputs(RankProgramImpl&);
    ~GraphInputs();
    void prepare();
    void set(int global, int local, int depth);
private:
    RankProgramImpl& owner_;
    std::unique_ptr<ContextPool> records_;
    std::vector<std::int64_t> states_;
    std::vector<AllocationId> allocations_;
    std::array<PinnedHostBuffer, 2> mappings_;
};
} // namespace ninfer::models::glm5next::detail
