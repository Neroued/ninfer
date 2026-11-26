#pragma once

#include "core/context_pool.h"
#include "models/inkling2/program/planning.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::inkling2::detail {
class RankProgramImpl;

[[nodiscard]] std::size_t graph_context_bytes(const ExecutionPlan&);

class GraphInputs {
public:
    explicit GraphInputs(RankProgramImpl&);
    ~GraphInputs();
    GraphInputs(const GraphInputs&)            = delete;
    GraphInputs& operator=(const GraphInputs&) = delete;

    void prepare();
    void set(std::int32_t rows, std::int32_t width);

private:
    RankProgramImpl& owner_;
    std::unique_ptr<ContextPool> records_;
    std::vector<AllocationId> allocations_;
    std::vector<std::int64_t> states_;
    PinnedHostBuffer mapping_;
};

} // namespace ninfer::models::inkling2::detail
