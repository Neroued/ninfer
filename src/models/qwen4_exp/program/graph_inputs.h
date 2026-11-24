#pragma once

#include "core/buffer.h"
#include "core/context_pool.h"
#include "models/qwen4_exp/execution/batch.h"

namespace ninfer::models::qwen4_exp::detail {
class RankProgramImpl;

class GraphInputs {
public:
    explicit GraphInputs(RankProgramImpl& p) : owner_(p) {}

    ~GraphInputs();
    void prepare();
    void batch(execution::BatchView, std::uint32_t visible);
    void stage();
    void stage_proposal(std::uint32_t rows, std::uint32_t depth, std::uint32_t visible);
    void stage_commit(std::uint32_t rows, std::uint32_t width);

private:
    RankProgramImpl& owner_;
    std::unique_ptr<ContextPool> records_;
    std::vector<AllocationId> allocations_;
    std::vector<std::int64_t> state_offsets_;
    std::array<PinnedHostBuffer, 2> mappings_;
};

} // namespace ninfer::models::qwen4_exp::detail
