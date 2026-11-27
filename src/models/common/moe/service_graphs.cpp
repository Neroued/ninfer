#include "models/common/moe/service_graphs.h"
#include "models/common/moe/step.h"
#include "core/arena.h"

namespace ninfer::models::moe {

void ExpertServiceGraphs::prepare(const BatchExecutionPlan& batches,
                                  const speculative::DecodeGraphRoutes& routes,
                                  GraphPreparation& preparation, MemoryArena& arena,
                                  CudaExecutionView cuda, const Pass& pass) {
    std::map<std::pair<bool, int>, std::uint32_t> passes;
    for (auto shape : batches.graph_shapes()) {
        for (const auto& route : routes.at(shape)) {
            const auto columns = shape.rows * route.width;
            passes.try_emplace({false, exchange_blocks(columns)}, columns);
            if (route.mtp_steps)
                passes.try_emplace({true, exchange_blocks(shape.rows)}, shape.rows);
        }
    }
    for (const auto& [key, columns] : passes) {
        const auto body = [&] { pass(key.first, columns); };
        if (preparation.capturing()) {
            auto& entry = graphs_.try_emplace(key).first->second;
            arena.reset_peak();
            entry.segment.prepare(cuda, body, &communication);
            entry.workspace_bytes = arena.peak_used();
            entry.segment.launch(cuda, &communication);
        } else {
            body();
        }
        preparation.checkpoint();
    }
}

void ExpertServiceGraphs::launch(bool draft, std::uint32_t columns, CudaExecutionView cuda) {
    auto& entry = graphs_.at({draft, exchange_blocks(columns)});
    entry.segment.launch(cuda, &communication);
    workspace_peak_ = std::max(workspace_peak_, entry.workspace_bytes);
}

} // namespace ninfer::models::moe
