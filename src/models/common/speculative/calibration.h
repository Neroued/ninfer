#pragma once

#include "models/common/speculative/cost_model.h"
#include "ninfer/observability/snapshot.h"

namespace ninfer::models::context {
class ContextDirectory;
}

namespace ninfer::models::speculative {

struct CalibrationProbeSpec {
    std::uint32_t context_tokens = 64;
    std::uint32_t logical_rows   = 1;
    RoundRoute route;
    // Row i belongs to owner i % owner_count, including for ragged probes.
    std::vector<std::uint32_t> verify_drafts;
    std::uint32_t warmup_launches = 1;
    // The runner can reduce this to one after measuring the warmup.
    std::uint32_t timed_launches = 2;
};

struct CalibrationConstraints {
    CostModelConfig model;
    const BatchExecutionPlan* batches = nullptr;
    ExecutionPath path                = ExecutionPath::Graph;
    const DecodeGraphRoutes* routes   = nullptr;
    std::uint32_t max_concurrency     = 1;
    std::uint32_t max_context         = 8192;
    std::uint32_t kv_page_tokens      = 64;
    std::uint64_t available_kv_pages  = 0; // Per required pool and per owner.
    double budget_ms                  = 6500.0;
};

// The runner enforces the wall-clock budget while executing this priority order.
[[nodiscard]] std::vector<CalibrationProbeSpec>
build_corner_calibration_schedule(const CalibrationConstraints&);

[[nodiscard]] std::uint32_t available_calibration_pages(const context::ContextDirectory&,
                                                        std::span<const PlacementDescriptor>);

} // namespace ninfer::models::speculative
