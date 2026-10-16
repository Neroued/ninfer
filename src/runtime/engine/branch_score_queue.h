#pragma once

#include "ninfer/branch_score.h"

#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <vector>

namespace ninfer::runtime {

// EngineCore's queue mutex guards admission; its worker settles every borrowed token span.
struct BranchScoreQueue {
    struct Job {
        std::vector<BranchScoreRow> rows;
        bool reuse_prefix = true;
        PreparationControl control;
        std::chrono::steady_clock::time_point pending_deadline;
        std::promise<BranchScoreResult> result;
    };

    BranchScoreQueue(bool enabled, bool cache_enabled)
        : enabled(enabled), cache_enabled(cache_enabled) {}

    const bool enabled;
    const bool cache_enabled;
    std::deque<std::shared_ptr<Job>> pending;
};

} // namespace ninfer::runtime
