#pragma once

namespace ninfer::runtime {

template <class Instance>
BranchScoreResult EngineCore<Instance>::score_branches(
    std::span<const BranchScoreRow> rows, const PreparationControl& control, bool reuse_prefix,
    std::shared_ptr<const models::qwen::PreparedPromptData> prompt) {
    auto job = std::make_shared<BranchScoreQueue::Job>();
    job->rows.assign(rows.begin(), rows.end());
    job->reuse_prefix     = reuse_prefix;
    job->prompt           = std::move(prompt);
    job->control          = control;
    job->pending_deadline = Clock::now() + pending_timeout_;
    if (control.deadline != Clock::time_point{}) {
        job->pending_deadline = std::min(job->pending_deadline, control.deadline);
    }
    auto future = job->result.get_future();
    {
        std::lock_guard lock(queue_mutex_);
        if (stopping_ || failed_) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        // Decision batches do not occupy generation lanes or its outstanding-request budget.
        if (decisions_.pending.size() >= kDecisionRequestCapacity - 1) {
            throw RequestError(RequestErrorKind::Overloaded, "decision request queue is full");
        }
        decisions_.pending.push_back(job);
    }
    queue_cv_.notify_one();
    if (future.wait_until(job->pending_deadline) == std::future_status::timeout) {
        std::lock_guard lock(queue_mutex_);
        const auto queued = std::find(decisions_.pending.begin(), decisions_.pending.end(), job);
        if (queued != decisions_.pending.end()) {
            decisions_.pending.erase(queued);
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "decision queue deadline exceeded");
        }
    }
    // Once claimed, the worker must settle before the caller releases its borrowed token spans.
    return future.get();
}

template <class Instance>
bool EngineCore<Instance>::progress_branch_job() {
    if constexpr (!requires(typename ModelContract::Session& session,
                            std::span<const BranchScoreRow> rows,
                            const PreparationControl& control) {
                      session.score_branches(rows, control, true);
                  }) {
        return false;
    } else {
        if (!instance_.session->branch_scoring_available()) { return false; }
        std::shared_ptr<BranchScoreQueue::Job> job;
        {
            std::lock_guard lock(queue_mutex_);
            if (decisions_.pending.empty()) { return false; }
            job = std::move(decisions_.pending.front());
            decisions_.pending.pop_front();
            decisions_.active = job;
        }
        try {
            if (Clock::now() >= job->pending_deadline) {
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "decision queue deadline exceeded");
            }
            auto control         = job->control;
            control.cancellation = CancellationView([&, original = control.cancellation] {
                if (original.requested()) { return true; }
                std::lock_guard lock(queue_mutex_);
                return stopping_;
            });
            auto scored          = instance_.session->score_branches(
                job->rows, control, job->reuse_prefix && decisions_.cache_enabled, job->prompt);
            if (!decisions_.cache_enabled) { (void)instance_.session->release_branch_cache(); }
            publish_runtime_stats();
            job->result.set_value(std::move(scored));
        } catch (const RequestError&) {
            if (instance_.executor->failed()) throw;
            const auto error = std::current_exception();
            publish_runtime_stats();
            job->result.set_exception(error);
        }
        {
            std::lock_guard lock(queue_mutex_);
            decisions_.active.reset();
        }
        return true;
    }
}

template <class Instance>
bool EngineCore<Instance>::release_branch_workspace_before_chat() {
    if constexpr (!requires(typename ModelContract::Session& session) {
                      session.release_branch_cache();
                  }) {
        return true;
    } else {
        if (!instance_.session->branch_scoring_available()) return true;
        bool chat_pending;
        {
            std::lock_guard lock(queue_mutex_);
            chat_pending = !pending_.empty();
        }
        if (chat_pending || !resident_empty() || !paused_.empty() || materializing_ ||
            context_owner_) {
            // Return decision-owned KV pages and StateImage leases; chat cache policy is unchanged.
            if (instance_.session->release_branch_cache()) {
                request_admission_check();
            }
            return true;
        }
        // End this decision-only cycle. A chat arriving after the queue check must start a
        // new cycle and release the overlay before admission can use general workspace.
        return false;
    }
}

} // namespace ninfer::runtime
