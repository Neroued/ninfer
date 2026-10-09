#pragma once

namespace ninfer::runtime {

template <class Instance>
BranchScoreResult EngineCore<Instance>::score_branches(std::span<const BranchScoreRow> rows,
                                                       const PreparationControl& control) {
    auto job = std::make_shared<BranchJob>();
    job->rows.assign(rows.begin(), rows.end());
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
        constexpr std::size_t kMaximumPendingDecisions = 16;
        if (branch_pending_.size() >= kMaximumPendingDecisions) {
            throw RequestError(RequestErrorKind::Overloaded, "decision request queue is full");
        }
        branch_pending_.push_back(job);
    }
    queue_cv_.notify_one();
    // Wait through cancellation cleanup: queued GPU work borrows these token spans.
    return future.get();
}

template <class Instance>
bool EngineCore<Instance>::progress_branch_job() {
    if (instance_.program->has_context_transaction()) { return false; }
    std::shared_ptr<BranchJob> job;
    {
        std::lock_guard lock(queue_mutex_);
        if (branch_pending_.empty()) { return false; }
        job = std::move(branch_pending_.front());
        branch_pending_.pop_front();
    }
    try {
        if (Clock::now() >= job->pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout, "decision queue deadline exceeded");
        }
        auto control         = job->control;
        control.cancellation = CancellationView([&, original = control.cancellation] {
            if (original.requested()) { return true; }
            std::lock_guard lock(queue_mutex_);
            return stopping_;
        });
        job->result.set_value(instance_.program->score_branches(job->rows, control));
    } catch (const RequestError&) {
        job->result.set_exception(std::current_exception());
    } catch (...) {
        job->result.set_exception(std::current_exception());
        throw;
    }
    return true;
}

template <class Instance>
bool EngineCore<Instance>::release_branch_workspace_before_chat() {
    bool chat_pending;
    {
        std::lock_guard lock(queue_mutex_);
        chat_pending = !pending_.empty();
    }
    if (chat_pending || !resident_empty() || !paused_.empty() || materializing_ || context_owner_) {
        // Only decision-owned pages are released. Native chat cache policy is unchanged.
        if (instance_.program->release_branch_cache()) {
            request_admission_check();
            scheduler_.capacity_released();
        }
        return true;
    }
    // End this decision-only cycle. A chat arriving after the queue check must start a
    // new cycle and release the overlay before admission can use general workspace.
    return false;
}

} // namespace ninfer::runtime
