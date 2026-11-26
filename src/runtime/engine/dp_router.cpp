#include "runtime/engine/dp_router.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {

DpRouter::DpRouter(DpRouterConfig config)
    : config_(config), healthy_count_(config.replica_count), replicas_(config.replica_count),
      matched_frontiers_(config.replica_count) {
    eligible_.reserve(config.replica_count);
    if (config_.replica_count == 0 || config_.local_lanes == 0 || config_.max_concurrency == 0) {
        throw std::invalid_argument(
            "DpRouter requires positive replica_count, local_lanes, and max_concurrency");
    }
    if (config_.max_pending_requests >
        std::numeric_limits<std::uint32_t>::max() - config_.max_concurrency) {
        throw std::overflow_error("DpRouter max_outstanding overflows uint32");
    }
    max_outstanding_ = config_.max_concurrency + config_.max_pending_requests;
}

OutstandingAssignment DpRouter::acquire_outstanding(Clock::time_point pending_deadline) {
    std::unique_lock lock(mutex_);
    if (fence_active_) {
        const bool ready = submission_cv_.wait_until(lock, pending_deadline, [&] {
            return !fence_active_ || stopping_ || failed_ || healthy_count_ == 0;
        });
        if (!ready) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired at a measurement boundary");
        }
    }
    if (stopping_ || failed_ || healthy_count_ == 0) {
        throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
    }
    if (outstanding_ >= max_outstanding_) {
        throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
    }
    if (next_request_id_ == 0 || next_pub_order_ == 0) {
        throw std::overflow_error("request identity space exhausted");
    }
    const std::uint64_t request_id = next_request_id_;
    requests_.try_emplace(request_id);
    ++next_request_id_;
    const std::uint64_t pub_order = next_pub_order_++;
    ++outstanding_;
    return {request_id, pub_order};
}

void DpRouter::abort_preparation(std::uint64_t request_id) noexcept {
    bool wake_fence = false;
    {
        std::lock_guard lock(mutex_);
        if (outstanding_ > 0) { --outstanding_; }
        requests_.erase(request_id);
        wake_fence = fence_active_ && request_id <= cut_request_id_ &&
                     (requests_.empty() || requests_.begin()->first > cut_request_id_);
    }
    if (wake_fence) { fence_cv_.notify_all(); }
}

DpRouter::ScheduleOutcome DpRouter::enqueue(PendingItem item, Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        prepare_schedule_locked(out, 1);
        auto& routing        = requests_.at(item.request_id);
        item.ownership_epoch = routing.epoch;
        if (stopping_ || failed_ || healthy_count_ == 0) {
            out.rejected.push_back({ExpiredItem::Reason::Unavailable, std::move(item)});
            return out;
        }
        item.rejected_revisions.resize(config_.replica_count);
        const auto pos = std::upper_bound(
            pending_.begin(), pending_.end(), item.request_id,
            [](std::uint64_t id, const PendingItem& existing) { return id < existing.request_id; });
        pending_.insert(pos, std::move(item));
        schedule_locked(now, out);
    }
    return out;
}

DpRouter::ScheduleOutcome
DpRouter::return_unadmitted(std::uint32_t replica_id, std::uint64_t observed_revision,
                            PendingItem item, std::optional<ReplicaStatusOffer> offer,
                            bool handoff_to_local_restore, Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        auto& replica = replicas_.at(replica_id);
        auto& routing = requests_.at(item.request_id);
        prepare_schedule_locked(out, 1, handoff_to_local_restore);
        const bool in_transit  = !offer;
        const bool unavailable = stopping_ || failed_ || healthy_count_ == 0;
        item.ownership_epoch   = routing.epoch + 1;
        if (unavailable) {
            out.rejected.push_back({ExpiredItem::Reason::Unavailable, std::move(item)});
        } else {
            item.rejected_revisions.resize(config_.replica_count);
            if (!handoff_to_local_restore) {
                item.rejected_revisions[replica_id] = observed_revision;
            }
            const auto pos = std::upper_bound(pending_.begin(), pending_.end(), item.request_id,
                                              [](std::uint64_t id, const PendingItem& existing) {
                                                  return id < existing.request_id;
                                              });
            pending_.insert(pos, std::move(item));
        }
        ++routing.epoch;
        rollback_bypasses_locked(routing);
        if (in_transit) {
            if (replica.in_transit_dispatches > 0) --replica.in_transit_dispatches;
        } else {
            update_offer_locked(replica, *offer);
        }
        release_permit_locked(replica_id, handoff_to_local_restore, out);
        schedule_locked(now, out);
    }
    return out;
}

void DpRouter::acknowledge_dispatch(std::uint32_t replica_id, std::uint32_t occupied_lanes,
                                    std::uint64_t lane_sequence) noexcept {
    std::lock_guard lock(mutex_);
    auto& replica = replicas_[replica_id];
    if (!replica.healthy) return;
    --replica.in_transit_dispatches;
    if (lane_sequence >= replica.lane_sequence) {
        replica.lane_sequence  = lane_sequence;
        replica.occupied_lanes = occupied_lanes;
    }
}

void DpRouter::commit_admission(std::uint64_t request_id) noexcept {
    std::lock_guard lock(mutex_);
    requests_.at(request_id).skipped.clear();
}

void DpRouter::cancel_admission(std::uint64_t request_id) noexcept {
    std::lock_guard lock(mutex_);
    rollback_bypasses_locked(requests_.at(request_id));
}

void DpRouter::drop_dispatch(std::uint32_t replica_id, std::uint64_t request_id) noexcept {
    std::lock_guard lock(mutex_);
    auto& replica = replicas_[replica_id];
    if (replica.healthy) --replica.in_transit_dispatches;
    --resident_permits_;
    auto& routing = requests_.at(request_id);
    ++routing.epoch;
    rollback_bypasses_locked(routing);
}

void DpRouter::update_offer_locked(ReplicaEntry& replica, const ReplicaStatusOffer& offer) {
    if (!replica.healthy || offer.epoch != replica.epoch) return;
    const auto lanes                          = replica.occupied_lanes;
    const auto sequence                       = replica.lane_sequence;
    const auto revision                       = replica.resource_revision;
    static_cast<ReplicaStatusOffer&>(replica) = offer;
    replica.resource_revision                 = std::max(revision, offer.resource_revision);
    if (offer.lane_sequence < sequence) {
        replica.occupied_lanes = lanes;
        replica.lane_sequence  = sequence;
    }
}

DpRouter::ScheduleOutcome DpRouter::update_replica_offer(std::uint32_t replica_id,
                                                         const ReplicaStatusOffer& offer,
                                                         Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        auto& replica = replicas_.at(replica_id);
        if (!replica.healthy || offer.epoch < replica.epoch) return out;
        const auto old_free = replica.effective_free_lanes(config_.local_lanes);
        const bool new_resources =
            offer.epoch > replica.epoch || offer.resource_revision > replica.resource_revision;
        const bool new_lanes = offer.lane_sequence >= replica.lane_sequence &&
                               offer.occupied_lanes < replica.occupied_lanes;
        const bool unblocked = replica.fresh_lanes_blocked && !offer.fresh_lanes_blocked;
        const bool schedule  = new_resources || new_lanes || unblocked;
        if (schedule) prepare_schedule_locked(out);
        if (offer.epoch > replica.epoch) {
            clear_replica_hints_locked(replica, replica_id);
            replica.epoch = offer.epoch;
        }
        update_offer_locked(replica, offer);
        if (schedule &&
            (new_resources || replica.effective_free_lanes(config_.local_lanes) > old_free))
            schedule_locked(now, out);
    }
    return out;
}

bool DpRouter::try_acquire_restore_permit(std::uint32_t replica_id) noexcept {
    std::lock_guard lock(mutex_);
    auto& replica = replicas_[replica_id];
    if (stopping_ || !replica.healthy) return false;
    if (replica.restore == RestorePermit::Granted) {
        replica.restore = RestorePermit::None;
        return true;
    }
    if (resident_permits_ < config_.max_concurrency) {
        ++resident_permits_;
        replica.restore = RestorePermit::None;
        return true;
    }
    replica.restore = RestorePermit::Waiting;
    return false;
}

DpRouter::ScheduleOutcome DpRouter::cancel_waiting_restore(std::uint32_t replica_id,
                                                           Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        auto& replica = replicas_.at(replica_id);
        if (replica.restore != RestorePermit::Granted) {
            replica.restore = RestorePermit::None;
            return out;
        }
        prepare_schedule_locked(out);
        replica.restore = RestorePermit::None;
        --resident_permits_;
        schedule_locked(now, out);
    }
    return out;
}

void DpRouter::release_permit_locked(std::uint32_t replica_id, bool handoff, ScheduleOutcome& out) {
    if (resident_permits_ == 0)
        throw std::logic_error("DpRouter residency permit release underflow");
    auto& replica = replicas_[replica_id];
    if (handoff && !stopping_ && replica.healthy && replica.restore != RestorePermit::Granted) {
        replica.restore = RestorePermit::Granted;
        out.restore_grants.push_back(replica_id);
    } else {
        --resident_permits_;
    }
}

DpRouter::ScheduleOutcome
DpRouter::release_or_handoff_residency(std::uint32_t replica_id, bool handoff_to_local_restore,
                                       std::optional<ReplicaStatusOffer> updated_offer,
                                       Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        auto& replica = replicas_.at(replica_id);
        prepare_schedule_locked(out, 0, handoff_to_local_restore);
        if (updated_offer) update_offer_locked(replica, *updated_offer);
        release_permit_locked(replica_id, handoff_to_local_restore, out);
        schedule_locked(now, out);
    }
    return out;
}

DpRouter::ScheduleOutcome DpRouter::release_failed_replica_residencies(std::uint32_t count,
                                                                       Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        if (count > resident_permits_) {
            throw std::logic_error(
                "DpRouter failed replica residency release exceeds held permits");
        }
        resident_permits_ -= count;
        prepare_schedule_locked(out);
        schedule_locked(now, out);
    }
    return out;
}

void DpRouter::mark_request_terminal(std::uint64_t request_id) noexcept {
    bool wake_fence = false;
    {
        std::lock_guard lock(mutex_);
        if (const auto it = requests_.find(request_id); it != requests_.end()) {
            rollback_bypasses_locked(it->second);
            requests_.erase(it);
        }
        wake_fence = fence_active_ && request_id <= cut_request_id_ &&
                     (requests_.empty() || requests_.begin()->first > cut_request_id_);
    }
    if (wake_fence) { fence_cv_.notify_all(); }
}

void DpRouter::release_outstanding_permit() noexcept {
    std::lock_guard lock(mutex_);
    if (outstanding_ > 0) { --outstanding_; }
}

DpRouter::ScheduleOutcome DpRouter::sweep_and_schedule(Clock::time_point now) {
    ScheduleOutcome out;
    {
        std::lock_guard lock(mutex_);
        prepare_schedule_locked(out);
        schedule_locked(now, out);
    }
    return out;
}

void DpRouter::apply_prefix_delta(std::uint32_t replica_id, std::uint64_t epoch,
                                  const PrefixIndexDelta<PrefixKey>& delta) {
    std::lock_guard lock(mutex_);
    if (replica_id >= replicas_.size()) return;
    auto& replica = replicas_[replica_id];
    if (!replica.healthy || replica.epoch != epoch) return;

    switch (delta.kind) {
    case PrefixIndexDelta<PrefixKey>::Kind::Insert: {
        auto& ref = replica.owned_hints[delta.key];
        ++ref;
        if (ref == 1) {
            hints_[delta.key][replica_id] = 1;
            ++active_frontiers_[delta.frontier];
        } else {
            hints_[delta.key][replica_id] = ref;
        }
        break;
    }
    case PrefixIndexDelta<PrefixKey>::Kind::Erase: {
        const auto owned = replica.owned_hints.find(delta.key);
        if (owned == replica.owned_hints.end()) break;
        if (--owned->second == 0) {
            replica.owned_hints.erase(owned);
            const auto bucket = hints_.find(delta.key);
            if (bucket != hints_.end()) {
                bucket->second.erase(replica_id);
                if (bucket->second.empty()) { hints_.erase(bucket); }
            }
            const auto f = active_frontiers_.find(delta.frontier);
            if (f != active_frontiers_.end() && --f->second == 0) { active_frontiers_.erase(f); }
        } else {
            hints_[delta.key][replica_id] = owned->second;
        }
        break;
    }
    case PrefixIndexDelta<PrefixKey>::Kind::Clear:
        clear_replica_hints_locked(replica, replica_id);
        break;
    }
}

void DpRouter::reset_replica_epoch(std::uint32_t replica_id, std::uint64_t new_epoch) {
    std::lock_guard lock(mutex_);
    if (replica_id >= replicas_.size()) return;
    auto& replica = replicas_[replica_id];
    clear_replica_hints_locked(replica, replica_id);
    replica.epoch = new_epoch;
}

DpRouter::FailureIsolationResult DpRouter::mark_replica_unhealthy(std::uint32_t replica_id,
                                                                  Clock::time_point now) {
    FailureIsolationResult result;
    {
        std::lock_guard lock(mutex_);
        if (replica_id >= replicas_.size()) {
            throw std::out_of_range("DpRouter replica_id out of range");
        }
        auto& replica = replicas_[replica_id];
        if (!replica.healthy) {
            result.newly_isolated = false;
            result.healthy_left   = healthy_count_;
            return result;
        }
        replica.healthy = false;
        ++replica.epoch;
        clear_replica_hints_locked(replica, replica_id);
        replica.occupied_lanes        = 0;
        replica.in_transit_dispatches = 0;
        if (replica.restore == RestorePermit::Granted) --resident_permits_;
        replica.restore = RestorePermit::None;
        if (healthy_count_ > 0) { --healthy_count_; }
        failed_               = healthy_count_ == 0;
        result.newly_isolated = true;
        result.healthy_left   = healthy_count_;
        submission_cv_.notify_all();
        fence_cv_.notify_all();
        if (!failed_ && !stopping_) {
            prepare_schedule_locked(result.schedule);
            schedule_locked(now, result.schedule);
        }
    }
    return result;
}

void DpRouter::request_stop() noexcept {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        for (auto& replica : replicas_) {
            if (replica.restore == RestorePermit::Granted) --resident_permits_;
            replica.restore = RestorePermit::None;
        }
    }
    submission_cv_.notify_all();
    fence_cv_.notify_all();
}

std::optional<DpRouter::PendingItem> DpRouter::take_unavailable() noexcept {
    std::lock_guard lock(mutex_);
    if ((!stopping_ && !failed_) || pending_.empty()) return {};
    auto item = std::move(pending_.front());
    pending_.pop_front();
    return item;
}

DpRouter::SubmissionCutFence DpRouter::begin_submission_cut() {
    std::lock_guard lock(mutex_);
    if (stopping_ || failed_ || healthy_count_ == 0) {
        throw std::runtime_error("inference engine is unavailable");
    }
    if (fence_active_) { throw std::logic_error("a measurement boundary is already pending"); }
    std::vector<std::uint32_t> healthy;
    healthy.reserve(healthy_count_);
    for (std::uint32_t r = 0; r < replicas_.size(); ++r) {
        if (replicas_[r].healthy) { healthy.push_back(r); }
    }
    fence_active_   = true;
    cut_request_id_ = next_request_id_ - 1;
    return {cut_request_id_, std::move(healthy)};
}

bool DpRouter::wait_pre_cut_terminal(std::span<const std::uint32_t> required_healthy_replicas) {
    std::unique_lock lock(mutex_);
    fence_cv_.wait(lock, [&] {
        if (stopping_ || failed_) return true;
        for (const std::uint32_t r : required_healthy_replicas) {
            if (r >= replicas_.size() || !replicas_[r].healthy) return true;
        }
        return requests_.empty() || requests_.begin()->first > cut_request_id_;
    });
    if (stopping_ || failed_) return false;
    for (const std::uint32_t r : required_healthy_replicas) {
        if (r >= replicas_.size() || !replicas_[r].healthy) return false;
    }
    return true;
}

std::pair<bool, bool>
DpRouter::wait_pre_cut_terminal_for(std::span<const std::uint32_t> required_healthy_replicas,
                                    std::chrono::milliseconds step) {
    std::unique_lock lock(mutex_);
    const auto check_healthy = [&]() noexcept {
        if (stopping_ || failed_) return false;
        for (const std::uint32_t r : required_healthy_replicas) {
            if (r >= replicas_.size() || !replicas_[r].healthy) return false;
        }
        return true;
    };
    const auto all_done = [&]() noexcept {
        return requests_.empty() || requests_.begin()->first > cut_request_id_;
    };
    fence_cv_.wait_for(lock, step, [&] { return !check_healthy() || all_done(); });
    if (!check_healthy()) return {true, false};
    return {all_done(), true};
}

void DpRouter::end_submission_cut() noexcept {
    {
        std::lock_guard lock(mutex_);
        fence_active_ = false;
    }
    submission_cv_.notify_all();
}

std::uint32_t DpRouter::outstanding_count() const noexcept {
    std::lock_guard lock(mutex_);
    return outstanding_;
}

std::uint32_t DpRouter::resident_permit_count() const noexcept {
    std::lock_guard lock(mutex_);
    return resident_permits_;
}

std::size_t DpRouter::pending_count() const noexcept {
    std::lock_guard lock(mutex_);
    return pending_.size();
}

std::uint32_t DpRouter::healthy_replica_count() const noexcept {
    std::lock_guard lock(mutex_);
    return healthy_count_;
}

bool DpRouter::is_replica_healthy(std::uint32_t replica_id) const noexcept {
    std::lock_guard lock(mutex_);
    return replica_id < replicas_.size() && replicas_[replica_id].healthy;
}

bool DpRouter::is_available() const noexcept {
    std::lock_guard lock(mutex_);
    return !stopping_ && !failed_ && healthy_count_ > 0;
}

std::size_t DpRouter::active_hint_key_count() const noexcept {
    std::lock_guard lock(mutex_);
    return hints_.size();
}

std::vector<DpRouter::PendingItem> DpRouter::snapshot_pending() const {
    std::lock_guard lock(mutex_);
    return {pending_.begin(), pending_.end()};
}

void DpRouter::clear_replica_hints_locked(ReplicaEntry& replica,
                                          std::uint32_t replica_id) noexcept {
    for (const auto& [key, count] : replica.owned_hints) {
        (void)count;
        const auto bucket = hints_.find(key);
        if (bucket != hints_.end()) {
            bucket->second.erase(replica_id);
            if (bucket->second.empty()) { hints_.erase(bucket); }
        }
        const auto f = active_frontiers_.find(key.frontier);
        if (f != active_frontiers_.end() && --f->second == 0) { active_frontiers_.erase(f); }
    }
    replica.owned_hints.clear();
}

std::map<std::uint64_t, std::uint64_t> DpRouter::snapshot_request_epochs() const {
    std::lock_guard lock(mutex_);
    std::map<std::uint64_t, std::uint64_t> epochs;
    for (const auto& [id, request] : requests_) epochs.emplace(id, request.epoch);
    return epochs;
}

void DpRouter::rollback_bypasses_locked(RequestRouting& routing) noexcept {
    for (auto id : routing.skipped) {
        if (const auto older = requests_.find(id); older != requests_.end())
            --older->second.bypasses;
    }
    routing.skipped.clear();
}

void DpRouter::prepare_schedule_locked(ScheduleOutcome& out, std::size_t extra_pending,
                                       bool local_handoff) {
    const auto pending = pending_.size() + extra_pending;
    out.rejected.reserve(pending);
    out.dispatches.reserve(std::min<std::size_t>(pending, config_.max_concurrency));
    const auto waiting = std::count_if(replicas_.begin(), replicas_.end(), [](const auto& replica) {
        return replica.restore == RestorePermit::Waiting;
    });
    out.restore_grants.reserve(waiting + static_cast<std::size_t>(local_handoff));
}

void DpRouter::schedule_locked(Clock::time_point now, ScheduleOutcome& out) {
    if (stopping_ || failed_) return;
    // 1. Sweep cancelled or expired pending entries first so dead entries never block FIFO or
    // hold up bypass budgets.
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->cancelled != nullptr && it->cancelled->load(std::memory_order_acquire)) {
            out.rejected.push_back({ExpiredItem::Reason::Cancelled, std::move(*it)});
            it = pending_.erase(it);
            continue;
        }
        if (now >= it->deadline) {
            out.rejected.push_back({ExpiredItem::Reason::QueueTimeout, std::move(*it)});
            it = pending_.erase(it);
            continue;
        }
        ++it;
    }

    // 2. Strict restore priority: grant residency permits to runnable paused_ requests across
    // healthy replicas BEFORE dispatching any fresh request from central pending_.
    while (resident_permits_ < config_.max_concurrency) {
        std::optional<std::uint32_t> chosen;
        for (std::uint32_t offset = 0; offset < replicas_.size(); ++offset) {
            const std::uint32_t r = (restore_cursor_ + offset) % config_.replica_count;
            if (replicas_[r].healthy && replicas_[r].restore == RestorePermit::Waiting) {
                chosen = r;
                break;
            }
        }
        if (!chosen) break;
        auto& replica   = replicas_[*chosen];
        replica.restore = RestorePermit::Granted;
        ++resident_permits_;
        restore_cursor_ = (*chosen + 1) % config_.replica_count;
        out.restore_grants.push_back(*chosen);
    }

    // Check whether any healthy replica STILL has runnable paused_ requests waiting for a permit.
    // If so, global residency is saturated and central pending cannot advance.
    if (resident_permits_ >= config_.max_concurrency || pending_.empty() || healthy_count_ == 0) {
        return;
    }

    // 3. Dispatch from central pending_ using bounded FIFO bypass (max_concurrency).
    while (resident_permits_ < config_.max_concurrency && !pending_.empty()) {
        bool any_free_lane = false;
        for (const auto& rep : replicas_) {
            if (rep.effective_free_lanes(config_.local_lanes) > 0) {
                any_free_lane = true;
                break;
            }
        }
        if (!any_free_lane) break;

        bool dispatched_one = false;
        for (std::size_t idx = 0; idx < pending_.size(); ++idx) {
            auto& candidate = pending_[idx];
            auto& routing   = requests_.at(candidate.request_id);
            eligible_.clear();
            for (std::uint32_t r = 0; r < config_.replica_count; ++r) {
                const auto& rep = replicas_[r];
                if (rep.effective_free_lanes(config_.local_lanes) == 0) continue;
                if (r < candidate.rejected_revisions.size() &&
                    candidate.rejected_revisions[r].has_value() &&
                    rep.resource_revision <= *candidate.rejected_revisions[r]) {
                    continue;
                }
                eligible_.push_back(r);
            }
            if (eligible_.empty()) {
                if (routing.bypasses >= config_.max_concurrency) break;
                continue;
            }

            std::fill(matched_frontiers_.begin(), matched_frontiers_.end(), 0);
            if (config_.replica_count > 1 && candidate.max_reusable_frontier > 0 &&
                candidate.key_at && !active_frontiers_.empty()) {
                try {
                    for (const auto& [frontier, count] : active_frontiers_) {
                        (void)count;
                        if (frontier > candidate.max_reusable_frontier) break;
                        const auto bucket = hints_.find(candidate.key_at(frontier));
                        if (bucket == hints_.end()) continue;
                        for (const auto r : eligible_)
                            if (bucket->second.contains(r)) matched_frontiers_[r] = frontier;
                    }
                } catch (...) {
                    // Prefix hints are advisory. Capacity routing still works without a key.
                    std::fill(matched_frontiers_.begin(), matched_frontiers_.end(), 0);
                }
            }

            const auto capacity_tier = [&](std::uint32_t r) noexcept -> int {
                const auto& rep = replicas_[r];
                const auto required =
                    std::max(candidate.estimated_required_bytes, rep.first_unit_bytes);
                if (rep.metadata_free_bytes < rep.first_unit_metadata_bytes) return 0;
                if (rep.min_rank_free_bytes >= required) return 2;
                if (rep.min_rank_reclaimable_bytes >= required) return 1;
                return 0;
            };
            const auto rr_distance = [&](std::uint32_t r) noexcept -> std::uint32_t {
                return (r + config_.replica_count - rr_cursor_) % config_.replica_count;
            };

            std::uint32_t best = eligible_.front();
            for (std::size_t i = 1; i < eligible_.size(); ++i) {
                const std::uint32_t r = eligible_[i];
                const auto mf_r       = matched_frontiers_[r];
                const auto mf_best    = matched_frontiers_[best];
                if (mf_r != mf_best) {
                    if (mf_r > mf_best) best = r;
                    continue;
                }
                const int tier_r    = capacity_tier(r);
                const int tier_best = capacity_tier(best);
                if (tier_r != tier_best) {
                    if (tier_r > tier_best) best = r;
                    continue;
                }
                const auto load_r    = replicas_[r].active_context_tokens;
                const auto load_best = replicas_[best].active_context_tokens;
                if (load_r != load_best) {
                    if (load_r < load_best) best = r;
                    continue;
                }
                const auto rec_r    = replicas_[r].min_rank_reclaimable_bytes;
                const auto rec_best = replicas_[best].min_rank_reclaimable_bytes;
                if (rec_r != rec_best) {
                    if (rec_r > rec_best) best = r;
                    continue;
                }
                if (rr_distance(r) < rr_distance(best)) { best = r; }
            }

            try {
                routing.skipped.reserve(idx);
            } catch (const std::bad_alloc&) {
                // Earlier grants remain deliverable. This candidate still owns its queue entry.
                return;
            }
            for (std::size_t k = 0; k < idx; ++k) {
                const auto id = pending_[k].request_id;
                routing.skipped.push_back(id);
                ++requests_.at(id).bypasses;
            }
            candidate.ownership_epoch = ++routing.epoch;
            out.dispatches.push_back({best, matched_frontiers_[best], std::move(candidate)});
            pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(idx));
            ++resident_permits_;
            ++replicas_[best].in_transit_dispatches;
            rr_cursor_     = (best + 1) % config_.replica_count;
            dispatched_one = true;
            break;
        }
        if (!dispatched_one) break;
    }
}

} // namespace ninfer::runtime
