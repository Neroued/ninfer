#pragma once

#include "ninfer/types.h"
#include "runtime/contract/recovery.h"
#include "runtime/engine/context_cache/prefix_index.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::runtime {

struct DpRouterConfig {
    std::uint32_t replica_count        = 1;
    std::uint32_t local_lanes          = 1;
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
};

struct ReplicaStatusOffer {
    std::uint64_t epoch                      = 1;
    std::uint32_t occupied_lanes             = 0;
    std::uint64_t min_rank_free_bytes        = 0;
    std::uint64_t min_rank_reclaimable_bytes = 0;
    std::uint64_t metadata_free_bytes        = 0;
    std::uint64_t active_context_tokens      = 0;
    std::uint64_t resource_revision          = 0;
    std::uint64_t lane_sequence              = 0;
    bool fresh_lanes_blocked                 = false;
    std::uint64_t first_unit_bytes           = 0;
    std::uint64_t first_unit_metadata_bytes  = 0;
};

struct OutstandingAssignment {
    std::uint64_t request_id        = 0;
    std::uint64_t publication_order = 0;
};

class DpRouter {
public:
    using Clock       = std::chrono::steady_clock;
    using KeyProvider = std::function<PrefixKey(std::uint32_t frontier)>;

    struct PendingItem {
        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        std::shared_ptr<void> payload;
        const std::atomic<bool>* cancelled     = nullptr;
        Clock::time_point deadline             = Clock::time_point::max();
        std::uint32_t max_reusable_frontier    = 0;
        std::uint64_t estimated_required_bytes = 0;
        KeyProvider key_at                     = {};
        std::uint64_t ownership_epoch          = 0;
        std::vector<std::optional<std::uint64_t>> rejected_revisions;
    };

    struct DispatchGrant {
        std::uint32_t replica_id       = 0;
        std::uint32_t matched_frontier = 0;
        PendingItem item;
    };

    struct ExpiredItem {
        enum class Reason : std::uint8_t { Cancelled, QueueTimeout, Unavailable };
        Reason reason = Reason::QueueTimeout;
        PendingItem item;
    };

    struct ScheduleOutcome {
        std::vector<DispatchGrant> dispatches;
        std::vector<ExpiredItem> rejected;
        std::vector<std::uint32_t> restore_grants;
    };

    struct FailureIsolationResult {
        bool newly_isolated        = false;
        std::uint32_t healthy_left = 0;
        ScheduleOutcome schedule;
    };

    struct SubmissionCutFence {
        std::uint64_t cut_request_id = 0;
        std::vector<std::uint32_t> healthy_participants;
    };

    explicit DpRouter(DpRouterConfig config);

    DpRouter(const DpRouter&)            = delete;
    DpRouter& operator=(const DpRouter&) = delete;

    // Step 1 of submission: waits past any active submission-cut fence (respecting deadline),
    // enforces GlobalOutstandingGate (C + max_pending_requests), and assigns monotonic IDs.
    [[nodiscard]] OutstandingAssignment acquire_outstanding(Clock::time_point pending_deadline);

    // Releases both engine-terminal tracking and outstanding permit when preparation fails
    // before enqueueing.
    void abort_preparation(std::uint64_t request_id) noexcept;

    // Enqueues a prepared request into the central pending queue and runs a scheduling pass.
    [[nodiscard]] ScheduleOutcome enqueue(PendingItem item, Clock::time_point now = Clock::now());

    // Returns an unadmitted request that exhausted its local placements at observed_revision
    // or is lending its permit to a local restore. Only capacity rejection suppresses a retry.
    // Releases the request's residency permit (handing off to waiting restores first) and
    // re-queues the request in original request_id order.
    [[nodiscard]] ScheduleOutcome
    return_unadmitted(std::uint32_t replica_id, std::uint64_t observed_revision, PendingItem item,
                      std::optional<ReplicaStatusOffer> offer = {},
                      bool handoff_to_local_restore = false, Clock::time_point now = Clock::now());

    // Called by replica_id when a dispatched request leaves in-transit state and becomes counted
    // in replica's occupied_lanes (or immediately settles before entering a lane).
    void acknowledge_dispatch(std::uint32_t replica_id, std::uint32_t occupied_lanes,
                              std::uint64_t lane_sequence) noexcept;

    // A successful first binding commits any reserved FIFO bypasses.
    void commit_admission(std::uint64_t request_id) noexcept;
    void cancel_admission(std::uint64_t request_id) noexcept;
    // Settles an in-transit grant if it could neither enter a replica nor be requeued.
    void drop_dispatch(std::uint32_t replica_id, std::uint64_t request_id) noexcept;

    // Updates replica's capacity/lane offer and triggers scheduling if new capacity appeared.
    [[nodiscard]] ScheduleOutcome update_replica_offer(std::uint32_t replica_id,
                                                       const ReplicaStatusOffer& offer,
                                                       Clock::time_point now = Clock::now());

    // Non-blocking restore residency permit request for a runnable paused_ request on replica_id.
    // Never blocks the replica thread. If global residency is full, registers waiting demand so
    // the next released residency permit goes strictly to waiting restores before central pending.
    [[nodiscard]] bool try_acquire_restore_permit(std::uint32_t replica_id) noexcept;

    // Cancels waiting demand or returns a grant that the replica has yet to claim.
    [[nodiscard]] ScheduleOutcome cancel_waiting_restore(std::uint32_t replica_id,
                                                         Clock::time_point now = Clock::now());

    // Called when a resident/materializing request vacates its replica slot (either finishing,
    // aborting, or completing pause). When handoff_to_local_restore is true, transfers the
    // residency permit directly in-place to a local paused_ request without exposing it to the
    // global pool. Otherwise returns the permit to the global pool (waking waiting restores first,
    // then central pending).
    [[nodiscard]] ScheduleOutcome
    release_or_handoff_residency(std::uint32_t replica_id, bool handoff_to_local_restore,
                                 std::optional<ReplicaStatusOffer> updated_offer = std::nullopt,
                                 Clock::time_point now                           = Clock::now());

    // Releases bulk residency permits owned by a failed replica during step 2 of isolation.
    [[nodiscard]] ScheduleOutcome
    release_failed_replica_residencies(std::uint32_t count, Clock::time_point now = Clock::now());

    // Marks request_id as engine-terminal (called when terminal snapshot/result is published).
    // Wakes any active submission-cut measurement fence once all id <= cut_request_id finish.
    void mark_request_terminal(std::uint64_t request_id) noexcept;

    // Releases the caller-held GlobalOutstandingGate permit (called when BOTH terminal result
    // is published AND GenerationHandle consumer is released).
    void release_outstanding_permit() noexcept;

    // Sweeps cancelled or timed-out central pending requests and runs a scheduling pass.
    [[nodiscard]] ScheduleOutcome sweep_and_schedule(Clock::time_point now = Clock::now());

    // Applies ordered PrefixIndex deltas from replica_id at epoch. Stale epochs or unhealthy
    // replicas are ignored.
    void apply_prefix_delta(std::uint32_t replica_id, std::uint64_t epoch,
                            const PrefixIndexDelta<PrefixKey>& delta);

    // Advances replica's epoch and atomically invalidates all prior prefix hints for replica_id.
    void reset_replica_epoch(std::uint32_t replica_id, std::uint64_t new_epoch);

    // Step 1 of two-step idempotent failure isolation: marks replica_id unroutable, invalidates
    // its prefix hints and slot offers and wakes submission/fence waiters. The caller takes
    // unavailable pending requests if no healthy replicas remain.
    [[nodiscard]] FailureIsolationResult
    mark_replica_unhealthy(std::uint32_t replica_id, Clock::time_point now = Clock::now());

    // Stops routing and rejects new submissions.
    void request_stop() noexcept;

    // Transfers the next pending request to the caller for settlement after stop or total failure.
    [[nodiscard]] std::optional<PendingItem> take_unavailable() noexcept;

    // Submission-cut measurement fence lifecycle:
    // 1. begin_submission_cut() records cut_request_id under the ID lock and blocks new submit()s.
    // 2. wait_pre_cut_terminal() waits until all requests with id <= cut_request_id reach terminal
    //    state (or a participating healthy replica fails, returning false).
    // 3. end_submission_cut() re-opens submissions and wakes blocked callers.
    [[nodiscard]] SubmissionCutFence begin_submission_cut();
    [[nodiscard]] bool
    wait_pre_cut_terminal(std::span<const std::uint32_t> required_healthy_replicas);
    // Bounded wait step returning {done, healthy_ok}. Allows caller to sweep pending deadlines
    // periodically while waiting for pre-cut requests to drain.
    [[nodiscard]] std::pair<bool, bool>
    wait_pre_cut_terminal_for(std::span<const std::uint32_t> required_healthy_replicas,
                              std::chrono::milliseconds step);
    void end_submission_cut() noexcept;

    // Inspection helpers for observability and unit tests.
    [[nodiscard]] std::uint32_t outstanding_count() const noexcept;
    [[nodiscard]] std::uint32_t resident_permit_count() const noexcept;
    [[nodiscard]] std::size_t pending_count() const noexcept;
    [[nodiscard]] std::uint32_t healthy_replica_count() const noexcept;
    [[nodiscard]] bool is_replica_healthy(std::uint32_t replica_id) const noexcept;
    [[nodiscard]] bool is_available() const noexcept;
    [[nodiscard]] std::size_t active_hint_key_count() const noexcept;
    [[nodiscard]] std::vector<PendingItem> snapshot_pending() const;
    [[nodiscard]] std::map<std::uint64_t, std::uint64_t> snapshot_request_epochs() const;

private:
    struct PrefixKeyHash {
        std::size_t operator()(const PrefixKey& k) const noexcept {
            std::size_t h = k.digests[0] ^ (k.digests[1] * 0x9e3779b97f4a7c15ULL);
            h ^= (static_cast<std::size_t>(k.frontier) << 32U) | k.domain;
            return h;
        }
    };

    enum class RestorePermit { None, Waiting, Granted };

    struct ReplicaEntry : ReplicaStatusOffer {
        bool healthy                        = true;
        std::uint32_t in_transit_dispatches = 0;
        RestorePermit restore               = RestorePermit::None;
        std::unordered_map<PrefixKey, std::uint32_t, PrefixKeyHash> owned_hints;

        [[nodiscard]] std::uint32_t effective_free_lanes(std::uint32_t local_lanes) const noexcept {
            if (!healthy || fresh_lanes_blocked) return 0;
            const std::uint32_t used = occupied_lanes + in_transit_dispatches;
            return used < local_lanes ? local_lanes - used : 0;
        }
    };

    struct RequestRouting {
        std::uint64_t epoch    = 0;
        std::uint32_t bypasses = 0;
        std::vector<std::uint64_t> skipped;
    };

    void rollback_bypasses_locked(RequestRouting& routing) noexcept;
    void prepare_schedule_locked(ScheduleOutcome& out, std::size_t extra_pending = 0,
                                 bool local_handoff = false);
    void update_offer_locked(ReplicaEntry& replica, const ReplicaStatusOffer& offer);
    void release_permit_locked(std::uint32_t replica_id, bool handoff, ScheduleOutcome& out);
    void clear_replica_hints_locked(ReplicaEntry& replica, std::uint32_t replica_id) noexcept;
    void schedule_locked(Clock::time_point now, ScheduleOutcome& out);

    DpRouterConfig config_;
    std::uint32_t max_outstanding_ = 1;

    mutable std::mutex mutex_;
    std::condition_variable submission_cv_;
    std::condition_variable fence_cv_;

    bool stopping_                 = false;
    bool failed_                   = false;
    std::uint32_t healthy_count_   = 0;
    std::uint64_t next_request_id_ = 1;
    std::uint64_t next_pub_order_  = 1;

    std::uint32_t outstanding_      = 0;
    std::uint32_t resident_permits_ = 0;
    std::uint32_t rr_cursor_        = 0;
    std::uint32_t restore_cursor_   = 0;

    bool fence_active_            = false;
    std::uint64_t cut_request_id_ = 0;
    std::map<std::uint64_t, RequestRouting> requests_;
    std::vector<std::uint32_t> eligible_;
    std::vector<std::uint32_t> matched_frontiers_;

    std::vector<ReplicaEntry> replicas_;
    std::deque<PendingItem> pending_;

    // PrefixKey already embeds {digest, frontier, domain}.
    // Maps PrefixKey -> (replica_id -> refcount).
    std::unordered_map<PrefixKey, std::unordered_map<std::uint32_t, std::uint32_t>, PrefixKeyHash>
        hints_;
    // Sorted active frontiers -> total replica-key pairs at that frontier.
    std::map<std::uint32_t, std::size_t> active_frontiers_;
};

} // namespace ninfer::runtime
