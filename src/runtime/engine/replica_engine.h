#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/context_capacity.h"
#include "runtime/engine/context_cache/placement_policy.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/dp_router.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/speculative_calibration.h"
#include "runtime/executor/contract.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

struct OwnedRequestObservation {
    RequestSnapshot snapshot;
    std::uint64_t ownership_epoch = 0;
};

template <class ModelContract>
class ReplicaCoordinatorHooks {
public:
    using Request = RequestRecord<ModelContract>;

    virtual ~ReplicaCoordinatorHooks() = default;

    [[nodiscard]] virtual bool prefix_routing_enabled() const noexcept = 0;
    [[nodiscard]] virtual bool bounce_enabled() const noexcept         = 0;

    virtual void on_prefix_delta(std::uint32_t replica_id, std::uint64_t epoch,
                                 const PrefixIndexDelta<PrefixKey>& delta) = 0;

    virtual void acknowledge_dispatch(std::uint32_t replica_id, std::uint32_t occupied_lanes,
                                      std::uint64_t lane_sequence) noexcept = 0;

    virtual void commit_admission(std::uint64_t request_id) noexcept      = 0;
    virtual void cancel_admission(std::uint64_t request_id) noexcept      = 0;
    [[nodiscard]] virtual std::uint64_t next_frontier_decision() noexcept = 0;

    virtual void on_replica_offer(std::uint32_t replica_id, const ReplicaStatusOffer& offer) = 0;

    virtual void bounce_unadmitted(std::uint32_t replica_id, const ReplicaStatusOffer& offer,
                                   bool handoff_to_local_restore,
                                   std::shared_ptr<Request> request) = 0;

    [[nodiscard]] virtual bool acquire_restore_permit(std::uint32_t replica_id) noexcept = 0;
    virtual void cancel_waiting_restore(std::uint32_t replica_id)                        = 0;

    virtual void release_or_handoff_residency(std::uint32_t replica_id,
                                              bool handoff_to_local_restore,
                                              const ReplicaStatusOffer& updated_offer) = 0;

    virtual void on_request_terminal(std::uint64_t request_id, FinishReason reason,
                                     bool failed) noexcept = 0;
    virtual void release_outstanding_permit() noexcept     = 0;

    virtual void on_replica_failing(std::uint32_t replica_id) noexcept               = 0;
    virtual void on_replica_failed_release_residencies(std::uint32_t count) noexcept = 0;
};

template <class Instance>
class ReplicaEngine {

public:
    static constexpr EnginePurpose purpose = EnginePurpose::Generation;

    using ModelContract      = typename Instance::ModelContract;
    using ContextAssessment  = typename ModelContract::Session::ContextAssessment;
    using ExecutionUnit      = typename ModelContract::ExecutionUnit;
    using UnitKind           = typename ModelContract::ExecutionUnitKind;
    using Checkpoint         = typename ModelContract::CheckpointHandle;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using PrefillMembership  = typename Scheduling::PrefillMembership;
    using ResourceManagement = ResourceManager<ModelContract>;
    using Hooks              = ReplicaCoordinatorHooks<ModelContract>;
    using Clock              = std::chrono::steady_clock;

    ReplicaEngine(Instance& instance, const EngineOptions& options,
                  ContextMachineCostModel context_cost, Hooks& hooks, std::uint32_t replica_id,
                  std::uint32_t local_lanes, std::shared_future<void> start)
        : instance_(instance), hooks_(hooks), replica_id_(replica_id),
          epoch_(instance.executor->epoch().value ? instance.executor->epoch().value : 1),
          max_context_(options.max_context), max_concurrency_(local_lanes),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          speculative_positions_(options.speculative.backend == SpeculativeBackend::None
                                     ? 0
                                     : options.speculative.draft_tokens),
          resources_(options.context_cache.enabled, std::move(context_cost),
                     options.context_cache.max_cached_checkpoints_per_replica),
          slots_(max_concurrency_), capture_decisions_(max_concurrency_),
          execution_decisions_(max_concurrency_), control_executed_(max_concurrency_),
          reserved_units_(max_concurrency_), current_decode_requests_(max_concurrency_),
          round_membership_(max_concurrency_), control_membership_(max_concurrency_),
          prefill_membership_(max_concurrency_), prefill_excluded_(max_concurrency_),
          cancellation_snapshot_(max_concurrency_), row_flags_(max_concurrency_),
          commit_decisions_(max_concurrency_), commit_preview_(max_concurrency_) {
        if (max_concurrency_ == 0) {
            throw std::invalid_argument("ReplicaEngine requires at least one local lane");
        }
        if (!instance_.context_capacities.empty()) {
            initial_token_capacity_estimate_ =
                estimate_context_token_capacity(instance_.context_capacities);
        }
        if (hooks_.prefix_routing_enabled()) {
            resources_.set_prefix_observer([this](const auto& delta) {
                if constexpr (std::is_same_v<std::decay_t<decltype(delta.key)>, PrefixKey>) {
                    hooks_.on_prefix_delta(replica_id_, epoch_, delta);
                }
            });
        }
        current_decode_lanes_.reserve(max_concurrency_);
        reservation_candidates_.reserve(max_concurrency_);
        cumulative_stats_.context_transfers.kv.resize(instance_.session->kv_pools().size());
        for (const auto& pool : instance_.session->kv_pools()) {
            cumulative_stats_.kv_pools.push_back({pool.id.index, std::string(pool.label), 0});
        }
        cumulative_stats_.speculative_attempted_per_position.resize(speculative_positions_);
        cumulative_stats_.speculative_accepted_per_position.resize(speculative_positions_);
        for (const auto& placement : instance_.session->placement_descriptions())
            cumulative_stats_.placements.push_back({.id = placement.id, .replica_id = replica_id_});
        stats_snapshot_             = cumulative_stats_;
        published_snapshot_.runtime = cumulative_stats_;
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        paused_.reserve(max_outstanding_);
        request_observations_.reserve(max_outstanding_ + 1);
        published_requests_.reserve(max_outstanding_);
        published_snapshot_.decode_by_batch.resize(max_concurrency_);
        const auto frames = instance_.executor->capacity();
        flights_.resize(frames.compute_frames);
        waiting_decode_lanes_.reserve(max_concurrency_);
        execution_observations_.reserve(frames.compute_frames + frames.context_frames);
        published_snapshot_.in_flight.reserve(frames.compute_frames + frames.context_frames);
        std::promise<void> bound;
        bound_ = bound.get_future();
        std::promise<std::optional<SpeculativeStartupSummary>> startup;
        started_ = startup.get_future();
        worker_  = std::thread([this, spec = options.speculative,
                               observer = options.startup_observer, bound = std::move(bound),
                               start = std::move(start), startup = std::move(startup)]() mutable {
            try {
                instance_.executor->bind_to_current_thread();
                instance_.executor->set_peer_wakeup([this] { wake(); });
                bound.set_value();
            } catch (...) {
                bound.set_exception(std::current_exception());
                startup.set_exception(std::current_exception());
                return;
            }
            try {
                start.get();
                auto summary = prepare_speculative_startup(spec, observer);
                instance_.executor->finish_startup(summary && summary->calibrated);
                if (hooks_.prefix_routing_enabled()) quote_first_unit();
                if (replica_id_ == 0) {
                    StartupPhaseScope phase(observer, StartupPhase::EngineFinalize);
                    publish_runtime_stats();
                    phase.complete();
                } else {
                    publish_runtime_stats();
                }
                observer = {};
                startup.set_value(std::move(summary));
            } catch (...) {
                observer         = {};
                const auto error = std::current_exception();
                instance_.executor->fail_from_exception(error);
                instance_.session->fail_all_cleanup();
                startup.set_exception(error);
                return;
            }
            worker_loop();
        });
    }

    void wait_bound() { bound_.get(); }

    [[nodiscard]] auto wait_started() { return started_.get(); }

    [[nodiscard]] std::uint32_t replica_id() const noexcept { return replica_id_; }

    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

    [[nodiscard]] std::uint32_t local_lanes() const noexcept { return max_concurrency_; }

    ~ReplicaEngine() noexcept {
        request_stop();
        join();
    }

    void join() noexcept {
        if (worker_.joinable()) { worker_.join(); }
        instance_.executor->set_peer_wakeup({});
    }

    // May be called while consumers are waiting; only the worker releases model resources.
    void request_stop() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
    }

    void wake() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            wake_requested_ = true;
        }
        queue_cv_.notify_one();
    }

    void notify_restore_grant() noexcept {
        request_admission_check();
        wake();
    }

    void notify_cancellation() noexcept {
        pending_scan_requested_.store(true, std::memory_order_release);
        request_admission_check();
        wake();
    }

    // Accepts a request dispatched by DpRouter with an attached global residency permit.
    // Returns false without modifying request if this replica is stopping or failed.
    [[nodiscard]] bool accept_dispatch(DpRouter::PendingItem& item) {
        auto request = std::static_pointer_cast<Request>(item.payload);
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) { return false; }
            const auto pos = std::upper_bound(
                pending_.begin(), pending_.end(), request->id,
                [](std::uint64_t id, const std::shared_ptr<Request>& r) { return id < r->id; });
            pending_.insert(pos, request);
            request->replica_id             = replica_id_;
            request->ownership_epoch        = item.ownership_epoch;
            request->rejected_revisions     = std::move(item.rejected_revisions);
            request->holds_residency_permit = true;
            request->counts_dispatch_lane   = true;
            ++occupied_lanes_;
            ++submitted_requests_;
            pending_scan_requested_.store(true, std::memory_order_release);
            hooks_.acknowledge_dispatch(replica_id_, occupied_lanes_, ++lane_sequence_);
            request_admission_check();
        }
        queue_cv_.notify_one();
        return true;
    }

    [[nodiscard]] ReplicaSnapshot
    replica_snapshot(std::vector<OwnedRequestObservation>* out_requests = nullptr,
                     std::uint64_t* frontier_decision                   = nullptr) const {
        std::lock_guard lock(stats_mutex_);
        ReplicaSnapshot out;
        out.replica_id = replica_id_;
        out.healthy    = published_snapshot_.status != EngineStatus::Failed;
        out.status     = published_snapshot_.status;
        out.available_context_tokens_estimate =
            published_snapshot_.available_context_tokens_estimate;
        out.progress_owner  = published_snapshot_.progress_owner;
        out.runtime         = published_snapshot_.runtime;
        out.executions      = published_snapshot_.executions;
        out.decode_by_batch = published_snapshot_.decode_by_batch;
        out.in_flight       = published_snapshot_.in_flight;
        out.ranks           = published_snapshot_.ranks;
        if (frontier_decision) *frontier_decision = published_frontier_decision_;
        if (out_requests) { *out_requests = published_requests_; }
        return out;
    }

    [[nodiscard]] std::future<void> start_measurement(MeasurementCommand command) {
        auto request     = std::make_unique<MeasurementRequest>();
        request->command = command;
        auto result      = request->completion.get_future();
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) throw std::runtime_error("inference engine is unavailable");
            if (measurement_) throw std::logic_error("a measurement boundary is already pending");
            measurement_    = std::move(request);
            wake_requested_ = true;
        }
        queue_cv_.notify_all();
        return result;
    }

    ReplicaEngine(const ReplicaEngine&)            = delete;
    ReplicaEngine& operator=(const ReplicaEngine&) = delete;

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

private:
    struct MeasurementRequest {
        MeasurementCommand command{};
        std::promise<void> completion;
    };

    bool process_measurement();

    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    enum class CycleStep : std::uint8_t { Boundary, Control, Decode, Prefill, Complete };

    struct UnitPermit {
        ExecutionUnit unit;
        ResidencyId residency;
    };

    struct ComputeFlight {
        ExecutionTicket ticket;
        UnitKind kind = UnitKind::Decode;
        std::vector<std::shared_ptr<Request>> requests;
        std::vector<SequenceHandle> sequences;
        std::vector<ResidencyId> residencies;
        std::vector<std::uint32_t> lanes;
        std::vector<std::uint8_t> cancelled;
        std::vector<std::uint8_t> progressed;
        std::optional<std::uint64_t> retired_ns;
        bool semantic_done = false;
        std::vector<TokenId> control_tokens;
        std::uint32_t control_stride = 0;
        std::vector<std::size_t> generated_sizes;
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept;

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
        bool decode        = false;
    };

    struct ActiveExposureSet {
        HostWorkClass work_class;
        std::vector<ActiveExposure> entries;
    };

    using ExposureSnapshot = std::shared_ptr<const ActiveExposureSet>;

    struct PendingCompletion {
        std::shared_ptr<Request> request;
        GenerationResult result;
        std::exception_ptr error;
        HostExposureClock exposure_end;
        bool ready = false;
    };

    struct CompletionExposure {
        ExposureSnapshot exposed;
        HostExposureClock first;
    };

    [[nodiscard]] HostExposureClock exposure_clock() const noexcept;
    [[nodiscard]] RequestHostTiming request_host_timing(const Request&) const noexcept;
    bool completion_prepared(const std::shared_ptr<Request>&) const;
    PendingCompletion& retain_completion(const std::shared_ptr<Request>&);
    void finish_completion_exposure() noexcept;
    void publish_completions();

    class CompletionScope {
    public:
        explicit CompletionScope(ReplicaEngine& owner)
            : owner_(owner), owns_(owner.completion_scope_depth_++ == 0) {}

        ~CompletionScope() {
            if (!active_) return;
            --owner_.completion_scope_depth_;
            if (owns_) owner_.finish_completion_exposure();
        }

        CompletionScope(const CompletionScope&)            = delete;
        CompletionScope& operator=(const CompletionScope&) = delete;

        void publish() {
            if (!owns_) return;
            owner_.finish_completion_exposure();
            owner_.publish_completions();
            --owner_.completion_scope_depth_;
            active_ = false;
        }
    private:
        ReplicaEngine& owner_;
        bool owns_, active_ = true;
    };

    struct CommitPreview {
        FinishReason reason{};
        ContinuationAction continuation{};
        std::size_t generated_size = 0;
        bool cancelled             = false;
        bool failed                = false;
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ExposureSnapshot exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept;

    [[nodiscard]] ExposureSnapshot active_exposure_set() const;

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const;

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {});

    void restore_host_work_class(const ActiveExposureSet&) noexcept;
    void set_resident(std::uint32_t lane, std::shared_ptr<Request>) noexcept;
    void set_placement_load(Request&, std::optional<PlacementId>, std::uint64_t tokens);

    void add_class_host_time(HostWorkClass, std::uint64_t host_ns,
                             std::uint64_t scheduler_yield_ns) noexcept;

    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept;

    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept;

    void record_coordinator_call(CoordinatorCall call, std::uint64_t elapsed,
                                 const ActiveExposureSet& exposed) noexcept;

    void finish_execution_call(const HostPhaseMeasurement& measurement,
                               CoordinatorCall call) noexcept;

    void record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::*invocation_member,
                       Clock::time_point started) noexcept;

    class EnginePhaseScope {
    public:
        EnginePhaseScope(ReplicaEngine& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void checkpoint() {
            if (!active_) { return; }
            owner_.finish_engine_phase(measurement_, phase_);
            active_      = false;
            measurement_ = owner_.begin_host_phase();
            active_      = true;
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        ReplicaEngine& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ExecutionCallScope {
    public:
        explicit ExecutionCallScope(ReplicaEngine& owner,
                                    CoordinatorCall call = CoordinatorCall::Session)
            : owner_(owner), call_(call), measurement_(owner.begin_host_phase()) {}

        ~ExecutionCallScope() noexcept { finish(); }

        ExecutionCallScope(const ExecutionCallScope&)            = delete;
        ExecutionCallScope& operator=(const ExecutionCallScope&) = delete;

        void finish() noexcept {
            if (!active_) { return; }
            owner_.finish_execution_call(measurement_, call_);
            active_ = false;
        }

    private:
        ReplicaEngine& owner_;
        CoordinatorCall call_;
        HostPhaseMeasurement measurement_;
        bool active_ = true;
    };

    void release_dispatch_lane(Request& request) {
        std::lock_guard lock(queue_mutex_);
        if (!request.counts_dispatch_lane) return;
        request.counts_dispatch_lane = false;
        --occupied_lanes_;
        ++lane_sequence_;
    }

    void quote_first_unit() {
        auto& session = *instance_.session;
        ResolvedExecutionOptions options;
        options.requested_output_tokens = 1;
        const auto base =
            session.plan_request(instance_.frontend.prepare_tokens({0}, false), options);
        // A one-token root prefill is a portable lower bound. Native binding still checks the
        // actual prompt, selected placement, reuse and record classes.
        std::optional<std::pair<std::uint64_t, std::uint64_t>> best;
        for (const auto placement : session.placements(base)) {
            const auto source = session.inspect_source(base, {}, false, {}, {}, placement);
            if (!source) continue;
            const auto binding =
                session.assess_binding(base, LaneId{0}, *source, nullptr, UnitKind::Prefill, 0);
            if (!binding.source_valid || !binding.capacity_possible) continue;
            std::uint64_t bytes = 0;
            for (const auto& domain : binding.assessment.placement.domains) {
                if (domain.domain.memory == decltype(domain.domain.memory)::Device)
                    bytes = std::max<std::uint64_t>(bytes, domain.minimum_bytes);
            }
            const auto quote = std::pair{
                bytes, static_cast<std::uint64_t>(binding.assessment.metadata_required_bytes)};
            if (!best || quote < *best) best = quote;
        }
        if (best) {
            first_unit_bytes_          = best->first;
            first_unit_metadata_bytes_ = best->second;
        }
    }

    template <class PhysicalUsage>
    [[nodiscard]] ReplicaStatusOffer compute_status_offer(const PhysicalUsage& physical) {
        ReplicaStatusOffer offer;
        offer.epoch                     = epoch_;
        offer.first_unit_bytes          = first_unit_bytes_;
        offer.first_unit_metadata_bytes = first_unit_metadata_bytes_;
        for (const auto& req : slots_) {
            if (req) offer.active_context_tokens += req->placement_tokens;
        }
        if (materializing_)
            offer.active_context_tokens += materializing_->request->placement_tokens;
        {
            std::lock_guard lock(queue_mutex_);
            offer.occupied_lanes      = occupied_lanes_;
            offer.lane_sequence       = lane_sequence_;
            offer.fresh_lanes_blocked = !paused_.empty();
        }

        bool capacity_increased       = false;
        bool any_device_pool          = false;
        std::uint64_t min_free        = UINT64_MAX;
        std::uint64_t min_reclaimable = UINT64_MAX;
        previous_pool_capacity_.resize(physical.pools.size());
        for (std::size_t i = 0; i < physical.pools.size(); ++i) {
            const auto& pool = physical.pools[i];
            const auto free =
                pool.assigned_free_bytes + pool.unassigned_free_bytes + pool.tail_bytes;
            const auto reclaimable = free + pool.cached_bytes;
            const auto previous    = previous_pool_capacity_[i];
            capacity_increased |= free > previous.first || reclaimable > previous.second;
            previous_pool_capacity_[i] = {free, reclaimable};
            if (pool.domain != ContextMemoryDomain::Device || pool.capacity_bytes == 0) continue;
            any_device_pool = true;
            min_free        = std::min(min_free, free);
            min_reclaimable = std::min(min_reclaimable, reclaimable);
        }
        offer.min_rank_free_bytes        = any_device_pool ? min_free : 0;
        offer.min_rank_reclaimable_bytes = any_device_pool ? min_reclaimable : 0;
        const auto metadata_used =
            std::min(physical.metadata.budget_bytes,
                     physical.metadata.allocated_bytes + physical.metadata.reserved_bytes);
        offer.metadata_free_bytes = physical.metadata.budget_bytes - metadata_used;
        capacity_increased |= offer.metadata_free_bytes > previous_metadata_free_;
        previous_metadata_free_     = offer.metadata_free_bytes;
        const auto catalog_revision = instance_.session->checkpoint_catalog_revision();
        const auto lease_revision   = instance_.session->checkpoint_lease_revision();
        const bool eviction_changed = catalog_revision != offered_catalog_revision_ ||
                                      lease_revision != offered_lease_revision_;
        offered_catalog_revision_ = catalog_revision;
        offered_lease_revision_   = lease_revision;
        if (capacity_increased || eviction_changed) {
            ++resource_revision_;
            request_admission_check();
        }
        offer.resource_revision = resource_revision_;
        return offer;
    }

    [[nodiscard]] std::shared_ptr<Request>
    find_unpermitted_paused_restore_target(const Request* pausing_victim = nullptr) const noexcept {
        const auto owner = scheduler_.progress_owner();
        if (!owner) return nullptr;
        const auto it = std::find_if(paused_.begin(), paused_.end(), [&](const auto& r) {
            return r.get() != pausing_victim && r->id == *owner &&
                   !r->cancelled.load(std::memory_order_acquire) && !r->holds_residency_permit;
        });
        return it != paused_.end() ? *it : nullptr;
    }

    void release_or_handoff_request_residency(const std::shared_ptr<Request>& request,
                                              const Request* pausing_victim = nullptr) {
        if (!request || !request->holds_residency_permit) return;
        release_dispatch_lane(*request);
        const auto target = find_unpermitted_paused_restore_target(pausing_victim);
        synchronize_restore_target(target);
        const auto offer = compute_status_offer(instance_.session->physical_usage());
        hooks_.release_or_handoff_residency(replica_id_, target != nullptr, offer);
        request->holds_residency_permit = false;
        if (target) {
            waiting_restore_target_ = target->id;
            (void)claim_restore_permit(*target);
        }
    }

    void synchronize_restore_target(const std::shared_ptr<Request>& target) {
        if (waiting_restore_target_ && (!target || target->id != *waiting_restore_target_)) {
            hooks_.cancel_waiting_restore(replica_id_);
            waiting_restore_target_.reset();
        }
    }

    bool claim_restore_permit(Request& target) noexcept {
        if (hooks_.acquire_restore_permit(replica_id_)) {
            target.holds_residency_permit = true;
            waiting_restore_target_.reset();
            request_admission_check();
            return true;
        }
        waiting_restore_target_ = target.id;
        return false;
    }

    void bounce_pending_request(const std::shared_ptr<Request>& request,
                                const std::shared_ptr<Request>& restore_target = {});

    void publish_resource_offer() {
        hooks_.on_replica_offer(replica_id_,
                                compute_status_offer(instance_.session->physical_usage()));
    }

    void publish_runtime_stats();
    [[nodiscard]] RequestSnapshot observe_request(const Request&) const;
    void record_completion(const std::shared_ptr<Request>&, FinishReason, bool failed);

    // Coalesces admission-visible queue/resource changes. Ordinary prefill/decode progress does not
    // re-arm a temporarily blocked inspection.
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
        schedule_revision_.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        BeginSummary summary;
        bool resumed = false;
        typename ResourceManagement::SourceChoice source;
    };

    struct AdmissionDecision {
        std::shared_ptr<Request> request;
        bool restoring = false;
        std::vector<typename ResourceManagement::SourceChoice> sources;
        std::size_t current = 0;
        std::optional<typename ResourceManagement::ReclaimCursor> reclaim;
    };

    struct ExecutionDecision {
        ExecutionUnit unit;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    struct CaptureDecision {
        SequenceHandle sequence;
        std::uint32_t frontier = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
        std::optional<typename ResourceManagement::ReclaimActionQuote> pending;
    };

    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request, std::uint32_t accepted_tokens);

    void record_first_output(const std::shared_ptr<Request>& request);

    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt,
                       EnginePhaseScope* phase                           = nullptr);

    // Model-format failures settle only this already-committed row. Native abort still owns the
    // sequence cleanup; decoder errors must not escape to the Engine-wide invariant handler.
    bool fail_output(const std::shared_ptr<Request>& request, std::uint32_t lane);

    void publish_prompt_progress(const std::shared_ptr<Request>& request);

    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin);


    bool prepare_completion(const std::shared_ptr<Request>& request);

    void release_request(const std::shared_ptr<Request>& request) noexcept;

    // The worker publishes the matching scheduling snapshot before making the response visible.
    void publish_completion(const std::shared_ptr<Request>& request, GenerationResult result,
                            std::exception_ptr error = {});

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error);

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason);

    void complete_cancelled(const std::shared_ptr<Request>& request);

    [[nodiscard]] std::span<const std::uint64_t> snapshot_cancellations() noexcept;

    bool settle_terminal_requests(HostPhaseMeasurement& boundary);

    void cancel_active_requests(std::span<const std::uint64_t> cancelled_at_boundary,
                                HostPhaseMeasurement& boundary);

    [[nodiscard]] bool expire_pending_requests();

    void commit_pending(PendingBatch&& pending, ComputeFlight& flight);

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next);

    bool prepare_semantic_capture(const std::shared_ptr<Request>& request,
                                  bool allow_reclaim = true);

    void resolve_prefill_progress(typename ModelContract::PrefillProgress&& progress,
                                  ComputeFlight& flight);

    void ensure_base_plan(const std::shared_ptr<Request>& request);

    [[nodiscard]] std::optional<std::uint32_t> free_lane() const noexcept;

    [[nodiscard]] bool resident_empty() const noexcept;

    void record_context_work(const typename ModelContract::ContextProgress& progress,
                             const std::shared_ptr<Request>& owner);

    void resolve_context_progress(typename ModelContract::ContextProgress&& progress,
                                  HostPhaseMeasurement& boundary);

    ReclaimProgress reclaim(const ContextAssessment& assessment, bool allow_pause,
                            std::span<const Checkpoint> excluded               = {},
                            typename ResourceManagement::ReclaimCursor* cursor = nullptr);

    bool pause_reclaim_victim(const ContextAssessment&,
                              typename ResourceManagement::ReclaimCursor* = nullptr);

    enum class PauseGoal { ReleaseResidency, ReleaseCapacity, RelocateOnly };
    bool pause_resident(std::uint32_t lane, PauseGoal goal = PauseGoal::ReleaseResidency,
                        const ContextAssessment* pressure           = nullptr,
                        typename ResourceManagement::ReclaimCursor* = nullptr);

    void order_binding_sources(const std::shared_ptr<Request>&, LaneId,
                               std::vector<typename ResourceManagement::SourceChoice>&);

    void finish_source_wait(const std::shared_ptr<Request>& request,
                            Clock::time_point now) noexcept;

    void retain_admission_source(const std::shared_ptr<Request>& request,
                                 const typename ResourceManagement::SourceChoice& choice,
                                 bool restoring);

    bool try_admit_one(bool restoring);

    void observe_scheduling(const std::shared_ptr<Request>& request,
                            GenerationSchedulingTransition transition, Clock::time_point now = {});

    void update_recovery(const std::shared_ptr<Request>& request, bool released = false);

    enum class ResidentReservation { Ready, Unavailable, Pending };

    [[nodiscard]] ExecutionUnit execution_unit(const Request& request) const;

    [[nodiscard]] ResidentReservation reserve_resident_unit(std::uint32_t lane, ExecutionUnit unit,
                                                            bool allow_reclaim = true);
    void reserve_resident_units();
    void build_prefill_membership(const PrefillBudgetPolicy& policy,
                                  std::optional<SequenceHandle> primary    = {},
                                  std::optional<std::uint32_t> primary_cap = {});
    void release_unused_prefill_units(std::optional<std::uint32_t> primary = {});
    [[nodiscard]] bool resident_runnable(std::uint32_t lane) const;
    [[nodiscard]] bool schedule_prefill();

    [[nodiscard]] bool unit_runnable(std::uint32_t lane) const;

    [[nodiscard]] ComputeFlight freeze_members(std::span<const std::uint32_t> lanes,
                                               UnitKind kind) const;

    [[nodiscard]] bool start_prefill(const PrefillMembership& membership);

    [[nodiscard]] bool start_decode(const RoundMembership& membership);

    void discard_control_preview(const ComputeFlight& flight) noexcept;

    [[nodiscard]] bool start_control(const ControlMembership& membership);

    void resolve_control(ComputeFlight& flight);

    void supply_decision(ExecutionTicket ticket, const DecisionRequest& decision);

    void record_decision_upload(const DecisionConsumed& consumed);

    [[nodiscard]] bool poll_execution(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool execution_outstanding() const noexcept;

    [[nodiscard]] bool compute_full() const noexcept;
    void install_flight(ComputeFlight);
    ComputeFlight& flight_for(ExecutionTicket);
    void release_completed(ComputeFlight&);
    void set_wait_work_class();
    void set_flight_work_class(const ComputeFlight&);

    void record_service(ComputeFlight& flight) noexcept;
    void log_worker_failure(std::exception_ptr error) const noexcept;
    void fail_all(std::exception_ptr error) noexcept;

    void begin_cycle(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool advance_cycle();

    void worker_loop() noexcept;

    std::optional<SpeculativeStartupSummary>
    prepare_speculative_startup(const SpeculativeOptions& options,
                                const StartupObserver& observer) {
        if (options.backend == SpeculativeBackend::None) return {};
        SpeculativeStartupSummary summary{.backend       = options.backend,
                                          .length_policy = options.length_policy,
                                          .draft_window  = options.draft_tokens,
                                          .target_widths =
                                              instance_.session->decode_target_widths()};
        if (options.length_policy == SpeculativeLengthPolicy::Adaptive) {
            if (const auto calibration = calibrate_speculative(instance_, observer)) {
                summary.calibrated           = true;
                summary.calibration_anchors  = calibration->anchors;
                summary.calibration_contexts = calibration->contexts;
                summary.calibration_seconds  = calibration->seconds;
            }
        }
        return summary;
    }

    Instance& instance_;
    Hooks& hooks_;
    const std::uint32_t replica_id_                = 0;
    const std::uint64_t epoch_                     = 1;
    std::uint64_t initial_token_capacity_estimate_ = 0;
    std::uint64_t resource_revision_               = 1;
    std::optional<std::uint64_t> waiting_restore_target_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> previous_pool_capacity_;
    std::uint64_t previous_metadata_free_    = 0;
    std::uint64_t first_unit_bytes_          = 0;
    std::uint64_t first_unit_metadata_bytes_ = 0;
    std::uint32_t occupied_lanes_            = 0; // queue_mutex_
    std::uint64_t lane_sequence_             = 0; // queue_mutex_
    std::uint64_t offered_catalog_revision_  = 0;
    std::uint64_t offered_lease_revision_    = 0;

    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::uint32_t speculative_positions_;
    std::future<void> bound_;
    std::future<std::optional<SpeculativeStartupSummary>> started_;
    ResourceManagement resources_;
    PlacementId next_placement_tie_;

    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::unique_ptr<MeasurementRequest> measurement_;
    bool wake_requested_              = false;
    std::uint64_t submitted_requests_ = 0;
    std::vector<std::shared_ptr<Request>> slots_;
    std::optional<MaterializingRequest> materializing_;
    std::optional<AdmissionDecision> admission_decision_;
    std::map<std::uint32_t, PlacementLoad> placement_loads_;
    std::uint64_t admission_generation_ = 1;
    std::vector<std::optional<CaptureDecision>> capture_decisions_;
    std::vector<std::optional<ExecutionDecision>> execution_decisions_;
    std::shared_ptr<Request> context_owner_;
    std::vector<std::shared_ptr<Request>> paused_;
    Scheduling scheduler_;
    std::vector<std::optional<ComputeFlight>> flights_;
    std::vector<std::uint32_t> waiting_decode_lanes_;
    CycleStep cycle_step_                 = CycleStep::Boundary;
    std::uint32_t prefill_coalesce_waits_ = 0;
    std::uint64_t last_coalesced_request_ = 0;
    std::vector<std::uint8_t> control_executed_;
    std::vector<std::optional<UnitPermit>> reserved_units_;
    std::atomic<bool> admission_check_pending_{false};
    std::atomic<std::uint64_t> schedule_revision_{0};
    std::uint64_t published_revision_   = 0;
    Clock::time_point pending_deadline_ = Clock::time_point::max();
    std::atomic<bool> pending_scan_requested_{true};
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::vector<std::uint32_t> current_decode_lanes_;
    std::vector<std::uint64_t> current_decode_requests_;
    mutable ExposureSnapshot active_exposure_;
    std::optional<CompletionExposure> completion_exposure_;
    std::uint32_t completion_scope_depth_ = 0;
    std::vector<PendingCompletion> pending_completions_;
    RoundMembership round_membership_;
    ControlMembership control_membership_;
    PrefillMembership prefill_membership_;
    std::vector<std::uint8_t> prefill_excluded_;
    std::vector<std::uint32_t> reservation_candidates_;
    std::vector<std::uint64_t> cancellation_snapshot_;
    std::vector<std::uint8_t> row_flags_;
    std::vector<CommitDecision> commit_decisions_;
    std::vector<CommitPreview> commit_preview_;
    RuntimeStats cumulative_stats_;
    RuntimeStats stats_snapshot_;
    EngineSnapshot published_snapshot_;
    std::uint64_t frontier_decision_ = 0, published_frontier_decision_ = 0;
    std::vector<OwnedRequestObservation> request_observations_, published_requests_;
    std::vector<ExecutionProgress> execution_observations_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime

#include "runtime/engine/engine_metrics.inl"
#include "runtime/engine/engine_requests.inl"
#include "runtime/engine/engine_schedule.inl"
#include "runtime/engine/engine_execution.inl"
