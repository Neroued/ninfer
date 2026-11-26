#pragma once

#include "runtime/engine/dp_router.h"
#include "runtime/engine/instance.h"
#include "runtime/engine/replica_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class EngineCoordinator final : private ReplicaCoordinatorHooks<typename Instance::ModelContract> {
public:
    static constexpr EnginePurpose purpose = EnginePurpose::Generation;

    using ModelContract  = typename Instance::ModelContract;
    using Replica        = typename Instance::Replica;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;
    using Request        = RequestRecord<ModelContract>;
    using Clock          = std::chrono::steady_clock;

    EngineCoordinator(Instance& instance, const EngineOptions& options,
                      ContextMachineCostModel context_cost)
        : instance_(instance), dp_count_(static_cast<std::uint32_t>(instance.replicas.size())),
          local_lanes_(dp_count_ ? (options.max_concurrency + dp_count_ - 1U) / dp_count_ : 0U),
          max_context_(options.max_context), max_concurrency_(options.max_concurrency),
          speculative_positions_(options.speculative.backend == SpeculativeBackend::None
                                     ? 0
                                     : options.speculative.draft_tokens),
          context_cache_enabled_(options.context_cache.enabled),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          kv_pool_count_(instance.replicas.empty()
                             ? 0
                             : instance.replicas.front()->session->kv_pools().size()),
          prefix_domain_([&]() -> std::uint32_t {
              if (instance.replicas.empty()) return 0;
              if constexpr (requires { instance.replicas.front()->session->prefix_domain(); }) {
                  return instance.replicas.front()->session->prefix_domain();
              }
              return 0;
          }()),
          router_(DpRouterConfig{.replica_count        = dp_count_,
                                 .local_lanes          = local_lanes_,
                                 .max_concurrency      = options.max_concurrency,
                                 .max_pending_requests = options.max_pending_requests}) {
        if (dp_count_ == 0 || max_concurrency_ == 0 || options.max_pending_requests == 0 ||
            pending_timeout_.count() <= 0) {
            throw std::invalid_argument("EngineCoordinator bounds are invalid");
        }
        replicas_.reserve(dp_count_);
        try {
            for (std::uint32_t r = 0; r < dp_count_; ++r) {
                auto rep_engine = std::make_unique<ReplicaEngine<Replica>>(
                    *instance_.replicas[r], options,
                    instance_.replicas[r]->context_cost.value_or(context_cost),
                    static_cast<ReplicaCoordinatorHooks<ModelContract>&>(*this), r, local_lanes_);
                if (r == 0) speculative_startup_ = rep_engine->take_speculative_startup();
                replicas_.push_back(std::move(rep_engine));
            }
        } catch (...) {
            stop_and_join();
            replicas_.clear();
            throw;
        }
    }

    ~EngineCoordinator() noexcept {
        stop_and_join();
        replicas_.clear();
    }

    EngineCoordinator(const EngineCoordinator&)            = delete;
    EngineCoordinator& operator=(const EngineCoordinator&) = delete;

    [[nodiscard]] std::optional<SpeculativeStartupSummary> take_speculative_startup() noexcept {
        return std::exchange(speculative_startup_, std::nullopt);
    }

    void request_stop() noexcept {
        stopping_.store(true, std::memory_order_release);
        router_.request_stop();
        settle_unavailable_requests();
        for (auto& rep : replicas_) {
            if (rep) rep->request_stop();
        }
    }

    [[nodiscard]] bool is_available() const noexcept {
        return !stopping_.load(std::memory_order_acquire) && router_.is_available();
    }

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        [[nodiscard]] std::uint64_t request_id() const noexcept {
            return request_ ? request_->id : 0;
        }

        void wait_for_admission(const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            owner_->wait_for_admission(request_, cancellation);
        }

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCoordinator* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(EngineCoordinator& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCoordinator* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class EngineCoordinator;
    };

    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        const auto ids = router_.acquire_outstanding(pending_deadline);

        std::shared_ptr<Request> request;
        DpRouter::ScheduleOutcome outcome;
        try {
            const auto constraint_started =
                observation.constraint_timings ? Clock::now() : submitted;
            auto output = instance_.frontend.make_output_session(
                prompt, options.stop, options.output, options.execution.thinking,
                options.constraint, options.tool_choice);
            if (Clock::now() >= pending_deadline) {
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "inference request expired during grammar preparation");
            }
            const std::uint32_t capacity_output =
                max_context_ - prompt_summary.prompt_tokens + static_cast<std::uint32_t>(1);
            try {
                output.validate_generation_capacity(
                    std::min(options.execution.requested_output_tokens, capacity_output));
            } catch (const std::invalid_argument& error) {
                throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                                   error.what());
            }
            const auto ready = Clock::now();
            output.observe_constraint(
                observation.constraint_timings,
                std::chrono::duration<double>(ready - constraint_started).count());
            prepare_seconds += std::chrono::duration<double>(ready - submitted).count();

            request = std::make_shared<Request>(
                ids.request_id, ids.publication_order, std::move(prompt), std::move(output),
                prompt_summary, prepare_seconds, std::move(options), consumer_mode,
                std::move(observation), pending_deadline, ready);
            request->speculative_stats.attempted_per_position.resize(speculative_positions_);
            request->speculative_stats.accepted_per_position.resize(speculative_positions_);
            request->context_transfers.resize(kv_pool_count_ + 1);
            if (dp_count_ > 1 && request->options.execution.allow_prefix_reuse &&
                context_cache_enabled_) {
                request->prefix_keyer.emplace(request->prompt, prefix_domain_);
            }
            if (!is_available()) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            outcome = router_.enqueue(pending_item(request));
        } catch (...) {
            router_.abort_preparation(ids.request_id);
            throw;
        }

        central_submitted_.fetch_add(1, std::memory_order_release);
        apply_schedule_outcome(std::move(outcome));
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] EngineSnapshot measure(MeasurementCommand command) {
        std::unique_lock measure_lock(measure_mutex_, std::try_to_lock);
        if (!measure_lock.owns_lock()) {
            throw std::logic_error("a measurement boundary is already pending");
        }
        const bool begin = command.action == MeasurementAction::Begin;
        const bool end   = command.action == MeasurementAction::End;
        if ((begin || command.action == MeasurementAction::Reset) &&
            !measurement_participants_.empty())
            throw std::logic_error("a measurement window is already active");
        if (end && measurement_participants_.empty())
            throw std::logic_error("no measurement window is active");

        struct CutGuard {
            DpRouter& router;
            bool active = false;

            ~CutGuard() {
                if (active) router.end_submission_cut();
            }
        } cut_guard{router_};

        std::vector<std::uint32_t> participants;
        try {
            auto fence       = router_.begin_submission_cut();
            cut_guard.active = true;
            if (begin)
                measurement_participants_ = std::move(fence.healthy_participants);
            else if (!end)
                participants = std::move(fence.healthy_participants);
            const auto& required = begin || end ? measurement_participants_ : participants;
            for (;;) {
                apply_schedule_outcome(router_.sweep_and_schedule());
                const auto [done, healthy_ok] =
                    router_.wait_pre_cut_terminal_for(required, std::chrono::milliseconds(10));
                if (!healthy_ok) {
                    throw std::runtime_error(
                        "inference engine replica failed while draining measurement cut");
                }
                if (done) break;
            }

            std::vector<std::future<void>> futures;
            futures.reserve(required.size());
            std::exception_ptr error;
            for (const auto r : required) {
                try {
                    futures.push_back(replicas_[r]->start_measurement(command));
                } catch (...) {
                    if (!error) error = std::current_exception();
                }
            }
            for (auto& future : futures) {
                try {
                    future.get();
                } catch (...) {
                    if (!error) error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            for (const auto r : required) {
                if (!replicas_[r]->is_available())
                    throw std::runtime_error("inference engine replica failed during measurement");
            }
            auto result = snapshot();
            if (end) measurement_participants_.clear();
            return result;
        } catch (...) {
            const auto& rollback = begin || end ? measurement_participants_ : participants;
            for (const auto r : rollback) {
                if (!replicas_[r]->is_available()) continue;
                try {
                    replicas_[r]->start_measurement({MeasurementAction::Cancel}).get();
                } catch (...) {}
            }
            if (begin || end) measurement_participants_.clear();
            throw;
        }
    }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        EngineSnapshot out;
        out.stamp = {snapshot_sequence_.fetch_add(1, std::memory_order_relaxed) + 1,
                     observation_time_ns()};
        out.runtime.speculative_attempted_per_position.resize(speculative_positions_);
        out.runtime.speculative_accepted_per_position.resize(speculative_positions_);

        std::vector<OwnedRequestObservation> all_requests;
        if (options.include_requests) {
            for (const auto& pending : router_.snapshot_pending()) {
                const auto req = std::static_pointer_cast<Request>(pending.payload);
                if (!req) continue;
                all_requests.push_back({observe_central_request(*req), pending.ownership_epoch});
            }
        }

        bool any_healthy = false;
        out.replicas.reserve(dp_count_);
        std::map<std::pair<std::uint32_t, int>, PhysicalDeviceMemorySample> devices;
        std::map<std::uint32_t, PhysicalNodeMemorySample> nodes;

        std::uint64_t latest_frontier_decision = 0;
        for (std::uint32_t r = 0; r < dp_count_; ++r) {
            std::vector<OwnedRequestObservation> rep_requests;
            std::uint64_t frontier_decision = 0;
            auto rep_snap                   = replicas_[r]->replica_snapshot(
                options.include_requests ? &rep_requests : nullptr, &frontier_decision);
            if (frontier_decision > latest_frontier_decision) {
                latest_frontier_decision = frontier_decision;
                out.runtime.last_selected_frontier_tokens =
                    rep_snap.runtime.last_selected_frontier_tokens;
            }
            if (options.include_requests) {
                all_requests.insert(all_requests.end(), rep_requests.begin(), rep_requests.end());
            }

            if (rep_snap.healthy) {
                any_healthy = true;
                out.available_context_tokens_estimate += rep_snap.available_context_tokens_estimate;
                if (!out.progress_owner && rep_snap.progress_owner) {
                    out.progress_owner         = rep_snap.progress_owner;
                    out.progress_owner_replica = r;
                }
            } else {
                rep_snap.available_context_tokens_estimate = 0;
            }

            const auto& rr = rep_snap.runtime;
            accumulate_runtime_counters(out.runtime, rr, rep_snap.healthy);

            for (std::size_t k = 0; k < out.executions.size(); ++k) {
                merge_execution_counters(out.executions[k], rep_snap.executions[k]);
            }
            if (out.decode_by_batch.size() < rep_snap.decode_by_batch.size())
                out.decode_by_batch.resize(rep_snap.decode_by_batch.size());
            for (std::size_t b = 0; b < rep_snap.decode_by_batch.size(); ++b) {
                merge_execution_counters(out.decode_by_batch[b], rep_snap.decode_by_batch[b]);
            }
            if (rep_snap.healthy) {
                out.in_flight.insert(out.in_flight.end(), rep_snap.in_flight.begin(),
                                     rep_snap.in_flight.end());
            }

            if constexpr (requires { instance_.replicas[r]->executor->rank_descriptions(); }) {
                const auto descs = instance_.replicas[r]->executor->rank_descriptions();
                for (std::size_t i = 0; i < rep_snap.ranks.size(); ++i) {
                    const auto& rank_snap = rep_snap.ranks[i];
                    out.ranks.push_back(rank_snap);
                    if (rank_snap.physical_memory && i < descs.size()) {
                        const auto key = std::pair{descs[i].node, descs[i].device};
                        auto it        = devices.find(key);
                        if (it == devices.end() ||
                            rank_snap.physical_memory->sampled_ns >= it->second.sample.sampled_ns) {
                            devices[key] = {descs[i].node, descs[i].device,
                                            *rank_snap.physical_memory};
                        }
                        auto nit = nodes.find(descs[i].node);
                        if (nit == nodes.end() ||
                            rank_snap.physical_memory->sampled_ns >= nit->second.sampled_ns) {
                            nodes[descs[i].node] = {
                                descs[i].node, rank_snap.physical_memory->system_total_bytes,
                                rank_snap.physical_memory->system_available_bytes,
                                rank_snap.physical_memory->shared_physical_memory,
                                rank_snap.physical_memory->sampled_ns};
                        }
                    }
                }
            } else {
                out.ranks.insert(out.ranks.end(), rep_snap.ranks.begin(), rep_snap.ranks.end());
            }

            out.replicas.push_back(std::move(rep_snap));
        }

        out.runtime.requests.submitted = central_submitted_.load(std::memory_order_acquire);
        out.runtime.requests.completed = central_completed_.load(std::memory_order_acquire);
        out.runtime.requests.cancelled = central_cancelled_.load(std::memory_order_acquire);
        out.runtime.requests.failed    = central_failed_.load(std::memory_order_acquire);
        out.runtime.waiting_requests += static_cast<std::uint32_t>(router_.pending_count());

        std::sort(out.ranks.begin(), out.ranks.end(),
                  [](const auto& a, const auto& b) { return a.rank < b.rank; });
        for (auto& [k, v] : devices) out.physical_devices.push_back(std::move(v));
        for (auto& [k, v] : nodes) out.physical_nodes.push_back(std::move(v));

        if (options.include_requests) {
            auto& requests = out.requests.emplace();
            requests.reserve(all_requests.size());
            const auto epochs = router_.snapshot_request_epochs();
            for (auto& observation : all_requests) {
                const auto current = epochs.find(observation.snapshot.id);
                if (current != epochs.end() && current->second == observation.ownership_epoch)
                    requests.push_back(std::move(observation.snapshot));
            }
            std::sort(requests.begin(), requests.end(),
                      [](const auto& a, const auto& b) { return a.id < b.id; });
            requests.erase(std::unique(requests.begin(), requests.end(),
                                       [](const auto& a, const auto& b) { return a.id == b.id; }),
                           requests.end());
        }

        out.status = stopping_.load(std::memory_order_acquire) ? EngineStatus::Stopping
                     : !any_healthy                            ? EngineStatus::Failed
                                                               : EngineStatus::Ready;
        return out;
    }

private:
    void stop_and_join() noexcept {
        request_stop();
        for (auto& replica : replicas_) replica->join();
    }

    static DpRouter::PendingItem pending_item(const std::shared_ptr<Request>& request) {
        DpRouter::PendingItem item{
            .request_id         = request->id,
            .publication_order  = request->publication_order,
            .payload            = request,
            .cancelled          = &request->cancelled,
            .deadline           = request->deadline,
            .rejected_revisions = request->rejected_revisions,
        };
        if (request->prefix_keyer && request->prefix_keyer->max_frontier() > 0) {
            item.max_reusable_frontier = request->prefix_keyer->max_frontier();
            item.key_at                = [keyer = &*request->prefix_keyer](std::uint32_t frontier) {
                return keyer->key_at(frontier);
            };
        }
        return item;
    }

    // ReplicaCoordinatorHooks implementation
    bool prefix_routing_enabled() const noexcept override { return dp_count_ > 1; }

    bool bounce_enabled() const noexcept override { return dp_count_ > 1; }

    void on_prefix_delta(std::uint32_t replica_id, std::uint64_t epoch,
                         const PrefixIndexDelta<PrefixKey>& delta) override {
        router_.apply_prefix_delta(replica_id, epoch, delta);
    }

    void acknowledge_dispatch(std::uint32_t replica_id, std::uint32_t occupied_lanes,
                              std::uint64_t lane_sequence) noexcept override {
        router_.acknowledge_dispatch(replica_id, occupied_lanes, lane_sequence);
    }

    void commit_admission(std::uint64_t request_id) noexcept override {
        router_.commit_admission(request_id);
    }

    void cancel_admission(std::uint64_t request_id) noexcept override {
        router_.cancel_admission(request_id);
    }

    std::uint64_t next_frontier_decision() noexcept override {
        return frontier_decision_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    void on_replica_offer(std::uint32_t replica_id, const ReplicaStatusOffer& offer) override {
        apply_schedule_outcome(router_.update_replica_offer(replica_id, offer));
    }

    void bounce_unadmitted(std::uint32_t replica_id, const ReplicaStatusOffer& offer,
                           bool handoff_to_local_restore,
                           std::shared_ptr<Request> request) override {
        apply_schedule_outcome(router_.return_unadmitted(replica_id, offer.resource_revision,
                                                         pending_item(request), offer,
                                                         handoff_to_local_restore));
    }

    bool acquire_restore_permit(std::uint32_t replica_id) noexcept override {
        return router_.try_acquire_restore_permit(replica_id);
    }

    void cancel_waiting_restore(std::uint32_t replica_id) override {
        apply_schedule_outcome(router_.cancel_waiting_restore(replica_id));
    }

    void release_or_handoff_residency(std::uint32_t replica_id, bool handoff_to_local_restore,
                                      const ReplicaStatusOffer& updated_offer) override {
        apply_schedule_outcome(router_.release_or_handoff_residency(
            replica_id, handoff_to_local_restore, updated_offer));
    }

    void on_request_terminal(std::uint64_t request_id, FinishReason reason,
                             bool failed) noexcept override {
        auto& count = failed                              ? central_failed_
                      : reason == FinishReason::Cancelled ? central_cancelled_
                                                          : central_completed_;
        count.fetch_add(1, std::memory_order_release);
        router_.mark_request_terminal(request_id);
    }

    void release_outstanding_permit() noexcept override { router_.release_outstanding_permit(); }

    void on_replica_failing(std::uint32_t replica_id) noexcept override {
        try {
            auto iso = router_.mark_replica_unhealthy(replica_id);
            apply_schedule_outcome(std::move(iso.schedule));
        } catch (...) {}
        settle_unavailable_requests();
    }

    void on_replica_failed_release_residencies(std::uint32_t count) noexcept override {
        try {
            apply_schedule_outcome(router_.release_failed_replica_residencies(count));
        } catch (...) {}
    }

    void settle_unavailable_requests() noexcept {
        while (auto item = router_.take_unavailable()) {
            auto request = std::static_pointer_cast<Request>(std::move(item->payload));
            settle_central_rejected(request, DpRouter::ExpiredItem::Reason::Unavailable);
        }
    }

    void apply_schedule_outcome(DpRouter::ScheduleOutcome outcome) noexcept {
        for (auto& grant : outcome.dispatches) {
            try {
                if (grant.replica_id < replicas_.size() &&
                    replicas_[grant.replica_id]->accept_dispatch(grant.item)) {
                    continue;
                }
            } catch (const std::bad_alloc&) {
                // Local queue insertion did not take ownership. Reroute the intact grant.
            }
            auto request = std::static_pointer_cast<Request>(grant.item.payload);
            try {
                auto retry =
                    router_.return_unadmitted(grant.replica_id, UINT64_MAX, std::move(grant.item));
                apply_schedule_outcome(std::move(retry));
            } catch (const std::bad_alloc&) {
                router_.drop_dispatch(grant.replica_id, request->id);
                settle_central_rejected(request, DpRouter::ExpiredItem::Reason::Unavailable);
            }
        }
        for (auto& expired : outcome.rejected) {
            auto req = std::static_pointer_cast<Request>(expired.item.payload);
            settle_central_rejected(req, expired.reason);
        }
        for (const auto r : outcome.restore_grants) {
            if (r < replicas_.size()) replicas_[r]->notify_restore_grant();
        }
    }

    void settle_central_rejected(const std::shared_ptr<Request>& req,
                                 DpRouter::ExpiredItem::Reason reason) noexcept {
        if (!req) return;
        GenerationResult result;
        std::exception_ptr error;
        if (reason == DpRouter::ExpiredItem::Reason::Cancelled) {
            try {
                (void)req->output.preview_terminal(FinishReason::Cancelled);
                (void)req->output.commit_preview();
            } catch (...) {}
            result.prompt                        = req->prompt_summary;
            result.finish_reason                 = FinishReason::Cancelled;
            result.stats.engine_request_id       = req->id;
            result.stats.timings.prepare_seconds = req->prepare_seconds;
            const auto queued                    = Clock::now() - req->submitted;
            result.stats.engine_timing.queue_wait_seconds =
                std::chrono::duration<double>(queued).count();
            result.stats.timings.total_seconds =
                req->prepare_seconds + result.stats.engine_timing.queue_wait_seconds;
        } else {
            try {
                if (reason == DpRouter::ExpiredItem::Reason::QueueTimeout)
                    throw RequestError(RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission");
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            } catch (...) { error = std::current_exception(); }
        }

        bool release = false;
        {
            std::lock_guard lock(req->mutex);
            if (!req->response_done) {
                req->prompt = {};
                req->prefix_keyer.reset();
                req->result        = std::move(result);
                req->error         = std::move(error);
                req->response_done = true;
                on_request_terminal(req->id, req->result.finish_reason, req->error != nullptr);
                if (req->consumer_released && !req->capacity_released) {
                    req->capacity_released = true;
                    release                = true;
                }
            }
        }
        if (release) { router_.release_outstanding_permit(); }
        req->cv.notify_all();
    }

    void notify_request_cancelled(const std::shared_ptr<Request>& request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        try {
            apply_schedule_outcome(router_.sweep_and_schedule());
        } catch (...) {}
        for (auto& rep : replicas_) {
            if (rep) rep->notify_cancellation();
        }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->response_done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { router_.release_outstanding_permit(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        notify_request_cancelled(request);
        release_consumer(request);
    }

    void wait_for_admission(const std::shared_ptr<Request>& request,
                            const CancellationView& cancellation) {
        try {
            for (;;) {
                const bool cancelled = cancellation.requested();
                if (!cancelled && Clock::now() >= request->deadline) {
                    apply_schedule_outcome(router_.sweep_and_schedule());
                }
                std::unique_lock lock(request->mutex);
                if (request->error) {
                    const auto error      = request->error;
                    const auto replica_id = request->replica_id;
                    lock.unlock();
                    report_replica_error(*request, replica_id);
                    std::rethrow_exception(error);
                }
                if (cancelled || request->cancelled.load(std::memory_order_acquire) ||
                    (request->response_done &&
                     request->result.finish_reason == FinishReason::Cancelled)) {
                    throw RequestError(RequestErrorKind::Cancelled,
                                       "inference request cancelled while waiting for admission");
                }
                if (request->generation_start || request->response_done) { return; }
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->generation_start.has_value() || request->response_done;
                });
            }
        } catch (...) {
            notify_request_cancelled(request);
            throw;
        }
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            EngineCoordinator* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        bool scheduling_failed   = false;
        const auto fail_consumer = [&] {
            if (caller_error == nullptr) { caller_error = std::current_exception(); }
            notify_request_cancelled(request);
        };
        std::optional<GenerationStart> start;
        bool start_delivered = false;
        std::optional<PromptProgress> progress;
        std::vector<typename Request::StreamEvent> events;
        for (;;) {
            if (!start_delivered && Clock::now() >= request->deadline) {
                apply_schedule_outcome(router_.sweep_and_schedule());
            }
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done ||
                           (!start_delivered && request->generation_start) ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                if (!start_delivered && request->generation_start) {
                    start           = request->generation_start;
                    start_delivered = true;
                }
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                } catch (...) { fail_consumer(); }
            }
            for (auto& event : events) {
                if (const auto* scheduling =
                        std::get_if<std::unique_ptr<GenerationSchedulingObservation>>(&event)) {
                    if (scheduling_failed) { continue; }
                    try {
                        request->observation.scheduling(**scheduling);
                    } catch (...) {
                        scheduling_failed = true;
                        fail_consumer();
                    }
                } else if (const auto* first_token =
                               std::get_if<GenerationFirstTokenObservation>(&event)) {
                    if (caller_error == nullptr) {
                        try {
                            request->observation.first_token(*first_token);
                        } catch (...) { fail_consumer(); }
                    }
                } else if (caller_error == nullptr && sink != nullptr) {
                    try {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputEvent>(event)));
                        }
                    } catch (...) { fail_consumer(); }
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) { notify_request_cancelled(request); }
                } catch (...) { fail_consumer(); }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            // Terminal state is immutable for the sole consumer after response_done is observed.
            if (request->error != nullptr) {
                report_replica_error(*request, request->replica_id);
                std::rethrow_exception(request->error);
            }
            return std::move(request->result);
        }
    }

    static void report_replica_error(const Request& request,
                                     std::optional<std::uint32_t> replica_id) noexcept {
        if (!replica_id || !request.observation.error_replica) return;
        try {
            request.observation.error_replica(*replica_id);
        } catch (...) {
            // An observation failure cannot replace the exception that ended the request.
        }
    }

    static RequestSnapshot observe_central_request(const Request& request) {
        RequestSnapshot out;
        out.id       = request.id;
        out.phase    = RequestPhase::Waiting;
        const auto s = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           request.submitted.time_since_epoch())
                           .count();
        out.submitted_ns           = s > 0 ? static_cast<std::uint64_t>(s) : 0;
        out.prompt_tokens          = request.prompt_summary.prompt_tokens;
        out.logical_context_tokens = request.prompt_summary.prompt_tokens;
        out.output_token_limit     = request.options.execution.requested_output_tokens;
        return out;
    }

    static void merge_execution_counters(ExecutionCounters& dst, const ExecutionCounters& src) {
        dst.units += src.units;
        dst.rows += src.rows;
        dst.input_tokens += src.input_tokens;
        dst.scheduled_positions += src.scheduled_positions;
        for (auto field : {&ExecutionExposure::result, &ExecutionExposure::resolution,
                           &ExecutionExposure::retirement, &ExecutionExposure::total}) {
            auto& d       = dst.exposure.*field;
            const auto& s = src.exposure.*field;
            d.count += s.count;
            d.total_ns += s.total_ns;
            d.maximum_ns = std::max(d.maximum_ns, s.maximum_ns);
        }
        for (std::size_t b = 0; b < dst.latency.bins.size(); ++b) {
            dst.latency.bins[b] += src.latency.bins[b];
        }
    }

    static void merge_transfer_series(ContextTransferSeries& dst,
                                      const ContextTransferSeries& src) {
        for (std::size_t i = 0; i < dst.size(); ++i) {
            dst[i].bytes += src[i].bytes;
            dst[i].seconds += src[i].seconds;
            dst[i].images += src[i].images;
            dst[i].page_groups += src[i].page_groups;
            dst[i].copy_operations += src[i].copy_operations;
        }
    }

    static void accumulate_runtime_counters(RuntimeStats& dst, const RuntimeStats& src,
                                            bool healthy) {
        // Monotonic counters are accumulated across both healthy and failed replicas so cluster
        // deltas never jump backwards when a replica domain is isolated.
        dst.host_work.engine_boundary_ns += src.host_work.engine_boundary_ns;
        dst.host_work.session_call_ns += src.host_work.session_call_ns;
        dst.host_work.executor_poll_ns += src.host_work.executor_poll_ns;
        dst.host_work.engine_commit_output_ns += src.host_work.engine_commit_output_ns;
        dst.host_work.engine_maintenance_ns += src.host_work.engine_maintenance_ns;
        dst.host_work.scheduler_yield_ns += src.host_work.scheduler_yield_ns;
        dst.host_work.executor_flush_ns += src.host_work.executor_flush_ns;
        dst.host_work.decode_host_ns += src.host_work.decode_host_ns;
        dst.host_work.decode_scheduler_yield_ns += src.host_work.decode_scheduler_yield_ns;
        dst.host_work.prefill_host_ns += src.host_work.prefill_host_ns;
        dst.host_work.prefill_scheduler_yield_ns += src.host_work.prefill_scheduler_yield_ns;
        dst.host_work.control_host_ns += src.host_work.control_host_ns;
        dst.host_work.control_scheduler_yield_ns += src.host_work.control_scheduler_yield_ns;
        dst.host_work.prefill_units += src.host_work.prefill_units;
        dst.host_work.control_units += src.host_work.control_units;
        dst.host_work.stats_publication_ns += src.host_work.stats_publication_ns;
        dst.host_work.stats_publication_invocations += src.host_work.stats_publication_invocations;

        dst.prompt_tokens += src.prompt_tokens;
        dst.generated_tokens += src.generated_tokens;
        dst.injected_control_tokens += src.injected_control_tokens;
        dst.speculative_rounds += src.speculative_rounds;
        dst.speculative_proposed_tokens += src.speculative_proposed_tokens;
        dst.speculative_draft_tokens += src.speculative_draft_tokens;
        dst.speculative_accepted_tokens += src.speculative_accepted_tokens;
        dst.speculative_bypass_steps += src.speculative_bypass_steps;
        dst.speculative_fallback_steps += src.speculative_fallback_steps;
        dst.speculative_committed_draft_tokens += src.speculative_committed_draft_tokens;
        for (std::size_t i = 0; i < std::min(dst.speculative_attempted_per_position.size(),
                                             src.speculative_attempted_per_position.size());
             ++i) {
            dst.speculative_attempted_per_position[i] += src.speculative_attempted_per_position[i];
            dst.speculative_accepted_per_position[i] += src.speculative_accepted_per_position[i];
        }
        dst.computed_prefill_tokens += src.computed_prefill_tokens;
        dst.committed_decode_tokens += src.committed_decode_tokens;
        dst.decode_rounds += src.decode_rounds;
        dst.decode_row_rounds += src.decode_row_rounds;
        dst.active_captures_completed += src.active_captures_completed;
        dst.active_captures_aborted += src.active_captures_aborted;
        dst.preemptions += src.preemptions;
        dst.snapshot_restores += src.snapshot_restores;
        dst.direct_restores += src.direct_restores;
        dst.replay_restores += src.replay_restores;
        dst.replayed_tokens += src.replayed_tokens;
        dst.root_selections += src.root_selections;
        dst.checkpoint_selections += src.checkpoint_selections;
        dst.reused_prompt_tokens += src.reused_prompt_tokens;

        dst.state_moves += src.state_moves;
        dst.state_forks += src.state_forks;
        dst.materialization_state_forks += src.materialization_state_forks;
        dst.state_restores += src.state_restores;
        merge_transfer_series(dst.context_transfers.state, src.context_transfers.state);
        if (dst.context_transfers.kv.size() < src.context_transfers.kv.size()) {
            dst.context_transfers.kv.resize(src.context_transfers.kv.size());
        }
        for (std::size_t i = 0; i < src.context_transfers.kv.size(); ++i) {
            merge_transfer_series(dst.context_transfers.kv[i], src.context_transfers.kv[i]);
        }

        const auto merge_prefix = [](PrefixCacheCounters& d, const PrefixCacheCounters& s,
                                     bool include_retained) {
            if (include_retained) d.retained += s.retained;
            d.published += s.published;
            d.released += s.released;
            d.reclaimed += s.reclaimed;
            d.adopted += s.adopted;
            d.reused_tokens += s.reused_tokens;
        };
        merge_prefix(dst.prefix_cache.input_replay, src.prefix_cache.input_replay, healthy);
        merge_prefix(dst.prefix_cache.long_anchor, src.prefix_cache.long_anchor, healthy);
        merge_prefix(dst.prefix_cache.shared_prefix, src.prefix_cache.shared_prefix, healthy);
        merge_prefix(dst.prefix_cache.continuation, src.prefix_cache.continuation, healthy);

        dst.pressure_spill_pages += src.pressure_spill_pages;
        dst.partial_tail_cow_pages += src.partial_tail_cow_pages;
        dst.actual_context_transfer_seconds += src.actual_context_transfer_seconds;

        if (!healthy) return;

        // Live gauges are summed only across healthy replicas.
        dst.running_requests += src.running_requests;
        dst.prefilling_requests += src.prefilling_requests;
        dst.decode_ready_requests += src.decode_ready_requests;
        dst.waiting_requests += src.waiting_requests;
        dst.paused_requests += src.paused_requests;
        dst.replaying_requests += src.replaying_requests;
        dst.materializing_requests += src.materializing_requests;
        dst.capture_pending_requests += src.capture_pending_requests;
        dst.terminal_pending_requests += src.terminal_pending_requests;

        if (dst.kv_pools.empty()) {
            dst.kv_pools = src.kv_pools;
        } else {
            for (std::size_t i = 0; i < std::min(dst.kv_pools.size(), src.kv_pools.size()); ++i) {
                dst.kv_pools[i].occupied_page_groups += src.kv_pools[i].occupied_page_groups;
            }
        }
        dst.context_pools.insert(dst.context_pools.end(), src.context_pools.begin(),
                                 src.context_pools.end());
        dst.context_resources.insert(dst.context_resources.end(), src.context_resources.begin(),
                                     src.context_resources.end());
        dst.placements.insert(dst.placements.end(), src.placements.begin(), src.placements.end());
        dst.context_metadata.budget_bytes += src.context_metadata.budget_bytes;
        dst.context_metadata.allocated_bytes += src.context_metadata.allocated_bytes;
        dst.context_metadata.reserved_bytes += src.context_metadata.reserved_bytes;
        dst.state_objects += src.state_objects;
        dst.history_objects += src.history_objects;
    }

    Instance& instance_;
    const std::uint32_t dp_count_;
    const std::uint32_t local_lanes_;
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const std::uint32_t speculative_positions_;
    const bool context_cache_enabled_;
    const std::chrono::milliseconds pending_timeout_;
    const std::size_t kv_pool_count_;
    const std::uint32_t prefix_domain_;

    DpRouter router_;
    std::vector<std::unique_ptr<ReplicaEngine<Replica>>> replicas_;
    std::optional<SpeculativeStartupSummary> speculative_startup_;

    std::mutex measure_mutex_;
    std::vector<std::uint32_t> measurement_participants_;
    std::atomic<bool> stopping_{false};
    std::atomic<std::uint64_t> central_submitted_{0};
    std::atomic<std::uint64_t> central_completed_{0};
    std::atomic<std::uint64_t> central_cancelled_{0};
    std::atomic<std::uint64_t> frontier_decision_sequence_{0};
    std::atomic<std::uint64_t> central_failed_{0};
    mutable std::atomic<std::uint64_t> snapshot_sequence_{0};
};

} // namespace ninfer::runtime
