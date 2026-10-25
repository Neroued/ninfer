#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/context_cache/placement_policy.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/executor/contract.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <limits>
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

template <class Instance>
class EngineCore {

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
    using ResourceManagement = ResourceManager<ModelContract>;
    using Clock              = std::chrono::steady_clock;

    EngineCore(Instance& instance, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), max_context_(options.max_context),
          max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          resources_(options.context_cache.enabled, std::move(context_cost)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        cumulative_stats_.context_transfers.kv.resize(instance_.session->kv_pools().size());
        for (const auto& pool : instance_.session->kv_pools()) {
            cumulative_stats_.kv_pools.push_back({pool.id.index, std::string(pool.label), 0});
        }
        stats_snapshot_             = cumulative_stats_;
        published_snapshot_.runtime = cumulative_stats_;
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        paused_.reserve(max_outstanding_);
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                instance_.executor->bind_to_current_thread();
                publish_runtime_stats();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~EngineCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

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

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class EngineCore;
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

        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        {
            std::unique_lock lock(queue_mutex_);
            if (!queue_cv_.wait_until(lock, pending_deadline,
                                      [&] { return stopping_ || failed_ || !measurement_; }))
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "inference request expired at a measurement boundary");
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            if (next_request_id_ == 0 || next_publication_order_ == 0) {
                throw std::overflow_error("request identity space exhausted");
            }
            ++outstanding_;
            ++preparing_requests_;
            request_id        = next_request_id_++;
            publication_order = next_publication_order_++;
        }

        std::shared_ptr<Request> request;
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
            request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                                std::move(output), prompt_summary, prepare_seconds,
                                                std::move(options), consumer_mode,
                                                std::move(observation), pending_deadline, ready);
            request->decode_by_batch.resize(max_concurrency_);
            request->context_transfers.resize(instance_.session->kv_pools().size() + 1);
        } catch (...) {
            {
                std::lock_guard lock(queue_mutex_);
                --preparing_requests_;
                --outstanding_;
            }
            queue_cv_.notify_all();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            --preparing_requests_;
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] EngineSnapshot snapshot() const {
        std::lock_guard lock(stats_mutex_);
        return published_snapshot_;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    [[nodiscard]] EngineSnapshot measure(MeasurementCommand command) {
        auto request     = std::make_unique<MeasurementRequest>();
        request->command = command;
        auto result      = request->completion.get_future();
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) throw std::runtime_error("inference engine is unavailable");
            if (measurement_) throw std::logic_error("a measurement boundary is already pending");
            measurement_ = std::move(request);
        }
        queue_cv_.notify_all();
        return result.get();
    }

private:
    struct MeasurementRequest {
        MeasurementCommand command{};
        std::promise<EngineSnapshot> completion;
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
        std::vector<bool> cancelled;
        std::vector<TokenId> control_tokens;
        std::uint32_t control_stride = 0;
        std::vector<std::size_t> generated_sizes;
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept;

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    struct ActiveExposureSet {
        std::array<ActiveExposure, kMaximumConcurrency> entries{};
        std::size_t size = 0;
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept;

    [[nodiscard]] ActiveExposureSet active_exposure_set() const;

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const;

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept;

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept;

    void add_class_host_time(std::uint64_t host_ns, std::uint64_t scheduler_yield_ns) noexcept;

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
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
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

        void checkpoint() noexcept {
            if (!active_) { return; }
            owner_.finish_engine_phase(measurement_, phase_);
            measurement_ = owner_.begin_host_phase();
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ExecutionCallScope {
    public:
        explicit ExecutionCallScope(EngineCore& owner,
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
        EngineCore& owner_;
        CoordinatorCall call_;
        HostPhaseMeasurement measurement_;
        bool active_ = true;
    };

    void publish_runtime_stats();

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation);

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
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    struct CaptureDecision {
        SequenceHandle sequence;
        std::uint32_t frontier = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
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

    void release_reserved_capacity() noexcept;

    void release_consumer(const std::shared_ptr<Request>& request) noexcept;

    void abandon_request(std::shared_ptr<Request> request) noexcept;

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept;

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept;

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error);

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason);

    void complete_cancelled(const std::shared_ptr<Request>& request);

    void complete_detached_cancelled(const std::shared_ptr<Request>& request);

    void remove_completed_slot(std::uint32_t lane);

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept;

    bool settle_terminal_requests(HostPhaseMeasurement& boundary);

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary);

    [[nodiscard]] bool expire_pending_requests();

    void commit_pending(PendingBatch&& pending, ComputeFlight flight);

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next);

    bool prepare_semantic_capture(const std::shared_ptr<Request>& request);

    void resolve_prefill_progress(const std::shared_ptr<Request>& request,
                                  typename ModelContract::PrefillProgress&& progress,
                                  ComputeFlight flight);

    void ensure_base_plan(const std::shared_ptr<Request>& request);

    [[nodiscard]] std::optional<std::uint32_t> free_lane() const noexcept;

    [[nodiscard]] bool resident_empty() const noexcept;

    [[nodiscard]] bool has_older_resident(std::uint64_t ticket) const noexcept;

    void record_context_work(const typename ModelContract::ContextProgress& progress,
                             const std::shared_ptr<Request>& owner);

    void resolve_context_progress(typename ModelContract::ContextProgress&& progress,
                                  HostPhaseMeasurement& boundary);

    ReclaimProgress reclaim(const ContextAssessment& assessment, bool allow_pause,
                            std::span<const Checkpoint> excluded               = {},
                            typename ResourceManagement::ReclaimCursor* cursor = nullptr,
                            std::optional<std::uint64_t> priority_ticket       = std::nullopt);

    bool pause_reclaim_victim(std::optional<std::uint64_t> priority_ticket,
                              const ContextAssessment&);

    enum class PauseGoal { YieldLane, ReleaseCapacity, RecoverCapacity, RelocateOnly };
    bool pause_resident(std::uint32_t lane, PauseGoal goal = PauseGoal::YieldLane,
                        const ContextAssessment* pressure = nullptr);

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

    enum class ReservationScope { Round, Prefill };

    [[nodiscard]] ExecutionUnit execution_unit(const Request& request) const;

    void reserve_resident_units(ReservationScope scope = ReservationScope::Round);
    [[nodiscard]] std::optional<std::uint32_t>
    next_prefill(const std::array<std::shared_ptr<Request>, kMaximumConcurrency>&) const;

    // Admission can pin an existing writer's tail after its reservation. Membership must
    // observe current context dependencies, including pauses selected during reclamation.
    [[nodiscard]] bool unit_runnable(std::uint32_t lane) const;

    [[nodiscard]] ComputeFlight freeze_members(std::span<const std::uint32_t> lanes,
                                               UnitKind kind) const;

    [[nodiscard]] bool start_prefill(std::uint32_t lane);

    [[nodiscard]] bool start_decode(const RoundMembership& membership);

    void discard_control_preview(const ComputeFlight& flight) noexcept;

    [[nodiscard]] bool start_control(const ControlMembership& membership);

    void resolve_control(const ComputeFlight& flight);

    void supply_decision(ExecutionTicket ticket, const DecisionRequest& decision);

    void record_decision_upload(const DecisionConsumed& consumed);

    [[nodiscard]] bool poll_execution(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool execution_outstanding() const noexcept;

    [[nodiscard]] bool compute_pending() const noexcept;

    void fail_all(std::exception_ptr error) noexcept;

    void begin_cycle(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool advance_cycle();

    void worker_loop() noexcept;

    Instance& instance_;
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    ResourceManagement resources_;
    PlacementId next_placement_tie_;

    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::unique_ptr<MeasurementRequest> measurement_;
    std::size_t preparing_requests_       = 0;
    std::size_t outstanding_              = 0;
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;
    std::optional<AdmissionDecision> admission_decision_;
    std::uint64_t admission_generation_ = 1;
    std::array<std::optional<CaptureDecision>, kMaximumConcurrency> capture_decisions_;
    std::shared_ptr<Request> context_owner_;
    std::vector<std::shared_ptr<Request>> paused_;
    Scheduling scheduler_;
    std::optional<ComputeFlight> compute_;
    std::optional<ComputeFlight> retiring_;
    CycleStep cycle_step_ = CycleStep::Boundary;
    std::array<bool, kMaximumConcurrency> control_executed_{};
    std::array<std::optional<UnitPermit>, kMaximumConcurrency> reserved_units_{};
    std::atomic<bool> admission_check_pending_{false};
    std::atomic<std::uint64_t> schedule_revision_{0};
    Clock::time_point pending_deadline_        = Clock::time_point::max();
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    RuntimeStats stats_snapshot_;
    EngineSnapshot published_snapshot_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime

#include "runtime/engine/engine_metrics.inl"

#include "runtime/engine/engine_requests.inl"
#include "runtime/engine/engine_schedule.inl"
#include "runtime/engine/engine_execution.inl"
