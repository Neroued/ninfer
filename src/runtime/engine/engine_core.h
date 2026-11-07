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
#include <cstdio>
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
          prefill_chunk_(options.prefill_chunk),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          speculative_positions_(options.speculative.backend == SpeculativeBackend::None
                                     ? 0
                                     : options.speculative.draft_tokens),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          resources_(options.context_cache.enabled, std::move(context_cost)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        scheduler_.configure_prefill(prefill_pacing(instance_.session->prefill_pack_tokens(),
                                                    prefill_chunk_, max_concurrency_));
        cumulative_stats_.context_transfers.kv.resize(instance_.session->kv_pools().size());
        for (const auto& pool : instance_.session->kv_pools()) {
            cumulative_stats_.kv_pools.push_back({pool.id.index, std::string(pool.label), 0});
        }
        cumulative_stats_.speculative_attempted_per_position.resize(speculative_positions_);
        cumulative_stats_.speculative_accepted_per_position.resize(speculative_positions_);
        for (const auto& placement : instance_.session->placement_descriptions())
            cumulative_stats_.placements.push_back({.id = placement.id});
        stats_snapshot_             = cumulative_stats_;
        published_snapshot_.runtime = cumulative_stats_;
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        paused_.reserve(max_outstanding_);
        request_observations_.reserve(max_outstanding_ + 1);
        published_requests_.reserve(max_outstanding_);
        published_snapshot_.decode_by_batch.resize(max_concurrency_);
        const auto frames = instance_.executor->capacity();
        execution_observations_.reserve(frames.compute_frames + frames.context_frames);
        published_snapshot_.in_flight.reserve(frames.compute_frames + frames.context_frames);
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
        request_stop();
        if (worker_.joinable()) { worker_.join(); }
    }

    // May be called while consumers are waiting; only the worker releases model resources.
    void request_stop() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    // The only definition of the prefill pacing constants. `pack_tokens` is
    // Session::prefill_pack_tokens(): the most tokens one prefill execution carries over all its
    // segments. Without a resolved chunk there is no unit size to budget, so a Throughput cycle
    // keeps one unit.
    //  - Throughput: one packed execution of pack_tokens per cycle, over up to max(2, C) units.
    //  - Interleave: at most kInterleaveChunks chunks of tokens per decode round, and no more
    //    than kInterleaveNs of predicted time; a lane without an estimate starts with
    //    kInterleaveColdTokens.
    //  - kInterleaveNs bounds the decode stall that the prefill work in front of a decode round
    //    may cause. The single 1024-token chunk per cycle that preceded this pacing cost 50 ms at
    //    depth 0 and 125 ms at 160K (measured prefill cost per token: 48 us at depth 0, 77 us at
    //    64K, 106 us at 128K, 120 us at 160K). 100 ms is that chunk at the middle of the contexts
    //    served beside decode, so a shallow cycle spends it on one whole chunk and a deep one on
    //    fewer tokens than the old chunk. A flight also has a fixed cost of roughly 14 ms
    //    (inferred from flights of a few hundred tokens) that the measured ns/token spreads over
    //    its tokens.
    //  - Time-derived units are multiples of kUnitGranularity, the chunk alignment.
    //  - Debt: a decode round takes 4-6 ms alone, but a lane that shares every cycle with 100 ms
    //    of prefill decodes at about a twentieth of its solo rate, for as long as the prefill
    //    lasts (tens of seconds at 128K-200K contexts). The Scheduler counts the prefill time
    //    that ran beside a waiting lane as a debt. kInterleaveDebtCapNs lets a short request
    //    prefill without waiting (three 100 ms cycles), and each Interleave cycle repays
    //    kInterleaveRepayNs, about a quarter of a 15-20 ms decode-only cycle, so a long prefill
    //    keeps about a fifth of the time while a lane decodes.
    static constexpr std::uint64_t kInterleaveNs         = 100'000'000;
    static constexpr std::uint32_t kInterleaveChunks     = 2;
    static constexpr std::uint32_t kInterleaveColdTokens = 1024;
    static constexpr std::uint32_t kUnitGranularity      = 128;
    static constexpr std::uint64_t kInterleaveDebtCapNs  = 300'000'000;
    static constexpr std::uint64_t kInterleaveRepayNs    = 4'000'000;

    [[nodiscard]] static PrefillPacing prefill_pacing(std::uint32_t pack_tokens,
                                                      std::uint32_t prefill_chunk,
                                                      std::uint32_t concurrency) noexcept {
        PrefillPacing pacing;
        pacing.interleave_ns = kInterleaveNs;
        if (prefill_chunk == 0) {
            pacing.throughput_units_per_cycle = 1;
            return pacing;
        }
        pacing.throughput_tokens_per_cycle = std::max(pack_tokens, prefill_chunk);
        pacing.throughput_units_per_cycle  = std::max(2U, concurrency);
        pacing.interleave_tokens_per_cycle = kInterleaveChunks * prefill_chunk;
        pacing.interleave_cold_tokens      = std::min(prefill_chunk, kInterleaveColdTokens);
        pacing.unit_granularity            = kUnitGranularity;
        pacing.interleave_debt_cap_ns      = kInterleaveDebtCapNs;
        pacing.interleave_repay_ns         = kInterleaveRepayNs;
        return pacing;
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
            request->speculative_stats.attempted_per_position.resize(speculative_positions_);
            request->speculative_stats.accepted_per_position.resize(speculative_positions_);
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
            ++submitted_requests_;
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        std::lock_guard lock(stats_mutex_);
        auto result = published_snapshot_;
        if (options.include_requests) result.requests = published_requests_;
        return result;
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
        std::array<bool, kMaximumConcurrency> progressed{};
        std::optional<std::uint64_t> retired_ns;
        bool service_accounted = false;
        std::vector<TokenId> control_tokens;
        std::uint32_t control_stride = 0;
        std::vector<std::size_t> generated_sizes;
        // Wall clock at the Session call that started a prefill or replay execution.
        Clock::time_point started{};
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
    [[nodiscard]] RequestSnapshot observe_request(const Request&) const;
    void record_completion(const std::shared_ptr<Request>&, FinishReason, bool failed);

    void wait_for_admission(const std::shared_ptr<Request>& request,
                            const CancellationView& cancellation);

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

    // ResourceManagement may release a checkpoint that a Capture or Demote transaction still
    // reads, so the Engine calls publish/finish/abandon/release_source only while the Session has
    // no context operation. A request that ends during one keeps its Session result and lane
    // release; only this resource tail is recorded here and replayed by begin_cycle.
    // `finished`: publish the continuation checkpoint, then finish its owner. Otherwise: release
    // the request's waiting source, then abandon its owner.
    struct DeferredSettlement {
        std::uint64_t request_id = 0;
        typename ResourceManagement::OwnerToken owner = 0;
        std::optional<Checkpoint> checkpoint;
        std::shared_ptr<const typename Request::BasePlan> base;
        std::uint64_t publication_order = 0;
        bool finished                   = false;
    };

    // Fixes, for one settle/cancel pass, whether resource tails must wait. The decision is taken
    // before the pass calls Session::finish/abort, because those calls queue their own residency
    // release and would otherwise make every tail look blocked.
    class SettlementPass {
    public:
        explicit SettlementPass(EngineCore& engine) : engine_(engine) {
            engine_.pass_blocked_ = engine_.resource_calls_blocked();
        }

        ~SettlementPass() { engine_.pass_blocked_.reset(); }

        SettlementPass(const SettlementPass&)            = delete;
        SettlementPass& operator=(const SettlementPass&) = delete;

    private:
        EngineCore& engine_;
    };

    // A context operation is in flight, or earlier tails are waiting and must stay ahead.
    [[nodiscard]] bool resource_calls_blocked() const noexcept {
        return !deferred_settlements_.empty() || instance_.session->has_context_operation();
    }

    // Runs the resource tail now, or records it when resource calls are blocked.
    void settle_resources(DeferredSettlement&& settlement);

    void apply_settlement(const DeferredSettlement& settlement);

    // Replays recorded tails in publication order while no context operation is in flight.
    void flush_deferred_settlements();

    // Whether the Session declines to end this lane's sequence now. A sequence already consumed
    // by commit has nothing left to end; only an execution that still names it defers settlement.
    [[nodiscard]] bool lane_terminal_blocked(const Request& request) const noexcept;

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

    bool prepare_completion(const std::shared_ptr<Request>& request);

    void release_request(const std::shared_ptr<Request>& request) noexcept;

    // The worker publishes the matching scheduling snapshot before making the response visible.
    void publish_completion(const std::shared_ptr<Request>& request, GenerationResult result,
                            std::exception_ptr error = {});

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error);

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason);

    void complete_cancelled(const std::shared_ptr<Request>& request);

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept;

    bool settle_terminal_requests(HostPhaseMeasurement& boundary);

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary);

    [[nodiscard]] bool expire_pending_requests();

    void commit_pending(PendingBatch&& pending, ComputeFlight flight);

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next);

    bool prepare_semantic_capture(const std::shared_ptr<Request>& request);

    using PrefillRow = typename decltype(std::declval<typename ModelContract::PrefillProgress&>()
                                             .rows)::value_type;

    // Applies the prefill result of one flight: charges every segment to the cycle budget, then
    // resolves its rows in member order.
    void resolve_prefill_flight(typename ModelContract::PrefillProgress&& progress,
                                ComputeFlight flight);

    // `flight` is consumed only by a row that carries a pending token, which exists only in a
    // single-segment flight.
    void resolve_prefill_progress(const std::shared_ptr<Request>& request, PrefillRow&& progress,
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

    bool pause_reclaim_victim(const ContextAssessment&);

    enum class PauseGoal { YieldLane, ReleaseCapacity, RelocateOnly };
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

    // Longest an admission that waits for capacity keeps the Engine in begin_cycle to chain its
    // reclaim steps (see begin_cycle).
    static constexpr std::chrono::milliseconds kAdmissionBurst{50};

    // Polls until the running context operation finishes or `deadline` passes. True when no
    // context operation remains.
    bool await_context_operation(HostPhaseMeasurement& boundary, Clock::time_point deadline);

    void observe_scheduling(const std::shared_ptr<Request>& request,
                            GenerationSchedulingTransition transition, Clock::time_point now = {});

    void update_recovery(const std::shared_ptr<Request>& request, bool released = false);

    enum class ReservationScope { Round, Prefill };

    [[nodiscard]] ExecutionUnit execution_unit(const Request& request) const;

    void reserve_resident_units(ReservationScope scope = ReservationScope::Round);

    using LaneSlots = std::array<std::shared_ptr<Request>, kMaximumConcurrency>;

    // The lanes that can take a prefill/replay unit in this reservation pass: resident lanes
    // without a pending capture, cancellation or Session execution, and not blocked by a
    // context operation.
    [[nodiscard]] LaneSlots prefill_slots() const;

    // Most tokens the next prefill/replay unit of every lane could process: the public prefill
    // chunk, limited by the prompt suffix a prefilling request still has to compute. Lanes
    // that are not in `slots` or do not prefill get 0.
    [[nodiscard]] PrefillUnitTokens prefill_unit_limits(const LaneSlots& slots) const;

    // Upper bound of the next unit of every lane: the Scheduler's caps for `limits`, with an
    // exclusive lane capped as if it were the only prefilling lane because it runs alone.
    [[nodiscard]] PrefillUnitTokens prefill_unit_caps(const LaneSlots& slots,
                                                      const PrefillUnitTokens& limits) const;

    // The lane the Session's Vision window continues, when it can run now.
    [[nodiscard]] std::optional<std::uint32_t> continuation_prefill_lane(const LaneSlots&) const;

    // The prefilling or replaying lane that must run alone: the progress owner, else the Vision
    // continuation lane. nullopt when every prefill lane may be packed.
    [[nodiscard]] std::optional<std::uint32_t> exclusive_prefill_lane(const LaneSlots&) const;

    [[nodiscard]] std::optional<std::uint32_t> next_prefill(const LaneSlots&,
                                                            const PrefillUnitTokens& caps) const;

    // Segments the Session may join into one prefill execution, bounded by the lane count.
    [[nodiscard]] std::uint32_t prefill_pack_limit() const;

    // The lanes of the next prefill execution: next_prefill() first, then the lanes the cycle
    // allowance admits behind it while the group may be packed. Packing starts only from a
    // prefilling lane, ends at the first Replay lane and excludes an exclusive lane. `caps` are
    // the tokens the lanes' permits carry.
    [[nodiscard]] std::vector<std::uint32_t>
    next_prefill_lanes(const LaneSlots&, const PrefillUnitTokens& caps) const;

    // Whether any lane that is not blocked by a capture waits for a decode or control round.
    [[nodiscard]] bool decode_waiting() const;

    // Admission can pin an existing writer's tail after its reservation. Membership must
    // observe current context dependencies, including pauses selected during reclamation.
    [[nodiscard]] bool unit_runnable(std::uint32_t lane) const;

    [[nodiscard]] ComputeFlight freeze_members(std::span<const std::uint32_t> lanes,
                                               UnitKind kind) const;

    // Starts one execution for a non-empty prefix of `lanes`, which are all prefilling or a
    // single replaying lane. Lanes the Session does not admit keep their sequences and lose
    // their Engine unit; the next cycle step reserves them again.
    [[nodiscard]] bool start_prefill(std::span<const std::uint32_t> lanes);

    [[nodiscard]] bool start_decode(const RoundMembership& membership);

    void discard_control_preview(const ComputeFlight& flight) noexcept;

    [[nodiscard]] bool start_control(const ControlMembership& membership);

    void resolve_control(ComputeFlight& flight);

    void supply_decision(ExecutionTicket ticket, const DecisionRequest& decision);

    void record_decision_upload(const DecisionConsumed& consumed);

    [[nodiscard]] bool poll_execution(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool execution_outstanding() const noexcept;

    [[nodiscard]] bool compute_pending() const noexcept;

    void record_service(ComputeFlight& flight) noexcept;
    void log_worker_failure(std::exception_ptr error) const noexcept;
    void fail_all(std::exception_ptr error) noexcept;

    void begin_cycle(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool advance_cycle();

    void worker_loop() noexcept;

    Instance& instance_;
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const std::uint32_t prefill_chunk_;
    const std::size_t max_outstanding_;
    const std::uint32_t speculative_positions_;
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
    std::uint64_t submitted_requests_     = 0;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;
    std::optional<AdmissionDecision> admission_decision_;
    std::uint64_t admission_generation_ = 1;
    std::array<std::optional<CaptureDecision>, kMaximumConcurrency> capture_decisions_;
    std::shared_ptr<Request> context_owner_;
    std::deque<DeferredSettlement> deferred_settlements_;
    std::optional<bool> pass_blocked_;
    std::vector<std::shared_ptr<Request>> paused_;
    Scheduling scheduler_;
    std::optional<ComputeFlight> compute_;
    std::optional<ComputeFlight> retiring_;
    CycleStep cycle_step_ = CycleStep::Boundary;
    std::array<bool, kMaximumConcurrency> control_executed_{};
    std::array<std::optional<UnitPermit>, kMaximumConcurrency> reserved_units_{};
    std::atomic<bool> admission_check_pending_{false};
    std::atomic<std::uint64_t> schedule_revision_{0};
    std::uint64_t published_revision_          = 0;
    Clock::time_point pending_deadline_        = Clock::time_point::max();
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    RuntimeStats stats_snapshot_;
    EngineSnapshot published_snapshot_;
    std::vector<RequestSnapshot> request_observations_, published_requests_;
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
